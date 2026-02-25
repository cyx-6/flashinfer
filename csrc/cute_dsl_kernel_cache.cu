/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>

#include <mutex>
#include <tuple>
#include <unordered_map>

#include "tvm_ffi_utils.h"

namespace flashinfer {

// Key: (dtype_code, H, weight_bias_scaled, enable_pdl)
using RMSNormKey = std::tuple<int64_t, int64_t, int64_t, bool>;

struct RMSNormKeyHash {
  std::size_t operator()(const RMSNormKey& k) const {
    // Combine hashes using XOR + bit shifts
    return std::hash<int64_t>{}(std::get<0>(k)) ^
           (std::hash<int64_t>{}(std::get<1>(k)) << 1) ^
           (std::hash<int64_t>{}(std::get<2>(k)) << 2) ^
           (std::hash<bool>{}(std::get<3>(k)) << 3);
  }
};

// Use raw pointers that are intentionally never freed to avoid static destruction
// order issues with Python references. The memory is reclaimed by the OS at process exit.
static std::unordered_map<RMSNormKey, tvm::ffi::Function, RMSNormKeyHash>* rmsnorm_kernel_cache =
    nullptr;
static std::mutex* rmsnorm_cache_mutex = nullptr;
static tvm::ffi::Function* rmsnorm_compile_callback = nullptr;

static void EnsureCacheInitialized() {
  if (!rmsnorm_cache_mutex) {
    rmsnorm_cache_mutex = new std::mutex();
  }
  if (!rmsnorm_kernel_cache) {
    rmsnorm_kernel_cache =
        new std::unordered_map<RMSNormKey, tvm::ffi::Function, RMSNormKeyHash>();
  }
}

void SetRMSNormCompileCallback(tvm::ffi::Function callback) {
  EnsureCacheInitialized();
  if (rmsnorm_compile_callback) {
    *rmsnorm_compile_callback = callback;
  } else {
    rmsnorm_compile_callback = new tvm::ffi::Function(callback);
  }
}

void RMSNormCuteCached(tvm::ffi::TensorView input, tvm::ffi::TensorView weight,
                       tvm::ffi::TensorView out, double eps, double weight_bias,
                       bool enable_pdl) {
  EnsureCacheInitialized();

  // Extract key info directly from tensor (no Python needed)
  int64_t dtype_code = encode_dlpack_dtype(input.dtype());
  int64_t H = input.size(-1);
  int64_t M;
  if (input.ndim() == 3) {
    M = input.size(0) * input.size(1);
  } else {
    M = input.size(0);
  }
  // Scale weight_bias for precision in hash key
  int64_t weight_bias_key = static_cast<int64_t>(weight_bias * 10000);

  RMSNormKey key = {dtype_code, H, weight_bias_key, enable_pdl};

  tvm::ffi::Function kernel;
  {
    std::lock_guard<std::mutex> lock(*rmsnorm_cache_mutex);
    auto it = rmsnorm_kernel_cache->find(key);
    if (it == rmsnorm_kernel_cache->end()) {
      // Cache miss - call Python callback to compile
      TVM_FFI_ICHECK(rmsnorm_compile_callback != nullptr)
          << "RMSNorm compile callback not set. Call set_rmsnorm_compile_callback first.";
      kernel = (*rmsnorm_compile_callback)(dtype_code, H, weight_bias, enable_pdl)
                   .cast<tvm::ffi::Function>();
      (*rmsnorm_kernel_cache)[key] = kernel;
    } else {
      kernel = it->second;
    }
  }

  // Reshape input and output to 2D if needed for kernel call
  // The kernel expects 2D tensors [M, H]
  if (input.ndim() == 3) {
    // Create 2D views
    int64_t shape_2d[2] = {M, H};
    int64_t stride_2d_in[2] = {input.stride(1), input.stride(2)};
    int64_t stride_2d_out[2] = {out.stride(1), out.stride(2)};

    DLManagedTensor input_managed;
    input_managed.dl_tensor.data = input.data_ptr();
    input_managed.dl_tensor.device = input.device();
    input_managed.dl_tensor.ndim = 2;
    input_managed.dl_tensor.dtype = input.dtype();
    input_managed.dl_tensor.shape = shape_2d;
    input_managed.dl_tensor.strides = stride_2d_in;
    input_managed.dl_tensor.byte_offset = 0;
    input_managed.manager_ctx = nullptr;
    input_managed.deleter = nullptr;

    DLManagedTensor out_managed;
    out_managed.dl_tensor.data = out.data_ptr();
    out_managed.dl_tensor.device = out.device();
    out_managed.dl_tensor.ndim = 2;
    out_managed.dl_tensor.dtype = out.dtype();
    out_managed.dl_tensor.shape = shape_2d;
    out_managed.dl_tensor.strides = stride_2d_out;
    out_managed.dl_tensor.byte_offset = 0;
    out_managed.manager_ctx = nullptr;
    out_managed.deleter = nullptr;

    tvm::ffi::TensorView input_2d(&input_managed.dl_tensor);
    tvm::ffi::TensorView out_2d(&out_managed.dl_tensor);

    kernel(input_2d, weight, out_2d, static_cast<int32_t>(M), static_cast<float>(eps));
  } else {
    // Already 2D, call directly
    kernel(input, weight, out, static_cast<int32_t>(M), static_cast<float>(eps));
  }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(flashinfer_set_rmsnorm_compile_callback, SetRMSNormCompileCallback);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(flashinfer_rmsnorm_cute_cached, RMSNormCuteCached);

// =============================================================================
// LayerNorm Caching
// =============================================================================

// Key: (dtype_code, gamma_dtype_code, H, enable_pdl)
using LayerNormKey = std::tuple<int64_t, int64_t, int64_t, bool>;

struct LayerNormKeyHash {
  std::size_t operator()(const LayerNormKey& k) const {
    return std::hash<int64_t>{}(std::get<0>(k)) ^
           (std::hash<int64_t>{}(std::get<1>(k)) << 1) ^
           (std::hash<int64_t>{}(std::get<2>(k)) << 2) ^
           (std::hash<bool>{}(std::get<3>(k)) << 3);
  }
};

static std::unordered_map<LayerNormKey, tvm::ffi::Function, LayerNormKeyHash>* layernorm_kernel_cache =
    nullptr;
static std::mutex* layernorm_cache_mutex = nullptr;
static tvm::ffi::Function* layernorm_compile_callback = nullptr;

static void EnsureLayerNormCacheInitialized() {
  if (!layernorm_cache_mutex) {
    layernorm_cache_mutex = new std::mutex();
  }
  if (!layernorm_kernel_cache) {
    layernorm_kernel_cache =
        new std::unordered_map<LayerNormKey, tvm::ffi::Function, LayerNormKeyHash>();
  }
}

void SetLayerNormCompileCallback(tvm::ffi::Function callback) {
  EnsureLayerNormCacheInitialized();
  if (layernorm_compile_callback) {
    *layernorm_compile_callback = callback;
  } else {
    layernorm_compile_callback = new tvm::ffi::Function(callback);
  }
}

void LayerNormCuteCached(tvm::ffi::TensorView out, tvm::ffi::TensorView input,
                         tvm::ffi::TensorView gamma, tvm::ffi::TensorView beta,
                         double eps, bool enable_pdl) {
  EnsureLayerNormCacheInitialized();

  int64_t dtype_code = encode_dlpack_dtype(input.dtype());
  int64_t gamma_dtype_code = encode_dlpack_dtype(gamma.dtype());
  int64_t H = input.size(-1);
  int64_t M = input.size(0);

  LayerNormKey key = {dtype_code, gamma_dtype_code, H, enable_pdl};

  tvm::ffi::Function kernel;
  {
    std::lock_guard<std::mutex> lock(*layernorm_cache_mutex);
    auto it = layernorm_kernel_cache->find(key);
    if (it == layernorm_kernel_cache->end()) {
      TVM_FFI_ICHECK(layernorm_compile_callback != nullptr)
          << "LayerNorm compile callback not set. Call set_layernorm_compile_callback first.";
      kernel = (*layernorm_compile_callback)(dtype_code, gamma_dtype_code, H, enable_pdl)
                   .cast<tvm::ffi::Function>();
      (*layernorm_kernel_cache)[key] = kernel;
    } else {
      kernel = it->second;
    }
  }

  kernel(out, input, gamma, beta, static_cast<int32_t>(M), static_cast<float>(eps));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(flashinfer_set_layernorm_compile_callback, SetLayerNormCompileCallback);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(flashinfer_layernorm_cute_cached, LayerNormCuteCached);

// =============================================================================
// Fused Add RMSNorm Caching
// =============================================================================

// Key: (dtype_code, H, weight_bias_scaled, enable_pdl) - same as RMSNormKey
using FusedAddRMSNormKey = std::tuple<int64_t, int64_t, int64_t, bool>;

struct FusedAddRMSNormKeyHash {
  std::size_t operator()(const FusedAddRMSNormKey& k) const {
    return std::hash<int64_t>{}(std::get<0>(k)) ^
           (std::hash<int64_t>{}(std::get<1>(k)) << 1) ^
           (std::hash<int64_t>{}(std::get<2>(k)) << 2) ^
           (std::hash<bool>{}(std::get<3>(k)) << 3);
  }
};

static std::unordered_map<FusedAddRMSNormKey, tvm::ffi::Function, FusedAddRMSNormKeyHash>*
    fused_add_rmsnorm_kernel_cache = nullptr;
static std::mutex* fused_add_rmsnorm_cache_mutex = nullptr;
static tvm::ffi::Function* fused_add_rmsnorm_compile_callback = nullptr;

static void EnsureFusedAddRMSNormCacheInitialized() {
  if (!fused_add_rmsnorm_cache_mutex) {
    fused_add_rmsnorm_cache_mutex = new std::mutex();
  }
  if (!fused_add_rmsnorm_kernel_cache) {
    fused_add_rmsnorm_kernel_cache =
        new std::unordered_map<FusedAddRMSNormKey, tvm::ffi::Function, FusedAddRMSNormKeyHash>();
  }
}

void SetFusedAddRMSNormCompileCallback(tvm::ffi::Function callback) {
  EnsureFusedAddRMSNormCacheInitialized();
  if (fused_add_rmsnorm_compile_callback) {
    *fused_add_rmsnorm_compile_callback = callback;
  } else {
    fused_add_rmsnorm_compile_callback = new tvm::ffi::Function(callback);
  }
}

void FusedAddRMSNormCuteCached(tvm::ffi::TensorView input, tvm::ffi::TensorView residual,
                                tvm::ffi::TensorView weight, double eps, double weight_bias,
                                bool enable_pdl) {
  EnsureFusedAddRMSNormCacheInitialized();

  int64_t dtype_code = encode_dlpack_dtype(input.dtype());
  int64_t H = input.size(-1);
  int64_t M = input.size(0);
  int64_t weight_bias_key = static_cast<int64_t>(weight_bias * 10000);

  FusedAddRMSNormKey key = {dtype_code, H, weight_bias_key, enable_pdl};

  tvm::ffi::Function kernel;
  {
    std::lock_guard<std::mutex> lock(*fused_add_rmsnorm_cache_mutex);
    auto it = fused_add_rmsnorm_kernel_cache->find(key);
    if (it == fused_add_rmsnorm_kernel_cache->end()) {
      TVM_FFI_ICHECK(fused_add_rmsnorm_compile_callback != nullptr)
          << "FusedAddRMSNorm compile callback not set. Call set_fused_add_rmsnorm_compile_callback first.";
      kernel = (*fused_add_rmsnorm_compile_callback)(dtype_code, H, weight_bias, enable_pdl)
                   .cast<tvm::ffi::Function>();
      (*fused_add_rmsnorm_kernel_cache)[key] = kernel;
    } else {
      kernel = it->second;
    }
  }

  kernel(input, residual, weight, static_cast<int32_t>(M), static_cast<float>(eps));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(flashinfer_set_fused_add_rmsnorm_compile_callback,
                              SetFusedAddRMSNormCompileCallback);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(flashinfer_fused_add_rmsnorm_cute_cached, FusedAddRMSNormCuteCached);

}  // namespace flashinfer
