# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# {$nv-internal-release file}

import torch
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import subprocess
import tvm_ffi
import os
from pathlib import Path
import time
import functools

import cutlass
from cutlass import Float32, Int32
from flashinfer.norm.utils import (
    FLOAT8_E4M3_MAX,
    COPY_BITS,
    rcp_approx_ftz,
    cvt_and_store_f32_to_e4m3,
    get_ptr_as_int64,
    warp_reduce,
    row_reduce_sum,
    predicate_k,
    compute_optimal_vec_size,
    compute_threads_per_row,
    make_tv_layout,
    _torch_dtype_to_str,
    get_cutlass_dtype,
    get_num_sm,
)


@functools.cache
def rmsnorm_gen(
    dtype: cutlass.Numeric,
    H: int,
    weight_bias: float = 0.0
):
    # Vectorization parameters: use optimal vec_size for warp utilization
    elem_bits = dtype.width
    max_vec_size = COPY_BITS // elem_bits  # 8 for float16/bfloat16, 4 for float32
    vec_size = compute_optimal_vec_size(H, max_vec_size)
    copy_bits = vec_size * elem_bits  # Actual bits per copy

    # Thread configuration
    threads_per_row = compute_threads_per_row(H, vec_size)
    num_threads = threads_per_row  # One row per block
    num_warps = max(threads_per_row // 32, 1)

    # Vectorization blocks
    num_vec_blocks = max(
        1, (H // vec_size + threads_per_row - 1) // threads_per_row
    )
    cols_per_tile = vec_size * num_vec_blocks * threads_per_row

    def _smem_size_in_bytes() -> int:
        return num_warps * 4

    @cute.kernel
    def cute_kernel(
        mX: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        M: Int32,
        eps: Float32,
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        """Device kernel for RMSNorm."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        H = self.H
        weight_bias = self.weight_bias
        threads_per_row = tv_layout.shape[0][0]
        num_warps = self.num_warps
        copy_bits = self.copy_bits

        # Allocate shared memory (only reduction buffer needed)
        smem = cutlass.utils.SmemAllocator()
        reduction_buffer = smem.allocate_tensor(
            Float32,
            cute.make_layout((num_warps,)),
            byte_alignment=4,
        )

        # Create identity tensor for coordinate tracking
        idX = cute.make_identity_tensor(mX.shape)

        # Slice for this row
        gX = cute.local_tile(mX, tiler_mn, (bidx, 0))
        gY = cute.local_tile(mY, tiler_mn, (bidx, 0))
        cX = cute.local_tile(idX, tiler_mn, (bidx, 0))

        # Expand weight to 2D for consistent tiling
        mW_2d = cute.prepend_ones(mW, up_to_rank=2)

        # Create TiledCopy for load and store (both use CopyUniversalOp for sync operations)
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=copy_bits,
        )

        tiled_copy = cute.make_tiled_copy(copy_atom, tv_layout, tiler_mn)
        thr_copy = tiled_copy.get_slice(tidx)

        # Partition tensors
        tXgX = thr_copy.partition_S(gX)
        tXgW = thr_copy.partition_S(mW_2d)
        tXgY = thr_copy.partition_D(gY)
        tXcX = thr_copy.partition_S(cX)

        # Register fragments - initialize to zero for proper handling of out-of-bounds threads
        tXrX = cute.make_rmem_tensor(tXgX.shape, mX.element_type)
        tXrW = cute.make_rmem_tensor(tXgW.shape, mW.element_type)
        tXrX.store(cute.zeros_like(tXrX, dtype=mX.element_type))
        tXrW.store(cute.zeros_like(tXrW, dtype=mW.element_type))

        # Bounds checking (column boundary only, row is always valid since grid=[M,1,1])
        tXpX = predicate_k(tXcX, limit=H)

        # ===================================================================
        # Phase 1: Load input from global to register
        # ===================================================================
        cute.copy(copy_atom, tXgX, tXrX, pred=tXpX)

        x = tXrX.load().to(Float32)
        x_sq = x * x
        sum_sq = row_reduce_sum(x_sq, threads_per_row, reduction_buffer)

        # Compute rstd = 1 / sqrt(mean(x^2) + eps)
        mean_sq = sum_sq / Float32(H)
        rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)

        # ===================================================================
        # Phase 2: Load weight from global to register
        # ===================================================================
        cute.copy(copy_atom, tXgW, tXrW, pred=tXpX)

        w = tXrW.load().to(Float32)

        # output = input * rstd * (weight + weight_bias)
        y = x * rstd * (w + Float32(weight_bias))

        # Store output using cute.copy with predicate
        tYrY = y.to(mY.element_type)
        tXrY = cute.make_rmem_tensor(tXgY.shape, mY.element_type)
        tXrY.store(tYrY)

        cute.copy(copy_atom, tXrY, tXgY, pred=tXpX)

    @cute.jit
    def cute_func(
        mX: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        M: Int32,
        eps: Float32,
        stream,
    ):
        """Launch the RMSNorm kernel."""
        tv_shape, tv_stride = make_tv_layout(
            threads_per_row,
            vec_size,
            num_vec_blocks,
        )
        tv_layout = cute.make_layout(tv_shape, stride=tv_stride)
        tiler_mn = (1, cols_per_tile)

        cute_kernel(mX, mW, mY, M, eps, tv_layout, tiler_mn).launch(
            grid=[M, 1, 1],
            block=[num_threads, 1, 1],
            smem=_smem_size_in_bytes(),
            stream=stream,
        )

    return cute_kernel, cute_func


def print_speed(name: str, speed: float) -> None:
    print(f"{name:<60} {speed} sec/call")

def benchmark_call(name, func, args, num_calls=10000):
    func(*args)
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_calls):
        func(*args)
    torch.cuda.synchronize()
    end_time = time.time()
    time_per_call = (end_time - start_time) / num_calls
    print_speed(name, time_per_call)


def benchmark_overhead():
    time.sleep(1)
    batch_size = 128
    hidden_size = 4096
    dtype = torch.bfloat16

    x_torch = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
    w_torch = torch.randn(hidden_size, device="cuda", dtype=dtype)
    out_torch = torch.empty_like(x_torch)

    x_cute = from_dlpack(x_torch, enable_tvm_ffi=True).mark_layout_dynamic()
    w_cute = from_dlpack(w_torch, enable_tvm_ffi=True).mark_layout_dynamic()
    out_cute = from_dlpack(out_torch, enable_tvm_ffi=True).mark_layout_dynamic()

    cute_kernel, cute_func = rmsnorm_gen(
        get_cutlass_dtype(_torch_dtype_to_str(x_torch.dtype)),
        hidden_size
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled_rmsnorm = cute.compile(cute_func, x_cute, w_cute, out_cute, Int32(1), Float32(1e-6), stream_fake, options="--enable-tvm-ffi")
    torch_args = [x_torch, w_torch, out_torch, Int32(batch_size), Float32(1e-6)]
    cute_args = [x_cute, w_cute, out_cute, Int32(batch_size), Float32(1e-6)]
    benchmark_call(f"[RMSNORM][JIT][TVM-FFI] call-with-torch-tensor", compiled_rmsnorm, torch_args)
    benchmark_call(f"[RMSNORM][JIT][TVM-FFI] call-with-cute-tensor", compiled_rmsnorm, cute_args)


def benchmark_e2e():
    time.sleep(1)
    batch_size = 128
    hidden_size = 4096
    dtype = torch.bfloat16

    x_torch = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
    w_torch = torch.randn(hidden_size, device="cuda", dtype=dtype)
    out_torch = torch.empty_like(x_torch)

    def torch_rmsnorm():
        x_cute = from_dlpack(x_torch, enable_tvm_ffi=True).mark_layout_dynamic()
        w_cute = from_dlpack(w_torch, enable_tvm_ffi=True).mark_layout_dynamic()
        out_cute = from_dlpack(out_torch, enable_tvm_ffi=True).mark_layout_dynamic()

        cute_kernel, cute_func = rmsnorm_gen(
            get_cutlass_dtype(_torch_dtype_to_str(x_torch.dtype)),
            hidden_size
        )
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        compiled_rmsnorm = cute.compile(cute_func, x_cute, w_cute, out_cute, Int32(1), Float32(1e-6), stream_fake, options="--enable-tvm-ffi")
        compiled_rmsnorm(x_torch, w_torch, out_torch, Int32(batch_size), Float32(1e-6))

    def cute_rmsnorm():
        x_cute = from_dlpack(x_torch, enable_tvm_ffi=True).mark_layout_dynamic()
        w_cute = from_dlpack(w_torch, enable_tvm_ffi=True).mark_layout_dynamic()
        out_cute = from_dlpack(out_torch, enable_tvm_ffi=True).mark_layout_dynamic()

        cute_kernel, cute_func = rmsnorm_gen(
            get_cutlass_dtype(_torch_dtype_to_str(x_torch.dtype)),
            hidden_size
        )
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        compiled_rmsnorm = cute.compile(cute_func, x_cute, w_cute, out_cute, Int32(1), Float32(1e-6), stream_fake, options="--enable-tvm-ffi")
        compiled_rmsnorm(x_cute, w_cute, out_cute, Int32(batch_size), Float32(1e-6))

    benchmark_call(f"[RMSNORM][JIT][TVM-FFI] call-with-torch-tensor", torch_rmsnorm, [], 100)
    benchmark_call(f"[RMSNORM][JIT][TVM-FFI] call-with-cute-tensor", cute_rmsnorm, [], 100)
    benchmark_call(f"[RMSNORM][JIT][TVM-FFI] call-with-torch-tensor", torch_rmsnorm, [], 100)
    benchmark_call(f"[RMSNORM][JIT][TVM-FFI] call-with-cute-tensor", cute_rmsnorm, [], 100)

def main():
    benchmark_overhead()
    benchmark_e2e()


if __name__ == "__main__":
    main()
