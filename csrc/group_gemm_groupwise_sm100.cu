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
#include <flashinfer/cutlass_utils.cuh>
#include <flashinfer/gemm/group_gemm_groupwise_sm100.cuh>

#include "pytorch_extension_utils.h"

using namespace flashinfer;

#define DISPATCH_PYTORCH_INPUT_OUTPUT_DTYPE(input_dtype, output_dtype, c_type_in, c_type_out, ...) \
  [&]() -> bool {                                                                                  \
    return DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(output_dtype, c_type_out, [&] {                    \
      return DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP8(input_dtype, c_type_in,                           \
                                                 [&] { return __VA_ARGS__(); });                   \
    });                                                                                            \
  }()

void CutlassGroupGemmGroupwiseScaledSM100(at::Tensor float_workspace_buffer,
                                          std::vector<at::Tensor> A, std::vector<at::Tensor> B,
                                          std::vector<at::Tensor> SFA, std::vector<at::Tensor> SFB,
                                          std::vector<at::Tensor> C) {
  int group_size = A.size();
  std::vector<std::tuple<int, int, int>> problem_shapes;
  for (int i = 0; i < group_size; ++i) {
    int m = A.at(i).size(0);
    int k = A.at(i).size(1);
    int n = B.at(i).size(0);
    problem_shapes.push_back(std::make_tuple(m, n, k));
  }

  const c10::cuda::OptionalCUDAGuard device_guard(float_workspace_buffer.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  DISPATCH_PYTORCH_INPUT_OUTPUT_DTYPE(
      A.at(0).scalar_type(), C.at(0).scalar_type(), c_type_in, c_type_out, [&] {
        using cutlass_t_in = cutlass_dtype_t<c_type_in>;
        using cutlass_t_out = cutlass_dtype_t<c_type_out>;
        std::vector<cutlass_t_in*> A_ptr;
        std::vector<cutlass_t_in*> B_ptr;
        std::vector<float*> SFA_ptr;
        std::vector<float*> SFB_ptr;
        std::vector<cutlass_t_out*> C_ptr;
        for (int i = 0; i < group_size; ++i) {
          A_ptr.push_back(static_cast<cutlass_t_in*>(A.at(i).data_ptr()));
          B_ptr.push_back(static_cast<cutlass_t_in*>(B.at(i).data_ptr()));
          SFA_ptr.push_back(static_cast<float*>(SFA.at(i).data_ptr()));
          SFB_ptr.push_back(static_cast<float*>(SFB.at(i).data_ptr()));
          C_ptr.push_back(static_cast<cutlass_t_out*>(C.at(i).data_ptr()));
        }
        auto status = flashinfer::gemm::CutlassGroupwiseScaledGroupGEMMSM100(
            static_cast<float*>(float_workspace_buffer.data_ptr()),
            float_workspace_buffer.element_size() * float_workspace_buffer.size(0), A_ptr, B_ptr,
            SFA_ptr, SFB_ptr, C_ptr, problem_shapes, stream);
        return true;
      });
}
