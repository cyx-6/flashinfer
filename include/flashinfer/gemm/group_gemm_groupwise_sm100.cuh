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
#ifndef FLASHINFER_GRUOP_GEMM_GROUPWISE_SM100_CUH_
#define FLASHINFER_GRUOP_GEMM_GROUPWISE_SM100_CUH_

#include <cassert>
#include <iterator>

#include "../allocator.h"
#include "../cutlass_utils.cuh"
#include "../utils.cuh"

namespace flashinfer {

namespace gemm {

using namespace cute;

template <typename DTypeIn, typename DTypeOut>
cudaError_t CutlassGroupwiseScaledGroupGEMMSM100(
    void* float_buffer, size_t float_buffer_size_in_bytes, std::vector<DTypeIn*> A_ptr_host,
    std::vector<DTypeIn*> B_ptr_host, std::vector<float*> SFA_ptr_host,
    std::vector<float*> SFB_ptr_host, std::vector<DTypeOut*> C_ptr_host,
    std::vector<std::tuple<int, int, int>> problem_shapes, cudaStream_t stream) {
  using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;  // <M,N,K> per group

  using ElementA = DTypeIn;                   // Element type for A matrix operand
  using LayoutA = cutlass::layout::RowMajor;  // Layout type for A matrix operand
  constexpr int AlignmentA =
      128 / cutlass::sizeof_bits<ElementA>::value;  // Memory access granularity/alignment of A
                                                    // matrix in units of elements (up to 16 bytes)

  using ElementB = DTypeIn;                      // Element type for B matrix operand
  using LayoutB = cutlass::layout::ColumnMajor;  // Layout type for B matrix operand
  constexpr int AlignmentB =
      128 / cutlass::sizeof_bits<ElementB>::value;  // Memory access granularity/alignment of A
                                                    // matrix in units of elements (up to 16 bytes)

  using ElementC = DTypeOut;                  // Element type for C and D matrix operands
  using LayoutC = cutlass::layout::RowMajor;  // Layout type for C and D matrix operands
  constexpr int AlignmentC =
      128 / cutlass::sizeof_bits<ElementC>::value;  // Memory access granularity/alignment of A
                                                    // matrix in units of elements (up to 16 bytes)

  using ElementD = ElementC;
  using LayoutD = LayoutC;
  constexpr int AlignmentD = AlignmentC;

  using ElementAccumulator = float;
  using ElementCompute = float;

  using MmaTileShape_MNK = Shape<_256, _128, _128>;
  using ClusterShape_MNK = Shape<_2, _1, _1>;

  constexpr int ScaleGranularityM = 1;
  constexpr int ScaleGranularityN = 128;
  constexpr int ScaleGranularityK = 128;
  using ScaleConfig =
      cutlass::detail::Sm100BlockwiseScaleConfig<ScaleGranularityM, ScaleGranularityN,
                                                 ScaleGranularityK>;

  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, MmaTileShape_MNK, ClusterShape_MNK,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute, ElementC,
      LayoutC*, AlignmentC, ElementD, LayoutC*, AlignmentD,
      cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, ElementA,
      cute::tuple<LayoutA*, LayoutSFA*>, AlignmentA, ElementB, cute::tuple<LayoutB*, LayoutSFB*>,
      AlignmentB, ElementAccumulator, MmaTileShape_MNK, ClusterShape_MNK,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedBlockwise2SmSm100>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop,
                                                          CollectiveEpilogue, void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;

  static_assert(
      cute::is_same_v<typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA, LayoutSFA>);
  static_assert(
      cute::is_same_v<typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB, LayoutSFB>);

  int group_size = problem_shapes.size();

  cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape> problem_sizes(
      group_size);

  cutlass::DeviceAllocation<const typename Gemm::ElementA*> A_ptr(group_size);
  cutlass::DeviceAllocation<const typename Gemm::ElementB*> B_ptr(group_size);
  cutlass::DeviceAllocation<const typename Gemm::ElementC*> C_ptr(group_size);
  cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput*> D_ptr(group_size);
  cutlass::DeviceAllocation<const ElementAccumulator*> SFA_ptr(group_size);
  cutlass::DeviceAllocation<const ElementAccumulator*> SFB_ptr(group_size);

  cutlass::DeviceAllocation<StrideA> stride_A(group_size);
  cutlass::DeviceAllocation<StrideB> stride_B(group_size);
  cutlass::DeviceAllocation<StrideC> stride_C(group_size);
  cutlass::DeviceAllocation<LayoutSFA> layout_SFA(group_size);
  cutlass::DeviceAllocation<LayoutSFB> layout_SFB(group_size);

  std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes_host;
  std::vector<StrideA> stride_A_host;
  std::vector<StrideB> stride_B_host;
  std::vector<StrideC> stride_C_host;
  std::vector<LayoutSFA> layout_SFA_host;
  std::vector<LayoutSFB> layout_SFB_host;

  for (int i = 0; i < group_size; ++i) {
    auto [m, n, k] = problem_shapes.at(i);
    auto gemm_layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, 1));
    auto gemm_layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));
    problem_sizes_host.push_back({m, n, k});
    stride_A_host.push_back(cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1}));
    stride_B_host.push_back(cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1}));
    stride_C_host.push_back(cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1}));
    layout_SFA_host.push_back(gemm_layout_SFA);
    layout_SFB_host.push_back(gemm_layout_SFB);
  }

  problem_sizes.copy_from_host(problem_sizes_host.data());
  A_ptr.copy_from_host(A_ptr_host.data());
  B_ptr.copy_from_host(B_ptr_host.data());
  C_ptr.copy_from_host(C_ptr_host.data());
  D_ptr.copy_from_host(C_ptr_host.data());
  SFA_ptr.copy_from_host(SFA_ptr_host.data());
  SFB_ptr.copy_from_host(SFB_ptr_host.data());
  stride_A.copy_from_host(stride_A_host.data());
  stride_B.copy_from_host(stride_B_host.data());
  stride_C.copy_from_host(stride_C_host.data());
  layout_SFA.copy_from_host(layout_SFA_host.data());
  layout_SFB.copy_from_host(layout_SFB_host.data());
  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  typename Gemm::Arguments arguments{cutlass::gemm::GemmUniversalMode::kGrouped,
                                     {group_size, problem_sizes.get(), problem_sizes_host.data()},
                                     {
                                         A_ptr.get(),
                                         stride_A.get(),
                                         B_ptr.get(),
                                         stride_B.get(),
                                         SFA_ptr.get(),
                                         layout_SFA.get(),
                                         SFB_ptr.get(),
                                         layout_SFB.get(),
                                     },
                                     {
                                         {},  // epilogue.thread
                                         C_ptr.get(),
                                         stride_C.get(),
                                         D_ptr.get(),
                                         stride_C.get(),
                                     },
                                     hw_info};
  auto& fusion_args = arguments.epilogue.thread;
  fusion_args.alpha = 1.0f;
  fusion_args.beta = 0.0f;

  Gemm gemm;

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  AlignedAllocator float_allocator(float_buffer, float_buffer_size_in_bytes);
  auto workspace_ptr = float_allocator.aligned_alloc<void>(
      workspace_size, 32 * 1024 * 1024, "sm100_groupwise_group_gemm_float_workspace");

  CUTLASS_CHECK(gemm.can_implement(arguments));
  CUTLASS_CHECK(gemm.initialize(arguments, workspace_ptr));
  CUTLASS_CHECK(gemm.run(stream));
  return cudaSuccess;
}

}  // namespace gemm

}  // namespace flashinfer

#endif  // FLASHINFER_GRUOP_GEMM_GROUPWISE_SM100_CUH_
