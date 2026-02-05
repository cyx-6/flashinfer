import torch
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import time
import functools

import cutlass
from cutlass import Float32, Int32
import nvtx
import math


COPY_BITS = 128


def get_cutlass_dtype(dtype: str) -> cutlass.dtype:
    dtype_map = {
        "float16": cutlass.Float16,
        "bfloat16": cutlass.BFloat16,
        "float32": cutlass.Float32,
        "float8_e5m2": cutlass.Float8E5M2,
        "float8_e4m3fn": cutlass.Float8E4M3FN,
        "float8_e8m0fnu": cutlass.Float8E8M0FNU,
        "float4_e2m1fn": cutlass.Float4E2M1FN,
    }
    return dtype_map[dtype]

def _torch_dtype_to_str(dtype: torch.dtype) -> str:
    dtype_map = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
        torch.float8_e4m3fn: "float8_e4m3fn",
    }
    return dtype_map[dtype]


def compute_optimal_vec_size(H: int, max_vec_size: int) -> int:
    """Compute vec_size that maximizes warp utilization.

    For small hidden sizes, using max vec_size may result in fewer than 32 threads,
    wasting warp resources. This function finds the largest vec_size that:
    1. Divides H evenly
    2. Results in at least 32 threads (one full warp)

    Examples:
    - H=128, max=8: vec_size=8 gives 16 threads, vec_size=4 gives 32 threads -> return 4
    - H=4096, max=8: vec_size=8 gives 512 threads -> return 8
    - H=111, max=8: no vec_size divides evenly with >=32 threads, use gcd -> return 1
    """
    # Try vec_sizes from largest to smallest
    for vec_size in [
        max_vec_size,
        max_vec_size // 2,
        max_vec_size // 4,
        max_vec_size // 8,
    ]:
        if vec_size < 1:
            continue
        if H % vec_size != 0:
            continue
        threads_needed = H // vec_size
        if threads_needed >= 32:
            return vec_size
    # Fallback: use gcd for correctness (handles odd sizes like 111)
    return math.gcd(max_vec_size, H)


def print_speed(name: str, speed: float) -> None:
    print(f"{name:<60} {speed} sec/call")

def benchmark_call(name, func, args, num_calls=10000):
    func(*args)
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_calls):
        # with nvtx.annotate("rmsnorm"):
            func(*args)
    torch.cuda.synchronize()
    end_time = time.time()
    time_per_call = (end_time - start_time) / num_calls
    print_speed(name, time_per_call)

class RMSNormKernel:
    """
    RMSNorm Kernel using CuTe-DSL.

    Computes: output = input / sqrt(mean(input^2) + eps) * (weight + weight_bias)

    Key optimizations:
    1. 128-bit vectorized loads for input and weight
    2. Two-stage reduction: warp shuffle + cross-warp shared memory
    3. All computations in FP32 for numerical stability
    """

    def __init__(
        self,
        dtype: cutlass.Numeric,
        H: int,
        weight_bias: float = 0.0,
    ):
        # Vectorization parameters: use optimal vec_size for warp utilization
        elem_bits = dtype.width
        max_vec_size = COPY_BITS // elem_bits  # 8 for float16/bfloat16, 4 for float32
        self.vec_size = compute_optimal_vec_size(H, max_vec_size)


    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        M: Int32,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        stream,
    ):
        """Launch the RMSNorm kernel."""


@functools.cache
def _get_compiled_rmsnorm_kernel(
    dtype_str: str, H: int, weight_bias: float, enable_pdl: bool
):
    """Get a compiled RMSNorm kernel using TVM-FFI."""
    dtype = get_cutlass_dtype(dtype_str)
    kernel_obj = RMSNormKernel(dtype, H, weight_bias)

    # Use symbolic size for dynamic M dimension
    sym_m = cute.sym_int()
    # Use symbolic stride for arbitrary row stride (last dim must be contiguous)
    sym_row_stride_x = cute.sym_int(divisibility=kernel_obj.vec_size)
    sym_row_stride_y = cute.sym_int(divisibility=kernel_obj.vec_size)

    # Create fake tensors with symbolic stride for arbitrary stride support
    x_fake = cute.runtime.make_fake_tensor(
        dtype, (sym_m, H), (sym_row_stride_x, 1), assumed_align=16
    )
    w_fake = cute.runtime.make_fake_compact_tensor(dtype, (H,), assumed_align=16)
    y_fake = cute.runtime.make_fake_tensor(
        dtype, (sym_m, H), (sym_row_stride_y, 1), assumed_align=16
    )

    # Create fake stream that uses environment stream at runtime
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    # Compile with TVM-FFI enabled
    compiled_kernel = cute.compile(
        kernel_obj,
        x_fake,
        w_fake,
        y_fake,
        Int32(1),  # Dummy M
        Float32(1e-6),  # Dummy eps
        enable_pdl,
        stream_fake,
        options="--enable-tvm-ffi",
    )

    return compiled_kernel


def benchmark_overhead():
    time.sleep(1)
    batch_size = 128
    hidden_size = 4096
    dtype = torch.bfloat16

    x_torch = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
    w_torch = torch.randn(hidden_size, device="cuda", dtype=dtype)
    out_torch = torch.empty_like(x_torch)

    dtype_str = _torch_dtype_to_str(x_torch.dtype)
    kernel = _get_compiled_rmsnorm_kernel(dtype_str, hidden_size, 0.0, False)

    torch_args = [x_torch, w_torch, out_torch, batch_size, 1e-6]
    benchmark_call(f"[RMSNORM][JIT][TVM-FFI] call-with-torch-tensor", kernel, torch_args)


def main():
    benchmark_overhead()


if __name__ == "__main__":
    main()
