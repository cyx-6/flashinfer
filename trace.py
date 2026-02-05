"""Benchmark CuTe DSL norm kernels vs CUDA JIT implementation."""

import torch
import argparse
import os
import sys
import nvtx
import numpy as np
from flashinfer.testing.utils import bench_gpu_time, bench_e2e_time

# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def bench_kernel(fn, warmup=10, iters=100):
    return (
        np.median(
            # bench_gpu_time(
            #     lambda: fn(),
            #     enable_cupti=True,
            #     dry_run_iters=10,
            #     repeat_iters=100,
            # )
            bench_e2e_time(
                lambda: fn(),
                dry_run_iters=10,
                repeat_iters=100,
                cold_l2_cache=False,
            )
        )
        * 1000
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=["cute", "cuda", "both"], default="both")
    args = parser.parse_args()

    # Test configurations: (batch_size, hidden_size, dtype)
    configs = [
        # (1, 4096, torch.bfloat16),
        # (32, 4096, torch.bfloat16),
        # (64, 4096, torch.bfloat16),
        # (128, 4096, torch.bfloat16),
        (256, 4096, torch.bfloat16),
        # (32, 8192, torch.bfloat16),
        # (64, 8192, torch.bfloat16),
        # (32, 16384, torch.bfloat16),
    ]

    print("=" * 90)
    print("RMSNorm Benchmark")
    print("=" * 90)

    if args.impl == "both":
        print(
            f"{'Config':25} {'CuTe DSL (us)':>15} {'CUDA JIT (us)':>15} {'Speedup':>10}"
        )
        print("-" * 90)

        for batch, hidden, dtype in configs:
            x = torch.randn(batch, hidden, dtype=dtype, device="cuda")
            w = torch.randn(hidden, dtype=dtype, device="cuda")
            out = torch.empty_like(x)

            # CuTe DSL
            from flashinfer.cute_dsl import rmsnorm_cute

            cute_time = bench_kernel(lambda: rmsnorm_cute(x, w, out, 1e-6, 0.0))

            # CUDA JIT
            from flashinfer.jit.norm import gen_norm_module
            import functools

            @functools.cache
            def get_norm_module():
                return gen_norm_module().build_and_load()

            cuda_time = bench_kernel(
                lambda: get_norm_module().rmsnorm(out, x, w, 1e-6, False)
            )

            speedup = cuda_time / cute_time
            config_str = f"({batch}, {hidden}, bf16)"
            print(
                f"{config_str:25} {cute_time:>15.2f} {cuda_time:>15.2f} {speedup:>9.2f}x"
            )

    else:
        impl_name = "CuTe DSL" if args.impl == "cute" else "CUDA JIT"
        print(f"{'Config':25} {f'{impl_name} (us)':>15}")
        print("-" * 50)

        for batch, hidden, dtype in configs:
            x = torch.randn(batch, hidden, dtype=dtype, device="cuda")
            w = torch.randn(hidden, dtype=dtype, device="cuda")
            out = torch.empty_like(x)

            rmsnorm_cute(x, w, out, 1e-6, 0.0)
            return

            if args.impl == "cute":
                from flashinfer.cute_dsl import rmsnorm_cute

                time_us = bench_kernel(lambda: rmsnorm_cute(x, w, out, 1e-6, 0.0))
            else:
                from flashinfer.jit.norm import gen_norm_module
                import functools

                @functools.cache
                def get_norm_module():
                    return gen_norm_module().build_and_load()

                time_us = bench_kernel(
                    lambda: get_norm_module().rmsnorm(out, x, w, 1e-6, False)
                )

            config_str = f"({batch}, {hidden}, bf16)"
            print(f"{config_str:25} {time_us:>15.2f}")

    print("=" * 90)


if __name__ == "__main__":
    main()