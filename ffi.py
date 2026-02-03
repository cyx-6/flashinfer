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

@cute.kernel
def device_add_one(a: cute.Tensor, b: cute.Tensor):
    b[0] = a[0] + 1

@cute.kernel
def device_add_one_4(a: cute.Tensor, b: cute.Tensor, a1: cute.Tensor, a2: cute.Tensor):
    b[0] = a[0] + 1


@cute.kernel
def device_add_one_6(a: cute.Tensor, b: cute.Tensor, a1: cute.Tensor, a2: cute.Tensor, a3: cute.Tensor, a4: cute.Tensor):
    b[0] = a[0] + 1


@cute.jit
def add_one(a: cute.Tensor, b: cute.Tensor):
    """b = a + 1"""
    device_add_one(a, b).launch(grid=(1, 1, 1), block=(1, 1, 1))

@cute.jit
def add_one_4(a: cute.Tensor, b: cute.Tensor, a1: cute.Tensor, a2: cute.Tensor):
    """b = a + 1"""
    device_add_one_4(a, b, a1, a2).launch(grid=(1, 1, 1), block=(1, 1, 1))


@cute.jit
def add_one_6(a: cute.Tensor, b: cute.Tensor, a1: cute.Tensor, a2: cute.Tensor, a3: cute.Tensor, a4: cute.Tensor):
    """b = a + 1"""
    device_add_one_6(a, b, a1, a2, a3, a4).launch(grid=(1, 1, 1), block=(1, 1, 1))

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


def get_shared_libs():
    libs = os.environ["CUTE_DSL_LIBS"].split(":")
    libs = [lib for lib in libs if Path(lib).exists()] + [tvm_ffi.libinfo.find_libtvm_ffi()]
    return libs


def benchmark_overhead(add_one, prefix: str, extra: int = 1):
    time.sleep(1)
    # compile the kernel with "--enable-tvm-ffi" option
    a_torch = torch.arange(10, dtype=torch.float32, device="cuda")
    b_torch = torch.zeros(10, dtype=torch.float32, device="cuda")
    a_cute = from_dlpack(a_torch, enable_tvm_ffi=True).mark_layout_dynamic()
    b_cute = from_dlpack(b_torch, enable_tvm_ffi=True).mark_layout_dynamic()
    # compile the kernel with "--enable-tvm-ffi" option
    compiled_add_one = cute.compile(add_one, a_cute, *([b_cute] * extra), options="--enable-tvm-ffi")
    shared_libs = get_shared_libs()
    Path.mkdir("./build", exist_ok=True)
    Path.unlink("./build/add_one_bench.o", missing_ok=True)
    Path.unlink("./build/add_one_bench.so", missing_ok=True)
    object_file_path = "./build/add_one_bench.o"
    lib_path = "./build/add_one_bench.so"
    # compile the object file to a shared library
    compiled_add_one.export_to_c(object_file_path, function_name="add_one")
    cmd = ["gcc", "-O3", "-flto", "-shared", "-o", lib_path, object_file_path, *shared_libs]
    subprocess.run(cmd, check=True)
    torch_args = [a_torch]
    for _ in range(extra):
        torch_args.append(torch.zeros(10, dtype=torch.float32, device="cuda"))
    cute_args = [a_cute]
    for _ in range(extra):
        cute_args.append(from_dlpack(torch.zeros(10, dtype=torch.float32, device="cuda"), enable_tvm_ffi=True).mark_layout_dynamic())
    benchmark_call(f"[{prefix}][JIT][TVM-FFI] call-with-torch-tensor", compiled_add_one, torch_args)
    benchmark_call(f"[{prefix}][JIT][TVM-FFI] call-with-cute-tensor", compiled_add_one, cute_args)
    time.sleep(1)
    aot_mod = tvm_ffi.load_module(lib_path)
    benchmark_call(f"[{prefix}][AOT][TVM-FFI] call-with-torch-tensor", aot_mod.add_one, torch_args)
    benchmark_call(f"[{prefix}][AOT][TVM-FFI] call-with-cute-tensor", aot_mod.add_one, cute_args)


def benchmark_old_overhead(add_one, prefix: str, extra: int = 1):
    time.sleep(1)
    # compile the kernel with "--enable-tvm-ffi" option
    a_torch = torch.arange(10, dtype=torch.float32, device="cuda")
    b_torch = torch.zeros(10, dtype=torch.float32, device="cuda")
    a_cute = from_dlpack(a_torch, enable_tvm_ffi=True).mark_layout_dynamic()
    b_cute = from_dlpack(b_torch, enable_tvm_ffi=True).mark_layout_dynamic()
    # compile the kernel with "--enable-tvm-ffi" option
    compiled_add_one = cute.compile(add_one, a_cute, *([b_cute] * extra))
    # run the compiled function by passing in cute.Tensor as input
    # you need to set enable_tvm_ffi=True for now
    cute_args = [a_cute]
    for _ in range(extra):
        cute_args.append(from_dlpack(a_torch, enable_tvm_ffi=True).mark_layout_dynamic())
    benchmark_call(f"[{prefix}][JIT][OLD] call-with-cute-tensor", compiled_add_one, cute_args)


def main():
    benchmark_overhead(add_one, "2-args", 1)
    benchmark_overhead(add_one_4, "4-args", 3)
    benchmark_overhead(add_one_6, "6-args", 5)

def main_old():
    benchmark_old_overhead(add_one, "2-args", 1)
    benchmark_old_overhead(add_one_4, "4-args", 3)
    benchmark_old_overhead(add_one_6, "6-args", 5)


if __name__ == "__main__":
    main()
    main_old()
