import os

os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")

import pytest
import torch
import triton
import triton.language as tl
from triton._C import libtriton
from triton.backends import backends
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.compiler.errors import CompilationError


if not hasattr(libtriton, "metax"):
    pytest.skip("MetaX backend not built in libtriton", allow_module_level=True)

if "metax" not in backends:
    pytest.skip("MetaX backend not discovered", allow_module_level=True)


_C550_TARGET = GPUTarget("maca", 80, 64)


@triton.jit
def _metax_int8_dot_kernel(lhs, rhs, out):
    offs_m = tl.arange(0, 16)
    offs_n = tl.arange(0, 16)
    offs_k = tl.arange(0, 32)
    a = tl.load(lhs + offs_m[:, None] * 32 + offs_k[None, :])
    b = tl.load(rhs + offs_k[:, None] * 16 + offs_n[None, :])
    acc = tl.dot(a, b)
    tl.store(out + offs_m[:, None] * 16 + offs_n[None, :], acc)


@triton.jit
def _metax_int8_narrow_dot_kernel(lhs, rhs, out):
    offs_m = tl.arange(0, 32)
    offs_n = tl.arange(0, 8)
    offs_k = tl.arange(0, 32)
    a = tl.load(lhs + offs_m[:, None] * 32 + offs_k[None, :])
    b = tl.load(rhs + offs_k[:, None] * 8 + offs_n[None, :])
    acc = tl.dot(a, b)
    tl.store(out + offs_m[:, None] * 8 + offs_n[None, :], acc)


def _compile_c550(signature):
    src = ASTSource(fn=_metax_int8_dot_kernel, signature=signature)
    return triton.compile(src, target=_C550_TARGET, options={"num_warps": 4})


def test_c550_signed_int8_dot_lowers_to_native_mma():
    compiled = _compile_c550({"lhs": "*i8", "rhs": "*i8", "out": "*i32"})
    assert "llvm.mxc.mma.i32.16x16x16i8" in compiled.asm["llir"]
    assert compiled.asm["mcfatbin"]


def test_c550_uint8_dot_fails_closed():
    with pytest.raises(CompilationError, match="only int8 supported"):
        _compile_c550({"lhs": "*u8", "rhs": "*u8", "out": "*i32"})


def test_c550_signed_int8_dot_below_native_tile_fails_closed():
    src = ASTSource(
        fn=_metax_int8_narrow_dot_kernel,
        signature={"lhs": "*i8", "rhs": "*i8", "out": "*i32"},
    )
    # The blocked i8 dot cannot use the generic FMA fallback because its
    # accumulator is i32, so rejecting the native tile must fail the MLIR pass.
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        triton.compile(src, target=_C550_TARGET, options={"num_warps": 4})


def test_c550_signed_int8_dot_device():
    if not torch.cuda.is_available():
        pytest.skip("requires a MetaX device")

    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "maca" or target.arch != 80:
        pytest.skip("requires a C550 / maca arch 80 target")

    torch.manual_seed(23)
    lhs = torch.randint(-4, 5, (16, 32), dtype=torch.int8, device="cuda")
    rhs = torch.randint(-4, 5, (32, 16), dtype=torch.int8, device="cuda")
    out = torch.empty((16, 16), dtype=torch.int32, device="cuda")

    compiled = _metax_int8_dot_kernel[(1, )](lhs, rhs, out, num_warps=4)
    torch.cuda.synchronize()

    expected = lhs.cpu().to(torch.int32) @ rhs.cpu().to(torch.int32)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
    assert "llvm.mxc.mma.i32.16x16x16i8" in compiled.asm["llir"]
