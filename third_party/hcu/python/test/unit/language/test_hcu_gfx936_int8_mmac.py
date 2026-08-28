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


if not hasattr(libtriton, "hcu"):
    pytest.skip("HCU backend not built in libtriton", allow_module_level=True)

if "hcu" not in backends:
    pytest.skip("HCU backend not discovered", allow_module_level=True)


_GFX936_TARGET = GPUTarget("hip", "gfx936", 64)


@triton.jit
def _hcu_int8_dot_kernel(lhs, rhs, out):
    offs_m = tl.arange(0, 16)
    offs_n = tl.arange(0, 16)
    offs_k = tl.arange(0, 32)
    a = tl.load(lhs + offs_m[:, None] * 32 + offs_k[None, :])
    b = tl.load(rhs + offs_k[:, None] * 16 + offs_n[None, :])
    acc = tl.dot(a, b)
    tl.store(out + offs_m[:, None] * 16 + offs_n[None, :], acc)


def _compile_gfx936(signature):
    src = ASTSource(fn=_hcu_int8_dot_kernel, signature=signature)
    return triton.compile(src, target=_GFX936_TARGET, options={"num_warps": 4})


def test_gfx936_signed_int8_dot_lowers_to_native_mmac():
    compiled = _compile_gfx936({"lhs": "*i8", "rhs": "*i8", "out": "*i32"})
    assert "llvm.hcu.mmac.i32.16x16x32.i8" in compiled.asm["llir"]
    assert "v_mmac_i32_16x16x32_i8" in compiled.asm["amdgcn"]
    assert compiled.asm["hsaco"]


def test_gfx936_uint8_dot_fails_closed():
    with pytest.raises(CompilationError, match="only int8 supported"):
        _compile_gfx936({"lhs": "*u8", "rhs": "*u8", "out": "*i32"})


def test_gfx936_signed_int8_dot_device():
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("requires an HCU device")

    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "hip" or target.arch != "gfx936":
        pytest.skip("requires a gfx936 HCU device")

    torch.manual_seed(17)
    lhs = torch.randint(-4, 5, (16, 32), dtype=torch.int8, device="cuda")
    rhs = torch.randint(-4, 5, (32, 16), dtype=torch.int8, device="cuda")
    out = torch.empty((16, 16), dtype=torch.int32, device="cuda")

    compiled = _hcu_int8_dot_kernel[(1, )](lhs, rhs, out, num_warps=4)
    torch.cuda.synchronize()

    expected = lhs.cpu().to(torch.int32) @ rhs.cpu().to(torch.int32)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
    assert "v_mmac_i32_16x16x32_i8" in compiled.asm["amdgcn"]
