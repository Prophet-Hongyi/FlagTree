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


if not hasattr(libtriton, "hcu"):
    pytest.skip("HCU backend not built in libtriton", allow_module_level=True)

if "hcu" not in backends:
    pytest.skip("HCU backend not discovered", allow_module_level=True)


_GFX936_TARGET = GPUTarget("hip", "gfx936", 64)
_FP16_MMAC_LLIR = "llvm.hcu.mmac.f32.16x16x16.f16"
_FP16_MMAC_ASM = "v_mmac_f32_16x16x16_f16"


@triton.jit
def _hcu_e2m1_dot_scaled_kernel(lhs, rhs, out):
    offsets_m = tl.arange(0, 16)
    offsets_n = tl.arange(0, 16)
    offsets_k_packed = tl.arange(0, 32)
    a = tl.load(lhs + offsets_m[:, None] * 32 + offsets_k_packed[None, :])
    b = tl.load(rhs + offsets_k_packed[:, None] * 16 + offsets_n[None, :])
    acc = tl.zeros((16, 16), dtype=tl.float32)
    result = tl.dot_scaled(
        a,
        None,
        "e2m1",
        b,
        None,
        "e2m1",
        acc,
        lhs_k_pack=True,
        rhs_k_pack=True,
    )
    tl.store(out + offsets_m[:, None] * 16 + offsets_n[None, :], result)


@triton.jit
def _hcu_scaled_e2m1_dot_kernel(lhs, lhs_scale, rhs, rhs_scale, out):
    offsets_m = tl.arange(0, 16)
    offsets_n = tl.arange(0, 16)
    offsets_k_packed = tl.arange(0, 32)
    offsets_scale = tl.arange(0, 2)
    a = tl.load(lhs + offsets_m[:, None] * 32 + offsets_k_packed[None, :])
    b = tl.load(rhs + offsets_k_packed[:, None] * 16 + offsets_n[None, :])
    a_scale = tl.load(lhs_scale + offsets_m[:, None] * 2 + offsets_scale[None, :])
    b_scale = tl.load(rhs_scale + offsets_n[:, None] * 2 + offsets_scale[None, :])
    acc = tl.zeros((16, 16), dtype=tl.float32)
    result = tl.dot_scaled(
        a,
        a_scale,
        "e2m1",
        b,
        b_scale,
        "e2m1",
        acc,
        lhs_k_pack=True,
        rhs_k_pack=True,
    )
    tl.store(out + offsets_m[:, None] * 16 + offsets_n[None, :], result)


def _compile_no_scale():
    source = ASTSource(
        fn=_hcu_e2m1_dot_scaled_kernel,
        signature={"lhs": "*u8", "rhs": "*u8", "out": "*fp32"},
    )
    return triton.compile(
        source,
        target=_GFX936_TARGET,
        options={"num_warps": 4, "num_stages": 1},
    )


def _pack_along_k(raw_codes, dim):
    if dim == 1:
        low = raw_codes[:, 0::2]
        high = raw_codes[:, 1::2]
    else:
        low = raw_codes[0::2, :]
        high = raw_codes[1::2, :]
    return low | (high << 4)


def _assert_software_e2m1_dot(compiled):
    assert compiled.asm["ttgir"].count("fp4_to_fp") == 2
    assert compiled.asm["llir"].count(_FP16_MMAC_LLIR + "(") == 5
    assert compiled.asm["amdgcn"].count(_FP16_MMAC_ASM) == 4
    assert "f8f6f4" not in compiled.asm["amdgcn"]


def test_gfx936_e2m1_dot_scaled_lowers_through_software_decode_and_fp16_mmac():
    compiled = _compile_no_scale()
    _assert_software_e2m1_dot(compiled)
    assert compiled.asm["hsaco"]


def test_gfx936_both_scaled_e2m1_dot_remains_fail_closed(capfd):
    source = ASTSource(
        fn=_hcu_scaled_e2m1_dot_kernel,
        signature={
            "lhs": "*u8",
            "lhs_scale": "*u8",
            "rhs": "*u8",
            "rhs_scale": "*u8",
            "out": "*fp32",
        },
    )
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        triton.compile(
            source,
            target=_GFX936_TARGET,
            options={"num_warps": 4, "num_stages": 1},
        )
    assert "Unsupported DotScaleOp found" in capfd.readouterr().err


def test_gfx936_e2m1_dot_scaled_device():
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("requires an HCU device")

    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "hip" or target.arch != "gfx936":
        pytest.skip("requires a gfx936 HCU device")

    codebook = torch.arange(16, dtype=torch.uint8)
    valuebook = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
    )
    lhs_index = ((torch.arange(16 * 64) * 7 + 3) % len(codebook)).reshape(16, 64)
    rhs_index = ((torch.arange(64 * 16) * 11 + 1) % len(codebook)).reshape(64, 16)
    lhs = _pack_along_k(codebook[lhs_index], dim=1).cuda()
    rhs = _pack_along_k(codebook[rhs_index], dim=0).cuda()
    out = torch.empty((16, 16), dtype=torch.float32, device="cuda")

    compiled = _hcu_e2m1_dot_scaled_kernel[(1,)](
        lhs,
        rhs,
        out,
        num_warps=4,
        num_stages=1,
    )
    torch.cuda.synchronize()

    expected = valuebook[lhs_index] @ valuebook[rhs_index]
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
    _assert_software_e2m1_dot(compiled)
    assert compiled.asm["hsaco"]
