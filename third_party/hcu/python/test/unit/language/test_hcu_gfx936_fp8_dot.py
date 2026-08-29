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
_NATIVE_FP8_DOWNCAST_ASM = (
    "v_cvt_pk_fp8_f32",
    "v_cvt_pk_bf8_f32",
    "v_cvt_scalef32_pk_fp8",
    "v_cvt_scalef32_pk_bf8",
)
_MIXED_OCP_FP8_CASES = (
    (
        "fp8e4nv",
        "fp8e5",
        tl.float8e4nv,
        tl.float8e5,
        [0xC0, 0xB8, 0x00, 0x38, 0x40],
        [0xC0, 0xBC, 0x00, 0x3C, 0x40],
    ),
    (
        "fp8e5",
        "fp8e4nv",
        tl.float8e5,
        tl.float8e4nv,
        [0xC0, 0xBC, 0x00, 0x3C, 0x40],
        [0xC0, 0xB8, 0x00, 0x38, 0x40],
    ),
)


@triton.jit
def _hcu_fp8_dot_kernel(lhs, rhs, out):
    offs_m = tl.arange(0, 16)
    offs_n = tl.arange(0, 16)
    offs_k = tl.arange(0, 32)
    a = tl.load(lhs + offs_m[:, None] * 32 + offs_k[None, :])
    b = tl.load(rhs + offs_k[:, None] * 16 + offs_n[None, :])
    acc = tl.zeros((16, 16), dtype=tl.float32)
    result = tl.dot(a, b, acc, out_dtype=tl.float32)
    tl.store(out + offs_m[:, None] * 16 + offs_n[None, :], result)


def _compile_gfx936(dtype, out_dtype="fp32", rhs_dtype=None):
    rhs_dtype = dtype if rhs_dtype is None else rhs_dtype
    src = ASTSource(
        fn=_hcu_fp8_dot_kernel,
        signature={
            "lhs": f"*{dtype}",
            "rhs": f"*{rhs_dtype}",
            "out": f"*{out_dtype}",
        },
    )
    return triton.compile(
        src,
        target=_GFX936_TARGET,
        options={"num_warps": 4, "num_stages": 1},
    )


@pytest.mark.parametrize("dtype", ["fp8e4nv", "fp8e5"])
def test_gfx936_ocp_fp8_dot_lowers_through_fp16_mmac(dtype):
    compiled = _compile_gfx936(dtype)

    assert compiled.asm["ttgir"].count("tt.fp_to_fp") == 2
    assert compiled.asm["llir"].count(_FP16_MMAC_LLIR + "(") == 3
    assert compiled.asm["amdgcn"].count(_FP16_MMAC_ASM) == 2
    assert ".fp8." not in compiled.asm["llir"]
    assert ".bf8." not in compiled.asm["llir"]
    assert "v_mmac_f32_16x16x32_fp8" not in compiled.asm["amdgcn"]
    assert "v_mmac_f32_16x16x32_bf8" not in compiled.asm["amdgcn"]
    assert compiled.asm["hsaco"]


@pytest.mark.parametrize(
    ("triton_dtype", "codes"),
    [
        (tl.float8e4nv, [0xC0, 0xB8, 0x00, 0x38, 0x40]),
        (tl.float8e5, [0xC0, 0xBC, 0x00, 0x3C, 0x40]),
    ],
)
def test_gfx936_ocp_fp8_dot_device(triton_dtype, codes):
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("requires an HCU device")

    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "hip" or target.arch != "gfx936":
        pytest.skip("requires a gfx936 HCU device")

    values = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32)
    raw_codes = torch.tensor(codes, dtype=torch.uint8)
    lhs_index = ((torch.arange(16 * 32) * 7 + 3) % len(codes)).reshape(16, 32)
    rhs_index = ((torch.arange(32 * 16) * 11 + 1) % len(codes)).reshape(32, 16)
    lhs_raw = raw_codes[lhs_index].to("cuda")
    rhs_raw = raw_codes[rhs_index].to("cuda")
    out = torch.empty((16, 16), dtype=torch.float32, device="cuda")

    compiled = _hcu_fp8_dot_kernel[(1, )](
        triton.reinterpret(lhs_raw, triton_dtype),
        triton.reinterpret(rhs_raw, triton_dtype),
        out,
        num_warps=4,
        num_stages=1,
    )
    torch.cuda.synchronize()

    expected = values[lhs_index] @ values[rhs_index]
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
    assert compiled.asm["ttgir"].count("tt.fp_to_fp") == 2
    assert compiled.asm["amdgcn"].count(_FP16_MMAC_ASM) == 2
    assert "v_mmac_f32_16x16x32_fp8" not in compiled.asm["amdgcn"]
    assert "v_mmac_f32_16x16x32_bf8" not in compiled.asm["amdgcn"]
    assert compiled.asm["hsaco"]


@pytest.mark.parametrize(
    (
        "lhs_dtype",
        "rhs_dtype",
        "_lhs_tl_dtype",
        "_rhs_tl_dtype",
        "_lhs_codes",
        "_rhs_codes",
    ),
    _MIXED_OCP_FP8_CASES,
)
def test_gfx936_mixed_ocp_fp8_dot_lowers_through_fp16_mmac(
    lhs_dtype, rhs_dtype, _lhs_tl_dtype, _rhs_tl_dtype, _lhs_codes, _rhs_codes
):
    compiled = _compile_gfx936(lhs_dtype, rhs_dtype=rhs_dtype)

    assert compiled.asm["ttgir"].count("tt.fp_to_fp") == 2
    assert compiled.asm["llir"].count(_FP16_MMAC_LLIR + "(") == 3
    assert compiled.asm["amdgcn"].count(_FP16_MMAC_ASM) == 2
    assert ".fp8." not in compiled.asm["llir"]
    assert ".bf8." not in compiled.asm["llir"]
    assert "v_mmac_f32_16x16x32_fp8" not in compiled.asm["amdgcn"]
    assert "v_mmac_f32_16x16x32_bf8" not in compiled.asm["amdgcn"]
    assert compiled.asm["hsaco"]


@pytest.mark.parametrize(
    (
        "_lhs_dtype",
        "_rhs_dtype",
        "lhs_tl_dtype",
        "rhs_tl_dtype",
        "lhs_codes",
        "rhs_codes",
    ),
    _MIXED_OCP_FP8_CASES,
)
def test_gfx936_mixed_ocp_fp8_dot_device(
    _lhs_dtype, _rhs_dtype, lhs_tl_dtype, rhs_tl_dtype, lhs_codes, rhs_codes
):
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("requires an HCU device")

    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "hip" or target.arch != "gfx936":
        pytest.skip("requires a gfx936 HCU device")

    values = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32)
    lhs_codes = torch.tensor(lhs_codes, dtype=torch.uint8)
    rhs_codes = torch.tensor(rhs_codes, dtype=torch.uint8)
    lhs_index = ((torch.arange(16 * 32) * 7 + 3) % len(lhs_codes)).reshape(16, 32)
    rhs_index = ((torch.arange(32 * 16) * 11 + 1) % len(rhs_codes)).reshape(32, 16)
    lhs_raw = lhs_codes[lhs_index].to("cuda")
    rhs_raw = rhs_codes[rhs_index].to("cuda")
    out = torch.empty((16, 16), dtype=torch.float32, device="cuda")

    compiled = _hcu_fp8_dot_kernel[(1, )](
        triton.reinterpret(lhs_raw, lhs_tl_dtype),
        triton.reinterpret(rhs_raw, rhs_tl_dtype),
        out,
        num_warps=4,
        num_stages=1,
    )
    torch.cuda.synchronize()

    expected = values[lhs_index] @ values[rhs_index]
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
    assert compiled.asm["ttgir"].count("tt.fp_to_fp") == 2
    assert compiled.asm["amdgcn"].count(_FP16_MMAC_ASM) == 2
    assert "v_mmac_f32_16x16x32_fp8" not in compiled.asm["amdgcn"]
    assert "v_mmac_f32_16x16x32_bf8" not in compiled.asm["amdgcn"]
    assert compiled.asm["hsaco"]


@pytest.mark.parametrize("dtype", ["fp8e4nv", "fp8e5"])
def test_gfx936_ocp_fp8_dot_output_uses_software_conversion(dtype):
    compiled = _compile_gfx936(dtype, out_dtype=dtype)

    assert compiled.asm["ttgir"].count("tt.fp_to_fp") == 3
    assert compiled.asm["llir"].count(_FP16_MMAC_LLIR + "(") == 3
    assert compiled.asm["amdgcn"].count(_FP16_MMAC_ASM) == 2
    assert "v_mmac_f32_16x16x32_fp8" not in compiled.asm["amdgcn"]
    assert "v_mmac_f32_16x16x32_bf8" not in compiled.asm["amdgcn"]
    assert not any(
        mnemonic in compiled.asm["amdgcn"] for mnemonic in _NATIVE_FP8_DOWNCAST_ASM
    )
    assert compiled.asm["hsaco"]


@pytest.mark.parametrize(
    ("triton_dtype", "codes"),
    [
        (tl.float8e4nv, [0xC0, 0xB8, 0x00, 0x38, 0x40]),
        (tl.float8e5, [0xC0, 0xBC, 0x00, 0x3C, 0x40]),
    ],
)
def test_gfx936_ocp_fp8_dot_output_device(triton_dtype, codes):
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("requires an HCU device")

    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "hip" or target.arch != "gfx936":
        pytest.skip("requires a gfx936 HCU device")

    raw_codes = torch.tensor(codes, dtype=torch.uint8)
    lhs_index = ((torch.arange(16 * 32) * 7 + 3) % len(codes)).reshape(16, 32)
    lhs_raw = raw_codes[lhs_index].to("cuda")
    rhs_raw = torch.zeros((32, 16), dtype=torch.uint8)
    rhs_raw.diagonal().fill_(codes[3])
    rhs_raw = rhs_raw.to("cuda")
    out_raw = torch.empty((16, 16), dtype=torch.uint8, device="cuda")

    compiled = _hcu_fp8_dot_kernel[(1, )](
        triton.reinterpret(lhs_raw, triton_dtype),
        triton.reinterpret(rhs_raw, triton_dtype),
        triton.reinterpret(out_raw, triton_dtype),
        num_warps=4,
        num_stages=1,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(out_raw.cpu(), lhs_raw[:, :16].cpu(), rtol=0, atol=0)
    assert compiled.asm["ttgir"].count("tt.fp_to_fp") == 3
    assert compiled.asm["amdgcn"].count(_FP16_MMAC_ASM) == 2
    assert "v_mmac_f32_16x16x32_fp8" not in compiled.asm["amdgcn"]
    assert "v_mmac_f32_16x16x32_bf8" not in compiled.asm["amdgcn"]
    assert not any(
        mnemonic in compiled.asm["amdgcn"] for mnemonic in _NATIVE_FP8_DOWNCAST_ASM
    )
    assert compiled.asm["hsaco"]
