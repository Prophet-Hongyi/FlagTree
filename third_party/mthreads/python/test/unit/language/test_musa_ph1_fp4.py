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

if not hasattr(libtriton, "mthreads"):
    pytest.skip("musa backend not built in libtriton", allow_module_level=True)

if "mthreads" not in backends:
    pytest.skip("musa backend not discovered", allow_module_level=True)

_PH1_TARGET = GPUTarget("musa", "ph1", 32)


@triton.jit
def _musa_e2m1_dot_scaled_kernel(lhs, rhs, out):
    offsets_m = tl.arange(0, 32)
    offsets_n = tl.arange(0, 32)
    offsets_k_packed = tl.arange(0, 32)
    a = tl.load(lhs + offsets_m[:, None] * 32 + offsets_k_packed[None, :])
    b = tl.load(rhs + offsets_k_packed[:, None] * 32 + offsets_n[None, :])
    acc = tl.zeros((32, 32), dtype=tl.float32)
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
    tl.store(out + offsets_m[:, None] * 32 + offsets_n[None, :], result)


@triton.jit
def _musa_lhs_scaled_e2m1_dot_kernel(lhs, lhs_scale, rhs, out):
    offsets_m = tl.arange(0, 32)
    offsets_n = tl.arange(0, 32)
    offsets_k_packed = tl.arange(0, 32)
    offsets_scale = tl.arange(0, 2)
    a = tl.load(lhs + offsets_m[:, None] * 32 + offsets_k_packed[None, :])
    a_scale = tl.load(lhs_scale + offsets_m[:, None] * 2 + offsets_scale[None, :])
    b = tl.load(rhs + offsets_k_packed[:, None] * 32 + offsets_n[None, :])
    acc = tl.zeros((32, 32), dtype=tl.float32)
    result = tl.dot_scaled(
        a,
        a_scale,
        "e2m1",
        b,
        None,
        "e2m1",
        acc,
        lhs_k_pack=True,
        rhs_k_pack=True,
    )
    tl.store(out + offsets_m[:, None] * 32 + offsets_n[None, :], result)


@triton.jit
def _musa_rhs_scaled_e2m1_dot_kernel(lhs, rhs, rhs_scale, out):
    offsets_m = tl.arange(0, 32)
    offsets_n = tl.arange(0, 32)
    offsets_k_packed = tl.arange(0, 32)
    offsets_scale = tl.arange(0, 2)
    a = tl.load(lhs + offsets_m[:, None] * 32 + offsets_k_packed[None, :])
    b = tl.load(rhs + offsets_k_packed[:, None] * 32 + offsets_n[None, :])
    b_scale = tl.load(rhs_scale + offsets_n[:, None] * 2 + offsets_scale[None, :])
    acc = tl.zeros((32, 32), dtype=tl.float32)
    result = tl.dot_scaled(
        a,
        None,
        "e2m1",
        b,
        b_scale,
        "e2m1",
        acc,
        lhs_k_pack=True,
        rhs_k_pack=True,
    )
    tl.store(out + offsets_m[:, None] * 32 + offsets_n[None, :], result)


@triton.jit
def _musa_both_scaled_e2m1_dot_kernel(lhs, lhs_scale, rhs, rhs_scale, out):
    offsets_m = tl.arange(0, 32)
    offsets_n = tl.arange(0, 32)
    offsets_k_packed = tl.arange(0, 32)
    offsets_scale = tl.arange(0, 2)
    a = tl.load(lhs + offsets_m[:, None] * 32 + offsets_k_packed[None, :])
    a_scale = tl.load(lhs_scale + offsets_m[:, None] * 2 + offsets_scale[None, :])
    b = tl.load(rhs + offsets_k_packed[:, None] * 32 + offsets_n[None, :])
    b_scale = tl.load(rhs_scale + offsets_n[:, None] * 2 + offsets_scale[None, :])
    acc = tl.zeros((32, 32), dtype=tl.float32)
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
    tl.store(out + offsets_m[:, None] * 32 + offsets_n[None, :], result)


@triton.jit
def _musa_both_scaled_e2m1_fast_math_kernel(lhs, lhs_scale, rhs, rhs_scale, out, FAST_MATH: tl.constexpr):
    offsets_m = tl.arange(0, 32)
    offsets_n = tl.arange(0, 32)
    offsets_k_packed = tl.arange(0, 32)
    offsets_scale = tl.arange(0, 2)
    a = tl.load(lhs + offsets_m[:, None] * 32 + offsets_k_packed[None, :])
    a_scale = tl.load(lhs_scale + offsets_m[:, None] * 2 + offsets_scale[None, :])
    b = tl.load(rhs + offsets_k_packed[:, None] * 32 + offsets_n[None, :])
    b_scale = tl.load(rhs_scale + offsets_n[:, None] * 2 + offsets_scale[None, :])
    acc = tl.zeros((32, 32), dtype=tl.float32)
    result = tl.dot_scaled(
        a,
        a_scale,
        "e2m1",
        b,
        b_scale,
        "e2m1",
        acc,
        fast_math=FAST_MATH,
        lhs_k_pack=True,
        rhs_k_pack=True,
    )
    tl.store(out + offsets_m[:, None] * 32 + offsets_n[None, :], result)


def _compile_ph1_e2m1_dot():
    source = ASTSource(
        fn=_musa_e2m1_dot_scaled_kernel,
        signature={"lhs": "*u8", "rhs": "*u8", "out": "*fp32"},
    )
    return triton.compile(source, target=_PH1_TARGET, options={"num_warps": 4})


def _compile_ph1_scaled_e2m1_dot(side):
    if side == "lhs":
        fn = _musa_lhs_scaled_e2m1_dot_kernel
        signature = {"lhs": "*u8", "lhs_scale": "*u8", "rhs": "*u8", "out": "*fp32"}
    elif side == "rhs":
        fn = _musa_rhs_scaled_e2m1_dot_kernel
        signature = {"lhs": "*u8", "rhs": "*u8", "rhs_scale": "*u8", "out": "*fp32"}
    else:
        assert side == "both"
        fn = _musa_both_scaled_e2m1_dot_kernel
        signature = {
            "lhs": "*u8",
            "lhs_scale": "*u8",
            "rhs": "*u8",
            "rhs_scale": "*u8",
            "out": "*fp32",
        }
    return triton.compile(
        ASTSource(fn=fn, signature=signature),
        target=_PH1_TARGET,
        options={"num_warps": 4},
    )


def _compile_ph1_fast_math(fast_math):
    source = ASTSource(
        fn=_musa_both_scaled_e2m1_fast_math_kernel,
        signature={
            "lhs": "*u8",
            "lhs_scale": "*u8",
            "rhs": "*u8",
            "rhs_scale": "*u8",
            "out": "*fp32",
            "FAST_MATH": "constexpr",
        },
        constexprs={"FAST_MATH": fast_math},
    )
    return triton.compile(source, target=_PH1_TARGET, options={"num_warps": 4})


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
    mma_lines = [line for line in compiled.asm["llir"].splitlines() if "llvm.musa." in line and ".mma" in line]
    assert mma_lines
    assert any("llvm.musa.sqmma.bfmma.m32n32k64.mma" in line for line in mma_lines)
    assert all(".e2m1." not in line for line in mma_lines)


def _assert_e8m0_nan_guard(compiled, fast_math):
    expected = 0 if fast_math else 2
    assert compiled.asm["ttgir"].count("arith.cmpi") == expected
    assert compiled.asm["ttgir"].count("arith.select") == expected


def test_musa_ph1_e2m1_dot_scaled_lowers_through_bf16_mma():
    compiled = _compile_ph1_e2m1_dot()
    _assert_software_e2m1_dot(compiled)
    assert compiled.asm["mubin"]


@pytest.mark.parametrize("side", ["lhs", "rhs", "both"])
def test_musa_ph1_scaled_e2m1_dot_lowers_through_bf16_mma(side):
    compiled = _compile_ph1_scaled_e2m1_dot(side)
    _assert_software_e2m1_dot(compiled)
    assert compiled.asm["mubin"]


@pytest.mark.parametrize("fast_math", [False, True])
def test_musa_ph1_e2m1_fast_math_lowers_through_bf16_mma(fast_math):
    compiled = _compile_ph1_fast_math(fast_math)
    _assert_software_e2m1_dot(compiled)
    _assert_e8m0_nan_guard(compiled, fast_math)
    assert compiled.asm["mubin"]


def test_musa_ph1_e2m1_dot_scaled_device():
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("requires a MUSA device")
    if tuple(torch.musa.get_device_capability(0)) != (3, 1):
        pytest.skip("requires a PH1 device")

    codebook = torch.tensor([0xC, 0xA, 0x0, 0x2, 0x4], dtype=torch.uint8)
    valuebook = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32)
    lhs_index = ((torch.arange(32 * 64) * 7 + 3) % len(codebook)).reshape(32, 64)
    rhs_index = ((torch.arange(64 * 32) * 11 + 1) % len(codebook)).reshape(64, 32)
    lhs_packed = _pack_along_k(codebook[lhs_index], dim=1).to("musa")
    rhs_packed = _pack_along_k(codebook[rhs_index], dim=0).to("musa")
    output = torch.empty((32, 32), dtype=torch.float32, device="musa")

    compiled = _musa_e2m1_dot_scaled_kernel[(1, )](
        lhs_packed,
        rhs_packed,
        output,
        num_warps=4,
    )
    torch.musa.synchronize()

    expected = valuebook[lhs_index] @ valuebook[rhs_index]
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    _assert_software_e2m1_dot(compiled)
    assert compiled.asm["mubin"]


@pytest.mark.parametrize("side", ["lhs", "rhs", "both"])
def test_musa_ph1_scaled_e2m1_dot_device(side):
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("requires a MUSA device")
    if tuple(torch.musa.get_device_capability(0)) != (3, 1):
        pytest.skip("requires a PH1 device")

    codebook = torch.arange(8, dtype=torch.uint8)
    valuebook = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
    rows = torch.arange(32)[:, None]
    cols = torch.arange(32)[None, :]
    lhs_k = torch.arange(64)[None, :]
    rhs_k = torch.arange(64)[:, None]
    lhs_index = (rows * 3 + lhs_k * 5 + lhs_k // 7) % 8
    rhs_index = (rhs_k * 3 + cols * 5 + rhs_k // 11) % 8
    lhs_values = valuebook[lhs_index]
    rhs_values = valuebook[rhs_index]
    lhs_packed = _pack_along_k(codebook[lhs_index], dim=1).to("musa")
    rhs_packed = _pack_along_k(codebook[rhs_index], dim=0).to("musa")
    lhs_scale_codes = torch.tensor(
        [[0x7E + ((row + 2 * block) % 4) for block in range(2)] for row in range(32)],
        dtype=torch.uint8,
    )
    rhs_scale_codes = torch.tensor(
        [[0x7E + ((3 * col + block + 1) % 4) for block in range(2)] for col in range(32)],
        dtype=torch.uint8,
    )
    lhs_scale_values = torch.pow(2.0, lhs_scale_codes.to(torch.int32) - 127).to(torch.float32)
    rhs_scale_values = torch.pow(2.0, rhs_scale_codes.to(torch.int32) - 127).to(torch.float32)
    output = torch.empty((32, 32), dtype=torch.float32, device="musa")

    if side == "lhs":
        compiled = _musa_lhs_scaled_e2m1_dot_kernel[(1, )](
            lhs_packed,
            lhs_scale_codes.to("musa"),
            rhs_packed,
            output,
            num_warps=4,
        )
        expected = (lhs_values * lhs_scale_values.repeat_interleave(32, dim=1)) @ rhs_values
    elif side == "rhs":
        compiled = _musa_rhs_scaled_e2m1_dot_kernel[(1, )](
            lhs_packed,
            rhs_packed,
            rhs_scale_codes.to("musa"),
            output,
            num_warps=4,
        )
        scaled_rhs = rhs_values * rhs_scale_values.repeat_interleave(32, dim=1).transpose(0, 1)
        expected = lhs_values @ scaled_rhs
    else:
        compiled = _musa_both_scaled_e2m1_dot_kernel[(1, )](
            lhs_packed,
            lhs_scale_codes.to("musa"),
            rhs_packed,
            rhs_scale_codes.to("musa"),
            output,
            num_warps=4,
        )
        scaled_lhs = lhs_values * lhs_scale_values.repeat_interleave(32, dim=1)
        scaled_rhs = rhs_values * rhs_scale_values.repeat_interleave(32, dim=1).transpose(0, 1)
        expected = scaled_lhs @ scaled_rhs
    torch.musa.synchronize()

    actual = output.cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(actual, lhs_values @ rhs_values)
    _assert_software_e2m1_dot(compiled)
    assert compiled.asm["mubin"]


@pytest.mark.parametrize("fast_math", [False, True])
def test_musa_ph1_e2m1_scale_special_value_device(fast_math):
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("requires a MUSA device")
    if tuple(torch.musa.get_device_capability(0)) != (3, 1):
        pytest.skip("requires a PH1 device")

    lhs_packed = torch.full((32, 32), 0x22, dtype=torch.uint8, device="musa")
    rhs_packed = torch.full((32, 32), 0x22, dtype=torch.uint8, device="musa")
    scale_code = 0x7F if fast_math else 0xFF
    lhs_scale = torch.full((32, 2), scale_code, dtype=torch.uint8, device="musa")
    rhs_scale = torch.full((32, 2), scale_code, dtype=torch.uint8, device="musa")
    output = torch.empty((32, 32), dtype=torch.float32, device="musa")

    compiled = _musa_both_scaled_e2m1_fast_math_kernel[(1, )](
        lhs_packed,
        lhs_scale,
        rhs_packed,
        rhs_scale,
        output,
        FAST_MATH=fast_math,
        num_warps=4,
    )
    torch.musa.synchronize()

    actual = output.cpu()
    if fast_math:
        torch.testing.assert_close(actual, torch.full_like(actual, 64.0), rtol=0, atol=0)
    else:
        assert torch.isnan(actual).all()
    _assert_software_e2m1_dot(compiled)
    _assert_e8m0_nan_guard(compiled, fast_math)
    assert compiled.asm["mubin"]
