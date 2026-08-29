import pytest
import torch
import triton
import triton.language as tl
from triton._internal_testing import is_ppu

_PPU0010_INT8_MMA = "ppu.mma.sync.aligned.m16n16k32.row.col.satfinite.s32.s8.s8.s32"
_PPU0010_BF16_MMA = "ppu.mma.sync.aligned.m16n16k16.row.col.f32.bf16.bf16.f32"


@triton.jit
def _int8_dot_quantize_kernel(lhs, rhs, out, OUTPUT_SCALE: tl.constexpr):
    offsets_m = tl.arange(0, 16)
    offsets_n = tl.arange(0, 16)
    offsets_k = tl.arange(0, 32)
    a = tl.load(lhs + offsets_m[:, None] * 32 + offsets_k[None, :])
    b = tl.load(rhs + offsets_k[:, None] * 16 + offsets_n[None, :])
    acc = tl.dot(a, b)
    result = tl.quantize(acc.to(tl.float32), OUTPUT_SCALE, dtype=tl.int8)
    tl.store(out + offsets_m[:, None] * 16 + offsets_n[None, :], result)


@triton.jit
def _int8_storage_bf16_dot_quantize_kernel(
    lhs,
    rhs,
    out,
    INPUT_SCALE: tl.constexpr,
    OUTPUT_SCALE: tl.constexpr,
):
    offsets_m = tl.arange(0, 16)
    offsets_n = tl.arange(0, 16)
    offsets_k = tl.arange(0, 32)
    lhs_q = tl.load(lhs + offsets_m[:, None] * 32 + offsets_k[None, :])
    rhs_q = tl.load(rhs + offsets_k[:, None] * 16 + offsets_n[None, :])
    a = tl.dequantize(lhs_q, INPUT_SCALE, dtype=tl.bfloat16)
    b = tl.dequantize(rhs_q, INPUT_SCALE, dtype=tl.bfloat16)
    acc = tl.dot(a, b, out_dtype=tl.float32)
    result = tl.quantize(acc, OUTPUT_SCALE, dtype=tl.int8)
    tl.store(out + offsets_m[:, None] * 16 + offsets_n[None, :], result)


@triton.jit
def _uint8_storage_bf16_dot_quantize_kernel(
    lhs,
    rhs,
    out,
    INPUT_SCALE: tl.constexpr,
    LHS_ZERO_POINT: tl.constexpr,
    RHS_ZERO_POINT: tl.constexpr,
    OUTPUT_SCALE: tl.constexpr,
    OUTPUT_ZERO_POINT: tl.constexpr,
):
    offsets_m = tl.arange(0, 16)
    offsets_n = tl.arange(0, 16)
    offsets_k = tl.arange(0, 32)
    lhs_q = tl.load(lhs + offsets_m[:, None] * 32 + offsets_k[None, :])
    rhs_q = tl.load(rhs + offsets_k[:, None] * 16 + offsets_n[None, :])
    a = tl.dequantize(
        lhs_q, INPUT_SCALE, dtype=tl.bfloat16, zero_point=LHS_ZERO_POINT
    )
    b = tl.dequantize(
        rhs_q, INPUT_SCALE, dtype=tl.bfloat16, zero_point=RHS_ZERO_POINT
    )
    acc = tl.dot(a, b, out_dtype=tl.float32)
    result = tl.quantize(
        acc,
        OUTPUT_SCALE,
        dtype=tl.uint8,
        zero_point=OUTPUT_ZERO_POINT,
    )
    tl.store(out + offsets_m[:, None] * 16 + offsets_n[None, :], result)


def _int8_inputs(device):
    lhs = (((torch.arange(16 * 32) * 7 + 3) % 17) - 8).reshape(16, 32).to(torch.int8)
    rhs = (((torch.arange(32 * 16) * 11 + 5) % 17) - 8).reshape(32, 16).to(torch.int8)
    return lhs.to(device), rhs.to(device)


def _uint8_inputs(device):
    lhs = ((torch.arange(16 * 32) * 13 + 17) % 256).reshape(16, 32).to(torch.uint8)
    rhs = ((torch.arange(32 * 16) * 29 + 5) % 256).reshape(32, 16).to(torch.uint8)
    return lhs.to(device), rhs.to(device)


def _quantize_reference(values, scale):
    return torch.clamp(torch.round(values.to(torch.float32) / scale), -128, 127).to(torch.int8)


def _quantize_uint8_reference(values, scale, zero_point):
    rounded = torch.round(values.to(torch.float32) / scale) + zero_point
    return torch.clamp(rounded, 0, 255).to(torch.uint8)


def test_ppu0010_native_int8_dot_has_explicit_int8_output_epilogue(device):
    if not is_ppu():
        pytest.skip("requires the PPU backend")

    lhs, rhs = _int8_inputs(device)
    output = torch.empty((16, 16), dtype=torch.int8, device=device)
    compiled = _int8_dot_quantize_kernel[(1, )](
        lhs,
        rhs,
        output,
        OUTPUT_SCALE=8.0,
        num_warps=1,
        num_stages=1,
    )

    accumulator = lhs.cpu().to(torch.int32) @ rhs.cpu().to(torch.int32)
    expected = _quantize_reference(accumulator, 8.0)
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    assert _PPU0010_INT8_MMA in compiled.asm["llir"]
    assert _PPU0010_BF16_MMA not in compiled.asm["llir"]
    assert compiled.asm["hgbin"]


def test_ppu0010_int8_storage_bf16_compute_int8_output(device):
    if not is_ppu():
        pytest.skip("requires the PPU backend")

    lhs, rhs = _int8_inputs(device)
    output = torch.empty((16, 16), dtype=torch.int8, device=device)
    compiled = _int8_storage_bf16_dot_quantize_kernel[(1, )](
        lhs,
        rhs,
        output,
        INPUT_SCALE=0.5,
        OUTPUT_SCALE=4.0,
        num_warps=1,
        num_stages=1,
    )

    lhs_dequantized = (lhs.cpu().to(torch.float32) * 0.5).to(torch.bfloat16).to(torch.float32)
    rhs_dequantized = (rhs.cpu().to(torch.float32) * 0.5).to(torch.bfloat16).to(torch.float32)
    expected = _quantize_reference(lhs_dequantized @ rhs_dequantized, 4.0)
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    assert compiled.asm["llir"].count(_PPU0010_BF16_MMA) == 2
    assert _PPU0010_INT8_MMA not in compiled.asm["llir"]
    assert compiled.asm["hgbin"]


def test_ppu0010_uint8_storage_bf16_compute_uint8_output(device):
    if not is_ppu():
        pytest.skip("requires the PPU backend")

    lhs, rhs = _uint8_inputs(device)
    output = torch.empty((16, 16), dtype=torch.uint8, device=device)
    compiled = _uint8_storage_bf16_dot_quantize_kernel[(1, )](
        lhs,
        rhs,
        output,
        INPUT_SCALE=0.0625,
        LHS_ZERO_POINT=127,
        RHS_ZERO_POINT=129,
        OUTPUT_SCALE=1.0,
        OUTPUT_ZERO_POINT=113,
        num_warps=1,
        num_stages=1,
    )

    lhs_dequantized = ((lhs.cpu().to(torch.float32) - 127) * 0.0625).to(torch.bfloat16)
    rhs_dequantized = ((rhs.cpu().to(torch.float32) - 129) * 0.0625).to(torch.bfloat16)
    expected = _quantize_uint8_reference(
        lhs_dequantized.to(torch.float32) @ rhs_dequantized.to(torch.float32),
        1.0,
        113,
    )
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    assert compiled.asm["llir"].count(_PPU0010_BF16_MMA) == 2
    assert _PPU0010_INT8_MMA not in compiled.asm["llir"]
    assert compiled.asm["hgbin"]
