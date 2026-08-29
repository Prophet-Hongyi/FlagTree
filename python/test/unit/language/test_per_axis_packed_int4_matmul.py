import numpy as np
import pytest
import torch
import triton
import triton.language as tl


BLOCK_M = 32
BLOCK_N = 32
PACKED_K = 32
LOGICAL_K = 2 * PACKED_K


@triton.jit
def _quantize_pack_per_axis_kernel(
    input_ptr,
    scale_ptr,
    zero_point_ptr,
    packed_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PACKED_K: tl.constexpr,
    SIGNED: tl.constexpr,
    PACK_LHS: tl.constexpr,
):
    reduction = tl.arange(0, PACKED_K)
    if PACK_LHS:
        axis = tl.arange(0, BLOCK_M)
        low_offsets = axis[:, None] * (2 * PACKED_K) + 2 * reduction[None, :]
        high_offsets = low_offsets + 1
        packed_offsets = axis[:, None] * PACKED_K + reduction[None, :]
        scale = tl.load(scale_ptr + axis)[:, None]
        zero_point = tl.load(zero_point_ptr + axis)[:, None]
    else:
        axis = tl.arange(0, BLOCK_N)
        low_offsets = (2 * reduction[:, None]) * BLOCK_N + axis[None, :]
        high_offsets = low_offsets + BLOCK_N
        packed_offsets = reduction[:, None] * BLOCK_N + axis[None, :]
        scale = tl.load(scale_ptr + axis)[None, :]
        zero_point = tl.load(zero_point_ptr + axis)[None, :]

    low = tl.load(input_ptr + low_offsets)
    high = tl.load(input_ptr + high_offsets)
    if SIGNED:
        low = tl.quantize(low, scale, dtype=tl.int8, zero_point=zero_point, qmin=-8, qmax=7)
        high = tl.quantize(high, scale, dtype=tl.int8, zero_point=zero_point, qmin=-8, qmax=7)
    else:
        low = tl.quantize(low, scale, dtype=tl.uint8, zero_point=zero_point, qmin=0, qmax=15)
        high = tl.quantize(high, scale, dtype=tl.uint8, zero_point=zero_point, qmin=0, qmax=15)

    packed = tl.pack_int4(low, high, signed=SIGNED)
    tl.store(packed_ptr + packed_offsets, packed)


@triton.jit
def _per_axis_packed_int4_matmul_kernel(
    lhs_ptr,
    rhs_ptr,
    lhs_scale_ptr,
    rhs_scale_ptr,
    lhs_zero_point_ptr,
    rhs_zero_point_ptr,
    output_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PACKED_K: tl.constexpr,
    SIGNED: tl.constexpr,
):
    rows = tl.arange(0, BLOCK_M)
    columns = tl.arange(0, BLOCK_N)
    reduction = tl.arange(0, PACKED_K)

    lhs_packed = tl.load(lhs_ptr + rows[:, None] * PACKED_K + reduction[None, :])
    rhs_packed = tl.load(rhs_ptr + reduction[:, None] * BLOCK_N + columns[None, :])
    lhs_low, lhs_high = tl.unpack_int4(lhs_packed, signed=SIGNED)
    rhs_low, rhs_high = tl.unpack_int4(rhs_packed, signed=SIGNED)

    if not SIGNED:
        lhs_zero_point = tl.load(lhs_zero_point_ptr + rows).to(tl.int16)[:, None]
        rhs_zero_point = tl.load(rhs_zero_point_ptr + columns).to(tl.int16)[None, :]
        lhs_low = (lhs_low.to(tl.int16) - lhs_zero_point).to(tl.int8)
        lhs_high = (lhs_high.to(tl.int16) - lhs_zero_point).to(tl.int8)
        rhs_low = (rhs_low.to(tl.int16) - rhs_zero_point).to(tl.int8)
        rhs_high = (rhs_high.to(tl.int16) - rhs_zero_point).to(tl.int8)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    accumulator = tl.dot(lhs_low, rhs_low, accumulator, out_dtype=tl.int32)
    accumulator = tl.dot(lhs_high, rhs_high, accumulator, out_dtype=tl.int32)

    lhs_scale = tl.load(lhs_scale_ptr + rows)[:, None]
    rhs_scale = tl.load(rhs_scale_ptr + columns)[None, :]
    output = accumulator.to(tl.float32) * lhs_scale * rhs_scale
    tl.store(output_ptr + rows[:, None] * BLOCK_N + columns[None, :], output)


def _pack_adjacent_k_values(low, high):
    low_bits = (low.to(torch.int16) & 0xF).to(torch.uint8)
    high_bits = (high.to(torch.int16) & 0xF).to(torch.uint8)
    return low_bits | (high_bits << 4)


def _quantize_reference(values, scale, zero_point, signed):
    qmin, qmax = (-8, 7) if signed else (0, 15)
    dtype = torch.int8 if signed else torch.uint8
    return (torch.round(values / scale) + zero_point).clamp(qmin, qmax).to(dtype)


def _make_per_axis_inputs(signed):
    lhs_scale = torch.tensor([0.125, 0.25, 0.5, 1.0], dtype=torch.float32).repeat(BLOCK_M // 4)
    rhs_scale = torch.tensor([0.25, 0.5, 1.0, 2.0], dtype=torch.float32).repeat(BLOCK_N // 4)
    if signed:
        lhs_zero_point = torch.zeros(BLOCK_M, dtype=torch.int32)
        rhs_zero_point = torch.zeros(BLOCK_N, dtype=torch.int32)
    else:
        lhs_zero_point = (torch.arange(BLOCK_M, dtype=torch.int32) % 7) + 4
        rhs_zero_point = (torch.arange(BLOCK_N, dtype=torch.int32) % 8) + 3

    lhs_steps = (
        (torch.arange(BLOCK_M * LOGICAL_K, dtype=torch.int64) * 7 + 3) % 41 - 20
    ).reshape(BLOCK_M, LOGICAL_K)
    rhs_steps = (
        (torch.arange(LOGICAL_K * BLOCK_N, dtype=torch.int64) * 11 + 5) % 43 - 21
    ).reshape(LOGICAL_K, BLOCK_N)
    lhs = lhs_steps.to(torch.float32) * 0.5 * lhs_scale[:, None]
    rhs = rhs_steps.to(torch.float32) * 0.5 * rhs_scale[None, :]

    lhs_quantized = _quantize_reference(lhs, lhs_scale[:, None], lhs_zero_point[:, None], signed)
    rhs_quantized = _quantize_reference(rhs, rhs_scale[None, :], rhs_zero_point[None, :], signed)
    return lhs, rhs, lhs_scale, rhs_scale, lhs_zero_point, rhs_zero_point, lhs_quantized, rhs_quantized


@pytest.mark.parametrize("signed", [True, False], ids=["symmetric-int4", "affine-uint4"])
def test_per_axis_quantize_pack_int4_matmul(device, signed):
    (
        lhs,
        rhs,
        lhs_scale,
        rhs_scale,
        lhs_zero_point,
        rhs_zero_point,
        lhs_quantized,
        rhs_quantized,
    ) = _make_per_axis_inputs(signed)

    lhs_packed = torch.empty((BLOCK_M, PACKED_K), dtype=torch.uint8, device=device)
    rhs_packed = torch.empty((PACKED_K, BLOCK_N), dtype=torch.uint8, device=device)
    lhs_pack_program = _quantize_pack_per_axis_kernel[(1, )](
        lhs.to(device),
        lhs_scale.to(device),
        lhs_zero_point.to(device),
        lhs_packed,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        PACKED_K=PACKED_K,
        SIGNED=signed,
        PACK_LHS=True,
    )
    rhs_pack_program = _quantize_pack_per_axis_kernel[(1, )](
        rhs.to(device),
        rhs_scale.to(device),
        rhs_zero_point.to(device),
        rhs_packed,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        PACKED_K=PACKED_K,
        SIGNED=signed,
        PACK_LHS=False,
    )

    expected_lhs_packed = _pack_adjacent_k_values(lhs_quantized[:, 0::2], lhs_quantized[:, 1::2])
    expected_rhs_packed = _pack_adjacent_k_values(rhs_quantized[0::2, :], rhs_quantized[1::2, :])
    torch.testing.assert_close(lhs_packed.cpu(), expected_lhs_packed, rtol=0, atol=0)
    torch.testing.assert_close(rhs_packed.cpu(), expected_rhs_packed, rtol=0, atol=0)

    output = torch.empty((BLOCK_M, BLOCK_N), dtype=torch.float32, device=device)
    dot_program = _per_axis_packed_int4_matmul_kernel[(1, )](
        lhs_packed,
        rhs_packed,
        lhs_scale.to(device),
        rhs_scale.to(device),
        lhs_zero_point.to(device),
        rhs_zero_point.to(device),
        output,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        PACKED_K=PACKED_K,
        SIGNED=signed,
    )

    lhs_centered = lhs_quantized.to(torch.int32) - lhs_zero_point[:, None]
    rhs_centered = rhs_quantized.to(torch.int32) - rhs_zero_point[None, :]
    expected_accumulator = np.matmul(lhs_centered.numpy(), rhs_centered.numpy())
    expected = torch.from_numpy(expected_accumulator).to(torch.float32)
    expected *= lhs_scale[:, None] * rhs_scale[None, :]
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)

    for program in (lhs_pack_program, rhs_pack_program):
        assert "arith.fptosi" in program.asm["ttir"]
        assert "xi8" in program.asm["ttir"]
    assert dot_program.asm["ttir"].count("tt.dot") == 2
    assert "xi32" in dot_program.asm["ttir"]
    assert "f32" in dot_program.asm["ttir"]
