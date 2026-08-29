import numpy as np
import pytest
import torch
import triton
import triton.language as tl


PACKED_K = 32
MATMUL_SHAPES = ((16, 32), (32, 32), (32, 64))


@triton.jit
def _packed_4bit_storage_io_matmul_kernel(
    lhs_ptr,
    rhs_ptr,
    lhs_scale_ptr,
    rhs_scale_ptr,
    output_scale_ptr,
    output_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PACKED_N: tl.constexpr,
    PACKED_K: tl.constexpr,
    SIGNED: tl.constexpr,
    INPUT_ZERO_POINT: tl.constexpr,
    OUTPUT_ZERO_POINT: tl.constexpr,
):
    rows = tl.arange(0, BLOCK_M)
    packed_columns = tl.arange(0, PACKED_N)
    reduction = tl.arange(0, PACKED_K)

    lhs_packed = tl.load(lhs_ptr + rows[:, None] * PACKED_K + reduction[None, :])
    lhs_low, lhs_high = tl.unpack_int4(lhs_packed, signed=SIGNED)

    low_columns = 2 * packed_columns
    high_columns = low_columns + 1
    rhs_low_packed = tl.load(rhs_ptr + reduction[:, None] * BLOCK_N + low_columns[None, :])
    rhs_high_packed = tl.load(rhs_ptr + reduction[:, None] * BLOCK_N + high_columns[None, :])
    rhs_low_low, rhs_low_high = tl.unpack_int4(rhs_low_packed, signed=SIGNED)
    rhs_high_low, rhs_high_high = tl.unpack_int4(rhs_high_packed, signed=SIGNED)

    if not SIGNED:
        lhs_low = (lhs_low.to(tl.int16) - INPUT_ZERO_POINT).to(tl.int8)
        lhs_high = (lhs_high.to(tl.int16) - INPUT_ZERO_POINT).to(tl.int8)
        rhs_low_low = (rhs_low_low.to(tl.int16) - INPUT_ZERO_POINT).to(tl.int8)
        rhs_low_high = (rhs_low_high.to(tl.int16) - INPUT_ZERO_POINT).to(tl.int8)
        rhs_high_low = (rhs_high_low.to(tl.int16) - INPUT_ZERO_POINT).to(tl.int8)
        rhs_high_high = (rhs_high_high.to(tl.int16) - INPUT_ZERO_POINT).to(tl.int8)

    low_accumulator = tl.zeros((BLOCK_M, PACKED_N), dtype=tl.int32)
    low_accumulator = tl.dot(lhs_low, rhs_low_low, low_accumulator, out_dtype=tl.int32)
    low_accumulator = tl.dot(lhs_high, rhs_low_high, low_accumulator, out_dtype=tl.int32)
    high_accumulator = tl.zeros((BLOCK_M, PACKED_N), dtype=tl.int32)
    high_accumulator = tl.dot(lhs_low, rhs_high_low, high_accumulator, out_dtype=tl.int32)
    high_accumulator = tl.dot(lhs_high, rhs_high_high, high_accumulator, out_dtype=tl.int32)

    lhs_scale = tl.load(lhs_scale_ptr + rows)[:, None]
    low_rhs_scale = tl.load(rhs_scale_ptr + low_columns)[None, :]
    high_rhs_scale = tl.load(rhs_scale_ptr + high_columns)[None, :]
    output_scale = tl.load(output_scale_ptr + rows)[:, None]
    low = low_accumulator.to(tl.float32) * lhs_scale * low_rhs_scale
    high = high_accumulator.to(tl.float32) * lhs_scale * high_rhs_scale

    if SIGNED:
        low = tl.quantize(
            low,
            output_scale,
            dtype=tl.int8,
            zero_point=OUTPUT_ZERO_POINT,
            qmin=-8,
            qmax=7,
            rounding="rtne",
        )
        high = tl.quantize(
            high,
            output_scale,
            dtype=tl.int8,
            zero_point=OUTPUT_ZERO_POINT,
            qmin=-8,
            qmax=7,
            rounding="rtne",
        )
    else:
        low = tl.quantize(
            low,
            output_scale,
            dtype=tl.uint8,
            zero_point=OUTPUT_ZERO_POINT,
            qmin=0,
            qmax=15,
            rounding="rtne",
        )
        high = tl.quantize(
            high,
            output_scale,
            dtype=tl.uint8,
            zero_point=OUTPUT_ZERO_POINT,
            qmin=0,
            qmax=15,
            rounding="rtne",
        )

    packed = tl.pack_int4(low, high, signed=SIGNED)
    tl.store(output_ptr + rows[:, None] * PACKED_N + packed_columns[None, :], packed)


def _pack_adjacent(values, axis):
    low = values.index_select(axis, torch.arange(0, values.shape[axis], 2))
    high = values.index_select(axis, torch.arange(1, values.shape[axis], 2))
    low_bits = (low.to(torch.int16) & 0xF).to(torch.uint8)
    high_bits = (high.to(torch.int16) & 0xF).to(torch.uint8)
    return low_bits | (high_bits << 4)


def _make_inputs(signed, block_m, block_n):
    lhs_centered = (
        (torch.arange(block_m * 2 * PACKED_K, dtype=torch.int64) * 7 + 3) % 5
        - 2
    ).reshape(block_m, 2 * PACKED_K)
    rhs_centered = (
        (torch.arange(2 * PACKED_K * block_n, dtype=torch.int64) * 11 + 5) % 5
        - 2
    ).reshape(2 * PACKED_K, block_n)
    input_zero_point = 0 if signed else 7
    output_zero_point = 0 if signed else 7
    dtype = torch.int8 if signed else torch.uint8
    lhs = (lhs_centered + input_zero_point).to(dtype)
    rhs = (rhs_centered + input_zero_point).to(dtype)
    lhs_scale = torch.tensor([0.125, 0.25, 0.5, 1.0], dtype=torch.float32).repeat(
        block_m // 4
    )
    rhs_scale = torch.tensor([0.25, 0.5, 1.0, 0.125], dtype=torch.float32).repeat(
        block_n // 4
    )
    output_scale = torch.tensor([3.0, 5.0, 7.0, 9.0], dtype=torch.float32).repeat(
        block_m // 4
    )
    return (
        lhs,
        rhs,
        lhs_centered.to(torch.int32),
        rhs_centered.to(torch.int32),
        lhs_scale,
        rhs_scale,
        output_scale,
        input_zero_point,
        output_zero_point,
    )


@pytest.mark.parametrize("signed", [True, False], ids=["int4", "uint4"])
@pytest.mark.parametrize(
    "block_m,block_n", MATMUL_SHAPES, ids=["m16n32", "m32n32", "m32n64"]
)
def test_packed_4bit_storage_input_output_matmul(device, signed, block_m, block_n):
    (
        lhs,
        rhs,
        lhs_centered,
        rhs_centered,
        lhs_scale,
        rhs_scale,
        output_scale,
        input_zero_point,
        output_zero_point,
    ) = _make_inputs(signed, block_m, block_n)
    lhs_packed = _pack_adjacent(lhs, axis=1)
    rhs_packed = _pack_adjacent(rhs, axis=0)
    packed_n = block_n // 2
    output_packed = torch.empty((block_m, packed_n), dtype=torch.uint8, device=device)

    program = _packed_4bit_storage_io_matmul_kernel[(1, )](
        lhs_packed.to(device),
        rhs_packed.to(device),
        lhs_scale.to(device),
        rhs_scale.to(device),
        output_scale.to(device),
        output_packed,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        PACKED_N=packed_n,
        PACKED_K=PACKED_K,
        SIGNED=signed,
        INPUT_ZERO_POINT=input_zero_point,
        OUTPUT_ZERO_POINT=output_zero_point,
    )

    accumulator = torch.from_numpy(
        np.matmul(lhs_centered.numpy(), rhs_centered.numpy())
    ).to(torch.float32)
    expected = accumulator * lhs_scale[:, None] * rhs_scale[None, :]
    qmin, qmax = (-8, 7) if signed else (0, 15)
    expected = (
        torch.round(expected / output_scale[:, None]) + output_zero_point
    ).clamp(qmin, qmax).to(torch.int8 if signed else torch.uint8)
    expected_packed = _pack_adjacent(expected, axis=1)

    torch.testing.assert_close(output_packed.cpu(), expected_packed, rtol=0, atol=0)
    assert program.asm["ttir"].count("tt.dot") == 4
    assert "xi8" in program.asm["ttir"]
    assert "xi32" in program.asm["ttir"]
    assert "arith.fptosi" in program.asm["ttir"]
