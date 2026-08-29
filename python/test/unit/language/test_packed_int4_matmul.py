import numpy as np
import pytest
import torch
import triton
import triton.language as tl


BLOCK_M = 32
BLOCK_N = 32
PACKED_K = 32
LOGICAL_K = 2 * PACKED_K
INT4_SCALE = 0.25


@triton.jit
def _packed_int4_dot_kernel(
    lhs_ptr,
    rhs_ptr,
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
        # UINT4 values are in [0, 15], so this conversion is exact. tl.dot only
        # admits signed INT8 and must not infer unsigned arithmetic from i8 IR.
        lhs_low = lhs_low.to(tl.int8)
        lhs_high = lhs_high.to(tl.int8)
        rhs_low = rhs_low.to(tl.int8)
        rhs_high = rhs_high.to(tl.int8)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    accumulator = tl.dot(lhs_low, rhs_low, accumulator, out_dtype=tl.int32)
    accumulator = tl.dot(lhs_high, rhs_high, accumulator, out_dtype=tl.int32)
    tl.store(output_ptr + rows[:, None] * BLOCK_N + columns[None, :], accumulator)


@triton.jit
def _quantize_pack_int4_kernel(
    input_ptr,
    packed_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PACKED_K: tl.constexpr,
    LOGICAL_K: tl.constexpr,
    SCALE: tl.constexpr,
    SIGNED: tl.constexpr,
    PACK_LHS: tl.constexpr,
):
    reduction = tl.arange(0, PACKED_K)
    high_valid = 2 * reduction + 1 < LOGICAL_K
    if PACK_LHS:
        rows = tl.arange(0, BLOCK_M)
        low_offsets = rows[:, None] * LOGICAL_K + 2 * reduction[None, :]
        high_offsets = low_offsets + 1
        packed_offsets = rows[:, None] * PACKED_K + reduction[None, :]
        high_mask = high_valid[None, :]
    else:
        columns = tl.arange(0, BLOCK_N)
        low_offsets = (2 * reduction[:, None]) * BLOCK_N + columns[None, :]
        high_offsets = low_offsets + BLOCK_N
        packed_offsets = reduction[:, None] * BLOCK_N + columns[None, :]
        high_mask = high_valid[:, None]

    low = tl.load(input_ptr + low_offsets)
    high = tl.load(input_ptr + high_offsets, mask=high_mask, other=0.0)
    if SIGNED:
        low = tl.quantize(low, SCALE, dtype=tl.int8, qmin=-8, qmax=7)
        high = tl.quantize(high, SCALE, dtype=tl.int8, qmin=-8, qmax=7)
    else:
        low = tl.quantize(low, SCALE, dtype=tl.uint8, qmin=0, qmax=15)
        high = tl.quantize(high, SCALE, dtype=tl.uint8, qmin=0, qmax=15)

    packed = tl.pack_int4(low, high, signed=SIGNED)
    tl.store(packed_ptr + packed_offsets, packed)


def _pack_adjacent_k_values(low, high):
    low_bits = (low.to(torch.int16) & 0xF).to(torch.uint8)
    high_bits = (high.to(torch.int16) & 0xF).to(torch.uint8)
    return low_bits | (high_bits << 4)


def _make_inputs(signed):
    lhs_values = torch.arange(BLOCK_M * LOGICAL_K, dtype=torch.int64).reshape(BLOCK_M, LOGICAL_K)
    rhs_values = torch.arange(LOGICAL_K * BLOCK_N, dtype=torch.int64).reshape(LOGICAL_K, BLOCK_N)
    if signed:
        lhs = (lhs_values % 16 - 8).to(torch.int8)
        rhs = ((rhs_values * 3 + 5) % 16 - 8).to(torch.int8)
    else:
        lhs = (lhs_values % 16).to(torch.uint8)
        rhs = ((rhs_values * 3 + 5) % 16).to(torch.uint8)
    return lhs, rhs


def _make_quantized_inputs(signed, logical_k):
    lhs_steps = (
        (torch.arange(BLOCK_M * logical_k, dtype=torch.int64) * 7 + 3) % 41 - 20
    ).reshape(BLOCK_M, logical_k)
    rhs_steps = (
        (torch.arange(logical_k * BLOCK_N, dtype=torch.int64) * 11 + 5) % 51 - 15
    ).reshape(logical_k, BLOCK_N)
    lhs = lhs_steps.to(torch.float32) * (INT4_SCALE / 2)
    rhs = rhs_steps.to(torch.float32) * (INT4_SCALE / 2)
    qmin, qmax = (-8, 7) if signed else (0, 15)
    dtype = torch.int8 if signed else torch.uint8
    lhs_quantized = torch.round(lhs / INT4_SCALE).clamp(qmin, qmax).to(dtype)
    rhs_quantized = torch.round(rhs / INT4_SCALE).clamp(qmin, qmax).to(dtype)
    return lhs, rhs, lhs_quantized, rhs_quantized


def _pad_high_k_nibble(values, axis):
    if values.shape[axis] % 2 == 0:
        return values
    padding_shape = list(values.shape)
    padding_shape[axis] = 1
    padding = torch.zeros(padding_shape, dtype=values.dtype)
    return torch.cat((values, padding), dim=axis)


@pytest.mark.parametrize("signed", [True, False], ids=["int4", "uint4"])
def test_packed_int4_storage_uses_explicit_int8_dot(device, signed):
    lhs, rhs = _make_inputs(signed)
    lhs_packed = _pack_adjacent_k_values(lhs[:, 0::2], lhs[:, 1::2]).to(device)
    rhs_packed = _pack_adjacent_k_values(rhs[0::2, :], rhs[1::2, :]).to(device)
    output = torch.empty((BLOCK_M, BLOCK_N), dtype=torch.int32, device=device)

    program = _packed_int4_dot_kernel[(1, )](
        lhs_packed,
        rhs_packed,
        output,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        PACKED_K=PACKED_K,
        SIGNED=signed,
    )

    expected_array = np.matmul(lhs.numpy().astype(np.int32), rhs.numpy().astype(np.int32))
    expected = torch.from_numpy(expected_array)
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)

    ttir = program.asm["ttir"]
    assert ttir.count("tt.dot") == 2
    assert "xi8" in ttir
    assert "xi32" in ttir


@pytest.mark.parametrize("signed", [True, False], ids=["int4", "uint4"])
@pytest.mark.parametrize("logical_k", [2 * PACKED_K, 2 * PACKED_K - 1], ids=["even-k", "odd-k"])
def test_quantize_pack_int4_storage_feeds_int8_dot(device, signed, logical_k):
    lhs, rhs, lhs_quantized, rhs_quantized = _make_quantized_inputs(signed, logical_k)
    lhs_packed = torch.empty((BLOCK_M, PACKED_K), dtype=torch.uint8, device=device)
    rhs_packed = torch.empty((PACKED_K, BLOCK_N), dtype=torch.uint8, device=device)

    lhs_pack_program = _quantize_pack_int4_kernel[(1, )](
        lhs.to(device),
        lhs_packed,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        PACKED_K=PACKED_K,
        LOGICAL_K=logical_k,
        SCALE=INT4_SCALE,
        SIGNED=signed,
        PACK_LHS=True,
    )
    rhs_pack_program = _quantize_pack_int4_kernel[(1, )](
        rhs.to(device),
        rhs_packed,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        PACKED_K=PACKED_K,
        LOGICAL_K=logical_k,
        SCALE=INT4_SCALE,
        SIGNED=signed,
        PACK_LHS=False,
    )

    lhs_quantized_padded = _pad_high_k_nibble(lhs_quantized, axis=1)
    rhs_quantized_padded = _pad_high_k_nibble(rhs_quantized, axis=0)
    expected_lhs_packed = _pack_adjacent_k_values(
        lhs_quantized_padded[:, 0::2], lhs_quantized_padded[:, 1::2]
    )
    expected_rhs_packed = _pack_adjacent_k_values(
        rhs_quantized_padded[0::2, :], rhs_quantized_padded[1::2, :]
    )
    torch.testing.assert_close(lhs_packed.cpu(), expected_lhs_packed, rtol=0, atol=0)
    torch.testing.assert_close(rhs_packed.cpu(), expected_rhs_packed, rtol=0, atol=0)

    output = torch.empty((BLOCK_M, BLOCK_N), dtype=torch.int32, device=device)
    dot_program = _packed_int4_dot_kernel[(1, )](
        lhs_packed,
        rhs_packed,
        output,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        PACKED_K=PACKED_K,
        SIGNED=signed,
    )

    expected_array = np.matmul(
        lhs_quantized.numpy().astype(np.int32),
        rhs_quantized.numpy().astype(np.int32),
    )
    torch.testing.assert_close(
        output.cpu(), torch.from_numpy(expected_array), rtol=0, atol=0
    )

    for program in (lhs_pack_program, rhs_pack_program):
        assert "arith.fptosi" in program.asm["ttir"]
        assert "xi8" in program.asm["ttir"]
    assert dot_program.asm["ttir"].count("tt.dot") == 2
    assert "xi32" in dot_program.asm["ttir"]
