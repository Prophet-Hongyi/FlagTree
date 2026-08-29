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
