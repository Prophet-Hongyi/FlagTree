import pytest
import torch
import triton
import triton.language as tl


BLOCK_M = 32
BLOCK_N = 32
PACKED_K = 32


@triton.jit
def _packed_4bit_accumulator_wrap_kernel(
    lhs_ptr,
    rhs_ptr,
    output_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PACKED_K: tl.constexpr,
    SIGNED: tl.constexpr,
    ZERO_POINT: tl.constexpr,
    ACCUMULATOR_INIT: tl.constexpr,
):
    rows = tl.arange(0, BLOCK_M)
    columns = tl.arange(0, BLOCK_N)
    reduction = tl.arange(0, PACKED_K)
    lhs_packed = tl.load(
        lhs_ptr + rows[:, None] * PACKED_K + reduction[None, :]
    )
    rhs_packed = tl.load(
        rhs_ptr + reduction[:, None] * BLOCK_N + columns[None, :]
    )
    lhs_low, lhs_high = tl.unpack_int4(lhs_packed, signed=SIGNED)
    rhs_low, rhs_high = tl.unpack_int4(rhs_packed, signed=SIGNED)

    if not SIGNED:
        lhs_low = (lhs_low.to(tl.int16) - ZERO_POINT).to(tl.int8)
        lhs_high = (lhs_high.to(tl.int16) - ZERO_POINT).to(tl.int8)
        rhs_low = (rhs_low.to(tl.int16) - ZERO_POINT).to(tl.int8)
        rhs_high = (rhs_high.to(tl.int16) - ZERO_POINT).to(tl.int8)

    accumulator = tl.full(
        (BLOCK_M, BLOCK_N), ACCUMULATOR_INIT, dtype=tl.int32
    )
    accumulator = tl.dot(
        lhs_low, rhs_low, accumulator, out_dtype=tl.int32
    )
    accumulator = tl.dot(
        lhs_high, rhs_high, accumulator, out_dtype=tl.int32
    )
    tl.store(
        output_ptr + rows[:, None] * BLOCK_N + columns[None, :],
        accumulator,
    )


def _repeat_nibble(value):
    bits = value & 0xF
    return bits | (bits << 4)


@pytest.mark.parametrize(
    "signed,lhs_value,rhs_value,accumulator_init",
    [
        (True, -8, -8, 2_147_481_600),
        (True, -8, 7, -2_147_481_600),
        (False, 15, 15, 2_147_481_600),
        (False, 0, 15, -2_147_481_600),
    ],
    ids=[
        "int4-positive-overflow",
        "int4-negative-underflow",
        "uint4-positive-overflow",
        "uint4-negative-underflow",
    ],
)
def test_packed_4bit_dot_accumulator_wraps_modulo_2_to_32(
    device, signed, lhs_value, rhs_value, accumulator_init
):
    lhs = torch.full(
        (BLOCK_M, PACKED_K),
        _repeat_nibble(lhs_value),
        dtype=torch.uint8,
        device=device,
    )
    rhs = torch.full(
        (PACKED_K, BLOCK_N),
        _repeat_nibble(rhs_value),
        dtype=torch.uint8,
        device=device,
    )
    output = torch.empty((BLOCK_M, BLOCK_N), dtype=torch.int32, device=device)
    zero_point = 0 if signed else 7

    program = _packed_4bit_accumulator_wrap_kernel[(1, )](
        lhs,
        rhs,
        output,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        PACKED_K=PACKED_K,
        SIGNED=signed,
        ZERO_POINT=zero_point,
        ACCUMULATOR_INIT=accumulator_init,
    )

    lhs_centered = lhs_value - zero_point
    rhs_centered = rhs_value - zero_point
    logical_k = 2 * PACKED_K
    mathematical_result = (
        accumulator_init + lhs_centered * rhs_centered * logical_k
    )
    expected = (mathematical_result + 2**31) % 2**32 - 2**31
    torch.testing.assert_close(
        output.cpu(), torch.full_like(output.cpu(), expected), rtol=0, atol=0
    )
    assert program.asm["ttir"].count("tt.dot") == 2
    assert "xi8" in program.asm["ttir"]
    assert "xi32" in program.asm["ttir"]
