import pytest
import torch
import triton
import triton.language as tl


BLOCK_M = 32
BLOCK_N = 32


@triton.jit
def _int8_accumulator_wrap_kernel(
    lhs_ptr,
    rhs_ptr,
    output_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ACCUMULATOR_INIT: tl.constexpr,
):
    rows = tl.arange(0, BLOCK_M)
    columns = tl.arange(0, BLOCK_N)
    reduction = tl.arange(0, BLOCK_K)
    lhs = tl.load(lhs_ptr + rows[:, None] * BLOCK_K + reduction[None, :])
    rhs = tl.load(rhs_ptr + reduction[:, None] * BLOCK_N + columns[None, :])
    accumulator = tl.full(
        (BLOCK_M, BLOCK_N), ACCUMULATOR_INIT, dtype=tl.int32
    )
    result = tl.dot(lhs, rhs, accumulator, out_dtype=tl.int32)
    tl.store(output_ptr + rows[:, None] * BLOCK_N + columns[None, :], result)


@pytest.mark.parametrize(
    "lhs_value,accumulator_init",
    [
        (1, 2_147_483_632),
        (-1, -2_147_483_632),
    ],
    ids=["positive-overflow", "negative-underflow"],
)
@pytest.mark.parametrize(
    "block_k", [32, 64], ids=["single-k-tile", "multiple-k-tiles"]
)
def test_signed_int8_dot_accumulator_wraps_modulo_2_to_32(
    device, lhs_value, accumulator_init, block_k
):
    lhs = torch.full(
        (BLOCK_M, block_k), lhs_value, dtype=torch.int8, device=device
    )
    rhs = torch.ones((block_k, BLOCK_N), dtype=torch.int8, device=device)
    output = torch.empty((BLOCK_M, BLOCK_N), dtype=torch.int32, device=device)

    program = _int8_accumulator_wrap_kernel[(1, )](
        lhs,
        rhs,
        output,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=block_k,
        ACCUMULATOR_INIT=accumulator_init,
    )

    mathematical_result = accumulator_init + lhs_value * block_k
    expected = (mathematical_result + 2**31) % 2**32 - 2**31
    torch.testing.assert_close(
        output.cpu(), torch.full_like(output.cpu(), expected), rtol=0, atol=0
    )
    assert program.asm["ttir"].count("tt.dot") == 1
    assert "xi8" in program.asm["ttir"]
    assert "xi32" in program.asm["ttir"]
