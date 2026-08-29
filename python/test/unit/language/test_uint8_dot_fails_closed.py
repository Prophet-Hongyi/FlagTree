import pytest
import torch
import triton
import triton.language as tl
from triton.compiler.errors import CompilationError


BLOCK_M = 32
BLOCK_N = 32
BLOCK_K = 32


@triton.jit
def _unsupported_uint8_dot_kernel(
    lhs_ptr,
    rhs_ptr,
    output_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    rows = tl.arange(0, BLOCK_M)
    columns = tl.arange(0, BLOCK_N)
    reduction = tl.arange(0, BLOCK_K)
    lhs = tl.load(lhs_ptr + rows[:, None] * BLOCK_K + reduction[None, :])
    rhs = tl.load(rhs_ptr + reduction[:, None] * BLOCK_N + columns[None, :])
    result = tl.dot(lhs, rhs, out_dtype=tl.int32)
    tl.store(output_ptr + rows[:, None] * BLOCK_N + columns[None, :], result)


@pytest.mark.parametrize(
    "lhs_dtype,rhs_dtype,error",
    [
        (torch.uint8, torch.uint8, "only int8 supported"),
        (torch.int8, torch.uint8, "Both operands must be same dtype"),
        (torch.uint8, torch.int8, "Both operands must be same dtype"),
    ],
    ids=["uint8-uint8", "int8-uint8", "uint8-int8"],
)
def test_uint8_dot_fails_before_backend_lowering(
    device, lhs_dtype, rhs_dtype, error
):
    lhs = torch.ones((BLOCK_M, BLOCK_K), dtype=lhs_dtype, device=device)
    rhs = torch.ones((BLOCK_K, BLOCK_N), dtype=rhs_dtype, device=device)
    output = torch.empty((BLOCK_M, BLOCK_N), dtype=torch.int32, device=device)

    with pytest.raises(CompilationError, match=error):
        _unsupported_uint8_dot_kernel[(1, )](
            lhs,
            rhs,
            output,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )
