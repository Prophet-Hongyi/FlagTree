import pytest
import triton
import triton.language as tl
from triton.compiler.errors import CompilationError


@triton.jit
def _invalid_dot_scaled_contract(CASE: tl.constexpr):
    M: tl.constexpr = 16
    N: tl.constexpr = 16
    K: tl.constexpr = 32

    if CASE == "invalid_format":
        lhs = tl.full((M, K), 0, tl.uint8)
        rhs = tl.full((K, N), 0, tl.uint8)
        tl.dot_scaled(lhs, None, "int4", rhs, None, "e5m2")
    elif CASE == "invalid_e2m1_storage":
        lhs = tl.full((M, K // 2), 0, tl.int8)
        rhs = tl.full((K // 2, N), 0, tl.uint8)
        tl.dot_scaled(lhs, None, "e2m1", rhs, None, "e2m1")
    elif CASE == "invalid_e4m3_storage":
        lhs = tl.full((M, K), 0, tl.int16)
        rhs = tl.full((K, N), 0, tl.uint8)
        tl.dot_scaled(lhs, None, "e4m3", rhs, None, "e4m3")
    elif CASE == "invalid_non_k_pack":
        lhs = tl.full((M, K), 0, tl.uint8)
        rhs = tl.full((K, N), 0, tl.uint8)
        tl.dot_scaled(lhs, None, "e5m2", rhs, None, "e5m2", lhs_k_pack=False)
    elif CASE == "invalid_reduction_extent":
        lhs = tl.full((M, K // 2), 0, tl.uint8)
        rhs = tl.full((K // 4, N), 0, tl.uint8)
        tl.dot_scaled(lhs, None, "e2m1", rhs, None, "e2m1")
    elif CASE == "invalid_scale_shape":
        lhs = tl.full((M, K), 0, tl.uint8)
        rhs = tl.full((K, N), 0, tl.uint8)
        lhs_scale = tl.full((M, K // 32 + 1), 0, tl.uint8)
        rhs_scale = tl.full((N, K // 32), 0, tl.uint8)
        tl.dot_scaled(lhs, lhs_scale, "e5m2", rhs, rhs_scale, "e5m2")
    else:
        tl.static_assert(False, "unknown dot_scaled contract case")


def _exception_chain_text(exception):
    messages = []
    seen = set()
    while exception is not None and id(exception) not in seen:
        seen.add(id(exception))
        messages.append(str(exception))
        exception = exception.__cause__ or exception.__context__
    return "\n".join(messages)


@pytest.mark.parametrize(
    "case,error",
    [
        ("invalid_format", "Invalid float format: int4"),
        ("invalid_e2m1_storage", "e2m1 format must be packed as uint8"),
        ("invalid_e4m3_storage", "Unexpected dtype for e4m3"),
        ("invalid_non_k_pack", "only mxfp4 inputs can be packed"),
        ("invalid_reduction_extent", "Reduction dimension should pack the same number of elements"),
        ("invalid_scale_shape", "lhs_scale must be a tensor of shape"),
    ],
)
def test_dot_scaled_rejects_invalid_low_precision_contract(case, error):
    with pytest.raises(CompilationError) as exc_info:
        triton.compile(
            triton.compiler.ASTSource(fn=_invalid_dot_scaled_contract, signature={}, constexprs={"CASE": case}))

    assert error in _exception_chain_text(exc_info.value)
