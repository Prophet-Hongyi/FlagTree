import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _quantize_fp4_lhs_kernel(
    input_ptr,
    scale_ptr,
    packed_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    rows = tl.arange(0, M)[:, None]
    packed_k = tl.arange(0, K // 2)[None, :]
    low_k = packed_k * 2
    high_k = low_k + 1
    groups: tl.constexpr = K // GROUP_SIZE
    scale = tl.load(scale_ptr + rows * groups + low_k // GROUP_SIZE)
    low = tl.load(input_ptr + rows * K + low_k)
    high = tl.load(input_ptr + rows * K + high_k)
    packed = tl.quantize_fp4(low, high, scale)
    tl.store(packed_ptr + rows * (K // 2) + packed_k, packed)


@triton.jit
def _quantize_fp4_rhs_kernel(
    input_ptr,
    scale_ptr,
    packed_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    packed_k = tl.arange(0, K // 2)[:, None]
    columns = tl.arange(0, N)[None, :]
    low_k = packed_k * 2
    high_k = low_k + 1
    groups: tl.constexpr = K // GROUP_SIZE
    scale = tl.load(scale_ptr + columns * groups + low_k // GROUP_SIZE)
    low = tl.load(input_ptr + low_k * N + columns)
    high = tl.load(input_ptr + high_k * N + columns)
    packed = tl.quantize_fp4(low, high, scale)
    tl.store(packed_ptr + packed_k * N + columns, packed)


@triton.jit
def _dequantize_fp4_lhs_kernel(
    packed_ptr,
    scale_ptr,
    output_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
):
    rows = tl.arange(0, M)[:, None]
    packed_k = tl.arange(0, K // 2)[None, :]
    low_k = packed_k * 2
    high_k = low_k + 1
    groups: tl.constexpr = K // GROUP_SIZE
    packed = tl.load(packed_ptr + rows * (K // 2) + packed_k)
    scale = tl.load(scale_ptr + rows * groups + low_k // GROUP_SIZE)
    low, high = tl.dequantize_fp4(packed, scale, dtype=OUTPUT_DTYPE)
    tl.store(output_ptr + rows * K + low_k, low)
    tl.store(output_ptr + rows * K + high_k, high)


@triton.jit
def _fp4_scaled_matmul_kernel(
    lhs_ptr,
    rhs_ptr,
    lhs_scale_ptr,
    rhs_scale_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    rows = tl.arange(0, M)[:, None]
    columns = tl.arange(0, N)[None, :]
    packed_k = tl.arange(0, K // 2)
    groups: tl.constexpr = K // GROUP_SIZE
    group_offsets = tl.arange(0, groups)

    lhs = tl.load(lhs_ptr + rows * (K // 2) + packed_k[None, :])
    rhs = tl.load(rhs_ptr + packed_k[:, None] * N + columns)
    lhs_scale = tl.load(lhs_scale_ptr + rows * groups + group_offsets[None, :])
    scale_columns = tl.arange(0, N)[:, None]
    rhs_scale = tl.load(rhs_scale_ptr + scale_columns * groups + group_offsets[None, :])
    accumulator = tl.zeros((M, N), dtype=tl.float32)
    output = tl.dot_scaled(
        lhs,
        lhs_scale,
        "e2m1",
        rhs,
        rhs_scale,
        "e2m1",
        acc=accumulator,
        lhs_k_pack=True,
        rhs_k_pack=True,
    )
    tl.store(output_ptr + rows * N + columns, output)


@triton.jit
def _invalid_scaled_fp4_kernel(input_ptr, output_ptr, BLOCK_SIZE: tl.constexpr, CASE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    values = tl.load(input_ptr + offsets)
    if CASE == "quantize_input":
        result = tl.quantize_fp4(values.to(tl.int32), values.to(tl.int32), 1.0)
    elif CASE == "quantize_scale":
        result = tl.quantize_fp4(values, values, 0.0)
    elif CASE == "quantize_format":
        result = tl.quantize_fp4(values, values, 1.0, format="e3m0")
    elif CASE == "dequantize_input":
        result, _ = tl.dequantize_fp4(values.to(tl.int8), 1.0)
    elif CASE == "dequantize_scale":
        result, _ = tl.dequantize_fp4(values.to(tl.uint8), -1.0)
    elif CASE == "dequantize_dtype":
        result, _ = tl.dequantize_fp4(values.to(tl.uint8), 1.0, dtype=tl.int32)
    else:
        tl.static_assert(False, "unknown scaled FP4 case")
    tl.store(output_ptr + offsets, result)


def _exception_chain_text(exception):
    messages = []
    seen = set()
    while exception is not None and id(exception) not in seen:
        seen.add(id(exception))
        messages.append(str(exception))
        exception = exception.__cause__ or exception.__context__
    return "\n".join(messages)


def _assert_target_binary(program):
    binary_keys = ("hgbin", "mubin", "mcfatbin", "hsaco", "cubin", "npubin")
    assert any(key in program.asm and len(program.asm[key]) > 0 for key in binary_keys)


def _assert_exact_bits(actual, expected):
    assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))


def _make_scaled_e2m1_case(dtype):
    m = n = 32
    k = 64
    group_size = 32
    groups = k // group_size
    e2m1_values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
    )

    rows = torch.arange(m)[:, None]
    columns = torch.arange(n)[None, :]
    reduction = torch.arange(k)
    lhs_codes = ((rows * 5 + reduction[None, :] * 3 + reduction[None, :] // group_size) % 16).to(torch.uint8)
    rhs_codes = ((reduction[:, None] * 7 + columns * 11 + reduction[:, None] // group_size) % 16).to(torch.uint8)
    lhs_normalized = e2m1_values[lhs_codes.to(torch.long)]
    rhs_normalized = e2m1_values[rhs_codes.to(torch.long)]

    lhs_exponents = ((rows + torch.arange(groups)[None, :]) % 5) - 2
    rhs_exponents = ((torch.arange(n)[:, None] * 2 + torch.arange(groups)[None, :]) % 5) - 2
    lhs_scale = torch.pow(2.0, lhs_exponents.to(torch.float32))
    rhs_scale = torch.pow(2.0, rhs_exponents.to(torch.float32))
    lhs = (lhs_normalized * lhs_scale.repeat_interleave(group_size, dim=1)).to(dtype)
    rhs = (rhs_normalized * rhs_scale.T.repeat_interleave(group_size, dim=0)).to(dtype)

    lhs_packed = (lhs_codes[:, 0::2] | (lhs_codes[:, 1::2] << 4)).to(torch.uint8)
    rhs_packed = (rhs_codes[0::2, :] | (rhs_codes[1::2, :] << 4)).to(torch.uint8)
    lhs_scale_e8m0 = (lhs_exponents + 127).to(torch.uint8)
    rhs_scale_e8m0 = (rhs_exponents + 127).to(torch.uint8)
    reference = lhs.to(torch.float32) @ rhs.to(torch.float32)
    return lhs, rhs, lhs_scale, rhs_scale, lhs_packed, rhs_packed, lhs_scale_e8m0, rhs_scale_e8m0, reference


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_scaled_fp4_storage_feeds_dot_scaled(device, dtype):
    m = n = 32
    k = 64
    group_size = 32
    (lhs_cpu, rhs_cpu, lhs_scale_cpu, rhs_scale_cpu, expected_lhs_packed, expected_rhs_packed,
     lhs_scale_e8m0_cpu, rhs_scale_e8m0_cpu, reference) = _make_scaled_e2m1_case(dtype)

    lhs_packed = torch.empty_like(expected_lhs_packed, device=device)
    rhs_packed = torch.empty_like(expected_rhs_packed, device=device)
    lhs_dequantized = torch.empty_like(lhs_cpu, device=device)
    output = torch.empty((m, n), dtype=torch.float32, device=device)
    lhs_scale = lhs_scale_cpu.to(device)
    rhs_scale = rhs_scale_cpu.to(device)

    lhs_program = _quantize_fp4_lhs_kernel[(1, )](
        lhs_cpu.to(device), lhs_scale, lhs_packed, M=m, K=k, GROUP_SIZE=group_size
    )
    rhs_program = _quantize_fp4_rhs_kernel[(1, )](
        rhs_cpu.to(device), rhs_scale, rhs_packed, N=n, K=k, GROUP_SIZE=group_size
    )
    output_dtype = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
    }[dtype]
    dequantize_program = _dequantize_fp4_lhs_kernel[(1, )](
        lhs_packed, lhs_scale, lhs_dequantized, M=m, K=k, GROUP_SIZE=group_size, OUTPUT_DTYPE=output_dtype
    )
    dot_program = _fp4_scaled_matmul_kernel[(1, )](
        lhs_packed,
        rhs_packed,
        lhs_scale_e8m0_cpu.to(device),
        rhs_scale_e8m0_cpu.to(device),
        output,
        M=m,
        N=n,
        K=k,
        GROUP_SIZE=group_size,
    )

    assert torch.equal(lhs_packed.cpu(), expected_lhs_packed)
    assert torch.equal(rhs_packed.cpu(), expected_rhs_packed)
    _assert_exact_bits(lhs_dequantized.cpu(), lhs_cpu)
    # Backends may use different FP32 accumulation trees after the same exact
    # FP4 decode. Keep storage/dequantization bit-exact above, and bound only
    # the matrix reduction's legal rounding drift here.
    torch.testing.assert_close(output.cpu(), reference, atol=1.0e-4, rtol=5.0e-7)
    for program in (lhs_program, rhs_program, dequantize_program, dot_program):
        _assert_target_binary(program)
    for program in (lhs_program, rhs_program, dequantize_program):
        assert "f4E2M1FN" not in program.asm["ttir"]
    assert "tt.dot_scaled" in dot_program.asm["ttir"]


@pytest.mark.parametrize(
    "case,error",
    [
        ("quantize_input", "quantize_fp4 inputs must be tl.float16, tl.bfloat16, or tl.float32"),
        ("quantize_scale", "scale must be finite and greater than zero"),
        ("quantize_format", "encode_fp4 format must be 'e2m1'"),
        ("dequantize_input", "dequantize_fp4 input must be tl.uint8 storage"),
        ("dequantize_scale", "scale must be finite and greater than zero"),
        ("dequantize_dtype", "dequantize_fp4 dtype must be tl.float16, tl.bfloat16, or tl.float32"),
    ],
)
def test_scaled_fp4_rejects_unsupported_contracts(device, case, error):
    input = torch.zeros(16, dtype=torch.float16, device=device)
    output = torch.empty_like(input)
    with pytest.raises(triton.CompilationError) as exc_info:
        _invalid_scaled_fp4_kernel[(1, )](input, output, BLOCK_SIZE=16, CASE=case)
    assert error in _exception_chain_text(exc_info.value)
