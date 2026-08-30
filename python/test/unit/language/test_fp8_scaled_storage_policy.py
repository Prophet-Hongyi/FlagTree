import math

import pytest
import torch
import triton
import triton.language as tl

from low_precision_reference import E4M3FN, E4M3FNUZ, E5M2, E5M2FNUZ, decode_fp8, encode_fp8_rtne


FORMATS = {
    "e4m3fn": E4M3FN,
    "e4m3fnuz": E4M3FNUZ,
    "e5m2": E5M2,
    "e5m2fnuz": E5M2FNUZ,
}


@triton.jit
def _scaled_fp8_storage_kernel(
    input_ptr,
    scale_ptr,
    packed_ptr,
    output_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GRANULARITY: tl.constexpr,
    FORMAT: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
):
    groups: tl.constexpr = (K + GROUP_SIZE - 1) // GROUP_SIZE
    rows = tl.arange(0, M)[:, None]
    columns = tl.arange(0, K)[None, :]

    if GRANULARITY == "tensor":
        scale = tl.load(scale_ptr)
    elif GRANULARITY == "row":
        scale = tl.load(scale_ptr + rows)
    elif GRANULARITY == "group":
        scale = tl.load(scale_ptr + rows * groups + columns // GROUP_SIZE)
    else:
        tl.static_assert(False, "unknown scale granularity")

    values = tl.load(input_ptr + rows * K + columns)
    packed = tl.quantize_fp8(values, scale, format=FORMAT)
    output = tl.dequantize_fp8(packed, scale, format=FORMAT, dtype=OUTPUT_DTYPE)
    tl.store(packed_ptr + rows * K + columns, packed)
    tl.store(output_ptr + rows * K + columns, output)


@triton.jit
def _invalid_scaled_fp8_storage_kernel(
    input_ptr,
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
    CASE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    values = tl.load(input_ptr + offsets)
    if CASE == "quantize_input":
        result = tl.quantize_fp8(values.to(tl.int32), 1.0)
    elif CASE == "quantize_scale":
        result = tl.quantize_fp8(values, 0.0)
    elif CASE == "quantize_format":
        result = tl.quantize_fp8(values, 1.0, format="e4m3")
    elif CASE == "quantize_rounding":
        result = tl.quantize_fp8(values, 1.0, rounding="rtz")
    elif CASE == "quantize_overflow":
        result = tl.quantize_fp8(values, 1.0, overflow="nan")
    elif CASE == "dequantize_input":
        result = tl.dequantize_fp8(values.to(tl.int8), 1.0)
    elif CASE == "dequantize_dtype":
        result = tl.dequantize_fp8(values.to(tl.uint8), 1.0, dtype=tl.float64)
    else:
        tl.static_assert(False, "unknown scaled FP8 storage case")
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


def _assert_output(actual, expected):
    actual_nan = torch.isnan(actual)
    expected_nan = torch.isnan(expected)
    assert torch.equal(actual_nan, expected_nan)
    assert torch.equal(actual[~expected_nan].contiguous().view(torch.uint8),
                       expected[~expected_nan].contiguous().view(torch.uint8))


def _make_case(dtype, format_name, granularity):
    m = 16
    k = 64
    group_size = 16
    groups = k // group_size
    normalized_values = torch.tensor(
        [
            0.0,
            -0.0,
            0.125,
            -0.125,
            0.5,
            -0.5,
            1.0,
            -1.0,
            1.5,
            -1.5,
            3.0,
            -3.0,
            6.0,
            -6.0,
            1000.0,
            -1000.0,
            math.inf,
            -math.inf,
            math.nan,
            math.copysign(math.nan, -1.0),
        ],
        dtype=torch.float32,
    )
    scale_values = torch.tensor([0.75, 1.25, 1.75, 2.5, 3.25], dtype=torch.float32)
    rows = torch.arange(m)[:, None]
    columns = torch.arange(k)[None, :]
    normalized = normalized_values[(rows * 7 + columns * 5 + columns // group_size) % len(normalized_values)]

    if granularity == "tensor":
        scale = scale_values[1:2].clone()
        expanded_scale = scale.reshape(1, 1).expand(m, k)
    elif granularity == "row":
        scale = scale_values[(torch.arange(m) * 3) % len(scale_values)].clone()
        expanded_scale = scale[:, None].expand(m, k)
    else:
        group_rows = torch.arange(m)[:, None]
        group_columns = torch.arange(groups)[None, :]
        scale = scale_values[(group_rows * 2 + group_columns * 3) % len(scale_values)].clone()
        expanded_scale = scale.repeat_interleave(group_size, dim=1)

    input_values = (normalized * expanded_scale).to(dtype)
    reference_format = FORMATS[format_name]
    expected_packed = torch.tensor(
        [
            encode_fp8_rtne(float(value) / float(scale_value), reference_format)
            for value, scale_value in zip(input_values.flatten().tolist(), expanded_scale.flatten().tolist())
        ],
        dtype=torch.uint8,
    ).reshape(m, k)
    expected_output = torch.tensor(
        [
            decode_fp8(code, reference_format) * float(scale_value)
            for code, scale_value in zip(expected_packed.flatten().tolist(), expanded_scale.flatten().tolist())
        ],
        dtype=dtype,
    ).reshape(m, k)
    return input_values, scale, expected_packed, expected_output, group_size


@pytest.mark.parametrize(
    "format_name,granularity",
    [
        ("e4m3fn", "tensor"),
        ("e5m2", "row"),
        ("e4m3fnuz", "group"),
        ("e5m2fnuz", "group"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_fp8_scaled_storage_formats_carriers_and_broadcast(device, dtype, format_name, granularity):
    input_cpu, scale_cpu, expected_packed, expected_output, group_size = _make_case(
        dtype, format_name, granularity
    )
    packed = torch.empty_like(expected_packed, device=device)
    output = torch.empty_like(expected_output, device=device)
    output_dtype = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
    }[dtype]

    program = _scaled_fp8_storage_kernel[(1, )](
        input_cpu.to(device),
        scale_cpu.to(device),
        packed,
        output,
        M=input_cpu.shape[0],
        K=input_cpu.shape[1],
        GROUP_SIZE=group_size,
        GRANULARITY=granularity,
        FORMAT=format_name,
        OUTPUT_DTYPE=output_dtype,
    )

    assert torch.equal(packed.cpu(), expected_packed)
    _assert_output(output.cpu(), expected_output)
    _assert_target_binary(program)
    assert "f8E4M3FN" not in program.asm["ttir"]
    assert "f8E4M3FNUZ" not in program.asm["ttir"]
    assert "f8E5M2" not in program.asm["ttir"]


@pytest.mark.parametrize(
    "case,error",
    [
        ("quantize_input", "quantize_fp8 input must be"),
        ("quantize_scale", "scale must be finite"),
        ("quantize_format", "encode_fp8 format must be"),
        ("quantize_rounding", "encode_fp8 rounding must be"),
        ("quantize_overflow", "encode_fp8 overflow must be"),
        ("dequantize_input", "dequantize_fp8 input must be"),
        ("dequantize_dtype", "dequantize_fp8 dtype must be"),
    ],
)
def test_fp8_scaled_storage_rejects_unsupported_contracts(device, case, error):
    input = torch.zeros(16, dtype=torch.float32, device=device)
    output = torch.empty_like(input)
    with pytest.raises(Exception) as exc_info:
        _invalid_scaled_fp8_storage_kernel[(1, )](input, output, BLOCK_SIZE=16, CASE=case)
    assert error in _exception_chain_text(exc_info.value)
