import math

import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _quantize_kernel(
    input_ptr,
    scale_ptr,
    zero_point_ptr,
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
    ROUNDING: tl.constexpr,
    QMIN: tl.constexpr,
    QMAX: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    input = tl.load(input_ptr + offsets)
    scale = tl.load(scale_ptr)
    zero_point = tl.load(zero_point_ptr)
    output = tl.quantize(
        input,
        scale,
        dtype=OUTPUT_DTYPE,
        zero_point=zero_point,
        rounding=ROUNDING,
        qmin=QMIN,
        qmax=QMAX,
    )
    tl.store(output_ptr + offsets, output)


@triton.jit
def _dequantize_kernel(
    input_ptr,
    scale_ptr,
    zero_point_ptr,
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    input = tl.load(input_ptr + offsets)
    scale = tl.load(scale_ptr)
    zero_point = tl.load(zero_point_ptr)
    output = tl.dequantize(input, scale, dtype=OUTPUT_DTYPE, zero_point=zero_point)
    tl.store(output_ptr + offsets, output)


@triton.jit
def _scaled_fp8_roundtrip_kernel(input_ptr, scale_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    input = tl.load(input_ptr + offsets)
    scale = tl.load(scale_ptr)
    quantized = tl.quantize(input, scale, dtype=tl.float8e4nv, rounding="rtne")
    output = tl.dequantize(quantized, scale, dtype=tl.float32)
    tl.store(output_ptr + offsets, output)


@triton.jit
def _static_quantize_dequantize_kernel(input_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    input = tl.load(input_ptr + offsets)
    quantized = tl.quantize(input, 0.5, dtype=tl.int8, zero_point=0)
    output = tl.dequantize(quantized, 0.5, dtype=tl.float32, zero_point=0)
    tl.store(output_ptr + offsets, output)


@triton.jit
def _invalid_quantize_kernel(
    input_ptr,
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
    ROUNDING: tl.constexpr,
    QMIN: tl.constexpr,
    QMAX: tl.constexpr,
    ZERO_POINT: tl.constexpr,
    SCALE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    input = tl.load(input_ptr + offsets)
    output = tl.quantize(
        input,
        SCALE,
        dtype=OUTPUT_DTYPE,
        zero_point=ZERO_POINT,
        rounding=ROUNDING,
        qmin=QMIN,
        qmax=QMAX,
    )
    tl.store(output_ptr + offsets, output)


def _reference_quantize(input, *, scale, zero_point, qmin, qmax, rounding):
    scaled = input.float() / scale
    scaled = torch.where(torch.isnan(scaled), torch.zeros_like(scaled), scaled)
    scaled = torch.clamp(scaled, qmin - zero_point, qmax - zero_point)
    rounded = torch.round(scaled) if rounding == "rtne" else torch.trunc(scaled)
    return torch.clamp(rounded + zero_point, qmin, qmax)


def _exception_chain_text(exception):
    messages = []
    seen = set()
    while exception is not None and id(exception) not in seen:
        seen.add(id(exception))
        messages.append(str(exception))
        exception = exception.__cause__ or exception.__context__
    return "\n".join(messages)


@pytest.mark.parametrize(
    "output_dtype,torch_dtype,qmin,qmax,zero_point",
    [
        (tl.int8, torch.int8, -8, 7, 1),
        (tl.uint8, torch.uint8, 0, 15, 7),
    ],
)
@pytest.mark.parametrize("rounding", ["rtne", "rtz"])
def test_explicit_affine_quantize_device(device, output_dtype, torch_dtype, qmin, qmax, zero_point, rounding):
    input = torch.tensor(
        [
            -math.inf,
            -100.0,
            -2.5,
            -1.25,
            -0.75,
            -0.5,
            -0.25,
            -0.0,
            0.25,
            0.5,
            0.75,
            1.25,
            2.5,
            100.0,
            math.inf,
            math.nan,
        ],
        dtype=torch.float32,
        device=device,
    )
    scale = torch.tensor([0.5], dtype=torch.float32, device=device)
    zero_point_tensor = torch.tensor([zero_point], dtype=torch.int32, device=device)
    output = torch.empty(input.shape, dtype=torch_dtype, device=device)

    _quantize_kernel[(1, )](
        input,
        scale,
        zero_point_tensor,
        output,
        BLOCK_SIZE=input.numel(),
        OUTPUT_DTYPE=output_dtype,
        ROUNDING=rounding,
        QMIN=qmin,
        QMAX=qmax,
    )

    expected = _reference_quantize(
        input.cpu(),
        scale=0.5,
        zero_point=zero_point,
        qmin=qmin,
        qmax=qmax,
        rounding=rounding,
    ).to(torch_dtype)
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "input_dtype,values,zero_point",
    [
        (torch.int8, [-128, -8, -2, -1, 0, 1, 7, 127], -1),
        (torch.uint8, [0, 1, 2, 15, 127, 128, 254, 255], 127),
    ],
)
def test_explicit_affine_dequantize_widens_before_subtraction(device, input_dtype, values, zero_point):
    input = torch.tensor(values, dtype=input_dtype, device=device)
    scale = torch.tensor([0.25], dtype=torch.float32, device=device)
    zero_point_tensor = torch.tensor([zero_point], dtype=torch.int32, device=device)
    output = torch.empty(input.shape, dtype=torch.float32, device=device)

    _dequantize_kernel[(1, )](
        input,
        scale,
        zero_point_tensor,
        output,
        BLOCK_SIZE=8,
        OUTPUT_DTYPE=tl.float32,
    )

    expected = (input.cpu().to(torch.float32) - zero_point) * 0.25
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


def test_quantize_dequantize_accepts_static_parameters(device):
    input = torch.tensor(
        [
            -127.5,
            -100.0,
            -2.0,
            -1.25,
            -0.75,
            -0.5,
            -0.25,
            0.0,
            0.25,
            0.5,
            0.75,
            1.25,
            2.0,
            100.0,
            127.5,
            math.nan,
        ],
        dtype=torch.float32,
        device=device,
    )
    output = torch.empty_like(input)

    _static_quantize_dequantize_kernel[(1, )](input, output, BLOCK_SIZE=input.numel())

    quantized = _reference_quantize(
        input.cpu(),
        scale=0.5,
        zero_point=0,
        qmin=-128,
        qmax=127,
        rounding="rtne",
    ).to(torch.int8)
    expected = quantized.to(torch.float32) * 0.5
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


def test_scaled_fp8_quantize_dequantize_roundtrip_device(device):
    input = torch.tensor(
        [-12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, -0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0],
        dtype=torch.float32,
        device=device,
    )
    scale = torch.tensor([2.0], dtype=torch.float32, device=device)
    output = torch.empty_like(input)

    _scaled_fp8_roundtrip_kernel[(1, )](input, scale, output, BLOCK_SIZE=input.numel())

    torch.testing.assert_close(output.cpu(), input.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize(
    "output_dtype,rounding,qmin,qmax,zero_point,scale,error",
    [
        (tl.int16, "rtne", -128, 127, 0, 1.0, "tl.int8 or tl.uint8"),
        (tl.int8, "nearest", -128, 127, 0, 1.0, "rounding must be"),
        (tl.int8, "rtne", 7, 7, 7, 1.0, "qmin and qmax"),
        (tl.int8, "rtne", -128, 127, 0, 0.0, "scale must be finite"),
        (tl.float8e4nv, "rtne", None, None, 1, 1.0, "zero_point=0"),
    ],
)
def test_quantize_rejects_invalid_static_contracts(
    device,
    output_dtype,
    rounding,
    qmin,
    qmax,
    zero_point,
    scale,
    error,
):
    input = torch.zeros(16, dtype=torch.float32, device=device)
    output = torch.empty(16, dtype=torch.int8, device=device)
    with pytest.raises(triton.CompilationError) as exc_info:
        _invalid_quantize_kernel[(1, )](
            input,
            output,
            BLOCK_SIZE=input.numel(),
            OUTPUT_DTYPE=output_dtype,
            ROUNDING=rounding,
            QMIN=qmin,
            QMAX=qmax,
            ZERO_POINT=zero_point,
            SCALE=scale,
        )
    assert error in _exception_chain_text(exc_info.value)
