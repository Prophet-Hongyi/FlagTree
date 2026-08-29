import math

import pytest
import torch
import triton
import triton.language as tl

from low_precision_reference import E4M3FN, E5M2, decode_fp8, encode_fp8_rtne


@triton.jit
def _quantize_fp8_bytes_kernel(
    input_ptr,
    scale_ptr,
    output_ptr,
    N_ELEMENTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    FP8_DTYPE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    scale = tl.load(scale_ptr)
    quantized = tl.quantize(values, scale, dtype=FP8_DTYPE, rounding="rtne")
    tl.store(output_ptr + offsets, quantized, mask=mask)


def _format_config(format_name):
    if format_name == "e4m3fn":
        return E4M3FN, tl.float8e4nv, "f8E4M3FN", 2**-10
    if format_name == "e5m2":
        return E5M2, tl.float8e5, "f8E5M2", 2**-17
    raise AssertionError(f"unhandled FP8 format: {format_name}")


def _edge_values(fp8_format, half_min_subnormal):
    max_finite = decode_fp8(fp8_format.max_finite_code, fp8_format)
    previous_finite = decode_fp8(fp8_format.max_finite_code - 1, fp8_format)
    top_midpoint = (previous_finite + max_finite) / 2
    return [
        math.copysign(math.nan, -1.0),
        -math.inf,
        -2 * max_finite,
        -max_finite,
        -top_midpoint,
        -7 * half_min_subnormal,
        -5 * half_min_subnormal,
        -3 * half_min_subnormal,
        -half_min_subnormal,
        -0.0,
        0.0,
        half_min_subnormal,
        3 * half_min_subnormal,
        5 * half_min_subnormal,
        7 * half_min_subnormal,
        top_midpoint,
        max_finite,
        2 * max_finite,
        math.inf,
        math.nan,
    ]


@pytest.mark.parametrize("format_name", ["e4m3fn", "e5m2"])
def test_fp8_quantize_rtne_edge_bytes(device, format_name):
    fp8_format, fp8_dtype, ir_type, half_min_subnormal = _format_config(format_name)
    values = _edge_values(fp8_format, half_min_subnormal)
    input = torch.tensor(values, dtype=torch.float32, device=device)
    scale = torch.ones(1, dtype=torch.float32, device=device)
    output_storage = torch.empty(len(values), dtype=torch.uint8, device=device)
    output_fp8 = triton.reinterpret(output_storage, fp8_dtype)

    program = _quantize_fp8_bytes_kernel[(1, )](
        input,
        scale,
        output_fp8,
        N_ELEMENTS=len(values),
        BLOCK_SIZE=triton.next_power_of_2(len(values)),
        FP8_DTYPE=fp8_dtype,
    )

    actual_bytes = output_storage.cpu().tolist()
    expected_bytes = [encode_fp8_rtne(value, fp8_format) for value in values]
    for index, (value, actual, expected) in enumerate(
        zip(values, actual_bytes, expected_bytes)
    ):
        if math.isnan(value):
            # Triton's cast contract requires NaN to remain NaN, but does not
            # freeze a cross-backend NaN payload or sign encoding.
            assert math.isnan(decode_fp8(actual, fp8_format)), (
                f"{format_name}[{index}]: 0x{actual:02x} is not NaN"
            )
        else:
            assert actual == expected, (
                f"{format_name}[{index}]: actual=0x{actual:02x}, "
                f"expected=0x{expected:02x}"
            )
    assert "tt.fp_to_fp" in program.asm["ttir"]
    assert ir_type in program.asm["ttir"]
