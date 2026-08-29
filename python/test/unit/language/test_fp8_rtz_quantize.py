import math

import pytest
import torch
import triton
import triton.language as tl

from low_precision_reference import E5M2, encode_fp8_rtz


BLOCK_SIZE = 32


@triton.jit
def _quantize_e5m2_rtz_kernel(input_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    values = tl.load(input_ptr + offsets)
    quantized = tl.quantize(
        values,
        1.0,
        dtype=tl.float8e5,
        zero_point=0,
        rounding="rtz",
    )
    tl.store(output_ptr + offsets, quantized)


def _carrier_dtype(carrier):
    if carrier == "fp16":
        return torch.float16
    if carrier == "bf16":
        return torch.bfloat16
    if carrier == "fp32":
        return torch.float32
    raise AssertionError(f"unhandled carrier dtype: {carrier}")


def _input_values():
    half_min_subnormal = 2**-17
    values = torch.tensor(
        [
            -60000.0,
            -1.875,
            -1.375,
            -1.20,
            -1.0,
            -3 * half_min_subnormal,
            -half_min_subnormal,
            -0.0,
            0.0,
            half_min_subnormal,
            3 * half_min_subnormal,
            1.0,
            1.20,
            1.375,
            1.875,
            60000.0,
        ],
        dtype=torch.float32,
    )
    return values.repeat(2)


def _special_input_values():
    values = torch.tensor(
        [
            float("-nan"),
            -math.inf,
            -65504.0,
            -60000.0,
            -57344.0,
            -0.0,
            0.0,
            57344.0,
            60000.0,
            65504.0,
            math.inf,
            math.nan,
            -1.20,
            1.20,
            -2**-17,
            2**-17,
        ],
        dtype=torch.float32,
    )
    return values.repeat(2)


def _is_e5m2_nan(byte):
    return byte & 0x7C == 0x7C and byte & 0x03 != 0


@pytest.mark.parametrize("carrier", ["fp16", "bf16", "fp32"])
def test_fp8_e5m2_rtz_quantize_carrier_dtypes(device, carrier):
    input = _input_values().to(dtype=_carrier_dtype(carrier), device=device)
    output = torch.empty(BLOCK_SIZE, dtype=torch.uint8, device=device)

    program = _quantize_e5m2_rtz_kernel[(1, )](
        input,
        triton.reinterpret(output, tl.float8e5),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    expected = torch.tensor(
        [encode_fp8_rtz(value, E5M2) for value in input.cpu().float().tolist()],
        dtype=torch.uint8,
    )
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    assert "rounding = rtz" in program.asm["ttir"]


@pytest.mark.parametrize("carrier", ["fp16", "bf16", "fp32"])
def test_fp8_e5m2_rtz_preserves_special_value_categories(device, carrier):
    input = _special_input_values().to(dtype=_carrier_dtype(carrier), device=device)
    output = torch.empty(BLOCK_SIZE, dtype=torch.uint8, device=device)

    program = _quantize_e5m2_rtz_kernel[(1, )](
        input,
        triton.reinterpret(output, tl.float8e5),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    actual = output.cpu().tolist()
    for value, byte in zip(input.cpu().float().tolist(), actual):
        if math.isnan(value):
            # NaN payload and sign may be canonicalized by the carrier or
            # target, but the result must remain an E5M2 NaN.
            assert _is_e5m2_nan(byte), (carrier, value, byte)
        else:
            assert byte == encode_fp8_rtz(value, E5M2), (carrier, value, byte)
    assert "rounding = rtz" in program.asm["ttir"]
