import pytest
import torch
import triton
import triton.language as tl

from low_precision_reference import E4M3FN, E5M2, encode_fp8_rtne


BLOCK_SIZE = 32
SCALE = 0.5


@triton.jit
def _quantize_carrier_kernel(
    input_ptr,
    scale_ptr,
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
    ZERO_POINT: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    values = tl.load(input_ptr + offsets)
    scale = tl.load(scale_ptr)
    quantized = tl.quantize(
        values,
        scale,
        dtype=OUTPUT_DTYPE,
        zero_point=ZERO_POINT,
        rounding="rtne",
    )
    tl.store(output_ptr + offsets, quantized)


@triton.jit
def _dequantize_carrier_kernel(
    input_ptr,
    scale_ptr,
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
    ZERO_POINT: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    values = tl.load(input_ptr + offsets)
    scale = tl.load(scale_ptr)
    dequantized = tl.dequantize(
        values,
        scale,
        dtype=OUTPUT_DTYPE,
        zero_point=ZERO_POINT,
    )
    tl.store(output_ptr + offsets, dequantized)


def _carrier_config(carrier):
    if carrier == "fp16":
        return torch.float16, tl.float16, "f16"
    if carrier == "bf16":
        return torch.bfloat16, tl.bfloat16, "bf16"
    if carrier == "fp32":
        return torch.float32, tl.float32, "f32"
    raise AssertionError(f"unhandled carrier dtype: {carrier}")


def _storage_config(storage):
    if storage == "int8":
        return torch.int8, tl.int8, None, -3
    if storage == "e4m3fn":
        return torch.uint8, tl.float8e4nv, E4M3FN, 0
    if storage == "e5m2":
        return torch.uint8, tl.float8e5, E5M2, 0
    raise AssertionError(f"unhandled storage dtype: {storage}")


def _make_values(storage, zero_point):
    levels = torch.tensor([-6, -4, -2, -1, 0, 1, 2, 4, 6], dtype=torch.int8)
    levels = levels.repeat((BLOCK_SIZE + levels.numel() - 1) // levels.numel())[:BLOCK_SIZE]
    if storage == "int8":
        values = (levels.to(torch.float32) - zero_point) * SCALE
        expected_storage = levels
    else:
        values = levels.to(torch.float32) * SCALE
        expected_storage = levels
    return values, expected_storage


@pytest.mark.parametrize("carrier", ["fp16", "bf16", "fp32"])
@pytest.mark.parametrize("storage", ["int8", "e4m3fn", "e5m2"])
def test_low_precision_quantize_dequantize_carrier_dtypes(device, storage, carrier):
    torch_carrier, tl_carrier, carrier_ir = _carrier_config(carrier)
    torch_storage, tl_storage, fp8_format, zero_point = _storage_config(storage)
    values, expected_levels = _make_values(storage, zero_point)
    input = values.to(dtype=torch_carrier, device=device)
    scale = torch.tensor([SCALE], dtype=torch.float32, device=device)
    storage_output = torch.empty(BLOCK_SIZE, dtype=torch_storage, device=device)
    storage_arg = (
        storage_output
        if fp8_format is None
        else triton.reinterpret(storage_output, tl_storage)
    )

    quantize_program = _quantize_carrier_kernel[(1, )](
        input,
        scale,
        storage_arg,
        BLOCK_SIZE=BLOCK_SIZE,
        OUTPUT_DTYPE=tl_storage,
        ZERO_POINT=zero_point,
    )

    if fp8_format is None:
        expected_storage = expected_levels
    else:
        expected_storage = torch.tensor(
            [encode_fp8_rtne(value, fp8_format) for value in expected_levels.tolist()],
            dtype=torch.uint8,
        )
    torch.testing.assert_close(storage_output.cpu(), expected_storage, rtol=0, atol=0)

    output = torch.empty(BLOCK_SIZE, dtype=torch_carrier, device=device)
    dequantize_program = _dequantize_carrier_kernel[(1, )](
        storage_arg,
        scale,
        output,
        BLOCK_SIZE=BLOCK_SIZE,
        OUTPUT_DTYPE=tl_carrier,
        ZERO_POINT=zero_point,
    )

    expected_output = values.to(torch_carrier)
    torch.testing.assert_close(output.cpu(), expected_output, rtol=0, atol=0)
    assert carrier_ir in dequantize_program.asm["ttir"]
    if fp8_format is None:
        assert "arith.fptosi" in quantize_program.asm["ttir"]
    else:
        assert "tt.fp_to_fp" in quantize_program.asm["ttir"]
        assert "tt.fp_to_fp" in dequantize_program.asm["ttir"]
