import math

import pytest
import torch
import triton
import triton.language as tl

from low_precision_reference import E4M3FN, E5M2, decode_fp8


@triton.jit
def _dequantize_fp8_bytes_kernel(
    input_ptr,
    scale_ptr,
    output_ptr,
    N_ELEMENTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS
    encoded = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    scale = tl.load(scale_ptr)
    decoded = tl.dequantize(encoded, scale, dtype=tl.float32)
    tl.store(output_ptr + offsets, decoded, mask=mask)


def _format_config(format_name):
    if format_name == "e4m3fn":
        return (
            E4M3FN,
            tl.float8e4nv,
            "f8E4M3FN",
            [
                0x00,
                0x80,
                0x01,
                0x81,
                0x02,
                0x82,
                0x07,
                0x87,
                0x08,
                0x88,
                0x38,
                0xB8,
                0x7D,
                0xFD,
                0x7E,
                0xFE,
                0x7F,
                0xFF,
            ],
        )
    if format_name == "e5m2":
        return (
            E5M2,
            tl.float8e5,
            "f8E5M2",
            [
                0x00,
                0x80,
                0x01,
                0x81,
                0x02,
                0x82,
                0x03,
                0x83,
                0x04,
                0x84,
                0x3C,
                0xBC,
                0x7B,
                0xFB,
                0x7C,
                0xFC,
                0x7D,
                0xFD,
                0x7E,
                0xFE,
                0x7F,
                0xFF,
            ],
        )
    raise AssertionError(f"unhandled FP8 format: {format_name}")


@pytest.mark.parametrize("format_name", ["e4m3fn", "e5m2"])
def test_fp8_dequantize_edge_bytes(device, format_name):
    fp8_format, fp8_dtype, ir_type, encoded_bytes = _format_config(format_name)
    scale_value = 0.5
    input_storage = torch.tensor(encoded_bytes, dtype=torch.uint8, device=device)
    input_fp8 = triton.reinterpret(input_storage, fp8_dtype)
    scale = torch.tensor([scale_value], dtype=torch.float32, device=device)
    output = torch.empty(len(encoded_bytes), dtype=torch.float32, device=device)

    program = _dequantize_fp8_bytes_kernel[(1, )](
        input_fp8,
        scale,
        output,
        N_ELEMENTS=len(encoded_bytes),
        BLOCK_SIZE=triton.next_power_of_2(len(encoded_bytes)),
    )

    expected_values = [
        decode_fp8(encoded, fp8_format) * scale_value for encoded in encoded_bytes
    ]
    actual_values = output.cpu()
    expected = torch.tensor(expected_values, dtype=torch.float32)
    actual_bits = actual_values.view(torch.int32).tolist()
    expected_bits = expected.view(torch.int32).tolist()
    for index, (expected_value, actual_bit_pattern, expected_bit_pattern) in enumerate(
        zip(expected_values, actual_bits, expected_bits)
    ):
        if math.isnan(expected_value):
            assert math.isnan(actual_values[index].item()), (
                f"{format_name}[{index}]: decoded 0x{encoded_bytes[index]:02x} "
                "is not NaN"
            )
        else:
            assert actual_bit_pattern == expected_bit_pattern, (
                f"{format_name}[{index}]: decoded 0x{encoded_bytes[index]:02x} "
                f"to bits 0x{actual_bit_pattern & 0xFFFFFFFF:08x}, expected "
                f"0x{expected_bit_pattern & 0xFFFFFFFF:08x}"
            )

    assert "tt.fp_to_fp" in program.asm["ttir"]
    assert ir_type in program.asm["ttir"]
