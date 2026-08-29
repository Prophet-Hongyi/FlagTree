import numpy as np
import pytest
import torch
import triton
import triton.language as tl

from low_precision_reference import E4M3FN, E5M2, decode_fp8, encode_fp8_rtne


BLOCK_M = 32
BLOCK_N = 32
GROUPS = 2
GROUP_K = 32


@triton.jit
def _fp8_storage_io_matmul_kernel(
    lhs_ptr,
    rhs_ptr,
    lhs_scale_ptr,
    rhs_scale_ptr,
    output_scale_ptr,
    output_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUPS: tl.constexpr,
    GROUP_K: tl.constexpr,
    FP8_DTYPE: tl.constexpr,
):
    rows = tl.arange(0, BLOCK_M)
    columns = tl.arange(0, BLOCK_N)
    reduction = tl.arange(0, GROUP_K)
    output = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for group in tl.static_range(0, GROUPS):
        lhs_offsets = (rows[:, None] * GROUPS + group) * GROUP_K + reduction[None, :]
        rhs_offsets = (group * GROUP_K + reduction[:, None]) * BLOCK_N + columns[None, :]
        lhs = tl.load(lhs_ptr + lhs_offsets)
        rhs = tl.load(rhs_ptr + rhs_offsets)
        accumulator = tl.dot(lhs, rhs, out_dtype=tl.float32)

        lhs_scale = tl.load(lhs_scale_ptr + rows * GROUPS + group)[:, None]
        rhs_scale = tl.load(rhs_scale_ptr + columns * GROUPS + group)[None, :]
        output += accumulator * lhs_scale * rhs_scale

    output_scale = tl.load(output_scale_ptr + rows)[:, None]
    quantized = tl.quantize(
        output, output_scale, dtype=FP8_DTYPE, rounding="rtne"
    )
    tl.store(output_ptr + rows[:, None] * BLOCK_N + columns[None, :], quantized)


def _encode_fp8(values, fp8_format):
    flat = [encode_fp8_rtne(value, fp8_format) for value in values.reshape(-1).tolist()]
    return torch.tensor(flat, dtype=torch.uint8).reshape(values.shape)


def _decode_fp8(values, fp8_format):
    flat = [decode_fp8(value, fp8_format) for value in values.reshape(-1).tolist()]
    return torch.tensor(flat, dtype=torch.float32).reshape(values.shape)


def _format_config(format_name):
    if format_name == "e4m3fn":
        return E4M3FN, tl.float8e4nv, "f8E4M3FN"
    if format_name == "e5m2":
        return E5M2, tl.float8e5, "f8E5M2"
    raise AssertionError(f"unhandled FP8 format: {format_name}")


def _make_inputs():
    lhs = (
        (torch.arange(BLOCK_M * GROUPS * GROUP_K, dtype=torch.int64) * 7 + 3) % 5
        - 2
    ).reshape(BLOCK_M, GROUPS, GROUP_K).to(torch.float32)
    rhs = (
        (torch.arange(GROUPS * GROUP_K * BLOCK_N, dtype=torch.int64) * 11 + 5) % 5
        - 2
    ).reshape(GROUPS, GROUP_K, BLOCK_N).to(torch.float32)
    lhs_scale = torch.tensor(
        [[0.5, 1.0], [1.0, 0.25], [0.25, 0.5], [1.0, 1.0]],
        dtype=torch.float32,
    ).repeat(BLOCK_M // 4, 1)
    rhs_scale = torch.tensor(
        [[1.0, 0.5], [0.5, 1.0], [0.25, 1.0], [1.0, 0.25]],
        dtype=torch.float32,
    ).repeat(BLOCK_N // 4, 1)
    output_scale = torch.tensor([3.0, 5.0, 7.0, 9.0], dtype=torch.float32).repeat(
        BLOCK_M // 4
    )
    return lhs, rhs, lhs_scale, rhs_scale, output_scale


@pytest.mark.parametrize("format_name", ["e4m3fn", "e5m2"])
def test_fp8_storage_input_output_matmul(device, format_name):
    fp8_format, fp8_dtype, ir_type = _format_config(format_name)
    lhs, rhs, lhs_scale, rhs_scale, output_scale = _make_inputs()
    lhs_encoded = _encode_fp8(lhs, fp8_format)
    rhs_encoded = _encode_fp8(rhs, fp8_format)
    lhs_storage = lhs_encoded.to(device)
    rhs_storage = rhs_encoded.to(device)
    output_storage = torch.empty((BLOCK_M, BLOCK_N), dtype=torch.uint8, device=device)
    lhs_fp8 = triton.reinterpret(lhs_storage, fp8_dtype)
    rhs_fp8 = triton.reinterpret(rhs_storage, fp8_dtype)
    output_fp8 = triton.reinterpret(output_storage, fp8_dtype)

    program = _fp8_storage_io_matmul_kernel[(1, )](
        lhs_fp8,
        rhs_fp8,
        lhs_scale.to(device),
        rhs_scale.to(device),
        output_scale.to(device),
        output_fp8,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        GROUPS=GROUPS,
        GROUP_K=GROUP_K,
        FP8_DTYPE=fp8_dtype,
    )

    lhs_decoded = _decode_fp8(lhs_encoded, fp8_format)
    rhs_decoded = _decode_fp8(rhs_encoded, fp8_format)
    expected = torch.zeros((BLOCK_M, BLOCK_N), dtype=torch.float32)
    for group in range(GROUPS):
        group_accumulator = torch.from_numpy(
            np.matmul(lhs_decoded[:, group].numpy(), rhs_decoded[group].numpy())
        )
        expected += group_accumulator * lhs_scale[:, group, None] * rhs_scale[:, group][None, :]
    expected_encoded = _encode_fp8(expected / output_scale[:, None], fp8_format)

    torch.testing.assert_close(output_storage.cpu(), expected_encoded, rtol=0, atol=0)
    assert program.asm["ttir"].count("tt.dot") == GROUPS
    assert "tt.fp_to_fp" in program.asm["ttir"]
    assert ir_type in program.asm["ttir"]
