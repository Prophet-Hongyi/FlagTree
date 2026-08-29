import numpy as np
import pytest
import torch
import triton
import triton.language as tl

from low_precision_reference import E4M3FN, E5M2, decode_fp8, encode_fp8_rtne


LOGICAL_K = 64
FP8_MATMUL_ATOL = 2**-14


@triton.jit
def _quantize_fp8_per_group_kernel(
    input_ptr,
    scale_ptr,
    quantized_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUPS: tl.constexpr,
    GROUP_K: tl.constexpr,
    QUANTIZE_LHS: tl.constexpr,
    FP8_DTYPE: tl.constexpr,
):
    group = tl.program_id(0)
    reduction = tl.arange(0, GROUP_K)
    if QUANTIZE_LHS:
        axis = tl.arange(0, BLOCK_M)
        offsets = (axis[:, None] * GROUPS + group) * GROUP_K + reduction[None, :]
        scale = tl.load(scale_ptr + axis * GROUPS + group)[:, None]
    else:
        axis = tl.arange(0, BLOCK_N)
        offsets = (group * GROUP_K + reduction[:, None]) * BLOCK_N + axis[None, :]
        scale = tl.load(scale_ptr + axis * GROUPS + group)[None, :]

    values = tl.load(input_ptr + offsets)
    quantized = tl.quantize(values, scale, dtype=FP8_DTYPE, rounding="rtne")
    tl.store(quantized_ptr + offsets, quantized)


@triton.jit
def _tensor_scaled_fp8_matmul_kernel(
    lhs_ptr,
    rhs_ptr,
    lhs_scale_ptr,
    rhs_scale_ptr,
    output_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUPS: tl.constexpr,
    GROUP_K: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
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

    tl.store(
        output_ptr + rows[:, None] * BLOCK_N + columns[None, :],
        output.to(OUTPUT_DTYPE),
    )


def _encode_fp8(values, fp8_format):
    flat = [encode_fp8_rtne(value, fp8_format) for value in values.reshape(-1).tolist()]
    return torch.tensor(flat, dtype=torch.uint8).reshape(values.shape)


def _decode_fp8(values, fp8_format):
    flat = [decode_fp8(value, fp8_format) for value in values.reshape(-1).tolist()]
    return torch.tensor(flat, dtype=torch.float32).reshape(values.shape)


def _format_config(format_name):
    if format_name == "e4m3fn":
        return E4M3FN, tl.float8e4nv, "f8E4M3FN", 25, 12, 0.5
    if format_name == "e5m2":
        return E5M2, tl.float8e5, "f8E5M2", 13, 6, 1.0
    raise AssertionError(f"unhandled FP8 format: {format_name}")


def _output_config(output_name):
    if output_name == "fp32":
        return torch.float32, tl.float32, "f32"
    if output_name == "fp16":
        return torch.float16, tl.float16, "f16"
    if output_name == "bf16":
        return torch.bfloat16, tl.bfloat16, "bf16"
    raise AssertionError(f"unhandled output dtype: {output_name}")


def _make_inputs(block_m, block_n, groups, value_modulus, value_offset, value_step):
    group_k = LOGICAL_K // groups
    lhs_scale = torch.tensor(
        [[0.125, 0.5], [0.25, 1.0], [0.5, 0.125], [1.0, 0.25]], dtype=torch.float32
    ).repeat(block_m // 4, 1)[:, :groups].contiguous()
    rhs_scale = torch.tensor(
        [[0.25, 1.0], [0.5, 2.0], [1.0, 0.25], [2.0, 0.5]], dtype=torch.float32
    ).repeat(block_n // 4, 1)[:, :groups].contiguous()

    lhs_unscaled = (
        (torch.arange(block_m * LOGICAL_K, dtype=torch.int64) * 7 + 3) % value_modulus
        - value_offset
    ).reshape(block_m, groups, group_k).to(torch.float32) * value_step
    rhs_unscaled = (
        (torch.arange(LOGICAL_K * block_n, dtype=torch.int64) * 11 + 5) % value_modulus
        - value_offset
    ).reshape(groups, group_k, block_n).to(torch.float32) * value_step
    lhs = lhs_unscaled * lhs_scale[:, :, None]
    rhs = rhs_unscaled * rhs_scale.T[:, None, :]
    return lhs, rhs, lhs_scale, rhs_scale, lhs_unscaled, rhs_unscaled


@pytest.mark.parametrize("format_name", ["e4m3fn", "e5m2"])
@pytest.mark.parametrize("groups", [1, 2], ids=["per-axis", "per-group-k32"])
@pytest.mark.parametrize("output_name", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize(
    "block_m,block_n",
    [(16, 32), (32, 16), (32, 32)],
    ids=["m16n32", "m32n16", "m32n32"],
)
def test_tensor_scaled_fp8_matmul(
    device, block_m, block_n, output_name, groups, format_name
):
    group_k = LOGICAL_K // groups
    output_torch_dtype, output_triton_dtype, output_ir_type = _output_config(
        output_name
    )
    fp8_format, fp8_dtype, ir_type, value_modulus, value_offset, value_step = (
        _format_config(format_name)
    )
    lhs, rhs, lhs_scale, rhs_scale, lhs_unscaled, rhs_unscaled = _make_inputs(
        block_m, block_n, groups, value_modulus, value_offset, value_step
    )
    lhs_encoded = _encode_fp8(lhs_unscaled, fp8_format)
    rhs_encoded = _encode_fp8(rhs_unscaled, fp8_format)

    lhs_storage = torch.empty(lhs_encoded.shape, dtype=torch.uint8, device=device)
    rhs_storage = torch.empty(rhs_encoded.shape, dtype=torch.uint8, device=device)
    lhs_fp8 = triton.reinterpret(lhs_storage, fp8_dtype)
    rhs_fp8 = triton.reinterpret(rhs_storage, fp8_dtype)
    lhs_quantize_program = _quantize_fp8_per_group_kernel[(groups, )](
        lhs.to(device),
        lhs_scale.to(device),
        lhs_fp8,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        GROUPS=groups,
        GROUP_K=group_k,
        QUANTIZE_LHS=True,
        FP8_DTYPE=fp8_dtype,
    )
    rhs_quantize_program = _quantize_fp8_per_group_kernel[(groups, )](
        rhs.to(device),
        rhs_scale.to(device),
        rhs_fp8,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        GROUPS=groups,
        GROUP_K=group_k,
        QUANTIZE_LHS=False,
        FP8_DTYPE=fp8_dtype,
    )
    torch.testing.assert_close(lhs_storage.cpu(), lhs_encoded, rtol=0, atol=0)
    torch.testing.assert_close(rhs_storage.cpu(), rhs_encoded, rtol=0, atol=0)

    output = torch.empty(
        (block_m, block_n), dtype=output_torch_dtype, device=device
    )
    dot_program = _tensor_scaled_fp8_matmul_kernel[(1, )](
        lhs_fp8,
        rhs_fp8,
        lhs_scale.to(device),
        rhs_scale.to(device),
        output,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        GROUPS=groups,
        GROUP_K=group_k,
        OUTPUT_DTYPE=output_triton_dtype,
    )

    lhs_decoded = _decode_fp8(lhs_encoded, fp8_format)
    rhs_decoded = _decode_fp8(rhs_encoded, fp8_format)
    expected = torch.zeros((block_m, block_n), dtype=torch.float32)
    for group in range(groups):
        group_accumulator = torch.from_numpy(
            np.matmul(lhs_decoded[:, group].numpy(), rhs_decoded[group].numpy())
        )
        expected += group_accumulator * lhs_scale[:, group, None] * rhs_scale[:, group][None, :]
    expected = expected.to(output_torch_dtype)
    if output_torch_dtype == torch.float32:
        output_rtol = 0
        output_atol = FP8_MATMUL_ATOL
    else:
        output_rtol = torch.finfo(output_torch_dtype).eps
        output_atol = 0
    torch.testing.assert_close(
        output.cpu(), expected, rtol=output_rtol, atol=output_atol
    )

    for program in (lhs_quantize_program, rhs_quantize_program):
        assert "tt.fp_to_fp" in program.asm["ttir"]
        assert ir_type in program.asm["ttir"]
    assert dot_program.asm["ttir"].count("tt.dot") == groups
    assert ir_type in dot_program.asm["ttir"]
    assert "f32" in dot_program.asm["ttir"]
    assert f"!tt.ptr<{output_ir_type}>" in dot_program.asm["ttir"]
