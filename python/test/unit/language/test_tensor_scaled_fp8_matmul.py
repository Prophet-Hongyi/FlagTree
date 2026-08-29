import numpy as np
import pytest
import torch
import triton
import triton.language as tl

from low_precision_reference import E4M3FN, decode_fp8, encode_fp8_rtne


BLOCK_M = 32
BLOCK_N = 32
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
    quantized = tl.quantize(values, scale, dtype=tl.float8e4nv, rounding="rtne")
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

    tl.store(output_ptr + rows[:, None] * BLOCK_N + columns[None, :], output)


def _encode_e4m3fn(values):
    flat = [encode_fp8_rtne(value, E4M3FN) for value in values.reshape(-1).tolist()]
    return torch.tensor(flat, dtype=torch.uint8).reshape(values.shape)


def _decode_e4m3fn(values):
    flat = [decode_fp8(value, E4M3FN) for value in values.reshape(-1).tolist()]
    return torch.tensor(flat, dtype=torch.float32).reshape(values.shape)


def _make_inputs(groups):
    group_k = LOGICAL_K // groups
    lhs_scale = torch.tensor(
        [[0.125, 0.5], [0.25, 1.0], [0.5, 0.125], [1.0, 0.25]], dtype=torch.float32
    ).repeat(BLOCK_M // 4, 1)[:, :groups].contiguous()
    rhs_scale = torch.tensor(
        [[0.25, 1.0], [0.5, 2.0], [1.0, 0.25], [2.0, 0.5]], dtype=torch.float32
    ).repeat(BLOCK_N // 4, 1)[:, :groups].contiguous()

    lhs_unscaled = (
        (torch.arange(BLOCK_M * LOGICAL_K, dtype=torch.int64) * 7 + 3) % 25 - 12
    ).reshape(BLOCK_M, groups, group_k).to(torch.float32) * 0.5
    rhs_unscaled = (
        (torch.arange(LOGICAL_K * BLOCK_N, dtype=torch.int64) * 11 + 5) % 25 - 12
    ).reshape(groups, group_k, BLOCK_N).to(torch.float32) * 0.5
    lhs = lhs_unscaled * lhs_scale[:, :, None]
    rhs = rhs_unscaled * rhs_scale.T[:, None, :]
    lhs_encoded = _encode_e4m3fn(lhs_unscaled)
    rhs_encoded = _encode_e4m3fn(rhs_unscaled)
    return lhs, rhs, lhs_scale, rhs_scale, lhs_encoded, rhs_encoded


@pytest.mark.parametrize("groups", [1, 2], ids=["per-axis", "per-group-k32"])
def test_tensor_scaled_fp8_matmul(device, groups):
    group_k = LOGICAL_K // groups
    lhs, rhs, lhs_scale, rhs_scale, lhs_encoded, rhs_encoded = _make_inputs(groups)

    lhs_storage = torch.empty(lhs_encoded.shape, dtype=torch.uint8, device=device)
    rhs_storage = torch.empty(rhs_encoded.shape, dtype=torch.uint8, device=device)
    lhs_fp8 = triton.reinterpret(lhs_storage, tl.float8e4nv)
    rhs_fp8 = triton.reinterpret(rhs_storage, tl.float8e4nv)
    lhs_quantize_program = _quantize_fp8_per_group_kernel[(groups, )](
        lhs.to(device),
        lhs_scale.to(device),
        lhs_fp8,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        GROUPS=groups,
        GROUP_K=group_k,
        QUANTIZE_LHS=True,
    )
    rhs_quantize_program = _quantize_fp8_per_group_kernel[(groups, )](
        rhs.to(device),
        rhs_scale.to(device),
        rhs_fp8,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        GROUPS=groups,
        GROUP_K=group_k,
        QUANTIZE_LHS=False,
    )
    torch.testing.assert_close(lhs_storage.cpu(), lhs_encoded, rtol=0, atol=0)
    torch.testing.assert_close(rhs_storage.cpu(), rhs_encoded, rtol=0, atol=0)

    output = torch.empty((BLOCK_M, BLOCK_N), dtype=torch.float32, device=device)
    dot_program = _tensor_scaled_fp8_matmul_kernel[(1, )](
        lhs_fp8,
        rhs_fp8,
        lhs_scale.to(device),
        rhs_scale.to(device),
        output,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        GROUPS=groups,
        GROUP_K=group_k,
    )

    lhs_decoded = _decode_e4m3fn(lhs_encoded)
    rhs_decoded = _decode_e4m3fn(rhs_encoded)
    expected = torch.zeros((BLOCK_M, BLOCK_N), dtype=torch.float32)
    for group in range(groups):
        group_accumulator = torch.from_numpy(
            np.matmul(lhs_decoded[:, group].numpy(), rhs_decoded[group].numpy())
        )
        expected += group_accumulator * lhs_scale[:, group, None] * rhs_scale[:, group][None, :]
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=FP8_MATMUL_ATOL)

    for program in (lhs_quantize_program, rhs_quantize_program):
        assert "tt.fp_to_fp" in program.asm["ttir"]
        assert "f8E4M3FN" in program.asm["ttir"]
    assert dot_program.asm["ttir"].count("tt.dot") == groups
    assert "f8E4M3FN" in dot_program.asm["ttir"]
    assert "f32" in dot_program.asm["ttir"]
