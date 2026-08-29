import numpy as np
import pytest
import torch
import triton
import triton.language as tl


LOGICAL_K = 64


@triton.jit
def _quantize_int8_per_group_kernel(
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
    quantized = tl.quantize(values, scale, dtype=tl.int8, zero_point=0)
    tl.store(quantized_ptr + offsets, quantized)


@triton.jit
def _tensor_scaled_int8_matmul_kernel(
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

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        accumulator = tl.dot(lhs, rhs, accumulator, out_dtype=tl.int32)
        lhs_scale = tl.load(lhs_scale_ptr + rows * GROUPS + group)[:, None]
        rhs_scale = tl.load(rhs_scale_ptr + columns * GROUPS + group)[None, :]
        output += accumulator.to(tl.float32) * lhs_scale * rhs_scale

    tl.store(output_ptr + rows[:, None] * BLOCK_N + columns[None, :], output)


def _make_inputs(block_m, block_n, groups):
    group_k = LOGICAL_K // groups
    lhs_scale = torch.tensor(
        [[0.125, 0.5], [0.25, 1.0], [0.5, 0.125], [1.0, 0.25]], dtype=torch.float32
    ).repeat(block_m // 4, 1)[:, :groups].contiguous()
    rhs_scale = torch.tensor(
        [[0.25, 1.0], [0.5, 2.0], [1.0, 0.25], [2.0, 0.5]], dtype=torch.float32
    ).repeat(block_n // 4, 1)[:, :groups].contiguous()

    lhs_steps = (
        (torch.arange(block_m * LOGICAL_K, dtype=torch.int64) * 7 + 3) % 41 - 20
    ).reshape(block_m, groups, group_k)
    rhs_steps = (
        (torch.arange(LOGICAL_K * block_n, dtype=torch.int64) * 11 + 5) % 43 - 21
    ).reshape(groups, group_k, block_n)
    lhs = lhs_steps.to(torch.float32) * 0.5 * lhs_scale[:, :, None]
    rhs = rhs_steps.to(torch.float32) * 0.5 * rhs_scale.T[:, None, :]
    lhs_quantized = torch.round(lhs / lhs_scale[:, :, None]).clamp(-128, 127).to(torch.int8)
    rhs_quantized = torch.round(rhs / rhs_scale.T[:, None, :]).clamp(-128, 127).to(torch.int8)
    return lhs, rhs, lhs_scale, rhs_scale, lhs_quantized, rhs_quantized


@pytest.mark.parametrize("groups", [1, 2], ids=["per-axis", "per-group-k32"])
@pytest.mark.parametrize(
    "block_m,block_n",
    [(16, 32), (32, 16), (32, 32)],
    ids=["m16n32", "m32n16", "m32n32"],
)
def test_tensor_scaled_int8_matmul(device, block_m, block_n, groups):
    group_k = LOGICAL_K // groups
    lhs, rhs, lhs_scale, rhs_scale, lhs_quantized, rhs_quantized = _make_inputs(
        block_m, block_n, groups
    )

    lhs_storage = torch.empty_like(lhs_quantized, device=device)
    rhs_storage = torch.empty_like(rhs_quantized, device=device)
    lhs_quantize_program = _quantize_int8_per_group_kernel[(groups, )](
        lhs.to(device),
        lhs_scale.to(device),
        lhs_storage,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        GROUPS=groups,
        GROUP_K=group_k,
        QUANTIZE_LHS=True,
    )
    rhs_quantize_program = _quantize_int8_per_group_kernel[(groups, )](
        rhs.to(device),
        rhs_scale.to(device),
        rhs_storage,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        GROUPS=groups,
        GROUP_K=group_k,
        QUANTIZE_LHS=False,
    )
    torch.testing.assert_close(lhs_storage.cpu(), lhs_quantized, rtol=0, atol=0)
    torch.testing.assert_close(rhs_storage.cpu(), rhs_quantized, rtol=0, atol=0)

    output = torch.empty((block_m, block_n), dtype=torch.float32, device=device)
    dot_program = _tensor_scaled_int8_matmul_kernel[(1, )](
        lhs_storage,
        rhs_storage,
        lhs_scale.to(device),
        rhs_scale.to(device),
        output,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        GROUPS=groups,
        GROUP_K=group_k,
    )

    expected = torch.zeros((block_m, block_n), dtype=torch.float32)
    for group in range(groups):
        group_accumulator = torch.from_numpy(
            np.matmul(lhs_quantized[:, group].to(torch.int32).numpy(),
                      rhs_quantized[group].to(torch.int32).numpy())
        )
        expected += group_accumulator.to(torch.float32) * lhs_scale[:, group, None] * rhs_scale[:, group][None, :]
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)

    for program in (lhs_quantize_program, rhs_quantize_program):
        assert "arith.fptosi" in program.asm["ttir"]
        assert "xi8" in program.asm["ttir"]
    assert dot_program.asm["ttir"].count("tt.dot") == groups
    assert "xi32" in dot_program.asm["ttir"]
    assert "f32" in dot_program.asm["ttir"]
