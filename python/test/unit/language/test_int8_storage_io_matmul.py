import numpy as np
import pytest
import torch
import triton
import triton.language as tl


BLOCK_M = 32
BLOCK_N = 32
GROUPS = 2
GROUP_K = 32


@triton.jit
def _int8_storage_io_matmul_kernel(
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
    OUTPUT_ZERO_POINT: tl.constexpr,
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

    output_scale = tl.load(output_scale_ptr + rows)[:, None]
    quantized = tl.quantize(
        output,
        output_scale,
        dtype=tl.int8,
        zero_point=OUTPUT_ZERO_POINT,
        rounding="rtne",
    )
    tl.store(output_ptr + rows[:, None] * BLOCK_N + columns[None, :], quantized)


def _make_inputs():
    lhs = (
        (torch.arange(BLOCK_M * GROUPS * GROUP_K, dtype=torch.int64) * 7 + 3) % 9
        - 4
    ).reshape(BLOCK_M, GROUPS, GROUP_K).to(torch.int8)
    rhs = (
        (torch.arange(GROUPS * GROUP_K * BLOCK_N, dtype=torch.int64) * 11 + 5) % 11
        - 5
    ).reshape(GROUPS, GROUP_K, BLOCK_N).to(torch.int8)
    lhs_scale = torch.tensor(
        [[0.125, 0.5], [0.25, 1.0], [0.5, 0.125], [1.0, 0.25]],
        dtype=torch.float32,
    ).repeat(BLOCK_M // 4, 1)
    rhs_scale = torch.tensor(
        [[0.25, 1.0], [0.5, 2.0], [1.0, 0.25], [2.0, 0.5]],
        dtype=torch.float32,
    ).repeat(BLOCK_N // 4, 1)
    output_scale = torch.tensor([3.0, 5.0, 7.0, 9.0], dtype=torch.float32).repeat(
        BLOCK_M // 4
    )
    return lhs, rhs, lhs_scale, rhs_scale, output_scale


@pytest.mark.parametrize("output_zero_point", [0, -7], ids=["symmetric", "affine"])
def test_int8_storage_input_output_matmul(device, output_zero_point):
    lhs, rhs, lhs_scale, rhs_scale, output_scale = _make_inputs()
    output = torch.empty((BLOCK_M, BLOCK_N), dtype=torch.int8, device=device)

    program = _int8_storage_io_matmul_kernel[(1, )](
        lhs.to(device),
        rhs.to(device),
        lhs_scale.to(device),
        rhs_scale.to(device),
        output_scale.to(device),
        output,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        GROUPS=GROUPS,
        GROUP_K=GROUP_K,
        OUTPUT_ZERO_POINT=output_zero_point,
    )

    expected = torch.zeros((BLOCK_M, BLOCK_N), dtype=torch.float32)
    for group in range(GROUPS):
        group_accumulator = torch.from_numpy(
            np.matmul(
                lhs[:, group].to(torch.int32).numpy(),
                rhs[group].to(torch.int32).numpy(),
            )
        )
        expected += (
            group_accumulator.to(torch.float32)
            * lhs_scale[:, group, None]
            * rhs_scale[:, group][None, :]
        )
    expected = (
        torch.round(expected / output_scale[:, None]) + output_zero_point
    ).clamp(-128, 127).to(torch.int8)

    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    assert program.asm["ttir"].count("tt.dot") == GROUPS
    assert "xi8" in program.asm["ttir"]
    assert "xi32" in program.asm["ttir"]
    assert "arith.fptosi" in program.asm["ttir"]
