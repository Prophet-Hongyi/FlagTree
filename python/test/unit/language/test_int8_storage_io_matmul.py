import numpy as np
import pytest
import torch
import triton
import triton.language as tl


GROUPS = 2
GROUP_K = 32
MATMUL_SHAPES = ((16, 32), (32, 16), (32, 32))


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
    OUTPUT_DTYPE: tl.constexpr,
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
        dtype=OUTPUT_DTYPE,
        zero_point=OUTPUT_ZERO_POINT,
        rounding="rtne",
    )
    tl.store(output_ptr + rows[:, None] * BLOCK_N + columns[None, :], quantized)


def _make_inputs(block_m, block_n):
    rows = torch.arange(block_m, dtype=torch.int64)[:, None, None]
    groups = torch.arange(GROUPS, dtype=torch.int64)[None, :, None]
    reduction = torch.arange(GROUP_K, dtype=torch.int64)[None, None, :]
    lhs = ((rows * 3 + groups * 5 + reduction * 7) % 9 - 4).to(torch.int8)

    groups = torch.arange(GROUPS, dtype=torch.int64)[:, None, None]
    reduction = torch.arange(GROUP_K, dtype=torch.int64)[None, :, None]
    columns = torch.arange(block_n, dtype=torch.int64)[None, None, :]
    rhs = ((groups * 2 + reduction * 5 + columns * 3) % 11 - 5).to(torch.int8)
    lhs_scale = torch.tensor(
        [[0.125, 0.5], [0.25, 1.0], [0.5, 0.125], [1.0, 0.25]],
        dtype=torch.float32,
    ).repeat(block_m // 4, 1)
    rhs_scale = torch.tensor(
        [[0.25, 1.0], [0.5, 2.0], [1.0, 0.25], [2.0, 0.5]],
        dtype=torch.float32,
    ).repeat(block_n // 4, 1)
    output_scale = torch.tensor([0.25, 0.5, 1.0, 2.0], dtype=torch.float32).repeat(
        block_m // 4
    )
    return lhs, rhs, lhs_scale, rhs_scale, output_scale


@pytest.mark.parametrize(
    "output_dtype,torch_dtype,output_zero_point,qmin,qmax",
    [
        (tl.int8, torch.int8, 0, -128, 127),
        (tl.int8, torch.int8, -7, -128, 127),
        (tl.uint8, torch.uint8, 113, 0, 255),
    ],
    ids=["int8-symmetric", "int8-affine", "uint8-affine"],
)
@pytest.mark.parametrize(
    "block_m,block_n", MATMUL_SHAPES, ids=["m16n32", "m32n16", "m32n32"]
)
def test_int8_storage_input_output_matmul(
    device,
    block_m,
    block_n,
    output_dtype,
    torch_dtype,
    output_zero_point,
    qmin,
    qmax,
):
    lhs, rhs, lhs_scale, rhs_scale, output_scale = _make_inputs(block_m, block_n)
    output = torch.empty((block_m, block_n), dtype=torch_dtype, device=device)

    program = _int8_storage_io_matmul_kernel[(1, )](
        lhs.to(device),
        rhs.to(device),
        lhs_scale.to(device),
        rhs_scale.to(device),
        output_scale.to(device),
        output,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        GROUPS=GROUPS,
        GROUP_K=GROUP_K,
        OUTPUT_DTYPE=output_dtype,
        OUTPUT_ZERO_POINT=output_zero_point,
    )

    expected = torch.zeros((block_m, block_n), dtype=torch.float32)
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
    ).clamp(qmin, qmax).to(torch_dtype)

    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    # The oracle must exercise both saturation directions and non-saturated
    # values; otherwise an all-zero dot can make the epilogue look correct.
    assert int(expected.min()) == qmin
    assert int(expected.max()) == qmax
    assert torch.any((expected > qmin) & (expected < qmax))
    assert program.asm["ttir"].count("tt.dot") == GROUPS
    assert "xi8" in program.asm["ttir"]
    assert "xi32" in program.asm["ttir"]
    assert "arith.fptosi" in program.asm["ttir"]
