import pytest
import torch
import triton
import triton.language as tl


BLOCK_M = 32
BLOCK_N = 32
GROUPS = 2
GROUP_K = 32


@triton.jit
def _fp8_software_storage_matmul_kernel(
    lhs_storage_ptr,
    rhs_storage_ptr,
    lhs_scale_ptr,
    rhs_scale_ptr,
    output_scale_ptr,
    output_float_ptr,
    output_storage_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUPS: tl.constexpr,
    GROUP_K: tl.constexpr,
    FORMAT: tl.constexpr,
):
    rows = tl.arange(0, BLOCK_M)
    columns = tl.arange(0, BLOCK_N)
    reduction = tl.arange(0, GROUP_K)
    output = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for group in tl.static_range(0, GROUPS):
        lhs_offsets = (rows[:, None] * GROUPS + group) * GROUP_K + reduction[None, :]
        rhs_offsets = (group * GROUP_K + reduction[:, None]) * BLOCK_N + columns[None, :]
        lhs_storage = tl.load(lhs_storage_ptr + lhs_offsets)
        rhs_storage = tl.load(rhs_storage_ptr + rhs_offsets)
        lhs = tl.decode_fp8(lhs_storage, format=FORMAT, dtype=tl.float16)
        rhs = tl.decode_fp8(rhs_storage, format=FORMAT, dtype=tl.float16)
        accumulator = tl.dot(lhs, rhs, input_precision="ieee", out_dtype=tl.float32)

        lhs_scale = tl.load(lhs_scale_ptr + rows * GROUPS + group)[:, None]
        rhs_scale = tl.load(rhs_scale_ptr + columns * GROUPS + group)[None, :]
        output += accumulator * lhs_scale * rhs_scale

    output_scale = tl.load(output_scale_ptr + rows)[:, None]
    output_float = output / output_scale
    tl.store(
        output_float_ptr + rows[:, None] * BLOCK_N + columns[None, :],
        output_float,
    )
    output_storage = tl.encode_fp8(output_float, format=FORMAT)
    tl.store(
        output_storage_ptr + rows[:, None] * BLOCK_N + columns[None, :],
        output_storage,
    )


def _format_config(format_name):
    if format_name == "e4m3fn":
        return torch.float8_e4m3fn, 448.0
    if format_name == "e5m2":
        return torch.float8_e5m2, 57344.0
    raise AssertionError(f"unhandled FP8 format: {format_name}")


def _encode_fp8_storage(values, torch_fp8_dtype, max_finite):
    return values.clamp(-max_finite, max_finite).to(torch_fp8_dtype).view(torch.uint8)


def _make_inputs():
    lhs = (
        (torch.arange(BLOCK_M * GROUPS * GROUP_K, dtype=torch.int64) * 7 + 3) % 5 - 2
    ).reshape(BLOCK_M, GROUPS, GROUP_K).to(torch.float32)
    rhs = (
        (torch.arange(GROUPS * GROUP_K * BLOCK_N, dtype=torch.int64) * 11 + 5) % 5 - 2
    ).reshape(GROUPS, GROUP_K, BLOCK_N).to(torch.float32)
    lhs_scale = torch.tensor(
        [[0.5, 1.0], [1.0, 0.25], [0.25, 0.5], [1.0, 1.0]],
        dtype=torch.float32,
    ).repeat(BLOCK_M // 4, 1)
    rhs_scale = torch.tensor(
        [[1.0, 0.5], [0.5, 1.0], [0.25, 1.0], [1.0, 0.25]],
        dtype=torch.float32,
    ).repeat(BLOCK_N // 4, 1)
    output_scale = torch.tensor([2.0, 4.0, 8.0, 16.0], dtype=torch.float32).repeat(BLOCK_M // 4)
    return lhs, rhs, lhs_scale, rhs_scale, output_scale


@pytest.mark.parametrize("format_name", ["e4m3fn", "e5m2"])
def test_fp8_software_storage_input_output_matmul(device, format_name):
    torch_fp8_dtype, max_finite = _format_config(format_name)
    lhs, rhs, lhs_scale, rhs_scale, output_scale = _make_inputs()
    lhs_encoded = _encode_fp8_storage(lhs, torch_fp8_dtype, max_finite)
    rhs_encoded = _encode_fp8_storage(rhs, torch_fp8_dtype, max_finite)
    output_float = torch.empty((BLOCK_M, BLOCK_N), dtype=torch.float32, device=device)
    output_storage = torch.empty((BLOCK_M, BLOCK_N), dtype=torch.uint8, device=device)

    program = _fp8_software_storage_matmul_kernel[(1, )](
        lhs_encoded.to(device),
        rhs_encoded.to(device),
        lhs_scale.to(device),
        rhs_scale.to(device),
        output_scale.to(device),
        output_float,
        output_storage,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        GROUPS=GROUPS,
        GROUP_K=GROUP_K,
        FORMAT=format_name,
    )

    lhs_decoded = lhs_encoded.view(torch_fp8_dtype).to(torch.float32)
    rhs_decoded = rhs_encoded.view(torch_fp8_dtype).to(torch.float32)
    expected = torch.zeros((BLOCK_M, BLOCK_N), dtype=torch.float32)
    for group in range(GROUPS):
        expected += (
            torch.matmul(lhs_decoded[:, group], rhs_decoded[group])
            * lhs_scale[:, group, None]
            * rhs_scale[:, group][None, :]
        )
    expected_float = expected / output_scale[:, None]
    actual_float = output_float.cpu()
    expected_encoded = _encode_fp8_storage(actual_float, torch_fp8_dtype, max_finite)

    torch.testing.assert_close(actual_float, expected_float, rtol=0, atol=1e-6)
    torch.testing.assert_close(output_storage.cpu(), expected_encoded, rtol=0, atol=0)
    assert torch.count_nonzero(expected_encoded) > 0
    assert torch.unique(expected_encoded).numel() > 8
    assert program.asm["ttir"].count("tt.dot") == GROUPS
    assert "f8E4M3FN" not in program.asm["ttir"]
    assert "f8E5M2" not in program.asm["ttir"]
    binary_keys = ("hgbin", "mubin", "mcfatbin", "hsaco", "cubin", "npubin")
    assert any(key in program.asm and len(program.asm[key]) > 0 for key in binary_keys)
