import math

import pytest
import torch
import triton
import triton.language as tl

from low_precision_reference import pack_fp4_e2m1, unpack_fp4_e2m1


@triton.jit
def _scaled_fp4_storage_policy_kernel(
    input_ptr,
    scale_ptr,
    packed_ptr,
    output_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GRANULARITY: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
):
    pairs: tl.constexpr = (K + 1) // 2
    groups: tl.constexpr = (K + GROUP_SIZE - 1) // GROUP_SIZE
    rows = tl.arange(0, M)[:, None]
    pair_offsets = tl.arange(0, pairs)[None, :]
    low_k = pair_offsets * 2
    high_k = low_k + 1

    if GRANULARITY == "tensor":
        scale = tl.load(scale_ptr)
    elif GRANULARITY == "row":
        scale = tl.load(scale_ptr + rows)
    elif GRANULARITY == "group":
        scale = tl.load(scale_ptr + rows * groups + low_k // GROUP_SIZE)
    else:
        tl.static_assert(False, "unknown scale granularity")

    low = tl.load(input_ptr + rows * K + low_k)
    high = tl.load(input_ptr + rows * K + high_k, mask=high_k < K, other=0.0)
    packed = tl.quantize_fp4(low, high, scale)
    tl.store(packed_ptr + rows * pairs + pair_offsets, packed)

    output_low, output_high = tl.dequantize_fp4(packed, scale, dtype=OUTPUT_DTYPE)
    padded_k: tl.constexpr = pairs * 2
    tl.store(output_ptr + rows * padded_k + low_k, output_low)
    tl.store(output_ptr + rows * padded_k + high_k, output_high)


@triton.jit
def _scaled_fp4_special_values_kernel(
    low_ptr,
    high_ptr,
    scale_ptr,
    packed_ptr,
    output_low_ptr,
    output_high_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    low = tl.load(low_ptr + offsets, mask=mask, other=0.0)
    high = tl.load(high_ptr + offsets, mask=mask, other=0.0)
    scale = tl.load(scale_ptr)
    packed = tl.quantize_fp4(low, high, scale)
    output_low, output_high = tl.dequantize_fp4(packed, scale, dtype=OUTPUT_DTYPE)
    tl.store(packed_ptr + offsets, packed, mask=mask)
    tl.store(output_low_ptr + offsets, output_low, mask=mask)
    tl.store(output_high_ptr + offsets, output_high, mask=mask)


def _assert_target_binary(program):
    binary_keys = ("hgbin", "mubin", "mcfatbin", "hsaco", "cubin", "npubin")
    assert any(key in program.asm and len(program.asm[key]) > 0 for key in binary_keys)


def _assert_exact_bits(actual, expected):
    assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))


def _make_case(dtype, granularity):
    m = 16
    k = 64 if granularity == "tensor" else 63
    group_size = 16
    groups = (k + group_size - 1) // group_size
    padded_k = ((k + 1) // 2) * 2
    e2m1_values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
    )
    scale_values = torch.tensor([0.75, 1.25, 1.75, 2.5, 3.25], dtype=torch.float32)
    rows = torch.arange(m)[:, None]
    logical_k = torch.arange(k)[None, :]
    codes = ((rows * 7 + logical_k * 5 + logical_k // group_size) % 16).to(torch.uint8)

    if granularity == "tensor":
        scale = scale_values[1:2].clone()
        expanded_scale = scale.reshape(1, 1).expand(m, k)
    elif granularity == "row":
        scale = scale_values[(torch.arange(m) * 3) % len(scale_values)].clone()
        expanded_scale = scale[:, None].expand(m, k)
    else:
        group_rows = torch.arange(m)[:, None]
        group_columns = torch.arange(groups)[None, :]
        scale = scale_values[(group_rows * 2 + group_columns * 3) % len(scale_values)].clone()
        expanded_scale = scale.repeat_interleave(group_size, dim=1)[:, :k]

    logical_values = (e2m1_values[codes.to(torch.long)] * expanded_scale).to(dtype)
    padded_values = torch.zeros((m, padded_k), dtype=dtype)
    padded_values[:, :k] = logical_values
    padded_codes = torch.zeros((m, padded_k), dtype=torch.uint8)
    padded_codes[:, :k] = codes
    expected_packed = (padded_codes[:, 0::2] | (padded_codes[:, 1::2] << 4)).to(torch.uint8)
    return logical_values, scale, padded_values, expected_packed, k, group_size


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("granularity", ["tensor", "row", "group"])
def test_fp4_runtime_scale_broadcast_and_explicit_odd_tail(device, dtype, granularity):
    logical_values, scale_cpu, expected_output, expected_packed, k, group_size = _make_case(dtype, granularity)
    m = logical_values.shape[0]
    packed = torch.empty_like(expected_packed, device=device)
    output = torch.empty_like(expected_output, device=device)
    output_dtype = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
    }[dtype]

    program = _scaled_fp4_storage_policy_kernel[(1, )](
        logical_values.to(device),
        scale_cpu.to(device),
        packed,
        output,
        M=m,
        K=k,
        GROUP_SIZE=group_size,
        GRANULARITY=granularity,
        OUTPUT_DTYPE=output_dtype,
    )

    assert torch.equal(packed.cpu(), expected_packed)
    _assert_exact_bits(output.cpu(), expected_output)
    if k % 2:
        assert torch.count_nonzero(packed.cpu()[:, -1] & 0xF0) == 0
        assert torch.count_nonzero(output.cpu()[:, -1]) == 0
    _assert_target_binary(program)
    assert "f4E2M1FN" not in program.asm["ttir"]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_fp4_runtime_scale_special_values_and_rtne(device, dtype):
    scale_value = 1.25
    low_cpu = torch.tensor(
        [
            0.0,
            -0.0,
            math.inf,
            -math.inf,
            math.nan,
            math.copysign(math.nan, -1.0),
            7.5,
            -7.5,
            100.0,
            -100.0,
            0.3125,
            -0.3125,
            0.9375,
            -0.9375,
            1.5625,
            -1.5625,
        ],
        dtype=dtype,
    )
    high_cpu = low_cpu.flip(0).clone()
    scale_cpu = torch.tensor([scale_value], dtype=torch.float32)
    packed = torch.empty(low_cpu.shape, dtype=torch.uint8, device=device)
    output_low = torch.empty(low_cpu.shape, dtype=dtype, device=device)
    output_high = torch.empty(high_cpu.shape, dtype=dtype, device=device)
    output_dtype = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
    }[dtype]

    program = _scaled_fp4_special_values_kernel[(1, )](
        low_cpu.to(device),
        high_cpu.to(device),
        scale_cpu.to(device),
        packed,
        output_low,
        output_high,
        N=low_cpu.numel(),
        BLOCK_SIZE=16,
        OUTPUT_DTYPE=output_dtype,
    )

    expected_packed = torch.tensor(
        [
            pack_fp4_e2m1(float(low) / scale_value, float(high) / scale_value)
            for low, high in zip(low_cpu.tolist(), high_cpu.tolist())
        ],
        dtype=torch.uint8,
    )
    expected_pairs = [unpack_fp4_e2m1(value) for value in expected_packed.tolist()]
    expected_low = torch.tensor([pair[0] * scale_value for pair in expected_pairs], dtype=dtype)
    expected_high = torch.tensor([pair[1] * scale_value for pair in expected_pairs], dtype=dtype)

    assert torch.equal(packed.cpu(), expected_packed)
    _assert_exact_bits(output_low.cpu(), expected_low)
    _assert_exact_bits(output_high.cpu(), expected_high)
    _assert_target_binary(program)
    assert "f4E2M1FN" not in program.asm["ttir"]
