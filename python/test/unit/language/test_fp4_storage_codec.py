import math

import pytest
import torch
import triton
import triton.language as tl

from low_precision_reference import decode_fp4_e2m1, pack_fp4_e2m1, unpack_fp4_e2m1


@triton.jit
def _fp4_storage_roundtrip_kernel(
    low_ptr,
    high_ptr,
    storage_ptr,
    decoded_low_ptr,
    decoded_high_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    low = tl.load(low_ptr + offsets, mask=mask, other=0.0)
    high = tl.load(high_ptr + offsets, mask=mask, other=0.0)
    storage = tl.encode_fp4(low, high)
    decoded_low, decoded_high = tl.decode_fp4(storage, dtype=OUTPUT_DTYPE)
    tl.store(storage_ptr + offsets, storage, mask=mask)
    tl.store(decoded_low_ptr + offsets, decoded_low, mask=mask)
    tl.store(decoded_high_ptr + offsets, decoded_high, mask=mask)


@triton.jit
def _fp4_storage_decode_kernel(
    storage_ptr,
    decoded_low_ptr,
    decoded_high_ptr,
    BLOCK_SIZE: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    storage = tl.load(storage_ptr + offsets)
    decoded_low, decoded_high = tl.decode_fp4(storage, dtype=OUTPUT_DTYPE)
    tl.store(decoded_low_ptr + offsets, decoded_low)
    tl.store(decoded_high_ptr + offsets, decoded_high)


@triton.jit
def _invalid_fp4_storage_codec_kernel(
    input_ptr,
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
    CASE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    values = tl.load(input_ptr + offsets)
    if CASE == "encode_input":
        result = tl.encode_fp4(values.to(tl.int32), values.to(tl.int32))
    elif CASE == "encode_format":
        result = tl.encode_fp4(values, values, format="e3m0")
    elif CASE == "encode_rounding":
        result = tl.encode_fp4(values, values, rounding="rtz")
    elif CASE == "encode_overflow":
        result = tl.encode_fp4(values, values, overflow="nan")
    elif CASE == "decode_input":
        result, _ = tl.decode_fp4(values.to(tl.int8))
    elif CASE == "decode_format":
        result, _ = tl.decode_fp4(values.to(tl.uint8), format="e3m0")
    elif CASE == "decode_dtype":
        result, _ = tl.decode_fp4(values.to(tl.uint8), dtype=tl.int32)
    else:
        tl.static_assert(False, "unknown FP4 storage codec case")
    tl.store(output_ptr + offsets, result)


def _exception_chain_text(exception):
    messages = []
    seen = set()
    while exception is not None and id(exception) not in seen:
        seen.add(id(exception))
        messages.append(str(exception))
        exception = exception.__cause__ or exception.__context__
    return "\n".join(messages)


def _assert_software_storage_program(program):
    assert "f4E2M1FN" not in program.asm["ttir"]
    binary_keys = ("hgbin", "mubin", "mcfatbin", "hsaco", "cubin", "npubin")
    assert any(key in program.asm and len(program.asm[key]) > 0 for key in binary_keys)


def _boundary_corpus(dtype):
    representable = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
    midpoints = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)
    midpoint_neighbors = torch.stack(
        (
            torch.nextafter(midpoints, torch.full_like(midpoints, -math.inf)),
            midpoints,
            torch.nextafter(midpoints, torch.full_like(midpoints, math.inf)),
        ),
        dim=1,
    ).flatten()
    negative_representable = -representable
    negative_midpoint_neighbors = -midpoint_neighbors
    specials = torch.tensor(
        [math.inf, -math.inf, math.nan, math.copysign(math.nan, -1.0)],
        dtype=torch.float32,
    )
    values = torch.cat(
        (representable, negative_representable, midpoint_neighbors, negative_midpoint_neighbors, specials)
    )
    return values.to(dtype)


def _assert_exact_bits(actual, expected):
    assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_fp4_e2m1_storage_roundtrip_boundaries(device, dtype):
    low_cpu = _boundary_corpus(dtype)
    high_cpu = low_cpu.flip(0).clone()
    storage = torch.empty(low_cpu.shape, dtype=torch.uint8, device=device)
    decoded_low = torch.empty(low_cpu.shape, dtype=dtype, device=device)
    decoded_high = torch.empty(high_cpu.shape, dtype=dtype, device=device)

    output_dtype = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
    }[dtype]
    program = _fp4_storage_roundtrip_kernel[(1, )](
        low_cpu.to(device),
        high_cpu.to(device),
        storage,
        decoded_low,
        decoded_high,
        n_elements=low_cpu.numel(),
        BLOCK_SIZE=64,
        OUTPUT_DTYPE=output_dtype,
    )

    expected_storage = torch.tensor(
        [pack_fp4_e2m1(low, high) for low, high in zip(low_cpu.tolist(), high_cpu.tolist())],
        dtype=torch.uint8,
    )
    expected_pairs = [unpack_fp4_e2m1(value) for value in expected_storage.tolist()]
    expected_low = torch.tensor([pair[0] for pair in expected_pairs], dtype=dtype)
    expected_high = torch.tensor([pair[1] for pair in expected_pairs], dtype=dtype)

    assert torch.equal(storage.cpu(), expected_storage)
    _assert_exact_bits(decoded_low.cpu(), expected_low)
    _assert_exact_bits(decoded_high.cpu(), expected_high)
    _assert_software_storage_program(program)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_fp4_e2m1_decode_all_packed_bytes(device, dtype):
    storage_cpu = torch.arange(256, dtype=torch.uint8)
    decoded_low = torch.empty(256, dtype=dtype, device=device)
    decoded_high = torch.empty(256, dtype=dtype, device=device)
    output_dtype = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
        torch.float32: tl.float32,
    }[dtype]

    program = _fp4_storage_decode_kernel[(1, )](
        storage_cpu.to(device),
        decoded_low,
        decoded_high,
        BLOCK_SIZE=256,
        OUTPUT_DTYPE=output_dtype,
    )

    expected_low = torch.tensor([decode_fp4_e2m1(value & 0xF) for value in range(256)], dtype=dtype)
    expected_high = torch.tensor([decode_fp4_e2m1(value >> 4) for value in range(256)], dtype=dtype)
    _assert_exact_bits(decoded_low.cpu(), expected_low)
    _assert_exact_bits(decoded_high.cpu(), expected_high)
    _assert_software_storage_program(program)


@pytest.mark.parametrize(
    "case,error",
    [
        ("encode_input", "encode_fp4 inputs must be tl.float16, tl.bfloat16, or tl.float32"),
        ("encode_format", "encode_fp4 format must be 'e2m1'"),
        ("encode_rounding", "encode_fp4 rounding must be 'rtne'"),
        ("encode_overflow", "encode_fp4 overflow must be 'satfinite'"),
        ("decode_input", "decode_fp4 input must be tl.uint8 storage"),
        ("decode_format", "decode_fp4 format must be 'e2m1'"),
        ("decode_dtype", "decode_fp4 dtype must be tl.float16, tl.bfloat16, or tl.float32"),
    ],
)
def test_fp4_storage_codec_rejects_unsupported_contracts(device, case, error):
    input = torch.zeros(16, dtype=torch.float16, device=device)
    output = torch.empty_like(input)
    with pytest.raises(triton.CompilationError) as exc_info:
        _invalid_fp4_storage_codec_kernel[(1, )](input, output, BLOCK_SIZE=16, CASE=case)
    assert error in _exception_chain_text(exc_info.value)
