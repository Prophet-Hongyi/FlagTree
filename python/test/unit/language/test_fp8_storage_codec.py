import math

import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _fp8_storage_roundtrip_kernel(
    input_ptr,
    storage_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
    FORMAT: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    storage = tl.encode_fp8(values, format=FORMAT)
    decoded = tl.decode_fp8(storage, format=FORMAT, dtype=OUTPUT_DTYPE)
    tl.store(storage_ptr + offsets, storage, mask=mask)
    tl.store(output_ptr + offsets, decoded, mask=mask)


@triton.jit
def _fp8_storage_decode_kernel(
    storage_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    FORMAT: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    storage = tl.load(storage_ptr + offsets, mask=mask, other=0)
    tl.store(output_ptr + offsets, tl.decode_fp8(storage, format=FORMAT), mask=mask)


@triton.jit
def _invalid_fp8_storage_codec_kernel(
    input_ptr,
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
    CASE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    values = tl.load(input_ptr + offsets)
    if CASE == "encode_input":
        result = tl.encode_fp8(values.to(tl.float32))
    elif CASE == "encode_format":
        result = tl.encode_fp8(values, format="e4m3fnuz")
    elif CASE == "encode_rounding":
        result = tl.encode_fp8(values, rounding="rtz")
    elif CASE == "encode_overflow":
        result = tl.encode_fp8(values, overflow="nan")
    elif CASE == "decode_input":
        result = tl.decode_fp8(values.to(tl.int8))
    elif CASE == "decode_format":
        result = tl.decode_fp8(values.to(tl.uint8), format="e4m3fnuz")
    elif CASE == "decode_dtype":
        result = tl.decode_fp8(values.to(tl.uint8), dtype=tl.float64)
    else:
        tl.static_assert(False, "unknown FP8 storage codec case")
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
    assert "f8E4M3FN" not in program.asm["ttir"]
    assert "f8E5M2" not in program.asm["ttir"]
    binary_keys = ("hgbin", "mubin", "mcfatbin", "hsaco", "cubin", "npubin")
    assert any(key in program.asm and len(program.asm[key]) > 0 for key in binary_keys)


@pytest.mark.parametrize(
    "output_dtype,torch_dtype",
    [
        (tl.float16, torch.float16),
        (tl.bfloat16, torch.bfloat16),
        (tl.float32, torch.float32),
    ],
)
@pytest.mark.parametrize(
    "format,torch_fp8_dtype,max_finite",
    [
        ("e4m3fn", torch.float8_e4m3fn, 448.0),
        ("e5m2", torch.float8_e5m2, 57344.0),
    ],
)
def test_fp8_storage_roundtrip_device(
    device,
    output_dtype,
    torch_dtype,
    format,
    torch_fp8_dtype,
    max_finite,
):
    input_cpu = torch.tensor(
        [
            -math.inf,
            -465.0,
            -60000.0,
            -57344.0,
            -449.0,
            -448.0,
            -300.0,
            -1.1875,
            -1.0625,
            -0.015625,
            -0.0146484375,
            -0.0029296875,
            -0.001953125,
            -0.0009765625,
            -(2**-16),
            -(2**-17),
            -0.0,
            0.0,
            0.0009765625,
            2**-17,
            2**-16,
            0.001953125,
            0.0029296875,
            0.0146484375,
            0.015625,
            1.0625,
            1.1875,
            300.0,
            448.0,
            449.0,
            465.0,
            57344.0,
            60000.0,
            math.inf,
            math.nan,
        ],
        dtype=torch.float16,
    )
    input = input_cpu.to(device)
    storage = torch.empty(input_cpu.shape, dtype=torch.uint8, device=device)
    output = torch.empty(input_cpu.shape, dtype=torch_dtype, device=device)

    program = _fp8_storage_roundtrip_kernel[(1, )](
        input,
        storage,
        output,
        n_elements=input_cpu.numel(),
        BLOCK_SIZE=64,
        OUTPUT_DTYPE=output_dtype,
        FORMAT=format,
    )

    expected_fp8 = input_cpu.float().clamp(-max_finite, max_finite).to(torch_fp8_dtype)
    torch.testing.assert_close(storage.cpu(), expected_fp8.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(
        output.cpu(),
        expected_fp8.to(torch_dtype),
        rtol=0,
        atol=0,
        equal_nan=True,
    )

    _assert_software_storage_program(program)


@pytest.mark.parametrize(
    "format,torch_fp8_dtype,max_finite",
    [
        ("e4m3fn", torch.float8_e4m3fn, 448.0),
        ("e5m2", torch.float8_e5m2, 57344.0),
    ],
)
def test_fp8_storage_encode_all_fp16_bit_patterns(device, format, torch_fp8_dtype, max_finite):
    input_cpu = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.float16)
    input = input_cpu.to(device)
    storage = torch.empty(input_cpu.shape, dtype=torch.uint8, device=device)
    output = torch.empty_like(input)

    block_size = 256
    program = _fp8_storage_roundtrip_kernel[(triton.cdiv(input_cpu.numel(), block_size), )](
        input,
        storage,
        output,
        n_elements=input_cpu.numel(),
        BLOCK_SIZE=block_size,
        OUTPUT_DTYPE=tl.float16,
        FORMAT=format,
    )

    expected_fp8 = input_cpu.float().clamp(-max_finite, max_finite).to(torch_fp8_dtype)
    torch.testing.assert_close(storage.cpu(), expected_fp8.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(output.cpu(), expected_fp8.to(torch.float16), rtol=0, atol=0, equal_nan=True)
    _assert_software_storage_program(program)


@pytest.mark.parametrize(
    "format,torch_fp8_dtype,max_finite",
    [
        ("e4m3fn", torch.float8_e4m3fn, 448.0),
        ("e5m2", torch.float8_e5m2, 57344.0),
    ],
)
def test_fp8_storage_encode_all_bf16_bit_patterns(
    device,
    format,
    torch_fp8_dtype,
    max_finite,
):
    input_cpu = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    input = input_cpu.to(device)
    storage = torch.empty(input_cpu.shape, dtype=torch.uint8, device=device)
    output = torch.empty(input_cpu.shape, dtype=torch.float16, device=device)

    block_size = 256
    program = _fp8_storage_roundtrip_kernel[(triton.cdiv(input_cpu.numel(), block_size), )](
        input,
        storage,
        output,
        n_elements=input_cpu.numel(),
        BLOCK_SIZE=block_size,
        OUTPUT_DTYPE=tl.float16,
        FORMAT=format,
    )

    expected_fp8 = input_cpu.float().clamp(-max_finite, max_finite).to(torch_fp8_dtype)
    torch.testing.assert_close(storage.cpu(), expected_fp8.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(output.cpu(), expected_fp8.to(torch.float16), rtol=0, atol=0, equal_nan=True)
    _assert_software_storage_program(program)


@pytest.mark.parametrize(
    "format,torch_fp8_dtype",
    [
        ("e4m3fn", torch.float8_e4m3fn),
        ("e5m2", torch.float8_e5m2),
    ],
)
def test_fp8_storage_decode_all_byte_patterns(device, format, torch_fp8_dtype):
    storage_cpu = torch.arange(256, dtype=torch.int16).to(torch.uint8)
    storage = storage_cpu.to(device)
    output = torch.empty(storage_cpu.shape, dtype=torch.float16, device=device)

    program = _fp8_storage_decode_kernel[(1, )](
        storage,
        output,
        n_elements=storage_cpu.numel(),
        BLOCK_SIZE=256,
        FORMAT=format,
    )

    expected = storage_cpu.view(torch_fp8_dtype).to(torch.float16)
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0, equal_nan=True)
    _assert_software_storage_program(program)


@pytest.mark.parametrize(
    "case,error",
    [
        ("encode_input", "encode_fp8 input must be tl.float16 or tl.bfloat16"),
        ("encode_format", "encode_fp8 format must be 'e4m3fn' or 'e5m2'"),
        ("encode_rounding", "encode_fp8 rounding must be 'rtne'"),
        ("encode_overflow", "encode_fp8 overflow must be 'satfinite'"),
        ("decode_input", "decode_fp8 input must be tl.uint8 storage"),
        ("decode_format", "decode_fp8 format must be 'e4m3fn' or 'e5m2'"),
        ("decode_dtype", "decode_fp8 dtype must be tl.float16, tl.bfloat16, or tl.float32"),
    ],
)
def test_fp8_storage_codec_rejects_unsupported_contracts(device, case, error):
    input = torch.zeros(16, dtype=torch.float16, device=device)
    output = torch.empty_like(input)
    with pytest.raises(triton.CompilationError) as exc_info:
        _invalid_fp8_storage_codec_kernel[(1, )](input, output, BLOCK_SIZE=16, CASE=case)
    assert error in _exception_chain_text(exc_info.value)
