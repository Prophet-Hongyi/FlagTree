import math

import pytest
import torch
import triton
import triton.language as tl

from low_precision_reference import E4M3FNUZ, decode_fp8, encode_fp8_rtne


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
        result = tl.encode_fp8(values.to(tl.int32))
    elif CASE == "encode_format":
        result = tl.encode_fp8(values, format="e4m3")
    elif CASE == "encode_rounding":
        result = tl.encode_fp8(values, rounding="rtz")
    elif CASE == "encode_overflow":
        result = tl.encode_fp8(values, overflow="nan")
    elif CASE == "decode_input":
        result = tl.decode_fp8(values.to(tl.int8))
    elif CASE == "decode_format":
        result = tl.decode_fp8(values.to(tl.uint8), format="e4m3")
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
    assert "f8E4M3FNUZ" not in program.asm["ttir"]
    assert "f8E5M2" not in program.asm["ttir"]
    binary_keys = ("hgbin", "mubin", "mcfatbin", "hsaco", "cubin", "npubin")
    assert any(key in program.asm and len(program.asm[key]) > 0 for key in binary_keys)


def _fp32_storage_corpus(torch_fp8_dtype, max_finite_code, reference_format=None):
    if reference_format is None:
        finite_codes = torch.arange(max_finite_code + 1, dtype=torch.int16).to(torch.uint8)
        finite_values = finite_codes.view(torch_fp8_dtype).to(torch.float32)
    else:
        finite_values = torch.tensor(
            [decode_fp8(code, reference_format) for code in range(max_finite_code + 1)],
            dtype=torch.float32,
        )
    midpoints = (finite_values[:-1] + finite_values[1:]) * 0.5
    negative_inf = torch.full_like(midpoints, -math.inf)
    positive_inf = torch.full_like(midpoints, math.inf)
    midpoint_neighbors = torch.stack(
        (
            torch.nextafter(midpoints, negative_inf),
            midpoints,
            torch.nextafter(midpoints, positive_inf),
        ),
        dim=1,
    ).flatten()

    indices = torch.arange(65536, dtype=torch.int64)
    random_bits = (indices * 0x9E3779B1 + 0x7F4A7C15) & 0xFFFFFFFF
    random_values = random_bits.to(torch.int32).view(torch.float32)

    mantissas = torch.tensor(
        [
            0x000000,
            0x000001,
            0x07FFFE,
            0x07FFFF,
            0x080000,
            0x080001,
            0x0FFFFE,
            0x0FFFFF,
            0x100000,
            0x100001,
            0x1FFFFE,
            0x1FFFFF,
            0x200000,
            0x200001,
            0x3FFFFF,
            0x400000,
            0x400001,
            0x5FFFFF,
            0x600000,
            0x600001,
            0x7FFFFE,
            0x7FFFFF,
        ],
        dtype=torch.int64,
    )
    exponents = torch.arange(256, dtype=torch.int64) << 23
    structured_bits = (exponents[:, None] | mantissas[None, :]).flatten()
    structured_bits = torch.cat((structured_bits, structured_bits | 0x80000000))
    structured_values = structured_bits.to(torch.int32).view(torch.float32)

    special = torch.tensor(
        [
            -math.inf,
            -0.0,
            0.0,
            math.inf,
            math.nan,
            -math.nan,
        ],
        dtype=torch.float32,
    )
    return torch.cat(
        (
            finite_values,
            -finite_values,
            midpoint_neighbors,
            -midpoint_neighbors,
            random_values,
            structured_values,
            special,
        )
    )


def _reference_fnuz_storage(values):
    return torch.tensor(
        [encode_fp8_rtne(value, E4M3FNUZ) for value in values.to(torch.float32).tolist()],
        dtype=torch.uint8,
    )


def _reference_fnuz_decoded(storage, dtype):
    return torch.tensor(
        [decode_fp8(value, E4M3FNUZ) for value in storage.tolist()],
        dtype=torch.float32,
    ).to(dtype)


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
    "format,torch_fp8_dtype,max_finite,max_finite_code",
    [
        ("e4m3fn", torch.float8_e4m3fn, 448.0, 0x7E),
        ("e5m2", torch.float8_e5m2, 57344.0, 0x7B),
    ],
)
def test_fp8_storage_encode_fp32_boundary_corpus(
    device,
    format,
    torch_fp8_dtype,
    max_finite,
    max_finite_code,
):
    input_cpu = _fp32_storage_corpus(torch_fp8_dtype, max_finite_code)
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

    expected_fp8 = input_cpu.clamp(-max_finite, max_finite).to(torch_fp8_dtype)
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
    "output_dtype,torch_dtype",
    [
        (tl.float16, torch.float16),
        (tl.bfloat16, torch.bfloat16),
        (tl.float32, torch.float32),
    ],
)
def test_e4m3fnuz_storage_roundtrip_device(device, output_dtype, torch_dtype):
    input_cpu = torch.tensor(
        [
            -math.inf,
            -241.0,
            -240.0,
            -232.0,
            -1.0625,
            -(2**-7),
            -(7 * 2**-10),
            -(2**-10),
            -(2**-11),
            -0.0,
            0.0,
            2**-11,
            2**-10,
            7 * 2**-10,
            2**-7,
            1.0625,
            232.0,
            240.0,
            241.0,
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
        BLOCK_SIZE=32,
        OUTPUT_DTYPE=output_dtype,
        FORMAT="e4m3fnuz",
    )

    expected_storage = _reference_fnuz_storage(input_cpu)
    expected_output = _reference_fnuz_decoded(expected_storage, torch_dtype)
    torch.testing.assert_close(storage.cpu(), expected_storage, rtol=0, atol=0)
    torch.testing.assert_close(output.cpu(), expected_output, rtol=0, atol=0, equal_nan=True)
    _assert_software_storage_program(program)


@pytest.mark.parametrize("input_dtype", [torch.float16, torch.bfloat16])
def test_e4m3fnuz_storage_encode_all_16bit_patterns(device, input_dtype):
    input_cpu = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(input_dtype)
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
        FORMAT="e4m3fnuz",
    )

    expected_storage = _reference_fnuz_storage(input_cpu)
    expected_output = _reference_fnuz_decoded(expected_storage, torch.float16)
    torch.testing.assert_close(storage.cpu(), expected_storage, rtol=0, atol=0)
    torch.testing.assert_close(output.cpu(), expected_output, rtol=0, atol=0, equal_nan=True)
    _assert_software_storage_program(program)


def test_e4m3fnuz_storage_encode_fp32_boundary_corpus(device):
    input_cpu = _fp32_storage_corpus(None, 0x7F, reference_format=E4M3FNUZ)
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
        FORMAT="e4m3fnuz",
    )

    expected_storage = _reference_fnuz_storage(input_cpu)
    expected_output = _reference_fnuz_decoded(expected_storage, torch.float16)
    torch.testing.assert_close(storage.cpu(), expected_storage, rtol=0, atol=0)
    torch.testing.assert_close(output.cpu(), expected_output, rtol=0, atol=0, equal_nan=True)
    _assert_software_storage_program(program)


def test_e4m3fnuz_storage_decode_all_byte_patterns(device):
    storage_cpu = torch.arange(256, dtype=torch.int16).to(torch.uint8)
    storage = storage_cpu.to(device)
    output = torch.empty(storage_cpu.shape, dtype=torch.float16, device=device)

    program = _fp8_storage_decode_kernel[(1, )](
        storage,
        output,
        n_elements=storage_cpu.numel(),
        BLOCK_SIZE=256,
        FORMAT="e4m3fnuz",
    )

    expected = _reference_fnuz_decoded(storage_cpu, torch.float16)
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0, equal_nan=True)
    _assert_software_storage_program(program)


@pytest.mark.parametrize(
    "case,error",
    [
        ("encode_input", "encode_fp8 input must be tl.float16, tl.bfloat16, or tl.float32"),
        ("encode_format", "encode_fp8 format must be 'e4m3fn', 'e4m3fnuz', or 'e5m2'"),
        ("encode_rounding", "encode_fp8 rounding must be 'rtne'"),
        ("encode_overflow", "encode_fp8 overflow must be 'satfinite'"),
        ("decode_input", "decode_fp8 input must be tl.uint8 storage"),
        ("decode_format", "decode_fp8 format must be 'e4m3fn', 'e4m3fnuz', or 'e5m2'"),
        ("decode_dtype", "decode_fp8 dtype must be tl.float16, tl.bfloat16, or tl.float32"),
    ],
)
def test_fp8_storage_codec_rejects_unsupported_contracts(device, case, error):
    input = torch.zeros(16, dtype=torch.float16, device=device)
    output = torch.empty_like(input)
    with pytest.raises(triton.CompilationError) as exc_info:
        _invalid_fp8_storage_codec_kernel[(1, )](input, output, BLOCK_SIZE=16, CASE=case)
    assert error in _exception_chain_text(exc_info.value)
