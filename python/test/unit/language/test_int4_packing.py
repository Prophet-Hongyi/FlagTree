import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _int4_roundtrip_kernel(
    low_ptr,
    high_ptr,
    packed_ptr,
    decoded_low_ptr,
    decoded_high_ptr,
    BLOCK_SIZE: tl.constexpr,
    SIGNED: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    low = tl.load(low_ptr + offsets)
    high = tl.load(high_ptr + offsets)
    packed = tl.pack_int4(low, high, signed=SIGNED)
    decoded_low, decoded_high = tl.unpack_int4(packed, signed=SIGNED)
    tl.store(packed_ptr + offsets, packed)
    tl.store(decoded_low_ptr + offsets, decoded_low)
    tl.store(decoded_high_ptr + offsets, decoded_high)


@triton.jit
def _invalid_pack_int4_kernel(
    low_ptr,
    high_ptr,
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
    SIGNED: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    low = tl.load(low_ptr + offsets)
    high = tl.load(high_ptr + offsets)
    packed = tl.pack_int4(low, high, signed=SIGNED)
    tl.store(output_ptr + offsets, packed)


@triton.jit
def _invalid_unpack_int4_kernel(input_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    packed = tl.load(input_ptr + offsets)
    low, _ = tl.unpack_int4(packed)
    tl.store(output_ptr + offsets, low)


def _exception_chain_text(exception):
    messages = []
    seen = set()
    while exception is not None and id(exception) not in seen:
        seen.add(id(exception))
        messages.append(str(exception))
        exception = exception.__cause__ or exception.__context__
    return "\n".join(messages)


@pytest.mark.parametrize(
    "signed,dtype,low_values,high_values",
    [
        (True, torch.int8, [-8, -7, -1, 0, 1, 6, 7, -3], [7, 6, 1, 0, -1, -7, -8, 3]),
        (False, torch.uint8, [0, 1, 2, 7, 8, 13, 14, 15], [15, 14, 13, 8, 7, 2, 1, 0]),
    ],
)
def test_int4_pair_pack_unpack_device(device, signed, dtype, low_values, high_values):
    low = torch.tensor(low_values, dtype=dtype, device=device)
    high = torch.tensor(high_values, dtype=dtype, device=device)
    packed = torch.empty(low.shape, dtype=torch.uint8, device=device)
    decoded_low = torch.empty_like(low)
    decoded_high = torch.empty_like(high)

    _int4_roundtrip_kernel[(1, )](
        low,
        high,
        packed,
        decoded_low,
        decoded_high,
        BLOCK_SIZE=low.numel(),
        SIGNED=signed,
    )

    expected = (low.cpu().to(torch.uint8) & 0xF) | ((high.cpu().to(torch.uint8) & 0xF) << 4)
    torch.testing.assert_close(packed.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(decoded_low.cpu(), low.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(decoded_high.cpu(), high.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize(
    "dtype,signed,error",
    [
        (torch.int16, True, "signed INT4 packing requires tl.int8 inputs"),
        (torch.int8, False, "UINT4 packing requires tl.uint8 inputs"),
    ],
)
def test_pack_int4_rejects_wrong_physical_dtype(device, dtype, signed, error):
    low = torch.zeros(8, dtype=dtype, device=device)
    high = torch.zeros_like(low)
    output = torch.empty(8, dtype=torch.uint8, device=device)

    with pytest.raises(Exception) as exc_info:
        _invalid_pack_int4_kernel[(1, )](low, high, output, BLOCK_SIZE=8, SIGNED=signed)

    assert error in _exception_chain_text(exc_info.value)


def test_unpack_int4_requires_uint8_storage(device):
    packed = torch.zeros(8, dtype=torch.int8, device=device)
    output = torch.empty_like(packed)

    with pytest.raises(Exception) as exc_info:
        _invalid_unpack_int4_kernel[(1, )](packed, output, BLOCK_SIZE=8)

    assert "INT4 unpacking requires a tl.uint8 packed input" in _exception_chain_text(exc_info.value)
