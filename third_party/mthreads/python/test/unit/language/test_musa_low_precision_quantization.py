import os

os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")

import pytest
import torch
import triton
import triton.language as tl
from triton._C import libtriton

if not hasattr(libtriton, "mthreads"):
    pytest.skip("musa backend not built in libtriton", allow_module_level=True)

_PH1_INT8_MMA = "llvm.musa.sqmma.smma"
_PH1_BF16_MMA = "llvm.musa.sqmma.bfmma.m32n32k64.mma"


@triton.jit
def _int8_dot_quantize_kernel(lhs, rhs, out, OUTPUT_SCALE: tl.constexpr):
    offsets_m = tl.arange(0, 32)
    offsets_n = tl.arange(0, 32)
    offsets_k = tl.arange(0, 64)
    a = tl.load(lhs + offsets_m[:, None] * 64 + offsets_k[None, :])
    b = tl.load(rhs + offsets_k[:, None] * 32 + offsets_n[None, :])
    acc = tl.dot(a, b)
    result = tl.quantize(acc.to(tl.float32), OUTPUT_SCALE, dtype=tl.int8)
    tl.store(out + offsets_m[:, None] * 32 + offsets_n[None, :], result)


@triton.jit
def _int8_storage_bf16_dot_quantize_kernel(
    lhs,
    rhs,
    out,
    INPUT_SCALE: tl.constexpr,
    OUTPUT_SCALE: tl.constexpr,
):
    offsets_m = tl.arange(0, 32)
    offsets_n = tl.arange(0, 32)
    offsets_k = tl.arange(0, 64)
    lhs_q = tl.load(lhs + offsets_m[:, None] * 64 + offsets_k[None, :])
    rhs_q = tl.load(rhs + offsets_k[:, None] * 32 + offsets_n[None, :])
    a = tl.dequantize(lhs_q, INPUT_SCALE, dtype=tl.bfloat16)
    b = tl.dequantize(rhs_q, INPUT_SCALE, dtype=tl.bfloat16)
    acc = tl.dot(a, b, out_dtype=tl.float32)
    result = tl.quantize(acc, OUTPUT_SCALE, dtype=tl.int8)
    tl.store(out + offsets_m[:, None] * 32 + offsets_n[None, :], result)


def _int8_inputs():
    lhs = (((torch.arange(32 * 64) * 7 + 3) % 17) - 8).reshape(32, 64).to(torch.int8)
    rhs = (((torch.arange(64 * 32) * 11 + 5) % 17) - 8).reshape(64, 32).to(torch.int8)
    return lhs.to("musa"), rhs.to("musa")


def _quantize_reference(values, scale):
    return torch.clamp(torch.round(values.to(torch.float32) / scale), -128, 127).to(torch.int8)


def _require_ph1():
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("requires a MUSA device")
    if tuple(torch.musa.get_device_capability(0)) != (3, 1):
        pytest.skip("requires a PH1 device")


def test_ph1_native_int8_dot_has_explicit_int8_output_epilogue():
    _require_ph1()

    lhs, rhs = _int8_inputs()
    output = torch.empty((32, 32), dtype=torch.int8, device="musa")
    compiled = _int8_dot_quantize_kernel[(1, )](
        lhs,
        rhs,
        output,
        OUTPUT_SCALE=8.0,
        num_warps=4,
    )
    torch.musa.synchronize()

    accumulator = lhs.cpu().to(torch.int32) @ rhs.cpu().to(torch.int32)
    expected = _quantize_reference(accumulator, 8.0)
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    assert _PH1_INT8_MMA in compiled.asm["llir"]
    assert _PH1_BF16_MMA not in compiled.asm["llir"]
    assert compiled.asm["mubin"]


def test_ph1_int8_storage_bf16_compute_int8_output():
    _require_ph1()

    lhs, rhs = _int8_inputs()
    output = torch.empty((32, 32), dtype=torch.int8, device="musa")
    compiled = _int8_storage_bf16_dot_quantize_kernel[(1, )](
        lhs,
        rhs,
        output,
        INPUT_SCALE=0.5,
        OUTPUT_SCALE=4.0,
        num_warps=4,
    )
    torch.musa.synchronize()

    lhs_dequantized = (lhs.cpu().to(torch.float32) * 0.5).to(torch.bfloat16).to(torch.float32)
    rhs_dequantized = (rhs.cpu().to(torch.float32) * 0.5).to(torch.bfloat16).to(torch.float32)
    expected = _quantize_reference(lhs_dequantized @ rhs_dequantized, 4.0)
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    assert _PH1_BF16_MMA in compiled.asm["llir"]
    assert _PH1_INT8_MMA not in compiled.asm["llir"]
    assert compiled.asm["mubin"]
