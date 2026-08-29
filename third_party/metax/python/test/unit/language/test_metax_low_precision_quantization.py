import os

os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")

import pytest
import torch
import triton
import triton.language as tl
from triton._C import libtriton

if not hasattr(libtriton, "metax"):
    pytest.skip("MetaX backend not built in libtriton", allow_module_level=True)

_C550_INT8_MMA = "llvm.mxc.mma.i32.16x16x16i8"
_C550_FP16_MMA = "llvm.mxc.mma.f32.16x16x16f16"


@triton.jit
def _int8_dot_quantize_kernel(lhs, rhs, out, OUTPUT_SCALE: tl.constexpr):
    offsets_m = tl.arange(0, 16)
    offsets_n = tl.arange(0, 16)
    offsets_k = tl.arange(0, 32)
    a = tl.load(lhs + offsets_m[:, None] * 32 + offsets_k[None, :])
    b = tl.load(rhs + offsets_k[:, None] * 16 + offsets_n[None, :])
    acc = tl.dot(a, b)
    result = tl.quantize(acc.to(tl.float32), OUTPUT_SCALE, dtype=tl.int8)
    tl.store(out + offsets_m[:, None] * 16 + offsets_n[None, :], result)


@triton.jit
def _int8_storage_fp16_dot_quantize_kernel(
    lhs,
    rhs,
    out,
    INPUT_SCALE: tl.constexpr,
    OUTPUT_SCALE: tl.constexpr,
):
    offsets_m = tl.arange(0, 16)
    offsets_n = tl.arange(0, 16)
    offsets_k = tl.arange(0, 32)
    lhs_q = tl.load(lhs + offsets_m[:, None] * 32 + offsets_k[None, :])
    rhs_q = tl.load(rhs + offsets_k[:, None] * 16 + offsets_n[None, :])
    a = tl.dequantize(lhs_q, INPUT_SCALE, dtype=tl.float16)
    b = tl.dequantize(rhs_q, INPUT_SCALE, dtype=tl.float16)
    acc = tl.dot(a, b, out_dtype=tl.float32)
    result = tl.quantize(acc, OUTPUT_SCALE, dtype=tl.int8)
    tl.store(out + offsets_m[:, None] * 16 + offsets_n[None, :], result)


@triton.jit
def _uint8_storage_fp16_dot_quantize_kernel(
    lhs,
    rhs,
    out,
    INPUT_SCALE: tl.constexpr,
    LHS_ZERO_POINT: tl.constexpr,
    RHS_ZERO_POINT: tl.constexpr,
    OUTPUT_SCALE: tl.constexpr,
    OUTPUT_ZERO_POINT: tl.constexpr,
):
    offsets_m = tl.arange(0, 16)
    offsets_n = tl.arange(0, 16)
    offsets_k = tl.arange(0, 32)
    lhs_q = tl.load(lhs + offsets_m[:, None] * 32 + offsets_k[None, :])
    rhs_q = tl.load(rhs + offsets_k[:, None] * 16 + offsets_n[None, :])
    a = tl.dequantize(
        lhs_q, INPUT_SCALE, dtype=tl.float16, zero_point=LHS_ZERO_POINT
    )
    b = tl.dequantize(
        rhs_q, INPUT_SCALE, dtype=tl.float16, zero_point=RHS_ZERO_POINT
    )
    acc = tl.dot(a, b, out_dtype=tl.float32)
    result = tl.quantize(
        acc,
        OUTPUT_SCALE,
        dtype=tl.uint8,
        zero_point=OUTPUT_ZERO_POINT,
    )
    tl.store(out + offsets_m[:, None] * 16 + offsets_n[None, :], result)


@triton.jit
def _e4m3fn_rtz_quantize_kernel(input_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    values = tl.load(input_ptr + offsets)
    result = tl.quantize(
        values,
        1.0,
        dtype=tl.float8e4nv,
        zero_point=0,
        rounding="rtz",
    )
    tl.store(output_ptr + offsets, result)


def _int8_inputs():
    lhs = (((torch.arange(16 * 32) * 7 + 3) % 17) - 8).reshape(16, 32).to(torch.int8)
    rhs = (((torch.arange(32 * 16) * 11 + 5) % 17) - 8).reshape(32, 16).to(torch.int8)
    return lhs.to("cuda"), rhs.to("cuda")


def _uint8_inputs():
    lhs = ((torch.arange(16 * 32) * 13 + 17) % 256).reshape(16, 32).to(torch.uint8)
    rhs = ((torch.arange(32 * 16) * 29 + 5) % 256).reshape(32, 16).to(torch.uint8)
    return lhs.to("cuda"), rhs.to("cuda")


def _quantize_reference(values, scale):
    return torch.clamp(torch.round(values.to(torch.float32) / scale), -128, 127).to(torch.int8)


def _quantize_uint8_reference(values, scale, zero_point):
    rounded = torch.round(values.to(torch.float32) / scale) + zero_point
    return torch.clamp(rounded, 0, 255).to(torch.uint8)


def _require_c550():
    if not torch.cuda.is_available():
        pytest.skip("requires a MetaX device")
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "maca" or target.arch != 80:
        pytest.skip("requires a C550 / maca arch 80 target")


def test_c550_native_int8_dot_has_explicit_int8_output_epilogue():
    _require_c550()

    lhs, rhs = _int8_inputs()
    output = torch.empty((16, 16), dtype=torch.int8, device="cuda")
    compiled = _int8_dot_quantize_kernel[(1, )](
        lhs,
        rhs,
        output,
        OUTPUT_SCALE=8.0,
        num_warps=4,
        num_stages=1,
    )
    torch.cuda.synchronize()

    accumulator = lhs.cpu().to(torch.int32) @ rhs.cpu().to(torch.int32)
    expected = _quantize_reference(accumulator, 8.0)
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    assert _C550_INT8_MMA in compiled.asm["llir"]
    assert _C550_FP16_MMA not in compiled.asm["llir"]
    assert compiled.asm["mcfatbin"]


def test_c550_int8_storage_fp16_compute_int8_output():
    _require_c550()

    lhs, rhs = _int8_inputs()
    output = torch.empty((16, 16), dtype=torch.int8, device="cuda")
    # Keep this integration oracle away from exact RTNE midpoints, where a
    # sub-ULP FP16 MMA difference can legitimately change the quantized value.
    output_scale = 4.25
    compiled = _int8_storage_fp16_dot_quantize_kernel[(1, )](
        lhs,
        rhs,
        output,
        INPUT_SCALE=0.5,
        OUTPUT_SCALE=output_scale,
        num_warps=4,
        num_stages=1,
    )
    torch.cuda.synchronize()

    lhs_dequantized = (lhs.cpu().to(torch.float32) * 0.5).to(torch.float16).to(torch.float32)
    rhs_dequantized = (rhs.cpu().to(torch.float32) * 0.5).to(torch.float16).to(torch.float32)
    expected = _quantize_reference(lhs_dequantized @ rhs_dequantized, output_scale)
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    assert _C550_FP16_MMA in compiled.asm["llir"]
    assert _C550_INT8_MMA not in compiled.asm["llir"]
    assert compiled.asm["mcfatbin"]


def test_c550_uint8_storage_fp16_compute_uint8_output():
    _require_c550()

    lhs, rhs = _uint8_inputs()
    output = torch.empty((16, 16), dtype=torch.uint8, device="cuda")
    compiled = _uint8_storage_fp16_dot_quantize_kernel[(1, )](
        lhs,
        rhs,
        output,
        INPUT_SCALE=0.0625,
        LHS_ZERO_POINT=127,
        RHS_ZERO_POINT=129,
        OUTPUT_SCALE=1.0,
        OUTPUT_ZERO_POINT=113,
        num_warps=4,
        num_stages=1,
    )
    torch.cuda.synchronize()

    lhs_dequantized = ((lhs.cpu().to(torch.float32) - 127) * 0.0625).to(torch.float16)
    rhs_dequantized = ((rhs.cpu().to(torch.float32) - 129) * 0.0625).to(torch.float16)
    expected = _quantize_uint8_reference(
        lhs_dequantized.to(torch.float32) @ rhs_dequantized.to(torch.float32),
        1.0,
        113,
    )
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    assert _C550_FP16_MMA in compiled.asm["llir"]
    assert _C550_INT8_MMA not in compiled.asm["llir"]
    assert compiled.asm["mcfatbin"]


def test_c550_e4m3fn_rtz_quantize_fails_closed(capfd):
    _require_c550()

    values = torch.arange(32, dtype=torch.float32, device="cuda") / 10
    output = torch.empty(32, dtype=torch.uint8, device="cuda")
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        _e4m3fn_rtz_quantize_kernel[(1, )](
            values,
            triton.reinterpret(output, tl.float8e4nv),
            BLOCK_SIZE=32,
            num_warps=4,
            num_stages=1,
        )

    stderr = capfd.readouterr().err
    assert "Unsupported conversion from f32 to f8E4M3FN" in stderr
    assert "with rounding mode rtz" in stderr
