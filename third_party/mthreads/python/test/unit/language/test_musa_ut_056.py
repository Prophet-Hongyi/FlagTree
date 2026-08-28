import os

os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")

import pytest
import torch
import triton
import triton.language as tl
from triton._C import libtriton
from pathlib import Path
from triton.compiler.errors import CompilationError

if not hasattr(libtriton, "mthreads"):
    pytest.skip("musa backend not built in libtriton", allow_module_level=True)

from triton.backends import backends
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton._C.libtriton import ir


def _get_musa_backend():
    if "mthreads" not in backends:
        pytest.skip("musa backend not discovered")
    target = GPUTarget("musa", "ph1", 32)
    return backends["mthreads"].compiler(target)


def _compile_to_llir(fn, signature, constexprs=None):
    target = GPUTarget("musa", "ph1", 32)
    backend = _get_musa_backend()

    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)

    options = backend.parse_options({})
    module_map = backend.get_module_map()
    codegen_fns = backend.get_codegen_implementation(options)
    src = ASTSource(fn=fn, signature=signature, constexprs=constexprs or {})

    ttir = src.make_ir(target, options, codegen_fns, module_map, context)
    stages = {}
    backend.add_stages(stages, options, src.language)
    meta = {}
    ttir = stages["ttir"](ttir, meta)
    ttgir = stages["ttgir"](ttir, meta)
    llir = stages["llir"](ttgir, meta)
    return llir, meta


def test_musa_056_default_libdevice_path(fresh_knobs):
    backend = _get_musa_backend()
    from triton.backends.mthreads import compiler as mthreads_compiler

    with fresh_knobs.musa.scope():
        del fresh_knobs.musa.libdevice_path
        options = backend.parse_options({})

    expected = Path(mthreads_compiler.__file__).resolve().parent / "lib" / "libdevice.31.bc"
    assert Path(dict(options.extern_libs)["libdevice"]).resolve() == expected


def test_musa_056_libdevice_path_override(fresh_knobs, tmp_path):
    backend = _get_musa_backend()
    override = tmp_path / "libdevice.override.bc"
    override.write_bytes(b"")

    with fresh_knobs.musa.scope():
        fresh_knobs.musa.libdevice_path = str(override)
        options = backend.parse_options({})

    assert dict(options.extern_libs)["libdevice"] == str(override)


def test_musa_056_cast_compile_only():

    @triton.jit
    def kernel_cast(inp, out):
        offs = tl.arange(0, 64)
        x = tl.load(inp + offs)
        y = x.to(tl.float16)
        z = y.to(tl.float32)
        tl.store(out + offs, z)

    llir, _ = _compile_to_llir(kernel_cast, {"inp": "*fp32", "out": "*fp32"})
    assert "fptrunc" in llir
    assert "fpext" in llir


def test_musa_056_chained_dot_compile_only():

    @triton.jit
    def kernel_chained_dot(out):
        a = tl.full((16, 16), 1.0, tl.float16)
        b = tl.full((16, 16), 2.0, tl.float16)
        c = tl.dot(a, b)
        d = tl.dot(c.to(tl.float16), a)
        row = tl.sum(d, axis=1)
        offs = tl.arange(0, 16)
        tl.store(out + offs, row.to(tl.float32))

    llir, meta = _compile_to_llir(kernel_chained_dot, {"out": "*fp32"})
    assert "target datalayout" in llir
    assert "shared" in meta


@pytest.mark.parametrize("input_precision", ["bf16x3", "bf16x6"])
def test_musa_056_bf16xN_dot_compile_only(input_precision):

    @triton.jit
    def kernel_bf16_dot(out, INPUT_PRECISION: tl.constexpr):
        a = tl.full((16, 16), 1.0, tl.float32)
        b = tl.full((16, 16), 2.0, tl.float32)
        c = tl.dot(a, b, input_precision=INPUT_PRECISION, out_dtype=tl.float32)
        row = tl.sum(c, axis=1)
        offs = tl.arange(0, 16)
        tl.store(out + offs, row)

    llir, _ = _compile_to_llir(
        kernel_bf16_dot,
        {"out": "*fp32", "INPUT_PRECISION": "constexpr"},
        constexprs={"INPUT_PRECISION": input_precision},
    )
    assert "target datalayout" in llir


def test_musa_056_functional_vecmat_compile_only():

    @triton.jit
    def kernel_vecmat(inp, out):
        offs = tl.arange(0, 16)
        vec = tl.load(inp + offs)
        mat = tl.full((16, 16), 0.5, tl.float32)
        prod = mat * tl.expand_dims(vec, 0)
        red = tl.sum(prod, axis=1)
        tl.store(out + offs, red)

    llir, _ = _compile_to_llir(kernel_vecmat, {"inp": "*fp32", "out": "*fp32"})
    assert "fadd" in llir
    assert "fmul" in llir


def test_musa_056_constexpr_annotation_compile_only():

    @triton.jit
    def kernel_constexpr(inp, out, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        x = tl.load(inp + offs)
        tl.store(out + offs, x)

    llir, _ = _compile_to_llir(kernel_constexpr, {"inp": "*fp32", "out": "*fp32", "BLOCK": "constexpr"},
                               constexprs={"BLOCK": 32})
    assert "target datalayout" in llir


@triton.jit
def _musa_int8_dot_kernel(a, b, out):
    offs_m = tl.arange(0, 32)
    offs_n = tl.arange(0, 32)
    offs_k = tl.arange(0, 64)
    lhs = tl.load(a + offs_m[:, None] * 64 + offs_k[None, :])
    rhs = tl.load(b + offs_k[:, None] * 32 + offs_n[None, :])
    acc = tl.dot(lhs, rhs)
    tl.store(out + offs_m[:, None] * 32 + offs_n[None, :], acc)


def test_musa_ph1_signed_int8_dot_lowers_to_sqmma():
    llir, _ = _compile_to_llir(_musa_int8_dot_kernel, {"a": "*i8", "b": "*i8", "out": "*i32"})
    assert "llvm.musa.sqmma.smma" in llir


def test_musa_ph1_uint8_dot_fails_closed():
    with pytest.raises(CompilationError, match="only int8 supported"):
        _compile_to_llir(_musa_int8_dot_kernel, {"a": "*u8", "b": "*u8", "out": "*i32"})


def test_musa_ph1_signed_int8_dot_device():
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("requires a MUSA device")

    torch.manual_seed(17)
    lhs = torch.randint(-4, 5, (32, 64), dtype=torch.int8, device="musa")
    rhs = torch.randint(-4, 5, (64, 32), dtype=torch.int8, device="musa")
    out = torch.empty((32, 32), dtype=torch.int32, device="musa")

    compiled = _musa_int8_dot_kernel[(1, )](lhs, rhs, out, num_warps=4)
    torch.musa.synchronize()

    expected = lhs.cpu().to(torch.int32) @ rhs.cpu().to(torch.int32)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
    assert "llvm.musa.sqmma.smma" in compiled.asm["llir"]


_STANDARD_FP8_DOT_FORMATS = {
    "fp8e4nv": (tl.float8e4nv, "e4m3", [0xC0, 0xB8, 0x00, 0x38, 0x40]),
    "fp8e5": (tl.float8e5, "e5m2", [0xC0, 0xBC, 0x00, 0x3C, 0x40]),
}


@triton.jit
def _musa_standard_fp8_dot_kernel(lhs, rhs, out):
    offsets_m = tl.arange(0, 32)
    offsets_n = tl.arange(0, 32)
    offsets_k = tl.arange(0, 32)
    a = tl.load(lhs + offsets_m[:, None] * 32 + offsets_k[None, :])
    b = tl.load(rhs + offsets_k[:, None] * 32 + offsets_n[None, :])
    acc = tl.zeros((32, 32), dtype=tl.float32)
    result = tl.dot(a, b, acc, out_dtype=tl.float32)
    tl.store(out + offsets_m[:, None] * 32 + offsets_n[None, :], result)


@pytest.mark.parametrize("format_name", _STANDARD_FP8_DOT_FORMATS)
def test_musa_ph1_standard_fp8_dot_lowers_to_sqmma(format_name):
    _, intrinsic_type, _ = _STANDARD_FP8_DOT_FORMATS[format_name]
    llir, _ = _compile_to_llir(
        _musa_standard_fp8_dot_kernel,
        {"lhs": f"*{format_name}", "rhs": f"*{format_name}", "out": "*fp32"},
    )

    assert f"llvm.musa.sqmma.{intrinsic_type}.m32n32k32.mma" in llir
    assert "llvm.musa.wmma" not in llir


@pytest.mark.parametrize("format_name", _STANDARD_FP8_DOT_FORMATS)
def test_musa_ph1_standard_fp8_dot_device(format_name):
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("requires a MUSA device")
    if tuple(torch.musa.get_device_capability(0)) != (3, 1):
        pytest.skip("requires a PH1 device")

    dtype, intrinsic_type, raw_codes = _STANDARD_FP8_DOT_FORMATS[format_name]
    codebook = torch.tensor(raw_codes, dtype=torch.uint8)
    valuebook = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32)
    lhs_index = ((torch.arange(32 * 32) * 7 + 3) % len(codebook)).reshape(32, 32)
    rhs_index = ((torch.arange(32 * 32) * 11 + 1) % len(codebook)).reshape(32, 32)
    lhs_raw = codebook[lhs_index].to("musa")
    rhs_raw = codebook[rhs_index].to("musa")
    output = torch.empty((32, 32), dtype=torch.float32, device="musa")

    compiled = _musa_standard_fp8_dot_kernel[(1, )](
        triton.reinterpret(lhs_raw, dtype),
        triton.reinterpret(rhs_raw, dtype),
        output,
        num_warps=4,
    )
    torch.musa.synchronize()

    expected = valuebook[lhs_index] @ valuebook[rhs_index]
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    assert f"llvm.musa.sqmma.{intrinsic_type}.m32n32k32.mma" in compiled.asm["llir"]
    assert compiled.asm["mubin"]


_CUSTOM_FP8_FORMATS = {
    "fp8e4b15": (tl.float8e4b15, 4, 3, 15, False),
    "fp8e4b8": (tl.float8e4b8, 4, 3, 8, True),
    "fp8e5b16": (tl.float8e5b16, 5, 2, 16, True),
}


@triton.jit
def _musa_custom_fp8_upcast_kernel(src, out, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    values = tl.load(src + offsets)
    tl.store(out + offsets, values.to(tl.float32))


@triton.jit
def _musa_custom_fp8_dot_kernel(lhs, rhs, out):
    offsets = tl.arange(0, 16)
    a = tl.load(lhs + offsets[:, None] * 16 + offsets[None, :])
    b = tl.load(rhs + offsets[:, None] * 16 + offsets[None, :])
    result = tl.dot(a, b)
    tl.store(out + offsets[:, None] * 16 + offsets[None, :], result)


def _decode_custom_fp8(raw, exponent_bits, mantissa_bits, exponent_bias, nan_on_negzero):
    if nan_on_negzero and raw == 0x80:
        return float("nan")

    sign = -1.0 if raw & 0x80 else 1.0
    exponent = (raw >> mantissa_bits) & ((1 << exponent_bits) - 1)
    mantissa = raw & ((1 << mantissa_bits) - 1)
    if exponent == 0:
        magnitude = mantissa * 2.0**(1 - exponent_bias - mantissa_bits)
    else:
        magnitude = (1.0 + mantissa / 2.0**mantissa_bits) * 2.0**(exponent - exponent_bias)
    return sign * magnitude


def test_musa_ph1_fp8_storage_and_dot_capabilities_are_separate():
    options = _get_musa_backend().parse_options({})
    compute = set(options.supported_fp8_dtypes)
    storage = set(options.supported_fp8_storage_dtypes)
    custom = set(options.custom_fp8_dtypes)

    assert compute == {"fp8e4nv", "fp8e5"}
    assert custom == set(_CUSTOM_FP8_FORMATS)
    assert storage == compute | custom


@pytest.mark.parametrize("format_name", _CUSTOM_FP8_FORMATS)
def test_musa_ph1_custom_fp8_storage_upcast_compiles(format_name):
    llir, _ = _compile_to_llir(
        _musa_custom_fp8_upcast_kernel,
        {"src": f"*{format_name}", "out": "*fp32", "BLOCK": "constexpr"},
        constexprs={"BLOCK": 16},
    )
    assert "fmul" in llir
    assert "llvm.musa.sqmma" not in llir


@pytest.mark.parametrize("format_name", _CUSTOM_FP8_FORMATS)
def test_musa_ph1_custom_fp8_dot_fails_closed(format_name):
    with pytest.raises(CompilationError, match="not supported in this architecture for dot"):
        _compile_to_llir(
            _musa_custom_fp8_dot_kernel,
            {"lhs": f"*{format_name}", "rhs": f"*{format_name}", "out": "*fp32"},
        )


@pytest.mark.parametrize("format_name", _CUSTOM_FP8_FORMATS)
def test_musa_ph1_custom_fp8_storage_upcast_device(format_name):
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("requires a MUSA device")

    dtype, exponent_bits, mantissa_bits, exponent_bias, nan_on_negzero = _CUSTOM_FP8_FORMATS[format_name]
    raw_values = [0x00, 0x01, 0x08, 0x38, 0x40, 0x7F, 0x80, 0x88, 0xB8, 0xC0, 0xFF]
    padded_values = raw_values + [0x00] * (16 - len(raw_values))
    raw = torch.tensor(padded_values, dtype=torch.uint8, device="musa")
    output = torch.empty(16, dtype=torch.float32, device="musa")

    compiled = _musa_custom_fp8_upcast_kernel[(1, )](
        triton.reinterpret(raw, dtype), output, BLOCK=16, num_warps=4)
    torch.musa.synchronize()

    expected = torch.tensor([
        _decode_custom_fp8(value, exponent_bits, mantissa_bits, exponent_bias, nan_on_negzero)
        for value in raw_values
    ])
    torch.testing.assert_close(output.cpu()[:len(raw_values)], expected, rtol=0, atol=0, equal_nan=True)
    assert "fmul" in compiled.asm["llir"]
    assert "llvm.musa.sqmma" not in compiled.asm["llir"]
