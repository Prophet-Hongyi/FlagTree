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
