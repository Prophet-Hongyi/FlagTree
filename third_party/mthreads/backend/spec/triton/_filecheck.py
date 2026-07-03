import functools
import os
import inspect
import shutil
import subprocess
import tempfile

import triton
from triton.backends import backends
from triton.compiler import ASTSource, make_backend
from triton.backends.compiler import GPUTarget
from triton.experimental.gluon._runtime import GluonASTSource
from triton.runtime.jit import create_function_from_signature
from triton._C.libtriton import ir

# ===-----------------------------------------------------------------------===#
# filecheck_test
# ===-----------------------------------------------------------------------===#


def _get_stub_target() -> GPUTarget:
    backend_name = os.environ.get("TRITON_DEFAULT_BACKEND")
    if backend_name is None and len(backends) == 1:
        backend_name = next(iter(backends))

    if backend_name in ("nvidia", "cuda"):
        return GPUTarget("cuda", 100, 32)
    if backend_name in ("amd", "hip"):
        return GPUTarget("hip", "gfx942", 64)
    if backend_name in ("mthreads", "musa"):
        arch = os.environ.get("TRITON_OVERRIDE_ARCH") or os.environ.get("TRITON_MUSA_ARCH") or "ph1"
        return GPUTarget("musa", arch, 32)

    # Preserve the legacy frontend parser target when no backend is selected.
    return GPUTarget("cuda", 100, 32)


_triton_dir = os.path.dirname(__file__)
_filecheck_local = os.path.join(_triton_dir, "FileCheck")
# CMake copies FileCheck to ${TRITON_WHEEL_DIR}/FileCheck (the triton package root).
# This module lives at triton/backends/mthreads/spec/triton/_filecheck.py,
# so triton/ is 4 levels up from _triton_dir.
_triton_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_triton_dir))))
_filecheck_wheel = os.path.join(_triton_root, "FileCheck")
_filecheck_system = shutil.which("FileCheck")
# Search order: same directory, triton package root, system PATH
_filecheck_path = None
for _candidate in (_filecheck_local, _filecheck_wheel, _filecheck_system):
    if _candidate and os.path.isfile(_candidate):
        _filecheck_path = _candidate
        break

_MISSING_FILECHECK_MSG = ("FileCheck binary not found.  Install it with your package manager\n"
                          "  (e.g. apt-get install llvm-15-tools) or place it next to this module:\n"
                          f"  {_filecheck_local}")


def _get_filecheck_path():
    """Return the path to the FileCheck binary, or raise FileNotFoundError."""
    if _filecheck_path is None:
        raise FileNotFoundError(_MISSING_FILECHECK_MSG)
    return _filecheck_path


class MatchError(ValueError):

    def __init__(self, message, module_str):
        super().__init__(message)
        self.module_str = module_str

    def __str__(self):
        return f"{super().__str__()}\n{self.module_str}"


def run_filecheck(name, module_str, check_template):
    with tempfile.TemporaryDirectory() as tempdir:
        temp_module = os.path.join(tempdir, "module")
        with open(temp_module, "w") as temp:
            temp.write(module_str)

        temp_expected = os.path.join(tempdir, "expected")
        with open(temp_expected, "w") as temp:
            temp.write(check_template)

        try:
            subprocess.check_output(
                [_get_filecheck_path(), temp_expected, "--input-file", temp_module, "--dump-input-context=50"],
                stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as error:
            decoded = error.output.decode('unicode_escape')
            raise ValueError(decoded)


def run_parser(kernel_fn, args=(), kwargs=None, target=None):
    if kwargs is None:
        kwargs = {}
    if target is None:
        target = _get_stub_target()
    if "sanitize_overflow" not in kwargs:
        kwargs = dict(kwargs)
        kwargs["sanitize_overflow"] = False
    backend = make_backend(target)
    binder = create_function_from_signature(
        kernel_fn.signature,
        kernel_fn.params,
        backend,
    )

    bound_args, specialization, options = binder(*args, **kwargs)
    options, signature, constexprs, attrs = kernel_fn._pack_args(backend, kwargs, bound_args, specialization, options)
    source_cls = GluonASTSource if kernel_fn.is_gluon() else ASTSource
    src = source_cls(kernel_fn, signature, constexprs, attrs)

    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)

    codegen_fns = backend.get_codegen_implementation(options)
    module_map = backend.get_module_map()
    module = src.make_ir(target, options, codegen_fns, module_map, context)
    return module


def run_filecheck_test(kernel_fn):
    assert isinstance(kernel_fn, triton.runtime.JITFunction)
    check_template = inspect.getsource(kernel_fn.fn)
    if check_template is None:
        raise ValueError("kernel function must have a docstring with FileCheck template")
    mlir_module = run_parser(kernel_fn)

    run_filecheck("placeholder", mlir_module.str_nodebug(), check_template)


def filecheck_test(fn):

    @functools.wraps(fn)
    def test_fn():
        run_filecheck_test(fn)

    return test_fn
