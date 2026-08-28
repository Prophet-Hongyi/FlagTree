import os

os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")

import pytest
import torch
import triton
import triton.language as tl
from triton._C import libtriton
from triton.backends import backends
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


if not hasattr(libtriton, "metax"):
    pytest.skip("MetaX backend not built in libtriton", allow_module_level=True)

if "metax" not in backends:
    pytest.skip("MetaX backend not discovered", allow_module_level=True)


_C550_TARGET = GPUTarget("maca", 80, 64)
_FP8_FORMATS = (
    (
        "fp8e4nv",
        tl.float8e4nv,
        torch.tensor([0xC0, 0xB8, 0x00, 0x38, 0x40], dtype=torch.uint8),
    ),
    (
        "fp8e5",
        tl.float8e5,
        torch.tensor([0xC0, 0xBC, 0x00, 0x3C, 0x40], dtype=torch.uint8),
    ),
)


@triton.jit
def _metax_fp8_dot_kernel(lhs, rhs, out):
    offsets_m = tl.arange(0, 16)
    offsets_n = tl.arange(0, 16)
    offsets_k = tl.arange(0, 32)
    a = tl.load(lhs + offsets_m[:, None] * 32 + offsets_k[None, :])
    b = tl.load(rhs + offsets_k[:, None] * 16 + offsets_n[None, :])
    acc = tl.zeros((16, 16), dtype=tl.float32)
    result = tl.dot(a, b, acc, out_dtype=tl.float32)
    tl.store(out + offsets_m[:, None] * 16 + offsets_n[None, :], result)


def _compile_c550(dtype_name):
    src = ASTSource(
        fn=_metax_fp8_dot_kernel,
        signature={"lhs": f"*{dtype_name}", "rhs": f"*{dtype_name}", "out": "*fp32"},
    )
    return triton.compile(
        src,
        target=_C550_TARGET,
        options={"num_warps": 4, "num_stages": 1},
    )


@pytest.mark.parametrize("dtype_name", [case[0] for case in _FP8_FORMATS])
def test_c550_fp8_dot_uses_software_upcast_and_fp16_mma(dtype_name):
    compiled = _compile_c550(dtype_name)
    assert compiled.asm["ttgir"].count("tt.fp_to_fp") == 2
    assert "llvm.mxc.mma.f32.16x16x16f16" in compiled.asm["llir"]
    assert "llvm.mxc.mma.f32.16x16x32f8" not in compiled.asm["llir"]
    assert "llvm.mxc.mma.f32.16x16x32bf8" not in compiled.asm["llir"]
    assert compiled.asm["mcfatbin"]


@pytest.mark.parametrize("dtype_name,triton_dtype,codes", _FP8_FORMATS)
def test_c550_fp8_dot_device(dtype_name, triton_dtype, codes):
    del dtype_name
    if not torch.cuda.is_available():
        pytest.skip("requires a MetaX device")

    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "maca" or target.arch != 80:
        pytest.skip("requires a C550 / maca arch 80 target")

    values = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32)
    lhs_index = ((torch.arange(16 * 32) * 7 + 3) % len(codes)).reshape(16, 32)
    rhs_index = ((torch.arange(32 * 16) * 11 + 1) % len(codes)).reshape(32, 16)
    lhs_raw = codes[lhs_index].to("cuda")
    rhs_raw = codes[rhs_index].to("cuda")
    output = torch.empty((16, 16), dtype=torch.float32, device="cuda")

    compiled = _metax_fp8_dot_kernel[(1, )](
        triton.reinterpret(lhs_raw, triton_dtype),
        triton.reinterpret(rhs_raw, triton_dtype),
        output,
        num_warps=4,
        num_stages=1,
    )
    torch.cuda.synchronize()

    expected = values[lhs_index] @ values[rhs_index]
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=1.0e-5)
    assert compiled.asm["ttgir"].count("tt.fp_to_fp") == 2
    assert "llvm.mxc.mma.f32.16x16x16f16" in compiled.asm["llir"]
    assert "llvm.mxc.mma.f32.16x16x32f8" not in compiled.asm["llir"]
    assert "llvm.mxc.mma.f32.16x16x32bf8" not in compiled.asm["llir"]
