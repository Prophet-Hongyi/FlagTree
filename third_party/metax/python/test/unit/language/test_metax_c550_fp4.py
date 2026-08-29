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
    pytest.skip("metax backend not built in libtriton", allow_module_level=True)

if "metax" not in backends:
    pytest.skip("metax backend not discovered", allow_module_level=True)


_C550_TARGET = GPUTarget("maca", 80, 64)


@triton.jit
def _metax_e2m1_dot_scaled_kernel(lhs, rhs, out):
    offsets_m = tl.arange(0, 32)
    offsets_n = tl.arange(0, 32)
    offsets_k_packed = tl.arange(0, 32)
    a = tl.load(lhs + offsets_m[:, None] * 32 + offsets_k_packed[None, :])
    b = tl.load(rhs + offsets_k_packed[:, None] * 32 + offsets_n[None, :])
    acc = tl.zeros((32, 32), dtype=tl.float32)
    result = tl.dot_scaled(
        a,
        None,
        "e2m1",
        b,
        None,
        "e2m1",
        acc,
        lhs_k_pack=True,
        rhs_k_pack=True,
    )
    tl.store(out + offsets_m[:, None] * 32 + offsets_n[None, :], result)


@triton.jit
def _metax_scaled_e2m1_dot_kernel(lhs, lhs_scale, rhs, rhs_scale, out):
    offsets_m = tl.arange(0, 32)
    offsets_n = tl.arange(0, 32)
    offsets_k_packed = tl.arange(0, 32)
    offsets_scale = tl.arange(0, 2)
    a = tl.load(lhs + offsets_m[:, None] * 32 + offsets_k_packed[None, :])
    b = tl.load(rhs + offsets_k_packed[:, None] * 32 + offsets_n[None, :])
    a_scale = tl.load(lhs_scale + offsets_m[:, None] * 2 + offsets_scale[None, :])
    b_scale = tl.load(rhs_scale + offsets_n[:, None] * 2 + offsets_scale[None, :])
    acc = tl.zeros((32, 32), dtype=tl.float32)
    result = tl.dot_scaled(
        a,
        a_scale,
        "e2m1",
        b,
        b_scale,
        "e2m1",
        acc,
        lhs_k_pack=True,
        rhs_k_pack=True,
    )
    tl.store(out + offsets_m[:, None] * 32 + offsets_n[None, :], result)


def _compile_no_scale():
    source = ASTSource(
        fn=_metax_e2m1_dot_scaled_kernel,
        signature={"lhs": "*u8", "rhs": "*u8", "out": "*fp32"},
    )
    return triton.compile(source, target=_C550_TARGET, options={"num_warps": 4})


def _pack_along_k(raw_codes, dim):
    if dim == 1:
        low = raw_codes[:, 0::2]
        high = raw_codes[:, 1::2]
    else:
        low = raw_codes[0::2, :]
        high = raw_codes[1::2, :]
    return low | (high << 4)


def _assert_software_e2m1_dot(compiled):
    assert compiled.asm["ttgir"].count("fp4_to_fp") == 2
    assert "llvm.mxc.mma.f32.16x16x2bf16" in compiled.asm["llir"]
    assert "llvm.mxc.cvt.pk4.f4tobf16.scale" not in compiled.asm["llir"]


def test_c550_e2m1_dot_scaled_lowers_through_software_decode_and_bf16_mma():
    compiled = _compile_no_scale()
    _assert_software_e2m1_dot(compiled)
    assert compiled.asm["mcfatbin"]


def test_c550_scaled_e2m1_dot_remains_fail_closed(capfd):
    source = ASTSource(
        fn=_metax_scaled_e2m1_dot_kernel,
        signature={
            "lhs": "*u8",
            "lhs_scale": "*u8",
            "rhs": "*u8",
            "rhs_scale": "*u8",
            "out": "*fp32",
        },
    )
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        triton.compile(source, target=_C550_TARGET, options={"num_warps": 4})
    assert "failed to legalize operation 'tt.dot_scaled'" in capfd.readouterr().err


def test_c550_e2m1_dot_scaled_device():
    if not torch.cuda.is_available():
        pytest.skip("requires a MetaX device")
    if torch.cuda.get_device_capability(0) != (8, 0):
        pytest.skip("requires a C550 / maca arch 80 target")

    codebook = torch.tensor([0xC, 0xA, 0x0, 0x2, 0x4], dtype=torch.uint8)
    valuebook = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32)
    lhs_index = ((torch.arange(32 * 64) * 7 + 3) % len(codebook)).reshape(32, 64)
    rhs_index = ((torch.arange(64 * 32) * 11 + 1) % len(codebook)).reshape(64, 32)
    lhs_packed = _pack_along_k(codebook[lhs_index], dim=1).cuda()
    rhs_packed = _pack_along_k(codebook[rhs_index], dim=0).cuda()
    output = torch.empty((32, 32), dtype=torch.float32, device="cuda")

    compiled = _metax_e2m1_dot_scaled_kernel[(1,)](
        lhs_packed,
        rhs_packed,
        output,
        num_warps=4,
    )
    torch.cuda.synchronize()

    expected = valuebook[lhs_index] @ valuebook[rhs_index]
    # C550's BF16 MMA can leave sub-micro-unit FP32 residuals even when every
    # decoded operand is an exact integer.  Keep a strict absolute-only bound
    # so zeros are covered without hiding a format or packing error.
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=2e-6)
    _assert_software_e2m1_dot(compiled)
    assert compiled.asm["mcfatbin"]
