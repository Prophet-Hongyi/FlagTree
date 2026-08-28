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


if not hasattr(libtriton, "mthreads"):
    pytest.skip("musa backend not built in libtriton", allow_module_level=True)

if "mthreads" not in backends:
    pytest.skip("musa backend not discovered", allow_module_level=True)


_PH1_TARGET = GPUTarget("musa", "ph1", 32)


@triton.jit
def _musa_e2m1_dot_scaled_kernel(lhs, rhs, out):
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


def _compile_ph1_e2m1_dot():
    source = ASTSource(
        fn=_musa_e2m1_dot_scaled_kernel,
        signature={"lhs": "*u8", "rhs": "*u8", "out": "*fp32"},
    )
    return triton.compile(source, target=_PH1_TARGET, options={"num_warps": 4})


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
    mma_lines = [
        line
        for line in compiled.asm["llir"].splitlines()
        if "llvm.musa." in line and ".mma" in line
    ]
    assert mma_lines
    assert any("llvm.musa.sqmma.bfmma.m32n32k64.mma" in line for line in mma_lines)
    assert all(".e2m1." not in line for line in mma_lines)


def test_musa_ph1_e2m1_dot_scaled_lowers_through_bf16_mma():
    compiled = _compile_ph1_e2m1_dot()
    _assert_software_e2m1_dot(compiled)
    assert compiled.asm["mubin"]


def test_musa_ph1_e2m1_dot_scaled_device():
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("requires a MUSA device")
    if tuple(torch.musa.get_device_capability(0)) != (3, 1):
        pytest.skip("requires a PH1 device")

    codebook = torch.tensor([0xC, 0xA, 0x0, 0x2, 0x4], dtype=torch.uint8)
    valuebook = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32)
    lhs_index = ((torch.arange(32 * 64) * 7 + 3) % len(codebook)).reshape(32, 64)
    rhs_index = ((torch.arange(64 * 32) * 11 + 1) % len(codebook)).reshape(64, 32)
    lhs_packed = _pack_along_k(codebook[lhs_index], dim=1).to("musa")
    rhs_packed = _pack_along_k(codebook[rhs_index], dim=0).to("musa")
    output = torch.empty((32, 32), dtype=torch.float32, device="musa")

    compiled = _musa_e2m1_dot_scaled_kernel[(1, )](
        lhs_packed,
        rhs_packed,
        output,
        num_warps=4,
    )
    torch.musa.synchronize()

    expected = valuebook[lhs_index] @ valuebook[rhs_index]
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    _assert_software_e2m1_dot(compiled)
    assert compiled.asm["mubin"]
