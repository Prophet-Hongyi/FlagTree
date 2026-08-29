import pytest
import torch
import triton
import triton.language as tl
from triton import knobs
from triton._internal_testing import is_ppu
from triton.backends.compiler import GPUTarget


_PPU0010_TARGET = GPUTarget("cuda", 80, 32)
_PPU0010_BF16_MMA = "ppu.mma.sync.aligned.m16n16k16.row.col.f32.bf16.bf16.f32"


@triton.jit
def _e2m1_dot_scaled(lhs, rhs, out):
    offsets_m = tl.arange(0, 16)
    offsets_n = tl.arange(0, 16)
    offsets_k_packed = tl.arange(0, 16)
    a = tl.load(lhs + offsets_m[:, None] * 16 + offsets_k_packed[None, :])
    b = tl.load(rhs + offsets_k_packed[:, None] * 16 + offsets_n[None, :])
    acc = tl.zeros((16, 16), dtype=tl.float32)
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
    tl.store(out + offsets_m[:, None] * 16 + offsets_n[None, :], result)


@triton.jit
def _e2m1_lhs_scaled(lhs, lhs_scale, rhs, out):
    offsets_m = tl.arange(0, 16)
    offsets_n = tl.arange(0, 16)
    offsets_k_packed = tl.arange(0, 16)
    a = tl.load(lhs + offsets_m[:, None] * 16 + offsets_k_packed[None, :])
    a_scale = tl.load(lhs_scale + offsets_m[:, None])
    b = tl.load(rhs + offsets_k_packed[:, None] * 16 + offsets_n[None, :])
    acc = tl.zeros((16, 16), dtype=tl.float32)
    result = tl.dot_scaled(
        a,
        a_scale,
        "e2m1",
        b,
        None,
        "e2m1",
        acc,
        lhs_k_pack=True,
        rhs_k_pack=True,
    )
    tl.store(out + offsets_m[:, None] * 16 + offsets_n[None, :], result)


@triton.jit
def _e2m1_rhs_scaled(lhs, rhs, rhs_scale, out):
    offsets_m = tl.arange(0, 16)
    offsets_n = tl.arange(0, 16)
    offsets_k_packed = tl.arange(0, 16)
    a = tl.load(lhs + offsets_m[:, None] * 16 + offsets_k_packed[None, :])
    b = tl.load(rhs + offsets_k_packed[:, None] * 16 + offsets_n[None, :])
    b_scale = tl.load(rhs_scale + offsets_n[:, None])
    acc = tl.zeros((16, 16), dtype=tl.float32)
    result = tl.dot_scaled(
        a,
        None,
        "e2m1",
        b,
        b_scale,
        "e2m1",
        acc,
        lhs_k_pack=True,
        rhs_k_pack=True,
    )
    tl.store(out + offsets_m[:, None] * 16 + offsets_n[None, :], result)


@triton.jit
def _e2m1_both_scaled(lhs, lhs_scale, rhs, rhs_scale, out):
    offsets_m = tl.arange(0, 16)
    offsets_n = tl.arange(0, 16)
    offsets_k_packed = tl.arange(0, 16)
    a = tl.load(lhs + offsets_m[:, None] * 16 + offsets_k_packed[None, :])
    a_scale = tl.load(lhs_scale + offsets_m[:, None])
    b = tl.load(rhs + offsets_k_packed[:, None] * 16 + offsets_n[None, :])
    b_scale = tl.load(rhs_scale + offsets_n[:, None])
    acc = tl.zeros((16, 16), dtype=tl.float32)
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
    tl.store(out + offsets_m[:, None] * 16 + offsets_n[None, :], result)


def _compile_through_llir(kernel, signature):
    previous_hook = knobs.runtime.add_stages_inspection_hook

    def stop_before_hgbin(_backend, stages, _options, _language, _capability):
        stages["hgbin"] = lambda _src, _metadata: b""

    knobs.runtime.add_stages_inspection_hook = stop_before_hgbin
    try:
        source = triton.compiler.ASTSource(fn=kernel, signature=signature)
        return triton.compile(source, target=_PPU0010_TARGET)
    finally:
        knobs.runtime.add_stages_inspection_hook = previous_hook


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
    assert compiled.asm["llir"].count(_PPU0010_BF16_MMA) == 2
    mma_lines = [line for line in compiled.asm["llir"].splitlines() if "ppu.mma" in line]
    assert mma_lines
    assert all(".e2m1." not in line for line in mma_lines)
    assert "ppu.prmt.b32" in compiled.asm["llir"]


def _compile_scaled(side):
    if side == "lhs":
        kernel = _e2m1_lhs_scaled
        signature = {"lhs": "*u8", "lhs_scale": "*u8", "rhs": "*u8", "out": "*fp32"}
    elif side == "rhs":
        kernel = _e2m1_rhs_scaled
        signature = {"lhs": "*u8", "rhs": "*u8", "rhs_scale": "*u8", "out": "*fp32"}
    else:
        assert side == "both"
        kernel = _e2m1_both_scaled
        signature = {
            "lhs": "*u8",
            "lhs_scale": "*u8",
            "rhs": "*u8",
            "rhs_scale": "*u8",
            "out": "*fp32",
        }
    return _compile_through_llir(kernel, signature)


def test_ppu0010_e2m1_dot_scaled_lowers_through_bf16_mma():
    compiled = _compile_through_llir(
        _e2m1_dot_scaled,
        {"lhs": "*u8", "rhs": "*u8", "out": "*fp32"},
    )
    _assert_software_e2m1_dot(compiled)


@pytest.mark.parametrize("side", ["lhs", "rhs", "both"])
def test_ppu0010_scaled_e2m1_dot_lowers_through_bf16_mma(side):
    _assert_software_e2m1_dot(_compile_scaled(side))


def test_ppu0010_e2m1_dot_scaled_device(device):
    if not is_ppu():
        pytest.skip("requires the PPU backend")

    codebook = torch.tensor([0xC, 0xA, 0x0, 0x2, 0x4], dtype=torch.uint8)
    valuebook = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32)
    lhs_index = ((torch.arange(16 * 32) * 7 + 3) % len(codebook)).reshape(16, 32)
    rhs_index = ((torch.arange(32 * 16) * 11 + 1) % len(codebook)).reshape(32, 16)
    lhs_packed = _pack_along_k(codebook[lhs_index], dim=1).to(device)
    rhs_packed = _pack_along_k(codebook[rhs_index], dim=0).to(device)
    output = torch.empty((16, 16), dtype=torch.float32, device=device)

    compiled = _e2m1_dot_scaled[(1, )](
        lhs_packed,
        rhs_packed,
        output,
        num_warps=1,
        num_stages=1,
    )

    expected = valuebook[lhs_index] @ valuebook[rhs_index]
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    _assert_software_e2m1_dot(compiled)
    assert compiled.asm["hgbin"]


@pytest.mark.parametrize("side", ["lhs", "rhs", "both"])
def test_ppu0010_scaled_e2m1_dot_device(device, side):
    if not is_ppu():
        pytest.skip("requires the PPU backend")

    codebook = torch.arange(8, dtype=torch.uint8)
    valuebook = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
    rows = torch.arange(16)[:, None]
    cols = torch.arange(16)[None, :]
    lhs_k = torch.arange(32)[None, :]
    rhs_k = torch.arange(32)[:, None]
    lhs_index = (rows * 3 + lhs_k * 5 + lhs_k // 7) % 8
    rhs_index = (rhs_k * 3 + cols * 5 + rhs_k // 11) % 8
    lhs_values = valuebook[lhs_index]
    rhs_values = valuebook[rhs_index]
    lhs_packed = _pack_along_k(codebook[lhs_index], dim=1).to(device)
    rhs_packed = _pack_along_k(codebook[rhs_index], dim=0).to(device)
    lhs_scale_codes = torch.tensor([0x7E + (row % 4) for row in range(16)], dtype=torch.uint8)
    rhs_scale_codes = torch.tensor([0x7E + ((3 * col + 1) % 4) for col in range(16)], dtype=torch.uint8)
    lhs_scale_values = torch.pow(2.0, lhs_scale_codes.to(torch.int32) - 127).to(torch.float32)[:, None]
    rhs_scale_values = torch.pow(2.0, rhs_scale_codes.to(torch.int32) - 127).to(torch.float32)[None, :]
    output = torch.empty((16, 16), dtype=torch.float32, device=device)

    if side == "lhs":
        compiled = _e2m1_lhs_scaled[(1, )](
            lhs_packed,
            lhs_scale_codes.to(device),
            rhs_packed,
            output,
            num_warps=1,
            num_stages=1,
        )
        expected = (lhs_values * lhs_scale_values) @ rhs_values
    elif side == "rhs":
        compiled = _e2m1_rhs_scaled[(1, )](
            lhs_packed,
            rhs_packed,
            rhs_scale_codes.to(device),
            output,
            num_warps=1,
            num_stages=1,
        )
        expected = lhs_values @ (rhs_values * rhs_scale_values)
    else:
        compiled = _e2m1_both_scaled[(1, )](
            lhs_packed,
            lhs_scale_codes.to(device),
            rhs_packed,
            rhs_scale_codes.to(device),
            output,
            num_warps=1,
            num_stages=1,
        )
        expected = (lhs_values * lhs_scale_values) @ (rhs_values * rhs_scale_values)

    actual = output.cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(actual, lhs_values @ rhs_values)
    _assert_software_e2m1_dot(compiled)
    assert compiled.asm["hgbin"]
