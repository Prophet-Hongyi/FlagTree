import pytest
import torch
import triton
import triton.language as tl
from triton import knobs
from triton._internal_testing import is_ppu
from triton.backends.compiler import GPUTarget
from triton.backends.ppu.compiler import PPUBackend

_PPU0010_TARGET = GPUTarget("cuda", 80, 32)
_NATIVE_FP8_COMPAT_TARGET = GPUTarget("cuda", 90, 32)
_SIGNATURE = {"src": "*fp32", "dst": "*fp8e4nv", "BLOCK": "constexpr"}
_CONSTANTS = {"BLOCK": 64}


@triton.jit
def _fp_to_fp8e4m3(src, dst, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    values = tl.load(src + offsets)
    tl.store(dst + offsets, values.to(tl.float8e4nv))


@triton.jit
def _fp_to_fp8e4m3_rtz(src, dst, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    values = tl.load(src + offsets)
    tl.store(dst + offsets, values.to(tl.float8e4nv, fp_downcast_rounding="rtz"))


@triton.jit
def _fp16_to_fp8e5(src, dst, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    values = tl.load(src + offsets)
    tl.store(dst + offsets, values.to(tl.float8e5))


@triton.jit
def _fp8e4m3_to_fp32(src, dst, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    values = tl.load(src + offsets)
    tl.store(dst + offsets, values.to(tl.float32))


def _compile_through_llir(kernel, signature, target=_PPU0010_TARGET):
    previous_hook = knobs.runtime.add_stages_inspection_hook

    def stop_before_hgbin(_backend, stages, _options, _language, _capability):
        stages["hgbin"] = lambda _src, _metadata: b""

    knobs.runtime.add_stages_inspection_hook = stop_before_hgbin
    try:
        src = triton.compiler.ASTSource(fn=kernel, signature=signature, constexprs=_CONSTANTS)
        return triton.compile(src, target=target)
    finally:
        knobs.runtime.add_stages_inspection_hook = previous_hook


@pytest.mark.parametrize(
    "capability, expected",
    [
        (80, True),
        (88, False),
        (89, True),
        (90, True),
    ],
)
def test_ppu_advertises_fp8e4m3_by_capability(capability, expected):
    target = GPUTarget("cuda", capability, 32)
    options = PPUBackend(target).parse_options({})
    assert ("fp8e4nv" in options.supported_fp8_dtypes) is expected


def test_ppu_rejects_fp8e4m3_override_on_unsupported_capability():
    target = GPUTarget("cuda", 88, 32)
    with pytest.raises(ValueError, match="only supported on PPU capability 80 or >= 89"):
        PPUBackend(target).parse_options({"supported_fp8_dtypes": ("fp8e4nv", )})


def test_ppu0010_lowers_fp8e4m3_conversions_to_llir():
    downcast = _compile_through_llir(_fp_to_fp8e4m3, _SIGNATURE)
    assert "f8E4M3FN" in downcast.asm["ttir"]
    assert "e4m3x2" not in downcast.asm["llir"]

    upcast_signature = {"src": "*fp8e4nv", "dst": "*fp32", "BLOCK": "constexpr"}
    upcast = _compile_through_llir(_fp8e4m3_to_fp32, upcast_signature)
    assert "f8E4M3FN" in upcast.asm["ttir"]
    assert "e4m3x2" not in upcast.asm["llir"]


@pytest.mark.parametrize("capability", [89, 90])
def test_newer_ppu_capability_keeps_native_fp8e4m3_lowering(capability):
    target = GPUTarget("cuda", capability, 32)
    compiled = _compile_through_llir(_fp_to_fp8e4m3, _SIGNATURE, target=target)
    assert "f8E4M3FN" in compiled.asm["ttir"]
    assert "e4m3x2" in compiled.asm["llir"]


def test_ppu_conversion_selection_is_scoped_to_target():
    signature = {"src": "*fp16", "dst": "*fp8e5", "BLOCK": "constexpr"}
    software = _compile_through_llir(_fp16_to_fp8e5, signature, target=_PPU0010_TARGET)
    native = _compile_through_llir(_fp16_to_fp8e5, signature, target=_NATIVE_FP8_COMPAT_TARGET)

    assert "e5m2x2" not in software.asm["llir"]
    assert "e5m2x2" in native.asm["llir"]


def test_ppu0010_rejects_fp8e4m3_rtz(capfd):
    with pytest.raises(RuntimeError, match="PassManager::run failed"):
        _compile_through_llir(_fp_to_fp8e4m3_rtz, _SIGNATURE)
    assert "only FP32/FP16/BF16 RTNE encode" in capfd.readouterr().err


def test_issue_975_fp32_to_fp8e4m3fn_store(device):
    if not is_ppu():
        pytest.skip("requires the PPU backend")

    values = torch.tensor(
        [
            -448.0,
            -256.0,
            -16.0,
            -1.0,
            -0.001953125,
            -0.0,
            0.0,
            0.001953125,
            0.125,
            0.5,
            1.0,
            2.0,
            16.0,
            32.0,
            256.0,
            448.0,
        ],
        dtype=torch.float32,
    )
    expected = values.to(torch.float8_e4m3fn).view(torch.uint8)
    device_values = values.to(device)
    output = torch.empty(device_values.shape, dtype=torch.float8_e4m3fn, device=device)

    _fp_to_fp8e4m3[(1, )](device_values, output, BLOCK=device_values.numel())

    torch.testing.assert_close(output.cpu().view(torch.uint8), expected, rtol=0, atol=0)


def test_ppu0010_fp8e4m3_round_trip(device):
    if not is_ppu():
        pytest.skip("requires the PPU backend")

    values = torch.tensor(
        [-16.0, -8.0, -4.0, -2.0, -1.0, -0.5, -0.25, -0.125, 0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
        dtype=torch.float32,
        device=device,
    ).repeat(4)
    fp8_storage = torch.empty(values.shape, dtype=torch.uint8, device=device)
    restored = torch.empty_like(values)

    _fp_to_fp8e4m3[(1, )](
        values,
        triton.reinterpret(fp8_storage, tl.float8e4nv),
        BLOCK=values.numel(),
    )
    _fp8e4m3_to_fp32[(1, )](
        triton.reinterpret(fp8_storage, tl.float8e4nv),
        restored,
        BLOCK=values.numel(),
    )

    torch.testing.assert_close(restored, values, rtol=0, atol=0)


@pytest.mark.parametrize("src_dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_ppu0010_fp_to_fp8e4m3_rtne(device, src_dtype):
    if not is_ppu():
        pytest.skip("requires the PPU backend")

    values = torch.tensor(
        [
            -float("inf"),
            -465.0,
            -449.0,
            -448.0,
            -256.0,
            -16.0,
            -1.1875,
            -1.0625,
            -1.0,
            -0.015625,
            -0.0146484375,
            -0.0029296875,
            -0.001953125,
            -0.0009765625,
            -0.0,
            0.0,
            0.0009765625,
            0.001953125,
            0.0029296875,
            0.0146484375,
            0.015625,
            1.0,
            1.0625,
            1.1875,
            16.0,
            256.0,
            448.0,
            449.0,
            465.0,
            float("inf"),
            float("nan"),
            2.0,
        ],
        dtype=src_dtype,
    )
    expected = values.float().clamp(min=-448.0, max=448.0).to(torch.float8_e4m3fn).view(torch.uint8)
    device_values = values.to(device)
    fp8_storage = torch.empty(device_values.shape, dtype=torch.uint8, device=device)

    _fp_to_fp8e4m3[(1, )](
        device_values,
        triton.reinterpret(fp8_storage, tl.float8e4nv),
        BLOCK=device_values.numel(),
    )

    torch.testing.assert_close(fp8_storage.cpu(), expected, rtol=0, atol=0)
