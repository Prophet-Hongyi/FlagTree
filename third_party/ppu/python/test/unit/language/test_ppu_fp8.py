import pytest
import torch
import triton
import triton.language as tl
from triton import knobs
from triton._internal_testing import is_ppu
from triton.backends.compiler import GPUTarget
from triton.backends.ppu.compiler import PPUBackend
from triton.compiler.errors import CompilationError

_PPU0010_TARGET = GPUTarget("cuda", 80, 32)
_NATIVE_FP8_COMPAT_TARGET = GPUTarget("cuda", 90, 32)
_SIGNATURE = {"src": "*fp32", "dst": "*fp8e4nv", "BLOCK": "constexpr"}
_CONSTANTS = {"BLOCK": 64}
_PPU0010_FP16_MMA = "ppu.mma.sync.aligned.m16n16k16.row.col.f32.f16.f16.f32"
_PPU_NATIVE_E4M3_MMA = ".e4m3.e4m3."
_PPU_NATIVE_E4M3_CVT = "ppu.cvt.rn.satfinite.e4m3x2.f32"
_PPU_NATIVE_E5M2_MMA = ".e5m2.e5m2."
_PPU_NATIVE_E5M2_CVT = "ppu.cvt.rn.satfinite.e5m2x2.f16x2"


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
def _fp16_to_fp8e5_grid(src, dst, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    values = tl.load(src + offsets, mask=mask)
    tl.store(dst + offsets, values.to(tl.float8e5), mask=mask)


@triton.jit
def _fp8e4m3_to_fp32(src, dst, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    values = tl.load(src + offsets)
    tl.store(dst + offsets, values.to(tl.float32))


@triton.jit
def _fp8e4m3_dot(lhs, rhs, out, OUTPUT_FP8: tl.constexpr):
    offsets_m = tl.arange(0, 16)
    offsets_n = tl.arange(0, 16)
    offsets_k = tl.arange(0, 32)
    a = tl.load(lhs + offsets_m[:, None] * 32 + offsets_k[None, :])
    b = tl.load(rhs + offsets_k[:, None] * 16 + offsets_n[None, :])
    acc = tl.zeros((16, 16), dtype=tl.float32)
    result = tl.dot(a, b, acc, out_dtype=tl.float32)
    if OUTPUT_FP8:
        result = result.to(tl.float8e4nv)
    tl.store(out + offsets_m[:, None] * 16 + offsets_n[None, :], result)


@triton.jit
def _fp8e5m2_dot(lhs, rhs, out, OUTPUT_FP8: tl.constexpr):
    offsets_m = tl.arange(0, 16)
    offsets_n = tl.arange(0, 16)
    offsets_k = tl.arange(0, 32)
    a = tl.load(lhs + offsets_m[:, None] * 32 + offsets_k[None, :])
    b = tl.load(rhs + offsets_k[:, None] * 16 + offsets_n[None, :])
    acc = tl.zeros((16, 16), dtype=tl.float32)
    result = tl.dot(a, b, acc, out_dtype=tl.float32)
    if OUTPUT_FP8:
        result = result.to(tl.float8e5)
    tl.store(out + offsets_m[:, None] * 16 + offsets_n[None, :], result)


@triton.jit
def _fp8e4m3_dot_illegal_k(lhs, rhs, out):
    offsets_m = tl.arange(0, 16)
    offsets_n = tl.arange(0, 16)
    offsets_k = tl.arange(0, 16)
    a = tl.load(lhs + offsets_m[:, None] * 16 + offsets_k[None, :])
    b = tl.load(rhs + offsets_k[:, None] * 16 + offsets_n[None, :])
    acc = tl.zeros((16, 16), dtype=tl.float32)
    result = tl.dot(a, b, acc, out_dtype=tl.float32)
    tl.store(out + offsets_m[:, None] * 16 + offsets_n[None, :], result)


def _compile_through_llir(kernel, signature, target=_PPU0010_TARGET, constexprs=None):
    previous_hook = knobs.runtime.add_stages_inspection_hook

    def stop_before_hgbin(_backend, stages, _options, _language, _capability):
        stages["hgbin"] = lambda _src, _metadata: b""

    knobs.runtime.add_stages_inspection_hook = stop_before_hgbin
    try:
        src = triton.compiler.ASTSource(
            fn=kernel,
            signature=signature,
            constexprs=_CONSTANTS if constexprs is None else constexprs,
        )
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


@pytest.mark.parametrize("output_fp8", [False, True])
def test_ppu0010_fp8e4m3_dot_lowers_through_fp16_mma(output_fp8):
    output_type = "fp8e4nv" if output_fp8 else "fp32"
    compiled = _compile_through_llir(
        _fp8e4m3_dot,
        {
            "lhs": "*fp8e4nv",
            "rhs": "*fp8e4nv",
            "out": f"*{output_type}",
            "OUTPUT_FP8": "constexpr",
        },
        constexprs={"OUTPUT_FP8": output_fp8},
    )

    assert compiled.asm["ttgir"].count("tt.fp_to_fp") == 2 + int(output_fp8)
    assert compiled.asm["llir"].count(_PPU0010_FP16_MMA) == 2
    assert _PPU_NATIVE_E4M3_MMA not in compiled.asm["llir"]
    assert _PPU_NATIVE_E4M3_CVT not in compiled.asm["llir"]


def test_ppu0010_fp8e4m3_dot_rejects_k_below_32():
    with pytest.raises(CompilationError, match="K >= 32"):
        _compile_through_llir(
            _fp8e4m3_dot_illegal_k,
            {"lhs": "*fp8e4nv", "rhs": "*fp8e4nv", "out": "*fp32"},
            constexprs={},
        )


@pytest.mark.parametrize("output_fp8", [False, True])
def test_ppu0010_fp8e4m3_dot_device(device, output_fp8):
    if not is_ppu():
        pytest.skip("requires the PPU backend")

    codes = torch.tensor([0xC0, 0xB8, 0x00, 0x38, 0x40], dtype=torch.uint8)
    values = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32)
    lhs_index = ((torch.arange(16 * 32) * 7 + 3) % len(codes)).reshape(16, 32)
    rhs_index = ((torch.arange(32 * 16) * 11 + 1) % len(codes)).reshape(32, 16)
    lhs_raw = codes[lhs_index].to(device)
    rhs_raw = codes[rhs_index].to(device)
    expected = values[lhs_index] @ values[rhs_index]
    output = torch.empty(
        (16, 16),
        dtype=torch.uint8 if output_fp8 else torch.float32,
        device=device,
    )

    output_arg = triton.reinterpret(output, tl.float8e4nv) if output_fp8 else output
    compiled = _fp8e4m3_dot[(1, )](
        triton.reinterpret(lhs_raw, tl.float8e4nv),
        triton.reinterpret(rhs_raw, tl.float8e4nv),
        output_arg,
        OUTPUT_FP8=output_fp8,
        num_warps=1,
        num_stages=1,
    )

    if output_fp8:
        expected = expected.to(torch.float8_e4m3fn).view(torch.uint8)
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    assert compiled.asm["llir"].count(_PPU0010_FP16_MMA) == 2
    assert _PPU_NATIVE_E4M3_MMA not in compiled.asm["llir"]
    assert _PPU_NATIVE_E4M3_CVT not in compiled.asm["llir"]
    assert compiled.asm["hgbin"]


@pytest.mark.parametrize("output_fp8", [False, True])
def test_ppu0010_fp8e5m2_dot_lowers_through_fp16_mma(output_fp8):
    output_type = "fp8e5" if output_fp8 else "fp32"
    compiled = _compile_through_llir(
        _fp8e5m2_dot,
        {
            "lhs": "*fp8e5",
            "rhs": "*fp8e5",
            "out": f"*{output_type}",
            "OUTPUT_FP8": "constexpr",
        },
        constexprs={"OUTPUT_FP8": output_fp8},
    )

    assert compiled.asm["ttgir"].count("tt.fp_to_fp") == 2 + int(output_fp8)
    assert compiled.asm["llir"].count(_PPU0010_FP16_MMA) == 2
    assert _PPU_NATIVE_E5M2_MMA not in compiled.asm["llir"]
    assert _PPU_NATIVE_E5M2_CVT not in compiled.asm["llir"]


@pytest.mark.parametrize("output_fp8", [False, True])
def test_ppu0010_fp8e5m2_dot_device(device, output_fp8):
    if not is_ppu():
        pytest.skip("requires the PPU backend")

    codes = torch.tensor([0xC0, 0xBC, 0x00, 0x3C, 0x40], dtype=torch.uint8)
    values = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32)
    lhs_index = ((torch.arange(16 * 32) * 7 + 3) % len(codes)).reshape(16, 32)
    rhs_index = ((torch.arange(32 * 16) * 11 + 1) % len(codes)).reshape(32, 16)
    lhs_raw = codes[lhs_index].to(device)
    rhs_raw = codes[rhs_index].to(device)
    expected = values[lhs_index] @ values[rhs_index]
    output = torch.empty(
        (16, 16),
        dtype=torch.uint8 if output_fp8 else torch.float32,
        device=device,
    )

    output_arg = triton.reinterpret(output, tl.float8e5) if output_fp8 else output
    compiled = _fp8e5m2_dot[(1, )](
        triton.reinterpret(lhs_raw, tl.float8e5),
        triton.reinterpret(rhs_raw, tl.float8e5),
        output_arg,
        OUTPUT_FP8=output_fp8,
        num_warps=1,
        num_stages=1,
    )

    if output_fp8:
        expected = expected.to(torch.float8_e5m2).view(torch.uint8)
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    assert compiled.asm["llir"].count(_PPU0010_FP16_MMA) == 2
    assert _PPU_NATIVE_E5M2_MMA not in compiled.asm["llir"]
    assert _PPU_NATIVE_E5M2_CVT not in compiled.asm["llir"]
    assert compiled.asm["hgbin"]


def test_ppu0010_fp16_to_fp8e5m2_rtne_exhaustive(device):
    if not is_ppu():
        pytest.skip("requires the PPU backend")

    raw = torch.arange(65536, dtype=torch.int32).to(torch.uint16)
    values = raw.view(torch.float16)
    expected = values.to(torch.float8_e5m2).view(torch.uint8)
    expected = torch.where(
        (expected & 0x7f) == 0x7c,
        (expected & 0x80) | 0x7b,
        expected,
    )
    output = torch.empty(values.shape, dtype=torch.uint8, device=device)

    block = 256
    compiled = _fp16_to_fp8e5_grid[(triton.cdiv(values.numel(), block), )](
        values.to(device),
        triton.reinterpret(output, tl.float8e5),
        N=values.numel(),
        BLOCK=block,
        num_warps=4,
        num_stages=1,
    )

    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    assert _PPU_NATIVE_E5M2_CVT not in compiled.asm["llir"]
    assert compiled.asm["hgbin"]


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
