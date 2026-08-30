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
_FP8_TORCH_DTYPES = {
    "fp8e4nv": torch.float8_e4m3fn,
    "fp8e5": torch.float8_e5m2,
}
_FP8_DOWNCAST_CASES = (
    ("fp8e4nv", "llvm.mxc.cvt.pk4.f32tof8.cfg"),
    ("fp8e5", "llvm.mxc.cvt.pk4.f32tobf8.cfg"),
)
_MIXED_OCP_FP8_CASES = (
    (
        "fp8e4nv",
        "fp8e5",
        tl.float8e4nv,
        tl.float8e5,
        torch.tensor([0xC0, 0xB8, 0x00, 0x38, 0x40], dtype=torch.uint8),
        torch.tensor([0xC0, 0xBC, 0x00, 0x3C, 0x40], dtype=torch.uint8),
    ),
    (
        "fp8e5",
        "fp8e4nv",
        tl.float8e5,
        tl.float8e4nv,
        torch.tensor([0xC0, 0xBC, 0x00, 0x3C, 0x40], dtype=torch.uint8),
        torch.tensor([0xC0, 0xB8, 0x00, 0x38, 0x40], dtype=torch.uint8),
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


@triton.jit
def _metax_fp8_dot_fp8_output_kernel(lhs, rhs, out, OUTPUT_DTYPE: tl.constexpr):
    offsets_m = tl.arange(0, 16)
    offsets_n = tl.arange(0, 16)
    offsets_k = tl.arange(0, 32)
    a = tl.load(lhs + offsets_m[:, None] * 32 + offsets_k[None, :])
    b = tl.load(rhs + offsets_k[:, None] * 16 + offsets_n[None, :])
    acc = tl.zeros((16, 16), dtype=tl.float32)
    result = tl.dot(a, b, acc, out_dtype=tl.float32)
    result = result.to(OUTPUT_DTYPE, fp_downcast_rounding="rtne")
    tl.store(out + offsets_m[:, None] * 16 + offsets_n[None, :], result)


@triton.jit
def _metax_downcast_e4(src, dst):
    offsets = tl.arange(0, 128)
    values = tl.load(src + offsets)
    result = values.to(tl.float8e4nv, fp_downcast_rounding="rtne")
    tl.store(dst + offsets, result)


@triton.jit
def _metax_downcast_e5(src, dst):
    offsets = tl.arange(0, 128)
    values = tl.load(src + offsets)
    result = values.to(tl.float8e5, fp_downcast_rounding="rtne")
    tl.store(dst + offsets, result)


def _compile_c550(dtype_name, rhs_dtype_name=None):
    rhs_dtype_name = dtype_name if rhs_dtype_name is None else rhs_dtype_name
    src = ASTSource(
        fn=_metax_fp8_dot_kernel,
        signature={"lhs": f"*{dtype_name}", "rhs": f"*{rhs_dtype_name}", "out": "*fp32"},
    )
    return triton.compile(
        src,
        target=_C550_TARGET,
        options={"num_warps": 4, "num_stages": 1},
    )


def _compile_downcast(dtype_name, arch):
    kernel = _metax_downcast_e4 if dtype_name == "fp8e4nv" else _metax_downcast_e5
    src = ASTSource(
        fn=kernel,
        signature={"src": "*fp32", "dst": f"*{dtype_name}"},
    )
    return triton.compile(
        src,
        target=GPUTarget("maca", arch, 64),
        options={"num_warps": 4, "num_stages": 1},
    )


def _get_metax_backend_for_capability(capability):
    target = GPUTarget("maca", capability, 64)
    return backends["metax"].compiler(target)


@pytest.mark.parametrize(
    (
        "capability, architecture, ocp_conversion, custom_conversion, "
        "fp8_mma, fp4_conversion, int8_mma"
    ),
    [
        (80, "c550", "software", "software", "software", "software", "native"),
        (
            89,
            "unknown",
            "unsupported",
            "unsupported",
            "unsupported",
            "unsupported",
            "unsupported",
        ),
    ],
)
def test_metax_low_precision_target_features(
    capability,
    architecture,
    ocp_conversion,
    custom_conversion,
    fp8_mma,
    fp4_conversion,
    int8_mma,
):
    features = libtriton.metax.get_low_precision_target_features(capability)
    assert features["architecture"] == architecture
    assert features["ocp_fp8_conversion"] == ocp_conversion
    assert features["custom_fp8_conversion"] == custom_conversion
    assert features["fp8_mma"] == fp8_mma
    assert features["fp4_conversion"] == fp4_conversion
    assert features["signed_int8_mma"] == int8_mma


def test_unknown_metax_capability_fails_closed_for_fp8():
    backend = _get_metax_backend_for_capability(89)
    options = backend.parse_options({})
    assert options.supported_fp8_dtypes == ()
    assert not options.supports_batched_dot_scaled
    with pytest.raises(ValueError, match="not supported on MetaX capability 89"):
        backend.parse_options({"supported_fp8_dtypes": ("fp8e4nv",)})
    with pytest.raises(
        ValueError,
        match="batched dot_scaled is not supported on MetaX capability 89",
    ):
        backend.parse_options({"supports_batched_dot_scaled": True})


@pytest.mark.parametrize("dtype_name", [case[0] for case in _FP8_FORMATS])
def test_c550_fp8_dot_uses_software_upcast_and_fp16_mma(dtype_name):
    compiled = _compile_c550(dtype_name)
    assert compiled.asm["ttgir"].count("tt.fp_to_fp") == 2
    assert "llvm.mxc.mma.f32.16x16x16f16" in compiled.asm["llir"]
    assert "llvm.mxc.mma.f32.16x16x32f8" not in compiled.asm["llir"]
    assert "llvm.mxc.mma.f32.16x16x32bf8" not in compiled.asm["llir"]
    assert compiled.asm["mcfatbin"]


@pytest.mark.parametrize("dtype_name,hardware_intrinsic", _FP8_DOWNCAST_CASES)
def test_fp8_conversion_selector_is_target_local(dtype_name, hardware_intrinsic):
    c550 = _compile_downcast(dtype_name, 80)

    assert hardware_intrinsic not in c550.asm["llir"]
    assert c550.asm["mcfatbin"]


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


@pytest.mark.parametrize(
    (
        "lhs_dtype",
        "rhs_dtype",
        "_lhs_tl_dtype",
        "_rhs_tl_dtype",
        "_lhs_codes",
        "_rhs_codes",
    ),
    _MIXED_OCP_FP8_CASES,
)
def test_c550_mixed_ocp_fp8_dot_uses_software_upcast_and_fp16_mma(
    lhs_dtype, rhs_dtype, _lhs_tl_dtype, _rhs_tl_dtype, _lhs_codes, _rhs_codes
):
    compiled = _compile_c550(lhs_dtype, rhs_dtype)
    assert compiled.asm["ttgir"].count("tt.fp_to_fp") == 2
    assert "llvm.mxc.mma.f32.16x16x16f16" in compiled.asm["llir"]
    assert "llvm.mxc.mma.f32.16x16x32f8" not in compiled.asm["llir"]
    assert "llvm.mxc.mma.f32.16x16x32bf8" not in compiled.asm["llir"]
    assert compiled.asm["mcfatbin"]


@pytest.mark.parametrize(
    (
        "_lhs_dtype",
        "_rhs_dtype",
        "lhs_tl_dtype",
        "rhs_tl_dtype",
        "lhs_codes",
        "rhs_codes",
    ),
    _MIXED_OCP_FP8_CASES,
)
def test_c550_mixed_ocp_fp8_dot_device(
    _lhs_dtype, _rhs_dtype, lhs_tl_dtype, rhs_tl_dtype, lhs_codes, rhs_codes
):
    if not torch.cuda.is_available():
        pytest.skip("requires a MetaX device")

    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "maca" or target.arch != 80:
        pytest.skip("requires a C550 / maca arch 80 target")

    values = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32)
    lhs_index = ((torch.arange(16 * 32) * 7 + 3) % len(lhs_codes)).reshape(16, 32)
    rhs_index = ((torch.arange(32 * 16) * 11 + 1) % len(rhs_codes)).reshape(32, 16)
    lhs_raw = lhs_codes[lhs_index].to("cuda")
    rhs_raw = rhs_codes[rhs_index].to("cuda")
    output = torch.empty((16, 16), dtype=torch.float32, device="cuda")

    compiled = _metax_fp8_dot_kernel[(1, )](
        triton.reinterpret(lhs_raw, lhs_tl_dtype),
        triton.reinterpret(rhs_raw, rhs_tl_dtype),
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


@pytest.mark.parametrize("dtype_name,triton_dtype,codes", _FP8_FORMATS)
def test_c550_fp8_dot_fp8_output_device(dtype_name, triton_dtype, codes):
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
    output_raw = torch.empty((16, 16), dtype=torch.uint8, device="cuda")

    compiled = _metax_fp8_dot_fp8_output_kernel[(1, )](
        triton.reinterpret(lhs_raw, triton_dtype),
        triton.reinterpret(rhs_raw, triton_dtype),
        triton.reinterpret(output_raw, triton_dtype),
        OUTPUT_DTYPE=triton_dtype,
        num_warps=4,
        num_stages=1,
    )
    torch.cuda.synchronize()

    expected = values[lhs_index] @ values[rhs_index]
    expected_raw = expected.to(_FP8_TORCH_DTYPES[dtype_name]).view(torch.uint8)
    actual_raw = output_raw.cpu()
    raw_mismatch = actual_raw != expected_raw
    both_are_zero = ((actual_raw & 0x7F) == 0) & ((expected_raw & 0x7F) == 0)
    assert not torch.any(raw_mismatch & ~both_are_zero)
    assert compiled.asm["ttgir"].count("tt.fp_to_fp") == 3
    assert "llvm.mxc.mma.f32.16x16x16f16" in compiled.asm["llir"]
    assert "llvm.mxc.mma.f32.16x16x32f8" not in compiled.asm["llir"]
    assert "llvm.mxc.mma.f32.16x16x32bf8" not in compiled.asm["llir"]
    for _, hardware_intrinsic in _FP8_DOWNCAST_CASES:
        assert hardware_intrinsic not in compiled.asm["llir"]
