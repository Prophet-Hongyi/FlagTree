import pytest

from triton.backends.hcu.llvm17_mmac_compat import (
    LEGACY_FP16_MMAC,
    MAKE_BUFFER_RSRC,
    NEW_FP16_MMAC,
    PTR_BUFFER_LOAD_I16,
    PTR_BUFFER_LOAD_I32,
    PTR_BUFFER_LOAD_V2I32,
    PTR_BUFFER_STORE_F32,
    PTR_BUFFER_STORE_I8,
    PTR_BUFFER_STORE_I32,
    PTR_BUFFER_STORE_V2I32,
    PTR_BUFFER_STORE_V2F32,
    PTR_BUFFER_STORE_V4F32,
    RAW_BUFFER_LOAD_I16,
    RAW_BUFFER_LOAD_I32,
    RAW_BUFFER_LOAD_V2I32,
    RAW_BUFFER_STORE_F32,
    RAW_BUFFER_STORE_I8,
    RAW_BUFFER_STORE_I32,
    RAW_BUFFER_STORE_V2I32,
    RAW_BUFFER_STORE_V2F32,
    RAW_BUFFER_STORE_V4F32,
    LLVM17MmacBridgeError,
    bridge_gfx936_fp16_mmac_for_llvm17,
)


def _fp16_dot_module(
    lit="false", lts="false", stride=0, store_type="f32", load_type="i16"
):
    load_pointer = {
        "i16": PTR_BUFFER_LOAD_I16,
        "i32": PTR_BUFFER_LOAD_I32,
        "v2i32": PTR_BUFFER_LOAD_V2I32,
    }[load_type]
    load_ir_type = {"v2i32": "<2 x i32>"}.get(load_type, load_type)
    store_pointer = {
        "f32": PTR_BUFFER_STORE_F32,
        "i8": PTR_BUFFER_STORE_I8,
        "i32": PTR_BUFFER_STORE_I32,
        "v2i32": PTR_BUFFER_STORE_V2I32,
        "v2f32": PTR_BUFFER_STORE_V2F32,
        "v4f32": PTR_BUFFER_STORE_V4F32,
    }[store_type]
    store_raw_type = {
        "f32": "float",
        "i8": "i8",
        "i32": "i32",
        "v2i32": "<2 x i32>",
        "v2f32": "<2 x float>",
        "v4f32": "<4 x float>",
    }[store_type]
    store_value = {
        "f32": "0.0",
        "i8": "0",
        "i32": "0",
        "v2i32": "zeroinitializer",
        "v2f32": "zeroinitializer",
        "v4f32": "zeroinitializer",
    }[store_type]
    return f"""define amdgpu_kernel void @kernel(ptr addrspace(1) %input, ptr addrspace(1) %out, <4 x half> %a, <4 x half> %b) {{
  %input.rsrc = call ptr addrspace(8) @{MAKE_BUFFER_RSRC}(ptr addrspace(1) %input, i16 0, i32 1024, i32 159744)
  %packed = call {load_ir_type} @{load_pointer}(ptr addrspace(8) %input.rsrc, i32 0, i32 0, i32 0)
  %acc0 = call <4 x float> @{NEW_FP16_MMAC}(<4 x half> %a, <4 x half> %b, <4 x float> zeroinitializer, i1 {lit}, i1 {lts})
  %rsrc = call ptr addrspace(8) @{MAKE_BUFFER_RSRC}(ptr addrspace(1) %out, i16 {stride}, i32 1024, i32 159744)
  call void @{store_pointer}({store_raw_type} {store_value}, ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0)
  ret void
}}
declare <4 x float> @{NEW_FP16_MMAC}(<4 x half>, <4 x half>, <4 x float>, i1, i1)
declare ptr addrspace(8) @{MAKE_BUFFER_RSRC}(ptr addrspace(1), i16, i32, i32)
declare {load_ir_type} @{load_pointer}(ptr addrspace(8), i32, i32, i32)
declare void @{store_pointer}({store_raw_type}, ptr addrspace(8), i32, i32, i32)
"""


@pytest.mark.parametrize(
    ("store_type", "pointer_store", "raw_store"),
    [
        ("f32", PTR_BUFFER_STORE_F32, RAW_BUFFER_STORE_F32),
        ("i8", PTR_BUFFER_STORE_I8, RAW_BUFFER_STORE_I8),
        ("i32", PTR_BUFFER_STORE_I32, RAW_BUFFER_STORE_I32),
        ("v2i32", PTR_BUFFER_STORE_V2I32, RAW_BUFFER_STORE_V2I32),
        ("v2f32", PTR_BUFFER_STORE_V2F32, RAW_BUFFER_STORE_V2F32),
        ("v4f32", PTR_BUFFER_STORE_V4F32, RAW_BUFFER_STORE_V4F32),
    ],
)
def test_fp16_mmac_and_output_buffer_are_bridged(store_type, pointer_store, raw_store):
    bridged, stats = bridge_gfx936_fp16_mmac_for_llvm17(_fp16_dot_module(store_type=store_type))

    assert NEW_FP16_MMAC not in bridged
    assert LEGACY_FP16_MMAC in bridged
    assert MAKE_BUFFER_RSRC not in bridged
    assert PTR_BUFFER_LOAD_I16 not in bridged
    assert pointer_store not in bridged
    assert RAW_BUFFER_LOAD_I16 in bridged
    assert raw_store in bridged
    assert "ptr addrspace(8)" not in bridged
    assert stats.calls == 1
    assert stats.make_buffer_calls == 2
    assert stats.raw_buffer_load_calls == 1
    assert stats.raw_buffer_store_calls == 1


@pytest.mark.parametrize(
    ("load_type", "pointer_load", "raw_load"),
    [
        ("i16", PTR_BUFFER_LOAD_I16, RAW_BUFFER_LOAD_I16),
        ("i32", PTR_BUFFER_LOAD_I32, RAW_BUFFER_LOAD_I32),
        ("v2i32", PTR_BUFFER_LOAD_V2I32, RAW_BUFFER_LOAD_V2I32),
    ],
)
def test_fp16_mmac_fp8_input_buffer_loads_are_bridged(
    load_type, pointer_load, raw_load
):
    bridged, stats = bridge_gfx936_fp16_mmac_for_llvm17(
        _fp16_dot_module(load_type=load_type)
    )

    assert pointer_load not in bridged
    assert raw_load in bridged
    assert stats.raw_buffer_load_calls == 1


@pytest.mark.parametrize(("lit", "lts"), [("true", "false"), ("false", "true")])
def test_fp16_mmac_controls_fail_closed(lit, lts):
    with pytest.raises(LLVM17MmacBridgeError, match="only supports the legacy layout"):
        bridge_gfx936_fp16_mmac_for_llvm17(_fp16_dot_module(lit, lts))


@pytest.mark.parametrize(
    ("supported", "unsupported"),
    [
        (PTR_BUFFER_LOAD_I16, "llvm.amdgcn.raw.ptr.buffer.load.i8"),
        (PTR_BUFFER_STORE_F32, "llvm.amdgcn.raw.ptr.buffer.store.f16"),
    ],
)
def test_unknown_fp16_buffer_overload_fails_closed(supported, unsupported):
    source = _fp16_dot_module().replace(supported, unsupported)
    with pytest.raises(LLVM17MmacBridgeError, match="unsupported.*buffer overloads"):
        bridge_gfx936_fp16_mmac_for_llvm17(source)


def test_nonzero_fp16_buffer_stride_fails_closed():
    with pytest.raises(LLVM17MmacBridgeError, match="nonzero buffer stride"):
        bridge_gfx936_fp16_mmac_for_llvm17(_fp16_dot_module(stride=4))


def test_malformed_fp16_mmac_contract_fails_closed():
    call_prefix = f"call <4 x float> @{NEW_FP16_MMAC}(<4 x half> %a"
    source = _fp16_dot_module().replace(
        call_prefix,
        f"call <4 x float> @{NEW_FP16_MMAC}(<4 x i16> %a",
        1,
    )
    with pytest.raises(LLVM17MmacBridgeError, match="cannot parse FP16 MMAC contract"):
        bridge_gfx936_fp16_mmac_for_llvm17(source)
