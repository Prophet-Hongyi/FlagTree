import pytest

from triton.backends.hcu.llvm17_mmac_compat import (
    MAKE_BUFFER_RSRC,
    PTR_BUFFER_LOAD_F32,
    PTR_BUFFER_LOAD_I8,
    PTR_BUFFER_STORE_F32,
    PTR_BUFFER_STORE_I8,
    RAW_BUFFER_LOAD_F32,
    RAW_BUFFER_LOAD_I8,
    RAW_BUFFER_STORE_F32,
    RAW_BUFFER_STORE_I8,
    LLVM17MmacBridgeError,
    bridge_gfx936_buffer_contracts_for_llvm17,
)


def _scalar_buffer_module(load_type="f32", store_type="i8", stride=0):
    load_pointer = {
        "f32": PTR_BUFFER_LOAD_F32,
        "i8": PTR_BUFFER_LOAD_I8,
    }.get(load_type, f"llvm.amdgcn.raw.ptr.buffer.load.{load_type}")
    store_pointer = {
        "f32": PTR_BUFFER_STORE_F32,
        "i8": PTR_BUFFER_STORE_I8,
    }.get(store_type, f"llvm.amdgcn.raw.ptr.buffer.store.{store_type}")
    load_result_type = "float" if load_type == "f32" else load_type
    store_value_type = "float" if store_type == "f32" else store_type
    store_value = "0.0" if store_type == "f32" else "0"
    return f"""define amdgpu_kernel void @kernel(ptr addrspace(1) %input, ptr addrspace(1) %output) {{
  %input.rsrc = tail call ptr addrspace(8) @{MAKE_BUFFER_RSRC}(ptr addrspace(1) %input, i16 {stride}, i32 1024, i32 159744)
  %value = tail call {load_result_type} @{load_pointer}(ptr addrspace(8) %input.rsrc, i32 0, i32 0, i32 0)
  %output.rsrc = tail call ptr addrspace(8) @{MAKE_BUFFER_RSRC}(ptr addrspace(1) %output, i16 0, i32 1024, i32 159744)
  tail call void @{store_pointer}({store_value_type} {store_value}, ptr addrspace(8) %output.rsrc, i32 0, i32 0, i32 0)
  ret void
}}
declare ptr addrspace(8) @{MAKE_BUFFER_RSRC}(ptr addrspace(1), i16, i32, i32)
declare {load_result_type} @{load_pointer}(ptr addrspace(8), i32, i32, i32)
declare void @{store_pointer}({store_value_type}, ptr addrspace(8), i32, i32, i32)
"""


@pytest.mark.parametrize(
    ("load_type", "store_type", "raw_load", "raw_store"),
    [
        ("f32", "i8", RAW_BUFFER_LOAD_F32, RAW_BUFFER_STORE_I8),
        ("i8", "f32", RAW_BUFFER_LOAD_I8, RAW_BUFFER_STORE_F32),
    ],
)
def test_scalar_buffer_contracts_are_bridged(
    load_type, store_type, raw_load, raw_store
):
    bridged, stats = bridge_gfx936_buffer_contracts_for_llvm17(
        _scalar_buffer_module(load_type, store_type)
    )

    assert MAKE_BUFFER_RSRC not in bridged
    assert "llvm.amdgcn.raw.ptr.buffer" not in bridged
    assert "ptr addrspace(8)" not in bridged
    assert raw_load in bridged
    assert raw_store in bridged
    assert stats.make_buffer_calls == 2
    assert stats.raw_buffer_load_calls == 1
    assert stats.raw_buffer_store_calls == 1


def test_scalar_buffer_bridge_rejects_unobserved_overload():
    with pytest.raises(LLVM17MmacBridgeError, match="unsupported.*buffer overloads"):
        bridge_gfx936_buffer_contracts_for_llvm17(
            _scalar_buffer_module("i16", "f32")
        )


def test_scalar_buffer_bridge_rejects_nonzero_stride():
    with pytest.raises(LLVM17MmacBridgeError, match="nonzero buffer stride"):
        bridge_gfx936_buffer_contracts_for_llvm17(
            _scalar_buffer_module("f32", "i8", stride=1)
        )
