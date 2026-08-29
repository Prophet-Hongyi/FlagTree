import pytest

from triton.backends.hcu.llvm17_mmac_compat import (
    LEGACY_INT8_MMAC,
    MAKE_BUFFER_RSRC,
    NEW_INT8_MMAC,
    PTR_BUFFER_LOAD_I8,
    PTR_BUFFER_LOAD_I16,
    PTR_BUFFER_LOAD_I32,
    PTR_BUFFER_LOAD_V2I32,
    PTR_BUFFER_LOAD_V4I32,
    PTR_BUFFER_STORE_I16,
    PTR_BUFFER_STORE_I8,
    PTR_BUFFER_STORE_I32,
    PTR_BUFFER_STORE_V2I32,
    PTR_BUFFER_STORE_V2F32,
    PTR_BUFFER_STORE_V4F32,
    PTR_BUFFER_STORE_V4I32,
    RAW_BUFFER_LOAD_I16,
    RAW_BUFFER_LOAD_I8,
    RAW_BUFFER_LOAD_I32,
    RAW_BUFFER_LOAD_V2I32,
    RAW_BUFFER_LOAD_V4I32,
    RAW_BUFFER_STORE_I16,
    RAW_BUFFER_STORE_I8,
    RAW_BUFFER_STORE_I32,
    RAW_BUFFER_STORE_V2I32,
    RAW_BUFFER_STORE_V2F32,
    RAW_BUFFER_STORE_V4F32,
    RAW_BUFFER_STORE_V4I32,
    LLVM17Int8MmacBridgeError,
    bridge_gfx936_int8_mmac_for_llvm17,
)


def _int8_mmac_module(
    lit="false", clamp="false", lts="false", accumulator="zeroinitializer"
):
    return (
        "define amdgpu_kernel void @kernel() {\n"
        "entry:\n"
        f"  %result = tail call <4 x i32> @{NEW_INT8_MMAC}"
        f"(<2 x i32> %lhs, <2 x i32> %rhs, <4 x i32> {accumulator}, "
        f"i1 {lit}, i1 {clamp}, i1 {lts})\n"
        "  ret void\n"
        "}\n\n"
        f"declare <4 x i32> @{NEW_INT8_MMAC}"
        "(<2 x i32>, <2 x i32>, <4 x i32>, i1 immarg, i1, i1) #0\n"
    )


def _int8_dot_module(
    stride=0,
    load_type="i16",
    load_ir_type=None,
    store_type="i32",
    store_ir_type=None,
):
    load_ir_type = load_ir_type or load_type
    store_ir_type = store_ir_type or store_type
    store_value = "zeroinitializer" if store_ir_type.startswith("<") else "0"
    return (
        "define amdgpu_kernel void @kernel(ptr addrspace(1) %input, "
        "ptr addrspace(1) %output) {\n"
        "entry:\n"
        f"  %input.rsrc = tail call ptr addrspace(8) @{MAKE_BUFFER_RSRC}"
        f"(ptr addrspace(1) %input, i16 {stride}, i32 2147483646, i32 159744)\n"
        f"  %value = tail call {load_ir_type} "
        f"@llvm.amdgcn.raw.ptr.buffer.load.{load_type}"
        f"(ptr addrspace(8) %input.rsrc, i32 0, i32 0, i32 0)\n"
        f"  %result = tail call <4 x i32> @{NEW_INT8_MMAC}"
        "(<2 x i32> %lhs, <2 x i32> %rhs, <4 x i32> zeroinitializer, "
        "i1 false, i1 false, i1 false)\n"
        f"  %output.rsrc = tail call ptr addrspace(8) @{MAKE_BUFFER_RSRC}"
        "(ptr addrspace(1) %output, i16 0, i32 2147483646, i32 159744)\n"
        f"  tail call void @llvm.amdgcn.raw.ptr.buffer.store.{store_type}"
        f"({store_ir_type} {store_value}, ptr addrspace(8) %output.rsrc, "
        "i32 0, i32 0, i32 0)\n"
        "  ret void\n"
        "}\n\n"
        f"declare ptr addrspace(8) @{MAKE_BUFFER_RSRC}"
        "(ptr addrspace(1) readnone, i16, i32, i32)\n"
        f"declare {load_ir_type} @llvm.amdgcn.raw.ptr.buffer.load.{load_type}"
        "(ptr addrspace(8) readonly nocapture, i32, i32, i32 immarg)\n"
        f"declare void @llvm.amdgcn.raw.ptr.buffer.store.{store_type}"
        f"({store_ir_type}, ptr addrspace(8) writeonly nocapture, "
        "i32, i32, i32 immarg)\n"
        f"declare <4 x i32> @{NEW_INT8_MMAC}"
        "(<2 x i32>, <2 x i32>, <4 x i32>, i1 immarg, i1, i1)\n"
    )


def test_gfx936_llvm17_int8_mmac_bridge_preserves_legacy_contract():
    bridged, stats = bridge_gfx936_int8_mmac_for_llvm17(_int8_mmac_module())

    assert stats.calls == 1
    assert NEW_INT8_MMAC not in bridged
    assert bridged.count(LEGACY_INT8_MMAC) == 2
    assert "%flagtree.mmac.result.lhs_i64 = bitcast <2 x i32> %lhs to i64" in bridged
    assert "%flagtree.mmac.result.rhs_i64 = bitcast <2 x i32> %rhs to i64" in bridged


@pytest.mark.parametrize(
    "accumulator",
    [
        "%acc",
        "zeroinitializer",
        "<i32 2147483632, i32 -2147483632, i32 0, i32 17>",
    ],
)
def test_gfx936_llvm17_int8_mmac_bridge_preserves_supported_accumulators(
    accumulator,
):
    bridged, stats = bridge_gfx936_int8_mmac_for_llvm17(
        _int8_mmac_module(accumulator=accumulator)
    )

    assert stats.calls == 1
    assert f"<4 x i32> {accumulator})" in bridged


def test_gfx936_llvm17_int8_mmac_bridge_rejects_unknown_accumulator():
    with pytest.raises(
        LLVM17Int8MmacBridgeError, match="cannot parse signed INT8 MMAC contract"
    ):
        bridge_gfx936_int8_mmac_for_llvm17(
            _int8_mmac_module(accumulator="poison")
        )


@pytest.mark.parametrize(
    ("load_type", "load_ir_type", "pointer_load", "raw_load"),
    [
        ("i8", "i8", PTR_BUFFER_LOAD_I8, RAW_BUFFER_LOAD_I8),
        ("i16", "i16", PTR_BUFFER_LOAD_I16, RAW_BUFFER_LOAD_I16),
        ("i32", "i32", PTR_BUFFER_LOAD_I32, RAW_BUFFER_LOAD_I32),
        ("v2i32", "<2 x i32>", PTR_BUFFER_LOAD_V2I32, RAW_BUFFER_LOAD_V2I32),
        ("v4i32", "<4 x i32>", PTR_BUFFER_LOAD_V4I32, RAW_BUFFER_LOAD_V4I32),
    ],
)
@pytest.mark.parametrize(
    ("store_type", "store_ir_type", "pointer_store", "raw_store"),
    [
        ("i16", "i16", PTR_BUFFER_STORE_I16, RAW_BUFFER_STORE_I16),
        ("i32", "i32", PTR_BUFFER_STORE_I32, RAW_BUFFER_STORE_I32),
        ("v2i32", "<2 x i32>", PTR_BUFFER_STORE_V2I32, RAW_BUFFER_STORE_V2I32),
        ("v2f32", "<2 x float>", PTR_BUFFER_STORE_V2F32, RAW_BUFFER_STORE_V2F32),
        ("v4f32", "<4 x float>", PTR_BUFFER_STORE_V4F32, RAW_BUFFER_STORE_V4F32),
        ("v4i32", "<4 x i32>", PTR_BUFFER_STORE_V4I32, RAW_BUFFER_STORE_V4I32),
        ("i8", "i8", PTR_BUFFER_STORE_I8, RAW_BUFFER_STORE_I8),
    ],
)
def test_gfx936_llvm17_int8_dot_bridge_rewrites_observed_buffer_contracts(
    load_type,
    load_ir_type,
    pointer_load,
    raw_load,
    store_type,
    store_ir_type,
    pointer_store,
    raw_store,
):
    bridged, stats = bridge_gfx936_int8_mmac_for_llvm17(
        _int8_dot_module(
            load_type=load_type,
            load_ir_type=load_ir_type,
            store_type=store_type,
            store_ir_type=store_ir_type,
        )
    )

    assert stats.calls == 1
    assert stats.make_buffer_calls == 2
    assert stats.raw_buffer_load_calls == 1
    assert stats.raw_buffer_store_calls == 1
    assert MAKE_BUFFER_RSRC not in bridged
    assert pointer_load not in bridged
    assert pointer_store not in bridged
    assert bridged.count(raw_load) == 2
    assert bridged.count(raw_store) == 2
    assert "ptr addrspace(8)" not in bridged
    assert "%input.rsrc = bitcast i128 %flagtree.rsrc.input.rsrc.packed to <4 x i32>" in bridged


@pytest.mark.parametrize(
    "lit, clamp, lts",
    [("true", "false", "false"), ("false", "true", "false"), ("false", "false", "true")],
)
def test_gfx936_llvm17_int8_mmac_bridge_rejects_nonlegacy_layout(lit, clamp, lts):
    with pytest.raises(LLVM17Int8MmacBridgeError, match="only supports the legacy layout"):
        bridge_gfx936_int8_mmac_for_llvm17(_int8_mmac_module(lit, clamp, lts))


def test_gfx936_llvm17_int8_mmac_bridge_rejects_unknown_contract():
    source = _int8_mmac_module().replace("<2 x i32> %lhs", "<2 x i32> zeroinitializer", 1)
    with pytest.raises(LLVM17Int8MmacBridgeError, match="cannot parse signed INT8 MMAC contract"):
        bridge_gfx936_int8_mmac_for_llvm17(source)


@pytest.mark.parametrize("load_type, store_type", [("i64", "i32"), ("i16", "i64")])
def test_gfx936_llvm17_int8_dot_bridge_rejects_unknown_buffer_overload(
    load_type, store_type
):
    with pytest.raises(LLVM17Int8MmacBridgeError, match="unsupported.*buffer overloads"):
        bridge_gfx936_int8_mmac_for_llvm17(
            _int8_dot_module(load_type=load_type, store_type=store_type)
        )


def test_gfx936_llvm17_int8_dot_bridge_rejects_nonzero_buffer_stride():
    with pytest.raises(LLVM17Int8MmacBridgeError, match="nonzero buffer stride"):
        bridge_gfx936_int8_mmac_for_llvm17(_int8_dot_module(stride=4))
