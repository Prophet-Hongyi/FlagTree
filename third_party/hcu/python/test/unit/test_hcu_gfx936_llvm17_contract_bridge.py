"""Focused tests for the fail-closed gfx936 DTK17 LLIR contract bridge."""

from __future__ import annotations

import inspect

import pytest

from triton.backends.hcu import compiler_hcu
from triton.backends.hcu.llvm17_compat import (
    LLVM17ContractBridgeError,
    LLVM17ContractBridgeStats,
    bridge_gfx936_llvm17_contract,
)


def test_hcu_make_llir_defines_daz_control_constant() -> None:
    make_llir_source = inspect.getsource(compiler_hcu.HIPBackend.make_llir)
    assert (
        'hcu.set_bool_control_constant(llvm_mod, "__oclc_daz_opt", '
        "options.allow_flush_denorm)"
    ) in make_llir_source


def _module(
    *,
    stride: int = 0,
    lit: bool = False,
    lts: bool = False,
    base_call_attrs: str = "",
    resource_call_attrs: str = "",
) -> str:
    return f"""define amdgpu_kernel void @bridge_test(ptr addrspace(1) %base) {{
entry:
  %desc = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) {base_call_attrs}%base, i16 {stride}, i32 2147483646, i32 159744)
  %value = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) {resource_call_attrs}%desc, i32 0, i32 0, i32 0)
  %mma = tail call <4 x float> @llvm.hcu.mmac.f32.16x16x16.f16(<4 x half> zeroinitializer, <4 x half> zeroinitializer, <4 x float> zeroinitializer, i1 {str(lit).lower()}, i1 {str(lts).lower()}), !dbg !1
  tail call void @llvm.amdgcn.raw.ptr.buffer.store.v4i32(<4 x i32> %value, ptr addrspace(8) {resource_call_attrs}%desc, i32 0, i32 0, i32 0)
  ret void
}}

declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1), i16, i32, i32) #1
declare <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly nocapture, i32, i32, i32 immarg) #1
declare <4 x float> @llvm.hcu.mmac.f32.16x16x16.f16(<4 x half>, <4 x half>, <4 x float>, i1, i1) #1
declare void @llvm.amdgcn.raw.ptr.buffer.store.v4i32(<4 x i32>, ptr addrspace(8) writeonly nocapture, i32, i32, i32 immarg) #1
"""


def _bf16_module(*, lit: bool = False, lts: bool = False) -> str:
    return (
        _module(lit=lit, lts=lts)
        .replace("llvm.hcu.mmac.f32.16x16x16.f16", "llvm.hcu.mmac.f32.16x16x16.bf16")
        .replace("<4 x half>", "<4 x i16>")
    )


def test_bridge_rewrites_supported_legacy_contract() -> None:
    bridged, stats = bridge_gfx936_llvm17_contract(_module())
    assert stats == LLVM17ContractBridgeStats(
        make_buffer_calls=1,
        raw_buffer_load_calls=1,
        raw_buffer_store_calls=1,
        mmac_calls=1,
        mmac_f16_calls=1,
    )
    assert "llvm.amdgcn.mmac.f32.16x16x16f16" in bridged
    assert "llvm.amdgcn.raw.buffer.load.v4i32" in bridged
    assert "llvm.amdgcn.raw.buffer.store.v4i32" in bridged
    assert "bitcast i128" in bridged
    assert "ptr addrspace(8)" not in bridged
    assert "llvm.hcu.mmac" not in bridged
    assert "raw.ptr.buffer" not in bridged


def test_bridge_rewrites_observed_nonnull_resource_base() -> None:
    bridged, stats = bridge_gfx936_llvm17_contract(
        _module(base_call_attrs="nonnull ", resource_call_attrs="nonnull ")
    )
    assert stats.make_buffer_calls == 1
    assert "ptrtoint ptr addrspace(1) %base to i64" in bridged
    assert "llvm.amdgcn.make.buffer.rsrc" not in bridged
    assert "<4 x i32> nonnull" not in bridged


def test_bridge_rejects_unreviewed_resource_base_attribute() -> None:
    with pytest.raises(LLVM17ContractBridgeError, match="unbridged LLVM17 contract"):
        bridge_gfx936_llvm17_contract(_module(base_call_attrs="noundef "))


def test_bridge_rewrites_supported_bf16_mmac_contract() -> None:
    bridged, stats = bridge_gfx936_llvm17_contract(_bf16_module())
    assert stats.mmac_calls == 1
    assert stats.mmac_f16_calls == 0
    assert stats.mmac_bf16_calls == 1
    assert "llvm.amdgcn.mmac.f32.16x16x16bf16" in bridged
    assert "llvm.hcu.mmac" not in bridged


def test_bridge_rejects_bf16_layout_flags_not_representable_by_dtk17() -> None:
    with pytest.raises(LLVM17ContractBridgeError, match="cannot represent enabled LIT/LTS"):
        bridge_gfx936_llvm17_contract(_bf16_module(lit=True))


def test_bridge_rejects_unknown_mmac_variant() -> None:
    source = _module().replace(
        "llvm.hcu.mmac.f32.16x16x16.f16",
        "llvm.hcu.mmac.f32.16x16x16.i8",
    )
    with pytest.raises(LLVM17ContractBridgeError, match="unbridged LLVM17 contract"):
        bridge_gfx936_llvm17_contract(source)


def test_assembly_validator_distinguishes_f16_and_bf16_mmac() -> None:
    assembly = "\n".join(
        [
            "v_mmac_f32_16x16x16_f16 v[0:3], v[4:5], v[6:7]",
            "v_mmac_f32_16x16x16_bf16 v[8:11], v[12:13], v[14:15]",
            "v_mmac_f32_16x16x16_bf16 v[16:19], v[20:21], v[22:23]",
        ]
    )
    compiler_hcu._validate_gfx936_llvm17_bridge_assembly(
        assembly,
        LLVM17ContractBridgeStats(
            mmac_calls=3,
            mmac_f16_calls=1,
            mmac_bf16_calls=2,
        ),
    )


def test_assembly_validator_rejects_missing_bf16_mmac() -> None:
    with pytest.raises(compiler_hcu.HSACOError, match="MMAC lowering mismatch"):
        compiler_hcu._validate_gfx936_llvm17_bridge_assembly(
            "v_mmac_f32_16x16x16_f16 v[0:3], v[4:5], v[6:7]",
            LLVM17ContractBridgeStats(
                mmac_calls=2,
                mmac_f16_calls=1,
                mmac_bf16_calls=1,
            ),
        )


def test_bridge_accepts_multiple_declared_load_overloads() -> None:
    source = _module().replace(
        "  %value = tail call <4 x i32>",
        "  %scalar = tail call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %desc, i32 0, i32 0, i32 0)\n"
        "  %value = tail call <4 x i32>",
    ).replace(
        "declare <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32",
        "declare i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) readonly nocapture, i32, i32, i32 immarg) #1\n"
        "declare <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32",
    )
    bridged, stats = bridge_gfx936_llvm17_contract(source)
    assert stats.raw_buffer_load_calls == 2
    assert "llvm.amdgcn.raw.buffer.load.i32" in bridged
    assert "llvm.amdgcn.raw.buffer.load.v4i32" in bridged


def test_bridge_deduplicates_materialized_i64_vector_load_declaration() -> None:
    source = """define amdgpu_kernel void @bridge_test(ptr addrspace(1) %base) {
entry:
  %desc = tail call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %base, i16 0, i32 2147483646, i32 159744)
  %wide = tail call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %desc, i32 0, i32 0, i32 0)
  %indices = tail call <2 x i64> @llvm.amdgcn.raw.ptr.buffer.load.v2i64(ptr addrspace(8) %desc, i32 16, i32 0, i32 0)
  ret void
}

declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1), i16, i32, i32) #1
declare <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) readonly nocapture, i32, i32, i32 immarg) #1
declare <2 x i64> @llvm.amdgcn.raw.ptr.buffer.load.v2i64(ptr addrspace(8) readonly nocapture, i32, i32, i32 immarg) #1
"""
    bridged, stats = bridge_gfx936_llvm17_contract(
        source, materialize_i64_vector_loads=True
    )
    declaration = (
        "declare <4 x i32> @llvm.amdgcn.raw.buffer.load.v4i32"
        "(<4 x i32>, i32, i32, i32 immarg) #1"
    )
    assert bridged.count(declaration) == 1
    assert stats.i64_vector_load_materializations == 1
    assert stats.duplicate_declarations_suppressed == 1


@pytest.mark.parametrize("lit,lts", [(True, False), (False, True), (True, True)])
def test_bridge_rejects_unrepresentable_layout_flags(lit: bool, lts: bool) -> None:
    with pytest.raises(LLVM17ContractBridgeError, match="cannot represent enabled LIT/LTS"):
        bridge_gfx936_llvm17_contract(_module(lit=lit, lts=lts))


def test_bridge_rejects_nonzero_resource_stride() -> None:
    with pytest.raises(LLVM17ContractBridgeError, match="nonzero pointer-buffer stride"):
        bridge_gfx936_llvm17_contract(_module(stride=16))


def test_bridge_is_noop_without_legacy_contracts() -> None:
    source = "define void @plain() {\n  ret void\n}\n"
    bridged, stats = bridge_gfx936_llvm17_contract(source)
    assert bridged == source
    assert stats == LLVM17ContractBridgeStats()
