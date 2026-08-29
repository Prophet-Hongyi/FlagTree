"""RLC knobs must change HCU compile keys in the same process.

MThreads FlagGems 4→4 was a harness false negative when site `hash()`
omitted RLC and `@functools.lru_cache()` froze the first env. HCU follows that
cache-key contract: live enhance/mask/HCU atomic-writeback policy and gfx936
materialization gates, no MUSA policy ints.
Production pipeline is compiler_hcu; hoist stays a separate pass.
"""

from __future__ import annotations

import os

import pytest

from triton.backends.compiler import GPUTarget

compiler = pytest.importorskip("triton.backends.hcu.compiler_hcu")
HIPBackend = compiler.HIPBackend
_rlc_policy_signature = compiler._rlc_policy_signature
_gfx936_codegen_signature = compiler._gfx936_codegen_signature
_RLC_ENV_KEYS = (
    "FLAGTREE_HCU_RLC_ENHANCE",
    "FLAGTREE_HCU_RLC_PHASE_MASK",
    "FLAGTREE_HCU_RLC_ALLOW_ATOMIC_WRITEBACK_ORDER_CHANGE",
    "FLAGTREE_HCU_GFX936_F16_PAIR_MATERIALIZE",
    "FLAGTREE_HCU_GFX936_F32_BOX_MULLER_PAIR_MATERIALIZE",
    "FLAGTREE_HCU_GFX936_LLVM17_CONTRACT_BRIDGE",
    "FLAGTREE_HCU_GFX936_I64_VECTOR_LOAD_MATERIALIZE",
    "FLAGTREE_HCU_GFX936_I64_VECTOR_STORE_MATERIALIZE",
    "FLAGTREE_HCU_GFX936_I64_SCALAR_LOAD_MATERIALIZE",
    "FLAGTREE_HCU_GFX936_I64_SCALAR_STORE_MATERIALIZE",
    "FLAGTREE_HCU_GFX936_RESOURCE_PHI_MATERIALIZE",
)


def _backend():
    return HIPBackend(GPUTarget("hip", "gfx936", 64))


def _set_rlc(
    enhance: bool,
    mask: int = 15,
    pair_materialize: bool = False,
    f32_box_muller_pair_materialize: bool = False,
    llvm17_contract_bridge: bool = False,
    allow_atomic_writeback_order_change: bool = False,
):
    os.environ["FLAGTREE_HCU_RLC_ENHANCE"] = "1" if enhance else "0"
    os.environ["FLAGTREE_HCU_RLC_PHASE_MASK"] = str(mask)
    os.environ["FLAGTREE_HCU_RLC_ALLOW_ATOMIC_WRITEBACK_ORDER_CHANGE"] = (
        "1" if allow_atomic_writeback_order_change else "0"
    )
    os.environ["FLAGTREE_HCU_GFX936_F16_PAIR_MATERIALIZE"] = "1" if pair_materialize else "0"
    os.environ["FLAGTREE_HCU_GFX936_F32_BOX_MULLER_PAIR_MATERIALIZE"] = (
        "1" if f32_box_muller_pair_materialize else "0"
    )
    os.environ["FLAGTREE_HCU_GFX936_LLVM17_CONTRACT_BRIDGE"] = (
        "1" if llvm17_contract_bridge else "0"
    )
    for key in (
        "FLAGTREE_HCU_GFX936_I64_VECTOR_LOAD_MATERIALIZE",
        "FLAGTREE_HCU_GFX936_I64_VECTOR_STORE_MATERIALIZE",
        "FLAGTREE_HCU_GFX936_I64_SCALAR_LOAD_MATERIALIZE",
        "FLAGTREE_HCU_GFX936_I64_SCALAR_STORE_MATERIALIZE",
        "FLAGTREE_HCU_GFX936_RESOURCE_PHI_MATERIALIZE",
    ):
        os.environ[key] = "0"


def test_signature_tracks_in_process_rlc_flip():
    previous = {k: os.environ.get(k) for k in _RLC_ENV_KEYS}
    try:
        _set_rlc(False, 15)
        off = _rlc_policy_signature()
        _set_rlc(True, 15)
        on = _rlc_policy_signature()
        _set_rlc(True, 3)
        mask3 = _rlc_policy_signature()
        _set_rlc(True, 3, pair_materialize=True)
        pair_on = _rlc_policy_signature()
        _set_rlc(True, 3, pair_materialize=True, f32_box_muller_pair_materialize=True)
        f32_pair_on = _rlc_policy_signature()
        _set_rlc(
            True,
            3,
            pair_materialize=True,
            f32_box_muller_pair_materialize=True,
            allow_atomic_writeback_order_change=True,
        )
        atomic_order_on = _rlc_policy_signature()
        assert off != on, (off, on)
        assert on != mask3, (on, mask3)
        assert mask3 != pair_on, (mask3, pair_on)
        assert pair_on != f32_pair_on, (pair_on, f32_pair_on)
        assert f32_pair_on != atomic_order_on, (f32_pair_on, atomic_order_on)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_codegen_signature_tracks_in_process_bridge_flip():
    previous = {k: os.environ.get(k) for k in _RLC_ENV_KEYS}
    try:
        _set_rlc(False, llvm17_contract_bridge=False)
        off = _gfx936_codegen_signature()
        _set_rlc(False, llvm17_contract_bridge=True)
        on = _gfx936_codegen_signature()
        assert off == "0-0-0-0-0-0"
        assert on == "1-0-0-0-0-0"
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_codegen_signature_tracks_i64_vector_load_flip():
    previous = {k: os.environ.get(k) for k in _RLC_ENV_KEYS}
    try:
        _set_rlc(False, llvm17_contract_bridge=True)
        bridge_only = _gfx936_codegen_signature()
        os.environ["FLAGTREE_HCU_GFX936_I64_VECTOR_LOAD_MATERIALIZE"] = "1"
        vector_load_on = _gfx936_codegen_signature()
        assert bridge_only == "1-0-0-0-0-0"
        assert vector_load_on == "1-1-0-0-0-0"
        assert bridge_only != vector_load_on
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_backend_hash_tracks_in_process_rlc_flip():
    previous = {k: os.environ.get(k) for k in _RLC_ENV_KEYS}
    backend = _backend()
    try:
        _set_rlc(False, 15)
        try:
            off = backend.hash()
        except Exception as exc:
            pytest.skip(f"HCU clang unavailable: {exc}")
        _set_rlc(True, 15)
        on = backend.hash()
        _set_rlc(True, 3)
        mask3 = backend.hash()
        _set_rlc(True, 3, pair_materialize=True)
        pair_on = backend.hash()
        _set_rlc(True, 3, pair_materialize=True, f32_box_muller_pair_materialize=True)
        f32_pair_on = backend.hash()
        _set_rlc(
            True,
            3,
            pair_materialize=True,
            f32_box_muller_pair_materialize=True,
            allow_atomic_writeback_order_change=True,
        )
        atomic_order_on = backend.hash()
        assert off != on, (off, on)
        assert on != mask3, (on, mask3)
        assert mask3 != pair_on, (mask3, pair_on)
        assert pair_on != f32_pair_on, (pair_on, f32_pair_on)
        assert f32_pair_on != atomic_order_on, (f32_pair_on, atomic_order_on)
        assert "-rlc" in off and "-rlc" in on
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_options_include_live_rlc_policy():
    previous = {k: os.environ.get(k) for k in _RLC_ENV_KEYS}
    backend = _backend()
    try:
        _set_rlc(False)
        try:
            off = backend.parse_options({})
        except Exception as exc:
            pytest.skip(f"HCU parse_options unavailable: {exc}")
        _set_rlc(True)
        on = backend.parse_options({})
        _set_rlc(True, pair_materialize=True)
        pair_on = backend.parse_options({})
        _set_rlc(True, pair_materialize=True, f32_box_muller_pair_materialize=True)
        f32_pair_on = backend.parse_options({})
        _set_rlc(
            True,
            pair_materialize=True,
            f32_box_muller_pair_materialize=True,
            allow_atomic_writeback_order_change=True,
        )
        atomic_order_on = backend.parse_options({})
        _set_rlc(
            True,
            pair_materialize=True,
            f32_box_muller_pair_materialize=True,
            llvm17_contract_bridge=True,
        )
        bridge_on = backend.parse_options({})
        assert off.rlc_policy != on.rlc_policy, (off.rlc_policy, on.rlc_policy)
        assert on.rlc_policy != pair_on.rlc_policy, (on.rlc_policy, pair_on.rlc_policy)
        assert pair_on.rlc_policy != f32_pair_on.rlc_policy, (pair_on.rlc_policy, f32_pair_on.rlc_policy)
        assert f32_pair_on.rlc_policy != atomic_order_on.rlc_policy, (
            f32_pair_on.rlc_policy,
            atomic_order_on.rlc_policy,
        )
        assert not f32_pair_on.gfx936_llvm17_contract_bridge
        assert bridge_on.gfx936_llvm17_contract_bridge
        assert str(off) != str(on)
        assert off.hash() != on.hash()
        assert on.hash() != pair_on.hash()
        assert pair_on.hash() != f32_pair_on.hash()
        assert f32_pair_on.hash() != bridge_on.hash()
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
