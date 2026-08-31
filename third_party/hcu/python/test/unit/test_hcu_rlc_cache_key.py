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
_apply_hcu_rlc_policy = compiler._apply_hcu_rlc_policy
_gfx936_codegen_signature = compiler._gfx936_codegen_signature
_RLC_ENV_KEYS = (
    "FLAGTREE_HCU_RLC_ENHANCE",
    "FLAGTREE_HCU_RLC_PHASE_MASK",
    "FLAGTREE_HCU_RLC_PROFITABILITY_POLICY",
    "FLAGTREE_HCU_RLC_PRODUCT_LAUNCH_COUNT",
    "FLAGTREE_HCU_RLC_MIN_ADJUSTED_SAVED_COST_PER_TENSOR_OP",
    "FLAGTREE_HCU_RLC_PHASE3_SAVED_COST_MULTIPLIER",
    "FLAGTREE_HCU_RLC_MAX_EXTERNAL_USE_EDGES",
    "FLAGTREE_HCU_RLC_MIN_REMOVED_CONVERT_DENSITY_PER_1024_PROPOSAL_VALUES",
    "FLAGTREE_HCU_RLC_LOW_DENSITY_GLOBAL_WRITEBACK_MIN_MATH_OPS",
    "FLAGTREE_HCU_RLC_LOW_DENSITY_OUTPUT_HEAVY_MIN_COMPUTE_OPS",
    "FLAGTREE_HCU_RLC_LOW_DENSITY_ZERO_LOAD_MIN_ARITHMETIC_OPS",
    "FLAGTREE_HCU_RLC_LOW_DENSITY_LOOP_RESIDENT_MIN_SAVED_COST",
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
    os.environ["FLAGTREE_HCU_RLC_PROFITABILITY_POLICY"] = "0"
    os.environ["FLAGTREE_HCU_RLC_PRODUCT_LAUNCH_COUNT"] = "0"
    os.environ["FLAGTREE_HCU_RLC_MIN_ADJUSTED_SAVED_COST_PER_TENSOR_OP"] = "0"
    os.environ["FLAGTREE_HCU_RLC_PHASE3_SAVED_COST_MULTIPLIER"] = "0"
    os.environ["FLAGTREE_HCU_RLC_MAX_EXTERNAL_USE_EDGES"] = "0"
    os.environ["FLAGTREE_HCU_RLC_MIN_REMOVED_CONVERT_DENSITY_PER_1024_PROPOSAL_VALUES"] = "0"
    os.environ["FLAGTREE_HCU_RLC_LOW_DENSITY_GLOBAL_WRITEBACK_MIN_MATH_OPS"] = "0"
    os.environ["FLAGTREE_HCU_RLC_LOW_DENSITY_OUTPUT_HEAVY_MIN_COMPUTE_OPS"] = "0"
    os.environ["FLAGTREE_HCU_RLC_LOW_DENSITY_ZERO_LOAD_MIN_ARITHMETIC_OPS"] = "0"
    os.environ["FLAGTREE_HCU_RLC_LOW_DENSITY_LOOP_RESIDENT_MIN_SAVED_COST"] = "0"
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


def test_signature_tracks_profitability_contract_only_when_enabled():
    previous = {k: os.environ.get(k) for k in _RLC_ENV_KEYS}
    try:
        _set_rlc(True, 13)
        os.environ["FLAGTREE_HCU_RLC_PROFITABILITY_POLICY"] = "1"
        os.environ["FLAGTREE_HCU_RLC_PRODUCT_LAUNCH_COUNT"] = "1"
        os.environ["FLAGTREE_HCU_RLC_MIN_ADJUSTED_SAVED_COST_PER_TENSOR_OP"] = "1900"
        os.environ["FLAGTREE_HCU_RLC_PHASE3_SAVED_COST_MULTIPLIER"] = "3"
        os.environ["FLAGTREE_HCU_RLC_MAX_EXTERNAL_USE_EDGES"] = "2"
        os.environ["FLAGTREE_HCU_RLC_MIN_REMOVED_CONVERT_DENSITY_PER_1024_PROPOSAL_VALUES"] = "128"
        os.environ["FLAGTREE_HCU_RLC_LOW_DENSITY_GLOBAL_WRITEBACK_MIN_MATH_OPS"] = "8"
        os.environ["FLAGTREE_HCU_RLC_LOW_DENSITY_OUTPUT_HEAVY_MIN_COMPUTE_OPS"] = "128"
        os.environ["FLAGTREE_HCU_RLC_LOW_DENSITY_ZERO_LOAD_MIN_ARITHMETIC_OPS"] = "100"
        os.environ["FLAGTREE_HCU_RLC_LOW_DENSITY_LOOP_RESIDENT_MIN_SAVED_COST"] = "4194304"
        baseline = _rlc_policy_signature()
        os.environ["FLAGTREE_HCU_RLC_LOW_DENSITY_LOOP_RESIDENT_MIN_SAVED_COST"] = "4194305"
        assert baseline != _rlc_policy_signature()

        os.environ["FLAGTREE_HCU_RLC_PROFITABILITY_POLICY"] = "0"
        disabled = _rlc_policy_signature()
        os.environ["FLAGTREE_HCU_RLC_MIN_ADJUSTED_SAVED_COST_PER_TENSOR_OP"] = "9000"
        assert disabled == _rlc_policy_signature()

        _set_rlc(True, 3)
        os.environ["FLAGTREE_HCU_RLC_PROFITABILITY_POLICY"] = "1"
        no_owner = _rlc_policy_signature()
        os.environ["FLAGTREE_HCU_RLC_PRODUCT_LAUNCH_COUNT"] = "7"
        assert no_owner == _rlc_policy_signature()
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_profitability_module_attrs_are_explicit_and_fail_closed(monkeypatch):
    previous = {k: os.environ.get(k) for k in _RLC_ENV_KEYS}

    class FakeBuilder:
        @staticmethod
        def get_int32_attr(value):
            return value

        @staticmethod
        def get_int64_attr(value):
            return value

    class FakeModule:
        context = object()

        def __init__(self):
            self.attrs = {}

        def set_attr(self, name, value):
            self.attrs[name] = value

    monkeypatch.setattr(compiler.ir, "builder", lambda context: FakeBuilder())
    try:
        _set_rlc(True, 13)
        os.environ["FLAGTREE_HCU_RLC_PROFITABILITY_POLICY"] = "1"
        os.environ["FLAGTREE_HCU_RLC_PRODUCT_LAUNCH_COUNT"] = "1"
        os.environ["FLAGTREE_HCU_RLC_MIN_ADJUSTED_SAVED_COST_PER_TENSOR_OP"] = "1900"
        os.environ["FLAGTREE_HCU_RLC_PHASE3_SAVED_COST_MULTIPLIER"] = "3"
        os.environ["FLAGTREE_HCU_RLC_MAX_EXTERNAL_USE_EDGES"] = "2"
        os.environ["FLAGTREE_HCU_RLC_MIN_REMOVED_CONVERT_DENSITY_PER_1024_PROPOSAL_VALUES"] = "128"
        os.environ["FLAGTREE_HCU_RLC_LOW_DENSITY_GLOBAL_WRITEBACK_MIN_MATH_OPS"] = "8"
        os.environ["FLAGTREE_HCU_RLC_LOW_DENSITY_OUTPUT_HEAVY_MIN_COMPUTE_OPS"] = "128"
        os.environ["FLAGTREE_HCU_RLC_LOW_DENSITY_ZERO_LOAD_MIN_ARITHMETIC_OPS"] = "100"
        os.environ["FLAGTREE_HCU_RLC_LOW_DENSITY_LOOP_RESIDENT_MIN_SAVED_COST"] = "4194304"
        complete = FakeModule()
        _apply_hcu_rlc_policy(complete)
        assert complete.attrs == {
            "ttg.rlc-profitability-policy-enabled": 1,
            "ttg.rlc-product-launch-count": 1,
            "ttg.rlc-profitability-min-adjusted-saved-cost-per-tensor-op": 1900,
            "ttg.rlc-profitability-phase3-saved-cost-multiplier": 3,
            "ttg.rlc-profitability-max-external-use-edges": 2,
            "ttg.rlc-profitability-min-removed-convert-density-per-1024-proposal-values": 128,
            "ttg.rlc-profitability-low-density-global-writeback-min-math-ops": 8,
            "ttg.rlc-profitability-low-density-output-heavy-min-compute-ops": 128,
            "ttg.rlc-profitability-low-density-zero-load-min-arithmetic-ops": 100,
            "ttg.rlc-profitability-low-density-loop-resident-min-saved-cost": 4194304,
        }

        os.environ["FLAGTREE_HCU_RLC_PRODUCT_LAUNCH_COUNT"] = "0"
        os.environ["FLAGTREE_HCU_RLC_MIN_ADJUSTED_SAVED_COST_PER_TENSOR_OP"] = "0"
        incomplete = FakeModule()
        _apply_hcu_rlc_policy(incomplete)
        assert incomplete.attrs["ttg.rlc-profitability-policy-enabled"] == 1
        assert "ttg.rlc-product-launch-count" not in incomplete.attrs
        assert "ttg.rlc-profitability-min-adjusted-saved-cost-per-tensor-op" not in incomplete.attrs
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
