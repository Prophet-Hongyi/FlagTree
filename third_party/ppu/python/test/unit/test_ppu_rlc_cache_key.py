"""RLC knobs must change PPU compile keys in the same process.

The old 810E v2 product had neither the common JIT backend identity nor a
live PPU RLC policy in backend/options hashes. That makes an off/on flip reuse
the incumbent CompiledKernel. PPU follows the same cache-key contract as the
other domestic backends: live enhance/mask identity and no cached hash.
"""

from __future__ import annotations

import inspect
import os

import pytest

from triton.backends.compiler import GPUTarget
from triton.runtime import jit

compiler = pytest.importorskip("triton.backends.ppu.compiler")
PPUBackend = compiler.PPUBackend
_rlc_policy_signature = compiler._rlc_policy_signature
_apply_ppu_rlc_policy = compiler._apply_ppu_rlc_policy
_RLC_ENV_KEYS = (
    "FLAGTREE_PPU_RLC_ENHANCE",
    "FLAGTREE_PPU_RLC_PHASE_MASK",
    "FLAGTREE_PPU_RLC_PROFITABILITY_POLICY",
    "FLAGTREE_PPU_RLC_PRODUCT_LAUNCH_COUNT",
    "FLAGTREE_PPU_RLC_MIN_ADJUSTED_SAVED_COST_PER_TENSOR_OP",
    "FLAGTREE_PPU_RLC_PHASE3_SAVED_COST_MULTIPLIER",
    "FLAGTREE_PPU_RLC_MAX_EXTERNAL_USE_EDGES",
    "FLAGTREE_PPU_RLC_MIN_REMOVED_CONVERT_DENSITY_PER_1024_PROPOSAL_VALUES",
    "FLAGTREE_PPU_RLC_LOW_DENSITY_GLOBAL_WRITEBACK_MIN_MATH_OPS",
    "FLAGTREE_PPU_RLC_LOW_DENSITY_OUTPUT_HEAVY_MIN_COMPUTE_OPS",
    "FLAGTREE_PPU_RLC_LOW_DENSITY_ZERO_LOAD_MIN_ARITHMETIC_OPS",
)


def _backend():
    return PPUBackend(GPUTarget("cuda", 80, 32))


def _set_rlc(enhance: bool, mask: int = 15):
    os.environ["FLAGTREE_PPU_RLC_ENHANCE"] = "1" if enhance else "0"
    os.environ["FLAGTREE_PPU_RLC_PHASE_MASK"] = str(mask)


def _restore(previous):
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def test_signature_tracks_in_process_rlc_flip():
    previous = {key: os.environ.get(key) for key in _RLC_ENV_KEYS}
    try:
        _set_rlc(False, 15)
        off = _rlc_policy_signature()
        _set_rlc(True, 15)
        on = _rlc_policy_signature()
        _set_rlc(True, 3)
        mask3 = _rlc_policy_signature()
        assert off != on, (off, on)
        assert on != mask3, (on, mask3)
    finally:
        _restore(previous)


def test_backend_hash_tracks_in_process_rlc_flip():
    previous = {key: os.environ.get(key) for key in _RLC_ENV_KEYS}
    backend = _backend()
    try:
        _set_rlc(False, 15)
        try:
            off = backend.hash()
        except Exception as exc:
            pytest.skip(f"PPU compiler unavailable: {exc}")
        _set_rlc(True, 15)
        on = backend.hash()
        _set_rlc(True, 3)
        mask3 = backend.hash()
        assert off != on, (off, on)
        assert on != mask3, (on, mask3)
        assert "-rlc" in off and "-rlc" in on
    finally:
        _restore(previous)


def test_options_include_live_rlc_policy():
    previous = {key: os.environ.get(key) for key in _RLC_ENV_KEYS}
    backend = _backend()
    try:
        _set_rlc(False, 15)
        off = backend.parse_options({})
        _set_rlc(True, 15)
        on = backend.parse_options({})
        _set_rlc(True, 3)
        mask3 = backend.parse_options({})
        assert off.rlc_policy != on.rlc_policy, (off.rlc_policy, on.rlc_policy)
        assert on.rlc_policy != mask3.rlc_policy, (on.rlc_policy, mask3.rlc_policy)
        assert str(off) != str(on)
        assert off.hash() != on.hash()
        assert on.hash() != mask3.hash()
    finally:
        _restore(previous)


def test_signature_tracks_profitability_contract_only_when_enabled():
    previous = {key: os.environ.get(key) for key in _RLC_ENV_KEYS}
    try:
        _set_rlc(True, 13)
        os.environ["FLAGTREE_PPU_RLC_PROFITABILITY_POLICY"] = "1"
        os.environ["FLAGTREE_PPU_RLC_PRODUCT_LAUNCH_COUNT"] = "1"
        os.environ["FLAGTREE_PPU_RLC_MIN_ADJUSTED_SAVED_COST_PER_TENSOR_OP"] = "1900"
        os.environ["FLAGTREE_PPU_RLC_PHASE3_SAVED_COST_MULTIPLIER"] = "3"
        os.environ["FLAGTREE_PPU_RLC_MAX_EXTERNAL_USE_EDGES"] = "2"
        os.environ["FLAGTREE_PPU_RLC_MIN_REMOVED_CONVERT_DENSITY_PER_1024_PROPOSAL_VALUES"] = "128"
        os.environ["FLAGTREE_PPU_RLC_LOW_DENSITY_GLOBAL_WRITEBACK_MIN_MATH_OPS"] = "8"
        os.environ["FLAGTREE_PPU_RLC_LOW_DENSITY_OUTPUT_HEAVY_MIN_COMPUTE_OPS"] = "128"
        os.environ["FLAGTREE_PPU_RLC_LOW_DENSITY_ZERO_LOAD_MIN_ARITHMETIC_OPS"] = "100"
        baseline = _rlc_policy_signature()
        os.environ["FLAGTREE_PPU_RLC_LOW_DENSITY_ZERO_LOAD_MIN_ARITHMETIC_OPS"] = "101"
        assert baseline != _rlc_policy_signature()

        os.environ["FLAGTREE_PPU_RLC_PROFITABILITY_POLICY"] = "0"
        disabled = _rlc_policy_signature()
        os.environ["FLAGTREE_PPU_RLC_MIN_ADJUSTED_SAVED_COST_PER_TENSOR_OP"] = "9000"
        assert disabled == _rlc_policy_signature()

        os.environ["FLAGTREE_PPU_RLC_PROFITABILITY_POLICY"] = "1"
        _set_rlc(True, 3)
        no_owner = _rlc_policy_signature()
        os.environ["FLAGTREE_PPU_RLC_PRODUCT_LAUNCH_COUNT"] = "7"
        assert no_owner == _rlc_policy_signature()
    finally:
        _restore(previous)


def test_profitability_module_attrs_are_explicit_and_fail_closed(monkeypatch):
    previous = {key: os.environ.get(key) for key in _RLC_ENV_KEYS}

    class FakeBuilder:
        @staticmethod
        def get_int32_attr(value):
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
        os.environ["FLAGTREE_PPU_RLC_PROFITABILITY_POLICY"] = "1"
        os.environ["FLAGTREE_PPU_RLC_PRODUCT_LAUNCH_COUNT"] = "1"
        os.environ["FLAGTREE_PPU_RLC_MIN_ADJUSTED_SAVED_COST_PER_TENSOR_OP"] = "1900"
        os.environ["FLAGTREE_PPU_RLC_PHASE3_SAVED_COST_MULTIPLIER"] = "3"
        os.environ["FLAGTREE_PPU_RLC_MAX_EXTERNAL_USE_EDGES"] = "2"
        os.environ["FLAGTREE_PPU_RLC_MIN_REMOVED_CONVERT_DENSITY_PER_1024_PROPOSAL_VALUES"] = "128"
        os.environ["FLAGTREE_PPU_RLC_LOW_DENSITY_GLOBAL_WRITEBACK_MIN_MATH_OPS"] = "8"
        os.environ["FLAGTREE_PPU_RLC_LOW_DENSITY_OUTPUT_HEAVY_MIN_COMPUTE_OPS"] = "128"
        os.environ["FLAGTREE_PPU_RLC_LOW_DENSITY_ZERO_LOAD_MIN_ARITHMETIC_OPS"] = "100"
        complete = FakeModule()
        _apply_ppu_rlc_policy(complete)
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
        }

        os.environ["FLAGTREE_PPU_RLC_PRODUCT_LAUNCH_COUNT"] = "0"
        os.environ["FLAGTREE_PPU_RLC_MIN_ADJUSTED_SAVED_COST_PER_TENSOR_OP"] = "0"
        incomplete = FakeModule()
        _apply_ppu_rlc_policy(incomplete)
        assert incomplete.attrs["ttg.rlc-profitability-policy-enabled"] == 1
        assert "ttg.rlc-product-launch-count" not in incomplete.attrs
        assert "ttg.rlc-profitability-min-adjusted-saved-cost-per-tensor-op" not in incomplete.attrs
    finally:
        _restore(previous)


def test_installed_jit_cache_key_includes_backend_identity():
    source = inspect.getsource(jit.JITFunction.run)
    assert 'key = f"{key}-{backend.hash()}"' in source
