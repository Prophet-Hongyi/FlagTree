"""RLC knobs must change MetaX compile keys in the same process.

MThreads FlagGems 4→4 was a harness false negative when site `hash()`
omitted RLC and `@functools.lru_cache()` froze the first env. MetaX follows
that cache-key contract with its own live, fail-closed profitability knobs.
"""

from __future__ import annotations

import os

import pytest

from triton.backends.compiler import GPUTarget

compiler = pytest.importorskip("triton.backends.metax.compiler")
MACABackend = compiler.MACABackend
_rlc_policy_signature = compiler._rlc_policy_signature
_apply_metax_rlc_policy = compiler._apply_metax_rlc_policy


def _backend():
    return MACABackend(GPUTarget("maca", 80, 64))


def _set_rlc(enhance: bool, mask: int = 15):
    os.environ["FLAGTREE_METAX_RLC_ENHANCE"] = "1" if enhance else "0"
    os.environ["FLAGTREE_METAX_RLC_PHASE_MASK"] = str(mask)


def test_signature_tracks_in_process_rlc_flip():
    previous = {k: os.environ.get(k) for k in ("FLAGTREE_METAX_RLC_ENHANCE", "FLAGTREE_METAX_RLC_PHASE_MASK")}
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
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_backend_hash_tracks_in_process_rlc_flip_without_reprobing_mxcc(monkeypatch):
    previous = {k: os.environ.get(k) for k in ("FLAGTREE_METAX_RLC_ENHANCE", "FLAGTREE_METAX_RLC_PHASE_MASK")}
    backend = _backend()
    calls = []

    def fake_check_output(argv):
        calls.append(tuple(map(str, argv)))
        return b"mxcc version test\n"

    monkeypatch.setattr(compiler.subprocess, "check_output", fake_check_output)
    monkeypatch.delattr(MACABackend, "_cached_mxcc_version", raising=False)
    try:
        _set_rlc(False, 15)
        off = backend.hash()
        _set_rlc(True, 15)
        on = backend.hash()
        _set_rlc(True, 3)
        mask3 = backend.hash()
        assert off != on, (off, on)
        assert on != mask3, (on, mask3)
        assert "-rlc" in off and "-rlc" in on
        assert len(calls) == 1, calls
        assert calls[0][-1] == "--version"
    finally:
        monkeypatch.delattr(MACABackend, "_cached_mxcc_version", raising=False)
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_options_include_live_rlc_policy():
    previous = {k: os.environ.get(k) for k in ("FLAGTREE_METAX_RLC_ENHANCE", "FLAGTREE_METAX_RLC_PHASE_MASK")}
    backend = _backend()
    try:
        _set_rlc(False)
        try:
            off = backend.parse_options({})
        except Exception as exc:
            pytest.skip(f"MACA parse_options unavailable: {exc}")
        _set_rlc(True)
        on = backend.parse_options({})
        assert off.rlc_policy != on.rlc_policy, (off.rlc_policy, on.rlc_policy)
        assert str(off) != str(on)
        assert off.hash() != on.hash()
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_signature_tracks_profitability_contract_only_when_enabled():
    keys = (
        "FLAGTREE_METAX_RLC_ENHANCE",
        "FLAGTREE_METAX_RLC_PHASE_MASK",
        "FLAGTREE_METAX_RLC_PROFITABILITY_POLICY",
        "FLAGTREE_METAX_RLC_PRODUCT_LAUNCH_COUNT",
        "FLAGTREE_METAX_RLC_MIN_ADJUSTED_SAVED_COST_PER_TENSOR_OP",
        "FLAGTREE_METAX_RLC_PHASE3_SAVED_COST_MULTIPLIER",
        "FLAGTREE_METAX_RLC_MAX_EXTERNAL_USE_EDGES",
        "FLAGTREE_METAX_RLC_MIN_REMOVED_CONVERT_DENSITY_PER_1024_PROPOSAL_VALUES",
        "FLAGTREE_METAX_RLC_LOW_DENSITY_GLOBAL_WRITEBACK_MIN_MATH_OPS",
        "FLAGTREE_METAX_RLC_LOW_DENSITY_OUTPUT_HEAVY_MIN_COMPUTE_OPS",
        "FLAGTREE_METAX_RLC_LOW_DENSITY_ZERO_LOAD_MIN_ARITHMETIC_OPS",
    )
    previous = {key: os.environ.get(key) for key in keys}
    try:
        _set_rlc(True, 13)
        os.environ["FLAGTREE_METAX_RLC_PROFITABILITY_POLICY"] = "1"
        os.environ["FLAGTREE_METAX_RLC_PRODUCT_LAUNCH_COUNT"] = "1"
        os.environ["FLAGTREE_METAX_RLC_MIN_ADJUSTED_SAVED_COST_PER_TENSOR_OP"] = "2200"
        os.environ["FLAGTREE_METAX_RLC_PHASE3_SAVED_COST_MULTIPLIER"] = "3"
        os.environ["FLAGTREE_METAX_RLC_MAX_EXTERNAL_USE_EDGES"] = "0"
        os.environ["FLAGTREE_METAX_RLC_MIN_REMOVED_CONVERT_DENSITY_PER_1024_PROPOSAL_VALUES"] = "128"
        os.environ["FLAGTREE_METAX_RLC_LOW_DENSITY_GLOBAL_WRITEBACK_MIN_MATH_OPS"] = "8"
        os.environ["FLAGTREE_METAX_RLC_LOW_DENSITY_OUTPUT_HEAVY_MIN_COMPUTE_OPS"] = "128"
        os.environ["FLAGTREE_METAX_RLC_LOW_DENSITY_ZERO_LOAD_MIN_ARITHMETIC_OPS"] = "100"
        baseline = _rlc_policy_signature()
        os.environ["FLAGTREE_METAX_RLC_LOW_DENSITY_ZERO_LOAD_MIN_ARITHMETIC_OPS"] = "101"
        retuned = _rlc_policy_signature()
        assert baseline != retuned

        os.environ["FLAGTREE_METAX_RLC_PROFITABILITY_POLICY"] = "0"
        disabled = _rlc_policy_signature()
        os.environ["FLAGTREE_METAX_RLC_MIN_ADJUSTED_SAVED_COST_PER_TENSOR_OP"] = "9000"
        assert disabled == _rlc_policy_signature()

        os.environ["FLAGTREE_METAX_RLC_PROFITABILITY_POLICY"] = "1"
        _set_rlc(True, 3)
        no_owner = _rlc_policy_signature()
        os.environ["FLAGTREE_METAX_RLC_PRODUCT_LAUNCH_COUNT"] = "7"
        assert no_owner == _rlc_policy_signature()
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_profitability_module_attrs_are_explicit_and_fail_closed(monkeypatch):
    keys = (
        "FLAGTREE_METAX_RLC_ENHANCE",
        "FLAGTREE_METAX_RLC_PHASE_MASK",
        "FLAGTREE_METAX_RLC_PROFITABILITY_POLICY",
        "FLAGTREE_METAX_RLC_PRODUCT_LAUNCH_COUNT",
        "FLAGTREE_METAX_RLC_MIN_ADJUSTED_SAVED_COST_PER_TENSOR_OP",
        "FLAGTREE_METAX_RLC_PHASE3_SAVED_COST_MULTIPLIER",
        "FLAGTREE_METAX_RLC_MAX_EXTERNAL_USE_EDGES",
        "FLAGTREE_METAX_RLC_MIN_REMOVED_CONVERT_DENSITY_PER_1024_PROPOSAL_VALUES",
        "FLAGTREE_METAX_RLC_LOW_DENSITY_GLOBAL_WRITEBACK_MIN_MATH_OPS",
        "FLAGTREE_METAX_RLC_LOW_DENSITY_OUTPUT_HEAVY_MIN_COMPUTE_OPS",
        "FLAGTREE_METAX_RLC_LOW_DENSITY_ZERO_LOAD_MIN_ARITHMETIC_OPS",
    )
    previous = {key: os.environ.get(key) for key in keys}

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
        os.environ["FLAGTREE_METAX_RLC_PROFITABILITY_POLICY"] = "1"
        os.environ["FLAGTREE_METAX_RLC_PRODUCT_LAUNCH_COUNT"] = "1"
        os.environ["FLAGTREE_METAX_RLC_MIN_ADJUSTED_SAVED_COST_PER_TENSOR_OP"] = "2200"
        os.environ["FLAGTREE_METAX_RLC_PHASE3_SAVED_COST_MULTIPLIER"] = "3"
        os.environ["FLAGTREE_METAX_RLC_MAX_EXTERNAL_USE_EDGES"] = "2"
        os.environ["FLAGTREE_METAX_RLC_MIN_REMOVED_CONVERT_DENSITY_PER_1024_PROPOSAL_VALUES"] = "128"
        os.environ["FLAGTREE_METAX_RLC_LOW_DENSITY_GLOBAL_WRITEBACK_MIN_MATH_OPS"] = "8"
        os.environ["FLAGTREE_METAX_RLC_LOW_DENSITY_OUTPUT_HEAVY_MIN_COMPUTE_OPS"] = "128"
        os.environ["FLAGTREE_METAX_RLC_LOW_DENSITY_ZERO_LOAD_MIN_ARITHMETIC_OPS"] = "100"
        complete = FakeModule()
        _apply_metax_rlc_policy(complete)
        assert complete.attrs == {
            "ttg.rlc-profitability-policy-enabled": 1,
            "ttg.rlc-product-launch-count": 1,
            "ttg.rlc-profitability-min-adjusted-saved-cost-per-tensor-op": 2200,
            "ttg.rlc-profitability-phase3-saved-cost-multiplier": 3,
            "ttg.rlc-profitability-max-external-use-edges": 2,
            "ttg.rlc-profitability-min-removed-convert-density-per-1024-proposal-values": 128,
            "ttg.rlc-profitability-low-density-global-writeback-min-math-ops": 8,
            "ttg.rlc-profitability-low-density-output-heavy-min-compute-ops": 128,
            "ttg.rlc-profitability-low-density-zero-load-min-arithmetic-ops": 100,
        }

        os.environ["FLAGTREE_METAX_RLC_PRODUCT_LAUNCH_COUNT"] = "0"
        os.environ["FLAGTREE_METAX_RLC_MIN_ADJUSTED_SAVED_COST_PER_TENSOR_OP"] = "0"
        incomplete = FakeModule()
        _apply_metax_rlc_policy(incomplete)
        assert incomplete.attrs["ttg.rlc-profitability-policy-enabled"] == 1
        assert "ttg.rlc-product-launch-count" not in incomplete.attrs
        assert "ttg.rlc-profitability-min-adjusted-saved-cost-per-tensor-op" not in incomplete.attrs
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
