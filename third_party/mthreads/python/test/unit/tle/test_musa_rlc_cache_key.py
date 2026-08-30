"""RLC knobs must change MUSA compile keys in the same process.

The 2026-08-18 FlagGems 4→4 result was a harness false negative: site
`hash()` omitted RLC, `@functools.lru_cache()` froze the first env, and
`JITFunction.device_caches` keyed only on launch kwargs. MThreads loads
`triton/spec/mthreads/triton/runtime/jit.py`, not `triton/runtime/jit.py`.
"""

from __future__ import annotations

import os
import inspect

from triton.backends.compiler import GPUTarget
from triton.backends.mthreads import compiler as musa_compiler
from triton.backends.mthreads.compiler import MUSABackend, _rlc_policy_signature
from triton.runtime import jit


def _backend():
    return MUSABackend(GPUTarget("musa", 31, 32))


def _set_rlc(enhance: bool, mask: int = 15):
    os.environ["FLAGTREE_MUSA_RLC_ENHANCE"] = "1" if enhance else "0"
    os.environ["FLAGTREE_MUSA_RLC_PHASE_MASK"] = str(mask)


def test_backend_hash_tracks_in_process_rlc_flip():
    previous = {k: os.environ.get(k) for k in ("FLAGTREE_MUSA_RLC_ENHANCE", "FLAGTREE_MUSA_RLC_PHASE_MASK")}
    backend = _backend()
    try:
        _set_rlc(False, 15)
        off = backend.hash()
        sig_off = _rlc_policy_signature()
        _set_rlc(True, 15)
        on = backend.hash()
        sig_on = _rlc_policy_signature()
        _set_rlc(True, 3)
        mask3 = backend.hash()
        assert sig_off != sig_on, (sig_off, sig_on)
        assert off != on, (off, on)
        assert on != mask3, (on, mask3)
        assert "-rlc" in off and "-rlc" in on
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_options_include_live_rlc_policy():
    previous = {k: os.environ.get(k) for k in ("FLAGTREE_MUSA_RLC_ENHANCE", "FLAGTREE_MUSA_RLC_PHASE_MASK")}
    backend = _backend()
    try:
        _set_rlc(False)
        off = backend.parse_options({})
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


def test_installed_jit_cache_key_includes_backend_identity():
    """The MThreads spec overlay, not the common file, owns this runtime."""
    source = inspect.getsource(jit.JITFunction.run)
    assert 'key = f"{key}-{backend.hash()}"' in source


def test_backend_hash_tracks_atomic_writeback_policy():
    keys = (
        "FLAGTREE_MUSA_RLC_ENHANCE",
        "FLAGTREE_MUSA_RLC_PHASE_MASK",
        "FLAGTREE_MUSA_RLC_ATOMIC_WRITEBACK_MAX_ELEMS_PER_THREAD_RATIO",
    )
    previous = {key: os.environ.get(key) for key in keys}
    backend = _backend()
    try:
        _set_rlc(True, 5)
        key = "FLAGTREE_MUSA_RLC_ATOMIC_WRITEBACK_MAX_ELEMS_PER_THREAD_RATIO"
        os.environ[key] = "1"
        ratio1 = backend.hash()
        options1 = backend.parse_options({})
        os.environ[key] = "2"
        ratio2 = backend.hash()
        options2 = backend.parse_options({})
        assert ratio1 != ratio2, (ratio1, ratio2)
        assert options1.rlc_policy != options2.rlc_policy
        assert options1.hash() != options2.hash()
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_backend_hash_ignores_atomic_policy_without_phase2():
    keys = (
        "FLAGTREE_MUSA_RLC_ENHANCE",
        "FLAGTREE_MUSA_RLC_PHASE_MASK",
        "FLAGTREE_MUSA_RLC_ATOMIC_WRITEBACK_MAX_ELEMS_PER_THREAD_RATIO",
    )
    previous = {key: os.environ.get(key) for key in keys}
    backend = _backend()
    try:
        _set_rlc(True, 3)
        key = "FLAGTREE_MUSA_RLC_ATOMIC_WRITEBACK_MAX_ELEMS_PER_THREAD_RATIO"
        os.environ[key] = "1"
        ratio1 = backend.hash()
        options1 = backend.parse_options({})
        os.environ[key] = "2"
        ratio2 = backend.hash()
        options2 = backend.parse_options({})
        assert ratio1 == ratio2, (ratio1, ratio2)
        assert options1.rlc_policy == options2.rlc_policy
        assert options1.hash() == options2.hash()
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_backend_hash_tracks_int_to_fp_contiguity_policy():
    keys = (
        "FLAGTREE_MUSA_RLC_ENHANCE",
        "FLAGTREE_MUSA_RLC_PHASE_MASK",
        "FLAGTREE_MUSA_RLC_PRESERVE_INT_TO_FP_CONTIGUITY",
    )
    previous = {key: os.environ.get(key) for key in keys}
    backend = _backend()
    try:
        _set_rlc(True, 5)
        key = "FLAGTREE_MUSA_RLC_PRESERVE_INT_TO_FP_CONTIGUITY"
        os.environ[key] = "1"
        guarded = backend.hash()
        guarded_options = backend.parse_options({})
        os.environ[key] = "0"
        unguarded = backend.hash()
        unguarded_options = backend.parse_options({})
        assert guarded != unguarded, (guarded, unguarded)
        assert guarded_options.rlc_policy != unguarded_options.rlc_policy
        assert guarded_options.hash() != unguarded_options.hash()
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_backend_hash_ignores_int_to_fp_policy_without_phase2():
    keys = (
        "FLAGTREE_MUSA_RLC_ENHANCE",
        "FLAGTREE_MUSA_RLC_PHASE_MASK",
        "FLAGTREE_MUSA_RLC_PRESERVE_INT_TO_FP_CONTIGUITY",
    )
    previous = {key: os.environ.get(key) for key in keys}
    backend = _backend()
    try:
        _set_rlc(True, 3)
        key = "FLAGTREE_MUSA_RLC_PRESERVE_INT_TO_FP_CONTIGUITY"
        os.environ[key] = "1"
        guarded = backend.hash()
        guarded_options = backend.parse_options({})
        os.environ[key] = "0"
        unguarded = backend.hash()
        unguarded_options = backend.parse_options({})
        assert guarded == unguarded, (guarded, unguarded)
        assert guarded_options.rlc_policy == unguarded_options.rlc_policy
        assert guarded_options.hash() == unguarded_options.hash()
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_backend_hash_tracks_profitability_policy_contract():
    keys = (
        "FLAGTREE_MUSA_RLC_ENHANCE",
        "FLAGTREE_MUSA_RLC_PHASE_MASK",
        "FLAGTREE_MUSA_RLC_PROFITABILITY_POLICY",
        "FLAGTREE_MUSA_RLC_PRODUCT_LAUNCH_COUNT",
        "FLAGTREE_MUSA_RLC_MIN_ADJUSTED_SAVED_COST_PER_TENSOR_OP",
        "FLAGTREE_MUSA_RLC_PHASE3_SAVED_COST_MULTIPLIER",
        "FLAGTREE_MUSA_RLC_MAX_EXTERNAL_USE_EDGES",
    )
    previous = {key: os.environ.get(key) for key in keys}
    backend = _backend()
    try:
        _set_rlc(True, 13)
        os.environ["FLAGTREE_MUSA_RLC_PROFITABILITY_POLICY"] = "1"
        os.environ["FLAGTREE_MUSA_RLC_PRODUCT_LAUNCH_COUNT"] = "1"
        os.environ["FLAGTREE_MUSA_RLC_MIN_ADJUSTED_SAVED_COST_PER_TENSOR_OP"] = "1800"
        os.environ["FLAGTREE_MUSA_RLC_PHASE3_SAVED_COST_MULTIPLIER"] = "2"
        os.environ.pop("FLAGTREE_MUSA_RLC_MAX_EXTERNAL_USE_EDGES", None)
        calibrated = backend.hash()
        calibrated_options = backend.parse_options({})
        os.environ["FLAGTREE_MUSA_RLC_MAX_EXTERNAL_USE_EDGES"] = "0"
        baseline = backend.hash()
        baseline_options = backend.parse_options({})

        os.environ["FLAGTREE_MUSA_RLC_MIN_ADJUSTED_SAVED_COST_PER_TENSOR_OP"] = "1900"
        retuned = backend.hash()
        retuned_options = backend.parse_options({})
        os.environ["FLAGTREE_MUSA_RLC_PRODUCT_LAUNCH_COUNT"] = "2"
        two_launches = backend.hash()

        assert calibrated != baseline
        assert calibrated_options.rlc_policy != baseline_options.rlc_policy
        assert calibrated_options.hash() != baseline_options.hash()
        assert baseline != retuned
        assert baseline_options.rlc_policy != retuned_options.rlc_policy
        assert baseline_options.hash() != retuned_options.hash()
        assert retuned != two_launches
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_backend_hash_ignores_profitability_tuning_when_selector_is_off():
    keys = (
        "FLAGTREE_MUSA_RLC_ENHANCE",
        "FLAGTREE_MUSA_RLC_PHASE_MASK",
        "FLAGTREE_MUSA_RLC_PROFITABILITY_POLICY",
        "FLAGTREE_MUSA_RLC_PRODUCT_LAUNCH_COUNT",
        "FLAGTREE_MUSA_RLC_MIN_ADJUSTED_SAVED_COST_PER_TENSOR_OP",
    )
    previous = {key: os.environ.get(key) for key in keys}
    backend = _backend()
    try:
        _set_rlc(True, 13)
        os.environ["FLAGTREE_MUSA_RLC_PROFITABILITY_POLICY"] = "0"
        os.environ["FLAGTREE_MUSA_RLC_PRODUCT_LAUNCH_COUNT"] = "1"
        os.environ["FLAGTREE_MUSA_RLC_MIN_ADJUSTED_SAVED_COST_PER_TENSOR_OP"] = "1800"
        first = backend.hash()
        first_options = backend.parse_options({})
        os.environ["FLAGTREE_MUSA_RLC_PRODUCT_LAUNCH_COUNT"] = "7"
        os.environ["FLAGTREE_MUSA_RLC_MIN_ADJUSTED_SAVED_COST_PER_TENSOR_OP"] = "9000"
        second = backend.hash()
        second_options = backend.parse_options({})

        assert first == second
        assert first_options.rlc_policy == second_options.rlc_policy
        assert first_options.hash() == second_options.hash()
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_profitability_policy_module_attrs_are_explicit_and_fail_closed(monkeypatch):
    keys = (
        "FLAGTREE_MUSA_RLC_ENHANCE",
        "FLAGTREE_MUSA_RLC_PHASE_MASK",
        "FLAGTREE_MUSA_RLC_PROFITABILITY_POLICY",
        "FLAGTREE_MUSA_RLC_PRODUCT_LAUNCH_COUNT",
        "FLAGTREE_MUSA_RLC_MIN_ADJUSTED_SAVED_COST_PER_TENSOR_OP",
        "FLAGTREE_MUSA_RLC_PHASE3_SAVED_COST_MULTIPLIER",
        "FLAGTREE_MUSA_RLC_MAX_EXTERNAL_USE_EDGES",
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

    monkeypatch.setattr(musa_compiler.ir, "builder", lambda context: FakeBuilder())
    try:
        _set_rlc(True, 13)
        os.environ["FLAGTREE_MUSA_RLC_PROFITABILITY_POLICY"] = "1"
        os.environ["FLAGTREE_MUSA_RLC_PRODUCT_LAUNCH_COUNT"] = "1"
        os.environ["FLAGTREE_MUSA_RLC_MIN_ADJUSTED_SAVED_COST_PER_TENSOR_OP"] = "1800"
        os.environ["FLAGTREE_MUSA_RLC_PHASE3_SAVED_COST_MULTIPLIER"] = "2"
        os.environ.pop("FLAGTREE_MUSA_RLC_MAX_EXTERNAL_USE_EDGES", None)

        calibrated = FakeModule()
        musa_compiler._apply_musa_rlc_policy(calibrated)
        assert calibrated.attrs[
            "ttg.rlc-profitability-max-external-use-edges"] == 8

        os.environ["FLAGTREE_MUSA_RLC_MAX_EXTERNAL_USE_EDGES"] = "0"

        single_launch = FakeModule()
        musa_compiler._apply_musa_rlc_policy(single_launch)
        assert single_launch.attrs[
            "ttg.rlc-int-to-fp-vector-width-mask"] == 20
        assert single_launch.attrs["ttg.rlc-profitability-policy-enabled"] == 1
        assert single_launch.attrs["ttg.rlc-product-launch-count"] == 1
        assert single_launch.attrs[
            "ttg.rlc-profitability-min-adjusted-saved-cost-per-tensor-op"] == 1800
        assert single_launch.attrs[
            "ttg.rlc-profitability-phase3-saved-cost-multiplier"] == 2
        assert single_launch.attrs[
            "ttg.rlc-profitability-max-external-use-edges"] == 0

        os.environ["FLAGTREE_MUSA_RLC_PRODUCT_LAUNCH_COUNT"] = "0"
        unknown_launch = FakeModule()
        musa_compiler._apply_musa_rlc_policy(unknown_launch)
        assert unknown_launch.attrs["ttg.rlc-profitability-policy-enabled"] == 1
        assert "ttg.rlc-product-launch-count" not in unknown_launch.attrs
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
