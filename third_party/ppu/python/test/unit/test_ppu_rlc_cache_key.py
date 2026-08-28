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
_RLC_ENV_KEYS = ("FLAGTREE_PPU_RLC_ENHANCE", "FLAGTREE_PPU_RLC_PHASE_MASK")


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


def test_installed_jit_cache_key_includes_backend_identity():
    source = inspect.getsource(jit.JITFunction.run)
    assert 'key = f"{key}-{backend.hash()}"' in source
