"""RLC knobs must change MetaX compile keys in the same process.

MThreads FlagGems 4→4 was a harness false negative when site `hash()`
omitted RLC and `@functools.lru_cache()` froze the first env. MetaX
follows that cache-key contract: live enhance/mask, no MUSA policy ints.
"""

from __future__ import annotations

import os

import pytest

from triton.backends.compiler import GPUTarget

compiler = pytest.importorskip("triton.backends.metax.compiler")
MACABackend = compiler.MACABackend
_rlc_policy_signature = compiler._rlc_policy_signature


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
