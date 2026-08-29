from types import SimpleNamespace

import pytest

from triton.backends.hcu.compiler_hcu import HIPBackend


def _options(**overrides):
    values = {
        "arch": "gfx936",
        "sched_latency": "none",
        "wdra_enabled": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_clang17_uses_only_portable_codegen_args(monkeypatch):
    monkeypatch.setattr(HIPBackend, "_clang_major", staticmethod(lambda: 17))

    assert HIPBackend._get_clang_args({}, _options()) == [
        "-target",
        "amdgcn-amd-amdhsa",
        "-mcpu=gfx936:xnack-",
        "-O3",
    ]


@pytest.mark.parametrize(
    ("overrides", "contract"),
    [
        ({"wdra_enabled": True}, "wdra_enabled"),
        ({"sched_latency": "mmac5-ds6"}, "sched_latency=mmac5-ds6"),
    ],
)
def test_clang17_rejects_unavailable_tuning_options(monkeypatch, overrides, contract):
    monkeypatch.setattr(HIPBackend, "_clang_major", staticmethod(lambda: 17))

    with pytest.raises(ValueError, match=contract):
        HIPBackend._get_clang_args({}, _options(**overrides))


def test_clang18_preserves_hcu_tuning_args(monkeypatch):
    monkeypatch.setattr(HIPBackend, "_clang_major", staticmethod(lambda: 18))

    args = HIPBackend._get_clang_args({}, _options())

    assert "-mllvm=-check-valu-data-forward-hazards=0" in args
    assert "-mllvm=-disable-cluster-lds-memops=true" in args
    assert "-mllvm=-hcu-pre-emit-load-store-opt=false" in args
    assert "-mllvm=-support-768-vgprs=true" in args
    assert "-mllvm=-enable-hcu-approx-func-fp-math=true" in args
    assert "-mllvm=-hcu-update-wait-by-reverse-search=true" in args
