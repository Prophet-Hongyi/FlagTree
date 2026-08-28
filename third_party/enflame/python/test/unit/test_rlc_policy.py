import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


_PASSES_DIR = Path(__file__).resolve().parents[3] / "backend" / "passes"
_REPO_ROOT = Path(__file__).resolve().parents[5]
_PACKAGE = "_flagtree_enflame_passes_test"
_package = ModuleType(_PACKAGE)
_package.__path__ = [str(_PASSES_DIR)]
sys.modules.setdefault(_PACKAGE, _package)


def _load_source_module(name):
    qualified_name = f"{_PACKAGE}.{name}"
    spec = importlib.util.spec_from_file_location(qualified_name, _PASSES_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    return module


rlc = _load_source_module("_rlc")
gcu300 = _load_source_module("gcu300")
gcu400 = _load_source_module("gcu400")
RLC_POLICY_FIELDS = rlc.RLC_POLICY_FIELDS
rlc_policy_signature = rlc.rlc_policy_signature
validate_rlc_arch = rlc.validate_rlc_arch


class _Pipeline:

    def __init__(self):
        self.calls = []

    def add_pass(self, name, options=""):
        self.calls.append((name, options))


def _policy(**overrides):
    values = {
        "rlc_enhance": False,
        "rlc_phase_mask": 0xF,
        **{name: 0 for name, _ in RLC_POLICY_FIELDS},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("module", [gcu300, gcu400])
def test_rlc_wrapper_forwards_default_off_and_phase_mask(module):
    pipeline = _Pipeline()
    module.add_tritongpu_remove_layout_conversions(pipeline)
    module.add_tritongpu_remove_layout_conversions(pipeline, True, 0x5)

    assert pipeline.calls == [
        ("tritongpu-remove-layout-conversions", "enable-rlc-enhance=false rlc-phase-mask=15"),
        ("tritongpu-remove-layout-conversions", "enable-rlc-enhance=true rlc-phase-mask=5"),
    ]


@pytest.mark.parametrize("mask", [-1, 16, 31])
def test_rlc_wrapper_rejects_unknown_phase_bits(mask):
    with pytest.raises(ValueError, match="phase mask"):
        gcu300.add_tritongpu_remove_layout_conversions(_Pipeline(), True, mask)


def test_rlc_policy_signature_binds_every_effective_value():
    baseline = rlc_policy_signature(_policy())
    assert baseline == "0-15-0-0-0-0-0-0-0"
    assert rlc_policy_signature(_policy(rlc_enhance=True)) != baseline
    assert rlc_policy_signature(_policy(rlc_phase_mask=5)) != baseline
    for name, _ in RLC_POLICY_FIELDS:
        assert rlc_policy_signature(_policy(**{name: 1})) != baseline


def test_rlc_policy_rejects_negative_cost_override():
    with pytest.raises(ValueError, match="non-negative"):
        rlc_policy_signature(_policy(rlc_convert_cost_per_byte=-1))


def test_rlc_arch_scope_is_fail_closed():
    for arch in ("gcu300", "gcu400", "gcu410"):
        validate_rlc_arch(arch, True)
    validate_rlc_arch("gcu500", False)
    with pytest.raises(ValueError, match="supports only"):
        validate_rlc_arch("gcu500", True)


def test_enflame_rlc_build_and_pipeline_contract_is_wired():
    cmake = (_REPO_ROOT / "cmake" / "FlagTreeOptions.cmake").read_text()
    spec = (_REPO_ROOT / "third_party" / "enflame" / "spec_cpp" / "include" / "triton" / "Dialect" /
            "TritonGPU" / "Transforms" / "Passes.td").read_text()
    compiler = (_REPO_ROOT / "third_party" / "enflame" / "backend" / "compiler.py").read_text()

    assert 'FLAGTREE_BACKEND STREQUAL "enflame"' in cmake
    assert "-D__FLAGTREE_RLC_ENHANCE__" in cmake
    assert '"enable-rlc-enhance"' in spec
    assert '"rlc-phase-mask"' in spec
    assert compiler.count("add_tritongpu_remove_layout_conversions(pm, options.rlc_enhance") == 4
