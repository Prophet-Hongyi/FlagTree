from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[5]
RLC_HELPER = REPO_ROOT / "third_party" / "sunrise" / "backend" / "_rlc.py"
SPEC = importlib.util.spec_from_file_location("_sunrise_rlc_test", RLC_HELPER)
RLC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RLC)


def policy(**overrides):
    values = {
        "rlc_enhance": False,
        "rlc_phase_mask": 0xF,
        **{name: 0 for name, _ in RLC.RLC_POLICY_FIELDS},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class SunriseRlcPolicyTests(unittest.TestCase):
    def test_phase_mask_and_arch_scope_fail_closed(self):
        self.assertEqual(RLC.validate_rlc_phase_mask(0xF), 0xF)
        for mask in (-1, 16, 31):
            with self.subTest(mask=mask), self.assertRaisesRegex(ValueError, "phase mask"):
                RLC.validate_rlc_phase_mask(mask)
        for arch in ("S2", "s3"):
            self.assertIn(RLC.validate_rlc_arch(arch, True), {"s2", "s3"})
        RLC.validate_rlc_arch("future", False)
        with self.assertRaisesRegex(ValueError, "supports only"):
            RLC.validate_rlc_arch("future", True)

    def test_policy_signature_binds_every_effective_override(self):
        baseline = RLC.rlc_policy_signature(policy())
        self.assertEqual(baseline, "0-15-0-0-0-0-0-0-0")
        self.assertNotEqual(RLC.rlc_policy_signature(policy(rlc_enhance=True)), baseline)
        self.assertNotEqual(RLC.rlc_policy_signature(policy(rlc_phase_mask=5)), baseline)
        for name, _ in RLC.RLC_POLICY_FIELDS:
            with self.subTest(name=name):
                self.assertNotEqual(RLC.rlc_policy_signature(policy(**{name: 1})), baseline)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            RLC.rlc_policy_signature(policy(rlc_convert_cost_per_byte=-1))

    def test_common_algorithm_preserves_sunrise_legality_and_pipeline(self):
        cmake = (REPO_ROOT / "cmake" / "FlagTreeOptions.cmake").read_text(encoding="utf-8")
        common = (
            REPO_ROOT / "lib" / "Dialect" / "TritonGPU" / "Transforms" /
            "RemoveLayoutConversions.cpp"
        ).read_text(encoding="utf-8")
        shim = (
            REPO_ROOT / "third_party" / "sunrise" / "spec_cpp" / "lib" /
            "Dialect" / "TritonGPU" / "Transforms" / "RemoveLayoutConversions.cpp"
        ).read_text(encoding="utf-8")
        compiler = (
            REPO_ROOT / "third_party" / "sunrise" / "backend" / "compiler.py"
        ).read_text(encoding="utf-8")

        self.assertIn('FLAGTREE_BACKEND STREQUAL "sunrise"', cmake)
        self.assertIn("-D__FLAGTREE_RLC_ENHANCE__", cmake)
        self.assertIn("ttg.rlc-preserve-narrowing-trunc-layouts", common)
        self.assertIn("srcBits == 32 && dstBits == 16", common)
        self.assertIn("dstBits == 8", common)
        self.assertIn("__FLAGTREE_SUNRISE_RLC__", shim)
        self.assertIn("RemoveLayoutConversions.cpp", shim)
        self.assertEqual(
            compiler.count(
                "add_remove_layout_conversions(pm, opt.rlc_enhance, opt.rlc_phase_mask)"
            ),
            2,
        )
        self.assertIn("_rlc_policy_signature()", compiler)
        self.assertIn("rlc_preserve_narrowing_trunc_layouts", compiler)


if __name__ == "__main__":
    unittest.main()
