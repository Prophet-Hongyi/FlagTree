# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import importlib.util
from pathlib import Path
import sys
import unittest

_MODULE_PATH = Path(__file__).parents[2] / "backend" / "target_features.py"
_SPEC = importlib.util.spec_from_file_location("ascend_target_features", _MODULE_PATH)
target_features = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = target_features
_SPEC.loader.exec_module(target_features)


class LowPrecisionTargetFeaturesTest(unittest.TestCase):

    def test_exact_a5_targets_admit_existing_direct_fp8_dtypes(self):
        for arch in (
            "Ascend910_9579",
            "Ascend910_9581",
            "Ascend910_9589",
            "Ascend910_9599",
            "Ascend950PR_957c",
        ):
            with self.subTest(arch=arch):
                features = target_features.get_low_precision_target_features(arch)
                self.assertEqual(features.target_name, arch)
                self.assertIs(
                    features.direct_fp8_dtype_mode,
                    target_features.LowPrecisionMode.NATIVE,
                )
                self.assertEqual(
                    features.supported_fp8_dtypes,
                    (
                        "fp8e5",
                        "fp8e4b15",
                        "fp8e4nv",
                        "fp8e4b8",
                        "fp8e5b16",
                    ),
                )

    def test_non_a5_and_near_match_targets_fail_closed(self):
        for arch in (
            "",
            "Ascend910B1",
            "Ascend910_9382",
            "Ascend910_95",
            "Ascend910_9589-extra",
            "Ascend950",
            "Ascend950PR_unknown",
            "unknown",
        ):
            with self.subTest(arch=arch):
                features = target_features.get_low_precision_target_features(arch)
                self.assertEqual(features.target_name, arch or "unknown")
                self.assertIs(
                    features.direct_fp8_dtype_mode,
                    target_features.LowPrecisionMode.UNSUPPORTED,
                )
                self.assertEqual(features.supported_fp8_dtypes, ())


if __name__ == "__main__":
    unittest.main()
