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

from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class LowPrecisionMode(str, Enum):
    UNSUPPORTED = "unsupported"
    SOFTWARE = "software"
    NATIVE = "native"


_A5_ARCHES = frozenset(
    {
        "Ascend910_9579",
        "Ascend910_9581",
        "Ascend910_9589",
        "Ascend910_9599",
        "Ascend950PR_957c",
    }
)

# Preserve the existing A5 frontend contract. This tuple controls dtype
# admission only; binary acceptance and device support remain separate gates.
_A5_DIRECT_FP8_DTYPES = (
    "fp8e5",
    "fp8e4b15",
    "fp8e4nv",
    "fp8e4b8",
    "fp8e5b16",
)


@dataclass(frozen=True)
class LowPrecisionTargetFeatures:
    """Compiler-routing facts for one exact Ascend target string.

    Keep this object intentionally narrow. A feature is added only when its
    target selector has a source and validation contract. Unknown or near-match
    targets must stay unsupported rather than inherit features by prefix.
    """

    target_name: str
    direct_fp8_dtype_mode: LowPrecisionMode
    supported_fp8_dtypes: Tuple[str, ...]


def get_low_precision_target_features(arch: str) -> LowPrecisionTargetFeatures:
    if str(arch) in _A5_ARCHES:
        return LowPrecisionTargetFeatures(
            target_name=str(arch),
            direct_fp8_dtype_mode=LowPrecisionMode.NATIVE,
            supported_fp8_dtypes=_A5_DIRECT_FP8_DTYPES,
        )
    return LowPrecisionTargetFeatures(
        target_name=str(arch) or "unknown",
        direct_fp8_dtype_mode=LowPrecisionMode.UNSUPPORTED,
        supported_fp8_dtypes=(),
    )
