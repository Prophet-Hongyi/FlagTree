# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
# Copyright 2025-     FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Device-independent reference semantics for low-precision tests.

This module deliberately lives under ``python/test``.  It is an oracle for
backend tests, not a new public Triton API and not a backend implementation.
"""

from bisect import bisect_left
from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Iterable, Literal, Sequence


@dataclass(frozen=True)
class Float8Format:
    name: str
    exponent_bits: int
    mantissa_bits: int
    exponent_bias: int
    finite_only: bool
    max_finite_code: int
    canonical_nan_code: int


E4M3FN = Float8Format(
    name="e4m3fn",
    exponent_bits=4,
    mantissa_bits=3,
    exponent_bias=7,
    finite_only=True,
    max_finite_code=0x7E,
    canonical_nan_code=0x7F,
)

E5M2 = Float8Format(
    name="e5m2",
    exponent_bits=5,
    mantissa_bits=2,
    exponent_bias=15,
    finite_only=False,
    max_finite_code=0x7B,
    canonical_nan_code=0x7F,
)


def _validate_byte(value: int) -> int:
    value = int(value)
    if not 0 <= value <= 0xFF:
        raise ValueError(f"expected an 8-bit value, got {value}")
    return value


def decode_fp8(value: int, fmt: Float8Format) -> float:
    """Decode one physical FP8 byte according to ``fmt``."""

    value = _validate_byte(value)
    sign = -1.0 if value & 0x80 else 1.0
    magnitude = value & 0x7F
    mantissa_mask = (1 << fmt.mantissa_bits) - 1
    exponent_mask = (1 << fmt.exponent_bits) - 1
    mantissa = magnitude & mantissa_mask
    exponent = (magnitude >> fmt.mantissa_bits) & exponent_mask

    if exponent == exponent_mask:
        if fmt.finite_only:
            if mantissa == mantissa_mask:
                return math.copysign(math.nan, sign)
        elif mantissa == 0:
            return math.copysign(math.inf, sign)
        else:
            return math.copysign(math.nan, sign)

    if exponent == 0:
        if mantissa == 0:
            return math.copysign(0.0, sign)
        significand = mantissa / (1 << fmt.mantissa_bits)
        unbiased_exponent = 1 - fmt.exponent_bias
    else:
        significand = 1.0 + mantissa / (1 << fmt.mantissa_bits)
        unbiased_exponent = exponent - fmt.exponent_bias
    return sign * math.ldexp(significand, unbiased_exponent)


@lru_cache(maxsize=None)
def _positive_finite_values(fmt: Float8Format) -> tuple[float, ...]:
    return tuple(decode_fp8(code, fmt) for code in range(fmt.max_finite_code + 1))


def encode_fp8_rtne(value: float, fmt: Float8Format) -> int:
    """Encode with round-to-nearest-even and saturating overflow.

    This matches the software contract used by AMD's generic FP8 downcast:
    finite overflow and infinities clamp to the largest finite value, NaNs are
    canonicalized, and signed zero is preserved.
    """

    value = float(value)
    sign = 0x80 if math.copysign(1.0, value) < 0 else 0
    if math.isnan(value):
        return sign | fmt.canonical_nan_code

    magnitude = abs(value)
    if magnitude == 0.0:
        return sign

    candidates = _positive_finite_values(fmt)
    if math.isinf(magnitude) or magnitude >= candidates[-1]:
        return sign | fmt.max_finite_code

    upper = bisect_left(candidates, magnitude)
    if upper == 0:
        code = 0
    elif candidates[upper] == magnitude:
        code = upper
    else:
        lower = upper - 1
        lower_distance = magnitude - candidates[lower]
        upper_distance = candidates[upper] - magnitude
        if lower_distance < upper_distance:
            code = lower
        elif upper_distance < lower_distance:
            code = upper
        else:
            # Adjacent positive encodings are ordered by value.  At an exact
            # midpoint, the encoding whose retained significand LSB is zero is
            # the round-to-nearest-even result.
            code = lower if lower % 2 == 0 else upper
    return sign | code


def encode_fp8_rtz(value: float, fmt: Float8Format) -> int:
    """Encode with round-toward-zero.

    Finite values choose the largest representable magnitude no greater than
    the input magnitude.  Finite overflow therefore clamps to the largest
    finite value.  IEEE formats preserve infinity, finite-only formats clamp
    it, NaNs are canonicalized, and signed zero is preserved.

    For OCP E5M2 this is equivalent to AMD's software lowering: first convert
    FP32 to FP16 with RTZ, then retain the sign/exponent/top-two-mantissa byte.
    """

    value = float(value)
    sign = 0x80 if math.copysign(1.0, value) < 0 else 0
    if math.isnan(value):
        return sign | fmt.canonical_nan_code

    magnitude = abs(value)
    if magnitude == 0.0:
        return sign
    if math.isinf(magnitude):
        if fmt.finite_only:
            return sign | fmt.max_finite_code
        infinity_code = ((1 << fmt.exponent_bits) - 1) << fmt.mantissa_bits
        return sign | infinity_code

    candidates = _positive_finite_values(fmt)
    if magnitude >= candidates[-1]:
        return sign | fmt.max_finite_code

    upper = bisect_left(candidates, magnitude)
    code = upper if candidates[upper] == magnitude else max(0, upper - 1)
    return sign | code


def _encode_nibble(value: int, signed: bool) -> int:
    value = int(value)
    lower, upper = (-8, 7) if signed else (0, 15)
    if not lower <= value <= upper:
        kind = "INT4" if signed else "UINT4"
        raise ValueError(f"{kind} value {value} is outside [{lower}, {upper}]")
    return value & 0xF


def pack_nibbles(values: Iterable[int], *, signed: bool, pad_value: int = 0) -> bytes:
    """Pack INT4/UINT4 values low-nibble first, padding an odd tail."""

    nibbles = [_encode_nibble(value, signed) for value in values]
    pad = _encode_nibble(pad_value, signed)
    packed = bytearray()
    for index in range(0, len(nibbles), 2):
        low = nibbles[index]
        high = nibbles[index + 1] if index + 1 < len(nibbles) else pad
        packed.append(low | (high << 4))
    return bytes(packed)


def unpack_nibbles(packed: Iterable[int], *, count: int, signed: bool) -> list[int]:
    """Unpack exactly ``count`` low-nibble-first INT4/UINT4 values."""

    packed_bytes = [_validate_byte(value) for value in packed]
    if count < 0 or count > 2 * len(packed_bytes):
        raise ValueError(f"count {count} does not fit {len(packed_bytes)} packed bytes")

    result: list[int] = []
    for value in packed_bytes:
        result.extend((value & 0xF, value >> 4))
    result = result[:count]
    if signed:
        result = [value - 16 if value & 0x8 else value for value in result]
    return result


RoundingMode = Literal["rtne", "rtz"]


def _round(value: float, rounding: RoundingMode) -> int:
    if not math.isfinite(value):
        raise ValueError("quantization input must be finite")
    if rounding == "rtne":
        return round(value)
    if rounding == "rtz":
        return math.trunc(value)
    raise ValueError(f"unsupported rounding mode {rounding!r}")


def quantize_affine(
    values: Sequence[float],
    *,
    scale: float,
    zero_point: int,
    qmin: int,
    qmax: int,
    rounding: RoundingMode = "rtne",
) -> list[int]:
    """Reference affine quantization using ``scale``, not inverse scale."""

    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and greater than zero")
    if qmin >= qmax:
        raise ValueError("qmin must be smaller than qmax")
    if not qmin <= zero_point <= qmax:
        raise ValueError("zero_point must be within the quantized range")

    result = []
    for value in values:
        quantized = _round(float(value) / scale, rounding) + zero_point
        result.append(min(qmax, max(qmin, quantized)))
    return result


def dequantize_affine(values: Sequence[int], *, scale: float, zero_point: int) -> list[float]:
    """Reference inverse of :func:`quantize_affine`."""

    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and greater than zero")
    return [(int(value) - int(zero_point)) * scale for value in values]
