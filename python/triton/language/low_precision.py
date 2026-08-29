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

from __future__ import annotations

from ..runtime.jit import jit
from . import core, math


@jit
def _validate_quantization_scale(scale):
    if isinstance(scale, core.tensor):
        core.device_assert(
            (scale > 0) & (scale < float("inf")) & (scale == scale),
            "scale must be finite and greater than zero",
        )
    else:
        core.static_assert(scale > 0 and scale < float("inf"), "scale must be finite and greater than zero")
    return scale


@jit
def _validate_zero_point(zero_point, qmin: core.constexpr, qmax: core.constexpr):
    if isinstance(zero_point, core.tensor):
        core.static_assert(zero_point.dtype.is_int(), "zero_point must have an integer dtype")
        core.device_assert(
            (zero_point >= qmin) & (zero_point <= qmax),
            "zero_point must be within [qmin, qmax]",
        )
    else:
        core.static_assert(qmin <= zero_point and zero_point <= qmax, "zero_point must be within [qmin, qmax]")
    return zero_point


@jit
def _round_quantized(input, rounding: core.constexpr):
    core.static_assert(rounding == "rtne" or rounding == "rtz", "rounding must be 'rtne' or 'rtz'")
    if rounding == "rtz":
        return core.where(input < 0, math.ceil(input), math.floor(input))

    lower = math.floor(input)
    fraction = input - lower
    lower_is_even = (lower.to(core.int32) & 1) == 0
    return core.where(
        fraction < 0.5,
        lower,
        core.where(fraction > 0.5, lower + 1.0, core.where(lower_is_even, lower, lower + 1.0)),
    )


@jit
def quantize(
    input,
    scale,
    dtype: core.constexpr,
    zero_point=0,
    rounding: core.constexpr = "rtne",
    qmin: core.constexpr = None,
    qmax: core.constexpr = None,
):
    """Apply explicit affine integer or scaled FP8 quantization.

    ``scale`` and ``zero_point`` are values, so callers choose per-tensor,
    per-row, per-channel, or per-group granularity through normal Triton
    broadcasting. This function applies parameters; it never derives them.

    Integer quantization is
    ``clamp(round(input / scale) + zero_point, qmin, qmax)``. Version 1
    intentionally supports only INT8 and UINT8 physical outputs. A narrower
    logical range such as INT4 can be requested with explicit ``qmin`` and
    ``qmax`` before a separate packing step.

    FP8 quantization is a scaled cast. FP8 has no zero point or custom integer
    range, so ``zero_point`` must be zero and ``qmin``/``qmax`` must be absent.
    Backend FP8 dtype and rounding support is still checked by the existing
    cast lowering.
    """

    core.static_assert(input.dtype.is_floating(), "quantize input must have a floating-point dtype")
    core.static_assert(rounding == "rtne" or rounding == "rtz", "rounding must be 'rtne' or 'rtz'")
    scale = _validate_quantization_scale(scale)

    if dtype.is_fp8():
        core.static_assert(qmin is None and qmax is None, "FP8 quantization does not accept qmin or qmax")
        if isinstance(zero_point, core.tensor):
            core.static_assert(zero_point.dtype.is_int(), "zero_point must have an integer dtype")
            core.device_assert(zero_point == 0, "FP8 quantization requires zero_point=0")
        else:
            core.static_assert(zero_point == 0, "FP8 quantization requires zero_point=0")
        return (input.to(core.float32) / scale).to(dtype, fp_downcast_rounding=rounding)
    else:
        core.static_assert(dtype.is_int8() or dtype.is_uint8(), "integer quantize dtype must be tl.int8 or tl.uint8")
        dtype_min: core.constexpr = dtype.get_int_min_value()
        dtype_max: core.constexpr = dtype.get_int_max_value()
        _qmin: core.constexpr = dtype_min if qmin is None else qmin
        _qmax: core.constexpr = dtype_max if qmax is None else qmax
        core.static_assert(dtype_min <= _qmin and _qmin < _qmax and _qmax <= dtype_max,
                           "qmin and qmax must be ordered and within the output dtype range")
        zero_point = _validate_zero_point(zero_point, _qmin, _qmax)

        scaled = input.to(core.float32) / scale
        scaled = core.where(scaled != scaled, 0.0, scaled)
        scaled = core.maximum(_qmin - zero_point, core.minimum(scaled, _qmax - zero_point))
        rounded = _round_quantized(scaled, rounding)
        return core.maximum(_qmin, core.minimum(rounded + zero_point, _qmax)).to(dtype)


@jit
def dequantize(input, scale, dtype: core.constexpr = core.float32, zero_point=0):
    """Apply explicit affine integer or scaled FP8 dequantization.

    Integer inputs are widened to FP32 before subtracting ``zero_point`` so the
    subtraction cannot overflow in INT8/UINT8. FP8 inputs require a zero zero
    point and use the existing backend cast before multiplying by ``scale``.
    """

    core.static_assert(dtype.is_standard_floating(), "dequantize dtype must be FP16, BF16, FP32, or FP64")
    core.static_assert(input.dtype.is_int8() or input.dtype.is_uint8() or input.dtype.is_fp8(),
                       "dequantize input must be INT8, UINT8, or FP8")
    scale = _validate_quantization_scale(scale)

    if input.dtype.is_fp8():
        if isinstance(zero_point, core.tensor):
            core.static_assert(zero_point.dtype.is_int(), "zero_point must have an integer dtype")
            core.device_assert(zero_point == 0, "FP8 dequantization requires zero_point=0")
        else:
            core.static_assert(zero_point == 0, "FP8 dequantization requires zero_point=0")
        return (input.to(core.float32) * scale).to(dtype)
    else:
        input_fp32 = input.to(core.float32)
        return ((input_fp32 - zero_point) * scale).to(dtype)
