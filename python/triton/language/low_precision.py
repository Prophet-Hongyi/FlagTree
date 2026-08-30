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
    _validate_quantization_scale(scale)

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
        _validate_zero_point(zero_point, _qmin, _qmax)

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
    _validate_quantization_scale(scale)

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


@jit
def encode_fp8(
    input,
    format: core.constexpr = "e4m3fn",
    rounding: core.constexpr = "rtne",
    overflow: core.constexpr = "satfinite",
):
    """Encode floating-point values into one-byte FP8 storage.

    The return type is always ``tl.uint8``. It is a physical storage carrier,
    not a native FP8 tensor, so backends can implement storage and software
    dequantization without claiming native FP8 arithmetic.

    Supported formats are OCP E4M3FN/E5M2 and AMD E4M3FNUZ/E5M2FNUZ. All use
    round-to-nearest-even, saturating finite overflow, and FP16, BF16, or FP32
    input. The FNUZ formats have one unsigned zero and map NaN or infinity to
    their sole NaN encoding, ``0x80``. Unsupported combinations fail at compile
    time instead of silently changing numeric semantics.
    """

    core.static_assert(
        input.dtype.is_fp16() or input.dtype.is_bf16() or input.dtype.is_fp32(),
        "encode_fp8 input must be tl.float16, tl.bfloat16, or tl.float32",
    )
    core.static_assert(
        format == "e4m3fn" or format == "e4m3fnuz" or format == "e5m2" or format == "e5m2fnuz",
        "encode_fp8 format must be 'e4m3fn', 'e4m3fnuz', 'e5m2', or 'e5m2fnuz'",
    )
    core.static_assert(rounding == "rtne", "encode_fp8 rounding must be 'rtne'")
    core.static_assert(overflow == "satfinite", "encode_fp8 overflow must be 'satfinite'")

    # AMD's software FP16 -> E5M2FNUZ conversion first extends to FP32. The
    # extension is exact for both FP16 and BF16, and avoids inventing a second
    # 16-bit subnormal-rounding implementation that upstream does not use.
    if format == "e5m2fnuz":
        input = input.to(core.float32)

    if input.dtype.is_fp32():
        bits = input.to(core.int32, bitcast=True)
        absolute = bits & 0x7FFFFFFF
        sign = ((((bits >> 31) & 1) << 7).to(core.uint8))
        exponent = (absolute >> 23) & 0xFF
        mantissa = absolute & 0x007FFFFF
        is_nan = (exponent == 0xFF) & (mantissa != 0)
        is_nan_or_inf = exponent == 0xFF

        if format == "e4m3fn":
            # AMD's generic software conversion reduces the FP32 mantissa from
            # 23 bits to three with RTNE, rebases the exponent from 127 to 7,
            # and handles the eight E4M3FN subnormal bins explicitly.
            rounding_absolute = core.where(is_nan, 0x7F800000, absolute)
            rounding_bias = ((rounding_absolute & 0x00100000) >> 20) + 0x0007FFFF
            rounded = (rounding_absolute + rounding_bias) & 0x7FF00000
            rounded = core.maximum(rounded, 0x3C800000)
            encoded = (rounded - 0x3C000000) >> 20
            encoded = core.where(absolute > 0x43E7FFFF, 0x7E, encoded)

            halfway_points: core.constexpr = (
                0x3A800000,
                0x3B400000,
                0x3BA00000,
                0x3BE00000,
                0x3C100000,
                0x3C300000,
                0x3C500000,
                0x3C700000,
            )
            for i in core.static_range(7, -1, -1):
                if i % 2 == 0:
                    below_halfway = absolute <= halfway_points[i]
                else:
                    below_halfway = absolute < halfway_points[i]
                encoded = core.where(below_halfway, i, encoded)
        elif format == "e4m3fnuz":
            # Port AMD's generic E4M3FNUZ RTNE conversion. FNUZ increases the
            # exponent bias from 7 to 8, reserves 0x80 as its sole NaN, and
            # has only one zero. Finite overflow saturates at 0x7f (240).
            rounding_absolute = core.where(is_nan_or_inf, 0x7F800000, absolute)
            rounding_bias = ((rounding_absolute & 0x00100000) >> 20) + 0x0007FFFF
            rounded = (rounding_absolute + rounding_bias) & 0x7FF00000
            rounded = core.maximum(rounded, 0x3C000000)
            encoded = (rounded - 0x3B800000) >> 20
            encoded = core.where(absolute > 0x43700000, 0x7F, encoded)

            halfway_points: core.constexpr = (
                0x3A000000,
                0x3AC00000,
                0x3B200000,
                0x3B600000,
                0x3B900000,
                0x3BB00000,
                0x3BD00000,
                0x3BF00000,
            )
            for i in core.static_range(7, -1, -1):
                if i % 2 == 0:
                    below_halfway = absolute <= halfway_points[i]
                else:
                    below_halfway = absolute < halfway_points[i]
                encoded = core.where(below_halfway, i, encoded)
        elif format == "e5m2fnuz":
            # Port AMD's generic E5M2FNUZ RTNE conversion. The exponent bias
            # is 16, all exponent patterns are finite, 0x80 is the sole NaN,
            # and finite overflow saturates at 0x7f (57344).
            rounding_absolute = core.where(is_nan_or_inf, 0x7F800000, absolute)
            rounding_bias = ((rounding_absolute & 0x00200000) >> 21) + 0x000FFFFF
            # The sign bit was removed above, so 0x7fe00000 is equivalent to
            # AMD's unsigned 0xffe00000 mask without forcing vendor frontends
            # to materialize an unsigned 32-bit constant.
            rounded = (rounding_absolute + rounding_bias) & 0x7FE00000
            rounded = core.maximum(rounded, 0x38000000)
            encoded = (rounded - 0x37800000) >> 21
            encoded = core.where(absolute > 0x47600000, 0x7F, encoded)

            halfway_points: core.constexpr = (
                0x36800000,
                0x37400000,
                0x37A00000,
                0x37E00000,
            )
            for i in core.static_range(3, -1, -1):
                if i % 2 == 0:
                    below_halfway = absolute <= halfway_points[i]
                else:
                    below_halfway = absolute < halfway_points[i]
                encoded = core.where(below_halfway, i, encoded)
        else:
            # Port AMD's direct FP32 -> OCP E5M2 software path. Mapping the
            # exponent and subnormal bins before reducing the 23-bit mantissa
            # avoids the double rounding of an FP16 intermediate.
            exponent5 = core.where(exponent < 0x71, 0, exponent - 0x70)
            adjusted_mantissa = core.where(exponent < 0x6E, 0, mantissa)
            adjusted_mantissa = core.where(
                exponent == 0x6E,
                core.where(mantissa != 0, 0x00200000, 0),
                adjusted_mantissa,
            )
            adjusted_mantissa = core.where(
                exponent == 0x6F,
                core.where(mantissa >= 0x00400000, 0x00400000, 0x00200000),
                adjusted_mantissa,
            )
            adjusted_mantissa = core.where(
                exponent == 0x70,
                core.where(
                    mantissa > 0x00200000,
                    core.where(mantissa >= 0x00600000, 0x00800000, 0x00600000),
                    0x00400000,
                ),
                adjusted_mantissa,
            )
            significand = (exponent5 << 23) | adjusted_mantissa
            rounding_bias = ((significand & 0x00200000) >> 21) + 0x000FFFFF
            rounded = significand + rounding_bias
            rounded = core.where(
                (exponent > 0x8E) | (significand >= 0x0F700000),
                0x0F7FFFFF,
                rounded,
            )
            encoded = rounded >> 21

        if format == "e4m3fnuz" or format == "e5m2fnuz":
            encoded = core.where(is_nan_or_inf, 0x80, encoded.to(core.int32)).to(core.uint8)
            is_zero = (encoded == 0) & (is_nan_or_inf == 0)
            encoded = (encoded | sign).to(core.uint8)
            return core.where(is_zero, 0, encoded).to(core.uint8)
        else:
            encoded = core.where(is_nan, 0x7F, encoded).to(core.uint8)
            return (encoded | sign).to(core.uint8)

    else:
        # Preserve NaN and sign information from the original source. BF16
        # values at or above the smallest FP8 rounding midpoint are exactly
        # representable in FP16; smaller values encode as signed zero. Larger
        # values may safely overflow to FP16 infinity because this contract
        # saturates FP8 overflow.
        source_bits = input.to(core.int16, bitcast=True).to(core.int32) & 0xFFFF
        source_absolute = source_bits & 0x7FFF
        sign = ((source_bits & 0x8000) >> 8).to(core.uint8)
        if input.dtype.is_bf16():
            is_nan = ((source_absolute & 0x7F80) == 0x7F80) & ((source_absolute & 0x007F) != 0)
            # The top 128 absolute BF16 encodings are exactly NaN/Inf. Adding
            # 0x80 makes only that interval cross bit 15. Keep this as an
            # integer bit instead of a comparison: some vendor LLVM pipelines
            # otherwise replace the test with an unreliable BF16 comparison.
            fnuz_special_bit = ((source_absolute + 0x0080) >> 15) & 1
            input = input.to(core.float16)
        else:
            is_nan = ((source_absolute & 0x7C00) == 0x7C00) & ((source_absolute & 0x03FF) != 0)
            fnuz_special_bit = ((source_absolute + 0x0400) >> 15) & 1

        # Use a non-negative int32 carrier. Some backends cannot lower unsigned
        # 32-bit arithmetic without promoting it to an unsupported uint64 cast.
        bits = input.to(core.int16, bitcast=True).to(core.int32) & 0xFFFF
        absolute = bits & 0x7FFF

        if format == "e4m3fn":
            rounding_bias = ((absolute & 0x0080) >> 7) + 0x003F
            encoded = (absolute + rounding_bias) & 0xFF80
            encoded = core.maximum(encoded, 0x2400)
            encoded = ((encoded - 0x2000) >> 7).to(core.uint8)
            encoded = core.where(absolute > 0x5F40, 0x7E, encoded)

            halfway_points: core.constexpr = (
                0x1400,
                0x1A00,
                0x1D00,
                0x1F00,
                0x2080,
                0x2180,
                0x2280,
                0x2380,
            )
            for i in core.static_range(7, -1, -1):
                if i % 2 == 0:
                    below_halfway = absolute <= halfway_points[i]
                else:
                    below_halfway = absolute < halfway_points[i]
                encoded = core.where(below_halfway, i, encoded)
        elif format == "e4m3fnuz":
            rounding_bias = ((absolute & 0x0080) >> 7) + 0x003F
            encoded = (absolute + rounding_bias) & 0xFF80
            encoded = core.maximum(encoded, 0x2000)
            encoded = ((encoded - 0x1C00) >> 7).to(core.uint8)
            encoded = core.where(absolute > 0x5B80, 0x7F, encoded)

            halfway_points: core.constexpr = (
                0x1000,
                0x1600,
                0x1900,
                0x1B00,
                0x1C80,
                0x1D80,
                0x1E80,
                0x1F80,
            )
            for i in core.static_range(7, -1, -1):
                if i % 2 == 0:
                    below_halfway = absolute <= halfway_points[i]
                else:
                    below_halfway = absolute < halfway_points[i]
                encoded = core.where(below_halfway, i, encoded)
        else:
            # E5M2 and FP16 share the same five-bit exponent. Round the FP16
            # mantissa from ten bits to two, clamp overflow and infinities to
            # the largest finite E5M2 value, then retain the high byte.
            rounding_bias = ((absolute & 0x0100) >> 8) + 0x007F
            encoded = absolute + rounding_bias
            encoded = core.where(absolute >= 0x7B80, 0x7B00, encoded)
            encoded = (encoded >> 8).to(core.uint8)

        if format == "e4m3fnuz":
            encoded_i32 = encoded.to(core.int32)
            encoded = (encoded_i32 * (1 - fnuz_special_bit) + 0x80 * fnuz_special_bit).to(core.uint8)
            sign = core.where(encoded == 0, 0, sign).to(core.uint8)
            return (encoded | sign).to(core.uint8)
        else:
            encoded = core.where(is_nan, 0x7F, encoded)
            return (encoded | sign).to(core.uint8)


@jit
def decode_fp8(input, format: core.constexpr = "e4m3fn", dtype: core.constexpr = core.float16):
    """Decode one-byte FP8 storage into FP16, BF16, or FP32 values.

    ``input`` must be ``tl.uint8`` produced by :func:`encode_fp8` or an
    equivalent E4M3FN, E4M3FNUZ, E5M2, or E5M2FNUZ storage source. No native
    FP8 IR type is introduced.
    """

    core.static_assert(input.dtype.is_uint8(), "decode_fp8 input must be tl.uint8 storage")
    core.static_assert(
        format == "e4m3fn" or format == "e4m3fnuz" or format == "e5m2" or format == "e5m2fnuz",
        "decode_fp8 format must be 'e4m3fn', 'e4m3fnuz', 'e5m2', or 'e5m2fnuz'",
    )
    core.static_assert(
        dtype.is_fp16() or dtype.is_bf16() or dtype.is_fp32(),
        "decode_fp8 dtype must be tl.float16, tl.bfloat16, or tl.float32",
    )

    values_i32 = input.to(core.int32)
    if format == "e4m3fn":
        absolute = values_i32 & 0x7F
        sign = (values_i32 & 0x80) << 8
        fp16_bits = ((values_i32 << 8) & 0x7FFF) >> 1
        fp16_bits += 0x2000
        fp16_bits = core.where(absolute == 0x7F, 0x7E00, fp16_bits)

        denormals_and_zero: core.constexpr = (
            0x0000,
            0x1800,
            0x1C00,
            0x1E00,
            0x2000,
            0x2100,
            0x2200,
            0x2300,
        )
        for i in core.static_range(8):
            fp16_bits = core.where(absolute == i, denormals_and_zero[i], fp16_bits)
        fp16_bits |= sign
    elif format == "e4m3fnuz":
        absolute = values_i32 & 0x7F
        sign = (values_i32 & 0x80) << 8
        fp16_bits = ((values_i32 << 8) & 0x7FFF) >> 1
        fp16_bits += 0x1C00

        denormals_and_zero: core.constexpr = (
            0x0000,
            0x1400,
            0x1800,
            0x1A00,
            0x1C00,
            0x1D00,
            0x1E00,
            0x1F00,
        )
        for i in core.static_range(8):
            fp16_bits = core.where(absolute == i, denormals_and_zero[i], fp16_bits)
        fp16_bits |= sign
        fp16_bits = core.where(values_i32 == 0x80, 0x7E00, fp16_bits)
    elif format == "e5m2fnuz":
        # Port AMD's software E5M2FNUZ -> FP16 conversion. Normal encodings
        # decrement the exponent to account for the 16 -> 15 bias change;
        # exponent-zero and exponent-one values need separate subnormal paths.
        source_bits = values_i32 << 8
        exponent = source_bits & 0x7C00
        mantissa = source_bits & 0x0300
        sign = source_bits & 0x8000

        shifted_mantissa = mantissa >> 1
        exponent_zero_bits = sign | shifted_mantissa
        debiased_exponent = core.maximum(exponent, 0x0400) - 0x0400
        normal_bits = sign | mantissa | debiased_exponent
        exponent_one_bits = sign | shifted_mantissa | 0x0200
        fp16_bits = core.where(exponent == 0, exponent_zero_bits, normal_bits)
        fp16_bits = core.where(exponent == 0x0400, exponent_one_bits, fp16_bits)
        fp16_bits = core.where(values_i32 == 0x80, 0x7E00, fp16_bits)
    else:
        # E5M2 is a high-byte projection of FP16: sign and exponent widths are
        # identical and the two stored mantissa bits occupy FP16 bits 9-8.
        fp16_bits = values_i32 << 8

    decoded = fp16_bits.to(core.int16).to(core.float16, bitcast=True)
    return decoded.to(dtype)


@jit
def _fp8_storage_sign_bit(input):
    if input.dtype.is_fp32():
        source_bits = input.to(core.int32, bitcast=True)
        return (((source_bits >> 31) & 1) << 7).to(core.uint8)
    else:
        source_bits = input.to(core.int16, bitcast=True).to(core.int32) & 0xFFFF
        return ((source_bits & 0x8000) >> 8).to(core.uint8)


@jit
def quantize_fp8(
    input,
    scale,
    format: core.constexpr = "e4m3fn",
    rounding: core.constexpr = "rtne",
    overflow: core.constexpr = "satfinite",
):
    """Apply an explicit scale and encode one-byte FP8 storage.

    The return type is always ``tl.uint8``. ``scale`` follows normal Triton
    broadcasting and must be finite and positive. The caller chooses the
    scale granularity and owns any amax reduction, block layout, padding, or
    scale storage policy. This primitive deliberately does not create a native
    FP8 tensor or imply native FP8 arithmetic support.
    """

    core.static_assert(
        input.dtype.is_fp16() or input.dtype.is_bf16() or input.dtype.is_fp32(),
        "quantize_fp8 input must be tl.float16, tl.bfloat16, or tl.float32",
    )
    _validate_quantization_scale(scale)
    normalized = input.to(core.float32) / scale
    encoded = encode_fp8(normalized, format=format, rounding=rounding, overflow=overflow)

    # Positive scaling cannot change the source sign, but some vendor divide
    # lowerings canonicalize -0 or a NaN to a positive value. Restore the sign
    # for OCP storage. FNUZ retains its sole 0x80 NaN and unsigned zero while
    # finite nonzero values keep the original sign.
    sign = _fp8_storage_sign_bit(input)
    if format == "e4m3fnuz" or format == "e5m2fnuz":
        is_zero_or_nan = (encoded == 0) | (encoded == 0x80)
        signed = ((encoded & 0x7F) | sign).to(core.uint8)
        return core.where(is_zero_or_nan, encoded, signed).to(core.uint8)
    return ((encoded & 0x7F) | sign).to(core.uint8)


@jit
def dequantize_fp8(
    packed,
    scale,
    format: core.constexpr = "e4m3fn",
    dtype: core.constexpr = core.float32,
):
    """Decode one-byte FP8 storage and apply an explicit scale.

    The output has the same shape as ``packed``. Scale derivation, logical
    axis layout, and block-tail handling remain caller policy. The operation
    reuses the software storage codec and therefore does not require or claim
    a backend-native FP8 scalar type or conversion instruction.
    """

    core.static_assert(packed.dtype.is_uint8(), "dequantize_fp8 input must be tl.uint8 storage")
    core.static_assert(
        dtype.is_fp16() or dtype.is_bf16() or dtype.is_fp32(),
        "dequantize_fp8 dtype must be tl.float16, tl.bfloat16, or tl.float32",
    )
    _validate_quantization_scale(scale)
    decoded = decode_fp8(packed, format=format, dtype=dtype)
    scaled = (decoded.to(core.float32) * scale).to(dtype)
    # Preserve the decoded signed-zero payload even when a vendor multiply
    # canonicalizes it. FNUZ decode already represents its only zero as +0.
    return core.where(decoded == 0.0, decoded, scaled)


@jit
def _fp4_e2m1_sign_bit(input):
    # Read the sign from the original storage before widening. Some targets
    # canonicalize a FP16/BF16 NaN while converting it to FP32, which may lose
    # the source sign even though this storage codec has deterministic signed
    # saturation semantics.
    if input.dtype.is_fp32():
        source_bits = input.to(core.uint32, bitcast=True)
        return ((source_bits >> 31) << 3).to(core.uint8)
    else:
        source_bits16 = input.to(core.int16, bitcast=True).to(core.int32) & 0xFFFF
        return ((source_bits16 & 0x8000) >> 12).to(core.uint8)


@jit
def _encode_fp4_e2m1_value(input):
    sign = _fp4_e2m1_sign_bit(input)
    if input.dtype.is_fp32():
        source = input
        source_bits = input.to(core.uint32, bitcast=True)
    else:
        source = input.to(core.float32)
        source_bits = source.to(core.uint32, bitcast=True)
    absolute_bits = source_bits & 0x7FFFFFFF
    absolute = absolute_bits.to(core.float32, bitcast=True)

    # E2M1 positive magnitudes are 0, .5, 1, 1.5, 2, 3, 4, and 6.
    # The strict/inclusive midpoint comparisons select the encoding whose
    # retained mantissa bit is zero at an exact RTNE tie.
    magnitude = core.where(absolute > 0.25, 1, 0)
    magnitude = core.where(absolute >= 0.75, 2, magnitude)
    magnitude = core.where(absolute > 1.25, 3, magnitude)
    magnitude = core.where(absolute >= 1.75, 4, magnitude)
    magnitude = core.where(absolute > 2.5, 5, magnitude)
    magnitude = core.where(absolute >= 3.5, 6, magnitude)
    magnitude = core.where(absolute > 5.0, 7, magnitude)
    magnitude = core.where(absolute_bits >= 0x7F800000, 7, magnitude)
    return (magnitude.to(core.uint8) | sign).to(core.uint8)


@jit
def encode_fp4(
    low,
    high,
    format: core.constexpr = "e2m1",
    rounding: core.constexpr = "rtne",
    overflow: core.constexpr = "satfinite",
):
    """Encode and pack two logical FP4 values into one UINT8 element.

    Version 1 supports OCP E2M1. ``low`` occupies bits 0-3 and ``high``
    occupies bits 4-7. Inputs must have the same shape and floating-point
    dtype. E2M1 has signed zero and no NaN or infinity encodings; non-finite
    inputs and finite overflow saturate to ``+/-6``. Scaling stays explicit in
    the caller and is not hidden in this storage primitive.
    """

    core.static_assert(low.shape == high.shape, "encode_fp4 low and high must have the same shape")
    core.static_assert(low.dtype == high.dtype, "encode_fp4 low and high must have the same dtype")
    core.static_assert(
        low.dtype.is_fp16() or low.dtype.is_bf16() or low.dtype.is_fp32(),
        "encode_fp4 inputs must be tl.float16, tl.bfloat16, or tl.float32",
    )
    core.static_assert(format == "e2m1", "encode_fp4 format must be 'e2m1'")
    core.static_assert(rounding == "rtne", "encode_fp4 rounding must be 'rtne'")
    core.static_assert(overflow == "satfinite", "encode_fp4 overflow must be 'satfinite'")

    low_nibble = _encode_fp4_e2m1_value(low)
    high_nibble = _encode_fp4_e2m1_value(high)
    return (low_nibble | (high_nibble << 4)).to(core.uint8)


@jit
def _decode_fp4_e2m1_value(input, dtype: core.constexpr):
    value = input.to(core.uint32)
    exponent = (value >> 1) & 0x3
    mantissa = value & 0x1
    nonzero = (exponent | mantissa) != 0

    fp32_exponent = core.where(exponent == 0, 126, exponent + 126)
    fp32_mantissa = core.where(exponent == 0, 0, mantissa << 22)
    magnitude_bits = core.where(nonzero, (fp32_exponent << 23) | fp32_mantissa, 0)
    sign_bits = (value & 0x8) << 28
    fp32_bits = (magnitude_bits | sign_bits).to(core.uint32)
    return fp32_bits.to(core.float32, bitcast=True).to(dtype)


@jit
def decode_fp4(input, format: core.constexpr = "e2m1", dtype: core.constexpr = core.float16):
    """Decode low-first packed E2M1 storage into two floating tensors.

    The returned ``(low, high)`` tensors have the same shape as ``input``.
    Reshaping or interleaving them into a logical element axis is deliberately
    left to the caller so odd-tail and layout policy remain explicit.
    """

    core.static_assert(input.dtype.is_uint8(), "decode_fp4 input must be tl.uint8 storage")
    core.static_assert(format == "e2m1", "decode_fp4 format must be 'e2m1'")
    core.static_assert(
        dtype.is_fp16() or dtype.is_bf16() or dtype.is_fp32(),
        "decode_fp4 dtype must be tl.float16, tl.bfloat16, or tl.float32",
    )

    low = _decode_fp4_e2m1_value(input & 0xF, dtype)
    high = _decode_fp4_e2m1_value((input >> 4) & 0xF, dtype)
    return low, high


@jit
def quantize_fp4(
    low,
    high,
    scale,
    format: core.constexpr = "e2m1",
    rounding: core.constexpr = "rtne",
    overflow: core.constexpr = "satfinite",
):
    """Apply an explicit scale and pack two logical FP4 values.

    ``scale`` follows normal Triton broadcasting and must be finite and
    positive. The caller chooses its granularity and supplies one scale for
    each ``low``/``high`` pair. This primitive does not derive a block scale,
    reshape a logical axis, or choose an odd-tail policy.
    """

    core.static_assert(low.shape == high.shape, "quantize_fp4 low and high must have the same shape")
    core.static_assert(low.dtype == high.dtype, "quantize_fp4 low and high must have the same dtype")
    core.static_assert(
        low.dtype.is_fp16() or low.dtype.is_bf16() or low.dtype.is_fp32(),
        "quantize_fp4 inputs must be tl.float16, tl.bfloat16, or tl.float32",
    )
    _validate_quantization_scale(scale)
    normalized_low = low.to(core.float32) / scale
    normalized_high = high.to(core.float32) / scale
    packed = encode_fp4(normalized_low, normalized_high, format=format, rounding=rounding, overflow=overflow)
    # Positive scaling cannot change a value's sign, but vendor scalar divide
    # lowerings may canonicalize -0 to +0. Restore both sign bits from the
    # original carrier instead of trusting the normalized intermediate.
    low_sign = _fp4_e2m1_sign_bit(low)
    high_sign = _fp4_e2m1_sign_bit(high)
    return ((packed & 0x77) | low_sign | (high_sign << 4)).to(core.uint8)


@jit
def dequantize_fp4(
    packed,
    scale,
    format: core.constexpr = "e2m1",
    dtype: core.constexpr = core.float32,
):
    """Decode low-first FP4 storage and apply an explicit scale.

    The returned ``(low, high)`` tensors keep the packed tensor's shape. Scale
    derivation, logical-axis interleaving, and odd-tail handling remain caller
    policy rather than implicit storage behavior.
    """

    core.static_assert(packed.dtype.is_uint8(), "dequantize_fp4 input must be tl.uint8 storage")
    core.static_assert(
        dtype.is_fp16() or dtype.is_bf16() or dtype.is_fp32(),
        "dequantize_fp4 dtype must be tl.float16, tl.bfloat16, or tl.float32",
    )
    _validate_quantization_scale(scale)
    low, high = decode_fp4(packed, format=format, dtype=dtype)
    scaled_low = (low.to(core.float32) * scale).to(dtype)
    scaled_high = (high.to(core.float32) * scale).to(dtype)
    # Multiplication by a positive scale should preserve signed zero, but some
    # vendor LLVM pipelines canonicalize the result to +0. Keep the decoded
    # zero payload explicitly so storage round-trips remain deterministic.
    output_low = core.where(low == 0.0, low, scaled_low)
    output_high = core.where(high == 0.0, high, scaled_high)
    return output_low, output_high


@jit
def pack_int4(low, high, signed: core.constexpr = True):
    """Pack two logical INT4 or UINT4 values into each UINT8 element.

    ``low`` and ``high`` must have the same shape. The low value occupies bits
    0-3 and the high value occupies bits 4-7. Signed inputs use two's-complement
    nibbles and must be INT8 values in ``[-8, 7]``; unsigned inputs must be
    UINT8 values in ``[0, 15]``.

    This pair-wise primitive deliberately does not reshape tensors or infer an
    odd-tail policy. Callers packing an odd logical extent must provide an
    explicit padding value for ``high`` and retain the logical element count.
    """

    core.static_assert(signed == True or signed == False, "signed must be a compile-time bool")
    core.static_assert(low.shape == high.shape, "low and high must have the same shape")
    if signed:
        core.static_assert(
            low.dtype.is_int8() and high.dtype.is_int8(),
            "signed INT4 packing requires tl.int8 inputs",
        )
        core.device_assert(
            (low >= -8) & (low <= 7) & (high >= -8) & (high <= 7),
            "signed INT4 values must be within [-8, 7]",
        )
    else:
        core.static_assert(
            low.dtype.is_uint8() and high.dtype.is_uint8(),
            "UINT4 packing requires tl.uint8 inputs",
        )
        core.device_assert(
            (low <= 15) & (high <= 15),
            "UINT4 values must be within [0, 15]",
        )

    low_nibble = low.to(core.uint8) & 0xF
    high_nibble = high.to(core.uint8) & 0xF
    return (low_nibble | (high_nibble << 4)).to(core.uint8)


@jit
def unpack_int4(packed, signed: core.constexpr = True):
    """Unpack low-first UINT8 storage into two logical INT4 or UINT4 values.

    The returned tensors have the same shape as ``packed``. Signed results are
    sign-extended to INT8; unsigned results remain UINT8. Interleaving or
    reshaping the pair into a logical element axis is intentionally left to the
    caller so backend-independent storage semantics stay explicit.
    """

    core.static_assert(signed == True or signed == False, "signed must be a compile-time bool")
    core.static_assert(packed.dtype.is_uint8(), "INT4 unpacking requires a tl.uint8 packed input")

    low = packed & 0xF
    high = (packed >> 4) & 0xF
    if signed:
        low_i16 = low.to(core.int16)
        high_i16 = high.to(core.int16)
        low = core.where((low & 0x8) != 0, low_i16 - 16, low_i16).to(core.int8)
        high = core.where((high & 0x8) != 0, high_i16 - 16, high_i16).to(core.int8)
    return low, high
