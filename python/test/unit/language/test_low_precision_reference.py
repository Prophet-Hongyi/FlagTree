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

import math

import pytest

from low_precision_reference import (
    E4M3FN,
    E4M3FNUZ,
    E5M2,
    E5M2FNUZ,
    decode_fp8,
    dequantize_affine,
    encode_fp8_rtne,
    encode_fp8_rtz,
    pack_nibbles,
    quantize_affine,
    unpack_nibbles,
)


@pytest.mark.parametrize(
    "fmt, code, expected",
    [
        (E4M3FN, 0x00, 0.0),
        (E4M3FN, 0x80, -0.0),
        (E4M3FN, 0x01, 2**-9),
        (E4M3FN, 0x08, 2**-6),
        (E4M3FN, 0x38, 1.0),
        (E4M3FN, 0x7E, 448.0),
        (E4M3FN, 0xFE, -448.0),
        (E4M3FNUZ, 0x00, 0.0),
        (E4M3FNUZ, 0x01, 2**-10),
        (E4M3FNUZ, 0x08, 2**-7),
        (E4M3FNUZ, 0x40, 1.0),
        (E4M3FNUZ, 0x7F, 240.0),
        (E4M3FNUZ, 0xFF, -240.0),
        (E5M2, 0x01, 2**-16),
        (E5M2, 0x04, 2**-14),
        (E5M2, 0x3C, 1.0),
        (E5M2, 0x7B, 57344.0),
        (E5M2, 0xFB, -57344.0),
        (E5M2, 0x7C, math.inf),
        (E5M2, 0xFC, -math.inf),
        (E5M2FNUZ, 0x00, 0.0),
        (E5M2FNUZ, 0x01, 2**-17),
        (E5M2FNUZ, 0x04, 2**-15),
        (E5M2FNUZ, 0x40, 1.0),
        (E5M2FNUZ, 0x7F, 57344.0),
        (E5M2FNUZ, 0xFF, -57344.0),
    ],
)
def test_decode_fp8_known_encodings(fmt, code, expected):
    actual = decode_fp8(code, fmt)
    assert actual == expected
    if expected == 0.0:
        assert math.copysign(1.0, actual) == math.copysign(1.0, expected)


@pytest.mark.parametrize(
    "fmt, code",
    [
        (E4M3FN, 0x7F),
        (E4M3FN, 0xFF),
        (E4M3FNUZ, 0x80),
        (E5M2, 0x7D),
        (E5M2, 0xFF),
        (E5M2FNUZ, 0x80),
    ],
)
def test_decode_fp8_nan_encodings(fmt, code):
    assert math.isnan(decode_fp8(code, fmt))


@pytest.mark.parametrize("fmt", [E4M3FN, E4M3FNUZ, E5M2, E5M2FNUZ])
def test_all_finite_encodings_roundtrip(fmt):
    for magnitude in range(fmt.max_finite_code + 1):
        for sign in (0x00, 0x80):
            code = sign | magnitude
            if fmt.unsigned_zero and code == fmt.canonical_nan_code:
                continue
            assert encode_fp8_rtne(decode_fp8(code, fmt), fmt) == code
            assert encode_fp8_rtz(decode_fp8(code, fmt), fmt) == code


@pytest.mark.parametrize(
    "fmt, value, expected",
    [
        # Exact subnormal midpoints taken from the AMD software conversion
        # contract.  The retained low bit decides the tie.
        (E4M3FN, 2**-10, 0x00),
        (E4M3FN, 3 * 2**-10, 0x02),
        (E4M3FN, 5 * 2**-10, 0x02),
        (E4M3FN, 7 * 2**-10, 0x04),
        (E4M3FNUZ, 2**-11, 0x00),
        (E4M3FNUZ, 3 * 2**-11, 0x02),
        (E4M3FNUZ, 5 * 2**-11, 0x02),
        (E4M3FNUZ, 7 * 2**-11, 0x04),
        (E5M2, 2**-17, 0x00),
        (E5M2, 3 * 2**-17, 0x02),
        (E5M2, 5 * 2**-17, 0x02),
        (E5M2, 7 * 2**-17, 0x04),
        (E5M2FNUZ, 2**-18, 0x00),
        (E5M2FNUZ, 3 * 2**-18, 0x02),
        (E5M2FNUZ, 5 * 2**-18, 0x02),
        (E5M2FNUZ, 7 * 2**-18, 0x04),
        # E4M3FN top-bin ties exercise the finite-only exponent encoding.
        (E4M3FN, 400.0, 0x7C),
        (E4M3FN, 432.0, 0x7E),
        (E4M3FNUZ, 232.0, 0x7E),
        (E5M2FNUZ, 53248.0, 0x7E),
    ],
)
def test_encode_fp8_rtne_ties(fmt, value, expected):
    assert encode_fp8_rtne(value, fmt) == expected
    expected_negative = 0 if fmt.unsigned_zero and expected == 0 else expected | 0x80
    assert encode_fp8_rtne(-value, fmt) == expected_negative


@pytest.mark.parametrize(
    "fmt, value, expected",
    [
        (E4M3FN, math.inf, 0x7E),
        (E4M3FN, -math.inf, 0xFE),
        (E5M2, math.inf, 0x7B),
        (E5M2, -math.inf, 0xFB),
        (E4M3FN, math.nan, 0x7F),
        (E4M3FN, math.copysign(math.nan, -1.0), 0xFF),
        (E5M2, math.nan, 0x7F),
        (E5M2, math.copysign(math.nan, -1.0), 0xFF),
        (E4M3FNUZ, 241.0, 0x7F),
        (E4M3FNUZ, -241.0, 0xFF),
        (E4M3FNUZ, math.inf, 0x80),
        (E4M3FNUZ, -math.inf, 0x80),
        (E4M3FNUZ, math.nan, 0x80),
        (E4M3FNUZ, math.copysign(math.nan, -1.0), 0x80),
        (E5M2FNUZ, 60000.0, 0x7F),
        (E5M2FNUZ, -60000.0, 0xFF),
        (E5M2FNUZ, math.inf, 0x80),
        (E5M2FNUZ, -math.inf, 0x80),
        (E5M2FNUZ, math.nan, 0x80),
        (E5M2FNUZ, math.copysign(math.nan, -1.0), 0x80),
    ],
)
def test_encode_fp8_saturation_and_nan(fmt, value, expected):
    assert encode_fp8_rtne(value, fmt) == expected


@pytest.mark.parametrize("fmt", [E4M3FNUZ, E5M2FNUZ])
def test_encode_fnuz_has_one_unsigned_zero(fmt):
    assert encode_fp8_rtne(0.0, fmt) == 0x00
    assert encode_fp8_rtne(-0.0, fmt) == 0x00


@pytest.mark.parametrize(
    "fmt, value, expected",
    [
        (E4M3FN, 2**-10, 0x00),
        (E4M3FN, 3 * 2**-10, 0x01),
        (E4M3FN, 1.20, 0x39),
        (E4M3FN, 500.0, 0x7E),
        (E4M3FN, math.inf, 0x7E),
        (E5M2, 2**-17, 0x00),
        (E5M2, 3 * 2**-17, 0x01),
        (E5M2, 1.20, 0x3C),
        (E5M2, 1.375, 0x3D),
        (E5M2, 60000.0, 0x7B),
        (E5M2, math.inf, 0x7C),
    ],
)
def test_encode_fp8_rtz_boundaries(fmt, value, expected):
    assert encode_fp8_rtz(value, fmt) == expected
    assert encode_fp8_rtz(-value, fmt) == (expected | 0x80)


@pytest.mark.parametrize("fmt", [E4M3FN, E5M2])
def test_encode_fp8_rtz_nan_and_signed_zero(fmt):
    assert encode_fp8_rtz(0.0, fmt) == 0x00
    assert encode_fp8_rtz(-0.0, fmt) == 0x80
    assert encode_fp8_rtz(math.nan, fmt) == fmt.canonical_nan_code
    assert encode_fp8_rtz(math.copysign(math.nan, -1.0), fmt) == (0x80 | fmt.canonical_nan_code)


@pytest.mark.parametrize(
    "values, signed, expected",
    [
        ([1, 15, 2], False, bytes([0xF1, 0x02])),
        ([-8, -1, 7], True, bytes([0xF8, 0x07])),
        ([], False, b""),
    ],
)
def test_pack_nibbles_is_low_first_and_pads_odd_tail(values, signed, expected):
    assert pack_nibbles(values, signed=signed) == expected
    assert unpack_nibbles(expected, count=len(values), signed=signed) == values


@pytest.mark.parametrize("value, signed", [(-9, True), (8, True), (-1, False), (16, False)])
def test_pack_nibbles_rejects_out_of_range_values(value, signed):
    with pytest.raises(ValueError):
        pack_nibbles([value], signed=signed)


def test_unpack_nibbles_requires_a_valid_logical_count():
    with pytest.raises(ValueError):
        unpack_nibbles([0x12], count=3, signed=False)


def test_affine_int8_quantization_freezes_rounding_saturation_and_zero_point():
    values = [-100.0, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 100.0]
    assert quantize_affine(values, scale=0.5, zero_point=0, qmin=-2, qmax=2) == [-2, -2, -2, 0, 0, 2, 2, 2]
    assert quantize_affine(values, scale=0.5, zero_point=1, qmin=-8, qmax=7, rounding="rtz") == [
        -8,
        -1,
        0,
        1,
        1,
        2,
        3,
        7,
    ]


def test_affine_dequantization_uses_scale_not_inverse_scale():
    assert dequantize_affine([-8, 0, 7], scale=0.25, zero_point=-1) == [-1.75, 0.25, 2.0]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scale": 0.0, "zero_point": 0, "qmin": -128, "qmax": 127},
        {"scale": math.inf, "zero_point": 0, "qmin": -128, "qmax": 127},
        {"scale": 1.0, "zero_point": 128, "qmin": -128, "qmax": 127},
        {"scale": 1.0, "zero_point": 0, "qmin": 7, "qmax": 7},
    ],
)
def test_affine_quantization_rejects_invalid_contracts(kwargs):
    with pytest.raises(ValueError):
        quantize_affine([0.0], **kwargs)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_affine_quantization_rejects_non_finite_inputs(value):
    with pytest.raises(ValueError):
        quantize_affine([value], scale=1.0, zero_point=0, qmin=-128, qmax=127)
