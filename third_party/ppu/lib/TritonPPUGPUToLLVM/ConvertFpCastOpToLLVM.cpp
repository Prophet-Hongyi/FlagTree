/*
 * Copyright (c) 2026 T-Head Semiconductor Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

//===----------------------------------------------------------------------===//
//
// This file lowers PPU low-precision floating-point casts to LLVM dialect
// operations and PPU conversion sequences. It keeps software fallbacks and
// target-specific conversion selection separate from generic elementwise ops.
//
//===----------------------------------------------------------------------===//

#include "ConvertFpCastOpToLLVM.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "TritonPPUGPUToLLVM/TIXAsmFormat.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Support/LLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVMBase.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

using namespace mlir::triton::gpu;
using namespace mlir::triton::ppu;

namespace mlir::triton {

namespace gpu {
namespace {

/* ----- FP8E5M2 ------ */
// This data-type is the standard FP8E5M2 format

struct Fp8ConversionDesc {
  std::string tix;
  int inVecWidthBits;
  int outVecWidthBits;
  size_t numElements;
};

static const Fp8ConversionDesc Fp16_to_Fp8E5M2_RTNE(bool hasNativeFP) {
  Fp8ConversionDesc ret;
  if (!hasNativeFP) {
    ret = {"{                            \n"
           ".reg .b32 a<2>;              \n"
           "ppu.and.b32 a0, $1, 0xfffefffe;  \n"   // a0 &= 0xfffefffe
           "ppu.and.b32 a1, $2, 0xfffefffe;  \n"   // (strip lowest bit)
           "ppu.add.u32 a0, a0, 0x00800080;  \n"   // a0 += 0x00800080
           "ppu.add.u32 a1, a1, 0x00800080;  \n"   // (round to nearest)
           "ppu.prmt.b32 $0, a0, a1, 0x7531; \n\t" // output = a1a0
           "}",
           32, 32, 4};
  } else {
    ret = {"ppu.cvt.rn.satfinite.e5m2x2.f16x2 $0, $1; \n\t", 32, 16, 2};
  }
  return ret;
}

const Fp8ConversionDesc Fp16_to_Fp8E5M2_RTZ = {
    "{                            \n"
    ".reg .b32 a<2>;              \n"
    "ppu.and.b32 a0, $1, 0xfffefffe;  \n"   // a0 &= 0xfffefffe
    "ppu.and.b32 a1, $2, 0xfffefffe;  \n"   // (strip lowest bit)
    "ppu.prmt.b32 $0, a0, a1, 0x7531; \n\t" // output = a1a0
    "}",
    32, 32, 4};

static const Fp8ConversionDesc Fp8E5M2_to_Fp16(bool hasNativeFP) {
  Fp8ConversionDesc ret;
  if (!hasNativeFP) {
    ret = {"{                           \n"
           "ppu.prmt.b32 $0, 0, $2, 0x5140; \n\t"
           "ppu.prmt.b32 $1, 0, $2, 0x7362; \n\t"
           "}",
           32, 32, 4};
  } else {
    ret = {"ppu.cvt.rn.f16x2.e5m2x2 $0, $1; \n\t", 16, 32, 2};
  }
  return ret;
}

static const Fp8ConversionDesc Fp8E5M2_to_Bf16(bool hasNativeFP) {
  Fp8ConversionDesc ret;
  if (!hasNativeFP) {
    ret = {
        "{                                        \n"
        ".reg .b32 a<2>, b<2>, c<4>, d<4>, e112;  \n" // if input = 0xf1f2f3f4
        "ppu.mov.u32 e112, 0x77800000;                \n"
        "ppu.prmt.b32 a0, 0, $2, 0x5140;              \n" // a0 = 0xf300f400
        "ppu.prmt.b32 a1, 0, $2, 0x7362;              \n" // a1 = 0xf100f200
        "ppu.lop3.b32 b0, a0, 0x7fff7fff, 0, 0xc0;    \n" // b0 = a0 &
                                                          // 0x7fff7fff
        "ppu.lop3.b32 b1, a1, 0x7fff7fff, 0, 0xc0;    \n" // (strip sign)
        "ppu.shr.b32  b0, b0, 3;                      \n" // b0 >>= 3
        "ppu.shr.b32  b1, b1, 3;                      \n" // shift into bf16
                                                          // position
        "ppu.and.b32 c0, b0, 0xFFFF0000;              \n" // c0 = f3
        "ppu.shl.b32 c1, b0, 16;                      \n" // c1 = f4
        "ppu.and.b32 c2, b1, 0xFFFF0000;              \n" // c2 = f1
        "ppu.shl.b32 c3, b1, 16;                      \n" // c3 = f2
        "ppu.mul.f32 d0, c0, e112;                    \n" // d0 = c0 *
                                                          // 0x77800000
        "ppu.mul.f32 d1, c1, e112;                    \n" // d1 = c1 *
                                                          // 0x77800000
        "ppu.mul.f32 d2, c2, e112;                    \n" // d2 = c2 *
                                                          // 0x77800000
        "ppu.mul.f32 d3, c3, e112;                    \n" // d3 = c3 *
                                                          // 0x77800000
        "ppu.prmt.b32 b0, d0, d1, 0x3276;             \n" // b0 = 0xd3d4
        "ppu.prmt.b32 b1, d2, d3, 0x3276;             \n" // b1 = 0xd1d2
        "ppu.lop3.b32 $0, b0, 0x80008000, a0, 0xf8;   \n" // out0 =
                                                          // b0|(0x80008000&a0)
        "ppu.lop3.b32 $1, b1, 0x80008000, a1, 0xf8;   \n" // (restore sign)
        "}",
        32, 32, 4};
  } else {
    ret = {
        "{                                       \n"
        ".reg .b32 a<2>, b<2>;                  \n" // if input = 0xf1f2f3f4
        ".reg .b32 e112;                        \n"
        "ppu.mov.u32 e112, 0x77807780;              \n" // 2**112 represented as
                                                        // bf16x2
        "ppu.prmt.b32 a0, 0, $2, 0x5140;            \n" // a0 = 0xf300f400
        "ppu.prmt.b32 a1, 0, $2, 0x7362;            \n" // a1 = 0xf100f200
        "ppu.lop3.b32 b0, a0, 0x7fff7fff, 0, 0xc0;  \n" // b0 = a0 & 0x7fff7fff
        "ppu.lop3.b32 b1, a1, 0x7fff7fff, 0, 0xc0;  \n" // (strip sign)
        "ppu.shr.b32  b0, b0, 3;                    \n" // b0 >>= 3
        "ppu.shr.b32  b1, b1, 3;                    \n" // shift into bf16
                                                        // position
        "ppu.lop3.b32 b0, b0, 0x80008000, a0, 0xf8; \n" // out0 =
                                                        // b0|(0x80008000&a0)
        "ppu.lop3.b32 b1, b1, 0x80008000, a1, 0xf8; \n" // (restore sign)
        "ppu.mul.rn.bf16x2 $0, b0, e112;            \n" // b0.exp += 2**7-2**4
        "ppu.mul.rn.bf16x2 $1, b1, e112;            \n" // exponent compensate =
                                                        // 112
        "}",
        32, 32, 4};
  }
  return ret;
}

static const Fp8ConversionDesc Bf16_to_Fp8E5M2(bool hasNativeFP) {
  Fp8ConversionDesc ret;
  if (!hasNativeFP) {
    ret = {
        "{                                           \n" // bf16=fp8>>3 + 112<<7
        ".reg .u32 sign, sign<2>, nosign, nosign<2>; \n" // fp8_min = 0b00000000
        ".reg .u32 fp8_min, fp8_max, rn_;            \n" // fp8_max = 0b11111111
        "ppu.mov.u32 fp8_min, 0x38003800;                \n" // so bf16_min =
                                                             // 0x3800
        "ppu.mov.u32 fp8_max, 0x57e057e0;                \n" // so bf16_max =
                                                             // 0x57e0
        "ppu.mov.u32 rn_, 0x00100010;                    \n" // round to nearest
        "ppu.and.b32 sign0, $1, 0x80008000;              \n" // sign0=in0&0x80008000
        "ppu.and.b32 sign1, $2, 0x80008000;              \n" // (store sign)
        "ppu.prmt.b32 sign, sign0, sign1, 0x7531;        \n"
        "ppu.and.b32 nosign0, $1, 0x7fff7fff;            \n" // nosign0=in0&0x7fff7fff
        "ppu.and.b32 nosign1, $2, 0x7fff7fff;            \n" // (strip sign)

        // nosign = clamp(nosign, min, max)
        ".reg .u32 nosign_0_<2>, nosign_1_<2>;       \n"
        "ppu.and.b32 nosign_0_0, nosign0, 0xffff0000;    \n"
        "ppu.max.u32 nosign_0_0, nosign_0_0, 0x38000000; \n"
        "ppu.min.u32 nosign_0_0, nosign_0_0, 0x57e00000; \n"
        "ppu.and.b32 nosign_0_1, nosign0, 0x0000ffff;    \n"
        "ppu.max.u32 nosign_0_1, nosign_0_1, 0x3800;     \n"
        "ppu.min.u32 nosign_0_1, nosign_0_1, 0x57e0;     \n"
        "ppu.or.b32 nosign0, nosign_0_0, nosign_0_1;     \n"
        "ppu.and.b32 nosign_1_0, nosign1, 0xffff0000;    \n"
        "ppu.max.u32 nosign_1_0, nosign_1_0, 0x38000000; \n"
        "ppu.min.u32 nosign_1_0, nosign_1_0, 0x57e00000; \n"
        "ppu.and.b32 nosign_1_1, nosign1, 0x0000ffff;    \n"
        "ppu.max.u32 nosign_1_1, nosign_1_1, 0x3800;     \n"
        "ppu.min.u32 nosign_1_1, nosign_1_1, 0x57e0;     \n"
        "ppu.or.b32 nosign1, nosign_1_0, nosign_1_1;     \n"

        "ppu.add.u32 nosign0, nosign0, rn_;              \n" // nosign0 += rn_
        "ppu.add.u32 nosign1, nosign1, rn_;              \n" // (round to
                                                             // nearest)
        "ppu.sub.u32 nosign0, nosign0, 0x38003800;       \n" // nosign0-=0x38003800
        "ppu.sub.u32 nosign1, nosign1, 0x38003800;       \n" // (compensate
                                                             // offset)
        "ppu.shl.b32 nosign0, nosign0, 3;                \n" // nosign0 <<= 3
        "ppu.shl.b32 nosign1, nosign1, 3;                \n" // shift into to
                                                             // fp8e4
        "ppu.prmt.b32 nosign, nosign0, nosign1, 0x7531;  \n" // nosign0 =
                                                             // 0xf100f200
                                                             // nosign1 =
                                                             // 0xf300f400
                                                             // nosign =
                                                             // 0xf3f4f1f2
        "ppu.or.b32 $0, nosign, sign;                    \n" // restore sign
        "}",
        32, 32, 4};
  } else {
    ret = {"{                                       \n"
           ".reg .b16 a<2>;                         \n"
           ".reg .f32 b<2>;                         \n"
           "ppu.mov.b32 {a0, a1}, $1;                   \n"
           "ppu.cvt.f32.bf16 b0, a0;                    \n"
           "ppu.cvt.f32.bf16 b1, a1;                    \n"
           "ppu.cvt.rn.satfinite.e5m2x2.f32 $0, b1, b0; \n"
           "}",
           32, 16, 2};
  }
  return ret;
}

// Fp8E4M3 (x2) -> Fp16 (x2) (packed)
static const Fp8ConversionDesc Fp8E4M3Nv_to_Fp16 = {
    "{ \n"
    "ppu.cvt.rn.f16x2.e4m3x2 $0, $1; \n"
    "}",
    16, 32, 2};

// Fp16 (x2) -> Fp8E4M3 (x2) (packed)
static const Fp8ConversionDesc Fp16_to_Fp8E4M3Nv = {
    "{ \n"
    "ppu.cvt.rn.satfinite.e4m3x2.f16x2 $0, $1; \n"
    "}",
    32, 16, 2};

static const Fp8ConversionDesc Fp8E4M3Nv_to_Bf16(bool hasNativeFP) {
  Fp8ConversionDesc ret;
  // Fp8E4M3 (x2) -> Fp16 (x2) (packed)
  if (!hasNativeFP) {
    ret = {"{                                       \n"
           ".reg .b32 a;                            \n"
           ".reg .f16 a<2>;                         \n"
           ".reg .f32 b<2>;                         \n"
           ".reg .b16 c<2>;                         \n"
           "ppu.cvt.rn.f16x2.e4m3x2 a, $1;              \n"
           "ppu.mov.b32 {a0, a1}, a;                    \n"
           "ppu.cvt.f32.f16 b0, a0;                     \n"
           "ppu.cvt.f32.f16 b1, a1;                     \n"
           "ppu.cvt.rn.bf16.f32 c0, b0;                 \n"
           "ppu.cvt.rn.bf16.f32 c1, b1;                 \n"
           "ppu.mov.b32 $0, {c0, c1};                   \n"
           "}",
           16, 32, 2};
  } else {
    ret = {"{                                       \n"
           ".reg .b32 a;                            \n"
           ".reg .f16 a<2>;                         \n"
           ".reg .b16 b<2>;                         \n"
           "ppu.cvt.rn.f16x2.e4m3x2 a, $1;              \n"
           "ppu.mov.b32 {a0, a1}, a;                    \n"
           "ppu.cvt.bf16.f16 b0, a0;                    \n"
           "ppu.cvt.bf16.f16 b1, a1;                    \n"
           "ppu.mov.b32 $0, {b0, b1};                   \n"
           "}",
           16, 32, 2};
  }
  return ret;
}

// Bf16 (x2) -> Fp8E4M3 (x2) (packed)
static const Fp8ConversionDesc Bf16_to_Fp8E4M3Nv = {
    "{                                       \n"
    ".reg .b16 a<2>;                         \n"
    ".reg .f32 b<2>;                         \n"
    "ppu.mov.b32 {a0, a1}, $1;                   \n"
    "ppu.cvt.f32.bf16 b0, a0;                    \n"
    "ppu.cvt.f32.bf16 b1, a1;                    \n"
    "ppu.cvt.rn.satfinite.e4m3x2.f32 $0, b1, b0; \n"
    "}",
    32, 16, 2};

// Fp32 (x2) -> Fp8 (x2) (packed)
static const Fp8ConversionDesc Fp32_to_Fp8E4M3Nv = {
    "ppu.cvt.rn.satfinite.e4m3x2.f32  $0, $2, $1; \n", 32, 16, 2};
static const Fp8ConversionDesc Fp32_to_Fp8E5M2 = {
    "ppu.cvt.rn.satfinite.e5m2x2.f32 $0, $2, $1; \n", 32, 16, 2};

/* ----- Packed integer to BF16 ------ */
static const std::string S8_to_Bf16 =
    "{                                           \n"
    ".reg .s8 s<4>;                              \n"
    ".reg .f32 f<4>;                             \n"
    "ppu.mov.b32 {s0, s1, s2, s3}, $2;           \n" // unpack
    "ppu.cvt.rn.f32.s8 f0, s0;                   \n" // no s8->bf16
    "ppu.cvt.rn.f32.s8 f1, s1;                   \n" // fi[0:15] is always 0
    "ppu.cvt.rn.f32.s8 f2, s2;                   \n" //
    "ppu.cvt.rn.f32.s8 f3, s3;                   \n" //
    "ppu.prmt.b32 $0, f0, f1, 0x7632;            \n" // f32->bf16 + pack
    "ppu.prmt.b32 $1, f2, f3, 0x7632;            \n" //
    "}";
// Conversions have low throughput, rely on bit tricks instead of cvt
// instruction on SM90+.
static const std::string S8_to_Bf16_sm90 =
    "{                               \n"
    ".reg .b32 l<3>;                 \n"
    ".reg .b32 h<3>;                 \n"
    "ppu.prmt.b32 l0, $2, 0x43, 0x4140;  \n" // Unpack to shifted bf16.
    "ppu.prmt.b32 h0, $2, 0x43, 0x4342;  \n"
    "ppu.and.b32 l1, l0, 0xff7fff7f;     \n" // Zero the least exp bit.
    "ppu.and.b32 h1, h0, 0xff7fff7f;     \n"
    "ppu.and.b32 l2, l0, 0xff80ff80;     \n" // Zero the mantissa.
    "ppu.and.b32 h2, h0, 0xff80ff80;     \n"
    "ppu.sub.bf16x2 $0, l1, l2;          \n" // Subtract the offset.
    "ppu.sub.bf16x2 $1, h1, h2;          \n"
    "}";

typedef std::function<SmallVector<Value>(Location, ConversionPatternRewriter &,
                                         const SmallVector<Value> &)>
    ConverterT;

template <typename SrcFPType>
static Value downcastToFp8E4M3FNRTNEOneValue(
    Location loc, ConversionPatternRewriter &rewriter, Value value) {
  static_assert(std::is_same_v<SrcFPType, Float32Type> ||
                std::is_same_v<SrcFPType, Float16Type> ||
                std::is_same_v<SrcFPType, BFloat16Type>);

  auto b = TritonLLVMOpBuilder(loc, rewriter);
  constexpr bool isFp32 = std::is_same_v<SrcFPType, Float32Type>;
  constexpr bool isFp16 = std::is_same_v<SrcFPType, Float16Type>;
  IntegerType intType = isFp32 ? i32_ty : i16_ty;
  constexpr unsigned srcWidth = isFp32 ? 32 : 16;
  constexpr unsigned srcMantissaBits = isFp32 ? 23 : (isFp16 ? 10 : 7);
  constexpr unsigned srcExponentBits = isFp32 ? 8 : (isFp16 ? 5 : 8);
  constexpr unsigned srcBias = (1U << (srcExponentBits - 1)) - 1;
  constexpr unsigned dstMantissaBits = 3;
  constexpr unsigned dstBias = 7;
  constexpr unsigned reducedMantissaBits = srcMantissaBits - dstMantissaBits;

  auto intValue = [&](uint32_t constant) -> Value {
    if constexpr (isFp32)
      return b.i32_val(static_cast<int32_t>(constant));
    return b.i16_val(static_cast<int16_t>(constant));
  };

  Value bits = b.bitcast(value, intType);
  constexpr uint32_t signMask = 1U << (srcWidth - 1);
  constexpr uint32_t absoluteMask = signMask - 1;
  Value absolute = b.and_(bits, intValue(absoluteMask));
  Value sign = b.trunc(
      i8_ty, b.lshr(b.and_(bits, intValue(signMask)), intValue(srcWidth - 8)));

  constexpr uint32_t exponentMask = ((1U << srcExponentBits) - 1)
                                    << srcMantissaBits;
  constexpr uint32_t mantissaMask = (1U << srcMantissaBits) - 1;
  Value isNaN =
      b.and_(b.icmp_eq(b.and_(absolute, intValue(exponentMask)),
                       intValue(exponentMask)),
             b.icmp_ne(b.and_(absolute, intValue(mantissaMask)), intValue(0)));

  constexpr uint32_t baseRoundingBias = (1U << (reducedMantissaBits - 1)) - 1;
  constexpr uint32_t mantissaLSB = 1U << reducedMantissaBits;
  Value roundingBias = b.add(b.lshr(b.and_(absolute, intValue(mantissaLSB)),
                                    intValue(reducedMantissaBits)),
                             intValue(baseRoundingBias));
  Value fp8 = b.add(absolute, roundingBias);

  constexpr uint32_t reduceMantissaMask = static_cast<uint32_t>(
      ((1ULL << (1 + srcExponentBits + dstMantissaBits + 1)) - 1)
      << reducedMantissaBits);
  fp8 = b.and_(fp8, intValue(reduceMantissaMask));

  constexpr uint32_t dstMinimal =
      isFp32 ? 0x3c800000 : (isFp16 ? 0x2400 : 0x3c80);
  fp8 = b.umax(fp8, intValue(dstMinimal));

  constexpr uint32_t exponentBias = (srcBias - dstBias) << srcMantissaBits;
  fp8 = b.sub(fp8, intValue(exponentBias));
  fp8 = b.trunc(i8_ty, b.lshr(fp8, intValue(reducedMantissaBits)));

  constexpr uint32_t dstMaxOfSrcType = isFp32   ? 0x43e7ffff
                                       : isFp16 ? 0x5f40
                                                : 0x43e7;
  fp8 = b.select(b.icmp_ugt(absolute, intValue(dstMaxOfSrcType)),
                 b.i8_val(0x7e), fp8);

  constexpr std::array<uint32_t, 8> halfwayPoints =
      isFp32
          ? std::array<uint32_t, 8>{0x3a800000, 0x3b400000, 0x3ba00000,
                                    0x3be00000, 0x3c100000, 0x3c300000,
                                    0x3c500000, 0x3c700000}
          : (isFp16 ? std::array<uint32_t, 8>{0x1400, 0x1a00, 0x1d00, 0x1f00,
                                              0x2080, 0x2180, 0x2280, 0x2380}
                    : std::array<uint32_t, 8>{0x3a80, 0x3b40, 0x3ba0, 0x3be0,
                                              0x3c10, 0x3c30, 0x3c50, 0x3c70});
  for (int i = halfwayPoints.size() - 1; i >= 0; --i) {
    Value belowHalfway = i % 2 == 0
                             ? b.icmp_ule(absolute, intValue(halfwayPoints[i]))
                             : b.icmp_ult(absolute, intValue(halfwayPoints[i]));
    fp8 = b.select(belowHalfway, b.i8_val(i), fp8);
  }

  fp8 = b.select(isNaN, b.i8_val(0x7f), fp8);
  return b.or_(fp8, sign);
}

template <typename SrcFPType>
static SmallVector<Value>
downcastToFp8E4M3FNRTNESoftware(Location loc,
                                ConversionPatternRewriter &rewriter,
                                const SmallVector<Value> &values) {
  assert(values.size() == 4);
  SmallVector<Value> results;
  results.reserve(values.size());
  for (Value value : values)
    results.push_back(
        downcastToFp8E4M3FNRTNEOneValue<SrcFPType>(loc, rewriter, value));
  return results;
}

static Value upcastFp8E4M3FNToFp16OneValue(Location loc,
                                           ConversionPatternRewriter &rewriter,
                                           Value value) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto fp8x2VecTy = vec_ty(i8_ty, 2);
  Value bits = b.undef(fp8x2VecTy);
  bits = b.insert_element(fp8x2VecTy, bits, b.i8_val(0), b.i32_val(0));
  bits = b.insert_element(fp8x2VecTy, bits, value, b.i32_val(1));
  bits = b.bitcast(bits, i16_ty);

  Value sign = b.and_(bits, b.i16_val(0x8000));
  bits = b.and_(bits, b.i16_val(0x7fff));
  bits = b.lshr(bits, b.i16_val(1));
  bits = b.add(bits, b.i16_val(0x2000));

  Value absolute = b.and_(b.bitcast(value, i8_ty), b.i8_val(0x7f));
  bits = b.select(b.icmp_eq(absolute, b.i8_val(0x7f)), b.i16_val(0x7e00), bits);

  constexpr std::array<uint16_t, 8> denormalsAndZero = {
      0x0000, 0x1800, 0x1c00, 0x1e00, 0x2000, 0x2100, 0x2200, 0x2300};
  for (int i = 0; i < denormalsAndZero.size(); ++i)
    bits = b.select(b.icmp_eq(absolute, b.i8_val(i)),
                    b.i16_val(denormalsAndZero[i]), bits);

  bits = b.or_(bits, sign);
  return b.bitcast(bits, f16_ty);
}

static SmallVector<Value>
upcastFp8E4M3FNToFp16Software(Location loc, ConversionPatternRewriter &rewriter,
                              const SmallVector<Value> &values) {
  assert(values.size() == 4);
  SmallVector<Value> results;
  results.reserve(values.size());
  for (Value value : values)
    results.push_back(upcastFp8E4M3FNToFp16OneValue(loc, rewriter, value));
  return results;
}

static Value convertFp32ToBf16RTNE(Location loc,
                                   ConversionPatternRewriter &rewriter,
                                   Value value) {
  TIXBuilder builder;
  auto &cvt = *builder.create("ppu.cvt.rn.bf16.f32");
  auto result = builder.newOperand("=h");
  auto operand = builder.newOperand(value, "f");
  cvt(result, operand);
  return builder.launch(rewriter, loc, bf16_ty, false);
}

static SmallVector<Value>
upcastFp8E4M3FNToBf16Software(Location loc, ConversionPatternRewriter &rewriter,
                              const SmallVector<Value> &values) {
  SmallVector<Value> results;
  results.reserve(values.size());
  for (Value value : values) {
    Value fp16 = upcastFp8E4M3FNToFp16OneValue(loc, rewriter, value);
    Value fp32 = LLVM::FPExtOp::create(rewriter, loc, f32_ty, fp16);
    results.push_back(convertFp32ToBf16RTNE(loc, rewriter, fp32));
  }
  return results;
}

static ConverterT makeConverterFromTIX(const std::string &tixAsm, Type inType,
                                       Type outType,
                                       const int inVecWidthBits = 32,
                                       const int outVecWidthBits = 32) {
  ConverterT converter =
      [tixAsm, inType, outType, inVecWidthBits,
       outVecWidthBits](Location loc, ConversionPatternRewriter &rewriter,
                        const SmallVector<Value> &v) -> SmallVector<Value> {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    int numElements = v.size();
    assert(numElements == 4 || numElements == 2 && "invalid vector size");

    auto ctx = rewriter.getContext();
    int inBitwidth = inType.getIntOrFloatBitWidth();
    int outBitwidth = outType.getIntOrFloatBitWidth();
    // first, we pack `v` into 32-bit ints
    int inVecWidth = inVecWidthBits / inBitwidth;
    auto inVecTy = vec_ty(inType, inVecWidth);
    SmallVector<Value> inPacked(numElements / inVecWidth, b.undef(inVecTy));
    for (size_t i = 0; i < numElements; i++)
      inPacked[i / inVecWidth] = b.insert_element(
          inVecTy, inPacked[i / inVecWidth], v[i], b.i32_val(i % inVecWidth));
    for (size_t i = 0; i < inPacked.size(); i++)
      inPacked[i] = b.bitcast(inPacked[i], int_ty(inVecWidthBits));

    // then, we run the provided inline TIX
    int outVecWidth = outVecWidthBits / outBitwidth;
    int outNums = numElements / outVecWidth;
    TIXBuilder builder;
    SmallVector<TIXBuilder::Operand *> operands;
    auto outConstraint = outVecWidthBits == 16 ? "=h" : "=r";
    auto inConstraint = inVecWidthBits == 16 ? "h" : "r";
    for (int i = 0; i < outNums; i++) {
      operands.push_back(builder.newOperand(outConstraint));
    }

    for (Value inVal : inPacked) {
      operands.push_back(builder.newOperand(inVal, inConstraint));
    }

    auto &tixOp = *builder.create(tixAsm);
    tixOp(operands, /*onlyAttachMLIRArgs=*/true);
    auto outVecTy = vec_ty(outType, outVecWidth);
    SmallVector<Value> outPacked;
    if (outNums == 1)
      outPacked.push_back(builder.launch(rewriter, loc, outVecTy, false));
    else {
      auto outStructTy = struct_ty(SmallVector<Type>(outNums, outVecTy));
      auto outStruct = builder.launch(rewriter, loc, outStructTy, false);
      for (int i = 0; i < outNums; i++)
        outPacked.push_back(b.extract_val(outVecTy, outStruct, i));
    }
    // unpack the output
    SmallVector<Value> ret;
    for (size_t i = 0; i < numElements; i++)
      ret.push_back(b.extract_element(outType, outPacked[i / outVecWidth],
                                      b.i32_val(i % outVecWidth)));
    return ret;
  };
  return converter;
}

// Attempts to use vectorized conversions via inline TIX when possible.
struct FpToFpOpConversion
    : public ElementwiseOpConversionBase<FpToFpOp, FpToFpOpConversion> {
  using ElementwiseOpConversionBase<
      FpToFpOp, FpToFpOpConversion>::ElementwiseOpConversionBase;

  explicit FpToFpOpConversion(LLVMTypeConverter &typeConverter,
                              ModuleAxisInfoAnalysis &axisAnalysisPass,
                              int computeCapability,
                              PatternBenefit benefit = patternBenefitDefault)
      : ElementwiseOpConversionBase(typeConverter, axisAnalysisPass, benefit),
        computeCapability(computeCapability) {}

  static Value convertFp16ToFp32(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 const Value &v) {
    return LLVM::FPExtOp::create(rewriter, loc, f32_ty, v);
  }

  static Value convertFp32ToBf16(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 const Value &v, const RoundingMode rounding) {
    TIXBuilder builder;
    StringRef tix;
    switch (rounding) {
    case RoundingMode::RTNE:
      tix = "ppu.cvt.rn.bf16.f32";
      break;
    case RoundingMode::RTZ:
      tix = "ppu.cvt.rz.bf16.f32";
      break;
    default:
      emitError(loc) << "unsupported rounding mode for f32->bf16 conversion: "
                     << stringifyRoundingMode(rounding) << "\n";
      llvm::report_fatal_error(
          "unsupported rounding mode for f32->bf16 conversion: " +
          stringifyRoundingMode(rounding) + "\n");
    }
    auto &cvt = *builder.create(tix.str());
    auto res = builder.newOperand("=h");
    auto operand = builder.newOperand(v, "f");
    cvt(res, operand);
    return builder.launch(rewriter, loc, bf16_ty, false);
  }

  static Value convertFp32ToFp16(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 const Value &v, const RoundingMode rounding) {
    TIXBuilder builder;
    StringRef tix;
    switch (rounding) {
    case RoundingMode::RTNE:
      tix = "ppu.cvt.rn.f16.f32";
      break;
    case RoundingMode::RTZ:
      tix = "ppu.cvt.rz.f16.f32";
      break;
    default:
      emitError(loc) << "unsupported rounding mode for f32->f16 conversion: "
                     << stringifyRoundingMode(rounding) << "\n";
      llvm::report_fatal_error(
          "unsupported rounding mode for f32->f16 conversion: " +
          stringifyRoundingMode(rounding) + "\n");
    }
    auto &cvt = *builder.create(tix.str());
    auto res = builder.newOperand("=h");
    auto operand = builder.newOperand(v, "r");
    cvt(res, operand);
    return builder.launch(rewriter, loc, f16_ty, false);
  }

  std::pair<ConverterT, size_t>
  getConversionFunc(Type srcTy, Type dstTy,
                    std::optional<RoundingMode> roundingMode) const {
    auto F8E4M3TyID = TypeID::get<Float8E4M3FNType>();
    auto F8E5M2TyID = TypeID::get<Float8E5M2Type>();
    auto F16TyID = TypeID::get<Float16Type>();
    auto BF16TyID = TypeID::get<BFloat16Type>();
    auto F32TyID = TypeID::get<Float32Type>();
    auto F64TyID = TypeID::get<Float64Type>();

    auto undefRounding = static_cast<RoundingMode>(-1);

    bool involvesFp8E4M3 = llvm::isa<Float8E4M3FNType>(srcTy) ||
                           llvm::isa<Float8E4M3FNType>(dstTy);
    if (computeCapability == 80) {
      if (llvm::isa<Float8E4M3FNType>(dstTy) &&
          roundingMode == RoundingMode::RTNE) {
        if (srcTy.isF32())
          return {downcastToFp8E4M3FNRTNESoftware<Float32Type>, 4};
        if (srcTy.isF16())
          return {downcastToFp8E4M3FNRTNESoftware<Float16Type>, 4};
        if (srcTy.isBF16())
          return {downcastToFp8E4M3FNRTNESoftware<BFloat16Type>, 4};
      }
      if (llvm::isa<Float8E4M3FNType>(srcTy)) {
        if (dstTy.isF16())
          return {upcastFp8E4M3FNToFp16Software, 4};
        if (dstTy.isBF16())
          return {upcastFp8E4M3FNToBf16Software, 4};
      }
    }
    if (involvesFp8E4M3 && computeCapability < 89) {
      if (computeCapability == 80) {
        llvm::report_fatal_error(
            "Unsupported f8e4m3nv conversion on PPU capability 80; only "
            "FP32/FP16/BF16 RTNE encode and FP16/BF16 decode are supported\n");
      }
      llvm::report_fatal_error(
          "Conversion from/to f8e4m3nv is only supported on PPU capability "
          "80 or >= 89\n");
    }

    static DenseMap<std::tuple<TypeID, TypeID, RoundingMode>, Fp8ConversionDesc>
        srcMap = {
            // F8 -> F16
            {{F8E4M3TyID, F16TyID, undefRounding}, Fp8E4M3Nv_to_Fp16},
            {{F8E5M2TyID, F16TyID, undefRounding},
             Fp8E5M2_to_Fp16(computeCapability >= 89)},
            {{F16TyID, F8E4M3TyID, RoundingMode::RTNE}, Fp16_to_Fp8E4M3Nv},
            {{F16TyID, F8E5M2TyID, RoundingMode::RTNE},
             Fp16_to_Fp8E5M2_RTNE(computeCapability >= 89)},
            {{F16TyID, F8E5M2TyID, RoundingMode::RTZ}, Fp16_to_Fp8E5M2_RTZ},
            // F8 -> BF16
            // mul{.rnd}.bf16 and mul{.rnd}.bf16x2 requires sm_90 or higher.
            {{F8E5M2TyID, BF16TyID, undefRounding},
             Fp8E5M2_to_Bf16(computeCapability >= 80)},
            // cvt with .bf16.f16' requires .target sm_90 or higher
            {{F8E4M3TyID, BF16TyID, undefRounding},
             Fp8E4M3Nv_to_Bf16(computeCapability >= 89)},
            // BF16 -> F8
            {{BF16TyID, F8E5M2TyID, RoundingMode::RTNE},
             Bf16_to_Fp8E5M2(computeCapability >= 89)},
            {{BF16TyID, F8E4M3TyID, RoundingMode::RTNE}, Bf16_to_Fp8E4M3Nv},
            // F32 -> F8
            {{F32TyID, F8E4M3TyID, RoundingMode::RTNE}, Fp32_to_Fp8E4M3Nv},
            {{F32TyID, F8E5M2TyID, RoundingMode::RTNE}, Fp32_to_Fp8E5M2},
        };
    std::tuple<TypeID, TypeID, RoundingMode> key = {
        srcTy.getTypeID(), dstTy.getTypeID(),
        roundingMode.value_or(undefRounding)};
    if (srcMap.count(key) == 0) {
      llvm::errs() << "Unsupported conversion from " << srcTy << " to "
                   << dstTy;
      if (roundingMode.has_value())
        llvm::errs() << " with rounding mode "
                     << stringifyRoundingMode(roundingMode.value());
      llvm::errs() << "\n";
      llvm::report_fatal_error("Unsupported rounding mode for conversion.");
    }
    auto convDesc = srcMap.lookup(key);
    return {makeConverterFromTIX(
                convDesc.tix, getTypeConverter()->convertType(srcTy),
                getTypeConverter()->convertType(dstTy), convDesc.inVecWidthBits,
                convDesc.outVecWidthBits),
            convDesc.numElements};
  }

  SmallVector<Value> createDestOps(FpToFpOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto srcElementType = getElementType(op.getSrc());
    auto dstElementType = getElementType(op.getResult());
    auto roundingMode = op.getRounding();

    if (llvm::isa<Float8E5M2Type, Float8E4M3FNType>(dstElementType)) {
      assert(roundingMode.has_value() &&
             "Rounding mode must be specified for convertsions to fp8");

      // For now only RTNE is supported for conversions from fp16 to fp8
      if (!srcElementType.isF32() &&
          roundingMode.value() != RoundingMode::RTNE) {
        llvm::report_fatal_error(
            "Unsupported rounding mode for conversion to fp8: " +
            stringifyRoundingMode(roundingMode.value()) + "\n");
      }
    }

    if (srcElementType.isF16() && dstElementType.isF32()) {
      return llvm::to_vector(llvm::map_range(operands[0], [&](Value v) {
        return convertFp16ToFp32(loc, rewriter, v);
      }));
    }

    if (srcElementType.isF32() && dstElementType.isF16()) {
      assert(roundingMode.has_value() &&
             "rounding mode must be specified for fp32->fp16 conversion");
      SmallVector<Value> outVals;
      for (Value v : operands[0]) {
        outVals.push_back(
            convertFp32ToFp16(loc, rewriter, v, roundingMode.value()));
      }
      return outVals;
    }

    if (srcElementType.isF32() && dstElementType.isBF16()) {
      assert(roundingMode.has_value() &&
             "rounding mode must be specified for fp32->bf16 conversion");
      SmallVector<Value> outVals;
      for (Value v : operands[0]) {
        outVals.push_back(
            convertFp32ToBf16(loc, rewriter, v, roundingMode.value()));
      }
      return outVals;
    }

    bool useSoftwareFp8E4M3Downcast =
        computeCapability == 80 && llvm::isa<Float8E4M3FNType>(dstElementType);
    bool useFP16IntermediateSrc =
        srcElementType.isF32() && !useSoftwareFp8E4M3Downcast &&
        (!(computeCapability >= 89 &&
           (llvm::isa<Float8E4M3FNType, Float8E5M2Type>(dstElementType))) ||
         roundingMode.value() == RoundingMode::RTZ);
    bool isDstFP32 = dstElementType.isF32();
    Type srcType = useFP16IntermediateSrc ? f16_ty : srcElementType;
    Type dstType = isDstFP32 ? f16_ty : dstElementType;
    auto [cvtFunc, numElements] =
        getConversionFunc(srcType, dstType, roundingMode);
    SmallVector<Value> inVals;
    for (unsigned i = 0; i < std::min(numElements, operands.size()); i++) {
      inVals.push_back(operands[i][0]);
    }
    if (useFP16IntermediateSrc)
      for (Value &v : inVals)
        v = convertFp32ToFp16(loc, rewriter, v, RoundingMode::RTZ);
    inVals.resize(numElements, b.undef(typeConverter->convertType(srcType)));
    SmallVector<Value> outVals = cvtFunc(loc, rewriter, inVals);
    assert(outVals.size() == inVals.size());
    outVals.resize(std::min(numElements, operands.size()));
    if (isDstFP32)
      for (Value &v : outVals)
        v = convertFp16ToFp32(loc, rewriter, v);
    // Pack values
    return outVals;
  }

private:
  int computeCapability;
};

} // namespace
} // namespace gpu

} // namespace mlir::triton

SmallVector<Value> mlir::triton::ppu::convertS8ToBf16(
    Location loc, ConversionPatternRewriter &rewriter,
    const SmallVector<Value> &values, Type inType, Type outType,
    int computeCapability) {
  using namespace mlir::triton::gpu;

  assert(values.size() == 4);
  auto cvtFunc = makeConverterFromTIX(
      computeCapability >= 90 ? S8_to_Bf16_sm90 : S8_to_Bf16, inType,
      outType);
  SmallVector<Value> results = cvtFunc(loc, rewriter, values);
  assert(results.size() == values.size());
  return results;
}

void mlir::triton::ppu::populateFpCastOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, int computeCapability,
    PatternBenefit benefit) {
  using namespace mlir::triton::gpu;

  patterns.add<FpToFpOpConversion>(typeConverter, axisInfoAnalysis,
                                   computeCapability, benefit);
}
