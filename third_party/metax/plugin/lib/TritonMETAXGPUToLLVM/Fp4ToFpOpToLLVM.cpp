#include "PatternTritonGPUOpToLLVM.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace {

Value decodeE2M1Nibble(RewriterBase &rewriter, Location loc, Value nibble,
                       bool toFp16) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  int toMantissaBits = toFp16 ? 10 : 7;
  int toBias = toFp16 ? 15 : 127;
  int toPoint5 = toFp16 ? 0x3800 : 0x3f00;

  Value exponentMantissa = b.and_(nibble, b.i8_val(0x7));
  Value sign = b.and_(nibble, b.i8_val(0x8));
  Value exponentMantissa16 = b.zext(i16_ty, exponentMantissa);
  Value sign16 = b.zext(i16_ty, sign);

  // E2M1 is S.EE.M.  Construct the target FP16/BF16 bit pattern directly so
  // the lowering needs no target-specific FP4 instruction.
  Value bits = b.or_(
      b.shl(exponentMantissa16, b.i16_val(toMantissaBits - 1)),
      b.shl(sign16, b.i16_val(12)));
  Value normal = b.icmp_ne(b.and_(exponentMantissa, b.i8_val(0x6)),
                           b.i8_val(0));
  bits = b.select(
      normal,
      b.add(bits, b.i16_val((toBias - 1) << toMantissaBits)), bits);

  // +/-0.5 is the only E2M1 subnormal value.  Zero retains its sign bit.
  Value signBit = b.and_(bits, b.i16_val(0x8000));
  Value point5 = b.or_(b.i16_val(toPoint5), signBit);
  bits = b.select(b.icmp_eq(exponentMantissa, b.i8_val(0x1)), point5, bits);

  // MetaX's LLVM type converter represents BF16 register elements as i16.
  // FP16 remains a native LLVM half, while BF16 keeps the constructed bit
  // pattern in its ABI container and is interpreted as BF16 by later ops.
  return toFp16 ? b.bitcast(bits, f16_ty) : bits;
}

struct Fp4ToFpOpPattern : public ConvertOpToLLVMPattern<Fp4ToFpOp> {
  Fp4ToFpOpPattern(LLVMTypeConverter &typeConverter, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<Fp4ToFpOp>(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(Fp4ToFpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto elemType = op.getType().getElementType();
    assert((elemType == f16_ty || elemType == bf16_ty) &&
           "Fp4ToFpOp only supports fp16/bf16 results");
    bool toFp16 = elemType == f16_ty;

    auto packedValues = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    SmallVector<Value> results;
    results.reserve(packedValues.size() * 2);
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    for (Value packed : packedValues) {
      Value low = b.and_(packed, b.i8_val(0x0f));
      Value high = b.lshr(packed, b.i8_val(4));
      results.push_back(decodeE2M1Nibble(rewriter, loc, low, toFp16));
      results.push_back(decodeE2M1Nibble(rewriter, loc, high, toFp16));
    }

    Value result = packLLElements(loc, getTypeConverter(), results, rewriter,
                                  op.getType());
    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

void mlir::triton::METAX::populateFp4ToFpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<Fp4ToFpOpPattern>(typeConverter, benefit);
}
