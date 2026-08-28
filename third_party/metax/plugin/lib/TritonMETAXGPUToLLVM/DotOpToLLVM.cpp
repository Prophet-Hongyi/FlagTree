#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::MACAMmaEncodingAttr;

LogicalResult convertMMAMACA(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                             const LLVMTypeConverter *typeConverter,
                             ConversionPatternRewriter &rewriter);

namespace {
struct DotOpConversion : public ConvertOpToLLVMPattern<triton::DotOp> {
  using ConvertOpToLLVMPattern<triton::DotOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    // D = A * B + C
    Value A = op.getA();
    Value D = op.getResult();

    // Here we assume the DotOp's operands always comes from shared memory.
    auto AShapePerCTA = getShapePerCTA(A.getType());
    size_t reduceAxis = 1;
    unsigned K = AShapePerCTA[reduceAxis];
    bool isOuter = K == 1;

    MACAMmaEncodingAttr mmaLayout = dyn_cast<MACAMmaEncodingAttr>(
        cast<RankedTensorType>(D.getType()).getEncoding());
    if (mmaLayout) {
      if (!isOuter && supportMMA(op, mmaLayout.getVersionMajor(),
                                mmaLayout.getVersionMinor()))
        return convertMMAMACA(op, adaptor, getTypeConverter(), rewriter);
      return rewriter.notifyMatchFailure(
          op, "MMA dot operands were not legalized for the target");
    }

    if (isa<BlockedEncodingAttr>(
            cast<RankedTensorType>(D.getType()).getEncoding())) {
      auto aType = cast<RankedTensorType>(op.getA().getType()).getElementType();
      auto bType = cast<RankedTensorType>(op.getB().getType()).getElementType();
      auto dType = cast<RankedTensorType>(D.getType()).getElementType();
      if (aType != dType || bType != dType)
        return rewriter.notifyMatchFailure(
            op, "blocked dot operands must match the accumulator type");
      return convertFMADot(op, adaptor, getTypeConverter(), rewriter);
    }

    return rewriter.notifyMatchFailure(
        op, "unsupported dot layout when converting TritonGPU to LLVM");
  }
};
} // namespace

void mlir::triton::METAX::populateDotOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<DotOpConversion>(typeConverter, benefit);
}
