#include "tle/dialect/include/Conversion/TleToLLVM/FlagCxOpToLLVM/PutMemOrValueOpToLLVM.h"
#include "tle/dialect/include/Tools/FlagcxUtils.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/Support/raw_ostream.h"

namespace {
using namespace mlir;
namespace tle = mlir::triton::tle;

struct PutMemOrValueOpConversion
    : public ConvertOpToLLVMPattern<tle::PutMemOrValueOp> {
  PutMemOrValueOpConversion(LLVMTypeConverter &typeConverter,
                            PatternBenefit benefit)
      : ConvertOpToLLVMPattern(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(tle::PutMemOrValueOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::errs() << "[PutMemOrValueOpConversion]\n";
    return success();
  }
};

} // namespace

namespace mlir::triton::tle {

void populatePutMemOrValueOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                           RewritePatternSet &patterns,
                                           PatternBenefit benefit) {
  patterns.add<PutMemOrValueOpConversion>(typeConverter, benefit);
}
} // namespace mlir::triton::tle
