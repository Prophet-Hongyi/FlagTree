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

#include "ConvertFpCastOpToLLVM.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "TritonPPUGPUToLLVM/TIXAsmFormat.h"
#include "Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Support/LLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVMBase.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir::triton::gpu;
using namespace mlir::triton::ppu;

namespace mlir::triton {

namespace gpu {
namespace {

struct FDivOpConversion
    : ElementwiseOpConversionBase<arith::DivFOp, FDivOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::DivFOp, FDivOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(arith::DivFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    unsigned bitwidth = getIntOrFloatOrPtrBitWidth(elemTy);
    TIXBuilder builder;
    StringRef tix;
    Type resultTy;
    if (32 == bitwidth) {
      tix = "ppu.div.full.f32";
      resultTy = f32_ty;
    } else if (64 == bitwidth) {
      tix = "ppu.div.rn.f64";
      resultTy = f64_ty;
    } else {
      llvm::report_fatal_error("Unsupported bitwidth");
    }
    auto &div = *builder.create(tix.str());
    auto res = builder.newOperand(resultTy == f32_ty ? "=f" : "=d");
    auto lhs = builder.newOperand(operands[0][0], elemTy == f32_ty ? "f" : "d");
    auto rhs = builder.newOperand(operands[0][1], elemTy == f32_ty ? "f" : "d");
    div(res, lhs, rhs);
    return {builder.launch(rewriter, loc, resultTy, false)};
  }
};

struct PreciseSqrtOpConversion
    : ElementwiseOpConversionBase<PreciseSqrtOp, PreciseSqrtOpConversion> {
  using Base =
      ElementwiseOpConversionBase<PreciseSqrtOp, PreciseSqrtOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(PreciseSqrtOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    TIXBuilder builder;
    auto &sqrt = *builder.create("ppu.sqrt.rn.f32");
    auto res = builder.newOperand("=f");
    auto operand = builder.newOperand(operands[0][0], "f");
    sqrt(res, operand);
    return {builder.launch(rewriter, loc, elemTy, false)};
  }
};

struct PreciseDivFOpConversion
    : ElementwiseOpConversionBase<PreciseDivFOp, PreciseDivFOpConversion> {
  using Base =
      ElementwiseOpConversionBase<PreciseDivFOp, PreciseDivFOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(PreciseDivFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    TIXBuilder builder;
    auto &div = *builder.create("ppu.div.rn.f32");
    auto res = builder.newOperand("=f");
    auto lhs = builder.newOperand(operands[0][0], "f");
    auto rhs = builder.newOperand(operands[0][1], "f");
    div(res, lhs, rhs);
    return {builder.launch(rewriter, loc, elemTy, false)};
  }
};

// Uses inline tix to convert s8/u8 to bf16, since the
struct SIToFPOpConversion
    : ElementwiseOpConversionBase<arith::SIToFPOp, SIToFPOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::SIToFPOp, SIToFPOpConversion>;
  using Adaptor = typename Base::OpAdaptor;

  explicit SIToFPOpConversion(LLVMTypeConverter &typeConverter,
                              ModuleAxisInfoAnalysis &axisAnalysisPass,
                              int computeCapability,
                              PatternBenefit benefit = patternBenefitDefault)
      : ElementwiseOpConversionBase(typeConverter, axisAnalysisPass, benefit),
        computeCapability(computeCapability) {}

  SmallVector<Value> createDestOps(arith::SIToFPOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    Type inElemTy = getElementType(op.getIn());
    Type outElemTy = getElementType(op.getOut());
    if (outElemTy.isBF16() && inElemTy.isInteger(8) && operands.size() >= 4) {
      SmallVector<Value> inVals = {operands[0][0], operands[1][0],
                                   operands[2][0], operands[3][0]};
      return convertS8ToBf16(
          loc, rewriter, inVals, getTypeConverter()->convertType(inElemTy),
          getTypeConverter()->convertType(outElemTy), computeCapability);
    } else {
      return {LLVM::SIToFPOp::create(rewriter, loc, elemTy, operands[0][0])};
    }
  }

private:
  int computeCapability;
};

struct FPToSIOpConversion
    : ElementwiseOpConversionBase<arith::FPToSIOp, FPToSIOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::FPToSIOp, FPToSIOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(arith::FPToSIOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto inElemTy = getElementType(op.getIn());
    return {LLVM::FPToSIOp::create(rewriter, loc, elemTy, operands[0][0])};
  }
};

struct ExpOpConversionApprox
    : ElementwiseOpConversionBase<math::ExpOp, ExpOpConversionApprox> {
  using Base = ElementwiseOpConversionBase<math::ExpOp, ExpOpConversionApprox>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(math::ExpOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    // For non-FP32 input, call __ppu_expf for higher-precision calculation
    if (getIntOrFloatOrPtrBitWidth(elemTy) != 32)
      return {};

    const double log2e = 1.4426950408889634;
    Value prod = b.fmul(f32_ty, operands[0][0], b.f32_val(log2e));

    Type resultTy = operands[0][0].getType();
    TIXBuilder builder;
    auto &ex2 = *builder.create("ppu.ex2.approx.f32");
    auto res = builder.newOperand("=f");
    auto operand = builder.newOperand(prod, "f");
    ex2(res, operand);
    return {builder.launch(rewriter, loc, resultTy, false)};
  }
};

struct ClampFOpConversion
    : ElementwiseOpConversionBase<ClampFOp, ClampFOpConversion> {
  using Base = ElementwiseOpConversionBase<ClampFOp, ClampFOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  explicit ClampFOpConversion(LLVMTypeConverter &typeConverter,
                              ModuleAxisInfoAnalysis &axisAnalysisPass,
                              int computeCapability,
                              PatternBenefit benefit = patternBenefitDefault)
      : ElementwiseOpConversionBase(typeConverter, axisAnalysisPass, benefit),
        computeCapability(computeCapability) {}

  bool isClipPattern(ClampFOp op) const {
    bool xorsignAbsAvailable = (computeCapability >= 90);
    // Pattern matching the sequence of clamp(x, -limit, limit) to generate
    // more efficient TIX code. NOTE: This pattern matching is not general
    // enough, but it is sufficient. We detect only two cases here:
    // 1. where the "-limit" is computed as 0 - limit:
    //   %cst = arith.constant dense<0.000000e+00>
    //   %8 = tt.load %7, %2
    //   %11 = arith.subf %cst, %8
    //   %12 = tt.clamp %5, %11, %8
    // 2. where "-limit" and "limit" are constants.
    //   %cst_6 = arith.constant dense<-6.0000e+00>
    //   %cst_7 = arith.constant dense<6.0000e+00>
    //   %160 = tt.clamp %158, %cst_6, %cst_7
    bool patternFound = false;

    auto getSplatInitializer = [](Value v) -> std::optional<double> {
      if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
        if (auto attr = mlir::dyn_cast<DenseIntOrFPElementsAttr>(
                constOp.getValueAttr())) {
          if (attr.isSplat()) {
            return attr.getSplatValue<APFloat>().convertToDouble();
          }
        }
      }
      return std::nullopt;
    };

    if (xorsignAbsAvailable) {
      if (auto subOp = op.getOperand(1).getDefiningOp<arith::SubFOp>()) {
        if (subOp.getOperand(1) == op.getOperand(2)) {
          auto initializer = getSplatInitializer(subOp.getOperand(0));
          if (initializer.has_value() && initializer.value() == 0.0) {
            patternFound = true;
          }
        }
      } else {
        auto initializer1 = getSplatInitializer(op.getOperand(1));
        auto initializer2 = getSplatInitializer(op.getOperand(2));
        if (initializer1.has_value() && initializer2.has_value() &&
            initializer1.value() == -initializer2.value()) {
          patternFound = true;
        }
      }
    }
    return patternFound;
  }

  SmallVector<Value> emitOptimization(ClampFOp op,
                                      ConversionPatternRewriter &rewriter,
                                      Type elemTy,
                                      MultipleOperandsRange operands,
                                      Location loc) const {
    TIXBuilder builder;
    std::string tix = "ppu.min";
    if (op.getPropagateNan() == PropagateNan::ALL) {
      tix += ".NaN";
    }
    tix += ".xorsign.abs";
    if (elemTy.isF32()) {
      tix += ".f32";
    } else if (elemTy.isF16()) {
      tix += ".f16";
    }

    auto &minOp = *builder.create(tix);
    auto res = builder.newOperand(elemTy.isF32() ? "=f" : "=h");
    auto lhs = builder.newOperand(operands[0][0], elemTy.isF32() ? "f" : "h");
    auto rhs = builder.newOperand(operands[0][2], elemTy.isF32() ? "f" : "h");
    minOp(res, lhs, rhs);
    return {builder.launch(rewriter, loc, elemTy, false)};
  }

  SmallVector<Value> createDestOps(ClampFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    if (isClipPattern(op)) {
      return emitOptimization(op, rewriter, elemTy, operands, loc);
    }
    return {};
  }

private:
  int computeCapability;
};

template <typename TritonOp>
struct OpToExternCallConversion
    : public ElementwiseOpConversionBase<TritonOp,
                                         OpToExternCallConversion<TritonOp>> {
  using Base =
      ElementwiseOpConversionBase<TritonOp, OpToExternCallConversion<TritonOp>>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  explicit OpToExternCallConversion(LLVMTypeConverter &typeConverter,
                                    ModuleAxisInfoAnalysis &axisAnalysisPass,
                                    StringRef externFuncName,
                                    PatternBenefit benefit)
      : Base::ElementwiseOpConversionBase(typeConverter, axisAnalysisPass,
                                          benefit),
        funcName(externFuncName) {}

  SmallVector<Value> createDestOps(TritonOp op, Adaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    Type funcType = getFunctionType(elemTy, operands[0]);
    LLVM::LLVMFuncOp funcOp =
        appendOrGetExternFuncOp(rewriter, op, funcName, funcType);
    return {
        LLVM::createLLVMCallOp(rewriter, loc, funcOp, operands[0]).getResult()};
  }

private:
  StringRef funcName;
};
} // namespace
} // namespace gpu

} // namespace mlir::triton

void mlir::triton::ppu::populateElementwiseOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, int computeCapability,
    const TargetInfo &targetInfo, PatternBenefit benefit) {
  using namespace mlir::triton::gpu;

  mlir::triton::populateElementwiseOpToLLVMPatterns(
      typeConverter, patterns, axisInfoAnalysis, targetInfo, benefit);

#define POPULATE_OP(SRC_OP, DST_OP)                                            \
  patterns.add<ElementwiseOpConversion<SRC_OP, DST_OP>>(                       \
      typeConverter, axisInfoAnalysis, benefit)

  POPULATE_OP(arith::SubFOp, LLVM::FSubOp);
  POPULATE_OP(arith::AddFOp, LLVM::FAddOp);
  POPULATE_OP(arith::MulFOp, LLVM::FMulOp);

  POPULATE_OP(arith::ExtFOp, LLVM::FPExtOp);
  POPULATE_OP(arith::TruncFOp, LLVM::FPTruncOp);

#undef POPULATE_OP

  patterns.add<PreciseSqrtOpConversion>(typeConverter, axisInfoAnalysis,
                                        benefit);
  patterns.add<PreciseDivFOpConversion>(typeConverter, axisInfoAnalysis,
                                        benefit);
  patterns.add<FDivOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<FPToSIOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<SIToFPOpConversion>(typeConverter, axisInfoAnalysis,
                                   computeCapability, benefit);
  // ExpOpConversionApprox will try using ex2.approx if the input type is
  // FP32. For other input types, ExpOpConversionApprox will return failure and
  // ElementwiseOpConversion<math::ExpOp, math::ExpOp> defined below will call
  // __ppu_expf for higher-precision calculation
  patterns.add<ExpOpConversionApprox>(typeConverter, axisInfoAnalysis, benefit);
  bool hwNanPropagationSupported = computeCapability >= 80;
  mlir::triton::populateMinMaxFOpToLLVMPattern(
      typeConverter, patterns, axisInfoAnalysis, hwNanPropagationSupported,
      benefit);
  mlir::triton::populateClampFOpToLLVMPattern(
      typeConverter, patterns, axisInfoAnalysis, targetInfo, benefit);
}

void mlir::triton::ppu::populateClampFOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, int computeCapability,
    PatternBenefit benefit) {
  using namespace mlir::triton::gpu;

  patterns.add<ClampFOpConversion>(typeConverter, axisInfoAnalysis,
                                   computeCapability, benefit);
}
