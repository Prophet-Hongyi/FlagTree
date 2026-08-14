#pragma once
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/TMAEncodingUtils.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir::triton::nvidia_gpu {

constexpr inline int TMA_SIZE_BYTES = 128;
constexpr inline int TMA_ALIGN = 128;

inline bool isFp4Padded(Attribute encoding) {
  if (auto mmaEnc = dyn_cast<gpu::NVMMASharedEncodingAttr>(encoding))
    return mmaEnc.getFp4Padded();
  if (auto tmaEnc = dyn_cast<gpu::NVTMASharedEncodingAttr>(encoding))
    return tmaEnc.getFp4Padded();
  return false;
}

SmallVector<Value> translateTMAIndices(OpBuilder &builder, Location loc,
                                       Attribute encoding,
                                       SmallVector<Value> indices);

gpu::CTAEncodingAttr updateCTALayoutForShape(gpu::CTAEncodingAttr ctaLayout,
                                             ArrayRef<int64_t> shape);

gpu::SharedEncodingTrait
updateEncodingForShape(Operation *op, gpu::SharedEncodingTrait encoding,
                       RankedTensorType tensorType);

triton::gpu::SharedEncodingTrait
getEncodingFromDescriptor(Operation *op, RankedTensorType tensorType,
                          Value desc);

SmallVector<int64_t> getTMABlockShape(ArrayRef<int64_t> shapePerCTA,
                                      int elementBitWidth, int swizzleBytes,
                                      bool fp4Padded, bool transposed,
                                      bool packedSize);

inline SmallVector<int64_t> getTMABlockShape(Attribute encoding,
                                             Type elementType,
                                             ArrayRef<int64_t> shapePerCTA,
                                             bool packedSize) {
  auto info = getTMASharedLayoutInfo(encoding, elementType);
  if (failed(info))
    llvm::report_fatal_error("getTMABlockShape requires a TMA-compatible "
                             "shared encoding");
  return getTMABlockShape(shapePerCTA, info->elementBitWidth,
                          info->swizzlingByteWidth, info->fp4Padded,
                          info->transposed, packedSize);
}

inline SmallVector<int64_t> getTMABlockShape(RankedTensorType ty,
                                             bool packedSize) {
  auto shapePerCTA = gpu::getShapePerCTA(ty);
  return getTMABlockShape(ty.getEncoding(), ty.getElementType(), shapePerCTA,
                          packedSize);
}

inline SmallVector<int64_t> getTMABlockShape(triton::gpu::MemDescType ty,
                                             bool packedSize) {
  auto shapePerCTA = gpu::getShapePerCTA(ty);
  return getTMABlockShape(ty.getEncoding(), ty.getElementType(), shapePerCTA,
                          packedSize);
}

std::optional<int> getTMASwizzleMode(Operation *op, TensorDescType ty);

std::optional<int> getTMAElementType(Operation *op, TensorDescType ty);

LogicalResult createTMADesc(Value tmaPtr, MakeTensorDescOp op,
                            OpBuilder &builder);

} // namespace mlir::triton::nvidia_gpu
