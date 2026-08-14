#include "triton/Dialect/TritonNvidiaGPU/IR/TMAEncodingUtils.h"

#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::nvidia_gpu {

FailureOr<TMASharedLayoutInfo> getTMASharedLayoutInfo(
    Attribute encoding, Type elementType,
    std::function<InFlightDiagnostic()> emitError) {
  auto fail = [&](const Twine &message) -> FailureOr<TMASharedLayoutInfo> {
    if (emitError)
      emitError() << message;
    return failure();
  };

  if (auto nvmma =
          dyn_cast_or_null<gpu::NVMMASharedEncodingAttr>(encoding)) {
    return TMASharedLayoutInfo{
        nvmma.getSwizzlingByteWidth(),
        static_cast<unsigned>(nvmma.getAlignment()),
        nvmma.getTransposed(),
        nvmma.getElementBitWidth(),
        nvmma.getFp4Padded(),
    };
  }

  if (auto nvtma =
          dyn_cast_or_null<gpu::NVTMASharedEncodingAttr>(encoding)) {
    return TMASharedLayoutInfo{
        nvtma.getSwizzlingByteWidth(),
        static_cast<unsigned>(nvtma.getAlignment()),
        nvtma.getTransposed(),
        nvtma.getElementBitWidth(),
        nvtma.getFp4Padded(),
    };
  }

  auto shared = dyn_cast_or_null<gpu::SwizzledSharedEncodingAttr>(encoding);
  if (!shared) {
    if (emitError)
      emitError() << "unsupported shared encoding " << encoding;
    return failure();
  }

  // Generic shared layouts can describe arbitrary software address
  // permutations. TMA can write directly into the canonical, unswizzled
  // row-major subset without imposing an MMA storage layout.
  if (shared.getVec() != 1 || shared.getPerPhase() != 1 ||
      shared.getMaxPhase() != 1)
    return fail("generic TMA shared layouts must be unswizzled "
                "(vec=1, perPhase=1, maxPhase=1)");

  auto order = shared.getOrder();
  for (unsigned position = 0; position < order.size(); ++position) {
    if (order[position] != order.size() - position - 1)
      return fail("generic TMA shared layouts must use canonical row-major "
                  "order");
  }

  if (!elementType || !elementType.isIntOrFloat())
    return fail("generic TMA shared layouts require an integer or "
                "floating-point element type");

  return TMASharedLayoutInfo{
      /*swizzlingByteWidth=*/0,
      static_cast<unsigned>(shared.getAlignment()),
      /*transposed=*/false,
      elementType.getIntOrFloatBitWidth(),
      /*fp4Padded=*/false,
  };
}

} // namespace mlir::triton::nvidia_gpu
