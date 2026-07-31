#include "triton/Dialect/TritonNvidiaGPU/IR/TMAEncodingUtils.h"

#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::nvidia_gpu {

FailureOr<TMASharedLayoutInfo> getTMASharedLayoutInfo(
    Attribute encoding, Type elementType,
    std::function<InFlightDiagnostic()> emitError) {
  (void)elementType;
  auto fail = [&](const Twine &message,
                  auto detail) -> FailureOr<TMASharedLayoutInfo> {
    if (emitError)
      emitError() << message << detail;
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

  return fail("encoding does not carry the complete TMA swizzle, element "
              "type, and allocation-alignment contract, got ",
              encoding);
}

} // namespace mlir::triton::nvidia_gpu
