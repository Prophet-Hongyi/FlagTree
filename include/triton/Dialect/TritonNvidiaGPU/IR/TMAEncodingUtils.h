#ifndef TRITON_DIALECT_TRITONNVIDIAGPU_IR_TMAENCODINGUTILS_H_
#define TRITON_DIALECT_TRITONNVIDIAGPU_IR_TMAENCODINGUTILS_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Types.h"

#include <functional>

namespace mlir::triton::nvidia_gpu {

struct TMASharedLayoutInfo {
  unsigned swizzlingByteWidth;
  unsigned requiredAlignment;
  bool transposed;
  unsigned elementBitWidth;
  bool fp4Padded;
};

FailureOr<TMASharedLayoutInfo> getTMASharedLayoutInfo(
    Attribute encoding, Type elementType,
    std::function<InFlightDiagnostic()> emitError = {});

} // namespace mlir::triton::nvidia_gpu

#endif // TRITON_DIALECT_TRITONNVIDIAGPU_IR_TMAENCODINGUTILS_H_
