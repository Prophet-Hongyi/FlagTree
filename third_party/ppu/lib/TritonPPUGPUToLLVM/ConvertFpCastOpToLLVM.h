#ifndef TRITON_THIRD_PARTY_PPU_LIB_CONVERTFPCASTOPTOLLVM_H_
#define TRITON_THIRD_PARTY_PPU_LIB_CONVERTFPCASTOPTOLLVM_H_

#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::triton::ppu {

SmallVector<Value> convertS8ToBf16(Location loc,
                                   ConversionPatternRewriter &rewriter,
                                   const SmallVector<Value> &values,
                                   Type inType, Type outType,
                                   int computeCapability);

} // namespace mlir::triton::ppu

#endif // TRITON_THIRD_PARTY_PPU_LIB_CONVERTFPCASTOPTOLLVM_H_
