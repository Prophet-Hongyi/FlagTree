#include "tle/dialect/include/IR/Dialect.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/InliningUtils.h"
#include "tle/dialect/include/IR/Dialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "tle/dialect/include/IR/TleAttrDefs.cpp.inc"

#define GET_OP_CLASSES
#include "tle/dialect/include/IR/Ops.cpp.inc"

namespace mlir::triton::tle {
namespace {

struct TleInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  // TLE operations carry their dependencies explicitly through SSA values, so
  // cloning them into a Triton function needs no dialect-specific remapping.
  bool isLegalToInline(Operation *, Region *, bool,
                       IRMapping &) const final {
    return true;
  }
};

} // namespace

void TleDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "tle/dialect/include/IR/TleAttrDefs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "tle/dialect/include/IR/Ops.cpp.inc"
      >();
  addInterfaces<TleInlinerInterface>();
}
} // namespace mlir::triton::tle
