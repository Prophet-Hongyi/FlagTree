/*
 * Copyright 2025-     FlagOS Contributors
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

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include <cctype>
#include <limits>

#include "tle/dialect/include/IR/VerfiyUtils.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include <iostream>

namespace mlir::triton::tle {
namespace RemotePointers {
llvm::LogicalResult verifyDeviceSpace(mlir::Value src, mlir::Value result) {
  if (!src)
    return success();

  if (auto tensorTy = dyn_cast<RankedTensorType>(result.getType())) {
    auto ptr = dyn_cast<triton::PointerType>(tensorTy.getElementType());
    if (!ptr)
      return failure();
    return success();
  }
  return success();
}

llvm::LogicalResult verifyNodeSpace(mlir::Operation *op, mlir::Value src,
                                    mlir::Value comm, mlir::Value result) {
  if (!src)
    return op->emitOpError() << "node space remote pointers require dev_mem";
  if (!comm)
    return op->emitOpError() << "node space remote pointers require dev_comm";
  if (!src.getType().isSignlessInteger(64))
    return op->emitOpError() << "node space dev_mem must be a scalar i64";
  if (!comm.getType().isSignlessInteger(64))
    return op->emitOpError() << "node space dev_comm must be a scalar i64";

  auto resultPtrTy = dyn_cast<triton::PointerType>(result.getType());
  if (!resultPtrTy || resultPtrTy.getAddressSpace() != 1)
    return op->emitOpError()
           << "node space result must be a global tt.ptr (addrspace=1)";

  return success();
}
} // namespace RemotePointers

namespace DistributedBarrier {
llvm::LogicalResult verifyDeviceSpace(mlir::Operation *op, mlir::Value src) {

  auto kindAttr = op->getAttrOfType<StringAttr>("group_kind");
  auto barrierTypeAttr = op->getAttrOfType<StringAttr>("barrier_type");
  auto orderAttr = op->getAttrOfType<StringAttr>("order");

  if (kindAttr && barrierTypeAttr && orderAttr)
    return success();
  else
    return op->emitOpError()
           << "expects src, group_kind, barrier_type and order attributes to "
              "be present for device space distributed barrier";
}

} // namespace DistributedBarrier

} // namespace mlir::triton::tle
