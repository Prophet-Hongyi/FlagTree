#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "llvm/ADT/PriorityWorklist.h"
#include <algorithm>
#include <unordered_set>

namespace ttg = mlir::triton::gpu;

namespace {

struct UseInfo {
  TypedValue<TensorDescType> descriptor;
  Operation *use;
  Attribute desiredSharedEncoding;
  SmallVector<int64_t> shape;
  ttg::CTAEncodingAttr ctaLayout;
};

static bool isTMACompatibleEncoding(Attribute enc, Type elementType) {
  auto info = mlir::triton::nvidia_gpu::getTMASharedLayoutInfo(enc,
                                                               elementType);
  return succeeded(info) && !info->transposed;
}

Attribute findLoadEncodingFromUsers(Operation *op) {
  // Ignore multiple users and just pick the first compatible layout
  for (auto use : op->getUsers()) {
    if (auto alloc = dyn_cast<ttg::LocalAllocOp>(use)) {
      auto enc = alloc.getType().getEncoding();
      if (isTMACompatibleEncoding(enc, alloc.getType().getElementType()))
        return enc;
    } else if (auto store = dyn_cast<ttg::LocalStoreOp>(use)) {
      auto enc = store.getDst().getType().getEncoding();
      if (isTMACompatibleEncoding(
              enc, store.getDst().getType().getElementType()))
        return enc;
    }
  }
  return {};
}

SmallVector<int64_t> expandToRank(ArrayRef<int64_t> shape, int rank) {
  SmallVector<int64_t> result(rank, 1);
  assert(shape.size() <= rank);
  auto rankDiff = rank - shape.size();
  std::copy(shape.begin(), shape.end(), result.begin() + rankDiff);
  return result;
}

std::optional<UseInfo> getUseInfo(Operation *op) {
  UseInfo info;
  info.use = op;
#ifdef __TLE__
  if (auto copy = dyn_cast<ttg::TMACopyOp>(op)) {
    Value descriptor;
    ttg::MemDescType memDescTy;
    if (isa<TensorDescType>(copy.getSrc().getType()) &&
        isa<ttg::MemDescType>(copy.getDst().getType())) {
      descriptor = copy.getSrc();
      memDescTy = cast<ttg::MemDescType>(copy.getDst().getType());
    } else if (isa<ttg::MemDescType>(copy.getSrc().getType()) &&
               isa<TensorDescType>(copy.getDst().getType())) {
      descriptor = copy.getDst();
      memDescTy = cast<ttg::MemDescType>(copy.getSrc().getType());
    } else {
      return std::nullopt;
    }

    auto descriptorTy = cast<TensorDescType>(descriptor.getType());
    unsigned descriptorRank = descriptorTy.getBlockType().getRank();
    if (memDescTy.getRank() > descriptorRank)
      return std::nullopt;

    Attribute encoding = memDescTy.getEncoding();
    if (!isTMACompatibleEncoding(encoding, memDescTy.getElementType()))
      return std::nullopt;

    info.descriptor = cast<TypedValue<TensorDescType>>(descriptor);
    info.desiredSharedEncoding = encoding;
    info.ctaLayout = ttg::getCTALayout(encoding);
    info.shape = expandToRank(memDescTy.getShape(), descriptorRank);
    return info;
  }
#endif
  if (auto load = dyn_cast<DescriptorLoadOp>(op)) {
    info.descriptor = load.getDesc();
    info.desiredSharedEncoding = findLoadEncodingFromUsers(op);
    auto encoding = info.desiredSharedEncoding ? info.desiredSharedEncoding
                                               : load.getType().getEncoding();
    info.ctaLayout = ttg::getCTALayout(encoding);
    auto shape = load.getResult().getType().getShape();
    auto rank = load.getDesc().getType().getBlockType().getRank();
    info.shape = expandToRank(shape, rank);
    return info;
  }
  if (auto gather = dyn_cast<DescriptorGatherOp>(op)) {
    info.descriptor = gather.getDesc();
    info.desiredSharedEncoding = findLoadEncodingFromUsers(op);
    auto encoding = info.desiredSharedEncoding ? info.desiredSharedEncoding
                                               : gather.getType().getEncoding();
    info.ctaLayout = ttg::getCTALayout(encoding);
    auto shape = gather.getResult().getType().getShape();
    auto rank = gather.getDesc().getType().getBlockType().getRank();
    info.shape = expandToRank(shape, rank);
    return info;
  }
  if (auto store = dyn_cast<DescriptorStoreLikeOpInterface>(op)) {
    info.descriptor = store.getDesc();
    auto encoding = store.getSrc().getType().getEncoding();
    info.ctaLayout = ttg::getCTALayout(encoding);
    auto shape = store.getSrc().getType().getShape();
    auto rank = store.getDesc().getType().getBlockType().getRank();
    info.shape = expandToRank(shape, rank);
    return info;
  }
  return std::nullopt;
}

struct EncodingInfo {
  Attribute desiredEncoding;
  ttg::CTAEncodingAttr ctaLayout;
  // Shape may be different from the descriptor block shape for gather/scatter
  // use case
  SmallVector<int64_t> shape;
  bool forcedToDefault = false;

  bool operator==(const EncodingInfo &other) const {
    return desiredEncoding == other.desiredEncoding &&
           ctaLayout == other.ctaLayout &&
           forcedToDefault == other.forcedToDefault && shape == other.shape;
  }
};

} // namespace

template <> struct std::hash<EncodingInfo> {
  size_t operator()(const EncodingInfo &einfo) const {
    return llvm::hash_combine(einfo.desiredEncoding, einfo.ctaLayout,
                              einfo.forcedToDefault,
                              ArrayRef<int64_t>(einfo.shape));
  }
};

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUOPTIMIZEDESCRIPTORENCODINGPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

const EncodingInfo *internEncoding(std::unordered_set<EncodingInfo> &encodings,
                                   EncodingInfo info) {
  return &*encodings.insert(info).first;
}

EncodingInfo combineEncodings(const EncodingInfo &lhs, const EncodingInfo &rhs,
                              unsigned rank) {
  EncodingInfo result;
  // Always propagate forcedToDefault
  result.forcedToDefault = lhs.forcedToDefault || rhs.forcedToDefault;

  if (result.forcedToDefault)
    return result;

  if (lhs.shape.empty() || lhs.shape == rhs.shape)
    result.shape = rhs.shape;
  else if (rhs.shape.empty())
    result.shape = lhs.shape;
  else {
    assert(lhs.shape.size() == rhs.shape.size());
    auto rank = lhs.shape.size();
    result.shape.reserve(rank);
    for (int i = 0; i < rank; ++i)
      result.shape.push_back(std::min(lhs.shape[i], rhs.shape[i]));
  }

  SetVector<ttg::CTAEncodingAttr> ctaLayouts;
  if (lhs.ctaLayout)
    ctaLayouts.insert(lhs.ctaLayout);
  if (rhs.ctaLayout)
    ctaLayouts.insert(rhs.ctaLayout);

  switch (ctaLayouts.size()) {
  case 2:
    // if we find clashing CTALayouts, fallback to default
    result.ctaLayout =
        ttg::CTAEncodingAttr::getDefault(lhs.ctaLayout.getContext(), rank);
    break;
  case 1:
    result.ctaLayout = ctaLayouts[0];
    break;
  default:
    break;
  }

  SetVector<Attribute> desiredEncodings;
  if (lhs.desiredEncoding)
    desiredEncodings.insert(lhs.desiredEncoding);
  if (rhs.desiredEncoding)
    desiredEncodings.insert(rhs.desiredEncoding);

  switch (desiredEncodings.size()) {
  case 2:
    // if we find clashing encodings, fallback to default
    result.forcedToDefault = true;
    break;
  case 1:
    result.desiredEncoding = desiredEncodings[0];
    break;
  default:
    break;
  }
  return result;
}

Attribute getFallbackSharedEncoding(RankedTensorType tensorType,
                                    ttg::CTAEncodingAttr ctaLayout,
                                    ArrayRef<int64_t> usageShape) {
  auto ctx = tensorType.getContext();
  SmallVector<unsigned> order;
  for (int i = tensorType.getRank() - 1; i >= 0; --i)
    order.push_back(i);

  ArrayRef<int64_t> shape =
      usageShape.empty() ? tensorType.getShape() : usageShape;
  if (!ctaLayout)
    ctaLayout = ttg::CTAEncodingAttr::getDefault(ctx, tensorType.getRank());
  else if (ctaLayout.getRank() != tensorType.getRank())
    ctaLayout = updateCTALayoutForShape(ctaLayout, shape);

  return ttg::NVTMASharedEncodingAttr::get(ctx, shape, order, ctaLayout,
                                           tensorType.getElementType(),
                                           /*fp4Padded*/ false);
}

TensorDescType getTensorDescTypeWithEncoding(Operation *op,
                                             RankedTensorType existingTy,
                                             Attribute encoding) {
  auto sharedEnc = cast<triton::gpu::SharedEncodingTrait>(encoding);
  encoding = updateEncodingForShape(op, sharedEnc, existingTy);
  auto blockTy = existingTy.cloneWithEncoding(encoding);
  return TensorDescType::get(existingTy.getContext(), blockTy);
}

#ifdef __TLE__
SmallVector<Value> getWarpSpecializeTiedDescValues(ttg::WarpSpecializeOp wsOp,
                                                   unsigned operandNumber) {
  SmallVector<Value> values;
  auto capture = wsOp.getExplicitCaptures()[operandNumber];
  if (isa<TensorDescType>(capture.getType()))
    values.push_back(capture);
  for (Region *region : wsOp.getPartitionRegions()) {
    auto arg = region->getArgument(operandNumber);
    if (isa<TensorDescType>(arg.getType()))
      values.push_back(arg);
  }
  return values;
}

void syncWarpSpecializePartitionArgTypes(FuncOp func) {
  func.walk([](ttg::WarpSpecializeOp wsOp) {
    auto captures = wsOp.getExplicitCaptures();
    for (Region *region : wsOp.getPartitionRegions()) {
      for (auto [i, capture] : llvm::enumerate(captures))
        region->getArgument(i).setType(capture.getType());
    }
  });
}
#endif

void assignMemoryLayouts(ModuleOp &mod) {
  std::unordered_set<EncodingInfo> encodings;
  llvm::MapVector<TypedValue<TensorDescType>, const EncodingInfo *>
      valueToEncodingInfo;
  llvm::PriorityWorklist<TypedValue<triton::TensorDescType>> worklist;
  llvm::DenseMap<Value, SmallVector<Value>> equivalentValues;

  auto updateEncoding = [&](ArrayRef<Value> descValues, EncodingInfo info) {
    for (auto value : descValues) {
      auto typedVal = cast<TypedValue<TensorDescType>>(value);
      auto itr = valueToEncodingInfo.find(typedVal);
      if (itr != valueToEncodingInfo.end())
        info = combineEncodings(*itr->second, info,
                                typedVal.getType().getBlockType().getRank());
    }

    auto einfo = internEncoding(encodings, info);
    for (auto value : descValues) {
      auto typedVal = cast<TypedValue<TensorDescType>>(value);
      auto res = valueToEncodingInfo.try_emplace(typedVal, einfo);
      if (res.second) {
        worklist.insert(typedVal);
      } else if (res.first->second != einfo) {
        res.first->second = einfo;
        worklist.insert(typedVal);
      }
    }
  };

  auto connectEquivalentValues = [&](Value lhs, Value rhs) {
    if (!isa<TensorDescType>(lhs.getType()) ||
        !isa<TensorDescType>(rhs.getType()))
      return;
    equivalentValues[lhs].push_back(rhs);
    equivalentValues[rhs].push_back(lhs);
    updateEncoding({lhs, rhs}, EncodingInfo{});
  };

  // 1. Seed descriptor uses across the whole module. Function boundaries are
  // connected below instead of forcing every device function to the fallback
  // layout.
  for (auto func : mod.getOps<FuncOp>()) {
    if (!func.isExternal()) {
      for (auto blockArg : func.getBlocks().front().getArguments())
        if (auto desc = dyn_cast<TypedValue<TensorDescType>>(blockArg))
          updateEncoding({desc}, EncodingInfo{});
    }

    func.walk([&](Operation *op) {
      if (auto info = getUseInfo(op)) {
        updateEncoding(
            info->descriptor,
            EncodingInfo{info->desiredSharedEncoding, info->ctaLayout,
                         info->shape});
      } else {
        bool forcedToDefault = isa<ReinterpretTensorDescOp>(op);

#ifdef __TLE__
        if (auto wsOp = dyn_cast<ttg::WarpSpecializeOp>(op)) {
          for (auto [i, capture] :
               llvm::enumerate(wsOp.getExplicitCaptures())) {
            if (!isa<TensorDescType>(capture.getType()))
              continue;
            updateEncoding(getWarpSpecializeTiedDescValues(wsOp, i),
                           EncodingInfo{});
          }
        }
#endif

        EncodingInfo defaultInfo{{}, {}, {}, forcedToDefault};
        for (auto result : op->getResults())
          if (auto desc = dyn_cast<TypedValue<TensorDescType>>(result))
            updateEncoding({desc}, defaultInfo);

        for (auto arg : op->getOperands())
          if (auto desc = dyn_cast<TypedValue<TensorDescType>>(arg))
            updateEncoding({desc}, defaultInfo);
      }
    });
  }

  // A tensor descriptor's shared encoding is part of the function ABI. Tie
  // direct-call operands/results to the corresponding callee values so all
  // call sites and the callee are solved together.
  mod.walk([&](CallOp call) {
    auto callee = mod.lookupSymbol<FuncOp>(call.getCallee());
    if (!callee || callee.isExternal()) {
      EncodingInfo fallback{{}, {}, {}, /*forcedToDefault=*/true};
      for (Value value : call->getOperands())
        if (isa<TensorDescType>(value.getType()))
          updateEncoding({value}, fallback);
      for (Value value : call->getResults())
        if (isa<TensorDescType>(value.getType()))
          updateEncoding({value}, fallback);
      return;
    }

    Block &entry = callee.getBlocks().front();
    for (auto [operand, argument] :
         llvm::zip(call.getOperands(), entry.getArguments()))
      connectEquivalentValues(operand, argument);

    SmallVector<ReturnOp> returns;
    callee.walk([&](ReturnOp ret) { returns.push_back(ret); });
    for (auto [resultIndex, result] : llvm::enumerate(call.getResults())) {
      if (!isa<TensorDescType>(result.getType()))
        continue;
      for (ReturnOp ret : returns)
        connectEquivalentValues(result, ret.getOperand(resultIndex));
    }
  });

  // Functions with no call sites still need all return sites to agree on one
  // descriptor result ABI.
  for (auto func : mod.getOps<FuncOp>()) {
    if (func.isExternal())
      continue;
    SmallVector<ReturnOp> returns;
    func.walk([&](ReturnOp ret) { returns.push_back(ret); });
    for (unsigned resultIndex = 0;
         resultIndex < func.getFunctionType().getNumResults(); ++resultIndex) {
      Value representative;
      for (ReturnOp ret : returns) {
        Value value = ret.getOperand(resultIndex);
        if (!isa<TensorDescType>(value.getType()))
          continue;
        if (representative)
          connectEquivalentValues(representative, value);
        else
          representative = value;
      }
    }
  }

  // 2. Propagate encoding info through the graph until fixed point
  while (!worklist.empty()) {
    auto desc = worklist.pop_back_val();

    if (auto equivalent = equivalentValues.find(desc);
        equivalent != equivalentValues.end()) {
      for (Value value : equivalent->second)
        updateEncoding({desc, value}, EncodingInfo{});
    }

    // Propagate to users
    for (OpOperand &use : desc.getUses()) {
      auto op = use.getOwner();
      if (isa<scf::ForOp, scf::WhileOp>(op)) {
        auto offset = 3 * isa<scf::ForOp>(op);
        auto vals = getTiedArgs(op, use.getOperandNumber() - offset);
        updateEncoding(vals, EncodingInfo{});
      } else if (isa<scf::YieldOp>(op)) {
        auto vals = getTiedArgs(op->getParentOp(), use.getOperandNumber());
        updateEncoding(vals, EncodingInfo{});
#ifdef __TLE__
      } else if (auto wsOp = dyn_cast<ttg::WarpSpecializeOp>(op)) {
        auto vals =
            getWarpSpecializeTiedDescValues(wsOp, use.getOperandNumber());
        updateEncoding(vals, EncodingInfo{});
#endif
      }
    }

    // Propagate to defining ops
    if (auto opResult = dyn_cast<OpResult>(desc)) {
      auto definingOp = opResult.getOwner();
      if (isa<scf::ForOp, scf::WhileOp, scf::IfOp>(definingOp)) {
        auto vals = getTiedArgs(definingOp, opResult.getResultNumber());
        updateEncoding(vals, EncodingInfo{});
      }
    } else if (auto blockArg = dyn_cast<BlockArgument>(desc)) {
      auto parentOp = blockArg.getOwner()->getParentOp();
      if (isa<scf::ForOp, scf::WhileOp>(parentOp)) {
        auto offset = isa<scf::ForOp>(parentOp);
        auto vals = getTiedArgs(parentOp, blockArg.getArgNumber() - offset);
        updateEncoding(vals, EncodingInfo{});
#ifdef __TLE__
      } else if (auto partitions =
                     dyn_cast<ttg::WarpSpecializePartitionsOp>(parentOp)) {
        auto wsOp = cast<ttg::WarpSpecializeOp>(partitions->getParentOp());
        auto vals =
            getWarpSpecializeTiedDescValues(wsOp, blockArg.getArgNumber());
        updateEncoding(vals, EncodingInfo{});
#endif
      }
    }
  }

  // 3. Transfer propagated encodings into the graph, then synchronize every
  // function and call ABI from the rewritten SSA value types.
  for (auto &[desc, einfo] : valueToEncodingInfo) {
    auto existingTy = desc.getType().getBlockType();
    Attribute newEncoding;
    if (einfo->desiredEncoding) {
      newEncoding = einfo->desiredEncoding;
    } else if (einfo->forcedToDefault) {
      newEncoding = getFallbackSharedEncoding(existingTy, {}, {});
    } else {
      newEncoding =
          getFallbackSharedEncoding(existingTy, einfo->ctaLayout, einfo->shape);
    }
    desc.setType(getTensorDescTypeWithEncoding(desc.getDefiningOp(), existingTy,
                                               newEncoding));
  }

  auto ctx = mod.getContext();
  for (auto func : mod.getOps<FuncOp>()) {
#ifdef __TLE__
    syncWarpSpecializePartitionArgTypes(func);
#endif
    SmallVector<Type> argTys(func.getFunctionType().getInputs());
    if (!func.isExternal())
      argTys.assign(func.getBlocks().front().getArgumentTypes().begin(),
                    func.getBlocks().front().getArgumentTypes().end());

    SmallVector<Type> resultTys(func.getResultTypes());
    if (!func.isExternal()) {
      SmallVector<ReturnOp> returns;
      func.walk([&](ReturnOp ret) { returns.push_back(ret); });
      for (auto [resultIndex, resultTy] : llvm::enumerate(resultTys)) {
        auto descTy = dyn_cast<TensorDescType>(resultTy);
        if (!descTy)
          continue;

        bool foundReturnType = false;
        for (ReturnOp ret : returns) {
          if (auto returnTy =
                  dyn_cast<TensorDescType>(ret.getOperand(resultIndex).getType())) {
            resultTys[resultIndex] = returnTy;
            foundReturnType = true;
            break;
          }
        }
        if (!foundReturnType) {
          auto encoding =
              getFallbackSharedEncoding(descTy.getBlockType(), {}, {});
          resultTys[resultIndex] = getTensorDescTypeWithEncoding(
              nullptr, descTy.getBlockType(), encoding);
        }
      }
    }
    func.setFunctionType(FunctionType::get(ctx, argTys, resultTys));
  }
}

} // anonymous namespace

class TritonNvidiaGPUOptimizeDescriptorEncodingPass
    : public impl::TritonNvidiaGPUOptimizeDescriptorEncodingPassBase<
          TritonNvidiaGPUOptimizeDescriptorEncodingPass> {
public:
  using BaseT = TritonNvidiaGPUOptimizeDescriptorEncodingPassBase<
      TritonNvidiaGPUOptimizeDescriptorEncodingPass>;
  using BaseT::BaseT;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();
    assignMemoryLayouts(m);
  }
};

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
