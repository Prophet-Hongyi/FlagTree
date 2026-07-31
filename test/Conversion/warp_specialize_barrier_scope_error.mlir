// RUN: not triton-opt %s -convert-warp-specialize-to-llvm 2>&1 | FileCheck %s

module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 5 : i32} {

llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

// CHECK: error: barrier-bearing function is called from incompatible execution scopes
// CHECK: note: called from warp-group scope with barrier 0 and 128 participating threads
// CHECK: note: called from warp-group scope with barrier 2 and 32 participating threads
llvm.func internal @shared_barrier_helper() attributes {sym_visibility = "private"} {
  nvvm.barrier0
  llvm.return
}

llvm.func @incompatible_barrier_scopes() attributes {allocation.offset = 32 : i32} {
  ttg.warp_specialize() attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4>}
  default {
    llvm.call @shared_barrier_helper() : () -> ()
    ttg.warp_yield
  }
  partition0() num_warps(1) {
    llvm.call @shared_barrier_helper() : () -> ()
    ttg.warp_return
  } : () -> ()
  llvm.return
}

}
