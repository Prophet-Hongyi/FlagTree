// RUN: triton-opt %s -split-input-file -mlir-print-local-scope -allow-unregistered-dialect -convert-warp-specialize-to-llvm -canonicalize=region-simplify=disabled | FileCheck %s

module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 8 : i32} {

llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

// CHECK-LABEL: @do_not_remat_special_register_capture
llvm.func @do_not_remat_special_register_capture() attributes {allocation.offset = 0 : i32} {
  // CHECK-DAG: [[C1:%.*]] = llvm.mlir.constant(1 : i32)
  // CHECK-DAG: [[C4:%.*]] = llvm.mlir.constant(4 : i32)
  // CHECK: [[CTAID:%.*]] = nvvm.read.ptx.sreg.ctaid.x
  // CHECK-NEXT: [[PID:%.*]] = llvm.udiv [[CTAID]], [[C4]] : i32
  // CHECK: ^bb4:
  // CHECK-NEXT: "llvm.nvvm.barrier.cta.sync.all"([[C1]])
  // CHECK-NOT: nvvm.read.ptx.sreg.ctaid.x
  // CHECK-NOT: llvm.load
  // CHECK-NEXT: "use"([[PID]])
  // CHECK-NOT: !llvm.struct<packed (i32)>
  %c4 = llvm.mlir.constant(4 : i32) : i32
  %ctaid = nvvm.read.ptx.sreg.ctaid.x : i32
  %pid = llvm.udiv %ctaid, %c4 : i32
  ttg.warp_specialize(%pid) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4>}
  default {
    ttg.warp_yield
  }
  partition0(%arg0: i32) num_warps(1) {
    "use"(%arg0) : (i32) -> ()
    ttg.warp_return
  } : (i32) -> ()
  llvm.return
}

}

// -----

module attributes {ttg.maxnreg = 168 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 8 : i32} {

llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

// CHECK-LABEL: @rematerialize_worker_warp_id_after_register_deallocation
llvm.func @rematerialize_worker_warp_id_after_register_deallocation() attributes {allocation.offset = 0 : i32} {
  // CHECK: nvvm.setmaxregister decrease 80
  // CHECK-NEXT: [[REL_WID:%.*]] = llvm.inline_asm has_side_effects {{.*}} "mov.u32 $0, %tid.x;\0A\09shr.u32 $0, $0, 5;\0A\09sub.u32 $0, $0, 4;", "=r" : () -> i32
  ttg.warp_specialize() attributes {actualRegisters = array<i32: 240, 24>, allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4>}
  default {
    ttg.warp_yield
  }
  partition0() num_warps(1) {
    ttg.warp_return
  } : () -> ()
  llvm.return
}

}

// -----

module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 8 : i32} {

llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

// CHECK-LABEL: @capture_byval_kernel_argument
llvm.func @capture_byval_kernel_argument(
    %arg0: !llvm.ptr {llvm.byval = !llvm.array<128 x i8>}) attributes {allocation.offset = 0 : i32, tle.warp_specialize_kernel_argument_table_offsets = array<i32: 8>} {
  // CHECK: [[INIT_PTR:%.*]] = llvm.getelementptr {{.*}}[8]
  // CHECK-NEXT: llvm.store %arg0, [[INIT_PTR]] {alignment = 8 : i64}
  // CHECK: [[RELOAD_PTR:%.*]] = llvm.getelementptr {{.*}}[8]
  // CHECK-NEXT: [[RELOAD:%.*]] = llvm.load volatile [[RELOAD_PTR]] {alignment = 8 : i64}
  // CHECK-NEXT: "use"([[RELOAD]])
  // CHECK-NOT: llvm.store %arg0
  ttg.warp_specialize(%arg0) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4>}
  default {
    ttg.warp_yield
  }
  partition0(%arg1: !llvm.ptr) num_warps(1) {
    "use"(%arg1) : (!llvm.ptr) -> ()
    ttg.warp_return
  } : (!llvm.ptr) -> ()
  llvm.return
}

}

// -----

module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 8 : i32} {

llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

// CHECK-LABEL: @reload_default_region_kernel_argument
llvm.func @reload_default_region_kernel_argument(
    %arg0: i32) attributes {allocation.offset = 0 : i32, tle.warp_specialize_kernel_argument_table_offsets = array<i32: 8>} {
  // CHECK: [[INIT_PTR:%.*]] = llvm.getelementptr {{.*}}[8]
  // CHECK-NEXT: llvm.store %arg0, [[INIT_PTR]] {alignment = 8 : i64}
  // CHECK: [[RELOAD_PTR:%.*]] = llvm.getelementptr {{.*}}[8]
  // CHECK-NEXT: [[RELOAD:%.*]] = llvm.load volatile [[RELOAD_PTR]] {alignment = 8 : i64}
  // CHECK-NEXT: "default_use"([[RELOAD]])
  ttg.warp_specialize() attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 4>}
  default {
    "default_use"(%arg0) : (i32) -> ()
    ttg.warp_yield
  }
  partition0() num_warps(1) {
    ttg.warp_return
  } : () -> ()
  llvm.return
}

}
