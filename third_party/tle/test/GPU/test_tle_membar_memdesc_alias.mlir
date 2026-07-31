// RUN: triton-opt %s --allocate-shared-memory-nv='compute-capability=90 ptx-version=81' --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=81' -reconcile-unrealized-casts | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @disjoint_memdesc_aliases
  // CHECK-NOT: nvvm.barrier0
  // CHECK: llvm.return
  tt.func @disjoint_memdesc_aliases(%v0: tensor<64xi8, #blocked>, %v1: tensor<64xi8, #blocked>) {
    %offs = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %arena = ttg.local_alloc : () -> !ttg.memdesc<128xi8, #shared, #smem, mutable>
    %left = tle.memdesc_alias %arena {offset_bytes = 0 : i64} : !ttg.memdesc<128xi8, #shared, #smem, mutable> -> !ttg.memdesc<64xi8, #shared, #smem, mutable>
    %right = tle.memdesc_alias %arena {offset_bytes = 64 : i64} : !ttg.memdesc<128xi8, #shared, #smem, mutable> -> !ttg.memdesc<64xi8, #shared, #smem, mutable>
    %ptr0 = "tle.local_pointers"(%left, %offs) : (!ttg.memdesc<64xi8, #shared, #smem, mutable>, tensor<64xi32, #blocked>) -> tensor<64x!tt.ptr<i8, 3>, #blocked>
    %ptr1 = "tle.local_pointers"(%right, %offs) : (!ttg.memdesc<64xi8, #shared, #smem, mutable>, tensor<64xi32, #blocked>) -> tensor<64x!tt.ptr<i8, 3>, #blocked>
    tt.store %ptr0, %v0 : tensor<64x!tt.ptr<i8, 3>, #blocked>
    tt.store %ptr1, %v1 : tensor<64x!tt.ptr<i8, 3>, #blocked>
    tt.return
  }

  // CHECK-LABEL: llvm.func @overlapping_memdesc_aliases
  // CHECK: nvvm.barrier0
  // CHECK: llvm.return
  tt.func @overlapping_memdesc_aliases(%v0: tensor<64xi8, #blocked>, %v1: tensor<64xi8, #blocked>) {
    %offs = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %arena = ttg.local_alloc : () -> !ttg.memdesc<128xi8, #shared, #smem, mutable>
    %left = tle.memdesc_alias %arena {offset_bytes = 0 : i64} : !ttg.memdesc<128xi8, #shared, #smem, mutable> -> !ttg.memdesc<64xi8, #shared, #smem, mutable>
    %overlap = tle.memdesc_alias %arena {offset_bytes = 32 : i64} : !ttg.memdesc<128xi8, #shared, #smem, mutable> -> !ttg.memdesc<64xi8, #shared, #smem, mutable>
    %ptr0 = "tle.local_pointers"(%left, %offs) : (!ttg.memdesc<64xi8, #shared, #smem, mutable>, tensor<64xi32, #blocked>) -> tensor<64x!tt.ptr<i8, 3>, #blocked>
    %ptr1 = "tle.local_pointers"(%overlap, %offs) : (!ttg.memdesc<64xi8, #shared, #smem, mutable>, tensor<64xi32, #blocked>) -> tensor<64x!tt.ptr<i8, 3>, #blocked>
    tt.store %ptr0, %v0 : tensor<64x!tt.ptr<i8, 3>, #blocked>
    tt.store %ptr1, %v1 : tensor<64x!tt.ptr<i8, 3>, #blocked>
    tt.return
  }

  // Dynamic indices conservatively cover their typed aliases, not the entire
  // backing arena.
  // CHECK-LABEL: llvm.func @dynamic_indices_keep_disjoint_aliases
  // CHECK-NOT: nvvm.barrier0
  // CHECK: llvm.return
  tt.func @dynamic_indices_keep_disjoint_aliases(%indices: tensor<64xi32, #blocked>, %v0: tensor<64xi8, #blocked>, %v1: tensor<64xi8, #blocked>) {
    %arena = ttg.local_alloc : () -> !ttg.memdesc<128xi8, #shared, #smem, mutable>
    %left = tle.memdesc_alias %arena {offset_bytes = 0 : i64} : !ttg.memdesc<128xi8, #shared, #smem, mutable> -> !ttg.memdesc<64xi8, #shared, #smem, mutable>
    %right = tle.memdesc_alias %arena {offset_bytes = 64 : i64} : !ttg.memdesc<128xi8, #shared, #smem, mutable> -> !ttg.memdesc<64xi8, #shared, #smem, mutable>
    %ptr0 = "tle.local_pointers"(%left, %indices) : (!ttg.memdesc<64xi8, #shared, #smem, mutable>, tensor<64xi32, #blocked>) -> tensor<64x!tt.ptr<i8, 3>, #blocked>
    %ptr1 = "tle.local_pointers"(%right, %indices) : (!ttg.memdesc<64xi8, #shared, #smem, mutable>, tensor<64xi32, #blocked>) -> tensor<64x!tt.ptr<i8, 3>, #blocked>
    tt.store %ptr0, %v0 : tensor<64x!tt.ptr<i8, 3>, #blocked>
    tt.store %ptr1, %v1 : tensor<64x!tt.ptr<i8, 3>, #blocked>
    tt.return
  }
}
