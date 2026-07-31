// RUN: triton-opt %s --inline | FileCheck %s

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func private @local_pointer_worker(
      %arena: !ttg.memdesc<128xi8, #shared, #smem, mutable>,
      %index: i32) -> !tt.ptr<i8, 3> attributes {noinline = false} {
    %ptr = "tle.local_pointers"(%arena, %index) :
        (!ttg.memdesc<128xi8, #shared, #smem, mutable>, i32) -> !tt.ptr<i8, 3>
    tt.return %ptr : !tt.ptr<i8, 3>
  }

  // CHECK-NOT: tt.func private @local_pointer_worker
  // CHECK-LABEL: tt.func public @kernel
  // CHECK-NOT: tt.call
  // CHECK: "tle.local_pointers"
  // CHECK-NOT: tt.func private @local_pointer_worker
  tt.func public @kernel(%output: !tt.ptr<i8>) {
    %index = arith.constant 0 : i32
    %arena = ttg.local_alloc : () ->
        !ttg.memdesc<128xi8, #shared, #smem, mutable>
    %ptr = tt.call @local_pointer_worker(%arena, %index) :
        (!ttg.memdesc<128xi8, #shared, #smem, mutable>, i32) -> !tt.ptr<i8, 3>
    %value = tt.load %ptr : !tt.ptr<i8, 3>
    tt.store %output, %value : !tt.ptr<i8>
    tt.return
  }
}
