// RUN: triton-opt %s -triton-tle-downgrade-invalid-async-copy | FileCheck %s

// CHECK-LABEL: tt.func @required_bf16_1x256
// CHECK: %[[SRC:.*]] = ttg.convert_layout %{{.*}} : tensor<1x256x!tt.ptr<bf16>, #{{.*}}> -> tensor<1x256x!tt.ptr<bf16>, #[[VEC:.*]]>
// CHECK: %[[TOKEN:.*]] = ttg.async_copy_global_to_local %[[SRC]], %{{.*}} {contiguity = 2 : i32, tle.required_async_copy}
// CHECK-SAME: tensor<1x256x!tt.ptr<bf16>, #[[VEC]]> -> <1x256xbf16
// CHECK: %[[COMMIT:.*]] = ttg.async_commit_group tokens %[[TOKEN]]
// CHECK: ttg.async_wait %[[COMMIT]] {num = 0 : i32}
// CHECK-NOT: tt.load
// CHECK-NOT: ttg.local_store

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 8], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
  tt.func @required_bf16_1x256(
      %input: tensor<1x256x!tt.ptr<bf16>, #blocked> {
        tt.contiguity = dense<[1, 256]> : tensor<2xi32>,
        tt.divisibility = dense<[1, 16]> : tensor<2xi32>
      },
      %view: !ttg.memdesc<1x256xbf16, #shared, #smem, mutable>) {
    %token = ttg.async_copy_global_to_local %input, %view {tle.required_async_copy} : tensor<1x256x!tt.ptr<bf16>, #blocked> -> <1x256xbf16, #shared, #smem, mutable>
    %commit = ttg.async_commit_group tokens %token
    ttg.async_wait %commit {num = 0 : i32}
    tt.return
  }
}
