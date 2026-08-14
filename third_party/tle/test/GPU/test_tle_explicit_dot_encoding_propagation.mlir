// RUN: triton-opt %s -convert-triton-to-tritongpu='target=cuda:90 num-warps=4' | FileCheck %s

#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#lhs = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#rhs = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>
#row_mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 4], instrShape = [16, 8]}>
#row_lhs = #ttg.dot_op<{opIdx = 0, parent = #row_mma, kWidth = 2}>
#row_rhs = #ttg.dot_op<{opIdx = 1, parent = #row_mma, kWidth = 2}>

module {
  // CHECK-LABEL: tt.func public @chained_dot_accumulator
  tt.func public @chained_dot_accumulator() attributes {noinline = false} {
    %lhs_value = arith.constant dense<0.0> : tensor<32x32xbf16>
    %lhs_encoded = tle.encoding %lhs_value {target_encoding = #lhs} : tensor<32x32xbf16> -> tensor<32x32xbf16>
    %rhs_value = arith.constant dense<0.0> : tensor<32x8xbf16>
    %rhs_encoded = tle.encoding %rhs_value {target_encoding = #rhs} : tensor<32x8xbf16> -> tensor<32x8xbf16>
    %acc_value = arith.constant dense<0.0> : tensor<32x8xf32>
    %acc_encoded = tle.encoding %acc_value {target_encoding = #mma} : tensor<32x8xf32> -> tensor<32x8xf32>
    // CHECK: %[[FIRST:.*]] = tt.dot %{{.*}}, %{{.*}}, %{{.*}} {{.*}} -> tensor<32x8xf32, #mma>
    %first = tt.dot %lhs_encoded, %rhs_encoded, %acc_encoded : tensor<32x32xbf16> * tensor<32x8xbf16> -> tensor<32x8xf32>
    // CHECK-NEXT: %[[SECOND:.*]] = tt.dot %{{.*}}, %{{.*}}, %[[FIRST]] {{.*}} -> tensor<32x8xf32, #mma>
    %second = tt.dot %lhs_encoded, %rhs_encoded, %first : tensor<32x32xbf16> * tensor<32x8xbf16> -> tensor<32x8xf32>
    // CHECK-NOT: ttg.convert_layout
    tt.return
  }

  // An explicit C layout determines the dot result layout, but must not leak
  // through the result's elementwise users into an unrelated rank-changing
  // mask expression.
  // CHECK-LABEL: tt.func public @dot_accumulator_boundary
  tt.func public @dot_accumulator_boundary(%mask_value: tensor<1xi1>) attributes {noinline = false} {
    %lhs_value = arith.constant dense<0.0> : tensor<1x32xbf16>
    %lhs_encoded = tle.encoding %lhs_value {target_encoding = #row_lhs} : tensor<1x32xbf16> -> tensor<1x32xbf16>
    %rhs_value = arith.constant dense<0.0> : tensor<32x8xbf16>
    %rhs_encoded = tle.encoding %rhs_value {target_encoding = #row_rhs} : tensor<32x8xbf16> -> tensor<32x8xbf16>
    %acc_value = arith.constant dense<0.0> : tensor<1x8xf32>
    %acc_encoded = tle.encoding %acc_value {target_encoding = #row_mma} : tensor<1x8xf32> -> tensor<1x8xf32>
    // CHECK: %[[DOT:.*]] = tt.dot %{{.*}}, %{{.*}}, %{{.*}} {{.*}} -> tensor<1x8xf32, #{{mma[0-9]+}}>
    %dot = tt.dot %lhs_encoded, %rhs_encoded, %acc_encoded : tensor<1x32xbf16> * tensor<32x8xbf16> -> tensor<1x8xf32>
    // CHECK: tt.expand_dims {{.*}} : tensor<1xi1, #{{.*}}> -> tensor<1x1xi1, #{{.*}}>
    %mask_row = tt.expand_dims %mask_value {axis = 1 : i32} : tensor<1xi1> -> tensor<1x1xi1>
    // The same row mask also feeds an independent rank-3 memory mask.
    // CHECK: tt.expand_dims {{.*}} : tensor<1x1xi1, #{{.*}}> -> tensor<1x1x1xi1, #{{.*}}>
    %mask_volume = tt.expand_dims %mask_row {axis = 2 : i32} : tensor<1x1xi1> -> tensor<1x1x1xi1>
    %memory_mask = tt.broadcast %mask_volume : tensor<1x1x1xi1> -> tensor<1x4x8xi1>
    %mask = tt.broadcast %mask_row : tensor<1x1xi1> -> tensor<1x8xi1>
    %zero = arith.constant dense<0.0> : tensor<1x8xf32>
    // CHECK: arith.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<1x8xi1, #{{.*}}>, tensor<1x8xf32, #{{.*}}>
    %selected = arith.select %mask, %dot, %zero : tensor<1x8xi1>, tensor<1x8xf32>
    tt.return
  }

  // CHECK-LABEL: tt.func public @dot_accumulator_loop
  tt.func public @dot_accumulator_loop() attributes {noinline = false} {
    %lb = arith.constant 0 : i32
    %ub = arith.constant 1 : i32
    %step = arith.constant 1 : i32
    %lhs_value = arith.constant dense<0.0> : tensor<1x32xbf16>
    %lhs_encoded = tle.encoding %lhs_value {target_encoding = #row_lhs} : tensor<1x32xbf16> -> tensor<1x32xbf16>
    %rhs_value = arith.constant dense<0.0> : tensor<32x8xbf16>
    %rhs_encoded = tle.encoding %rhs_value {target_encoding = #row_rhs} : tensor<32x8xbf16> -> tensor<32x8xbf16>
    %init = arith.constant dense<0.0> : tensor<1x8xf32>
    // CHECK: scf.for
    // CHECK-SAME: -> (tensor<1x8xf32, #{{mma[0-9]+}}>)
    %result = scf.for %index = %lb to %ub step %step iter_args(%acc = %init) -> (tensor<1x8xf32>) : i32 {
      %acc_encoded = tle.encoding %acc {target_encoding = #row_mma} : tensor<1x8xf32> -> tensor<1x8xf32>
      %next = tt.dot %lhs_encoded, %rhs_encoded, %acc_encoded : tensor<1x32xbf16> * tensor<32x8xbf16> -> tensor<1x8xf32>
      // CHECK: scf.yield {{.*}} : tensor<1x8xf32, #{{mma[0-9]+}}>
      scf.yield %next : tensor<1x8xf32>
    }
    tt.return
  }
}
