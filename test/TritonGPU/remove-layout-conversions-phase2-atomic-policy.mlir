// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions="enable-rlc-enhance=true rlc-phase-mask=5" | FileCheck %s --check-prefixes=ALLOW,PRESERVE,PROFIT-REJECT,DENSITY-REJECT,CONTEXT-REJECT
// RUN: env FLAGTREE_RLC_TRACE_REJECTS=1 triton-opt %s -split-input-file -tritongpu-remove-layout-conversions="enable-rlc-enhance=true rlc-phase-mask=5" 2>&1 | FileCheck %s --check-prefix=POLICY-TRACE

// POLICY-TRACE-DAG: FLAGTREE_RLC_TRACE phase=2 outcome=reject reason=profitability-score-below-threshold{{.*}}online_adjusted_saved_cost_per_tensor_op={{[1-9][0-9]*}}
// POLICY-TRACE-DAG: FLAGTREE_RLC_TRACE phase=2 outcome=reject reason=profitability-removed-convert-density-below-threshold{{.*}}online_removed_convert_density_per_1024_proposal_values={{[1-9][0-9]*}}
// POLICY-TRACE-DAG: FLAGTREE_RLC_TRACE phase=policy outcome=reject reason=profitability-requires-single-product-launch{{.*}}online_launch_count=unknown

#parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#atomic = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "musa:31", "ttg.threads-per-warp" = 32 : i32} {
  // ALLOW-LABEL: tt.func @atomic_tail_allowed
  // ALLOW-NOT: ttg.convert_layout
  // ALLOW: tt.atomic_rmw
  tt.func @atomic_tail_allowed(%acc: tensor<16x1024xi64, #parent>, %base: !tt.ptr<i32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %offsets = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #atomic>
    %base_splat = tt.splat %base : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>, #atomic>
    %ptrs = tt.addptr %base_splat, %offsets : tensor<16x!tt.ptr<i32>, #atomic>, tensor<16xi32, #atomic>
    %mask = arith.constant dense<true> : tensor<16xi1, #atomic>
    scf.for %i = %c0 to %c1 step %c1 {
      %local_sum = "tt.reduce"(%acc) <{axis = 1 : i32}> ({
      ^bb0(%lhs: i64, %rhs: i64):
        %sum = arith.addi %lhs, %rhs : i64
        tt.reduce.return %sum : i64
      }) : (tensor<16x1024xi64, #parent>) -> tensor<16xi64, #ttg.slice<{dim = 1, parent = #parent}>>
      %converted = ttg.convert_layout %local_sum : tensor<16xi64, #ttg.slice<{dim = 1, parent = #parent}>> -> tensor<16xi64, #atomic>
      %value = arith.trunci %converted : tensor<16xi64, #atomic> to tensor<16xi32, #atomic>
      %unused = tt.atomic_rmw add, relaxed, gpu, %ptrs, %value, %mask : (tensor<16x!tt.ptr<i32>, #atomic>, tensor<16xi32, #atomic>, tensor<16xi1, #atomic>) -> tensor<16xi32, #atomic>
    }
    tt.return
  }
}

// -----

#parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#atomic = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.rlc-atomic-writeback-max-elements-per-thread-ratio" = 1 : i32, ttg.target = "musa:31", "ttg.threads-per-warp" = 32 : i32} {
  // PRESERVE-LABEL: tt.func @atomic_tail_policy_preserves
  // PRESERVE: ttg.convert_layout
  // PRESERVE: tt.atomic_rmw
  tt.func @atomic_tail_policy_preserves(%acc: tensor<16x1024xi64, #parent>, %base: !tt.ptr<i32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %offsets = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #atomic>
    %base_splat = tt.splat %base : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>, #atomic>
    %ptrs = tt.addptr %base_splat, %offsets : tensor<16x!tt.ptr<i32>, #atomic>, tensor<16xi32, #atomic>
    %mask = arith.constant dense<true> : tensor<16xi1, #atomic>
    scf.for %i = %c0 to %c1 step %c1 {
      %local_sum = "tt.reduce"(%acc) <{axis = 1 : i32}> ({
      ^bb0(%lhs: i64, %rhs: i64):
        %sum = arith.addi %lhs, %rhs : i64
        tt.reduce.return %sum : i64
      }) : (tensor<16x1024xi64, #parent>) -> tensor<16xi64, #ttg.slice<{dim = 1, parent = #parent}>>
      %converted = ttg.convert_layout %local_sum : tensor<16xi64, #ttg.slice<{dim = 1, parent = #parent}>> -> tensor<16xi64, #atomic>
      %value = arith.trunci %converted : tensor<16xi64, #atomic> to tensor<16xi32, #atomic>
      %unused = tt.atomic_rmw add, relaxed, gpu, %ptrs, %value, %mask : (tensor<16x!tt.ptr<i32>, #atomic>, tensor<16xi32, #atomic>, tensor<16xi1, #atomic>) -> tensor<16xi32, #atomic>
    }
    tt.return
  }
}

// -----

#parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#atomic = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.rlc-product-launch-count" = 1 : i32, "ttg.rlc-profitability-max-external-use-edges" = 0 : i64, "ttg.rlc-profitability-min-adjusted-saved-cost-per-tensor-op" = 1 : i64, "ttg.rlc-profitability-min-removed-convert-density-per-1024-proposal-values" = 1025 : i64, "ttg.rlc-profitability-phase3-saved-cost-multiplier" = 2 : i64, "ttg.rlc-profitability-policy-enabled" = 1 : i32, ttg.target = "musa:31", "ttg.threads-per-warp" = 32 : i32} {
  // DENSITY-REJECT-LABEL: tt.func @atomic_tail_density_rejects
  // DENSITY-REJECT: ttg.convert_layout
  // DENSITY-REJECT: tt.atomic_rmw
  tt.func @atomic_tail_density_rejects(%acc: tensor<16x1024xi64, #parent>, %base: !tt.ptr<i32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %offsets = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #atomic>
    %base_splat = tt.splat %base : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>, #atomic>
    %ptrs = tt.addptr %base_splat, %offsets : tensor<16x!tt.ptr<i32>, #atomic>, tensor<16xi32, #atomic>
    %mask = arith.constant dense<true> : tensor<16xi1, #atomic>
    scf.for %i = %c0 to %c1 step %c1 {
      %local_sum = "tt.reduce"(%acc) <{axis = 1 : i32}> ({
      ^bb0(%lhs: i64, %rhs: i64):
        %sum = arith.addi %lhs, %rhs : i64
        tt.reduce.return %sum : i64
      }) : (tensor<16x1024xi64, #parent>) -> tensor<16xi64, #ttg.slice<{dim = 1, parent = #parent}>>
      %converted = ttg.convert_layout %local_sum : tensor<16xi64, #ttg.slice<{dim = 1, parent = #parent}>> -> tensor<16xi64, #atomic>
      %value = arith.trunci %converted : tensor<16xi64, #atomic> to tensor<16xi32, #atomic>
      %unused = tt.atomic_rmw add, relaxed, gpu, %ptrs, %value, %mask : (tensor<16x!tt.ptr<i32>, #atomic>, tensor<16xi32, #atomic>, tensor<16xi1, #atomic>) -> tensor<16xi32, #atomic>
    }
    tt.return
  }
}

// -----

#parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#atomic = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.rlc-product-launch-count" = 1 : i32, "ttg.rlc-profitability-max-external-use-edges" = 0 : i64, "ttg.rlc-profitability-min-adjusted-saved-cost-per-tensor-op" = 9223372036854775807 : i64, "ttg.rlc-profitability-phase3-saved-cost-multiplier" = 2 : i64, "ttg.rlc-profitability-policy-enabled" = 1 : i32, ttg.target = "musa:31", "ttg.threads-per-warp" = 32 : i32} {
  // PROFIT-REJECT-LABEL: tt.func @atomic_tail_profitability_rejects
  // PROFIT-REJECT: ttg.convert_layout
  // PROFIT-REJECT: tt.atomic_rmw
  tt.func @atomic_tail_profitability_rejects(%acc: tensor<16x1024xi64, #parent>, %base: !tt.ptr<i32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %offsets = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #atomic>
    %base_splat = tt.splat %base : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>, #atomic>
    %ptrs = tt.addptr %base_splat, %offsets : tensor<16x!tt.ptr<i32>, #atomic>, tensor<16xi32, #atomic>
    %mask = arith.constant dense<true> : tensor<16xi1, #atomic>
    scf.for %i = %c0 to %c1 step %c1 {
      %local_sum = "tt.reduce"(%acc) <{axis = 1 : i32}> ({
      ^bb0(%lhs: i64, %rhs: i64):
        %sum = arith.addi %lhs, %rhs : i64
        tt.reduce.return %sum : i64
      }) : (tensor<16x1024xi64, #parent>) -> tensor<16xi64, #ttg.slice<{dim = 1, parent = #parent}>>
      %converted = ttg.convert_layout %local_sum : tensor<16xi64, #ttg.slice<{dim = 1, parent = #parent}>> -> tensor<16xi64, #atomic>
      %value = arith.trunci %converted : tensor<16xi64, #atomic> to tensor<16xi32, #atomic>
      %unused = tt.atomic_rmw add, relaxed, gpu, %ptrs, %value, %mask : (tensor<16x!tt.ptr<i32>, #atomic>, tensor<16xi32, #atomic>, tensor<16xi1, #atomic>) -> tensor<16xi32, #atomic>
    }
    tt.return
  }
}

// -----

#parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#atomic = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.rlc-profitability-max-external-use-edges" = 0 : i64, "ttg.rlc-profitability-min-adjusted-saved-cost-per-tensor-op" = 1 : i64, "ttg.rlc-profitability-phase3-saved-cost-multiplier" = 2 : i64, "ttg.rlc-profitability-policy-enabled" = 1 : i32, ttg.target = "musa:31", "ttg.threads-per-warp" = 32 : i32} {
  // CONTEXT-REJECT-LABEL: tt.func @atomic_tail_missing_launch_count
  // CONTEXT-REJECT: ttg.convert_layout
  // CONTEXT-REJECT: tt.atomic_rmw
  tt.func @atomic_tail_missing_launch_count(%acc: tensor<16x1024xi64, #parent>, %base: !tt.ptr<i32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %offsets = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #atomic>
    %base_splat = tt.splat %base : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>, #atomic>
    %ptrs = tt.addptr %base_splat, %offsets : tensor<16x!tt.ptr<i32>, #atomic>, tensor<16xi32, #atomic>
    %mask = arith.constant dense<true> : tensor<16xi1, #atomic>
    scf.for %i = %c0 to %c1 step %c1 {
      %local_sum = "tt.reduce"(%acc) <{axis = 1 : i32}> ({
      ^bb0(%lhs: i64, %rhs: i64):
        %sum = arith.addi %lhs, %rhs : i64
        tt.reduce.return %sum : i64
      }) : (tensor<16x1024xi64, #parent>) -> tensor<16xi64, #ttg.slice<{dim = 1, parent = #parent}>>
      %converted = ttg.convert_layout %local_sum : tensor<16xi64, #ttg.slice<{dim = 1, parent = #parent}>> -> tensor<16xi64, #atomic>
      %value = arith.trunci %converted : tensor<16xi64, #atomic> to tensor<16xi32, #atomic>
      %unused = tt.atomic_rmw add, relaxed, gpu, %ptrs, %value, %mask : (tensor<16x!tt.ptr<i32>, #atomic>, tensor<16xi32, #atomic>, tensor<16xi1, #atomic>) -> tensor<16xi32, #atomic>
    }
    tt.return
  }
}
