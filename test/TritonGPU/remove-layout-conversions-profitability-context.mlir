// RUN: triton-opt %s -tritongpu-remove-layout-conversions="enable-rlc-enhance=true rlc-phase-mask=5" | FileCheck %s
// RUN: env FLAGTREE_RLC_TRACE_REJECTS=1 triton-opt %s -tritongpu-remove-layout-conversions="enable-rlc-enhance=true rlc-phase-mask=5" 2>&1 | FileCheck %s --check-prefix=TRACE

#src = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#dst = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#reduce = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.rlc-int-to-fp-vector-width-mask" = 20 : i32, "ttg.rlc-preserve-int-to-fp-contiguity" = 1 : i32, "ttg.rlc-product-launch-count" = 1 : i32, "ttg.rlc-profitability-max-external-use-edges" = 8 : i64, "ttg.rlc-profitability-min-adjusted-saved-cost-per-tensor-op" = 1 : i64, "ttg.rlc-profitability-phase3-saved-cost-multiplier" = 2 : i64, "ttg.rlc-profitability-policy-enabled" = 1 : i32, ttg.target = "musa:31", "ttg.threads-per-warp" = 32 : i32} {
  // The writeback tail is closed and locally profitable. The reduction is in
  // an independent use-def component, so it remains an offline calibration
  // feature and must not veto this proposal.
  // CHECK-LABEL: tt.func @unrelated_reduction_does_not_veto_closed_tail
  // CHECK-NOT: ttg.convert_layout
  // CHECK: tt.reduce
  // CHECK-NOT: ttg.convert_layout
  // CHECK: tt.return
  tt.func @unrelated_reduction_does_not_veto_closed_tail(%input: !tt.ptr<i32>, %output: tensor<256x!tt.ptr<f32>, #dst>, %reduce_input: tensor<16x32xf32, #reduce>, %reduce_output: tensor<16x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #reduce}>>, %stride: i32) {
    %reduced = "tt.reduce"(%reduce_input) <{axis = 1 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      tt.reduce.return %sum : f32
    }) : (tensor<16x32xf32, #reduce>) -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #reduce}>>
    tt.store %reduce_output, %reduced : tensor<16x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #reduce}>>
    %index = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #src>
    %stride_splat = tt.splat %stride : i32 -> tensor<256xi32, #src>
    %offset = arith.muli %index, %stride_splat : tensor<256xi32, #src>
    %input_splat = tt.splat %input : !tt.ptr<i32> -> tensor<256x!tt.ptr<i32>, #src>
    %input_ptrs = tt.addptr %input_splat, %offset : tensor<256x!tt.ptr<i32>, #src>, tensor<256xi32, #src>
    %loaded = tt.load %input_ptrs : tensor<256x!tt.ptr<i32>, #src>
    %value = arith.sitofp %loaded : tensor<256xi32, #src> to tensor<256xf32, #src>
    %converted = ttg.convert_layout %value : tensor<256xf32, #src> -> tensor<256xf32, #dst>
    tt.store %output, %converted : tensor<256x!tt.ptr<f32>, #dst>
    tt.return
  }
}

// TRACE: FLAGTREE_RLC_TRACE phase=2 outcome=accept reason=committed
// TRACE-SAME: online_reduce_scan_ops=1 online_proposal_touches_reduction_or_scan=0
