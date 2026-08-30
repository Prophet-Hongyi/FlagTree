// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions="enable-rlc-enhance=true rlc-phase-mask=5" | FileCheck %s --check-prefixes=ESCAPE,STRICT
// RUN: env FLAGTREE_RLC_TRACE_REJECTS=1 triton-opt %s -split-input-file -tritongpu-remove-layout-conversions="enable-rlc-enhance=true rlc-phase-mask=5" 2>&1 | FileCheck %s --check-prefix=TRACE

#src = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#dst = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.rlc-product-launch-count" = 1 : i32, "ttg.rlc-profitability-low-density-global-writeback-min-math-ops" = 1 : i64, "ttg.rlc-profitability-max-external-use-edges" = 0 : i64, "ttg.rlc-profitability-min-adjusted-saved-cost-per-tensor-op" = 1 : i64, "ttg.rlc-profitability-min-removed-convert-density-per-1024-proposal-values" = 1025 : i64, "ttg.rlc-profitability-phase3-saved-cost-multiplier" = 2 : i64, "ttg.rlc-profitability-policy-enabled" = 1 : i32, ttg.target = "musa:31", "ttg.threads-per-warp" = 32 : i32} {
  // A low-density proposal may pass only when it is a closed, pure
  // global-writeback math network and removes at least one convert per store.
  // ESCAPE-LABEL: tt.func @math_writeback_escapes_density_gate
  // ESCAPE-NOT: ttg.convert_layout
  // ESCAPE: tt.store
  tt.func @math_writeback_escapes_density_gate(%output: tensor<256x!tt.ptr<f32>, #dst>) {
    %index = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #src>
    %input = arith.sitofp %index : tensor<256xi32, #src> to tensor<256xf32, #src>
    %value = math.sin %input : tensor<256xf32, #src>
    %converted = ttg.convert_layout %value : tensor<256xf32, #src> -> tensor<256xf32, #dst>
    tt.store %output, %converted : tensor<256x!tt.ptr<f32>, #dst>
    tt.return
  }
}

// -----

#src = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#dst = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.rlc-product-launch-count" = 1 : i32, "ttg.rlc-profitability-low-density-global-writeback-min-math-ops" = 1 : i64, "ttg.rlc-profitability-max-external-use-edges" = 0 : i64, "ttg.rlc-profitability-min-adjusted-saved-cost-per-tensor-op" = 1 : i64, "ttg.rlc-profitability-min-removed-convert-density-per-1024-proposal-values" = 1025 : i64, "ttg.rlc-profitability-phase3-saved-cost-multiplier" = 2 : i64, "ttg.rlc-profitability-policy-enabled" = 1 : i32, ttg.target = "musa:31", "ttg.threads-per-warp" = 32 : i32} {
  // A pure writeback is not sufficient by itself. With no qualifying math
  // work, the same low-density topology remains fail-closed.
  // STRICT-LABEL: tt.func @arithmetic_writeback_keeps_density_gate
  // STRICT: ttg.convert_layout
  // STRICT: tt.store
  tt.func @arithmetic_writeback_keeps_density_gate(%output: tensor<256x!tt.ptr<f32>, #dst>) {
    %index = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #src>
    %input = arith.sitofp %index : tensor<256xi32, #src> to tensor<256xf32, #src>
    %one = arith.constant dense<1.0> : tensor<256xf32, #src>
    %value = arith.addf %input, %one : tensor<256xf32, #src>
    %converted = ttg.convert_layout %value : tensor<256xf32, #src> -> tensor<256xf32, #dst>
    tt.store %output, %converted : tensor<256x!tt.ptr<f32>, #dst>
    tt.return
  }
}

// TRACE-DAG: FLAGTREE_RLC_TRACE phase=2 outcome=accept reason=committed{{.*}}online_low_density_global_writeback_math_eligible=1
// TRACE-DAG: FLAGTREE_RLC_TRACE phase=2 outcome=reject reason=profitability-removed-convert-density-below-threshold{{.*}}online_math_ops=0{{.*}}online_low_density_global_writeback_math_eligible=0
