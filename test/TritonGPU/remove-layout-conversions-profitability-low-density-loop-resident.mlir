// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions="enable-rlc-enhance=true rlc-phase-mask=5" | FileCheck %s --check-prefixes=HIGH-SAVINGS,BELOW-FLOOR
// RUN: env FLAGTREE_RLC_TRACE_REJECTS=1 triton-opt %s -split-input-file -tritongpu-remove-layout-conversions="enable-rlc-enhance=true rlc-phase-mask=5" 2>&1 | FileCheck %s --check-prefix=TRACE

#src = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#dst = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.rlc-product-launch-count" = 1 : i32, "ttg.rlc-profitability-low-density-loop-resident-min-saved-cost" = 1 : i64, "ttg.rlc-profitability-max-external-use-edges" = 0 : i64, "ttg.rlc-profitability-min-adjusted-saved-cost-per-tensor-op" = 1 : i64, "ttg.rlc-profitability-min-removed-convert-density-per-1024-proposal-values" = 1025 : i64, "ttg.rlc-profitability-phase3-saved-cost-multiplier" = 2 : i64, "ttg.rlc-profitability-policy-enabled" = 1 : i32, ttg.target = "musa:31", "ttg.threads-per-warp" = 32 : i32} {
  // A conversion executed inside a loop can amortize a large saved payload
  // despite low proposal density. The escape uses only online topology and a
  // backend-provided cost floor; no kernel or shape identity is consulted.
  // HIGH-SAVINGS-LABEL: tt.func @loop_resident_high_savings_escapes_density_gate
  // HIGH-SAVINGS-NOT: ttg.convert_layout
  // HIGH-SAVINGS: tt.store
  tt.func @loop_resident_high_savings_escapes_density_gate(%output: tensor<256x!tt.ptr<f16>, #dst>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    scf.for %iv = %c0_i32 to %c2_i32 step %c1_i32 : i32 {
      %index = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #src>
      %value = arith.sitofp %index : tensor<256xi32, #src> to tensor<256xf32, #src>
      %narrow = arith.truncf %value : tensor<256xf32, #src> to tensor<256xf16, #src>
      %converted = ttg.convert_layout %narrow : tensor<256xf16, #src> -> tensor<256xf16, #dst>
      tt.store %output, %converted : tensor<256x!tt.ptr<f16>, #dst>
    }
    tt.return
  }
}

// -----

#src = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#dst = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.rlc-product-launch-count" = 1 : i32, "ttg.rlc-profitability-low-density-loop-resident-min-saved-cost" = 9223372036854775807 : i64, "ttg.rlc-profitability-max-external-use-edges" = 0 : i64, "ttg.rlc-profitability-min-adjusted-saved-cost-per-tensor-op" = 1 : i64, "ttg.rlc-profitability-min-removed-convert-density-per-1024-proposal-values" = 1025 : i64, "ttg.rlc-profitability-phase3-saved-cost-multiplier" = 2 : i64, "ttg.rlc-profitability-policy-enabled" = 1 : i32, ttg.target = "musa:31", "ttg.threads-per-warp" = 32 : i32} {
  // The same topology remains behind the density gate when its online saved
  // cost is below the backend floor.
  // BELOW-FLOOR-LABEL: tt.func @loop_resident_below_cost_floor_keeps_density_gate
  // Generic cleanup can still simplify the converted chain after the policy
  // rejects it, so the reject trace below owns the cost-floor assertion.
  // BELOW-FLOOR: tt.store
  tt.func @loop_resident_below_cost_floor_keeps_density_gate(%output: tensor<256x!tt.ptr<f16>, #dst>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    scf.for %iv = %c0_i32 to %c2_i32 step %c1_i32 : i32 {
      %index = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #src>
      %value = arith.sitofp %index : tensor<256xi32, #src> to tensor<256xf32, #src>
      %narrow = arith.truncf %value : tensor<256xf32, #src> to tensor<256xf16, #src>
      %converted = ttg.convert_layout %narrow : tensor<256xf16, #src> -> tensor<256xf16, #dst>
      tt.store %output, %converted : tensor<256x!tt.ptr<f16>, #dst>
    }
    tt.return
  }
}

// TRACE-DAG: FLAGTREE_RLC_TRACE phase=2 outcome=accept reason=committed{{.*}}online_loop_resident=1{{.*}}online_low_density_loop_resident_high_savings_eligible=1
// TRACE-DAG: FLAGTREE_RLC_TRACE phase=2 outcome=reject reason=profitability-removed-convert-density-below-threshold{{.*}}online_loop_resident=1{{.*}}online_low_density_loop_resident_high_savings_eligible=0
