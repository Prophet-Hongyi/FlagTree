// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions="enable-rlc-enhance=true rlc-phase-mask=5" | FileCheck %s --check-prefixes=OUTPUT-HEAVY,BALANCED
// RUN: env FLAGTREE_RLC_TRACE_REJECTS=1 triton-opt %s -split-input-file -tritongpu-remove-layout-conversions="enable-rlc-enhance=true rlc-phase-mask=5" 2>&1 | FileCheck %s --check-prefix=TRACE

#src = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#dst = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.rlc-product-launch-count" = 1 : i32, "ttg.rlc-profitability-low-density-output-heavy-min-compute-ops" = 1 : i64, "ttg.rlc-profitability-max-external-use-edges" = 0 : i64, "ttg.rlc-profitability-min-adjusted-saved-cost-per-tensor-op" = 1 : i64, "ttg.rlc-profitability-min-removed-convert-density-per-1024-proposal-values" = 1025 : i64, "ttg.rlc-profitability-phase3-saved-cost-multiplier" = 2 : i64, "ttg.rlc-profitability-policy-enabled" = 1 : i32, ttg.target = "musa:31", "ttg.threads-per-warp" = 32 : i32} {
  // A low-density proposal may pass when online compute exceeds a backend
  // threshold, global writes outnumber reads, and convert removal covers both
  // writeback and external-use risk. No kernel or shape identity is consulted.
  // OUTPUT-HEAVY-LABEL: tt.func @output_heavy_compute_escapes_density_gate
  // OUTPUT-HEAVY-NOT: ttg.convert_layout
  // OUTPUT-HEAVY: tt.store
  tt.func @output_heavy_compute_escapes_density_gate(%output: tensor<256x!tt.ptr<f16>, #dst>) {
    %index = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #src>
    %value = arith.sitofp %index : tensor<256xi32, #src> to tensor<256xf32, #src>
    %narrow = arith.truncf %value : tensor<256xf32, #src> to tensor<256xf16, #src>
    %converted = ttg.convert_layout %narrow : tensor<256xf16, #src> -> tensor<256xf16, #dst>
    tt.store %output, %converted : tensor<256x!tt.ptr<f16>, #dst>
    tt.return
  }
}

// -----

#src = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#dst = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.rlc-product-launch-count" = 1 : i32, "ttg.rlc-profitability-low-density-output-heavy-min-compute-ops" = 1 : i64, "ttg.rlc-profitability-max-external-use-edges" = 0 : i64, "ttg.rlc-profitability-min-adjusted-saved-cost-per-tensor-op" = 1 : i64, "ttg.rlc-profitability-min-removed-convert-density-per-1024-proposal-values" = 1025 : i64, "ttg.rlc-profitability-phase3-saved-cost-multiplier" = 2 : i64, "ttg.rlc-profitability-policy-enabled" = 1 : i32, ttg.target = "musa:31", "ttg.threads-per-warp" = 32 : i32} {
  // Equal global read/write counts are not output-heavy. The independent side
  // chain keeps two loads and one store live while the converted writeback
  // remains a separate proposal. Generic cleanup may still simplify the
  // converted chain, so the reject trace owns the policy assertion.
  // BALANCED-LABEL: tt.func @balanced_read_write_keeps_density_gate
  // BALANCED: tt.load
  // BALANCED: tt.load
  // BALANCED: arith.addf
  // BALANCED: tt.store
  // BALANCED: arith.truncf
  // BALANCED: tt.store
  tt.func @balanced_read_write_keeps_density_gate(%lhs: tensor<256x!tt.ptr<f32>, #src>, %rhs: tensor<256x!tt.ptr<f32>, #src>, %side_output: tensor<256x!tt.ptr<f32>, #src>, %output: tensor<256x!tt.ptr<f16>, #dst>) {
    %lhs_value = tt.load %lhs : tensor<256x!tt.ptr<f32>, #src>
    %rhs_value = tt.load %rhs : tensor<256x!tt.ptr<f32>, #src>
    %side_value = arith.addf %lhs_value, %rhs_value : tensor<256xf32, #src>
    tt.store %side_output, %side_value : tensor<256x!tt.ptr<f32>, #src>
    %index = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #src>
    %input = arith.sitofp %index : tensor<256xi32, #src> to tensor<256xf32, #src>
    %one = arith.constant dense<1.0> : tensor<256xf32, #src>
    %value = arith.addf %input, %one : tensor<256xf32, #src>
    %narrow = arith.truncf %value : tensor<256xf32, #src> to tensor<256xf16, #src>
    %converted = ttg.convert_layout %narrow : tensor<256xf16, #src> -> tensor<256xf16, #dst>
    tt.store %output, %converted : tensor<256x!tt.ptr<f16>, #dst>
    tt.return
  }
}

// TRACE-DAG: FLAGTREE_RLC_TRACE phase=2 outcome=accept reason=committed{{.*}}online_compute_ops={{[1-9][0-9]*}}{{.*}}online_low_density_global_writeback_math_eligible=0{{.*}}online_low_density_output_heavy_compute_eligible=1
// TRACE-DAG: FLAGTREE_RLC_TRACE phase=2 outcome=reject reason=profitability-removed-convert-density-below-threshold{{.*}}online_global_load_ops=2 online_global_store_ops=2{{.*}}online_compute_ops={{[1-9][0-9]*}}{{.*}}online_low_density_output_heavy_compute_eligible=0
