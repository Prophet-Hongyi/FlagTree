// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions="enable-rlc-enhance=true rlc-phase-mask=5" | FileCheck %s --check-prefixes=ZERO-LOAD,HAS-MATH
// RUN: env FLAGTREE_RLC_TRACE_REJECTS=1 triton-opt %s -split-input-file -tritongpu-remove-layout-conversions="enable-rlc-enhance=true rlc-phase-mask=5" 2>&1 | FileCheck %s --check-prefix=TRACE

#src = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#dst = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.rlc-product-launch-count" = 1 : i32, "ttg.rlc-profitability-low-density-zero-load-min-arithmetic-ops" = 1 : i64, "ttg.rlc-profitability-max-external-use-edges" = 0 : i64, "ttg.rlc-profitability-min-adjusted-saved-cost-per-tensor-op" = 1 : i64, "ttg.rlc-profitability-min-removed-convert-density-per-1024-proposal-values" = 1025 : i64, "ttg.rlc-profitability-phase3-saved-cost-multiplier" = 2 : i64, "ttg.rlc-profitability-policy-enabled" = 1 : i32, ttg.target = "musa:31", "ttg.threads-per-warp" = 32 : i32} {
  // The tensor payload is synthesized from launch state: no global input is
  // read, no math-dialect operation is present, and the converted value is
  // written once. The backend threshold is the only calibration input.
  // ZERO-LOAD-LABEL: tt.func @zero_load_arithmetic_escapes_density_gate
  // ZERO-LOAD-NOT: ttg.convert_layout
  // ZERO-LOAD: tt.store
  tt.func @zero_load_arithmetic_escapes_density_gate(%output: tensor<256x!tt.ptr<f16>, #dst>) {
    %index = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #src>
    %one = arith.constant dense<1> : tensor<256xi32, #src>
    %mixed = arith.xori %index, %one : tensor<256xi32, #src>
    %value = arith.sitofp %mixed : tensor<256xi32, #src> to tensor<256xf32, #src>
    %narrow = arith.truncf %value : tensor<256xf32, #src> to tensor<256xf16, #src>
    %converted = ttg.convert_layout %narrow : tensor<256xf16, #src> -> tensor<256xf16, #dst>
    tt.store %output, %converted : tensor<256x!tt.ptr<f16>, #dst>
    tt.return
  }
}

// -----

#src = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#dst = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.rlc-product-launch-count" = 1 : i32, "ttg.rlc-profitability-low-density-zero-load-min-arithmetic-ops" = 1 : i64, "ttg.rlc-profitability-max-external-use-edges" = 0 : i64, "ttg.rlc-profitability-min-adjusted-saved-cost-per-tensor-op" = 1 : i64, "ttg.rlc-profitability-min-removed-convert-density-per-1024-proposal-values" = 1025 : i64, "ttg.rlc-profitability-phase3-saved-cost-multiplier" = 2 : i64, "ttg.rlc-profitability-policy-enabled" = 1 : i32, ttg.target = "musa:31", "ttg.threads-per-warp" = 32 : i32} {
  // A nearby zero-load affine network contains math-dialect work and remains
  // behind the density gate. This is the online distinction between the
  // positive arithmetic generator and the calibrated negative neighbor.
  // HAS-MATH-LABEL: tt.func @math_network_keeps_density_gate
  // HAS-MATH: math.fma
  // HAS-MATH: tt.store
  tt.func @math_network_keeps_density_gate(%output: tensor<256x!tt.ptr<f16>, #dst>) {
    %index = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #src>
    %value = arith.sitofp %index : tensor<256xi32, #src> to tensor<256xf32, #src>
    %zero = arith.constant dense<0.0> : tensor<256xf32, #src>
    %affine = math.fma %value, %value, %zero : tensor<256xf32, #src>
    %narrow = arith.truncf %affine : tensor<256xf32, #src> to tensor<256xf16, #src>
    %converted = ttg.convert_layout %narrow : tensor<256xf16, #src> -> tensor<256xf16, #dst>
    tt.store %output, %converted : tensor<256x!tt.ptr<f16>, #dst>
    tt.return
  }
}

// TRACE-DAG: FLAGTREE_RLC_TRACE phase=2 outcome=accept reason=committed{{.*}}online_global_load_ops=0 online_global_store_ops=1{{.*}}online_arithmetic_ops={{[1-9][0-9]*}} online_math_ops=0{{.*}}online_low_density_zero_load_arithmetic_eligible=1
// TRACE-DAG: FLAGTREE_RLC_TRACE phase=2 outcome=reject reason=profitability-removed-convert-density-below-threshold{{.*}}online_global_load_ops=0 online_global_store_ops=1{{.*}}online_math_ops=1{{.*}}online_low_density_zero_load_arithmetic_eligible=0
