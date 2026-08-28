// RUN: env FLAGTREE_RLC_TRACE_REJECTS=1 triton-opt %s -tritongpu-remove-layout-conversions="enable-rlc-enhance=true rlc-phase-mask=7" 2>&1 | FileCheck %s --check-prefix=TRACE
// RUN: env -u FLAGTREE_RLC_TRACE_REJECTS triton-opt %s -tritongpu-remove-layout-conversions="enable-rlc-enhance=true rlc-phase-mask=7" 2>&1 | FileCheck %s --check-prefix=QUIET

#src = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#dst = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // The converted source has a second tensor user. Phase 1b must reject the
  // seed instead of propagating a layout preference through both branches.
  // The direct store-tail proposal removes only one one-shot convert, so Phase
  // 2 must reject it as a weak proposal.
  tt.func @multi_use_one_shot_writeback(%base: !tt.ptr<f32>, %seed: i32) {
    %root = tt.splat %seed : i32 -> tensor<256xi32, #src>
    %root_f = arith.sitofp %root : tensor<256xi32, #src> to tensor<256xf32, #src>
    %value = math.sin %root_f : tensor<256xf32, #src>
    %one = arith.constant dense<1.0> : tensor<256xf32, #src>
    %side = arith.addf %value, %one : tensor<256xf32, #src>
    %converted = ttg.convert_layout %value : tensor<256xf32, #src> -> tensor<256xf32, #dst>
    %offsets = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #dst>
    %base_splat = tt.splat %base : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #dst>
    %ptrs = tt.addptr %base_splat, %offsets : tensor<256x!tt.ptr<f32>, #dst>, tensor<256xi32, #dst>
    tt.store %ptrs, %converted : tensor<256x!tt.ptr<f32>, #dst>
    tt.return
  }
}

// TRACE: FLAGTREE_RLC_TRACE phase=1b outcome=reject reason=source-or-result-multi-use op=ttg.convert_layout
// TRACE: FLAGTREE_RLC_TRACE phase=2 outcome=no-proposal reason=reduce-tail
// TRACE: FLAGTREE_RLC_TRACE phase=2 outcome=no-proposal reason=loop-tail
// TRACE: FLAGTREE_RLC_TRACE phase=2 outcome=reject reason=weak-one-shot-proposal
// QUIET-NOT: FLAGTREE_RLC_TRACE
