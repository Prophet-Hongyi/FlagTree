// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions="enable-rlc-enhance=true rlc-phase-mask=5" | FileCheck %s --check-prefixes=UNGUARDED,GUARDED
// RUN: env FLAGTREE_RLC_TRACE_REJECTS=1 triton-opt %s -split-input-file -tritongpu-remove-layout-conversions="enable-rlc-enhance=true rlc-phase-mask=5" 2>&1 | FileCheck %s --check-prefix=TRACE

#src = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#dst = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "musa:31", "ttg.threads-per-warp" = 32 : i32} {
  // The dynamic stride makes the load non-coalesced, so Phase 2 may retag the
  // closed load/sitofp component to the wider store layout.
  // UNGUARDED-LABEL: tt.func @int_to_fp_retag_allowed
  // UNGUARDED-NOT: ttg.convert_layout
  // UNGUARDED: tt.store
  tt.func @int_to_fp_retag_allowed(%input: !tt.ptr<i32>, %output: tensor<256x!tt.ptr<f32>, #dst>, %stride: i32) {
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

// -----

#src = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#dst = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.rlc-preserve-int-to-fp-contiguity" = 1 : i32, ttg.target = "musa:31", "ttg.threads-per-warp" = 32 : i32} {
  // GUARDED-LABEL: tt.func @int_to_fp_retag_preserved
  // GUARDED: arith.sitofp
  // GUARDED: ttg.convert_layout
  // GUARDED: tt.store
  tt.func @int_to_fp_retag_preserved(%input: !tt.ptr<i32>, %output: tensor<256x!tt.ptr<f32>, #dst>, %stride: i32) {
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

// TRACE: FLAGTREE_RLC_TRACE phase=2 outcome=reject reason=int-to-fp-contiguity-change
