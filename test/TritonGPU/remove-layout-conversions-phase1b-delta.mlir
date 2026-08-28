// RUN: triton-opt %s -tritongpu-remove-layout-conversions="enable-rlc-enhance=true rlc-phase-mask=0" | FileCheck %s --check-prefix=MASK0
// RUN: triton-opt %s -tritongpu-remove-layout-conversions="enable-rlc-enhance=true rlc-phase-mask=3" | FileCheck %s --check-prefix=DELTA
// RUN: env FLAGTREE_RLC_TRACE_REJECTS=1 triton-opt %s -tritongpu-remove-layout-conversions="enable-rlc-enhance=true rlc-phase-mask=3" 2>&1 | FileCheck %s --check-prefix=TRACE

#value = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#index = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#store = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // Two stages reduced from the real S5000 topk comparator graph. Legacy RLC
  // removes the predicate converts and leaves the single final writeback
  // convert. Phase 1b must reject both the predicate seeds (one external value
  // branch each) and the writeback seed (two external predicate branches).
  // Accepting a locally neutral predicate seed perturbs forward conflict
  // resolution and regresses the result from one to two converts.
  //
  // MASK0-LABEL: tt.func @shared_predicate_fanout
  // MASK0-COUNT-1: ttg.convert_layout
  // MASK0-NOT: ttg.convert_layout
  // MASK0: tt.return
  //
  // DELTA-LABEL: tt.func @shared_predicate_fanout
  // DELTA-COUNT-1: ttg.convert_layout
  // DELTA-NOT: ttg.convert_layout
  // DELTA: tt.return
  tt.func @shared_predicate_fanout(%value_arg: tensor<256xi32, #value>,
                                   %value_base: !tt.ptr<i32>,
                                   %index_base: !tt.ptr<i32>) {
    %value_zero = arith.constant dense<0> : tensor<256xi32, #value>
    %index_zero = arith.constant dense<0> : tensor<256xi32, #index>
    %value = arith.addi %value_arg, %value_zero : tensor<256xi32, #value>
    %index = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #index>

    %cond0 = arith.cmpi sgt, %value, %value_zero : tensor<256xi32, #value>
    %cond0_index = ttg.convert_layout %cond0 : tensor<256xi1, #value> -> tensor<256xi1, #index>
    %value0 = arith.select %cond0, %value, %value_zero : tensor<256xi1, #value>, tensor<256xi32, #value>
    %index0 = arith.select %cond0_index, %index, %index_zero : tensor<256xi1, #index>, tensor<256xi32, #index>
    %cond1 = arith.cmpi sgt, %value0, %value_zero : tensor<256xi32, #value>
    %cond1_index = ttg.convert_layout %cond1 : tensor<256xi1, #value> -> tensor<256xi1, #index>
    %value1 = arith.select %cond1, %value0, %value_zero : tensor<256xi1, #value>, tensor<256xi32, #value>
    %index1 = arith.select %cond1_index, %index0, %index_zero : tensor<256xi1, #index>, tensor<256xi32, #index>

    %value_offsets = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #value>
    %value_base_splat = tt.splat %value_base : !tt.ptr<i32> -> tensor<256x!tt.ptr<i32>, #value>
    %value_ptrs = tt.addptr %value_base_splat, %value_offsets : tensor<256x!tt.ptr<i32>, #value>, tensor<256xi32, #value>
    tt.store %value_ptrs, %value1 : tensor<256x!tt.ptr<i32>, #value>

    %converted = ttg.convert_layout %index1 : tensor<256xi32, #index> -> tensor<256xi32, #store>
    %index_offsets = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #store>
    %index_base_splat = tt.splat %index_base : !tt.ptr<i32> -> tensor<256x!tt.ptr<i32>, #store>
    %index_ptrs = tt.addptr %index_base_splat, %index_offsets : tensor<256x!tt.ptr<i32>, #store>, tensor<256xi32, #store>
    tt.store %index_ptrs, %converted : tensor<256x!tt.ptr<i32>, #store>
    tt.return
  }

  // A shared make_range is a safe legacy positive: the existing RLC can clone
  // it for the #value pointer chain while preserving the #index mask/store
  // chain, so both mask0 and Phase 1a+1b must remain convert-free. This keeps
  // the external-use guard from treating every shared value as a fanout risk;
  // only values that backward propagation would actually retag are counted.
  //
  // MASK0-LABEL: tt.func @shared_offsets_fanout
  // MASK0-NOT: ttg.convert_layout
  // MASK0: tt.return
  //
  // DELTA-LABEL: tt.func @shared_offsets_fanout
  // DELTA-NOT: ttg.convert_layout
  // DELTA: tt.return
  tt.func @shared_offsets_fanout(%value_base: !tt.ptr<i32>,
                                 %index_base: !tt.ptr<i32>,
                                 %value: tensor<256xi32, #value>,
                                 %limit: i32) {
    %offsets = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #index>
    %value_base_splat = tt.splat %value_base : !tt.ptr<i32> -> tensor<256x!tt.ptr<i32>, #index>
    %value_ptrs_index = tt.addptr %value_base_splat, %offsets : tensor<256x!tt.ptr<i32>, #index>, tensor<256xi32, #index>
    %value_ptrs = ttg.convert_layout %value_ptrs_index : tensor<256x!tt.ptr<i32>, #index> -> tensor<256x!tt.ptr<i32>, #value>
    tt.store %value_ptrs, %value : tensor<256x!tt.ptr<i32>, #value>

    %limit_splat = tt.splat %limit : i32 -> tensor<256xi32, #index>
    %mask = arith.cmpi slt, %offsets, %limit_splat : tensor<256xi32, #index>
    %index_base_splat = tt.splat %index_base : !tt.ptr<i32> -> tensor<256x!tt.ptr<i32>, #index>
    %index_ptrs = tt.addptr %index_base_splat, %offsets : tensor<256x!tt.ptr<i32>, #index>, tensor<256xi32, #index>
    tt.store %index_ptrs, %offsets, %mask : tensor<256x!tt.ptr<i32>, #index>
    tt.return
  }
}

// TRACE: FLAGTREE_RLC_TRACE phase=1b outcome=reject reason=producer-external-use-fanout op=ttg.convert_layout
