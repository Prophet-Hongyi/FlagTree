// RUN: triton-opt %s -tritongpu-remove-layout-conversions="enable-rlc-enhance=true rlc-phase-mask=3" | FileCheck %s
// RUN: triton-opt %s -tritongpu-remove-layout-conversions="enable-rlc-enhance=true rlc-phase-mask=5" | FileCheck %s
// RUN: triton-opt %s -tritongpu-remove-layout-conversions="enable-rlc-enhance=true rlc-phase-mask=15" | FileCheck %s
// Reduced from the first RLC input of FlagGems geometric on MUSA. Value,
// address and mask writebacks share the Philox producer cone: guarding only
// the log result lets another seed retag the same computation on a later pass.
// CHECK-DAG: #[[SCALAR:.*]] = #ttg.blocked<{sizePerThread = [1],
// CHECK-LABEL: tt.func public @geometric_kernel
// CHECK-COUNT-4: math.log {{.*}} : tensor<1024xf32, #[[SCALAR]]>
// CHECK: tt.return
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [16], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [16], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, "ttg.rlc-atomic-writeback-max-elements-per-thread-ratio" = 1 : i32, "ttg.rlc-int-to-fp-vector-width-mask" = 20 : i32, "ttg.rlc-preserve-int-to-fp-contiguity" = 1 : i32, ttg.target = "musa:31", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @geometric_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32}, %arg2: f32, %arg3: i32, %arg4: i32) attributes {noinline = false} {
    %c-766435501_i32 = arith.constant -766435501 : i32
    %c0_i32 = arith.constant 0 : i32
    %c-845247145_i32 = arith.constant -845247145 : i32
    %cst = arith.constant dense<4.6566126E-10> : tensor<1024xf32, #blocked>
    %cst_0 = arith.constant dense<-766435501> : tensor<1024xi32, #blocked>
    %cst_1 = arith.constant dense<-845247145> : tensor<1024xi32, #blocked>
    %cst_2 = arith.constant 1.000000e+00 : f32
    %cst_3 = arith.constant dense<0> : tensor<1024xi32, #blocked>
    %c32_i64 = arith.constant 32 : i64
    %c4294967295_i64 = arith.constant 4294967295 : i64
    %0 = arith.extsi %arg3 : i32 to i64
    %1 = arith.extsi %arg4 : i32 to i64
    %2 = arith.andi %1, %c4294967295_i64 : i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = tt.get_program_id x : i32
    %5 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %6 = tt.splat %4 : i32 -> tensor<1024xi32, #blocked>
    %7 = arith.addi %6, %5 : tensor<1024xi32, #blocked>
    %8 = tt.splat %3 : i32 -> tensor<1024xi32, #blocked>
    %9 = arith.addi %8, %7 : tensor<1024xi32, #blocked>
    %10 = arith.shrui %0, %c32_i64 : i64
    %11 = arith.andi %10, %c4294967295_i64 : i64
    %12 = arith.trunci %11 : i64 to i32
    %13 = arith.andi %0, %c4294967295_i64 : i64
    %14 = arith.trunci %13 : i64 to i32
    %15 = tt.mulhiui %c-845247145_i32, %c0_i32 : i32
    %16 = tt.splat %12 : i32 -> tensor<1024xi32, #blocked>
    %17 = arith.xori %16, %cst_0 : tensor<1024xi32, #blocked>
    %18 = tt.splat %14 : i32 -> tensor<1024xi32, #blocked>
    %19 = arith.xori %18, %cst_1 : tensor<1024xi32, #blocked>
    %20 = tt.mulhiui %c-766435501_i32, %15 : i32
    %21 = tt.splat %20 : i32 -> tensor<1024xi32, #blocked>
    %22 = arith.xori %21, %9 : tensor<1024xi32, #blocked>
    %23 = tt.splat %15 : i32 -> tensor<1024xi32, #blocked>
    %24 = tt.mulhiui %cst_1, %22 : tensor<1024xi32, #blocked>
    %25 = arith.xori %24, %17 : tensor<1024xi32, #blocked>
    %26 = arith.xori %23, %cst_0 : tensor<1024xi32, #blocked>
    %27 = tt.splat %14 : i32 -> tensor<1024xi32, #blocked>
    %28 = arith.xori %27, %cst_1 : tensor<1024xi32, #blocked>
    %29 = arith.xori %19, %cst_0 : tensor<1024xi32, #blocked>
    %30 = arith.muli %26, %cst_1 : tensor<1024xi32, #blocked>
    %31 = arith.muli %25, %cst_0 : tensor<1024xi32, #blocked>
    %32 = arith.xori %30, %cst_1 : tensor<1024xi32, #blocked>
    %33 = tt.splat %14 : i32 -> tensor<1024xi32, #blocked>
    %34 = arith.xori %32, %33 : tensor<1024xi32, #blocked>
    %35 = arith.xori %31, %cst_0 : tensor<1024xi32, #blocked>
    %36 = tt.splat %12 : i32 -> tensor<1024xi32, #blocked>
    %37 = arith.xori %35, %36 : tensor<1024xi32, #blocked>
    %38 = arith.muli %29, %cst_1 : tensor<1024xi32, #blocked>
    %39 = arith.muli %28, %cst_0 : tensor<1024xi32, #blocked>
    %40 = arith.xori %38, %cst_1 : tensor<1024xi32, #blocked>
    %41 = tt.splat %14 : i32 -> tensor<1024xi32, #blocked>
    %42 = arith.xori %40, %41 : tensor<1024xi32, #blocked>
    %43 = arith.xori %39, %cst_0 : tensor<1024xi32, #blocked>
    %44 = tt.splat %12 : i32 -> tensor<1024xi32, #blocked>
    %45 = arith.xori %43, %44 : tensor<1024xi32, #blocked>
    %46 = arith.muli %37, %cst_1 : tensor<1024xi32, #blocked>
    %47 = arith.muli %34, %cst_0 : tensor<1024xi32, #blocked>
    %48 = tt.mulhiui %cst_1, %45 : tensor<1024xi32, #blocked>
    %49 = arith.xori %48, %46 : tensor<1024xi32, #blocked>
    %50 = tt.splat %14 : i32 -> tensor<1024xi32, #blocked>
    %51 = arith.xori %49, %50 : tensor<1024xi32, #blocked>
    %52 = tt.mulhiui %cst_0, %42 : tensor<1024xi32, #blocked>
    %53 = arith.xori %52, %47 : tensor<1024xi32, #blocked>
    %54 = tt.splat %12 : i32 -> tensor<1024xi32, #blocked>
    %55 = arith.xori %53, %54 : tensor<1024xi32, #blocked>
    %56 = arith.muli %45, %cst_1 : tensor<1024xi32, #blocked>
    %57 = arith.muli %42, %cst_0 : tensor<1024xi32, #blocked>
    %58 = tt.mulhiui %cst_1, %55 : tensor<1024xi32, #blocked>
    %59 = arith.xori %58, %56 : tensor<1024xi32, #blocked>
    %60 = tt.splat %14 : i32 -> tensor<1024xi32, #blocked>
    %61 = arith.xori %59, %60 : tensor<1024xi32, #blocked>
    %62 = tt.mulhiui %cst_0, %51 : tensor<1024xi32, #blocked>
    %63 = arith.xori %62, %57 : tensor<1024xi32, #blocked>
    %64 = tt.splat %12 : i32 -> tensor<1024xi32, #blocked>
    %65 = arith.xori %63, %64 : tensor<1024xi32, #blocked>
    %66 = arith.muli %55, %cst_1 : tensor<1024xi32, #blocked>
    %67 = arith.muli %51, %cst_0 : tensor<1024xi32, #blocked>
    %68 = tt.mulhiui %cst_1, %65 : tensor<1024xi32, #blocked>
    %69 = arith.xori %68, %66 : tensor<1024xi32, #blocked>
    %70 = tt.splat %14 : i32 -> tensor<1024xi32, #blocked>
    %71 = arith.xori %69, %70 : tensor<1024xi32, #blocked>
    %72 = tt.mulhiui %cst_0, %61 : tensor<1024xi32, #blocked>
    %73 = arith.xori %72, %67 : tensor<1024xi32, #blocked>
    %74 = tt.splat %12 : i32 -> tensor<1024xi32, #blocked>
    %75 = arith.xori %73, %74 : tensor<1024xi32, #blocked>
    %76 = tt.mulhiui %cst_1, %75 : tensor<1024xi32, #blocked>
    %77 = tt.mulhiui %cst_0, %71 : tensor<1024xi32, #blocked>
    %78 = tt.bitcast %76 : tensor<1024xi32, #blocked> -> tensor<1024xi32, #blocked>
    %79 = arith.cmpi slt, %78, %cst_3 : tensor<1024xi32, #blocked>
    %80 = arith.select %79, %cst_3, %78 : tensor<1024xi1, #blocked>, tensor<1024xi32, #blocked>
    %81 = arith.sitofp %80 : tensor<1024xi32, #blocked> to tensor<1024xf32, #blocked>
    %82 = arith.mulf %81, %cst : tensor<1024xf32, #blocked>
    %83 = tt.bitcast %75 : tensor<1024xi32, #blocked> -> tensor<1024xi32, #blocked>
    %84 = arith.cmpi slt, %83, %cst_3 : tensor<1024xi32, #blocked>
    %85 = arith.select %84, %cst_3, %83 : tensor<1024xi1, #blocked>, tensor<1024xi32, #blocked>
    %86 = arith.sitofp %85 : tensor<1024xi32, #blocked> to tensor<1024xf32, #blocked>
    %87 = arith.mulf %86, %cst : tensor<1024xf32, #blocked>
    %88 = tt.bitcast %77 : tensor<1024xi32, #blocked> -> tensor<1024xi32, #blocked>
    %89 = arith.cmpi slt, %88, %cst_3 : tensor<1024xi32, #blocked>
    %90 = arith.select %89, %cst_3, %88 : tensor<1024xi1, #blocked>, tensor<1024xi32, #blocked>
    %91 = arith.sitofp %90 : tensor<1024xi32, #blocked> to tensor<1024xf32, #blocked>
    %92 = arith.mulf %91, %cst : tensor<1024xf32, #blocked>
    %93 = tt.bitcast %71 : tensor<1024xi32, #blocked> -> tensor<1024xi32, #blocked>
    %94 = arith.cmpi slt, %93, %cst_3 : tensor<1024xi32, #blocked>
    %95 = arith.select %94, %cst_3, %93 : tensor<1024xi1, #blocked>, tensor<1024xi32, #blocked>
    %96 = arith.sitofp %95 : tensor<1024xi32, #blocked> to tensor<1024xf32, #blocked>
    %97 = arith.mulf %96, %cst : tensor<1024xf32, #blocked>
    %98 = arith.subf %cst_2, %arg2 : f32
    %99 = math.log %98 : f32
    %100 = math.log %82 : tensor<1024xf32, #blocked>
    %101 = tt.splat %99 : f32 -> tensor<1024xf32, #blocked>
    %102 = arith.divf %100, %101 : tensor<1024xf32, #blocked>
    %103 = math.ceil %102 : tensor<1024xf32, #blocked>
    %104 = math.log %87 : tensor<1024xf32, #blocked>
    %105 = arith.divf %104, %101 : tensor<1024xf32, #blocked>
    %106 = math.ceil %105 : tensor<1024xf32, #blocked>
    %107 = math.log %92 : tensor<1024xf32, #blocked>
    %108 = arith.divf %107, %101 : tensor<1024xf32, #blocked>
    %109 = math.ceil %108 : tensor<1024xf32, #blocked>
    %110 = math.log %97 : tensor<1024xf32, #blocked>
    %111 = arith.divf %110, %101 : tensor<1024xf32, #blocked>
    %112 = math.ceil %111 : tensor<1024xf32, #blocked>
    %113 = tt.splat %4 : i32 -> tensor<1024xi32, #blocked>
    %114 = arith.addi %113, %5 : tensor<1024xi32, #blocked>
    %115 = tt.splat %arg1 : i32 -> tensor<1024xi32, #blocked>
    %116 = arith.cmpi slt, %114, %115 : tensor<1024xi32, #blocked>
    %117 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>, #blocked>
    %118 = tt.addptr %117, %114 : tensor<1024x!tt.ptr<f16>, #blocked>, tensor<1024xi32, #blocked>
    %119 = arith.truncf %103 : tensor<1024xf32, #blocked> to tensor<1024xf16, #blocked>
    %120 = ttg.convert_layout %118 : tensor<1024x!tt.ptr<f16>, #blocked> -> tensor<1024x!tt.ptr<f16>, #blocked1>
    %121 = ttg.convert_layout %119 : tensor<1024xf16, #blocked> -> tensor<1024xf16, #blocked1>
    %122 = ttg.convert_layout %116 : tensor<1024xi1, #blocked> -> tensor<1024xi1, #blocked1>
    tt.store %120, %121, %122 evictionPolicy = evict_first : tensor<1024x!tt.ptr<f16>, #blocked1>
    %123 = arith.cmpi slt, %114, %115 : tensor<1024xi32, #blocked>
    %124 = tt.addptr %117, %114 : tensor<1024x!tt.ptr<f16>, #blocked>, tensor<1024xi32, #blocked>
    %125 = arith.truncf %106 : tensor<1024xf32, #blocked> to tensor<1024xf16, #blocked>
    %126 = ttg.convert_layout %124 : tensor<1024x!tt.ptr<f16>, #blocked> -> tensor<1024x!tt.ptr<f16>, #blocked1>
    %127 = ttg.convert_layout %125 : tensor<1024xf16, #blocked> -> tensor<1024xf16, #blocked1>
    %128 = ttg.convert_layout %123 : tensor<1024xi1, #blocked> -> tensor<1024xi1, #blocked1>
    tt.store %126, %127, %128 evictionPolicy = evict_first : tensor<1024x!tt.ptr<f16>, #blocked1>
    %129 = arith.cmpi slt, %114, %115 : tensor<1024xi32, #blocked>
    %130 = tt.addptr %117, %114 : tensor<1024x!tt.ptr<f16>, #blocked>, tensor<1024xi32, #blocked>
    %131 = arith.truncf %109 : tensor<1024xf32, #blocked> to tensor<1024xf16, #blocked>
    %132 = ttg.convert_layout %130 : tensor<1024x!tt.ptr<f16>, #blocked> -> tensor<1024x!tt.ptr<f16>, #blocked1>
    %133 = ttg.convert_layout %131 : tensor<1024xf16, #blocked> -> tensor<1024xf16, #blocked1>
    %134 = ttg.convert_layout %129 : tensor<1024xi1, #blocked> -> tensor<1024xi1, #blocked1>
    tt.store %132, %133, %134 evictionPolicy = evict_first : tensor<1024x!tt.ptr<f16>, #blocked1>
    %135 = arith.cmpi slt, %114, %115 : tensor<1024xi32, #blocked>
    %136 = tt.addptr %117, %114 : tensor<1024x!tt.ptr<f16>, #blocked>, tensor<1024xi32, #blocked>
    %137 = arith.truncf %112 : tensor<1024xf32, #blocked> to tensor<1024xf16, #blocked>
    %138 = ttg.convert_layout %136 : tensor<1024x!tt.ptr<f16>, #blocked> -> tensor<1024x!tt.ptr<f16>, #blocked1>
    %139 = ttg.convert_layout %137 : tensor<1024xf16, #blocked> -> tensor<1024xf16, #blocked1>
    %140 = ttg.convert_layout %135 : tensor<1024xi1, #blocked> -> tensor<1024xi1, #blocked1>
    tt.store %138, %139, %140 evictionPolicy = evict_first : tensor<1024x!tt.ptr<f16>, #blocked1>
    tt.return
  }
}

