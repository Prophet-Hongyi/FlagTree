"""Focused IR tests for the HCU gfx936 fp16 pair-materialization gate."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from triton._C.libtriton import ir, passes
from triton.backends.compiler import GPUTarget
from triton.backends.hcu.compiler_hcu import HIPBackend


_PAIR_GATE = "FLAGTREE_HCU_GFX936_F16_PAIR_MATERIALIZE"
_ANCHOR = "tt.elementwise_inline_asm"


def _make_module(target: str, filler_count: int, *, reverse_stores: bool = False) -> str:
    filler_lines = []
    dummy_value = "%seed_splat"
    for index in range(filler_count):
        next_value = f"%dummy{index}"
        filler_lines.append(
            f"    {next_value} = arith.addi {dummy_value}, %one : "
            "tensor<64xi32, #blocked>"
        )
        dummy_value = next_value

    first_store = (
        "    tt.store %out0_splat, %trunc0, %mask : "
        "tensor<64x!tt.ptr<f16>, #blocked>"
    )
    second_store = (
        "    tt.store %out1_splat, %trunc1, %mask : "
        "tensor<64x!tt.ptr<f16>, #blocked>"
    )
    if reverse_stores:
        store_before_fillers = []
        stores_after_second = [second_store, first_store]
    else:
        store_before_fillers = [first_store]
        stores_after_second = [second_store]

    body = [
        "#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], "
        "warpsPerCTA = [1], order = [0]}>",
        "module attributes {\"ttg.num-ctas\" = 1 : i32, \"ttg.num-warps\" = 1 : i32, "
        f"ttg.target = \"{target}\", \"ttg.threads-per-warp\" = 64 : i32}} {{",
        "  tt.func public @pair_materialization_test(",
        "      %out0: !tt.ptr<f16>, %out1: !tt.ptr<f16>,",
        "      %dummy_out: !tt.ptr<i32>, %seed: i32) attributes {noinline = false} {",
        "    %one = arith.constant dense<1> : tensor<64xi32, #blocked>",
        "    %mask = arith.constant dense<true> : tensor<64xi1, #blocked>",
        "    %seed_splat = tt.splat %seed : i32 -> tensor<64xi32, #blocked>",
        "    %seed_plus_one = arith.addi %seed_splat, %one : tensor<64xi32, #blocked>",
        "    %producer0 = arith.sitofp %seed_splat : tensor<64xi32, #blocked> "
        "to tensor<64xf32, #blocked>",
        "    %producer1 = arith.sitofp %seed_plus_one : tensor<64xi32, #blocked> "
        "to tensor<64xf32, #blocked>",
        "    %out0_splat = tt.splat %out0 : !tt.ptr<f16> -> "
        "tensor<64x!tt.ptr<f16>, #blocked>",
        "    %out1_splat = tt.splat %out1 : !tt.ptr<f16> -> "
        "tensor<64x!tt.ptr<f16>, #blocked>",
        "    %dummy_out_splat = tt.splat %dummy_out : !tt.ptr<i32> -> "
        "tensor<64x!tt.ptr<i32>, #blocked>",
        "    %trunc0 = arith.truncf %producer0 : tensor<64xf32, #blocked> "
        "to tensor<64xf16, #blocked>",
        *store_before_fillers,
        *filler_lines,
        "    %trunc1 = arith.truncf %producer1 : tensor<64xf32, #blocked> "
        "to tensor<64xf16, #blocked>",
        *stores_after_second,
        f"    tt.store %dummy_out_splat, {dummy_value}, %mask : "
        "tensor<64x!tt.ptr<i32>, #blocked>",
        "    tt.return",
        "  }",
        "}",
    ]
    return "\n".join(body) + "\n"


def _run_rlc(module_text: str, *, pair_gate: bool | None) -> str:
    previous = os.environ.get(_PAIR_GATE)
    try:
        if pair_gate is None:
            os.environ.pop(_PAIR_GATE, None)
        else:
            os.environ[_PAIR_GATE] = "1" if pair_gate else "0"

        with tempfile.TemporaryDirectory(prefix="hcu-gfx936-pair-") as temp_dir:
            path = Path(temp_dir) / "pair.ttgir"
            path.write_text(module_text)
            context = ir.context()
            ir.load_dialects(context)
            HIPBackend(GPUTarget("hip", "gfx936", 64)).load_dialects(context)
            module = ir.parse_mlir_module(str(path), context)
            pm = ir.pass_manager(context)
            passes.ttgpuir.add_remove_layout_conversions(pm, False, 15)
            pm.run(module, "test_hcu_gfx936_f16_pair_materialization")
            return str(module)
    finally:
        if previous is None:
            os.environ.pop(_PAIR_GATE, None)
        else:
            os.environ[_PAIR_GATE] = previous


def _assert_anchor_before_f16_stores(module_text: str) -> None:
    lines = module_text.splitlines()
    anchor_lines = [index for index, line in enumerate(lines) if _ANCHOR in line]
    f16_store_lines = [
        index
        for index, line in enumerate(lines)
        if "tt.store" in line and "!tt.ptr<f16>" in line
    ]
    assert len(anchor_lines) == 1, module_text
    assert len(f16_store_lines) == 2, module_text
    assert anchor_lines[0] < min(f16_store_lines), module_text


def _exercise_match_boundaries() -> dict[str, int]:
    # First store plus 30 filler ops puts the second truncation at distance 31,
    # the last operation examined by the strict `< 32` search bound.
    within_window = _make_module("hip:gfx936", 30)
    default_off = _run_rlc(within_window, pair_gate=None)
    assert _ANCHOR not in default_off, default_off

    gate_on = _run_rlc(within_window, pair_gate=True)
    _assert_anchor_before_f16_stores(gate_on)

    # One additional filler puts the second truncation at distance 32 and must
    # leave the pair untouched.
    outside_window = _run_rlc(_make_module("hip:gfx936", 31), pair_gate=True)
    assert _ANCHOR not in outside_window, outside_window

    reversed_stores = _run_rlc(
        _make_module("hip:gfx936", 0, reverse_stores=True), pair_gate=True
    )
    assert _ANCHOR not in reversed_stores, reversed_stores

    wrong_target = _run_rlc(_make_module("hip:gfx942", 30), pair_gate=True)
    assert _ANCHOR not in wrong_target, wrong_target

    return {
        "default_off_anchors": default_off.count(_ANCHOR),
        "within_distance_31_anchors": gate_on.count(_ANCHOR),
        "outside_distance_32_anchors": outside_window.count(_ANCHOR),
        "reversed_store_anchors": reversed_stores.count(_ANCHOR),
        "gfx942_target_anchors": wrong_target.count(_ANCHOR),
    }


def test_hcu_gfx936_f16_pair_materialization_match_boundaries() -> None:
    assert _exercise_match_boundaries() == {
        "default_off_anchors": 0,
        "within_distance_31_anchors": 1,
        "outside_distance_32_anchors": 0,
        "reversed_store_anchors": 0,
        "gfx942_target_anchors": 0,
    }


if __name__ == "__main__":
    print(json.dumps(_exercise_match_boundaries(), sort_keys=True))
    print("HCU_GFX936_F16_PAIR_MATERIALIZATION_TEST_OK")
