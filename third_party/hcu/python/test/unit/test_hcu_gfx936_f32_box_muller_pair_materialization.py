"""Focused IR tests for the HCU gfx936 f32 Box-Muller pair gate."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from triton._C.libtriton import ir, passes
from triton.backends.compiler import GPUTarget
from triton.backends.hcu.compiler_hcu import HIPBackend


_PAIR_GATE = "FLAGTREE_HCU_GFX936_F32_BOX_MULLER_PAIR_MATERIALIZE"
_ANCHOR = "tt.elementwise_inline_asm"


def _make_module(
    target: str,
    *,
    pair_count: int = 1,
    reverse_stores: bool = False,
    shared_radius: bool = True,
    shared_angle: bool = True,
    box_muller: bool = True,
) -> str:
    assert pair_count in (1, 2)
    stores = [
        f"    tt.store %out{index}_splat, %z{index}, %mask : "
        "tensor<64x!tt.ptr<f32>, #blocked>"
        for index in range(pair_count * 2)
    ]
    if reverse_stores:
        stores = list(reversed(stores))
    second_radius = "%radius" if shared_radius else "%other"
    second_angle = "%angle" if shared_angle else "%other"
    if box_muller:
        products = [
            "    %cos = math.cos %angle : tensor<64xf32, #blocked>",
            "    %z0 = arith.mulf %radius, %cos : tensor<64xf32, #blocked>",
            f"    %sin = math.sin {second_angle} : tensor<64xf32, #blocked>",
            f"    %z1 = arith.mulf {second_radius}, %sin : tensor<64xf32, #blocked>",
        ]
        if pair_count == 2:
            products.extend(
                [
                    "    %cos1 = math.cos %radius : tensor<64xf32, #blocked>",
                    "    %z2 = arith.mulf %other, %cos1 : tensor<64xf32, #blocked>",
                    "    %sin1 = math.sin %radius : tensor<64xf32, #blocked>",
                    "    %z3 = arith.mulf %other, %sin1 : tensor<64xf32, #blocked>",
                ]
            )
    else:
        products = [
            "    %z0 = arith.mulf %radius, %angle : tensor<64xf32, #blocked>",
            "    %z1 = arith.mulf %radius, %other : tensor<64xf32, #blocked>",
        ]

    body = [
        "#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], "
        "warpsPerCTA = [1], order = [0]}>",
        "module attributes {\"ttg.num-ctas\" = 1 : i32, \"ttg.num-warps\" = 1 : i32, "
        f"ttg.target = \"{target}\", \"ttg.threads-per-warp\" = 64 : i32}} {{",
        "  tt.func public @box_muller_pair_test(",
        "      " + ", ".join(f"%out{i}: !tt.ptr<f32>" for i in range(pair_count * 2)) + ",",
        "      %seed: i32) "
        "attributes {noinline = false} {",
        "    %one = arith.constant dense<1> : tensor<64xi32, #blocked>",
        "    %mask = arith.constant dense<true> : tensor<64xi1, #blocked>",
        "    %seed_splat = tt.splat %seed : i32 -> tensor<64xi32, #blocked>",
        "    %seed_plus_one = arith.addi %seed_splat, %one : tensor<64xi32, #blocked>",
        "    %seed_plus_two = arith.addi %seed_plus_one, %one : tensor<64xi32, #blocked>",
        "    %radius = arith.sitofp %seed_splat : tensor<64xi32, #blocked> "
        "to tensor<64xf32, #blocked>",
        "    %angle = arith.sitofp %seed_plus_one : tensor<64xi32, #blocked> "
        "to tensor<64xf32, #blocked>",
        "    %other = arith.sitofp %seed_plus_two : tensor<64xi32, #blocked> "
        "to tensor<64xf32, #blocked>",
        *[
            f"    %out{i}_splat = tt.splat %out{i} : !tt.ptr<f32> -> "
            "tensor<64x!tt.ptr<f32>, #blocked>"
            for i in range(pair_count * 2)
        ],
        *products,
        *stores,
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

        with tempfile.TemporaryDirectory(prefix="hcu-gfx936-f32-box-") as temp_dir:
            path = Path(temp_dir) / "pair.ttgir"
            path.write_text(module_text)
            context = ir.context()
            ir.load_dialects(context)
            HIPBackend(GPUTarget("hip", "gfx936", 64)).load_dialects(context)
            module = ir.parse_mlir_module(str(path), context)
            pm = ir.pass_manager(context)
            passes.ttgpuir.add_remove_layout_conversions(pm, False, 15)
            pm.run(module, "test_hcu_gfx936_f32_box_muller_pair_materialization")
            return str(module)
    finally:
        if previous is None:
            os.environ.pop(_PAIR_GATE, None)
        else:
            os.environ[_PAIR_GATE] = previous


def _exercise_match_boundaries() -> dict[str, int]:
    compatible = _make_module("hip:gfx936")
    default_off = _run_rlc(compatible, pair_gate=None)
    assert _ANCHOR not in default_off, default_off

    gate_on = _run_rlc(compatible, pair_gate=True)
    lines = gate_on.splitlines()
    anchors = [i for i, line in enumerate(lines) if _ANCHOR in line]
    stores = [i for i, line in enumerate(lines) if "tt.store" in line]
    assert len(anchors) == 2 and len(stores) == 2 and max(anchors) < min(stores), gate_on
    assert gate_on.count('constraints = "=v,v"') == 2, gate_on
    assert "=v,=v" not in gate_on, gate_on

    four_output = _run_rlc(_make_module("hip:gfx936", pair_count=2), pair_gate=True)
    four_lines = four_output.splitlines()
    four_anchors = [i for i, line in enumerate(four_lines) if _ANCHOR in line]
    four_stores = [i for i, line in enumerate(four_lines) if "tt.store" in line]
    assert len(four_anchors) == 4 and len(four_stores) == 4, four_output
    assert (
        four_anchors[0]
        < four_anchors[1]
        < four_stores[0]
        < four_stores[1]
        < four_anchors[2]
        < four_anchors[3]
        < four_stores[2]
        < four_stores[3]
    ), four_output
    assert four_output.count('constraints = "=v,v"') == 4, four_output
    assert "=v,=v" not in four_output, four_output

    wrong_target = _run_rlc(_make_module("hip:gfx942"), pair_gate=True)
    reversed_stores = _run_rlc(_make_module("hip:gfx936", reverse_stores=True), pair_gate=True)
    split_radius = _run_rlc(_make_module("hip:gfx936", shared_radius=False), pair_gate=True)
    split_angle = _run_rlc(_make_module("hip:gfx936", shared_angle=False), pair_gate=True)
    non_box = _run_rlc(_make_module("hip:gfx936", box_muller=False), pair_gate=True)

    return {
        "default_off_anchors": default_off.count(_ANCHOR),
        "compatible_anchors": gate_on.count(_ANCHOR),
        "compatible_four_output_anchors": four_output.count(_ANCHOR),
        "gfx942_target_anchors": wrong_target.count(_ANCHOR),
        "reversed_store_anchors": reversed_stores.count(_ANCHOR),
        "split_radius_anchors": split_radius.count(_ANCHOR),
        "split_angle_anchors": split_angle.count(_ANCHOR),
        "non_box_anchors": non_box.count(_ANCHOR),
    }


def test_hcu_gfx936_f32_box_muller_pair_materialization_boundaries() -> None:
    assert _exercise_match_boundaries() == {
        "default_off_anchors": 0,
        "compatible_anchors": 2,
        "compatible_four_output_anchors": 4,
        "gfx942_target_anchors": 0,
        "reversed_store_anchors": 0,
        "split_radius_anchors": 0,
        "split_angle_anchors": 0,
        "non_box_anchors": 0,
    }


if __name__ == "__main__":
    print(json.dumps(_exercise_match_boundaries(), sort_keys=True))
    print("HCU_GFX936_F32_BOX_MULLER_PAIR_MATERIALIZATION_TEST_OK")
