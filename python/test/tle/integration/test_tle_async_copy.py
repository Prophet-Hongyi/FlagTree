"""End-to-end coverage for explicit two-stage TLE cp.async transport."""

import re

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle


@triton.jit
def _two_stage_copy_add_kernel(in_ptr, out_ptr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    smem = tle.gpu.alloc(
        [2, BLOCK],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    stage0 = smem.slot(0)
    stage1 = smem.slot(1)

    # Prologue: two separately committed groups occupy the two explicit slots.
    tle.gpu.copy(in_ptr + offsets, stage0, [BLOCK], is_async=True)
    tle.gpu.async_commit_group()
    tle.gpu.copy(in_ptr + BLOCK + offsets, stage1, [BLOCK], is_async=True)
    tle.gpu.async_commit_group()

    # At most the newest group remains while the oldest slot is consumed.
    tle.gpu.async_wait_group(1)
    values0 = tl.load(tle.gpu.local_ptr(stage0, (offsets, )))
    tl.store(out_ptr + offsets, values0 + 1.0)
    tle.gpu.async_wait_group(0)
    values1 = tl.load(tle.gpu.local_ptr(stage1, (offsets, )))
    tl.store(out_ptr + BLOCK + offsets, values1 + 2.0)


@triton.jit
def _two_stage_bf16_1x256_copy_kernel(in_ptr, out_ptr):
    row = tl.arange(0, 1)[:, None]
    col = tl.arange(0, 256)[None, :]
    offsets = row * 256 + col
    stages = tle.gpu.alloc(
        [2, 1, 256],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    stage0 = stages.slot(0)
    stage1 = stages.slot(1)

    tle.gpu.copy(in_ptr + offsets, stage0, [1, 256], is_async=True)
    tle.gpu.async_commit_group()
    tle.gpu.copy(in_ptr + 256 + offsets, stage1, [1, 256], is_async=True)
    tle.gpu.async_commit_group()

    tle.gpu.async_wait_group(1)
    values0 = tl.load(tle.gpu.local_ptr(stage0, (row, col), shape=(1, 256)))
    tl.store(out_ptr + offsets, values0)
    tle.gpu.async_wait_group(0)
    values1 = tl.load(tle.gpu.local_ptr(stage1, (row, col), shape=(1, 256)))
    tl.store(out_ptr + 256 + offsets, values1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
def test_two_stage_async_copy_executes_and_lowers_to_cp_async():
    block = 64
    inp = torch.arange(2 * block, device="cuda", dtype=torch.float32)
    out = torch.empty((2 * block, ), device="cuda", dtype=torch.float32)

    compiled = _two_stage_copy_add_kernel.warmup(
        inp,
        out,
        BLOCK=block,
        grid=(1, ),
        num_warps=4,
    )
    ttgir = compiled.asm["ttgir"]
    ptx = compiled.asm["ptx"]
    assert ttgir.count("ttg.async_copy_global_to_local") == 2
    assert "cp.async" in ptx
    assert "cp.async.commit_group" in ptx
    assert re.search(r"cp\.async\.wait_group\s+1", ptx)
    assert re.search(r"cp\.async\.wait_group\s+0", ptx)

    _two_stage_copy_add_kernel[(1, )](inp, out, BLOCK=block, num_warps=4)
    expected = torch.cat((inp[:block] + 1.0, inp[block:] + 2.0))
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
def test_two_stage_bf16_1x256_copy_legalizes_and_executes():
    inp = torch.arange(512, device="cuda", dtype=torch.bfloat16).reshape(2, 1, 256)
    out = torch.empty_like(inp)

    compiled = _two_stage_bf16_1x256_copy_kernel.warmup(
        inp,
        out,
        grid=(1, ),
        num_warps=8,
    )
    ttgir = compiled.asm["ttgir"]
    ptx = compiled.asm["ptx"]
    assert ttgir.count("ttg.async_copy_global_to_local") == 2
    assert ttgir.count("tle.required_async_copy") == 2
    assert ttgir.count("contiguity = 2 : i32") == 2
    assert "cp.async" in ptx
    assert "cp.async.commit_group" in ptx
    assert re.search(r"cp\.async\.wait_group\s+1", ptx)
    assert re.search(r"cp\.async\.wait_group\s+0", ptx)
    assert len(re.findall(r"bar\.sync\s+0", ptx)) >= 2

    _two_stage_bf16_1x256_copy_kernel[(1, )](inp, out, num_warps=8)
    torch.testing.assert_close(out, inp, atol=0, rtol=0)
