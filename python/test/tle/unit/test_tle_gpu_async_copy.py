"""Unit coverage for explicit TLE transport-level asynchronous copies."""

import re

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle


def _require_cuda():
    try:
        torch.cuda.init()
    except Exception as exc:
        pytest.skip(f"CUDA init failed: {exc}")


@pytest.fixture(scope="module", autouse=True)
def _cuda_guard():
    _require_cuda()


@triton.jit
def _two_group_async_copy_kernel(in_ptr, out_ptr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    stages = tle.gpu.alloc(
        [2, BLOCK],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    stage0 = stages.slot(0)
    stage1 = stages.slot(1)

    tle.gpu.copy(in_ptr + offsets, stage0, [BLOCK], is_async=True)
    tle.gpu.async_commit_group()
    tle.gpu.copy(in_ptr + BLOCK + offsets, stage1, [BLOCK], is_async=True)
    tle.gpu.async_commit_group()

    tle.gpu.async_wait_group(1)
    values0 = tl.load(tle.gpu.local_ptr(stage0, (offsets, )))
    tl.store(out_ptr + offsets, values0 + 1.0)
    tle.gpu.async_wait_group(0)
    values1 = tl.load(tle.gpu.local_ptr(stage1, (offsets, )))
    tl.store(out_ptr + BLOCK + offsets, values1 + 2.0)


@triton.jit
def _invalid_async_direction_kernel(out_ptr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    smem = tle.gpu.alloc(
        [BLOCK],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    tle.gpu.copy(smem, out_ptr + offsets, [BLOCK], is_async=True)


@triton.jit
def _invalid_async_wait_kernel(out_ptr, MAX_PENDING: tl.constexpr):
    tle.gpu.async_wait_group(MAX_PENDING)
    tl.store(out_ptr, 0.0)


@triton.jit
def _invalid_async_flag_kernel(in_ptr, FLAG: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    smem = tle.gpu.alloc(
        [BLOCK],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    tle.gpu.copy(in_ptr + offsets, smem, [BLOCK], is_async=FLAG)


@triton.jit
def _invalid_async_shape_kernel(in_ptr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    smem = tle.gpu.alloc(
        [BLOCK],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    tle.gpu.copy(in_ptr + offsets, smem, [BLOCK // 2], is_async=True)


@triton.jit
def _required_async_illegal_width_kernel(in_ptr):
    offsets = tl.arange(0, 1)
    smem = tle.gpu.alloc(
        [1],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    tle.gpu.copy(in_ptr + offsets, smem, [1], is_async=True)
    tle.gpu.async_commit_group()
    tle.gpu.async_wait_group(0)


@triton.jit
def _required_async_bf16_1x256_kernel(in_ptr, out_ptr):
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
    smem = stages.slot(0)
    tle.gpu.copy(in_ptr + offsets, smem, [1, 256], is_async=True)
    tle.gpu.async_commit_group()
    tle.gpu.async_wait_group(0)
    values = tl.load(tle.gpu.local_ptr(smem, (row, col), shape=(1, 256)))
    tl.store(out_ptr + offsets, values)


def test_async_copy_api_is_exported():
    assert hasattr(tle.gpu, "async_commit_group")
    assert hasattr(tle.gpu, "async_wait_group")


def test_async_copy_codegen_contains_copy_commit_wait1_and_wait0():
    block = 64
    inp = torch.empty((2 * block, ), device="cuda", dtype=torch.float32)
    out = torch.empty((2 * block, ), device="cuda", dtype=torch.float32)

    compiled = _two_group_async_copy_kernel.warmup(
        inp,
        out,
        BLOCK=block,
        grid=(1, ),
        num_warps=4,
    )
    ttgir = compiled.asm["ttgir"]
    assert ttgir.count("ttg.async_copy_global_to_local") == 2
    assert ttgir.count("tle.required_async_copy") == 2
    assert ttgir.count("ttg.async_commit_group") == 2
    assert ttgir.count("ttg.async_wait") == 2
    assert re.search(r"ttg\.async_wait[^\n]*num = 1 : i32", ttgir)
    assert re.search(r"ttg\.async_wait[^\n]*num = 0 : i32", ttgir)


def test_async_copy_rejects_non_global_to_shared_direction():
    out = torch.empty((64, ), device="cuda", dtype=torch.float32)
    with pytest.raises(triton.CompilationError, match="supports only a tl.tensor of global pointers"):
        _invalid_async_direction_kernel.warmup(out, BLOCK=64, grid=(1, ), num_warps=4)


def test_async_copy_rejects_non_bool_transport_selector():
    inp = torch.empty((64, ), device="cuda", dtype=torch.float32)
    with pytest.raises(triton.CompilationError, match="is_async must be a compile-time bool"):
        _invalid_async_flag_kernel.warmup(
            inp,
            FLAG=1,
            BLOCK=64,
            grid=(1, ),
            num_warps=4,
        )


def test_async_copy_rejects_partial_shape():
    inp = torch.empty((64, ), device="cuda", dtype=torch.float32)
    with pytest.raises(triton.CompilationError, match="requires shape to exactly match both operands"):
        _invalid_async_shape_kernel.warmup(inp, BLOCK=64, grid=(1, ), num_warps=4)


def test_required_async_copy_rejects_synchronous_downgrade():
    inp = torch.empty((1, ), device="cuda", dtype=torch.bfloat16)
    # MLIR pass diagnostics are emitted directly by the native pass manager;
    # the Python compiler surfaces the failed pipeline as RuntimeError.  The
    # dialect regression checks the precise diagnostic text.
    with pytest.raises(RuntimeError, match=r"PassManager::run failed"):
        _required_async_illegal_width_kernel.warmup(
            inp,
            grid=(1, ),
            num_warps=1,
        )


def test_required_async_copy_legalizes_contiguous_bf16_1x256():
    inp = torch.arange(256, device="cuda", dtype=torch.bfloat16).reshape(1, 256)
    out = torch.empty_like(inp)
    compiled = _required_async_bf16_1x256_kernel.warmup(
        inp,
        out,
        grid=(1, ),
        num_warps=8,
    )
    ttgir = compiled.asm["ttgir"]
    ptx = compiled.asm["ptx"]
    assert "ttg.async_copy_global_to_local" in ttgir
    assert "tle.required_async_copy" in ttgir
    assert "contiguity = 2 : i32" in ttgir
    assert "cp.async" in ptx
    assert re.search(r"cp\.async\.wait_group\s+0", ptx)
    assert re.search(r"bar\.sync\s+0", ptx)

    _required_async_bf16_1x256_kernel[(1, )](inp, out, num_warps=8)
    torch.testing.assert_close(out, inp, atol=0, rtol=0)


@pytest.mark.parametrize("max_pending", [-1, 8])
def test_async_wait_group_rejects_out_of_range_immediates(max_pending):
    out = torch.empty((1, ), device="cuda", dtype=torch.float32)
    with pytest.raises(triton.CompilationError, match=r"max_pending must be in \[0, 7\]"):
        _invalid_async_wait_kernel.warmup(
            out,
            MAX_PENDING=max_pending,
            grid=(1, ),
            num_warps=1,
        )
