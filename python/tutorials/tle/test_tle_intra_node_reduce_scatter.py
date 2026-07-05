"""
Intra-node Reduce-Scatter using FlagTree TLE (Triton Language Extension)
=========================================================================

This tutorial implements a single-node reduce-scatter operator using
FlagTree TLE.

"""

import os
import sys

import torch
import torch.distributed as dist

import triton
import triton.language as tl
import triton.experimental.tle.language as tle


@triton.jit
def scatter_kernel(
    input_ptr,
    dev_mem_ptr,
    M_per_rank,
    N,
    LOCAL_RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)

    num_tiles_m = tl.cdiv(M_per_rank, BLOCK_M)
    num_tiles_n = tl.cdiv(N, BLOCK_N)
    tiles_per_peer = num_tiles_m * num_tiles_n
    total_tiles = tiles_per_peer * WORLD_SIZE

    row_offs = tl.arange(0, BLOCK_M)
    col_offs = tl.arange(0, BLOCK_N)

    for tile_id in range(pid, total_tiles, num_pid):
        peer = tile_id // tiles_per_peer
        local_tile = tile_id % tiles_per_peer
        tile_m = local_tile // num_tiles_n
        tile_n = local_tile % num_tiles_n

        in_row = peer * M_per_rank + tile_m * BLOCK_M
        in_col = tile_n * BLOCK_N
        in_ptrs = (input_ptr + (in_row + row_offs[:, None]) * N + (in_col + col_offs[None, :]))
        data = tl.load(in_ptrs)

        out_row_in_peer = LOCAL_RANK * M_per_rank + tile_m * BLOCK_M
        out_col = tile_n * BLOCK_N
        out_offset_elems = out_row_in_peer * N + out_col

        remote_base = tle.remote(
            dev_mem_ptr,
            space="device",
            dtype=input_ptr.dtype.element_ty,
            shard_id=peer,
            offset=out_offset_elems,
        )
        remote_ptrs = (remote_base + row_offs[:, None] * N + col_offs[None, :])
        # 把当前 rank 的本地数据写到远程 peer 的 scatter buffer 里。
        tl.store(remote_ptrs, data)


@triton.jit
def ring_reduce_kernel(
    local_scatter_ptr,
    output_ptr,
    M_per_rank,
    N,
    LOCAL_RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)

    num_tiles_m = tl.cdiv(M_per_rank, BLOCK_M)
    num_tiles_n = tl.cdiv(N, BLOCK_N)
    total_tiles = num_tiles_m * num_tiles_n

    row_offs = tl.arange(0, BLOCK_M)
    col_offs = tl.arange(0, BLOCK_N)

    # swizzled starting slot, same idea as the NVSHMEM tutorial
    begin_idx = LOCAL_RANK

    for tile_id in range(pid, total_tiles, num_pid):
        tile_m = tile_id // num_tiles_n
        tile_n = tile_id % num_tiles_n

        row_in_shard = tile_m * BLOCK_M
        col = tile_n * BLOCK_N

        src_rank = (begin_idx + 1) % WORLD_SIZE
        row = src_rank * M_per_rank + row_in_shard
        ptrs = (local_scatter_ptr + (row + row_offs[:, None]) * N + (col + col_offs[None, :]))
        accum = tl.load(ptrs)

        for i in range(1, WORLD_SIZE):
            src_rank = (i + begin_idx + 1) % WORLD_SIZE
            row = src_rank * M_per_rank + row_in_shard
            ptrs = (local_scatter_ptr + (row + row_offs[:, None]) * N + (col + col_offs[None, :]))
            data = tl.load(ptrs)
            accum += data

        # store to local output
        out_row = row_in_shard
        out_col = col
        out_ptrs = (output_ptr + (out_row + row_offs[:, None]) * N + (out_col + col_offs[None, :]))
        tl.store(out_ptrs, accum)


def torch_reduce_scatter(input_tensor, group):
    M, N = input_tensor.shape
    world_size = dist.get_world_size(group)
    output = torch.empty((M // world_size, N), dtype=input_tensor.dtype, device=input_tensor.device)
    dist.reduce_scatter_tensor(output, input_tensor, group=group)
    return output


def main():

    mem_pool = tle.get_mem_pool()  # calls tle.init_communicator()

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)

    print(f"[Rank {rank}/{world_size}] Starting TLE reduce-scatter")

    if world_size < 2:
        print("This example needs at least 2 GPUs", file=sys.stderr)
        sys.exit(1)

    dtype = torch.bfloat16
    M, N = 8192, 16384
    M_per_rank = M // world_size

    input_tensor = torch.rand((M, N), dtype=dtype, device="cuda")

    with torch.cuda.use_mem_pool(mem_pool):
        scatter_buf = torch.empty((M, N), dtype=dtype, device="cuda").clone()

    _, dev_mem_ptr = tle.create_comm_tensor(scatter_buf)
    output = torch.empty((M_per_rank, N), dtype=dtype, device="cuda")

    stream = torch.cuda.current_stream()

    torch_output = torch_reduce_scatter(input_tensor, group=None)

    torch.cuda.synchronize()

    grid_scatter = lambda META: (min(
        triton.cdiv(M_per_rank, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]) * world_size,
        128,
    ), )
    with torch.cuda.stream(stream):
        scatter_kernel[grid_scatter](
            input_tensor,
            dev_mem_ptr,
            M_per_rank,
            N,
            LOCAL_RANK=local_rank,
            WORLD_SIZE=world_size,
            BLOCK_M=256,
            BLOCK_N=64,
            num_warps=4,
        )

    torch.cuda.synchronize()

    dist.barrier()

    grid_reduce = lambda META: (triton.cdiv(M_per_rank, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]), )
    with torch.cuda.stream(stream):
        ring_reduce_kernel[grid_reduce](
            scatter_buf,
            output,
            M_per_rank,
            N,
            LOCAL_RANK=local_rank,
            WORLD_SIZE=world_size,
            BLOCK_M=256,
            BLOCK_N=64,
            num_warps=4,
        )

    torch.cuda.synchronize()

    # Validate: compare TLE output against PyTorch/NCCL reference

    atol, rtol = 6e-2, 6e-2
    if torch.allclose(torch_output, output, atol=atol, rtol=rtol):
        print(f"[Rank {rank}] PASSED")
    else:
        print(f"[Rank {rank}] FAILED")
        print(f"[Rank {rank}] torch_output[:2,:4] = {torch_output[:2,:4]}")
        print(f"[Rank {rank}] tle_output[:2,:4] = {output[:2,:4]}")
        sys.exit(1)

    tle.cleanup_communicator()


if __name__ == "__main__":
    main()
