import os
import sys
from typing import Optional

import torch
import torch.distributed as dist

import triton
import triton.language as tl
import triton.runtime
import triton.experimental.tle.language as tle


# Kernel 1: scatter (optimized)
@triton.jit
def scatter_kernel_opt(
    input_ptr,
    local_scatter_ptr,
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

    row_offs = tl.arange(0, BLOCK_M)
    col_offs = tl.arange(0, BLOCK_N)

    # Every peer reserves a [M_per_rank, N] slot for this rank.
    slot_offset_elems = LOCAL_RANK * M_per_rank * N

    for step in range(WORLD_SIZE):
        peer = (LOCAL_RANK + step + 1) % WORLD_SIZE

        # Resolve remote base pointer once per peer.
        if peer == LOCAL_RANK:
            remote_base = local_scatter_ptr + slot_offset_elems
        else:
            remote_base = tle.remote(
                dev_mem_ptr,
                space="device",
                dtype=input_ptr.dtype.element_ty,
                shard_id=peer,
                offset=slot_offset_elems,
            )

        for local_tile in range(pid, tiles_per_peer, num_pid):
            tile_m = local_tile // num_tiles_n
            tile_n = local_tile % num_tiles_n

            in_row = peer * M_per_rank + tile_m * BLOCK_M
            in_col = tile_n * BLOCK_N
            in_ptrs = (input_ptr + (in_row + row_offs[:, None]) * N + (in_col + col_offs[None, :]))
            # Mask out rows/cols that are beyond the actual tensor boundary.
            in_row_mask = (in_row + row_offs[:, None]) < (peer + 1) * M_per_rank
            in_col_mask = (in_col + col_offs[None, :]) < N
            data = tl.load(in_ptrs, mask=in_row_mask & in_col_mask, other=0.0)

            out_row_in_peer = tile_m * BLOCK_M
            out_col = tile_n * BLOCK_N
            out_ptrs = (remote_base + (out_row_in_peer + row_offs[:, None]) * N + (out_col + col_offs[None, :]))
            out_row_mask = (out_row_in_peer + row_offs[:, None]) < M_per_rank
            tl.store(out_ptrs, data, mask=out_row_mask & in_col_mask)


# Kernel 2: ring reduce with TMA descriptors
@triton.jit
def ring_reduce_kernel_tma(
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

    c_desc = tl.make_tensor_descriptor(
        local_scatter_ptr,
        shape=[M_per_rank * WORLD_SIZE, N],
        strides=[N, 1],
        block_shape=[BLOCK_M, BLOCK_N],
    )

    output_desc = tl.make_tensor_descriptor(
        output_ptr,
        shape=[M_per_rank, N],
        strides=[N, 1],
        block_shape=[BLOCK_M, BLOCK_N],
    )

    begin_idx = LOCAL_RANK

    for tile_id in range(pid, total_tiles, num_pid):
        tile_m = tile_id // num_tiles_n
        tile_n = tile_id % num_tiles_n

        row_in_shard = tile_m * BLOCK_M
        col = tile_n * BLOCK_N

        src_rank = (begin_idx + 1) % WORLD_SIZE
        accum = c_desc.load([
            row_in_shard + src_rank * M_per_rank,
            col,
        ])

        for i in range(1, WORLD_SIZE):
            src_rank = (i + begin_idx + 1) % WORLD_SIZE
            data = c_desc.load([
                row_in_shard + src_rank * M_per_rank,
                col,
            ])
            accum += data

        output_desc.store([row_in_shard, col], accum)


# Host-side reference using PyTorch/NCCL
def torch_reduce_scatter(input_tensor, group):
    M, N = input_tensor.shape
    world_size = dist.get_world_size(group)
    output = torch.empty((M // world_size, N), dtype=input_tensor.dtype, device=input_tensor.device)
    dist.reduce_scatter_tensor(output, input_tensor, group=group)
    return output


# TLE reduce-scatter host wrapper (optimized scatter -> barrier -> TMA reduce)
def tle_reduce_scatter(
    input_tensor,
    scatter_buf,
    dev_mem_ptr,
    output,
    M_per_rank,
    N,
    local_rank,
    world_size,
    stream,
    num_sms: int = -1,
):
    # Scatter launch config mirrors the reduce two-tier design.
    if num_sms == -1:
        grid_scatter = lambda META: (triton.cdiv(M_per_rank, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]), )
        scatter_num_warps = 4
    else:
        grid_scatter = lambda META: (min(
            triton.cdiv(M_per_rank, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
            128,
        ), )
        scatter_num_warps = 8

    with torch.cuda.stream(stream):
        scatter_kernel_opt[grid_scatter](
            input_tensor,
            scatter_buf,
            dev_mem_ptr,
            M_per_rank,
            N,
            LOCAL_RANK=local_rank,
            WORLD_SIZE=world_size,
            BLOCK_M=256,
            BLOCK_N=128,
            num_warps=scatter_num_warps,
        )

    torch.cuda.synchronize()
    dist.barrier()

    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    # Reduce launch config aligned with 05-intra-node-reduce-scatter.py
    if num_sms == -1:
        grid_reduce = lambda META: (triton.cdiv(M_per_rank, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]), )
        with torch.cuda.stream(stream):
            ring_reduce_kernel_tma[grid_reduce](
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
    else:
        grid_reduce = lambda META: (min(
            triton.cdiv(M_per_rank, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
            num_sms,
        ), )
        with torch.cuda.stream(stream):
            ring_reduce_kernel_tma[grid_reduce](
                scatter_buf,
                output,
                M_per_rank,
                N,
                LOCAL_RANK=local_rank,
                WORLD_SIZE=world_size,
                BLOCK_M=256,
                BLOCK_N=128,
                num_warps=8,
            )


# Main
def main():
    mem_pool = tle.get_mem_pool()

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)

    print(f"[Rank {rank}/{world_size}] Starting TLE reduce-scatter (4096, 2048)")

    if world_size < 2:
        print("This example needs at least 2 GPUs", file=sys.stderr)
        sys.exit(1)

    dtype = torch.bfloat16
    M, N = 4096, 2048
    M_per_rank = M // world_size

    if M_per_rank < 256:
        print(f"M // world_size = {M_per_rank} < 256, skipping", file=sys.stderr)
        sys.exit(1)

    with torch.cuda.use_mem_pool(mem_pool):
        scatter_buf = torch.empty((M * N, ), dtype=dtype, device="cuda")
    _, scatter_dev_mem_ptr = tle.create_comm_tensor(scatter_buf)

    input_tensor = torch.rand((M, N), dtype=dtype, device="cuda")
    scatter_buf = scatter_buf.view(M, N)
    output = torch.empty((M_per_rank, N), dtype=dtype, device="cuda")
    stream = torch.cuda.current_stream()
    num_sms = torch.cuda.get_device_properties(local_rank).multi_processor_count

    torch_output = torch_reduce_scatter(input_tensor, group=None)
    torch.cuda.synchronize()

    # Correctness check.
    tle_reduce_scatter(
        input_tensor,
        scatter_buf,
        scatter_dev_mem_ptr,
        output,
        M_per_rank,
        N,
        local_rank,
        world_size,
        stream,
        num_sms=num_sms,
    )
    torch.cuda.synchronize()

    atol, rtol = 6e-2, 6e-2
    if torch.allclose(torch_output, output, atol=atol, rtol=rtol):
        print(f"[Rank {rank}] shape={(M, N)} PASSED")
    else:
        print(f"[Rank {rank}] shape={(M, N)} FAILED")
        print(f"[Rank {rank}] torch_output[:2,:4] = {torch_output[:2,:4]}")
        print(f"[Rank {rank}] tle_output[:2,:4] = {output[:2,:4]}")
        sys.exit(1)

    tle.cleanup_communicator()


if __name__ == "__main__":
    main()
