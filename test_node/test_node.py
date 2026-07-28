

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist
import triton
import triton.language as tl

import triton.experimental.tle.language as tle


# node 0 的 rank 0 访问 node 1的 rank 0 ...
@triton.jit
def _read_remote_world_rank_kernel(
    output_ptr,
    ctx: tl.constexpr,
    PEER_WORLD_RANK: tl.constexpr,
):
    offset = tl.program_id(0)
    remote_ptr = tle.remote(
        ctx,
        space="node",
        dtype=tl.float32,
        shard_id=PEER_WORLD_RANK,
        offset=offset,
    )
    tl.store(output_ptr + offset, tl.load(remote_ptr))


def _peer_on_next_node(rank: int, world_size: int, local_world_size: int) -> int:
    if world_size <= local_world_size:
        raise ValueError("node remote example requires at least two nodes")
    if world_size % local_world_size:
        raise ValueError("world_size must be divisible by LOCAL_WORLD_SIZE")
    return (rank + local_world_size) % world_size


def main() -> None:
    n_elements = int(os.environ.get("N_ELEMENTS", "64"))
    if n_elements <= 0:
        raise ValueError("N_ELEMENTS must be positive")

    mem_pool = tle.get_mem_pool()
    if mem_pool is None:
        raise RuntimeError("FlagCX MemPool is unavailable; check FlagCX library/header setup")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    peer_world_rank = _peer_on_next_node(rank, world_size, local_world_size)

    with torch.cuda.use_mem_pool(mem_pool):
         source = (
            torch.arange(n_elements, dtype=torch.float32, device="cuda") + rank * 1000
        ).clone()

    torch.cuda.synchronize()

    ctx = tle.create_dist_tensor(source)

    torch.cuda.synchronize()
    dist.barrier()

    output = torch.empty_like(source)

    _read_remote_world_rank_kernel[(n_elements,)](
        output,
        ctx=ctx,
        PEER_WORLD_RANK=peer_world_rank,
        num_warps=4,
    )
    torch.cuda.synchronize()

    expected = torch.arange(n_elements, dtype=torch.float32, device="cuda")
    expected += peer_world_rank * 1000
    local_ok = torch.equal(output, expected)
    result = torch.tensor([int(local_ok)], dtype=torch.int32, device="cuda")
    dist.all_reduce(result, op=dist.ReduceOp.MIN)

    if local_ok:
        print(
            f"[rank {rank}] read world rank {peer_world_rank} on another node: PASS",
            flush=True,
        )
    else:
        print(
            f"[rank {rank}] FAIL: got {output[:4].tolist()}, "
            f"expected {expected[:4].tolist()}",
            flush=True,
        )

    tle.cleanup_communicator()
    if not bool(result.item()):
        sys.exit(1)


if __name__ == "__main__":
    main()