"""
Intra-node AllGather with FlagTree TLE device remote pointers.
This tutorial implements a single-node all-gather operator using
FlagTree TLE.
Run with a FlagTree environment, for example:

    export FLAGCX_MEM_ENABLE=1
    export FLAGCX_USE_HETERO_COMM=1
    export FLAGCX_VMM_ENABLE=0
    export FLAGCX_P2P_DISABLE=1
    export CUDA_VISIBLE_DEVICES=0,1
    # Optional: set FLAGCX_IB_HCA to the HCA list for your machine.
    torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${SCRIPT_DIR}/05-intra-node-allgather.py"

If you explicitly disabled distributed support with USE_FLAGCX=0, USE_DIST=0,
or USE_TLE_DIST=0, it might be necessary to reset these settings before running this tutorial.
"""

import os

import torch
import torch.distributed as dist
import triton
import triton.language as tl
import triton.experimental.tle.language as tle


@triton.jit
def _all_gather_push_kernel(
    ag_ptr,
    signal_ptr,
    ag_dev_mem,
    signal_dev_mem,
    dev_comm_dptr,  # DevComm handle, used to query the current rank within the kernel
    mesh: tl.constexpr,
    ELEM_PER_RANK: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    SIGNAL_TARGET: tl.constexpr,
):
    peer = tl.program_id(0)
    local_rank = tle.shard_id(mesh, "device", comm_ptr=dev_comm_dptr)

    if peer != local_rank:
        dst_base = local_rank * ELEM_PER_RANK
        dst_ptr = tle.remote(
            ag_dev_mem,
            shard_id=peer,
            space="device",
            dtype=ag_ptr.dtype.element_ty,
            offset=dst_base,
        )
        for block_id in tl.range(0, NUM_BLOCKS):
            offsets = block_id * BLOCK + tl.arange(0, BLOCK)
            mask = offsets < ELEM_PER_RANK
            vals = tl.load(ag_ptr + dst_base + offsets, mask=mask, other=0.0)
            tl.store(dst_ptr + offsets, vals, mask=mask)

        remote_signal_ptr = tle.remote(
            signal_dev_mem,
            shard_id=peer,
            space="device",
            dtype=tl.int32,
            offset=local_rank,
        )
        tl.store(remote_signal_ptr, SIGNAL_TARGET, cache_modifier=".wt")
    else:
        tl.store(signal_ptr + local_rank, SIGNAL_TARGET)


def _rank_print(rank: int, *items):
    dist.barrier()
    for cur_rank in range(dist.get_world_size()):
        if cur_rank == rank:
            print(*items, flush=True)
        dist.barrier()


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    mem_pool = tle.get_mem_pool()
    if mem_pool is None:
        raise RuntimeError("FlagCX memory pool is unavailable; check FlagCX build and environment variables.")

    rank = dist.get_rank()  # Obtain rank and world_size
    world_size = dist.get_world_size()
    local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", str(world_size)))
    assert world_size == local_world_size, "This tutorial is designed for a single node"

    M = 8192
    N = 12288
    assert M % world_size == 0
    m_per_rank = M // world_size
    dtype = torch.float16
    device = torch.device("cuda", local_rank)

    local_data = torch.randn((m_per_rank, N), dtype=dtype, device=device)

    with torch.cuda.use_mem_pool(mem_pool):
        ag_buffer = torch.empty((M, N), dtype=dtype, device=device)
        signal = torch.empty((world_size, ), dtype=torch.int32, device=device)

    dev_comm_dptr, ag_dev_mem = tle.create_comm_tensor(ag_buffer)
    _, signal_dev_mem = tle.create_comm_tensor(signal)
    # ag_dev_mem is the device-side DevMem handle address created by FlagCX/TLE,
    # signal_dev_mem is used to remotely write to the peer's signal[local_rank].
    # dev_comm_dptr is used on the device side by tle.shard_id(..., comm_ptr=...) to query the current rank.

    golden = torch.empty((M, N), dtype=dtype, device=device)
    dist.all_gather_into_tensor(golden, local_data)

    ag_buffer.fill_(-1)
    ag_buffer[rank * m_per_rank:(rank + 1) * m_per_rank, :].copy_(local_data)
    signal.zero_()

    torch.cuda.synchronize()
    dist.barrier()

    elem_per_rank = m_per_rank * N
    block = 1024
    num_blocks = triton.cdiv(elem_per_rank, block)

    # 1D grid: Each producer CTA is responsible for pushing this rank's shard to a peer.
    # Immediately after writing the data, it performs a remote write to signal[local_rank].
    grid = (world_size, )
    mesh = tle.device_mesh(tle.MeshConfig(device=world_size))
    _all_gather_push_kernel[grid](
        ag_buffer,
        signal,
        ag_dev_mem,
        signal_dev_mem,
        dev_comm_dptr,
        mesh,
        ELEM_PER_RANK=elem_per_rank,
        BLOCK=block,
        NUM_BLOCKS=num_blocks,
        SIGNAL_TARGET=1,
        num_warps=4,
    )

    torch.cuda.synchronize()
    dist.barrier()

    _rank_print(rank, f"Rank {rank} FlagTree Result:", ag_buffer)
    _rank_print(rank, f"Rank {rank} FlagTree Signal:", signal)
    assert torch.allclose(golden, ag_buffer, atol=1e-5, rtol=1e-5)
    _rank_print(rank, f"Rank {rank} Pass!")

    tle.cleanup_communicator()


if __name__ == "__main__":
    main()
