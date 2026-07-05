"""
Intra-node AllGather with FlagTree TLE device remote pointers.

Run with a FlagTree environment, for example:

    export FLAGCX_MEM_ENABLE=1
    export FLAGCX_USE_HETERO_COMM=1
    export FLAGCX_VMM_ENABLE=0
    export FLAGCX_P2P_DISABLE=1
    export CUDA_VISIBLE_DEVICES=0,1
    # Optional: set FLAGCX_IB_HCA to the HCA list for your machine.
    torchrun --nproc_per_node=2 FlagTree/python/tutorials/tle/05-intra-node-allgather.py

If you explicitly disabled distributed support with USE_FLAGCX=0, USE_DIST=0,
or USE_TLE_DIST=0, unset it before running this tutorial.

This mirrors the Triton-distributed tutorial's symmetric-buffer pull path.
Each rank first writes its local shard into a FlagCX-registered all-gather
buffer, then a Triton kernel pulls peer shards through `tle.remote` device
pointers.
"""

import os

import torch
import torch.distributed as dist
import triton
import triton.language as tl
import triton.experimental.tle.language as tle


@triton.jit
def _all_gather_pull_kernel(ag_ptr,  #  当前 rank 本地的 all-gather 输出 buffer
                            ag_dev_mem,  # DevMem 句柄(远端内存句柄)，用来让 kernel 找到远端 rank 的 buffer
                            dev_comm_dptr,  # DevComm 句柄(通信上下文句柄)，用来在 kernel 内查询当前 rank
                            mesh: tl.constexpr, ELEM_PER_RANK: tl.constexpr,  #  每个 rank 的 shard 有多少个元素
                            BLOCK: tl.constexpr,  # 每个 Triton program 一次处理多少元素
                            ):
    peer = tl.program_id(0)
    block_id = tl.program_id(1)
    offsets = block_id * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < ELEM_PER_RANK
    local_rank = tle.shard_id(mesh, "device", comm_ptr=dev_comm_dptr)

    if peer != local_rank:
        src_base = peer * ELEM_PER_RANK
        src_ptr = tle.remote(
            ag_dev_mem,
            shard_id=peer,
            space="device",
            dtype=tl.float16,
            offset=src_base,
        )
        vals = tl.load(src_ptr + offsets, mask=mask, other=0.0)
        tl.store(ag_ptr + src_base + offsets, vals, mask=mask)


@triton.jit
def _mark_pull_done_kernel(signal_ptr, dev_comm_dptr, mesh: tl.constexpr):
    peer = tl.program_id(0)
    local_rank = tle.shard_id(mesh, "device", comm_ptr=dev_comm_dptr)
    if peer != local_rank:
        tl.store(signal_ptr + peer, 1)


def _rank_print(rank: int, *items):
    dist.barrier()
    for cur_rank in range(dist.get_world_size()):
        if cur_rank == rank:
            print(*items, flush=True)
        dist.barrier()


def main():
    mem_pool = tle.get_mem_pool()
    if mem_pool is None:
        raise RuntimeError("FlagCX memory pool is unavailable; check FlagCX build and environment variables.")

    rank = dist.get_rank()  # 获取 rank 和 world_size
    world_size = dist.get_world_size()  #
    local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", str(world_size)))
    assert world_size == local_world_size, "This tutorial is designed for a single node"

    M = 8192
    N = 12288
    assert M % world_size == 0
    m_per_rank = M // world_size
    dtype = torch.float16
    device = torch.device("cuda")

    local_data = torch.randn((m_per_rank, N), dtype=dtype, device=device)

    with torch.cuda.use_mem_pool(mem_pool):  # 在 FlagCX mem pool 里分配 ag_buffer, 后面要注册成 FlagCX/TLE 可解析的远端内存
        ag_buffer = torch.empty((M, N), dtype=dtype, device=device)

    # 一个cuda tensor，作为 signal，告诉每个 rank 其他 rank 的数据已经拉完了
    signal = torch.zeros((world_size, ), dtype=torch.int32, device=device)

    dev_comm_dptr, ag_dev_mem = tle.create_comm_tensor(ag_buffer)
    # ag_dev_mem  FlagCX/TLE 创建出来的 device-side DevMem 句柄地址，
    # 给 Triton kernel 用来解析远端地址。dev_comm_dptr 给设备侧
    # tle.shard_id(..., comm_ptr=...) 查询当前 rank 使用。

    golden = torch.empty((M, N), dtype=dtype, device=device)
    dist.all_gather_into_tensor(golden, local_data)  # 这里对比缺少了一个group参数

    ag_buffer.fill_(-1)
    ag_buffer[rank * m_per_rank:(rank + 1) * m_per_rank, :].copy_(local_data)
    signal.zero_()

    torch.cuda.synchronize()  # 同步当前 CUDA 设备，确保前面的 fill/copy/zero 已经完成
    dist.barrier()  # 跨 rank barrier，确保所有 rank 都已经把自己的 shard 写进自己的 ag_buffer，然后才允许其他 rank 去远程读取

    elem_per_rank = m_per_rank * N  # 每一个 rank 的 shard 有多少个元素
    block = 1024  # 每个 Triton program 处理 1024 个元素

    # grid 是二维：第一维遍历 world_size 个 peer，
    # 第二维遍历 peer shard 的所有 block
    grid = (world_size, triton.cdiv(elem_per_rank, block))  # 主要服务于tle.shard_id 和 tle.remote 等
    mesh = tle.device_mesh(tle.MeshConfig(device=world_size))  #
    _all_gather_pull_kernel[grid](
        ag_buffer,
        ag_dev_mem,
        dev_comm_dptr,
        mesh,
        ELEM_PER_RANK=elem_per_rank,
        BLOCK=block,
        num_warps=4,
    )
    _mark_pull_done_kernel[(world_size, )](signal, dev_comm_dptr, mesh, num_warps=1)

    torch.cuda.synchronize()  # rank内的gpu同步，保证所有 kernel 都执行完了
    dist.barrier()  # 跨rank的gpu 同步

    _rank_print(rank, f"Rank {rank} FlagTree Result:", ag_buffer)
    _rank_print(rank, f"Rank {rank} FlagTree Signal:", signal)
    assert torch.allclose(golden, ag_buffer, atol=1e-5, rtol=1e-5)
    _rank_print(rank, f"Rank {rank} Pass!")

    tle.cleanup_communicator()


if __name__ == "__main__":
    main()
