from __future__ import annotations

import dataclasses
from typing import Optional

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle


# Reduce-Scatter context: stores GPU/node organization, buffers, communication
# pointers, and CUDA streams.
@dataclasses.dataclass
class TleReduceScatter2DContext:

    # Auto-generated constructor.
    max_M: int
    N: int
    rank: int
    world_size: int
    local_world_size: int
    dtype: torch.dtype
    with_gemm_output: bool


    gemm_out_dev_comm_ptr: Optional[int]
    gemm_out_dev_mem_ptr: Optional[int]
    gemm_out_buf: Optional[torch.Tensor]
    scatter_buf: torch.Tensor
    rs_per_node_buf: torch.Tensor
    p2p_buf: torch.Tensor
    signal_buf: torch.Tensor
    scatter_dev_comm_ptr: int
    scatter_dev_mem_ptr: int
    rs_per_node_dev_comm_ptr: int
    rs_per_node_dev_mem_ptr: int
    p2p_dev_comm_ptr: int
    p2p_dev_mem_ptr: int
    signal_dev_comm_ptr: int
    signal_dev_mem_ptr: int


    reduction_stream: torch.cuda.Stream
    p2p_stream: torch.cuda.Stream
    num_sync_sms: int
    num_p2p_sms: int
    num_reduction_sms: int
    num_scatter_sms: int


    scatter_signal_buf: torch.Tensor = dataclasses.field(init=False)
    rs_per_node_signal_buf: torch.Tensor = dataclasses.field(init=False)
    local_rank: int = dataclasses.field(init=False)
    node_id: int = dataclasses.field(init=False)
    nnodes: int = dataclasses.field(init=False)
    _finalized: bool = dataclasses.field(init=False, default=False)

    # Compute local_rank, node_id, nnodes, scatter_signal_buf, and validate
    # arguments.
    def __post_init__(self):
        if self.world_size < 2:
            raise ValueError("TLE reduce-scatter requires at least two GPUs")
        if self.world_size % self.local_world_size:
            raise ValueError("world_size must be divisible by local_world_size")
        if self.max_M % self.world_size:
            raise ValueError("max_M must be divisible by world_size")
        self.local_rank = self.rank % self.local_world_size
        self.node_id = self.rank // self.local_world_size
        self.nnodes = self.world_size // self.local_world_size
        if self.nnodes != 1:
            raise NotImplementedError(
                "TLE has no node-level remote primitive yet; this example only "
                "implements the nnodes == 1 specialization")
        if self.local_rank != self.rank:
            raise ValueError("single-node rank must equal local_rank")
        if self.signal_buf.numel() < 2 * self.world_size:
            raise ValueError("signal_buf must contain scatter and per-node signals")
        if self.num_scatter_sms < 1:
            raise ValueError("num_scatter_sms must be positive")

        # Take the first world_size elements of signal_buf as scatter_signal_buf,
        # and the next world_size elements as rs_per_node_signal_buf.
        self.scatter_signal_buf = self.signal_buf[:self.world_size]
        # The last world_size elements of signal_buf are used as
        # rs_per_node_signal_buf for multi-node.
        self.rs_per_node_signal_buf = self.signal_buf[self.world_size:2 * self.world_size]


    @property
    def num_rs_sms(self) -> int:
        if self.nnodes == 1:
            return self.num_scatter_sms
        return (self.num_scatter_sms + self.num_sync_sms +
                self.num_p2p_sms + self.num_reduction_sms)
    def finalize(self):
        """Collectively release TLE/FlagCX after the context is no longer used."""
        if self._finalized:
            return
        torch.cuda.synchronize()
        tle.cleanup_communicator()
        self._finalized = True

    def reset_barriers(self):
        self.signal_buf.zero_()

# Intra-node scatter: write each target shard from this rank to the target rank.
@triton.jit
def _scatter_kernel(
    input_ptr,
    local_scatter_ptr,
    dev_mem_ptr,
    ready_ptr,
    M_per_rank,
    N,
    LOCAL_RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    SCATTER_NODE_SLICE_OFFSET_ELEMS: tl.constexpr,
    WAIT_FOR_READY: tl.constexpr,
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

    # On destination rank t, source rank LOCAL_RANK owns slot LOCAL_RANK.
    source_slot_offset_elems = LOCAL_RANK * M_per_rank * N
    for step in range(WORLD_SIZE):
        target_rank = (LOCAL_RANK + step + 1) % WORLD_SIZE

        # ready_ptr + target_rank points to the ready flag for the target rank
        # on the current GPU. Poll until the GEMM data produced by this GPU for
        # target_rank is complete.
        if WAIT_FOR_READY:
            while tl.atomic_add(ready_ptr + target_rank, 0, sem="acquire",
                                scope="gpu") == 0:
                pass

        if target_rank == LOCAL_RANK:
            remote_base = local_scatter_ptr + source_slot_offset_elems
        else:
            remote_base = tle.remote(
                dev_mem_ptr,
                space="device",
                dtype=input_ptr.dtype.element_ty,
                shard_id=target_rank,
                offset=SCATTER_NODE_SLICE_OFFSET_ELEMS + source_slot_offset_elems,
            )

        for local_tile in range(pid, tiles_per_peer, num_pid):
            tile_m = local_tile // num_tiles_n
            tile_n = local_tile % num_tiles_n
            input_row = target_rank * M_per_rank + tile_m * BLOCK_M
            input_col = tile_n * BLOCK_N
            input_ptrs = (input_ptr +
                          (input_row + row_offs[:, None]) * N +
                          input_col + col_offs[None, :])
            row_mask = (input_row + row_offs[:, None]
                        ) < (target_rank + 1) * M_per_rank
            col_mask = (input_col + col_offs[None, :]) < N
            values = tl.load(input_ptrs, mask=row_mask & col_mask, other=0.0)

            scatter_row = tile_m * BLOCK_M
            scatter_ptrs = (remote_base +
                            (scatter_row + row_offs[:, None]) * N +
                            input_col + col_offs[None, :])
            scatter_row_mask = (scatter_row + row_offs[:, None]) < M_per_rank
            tl.store(scatter_ptrs, values, mask=scatter_row_mask & col_mask)


# Device-side intra-node barrier: wait for all GPUs' scatter writes to become
# visible.
@triton.jit
def _device_barrier_kernel(dev_comm_ptr, WORLD_SIZE: tl.constexpr):
    # This is a FlagCX/TLE intra-node device barrier, not a CUDA CTA barrier.
    tle.distributed_barrier(
        comm_ptr=dev_comm_ptr,
        space="device",
        group_kind="block",
        barrier_kind="sync",
        order="acqrel",
        index=0,
    )


# TMA reduction: read each source on this GPU and write the summed result.
@triton.jit
def _ring_reduce_tma_kernel(
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
    scatter_desc = tl.make_tensor_descriptor(
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

    for tile_id in range(pid, total_tiles, num_pid):
        tile_m = tile_id // num_tiles_n
        tile_n = tile_id % num_tiles_n
        row = tile_m * BLOCK_M
        col = tile_n * BLOCK_N
        source_rank = (LOCAL_RANK + 1) % WORLD_SIZE
        accum = scatter_desc.load([row + source_rank * M_per_rank, col])

        for i in range(1, WORLD_SIZE):
            source_rank = (LOCAL_RANK + i + 1) % WORLD_SIZE
            accum += scatter_desc.load([row + source_rank * M_per_rank, col])
        output_desc.store([row, col], accum)


# Context factory: allocate and register all communication buffers.
def create_tle_reduce_scatter_2d_ctx(max_M: int, N: int, rank: int,
                                      world_size: int, local_world_size: int,
                                      dtype: torch.dtype,
                                      with_gemm_output: bool = False,
                                      reduction_stream: Optional[torch.cuda.Stream] = None,
                                      num_reduction_sms: int = 15,
                                      num_scatter_sms: int = 16) -> TleReduceScatter2DContext:

    if world_size < 2:
        raise ValueError("TLE reduce-scatter requires at least two GPUs")
    if world_size % local_world_size:
        raise ValueError("world_size must be divisible by local_world_size")
    if max_M % world_size:
        raise ValueError("max_M must be divisible by world_size")
    if world_size // local_world_size != 1:
        raise NotImplementedError(
            "TLE has no node-level remote primitive yet; this example only "
            "implements the nnodes == 1 specialization")

    # Buffer allocation.
    per_node_rows = max_M // local_world_size
    with torch.cuda.use_mem_pool(tle.get_mem_pool()):
        gemm_out_buf = (torch.empty((max_M, N), dtype=dtype, device="cuda")
                        if with_gemm_output else None)
        scatter_buf = torch.empty((max_M, N), dtype=dtype, device="cuda")
        rs_per_node_buf = torch.empty((per_node_rows, N), dtype=dtype,
                                      device="cuda")
        p2p_buf = torch.empty((per_node_rows, N), dtype=dtype, device="cuda")
        signal_buf = torch.empty((2 * world_size,), dtype=torch.int32,
                                 device="cuda")
    signal_buf.zero_()

    # Create multiple communication windows.
    scatter_dev_comm_ptr, scatter_dev_mem_ptr = tle.create_comm_tensor(scatter_buf)

    rs_per_node_dev_comm_ptr, rs_per_node_dev_mem_ptr = tle.create_comm_tensor(rs_per_node_buf)

    p2p_dev_comm_ptr, p2p_dev_mem_ptr = tle.create_comm_tensor(p2p_buf)

    signal_dev_comm_ptr, signal_dev_mem_ptr = tle.create_comm_tensor(signal_buf)


    gemm_out_dev_comm_ptr, gemm_out_dev_mem_ptr = ((None, None) if gemm_out_buf is None
                                                    else tle.create_comm_tensor(gemm_out_buf))
    return TleReduceScatter2DContext(
        max_M=max_M,
        N=N,
        rank=rank,
        world_size=world_size,
        local_world_size=local_world_size,
        dtype=dtype,
        with_gemm_output=with_gemm_output,
        gemm_out_buf=gemm_out_buf,
        scatter_buf=scatter_buf,
        gemm_out_dev_comm_ptr=gemm_out_dev_comm_ptr,
        gemm_out_dev_mem_ptr=gemm_out_dev_mem_ptr,
        rs_per_node_buf=rs_per_node_buf,
        p2p_buf=p2p_buf,
        signal_buf=signal_buf,
        scatter_dev_comm_ptr=scatter_dev_comm_ptr,
        scatter_dev_mem_ptr=scatter_dev_mem_ptr,
        rs_per_node_dev_comm_ptr=rs_per_node_dev_comm_ptr,
        rs_per_node_dev_mem_ptr=rs_per_node_dev_mem_ptr,
        p2p_dev_comm_ptr=p2p_dev_comm_ptr,
        p2p_dev_mem_ptr=p2p_dev_mem_ptr,
        signal_dev_comm_ptr=signal_dev_comm_ptr,
        signal_dev_mem_ptr=signal_dev_mem_ptr,
        reduction_stream=(reduction_stream if reduction_stream is not None else
                          torch.cuda.Stream(priority=-1)),
        p2p_stream=torch.cuda.Stream(priority=-1),
        num_sync_sms=0,
        num_p2p_sms=1,
        num_reduction_sms=num_reduction_sms,
        num_scatter_sms=num_scatter_sms,
    )


def _set_tma_allocator():
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)



# Perform intra-node scatter and partial reduction per target node.
# Per-target-node local Reduce-Scatter and P2P.
def reduce_scatter_for_each_node(
        input_tensor: torch.Tensor,
        stream: torch.cuda.Stream,
        ctx: TleReduceScatter2DContext,
        ready_flags: Optional[torch.Tensor] = None) -> torch.Tensor:

    # Intra-node Reduce-Scatter and the subsequent inter-node P2P.
    world_size = ctx.world_size
    local_world_size = ctx.local_world_size
    local_rank = ctx.local_rank
    reduction_stream = ctx.reduction_stream
    num_reduction_sms = ctx.num_reduction_sms
    nnodes = ctx.nnodes
    node_id = ctx.node_id
    rs_per_node_buf = ctx.rs_per_node_buf
    p2p_buf = ctx.p2p_buf
    M, N = input_tensor.shape
    M_per_rank = M // world_size
    M_per_node = M_per_rank * local_world_size

#    Set the number of scatter CTAs.
    scatter_grid = lambda META: (min(
        triton.cdiv(M_per_rank, META["BLOCK_M"]) *
        triton.cdiv(N, META["BLOCK_N"]),
        ctx.num_scatter_sms,
    ), )

    def reduce_launch_config(num_sms: int):
        if num_sms == -1:
            return (lambda META: (
                triton.cdiv(M_per_rank, META["BLOCK_M"]) *
                triton.cdiv(N, META["BLOCK_N"]), )), 64, 4
        return (lambda META: (min(
            triton.cdiv(M_per_rank, META["BLOCK_M"]) *
            triton.cdiv(N, META["BLOCK_N"]), num_sms), )), 128, 8

    # Plain RS has no upstream GEMM producer, so it does not wait on a signal;
    # when fused with GEMM, ready_flags[target_rank] indicates the corresponding
    # GEMM tile is complete.
    ready_ptr = ready_flags if ready_flags is not None else input_tensor

    with torch.cuda.stream(stream):
        for n in range(nnodes):
            # Same node-level swizzle as tutorial 06: the final round targets the
            # current node.
            cur_node_id = (node_id + n + 1) % nnodes

            # Take all rows belonging to the target node; for single-node this is
            # the full input.
            input_intra_node = input_tensor[
                cur_node_id * M_per_node:(cur_node_id + 1) * M_per_node]

            # Buffer where the target node's data will be stored.
            scatter_for_node = ctx.scatter_buf[
                cur_node_id * M_per_node:(cur_node_id + 1) * M_per_node]

            # Buffer holding the partial reduction result computed by this node for
            # the target node.
            rs_per_node_output = rs_per_node_buf[
                cur_node_id * M_per_rank:(cur_node_id + 1) * M_per_rank]

            # Signals are ordered by global rank; this round uses only the
            # local-rank segment for the target node.

            # Starting signal index for the target node.
            signal_start = cur_node_id * local_world_size

            # Flag used when coordinating with GEMM.
            if ready_flags is not None:
                ready_for_node = ready_flags[
                    signal_start:signal_start + local_world_size
                ]
            else:
                ready_for_node = input_tensor

            # dev_mem_ptr points to the full scatter_buf window; remote access
            # first jumps to the staging slice of the current target node. Offset is
            # in elements, not bytes.
            scatter_node_slice_offset_elems = cur_node_id * M_per_node * N

            # Scatter data for the target node within the current node.
            _scatter_kernel[scatter_grid](
                input_intra_node,
                scatter_for_node,
                ctx.scatter_dev_mem_ptr,
                ready_for_node,
                M_per_rank,
                N,
                LOCAL_RANK=local_rank,
                WORLD_SIZE=local_world_size,
                SCATTER_NODE_SLICE_OFFSET_ELEMS=scatter_node_slice_offset_elems,
                WAIT_FOR_READY=ready_flags is not None,
                BLOCK_M=256,
                BLOCK_N=128,
                num_warps=4,
            )

            # Wait for all GPUs' intra-node scatter writes to become visible.
            _device_barrier_kernel[(local_world_size, )](
                ctx.scatter_dev_comm_ptr, WORLD_SIZE=local_world_size)

            # Limit reduction SMs for non-final nodes; the final node uses the full
            # tile grid.
            node_reduce_sms = (-1 if n == nnodes - 1 else
                               num_reduction_sms)

            reduce_grid, reduce_block_n, reduce_warps = reduce_launch_config(
                node_reduce_sms)

            if stream is not reduction_stream:
                reduction_stream.wait_stream(stream)

            with torch.cuda.stream(reduction_stream):
                # Reduce only among the local_world_size GPUs of the current node.
                # Result is written to rs_per_node_output.
                _ring_reduce_tma_kernel[reduce_grid](
                    scatter_for_node,
                    rs_per_node_output,
                    M_per_rank,
                    N,
                    LOCAL_RANK=local_rank,
                    WORLD_SIZE=local_world_size,
                    BLOCK_M=256,
                    BLOCK_N=reduce_block_n,
                    num_warps=reduce_warps,
                )


                # TLE node-space remote/P2P primitive is not yet wired up.
                if nnodes > 1:
                    pass

    if stream is not reduction_stream:
        stream.wait_stream(reduction_stream)
    if nnodes == 1:
        return rs_per_node_buf[:M_per_rank * nnodes]
    return p2p_buf[:M_per_rank * nnodes]


def reduce_scatter_multi_node(
        input_tensor: torch.Tensor,
        stream: torch.cuda.Stream,
        ctx: TleReduceScatter2DContext,
        output: torch.Tensor,
        ready_flags: Optional[torch.Tensor] = None) -> torch.Tensor:

    M, N = input_tensor.shape
    M_per_rank = M // ctx.world_size
    ctx.p2p_stream.wait_stream(stream)

    # Intra-node reduce-scatter; returns each node's reduction result. In the
    # single-node case, this step completes the operation.
    rs_result_per_node = reduce_scatter_for_each_node(
        input_tensor, stream, ctx, ready_flags)


    final_grid = lambda META: (
        triton.cdiv(M_per_rank, META["BLOCK_M"]) *
        triton.cdiv(N, META["BLOCK_N"]), )

    # In single-node case, pass local_rank=0, world_size=1 to copy
    # rs_result_per_node directly to output.
    with torch.cuda.stream(stream):
        _ring_reduce_tma_kernel[final_grid](
            rs_result_per_node,
            output,
            M_per_rank,
            N,
            LOCAL_RANK=ctx.node_id,
            WORLD_SIZE=ctx.nnodes,
            BLOCK_M=256,
            BLOCK_N=64,
            num_warps=4,
        )
    return output


# Dispatch function that calls reduce_scatter_multi_node.
def reduce_scatter_2d_op(input_tensor: torch.Tensor,
                         ctx: TleReduceScatter2DContext,
                         output: Optional[torch.Tensor] = None,
                         ready_flags: Optional[torch.Tensor] = None) -> torch.Tensor:

    M, N = input_tensor.shape
    if input_tensor.dtype != ctx.dtype or N != ctx.N:
        raise ValueError("input shape/dtype does not match reduce-scatter context")
    if M > ctx.max_M or M % ctx.world_size:
        raise ValueError("M must divide world_size and fit in the context")
    M_per_rank = M // ctx.world_size
    if M_per_rank < 256:
        raise ValueError("M_per_rank must be >= 256 for the TMA reduce kernel")
    if output is None:
        output = torch.empty((M_per_rank, N), dtype=input_tensor.dtype,
                             device=input_tensor.device)
    if tuple(output.shape) != (M_per_rank, N):
        raise ValueError("output has an invalid reduce-scatter shape")
    if ready_flags is not None and ready_flags.numel() != ctx.world_size:
        raise ValueError("ready_flags must contain one entry per target rank")

    _set_tma_allocator()
    reduction_stream = ctx.reduction_stream
    scatter_stream = torch.cuda.current_stream()
    if scatter_stream is reduction_stream:
        raise ValueError("scatter_stream and reduction_stream must be distinct")


    reduction_stream.wait_stream(scatter_stream)

    with torch.cuda.stream(scatter_stream):
        _device_barrier_kernel[(ctx.world_size, )](ctx.scatter_dev_comm_ptr,
                                                    WORLD_SIZE=ctx.world_size)


    output = reduce_scatter_multi_node(input_tensor, scatter_stream, ctx,
                                       output, ready_flags)


    with torch.cuda.stream(scatter_stream):
        ctx.reset_barriers()
    return output

# PyTorch baseline implementation.
def torch_rs(input_tensor: torch.Tensor, TP_GROUP) -> torch.Tensor:
    output = torch.empty(
        (input_tensor.shape[0] // TP_GROUP.size(),
        input_tensor.shape[1]),
        dtype=input_tensor.dtype,
        device=input_tensor.device
    )
    dist.reduce_scatter_tensor(output, input_tensor, group=TP_GROUP)
    return output



def main():
    # get_mem_pool initializes both NCCL's process group and the FlagCX runtime.
    tle.get_mem_pool()
    rank = dist.get_rank()
    TP_GROUP = dist.group.WORLD
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    torch.cuda.set_device(local_rank)


    if world_size < 2:
        print("This example needs at least two GPUs", file=sys.stderr)
        return
    if local_world_size != world_size:
        raise NotImplementedError("TLE node-level reduce-scatter is not implemented")
    if torch.cuda.get_device_capability()[0] < 9:
        print("Skip: the TMA reduce kernel requires sm90 or newer")
        tle.cleanup_communicator()
        return

    dtype = torch.bfloat16

    M, N = 8192, 16384
    #
    ctx = create_tle_reduce_scatter_2d_ctx(M, N, rank, world_size,
                                           local_world_size, dtype)
    input_tensor = torch.rand((M, N), dtype=dtype, device="cuda")

    # PyTorch baseline implementation.
    torch_output = torch_rs(input_tensor, TP_GROUP)
    torch.cuda.synchronize()

    output = reduce_scatter_2d_op(input_tensor, ctx)
    torch.cuda.current_stream().wait_stream(ctx.reduction_stream)
    torch.cuda.synchronize()

    torch.testing.assert_close(torch_output, output, atol=6e-2, rtol=6e-2)
    print(f"[Rank {rank}] TLE single-node reduce-scatter passed: {tuple(output.shape)}")
    ctx.finalize()


if __name__ == "__main__":
    main()
