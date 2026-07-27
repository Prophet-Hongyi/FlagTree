from __future__ import annotations

import dataclasses
import os
import statistics
import sys
from typing import Optional

import torch
import torch.distributed as dist
import triton
import triton.runtime
import triton.language as tl
import triton.experimental.tle.language as tle

from tle_inter_node_reduce_scatter import (
    TleReduceScatter2DContext,
    create_tle_reduce_scatter_2d_ctx,
    reduce_scatter_2d_op,
)



@dataclasses.dataclass
class TleGemmReduceScatterContext:
    """Tutorial-08-style hierarchical GEMM+RS context."""

    rs_ctx: TleReduceScatter2DContext
    output_dtype: torch.dtype
    rs_stream: torch.cuda.Stream
    num_gemm_sms: int
    BLOCK_M: int = 128
    BLOCK_N: int = 256
    BLOCK_K: int = 64
    GROUP_M: int = 8
    STAGES: int = 3

    def __post_init__(self):
        # rs_stream drives scatter; reduction_stream consumes each completed
        # scatter stage. They must be distinct to create the pipeline.
        if self.rs_stream is self.rs_ctx.reduction_stream:
            raise ValueError("rs_stream and reduction_stream must be distinct")

    def finalize(self):
        """Match tutorial 08 by releasing the owned RS context."""
        self.rs_ctx.finalize()

    def get_gemm_out_buf(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.rs_ctx.gemm_out_buf is None:
            raise RuntimeError("GEMM context must reserve a symmetric GEMM output buffer")
        return self.rs_ctx.gemm_out_buf[:input_tensor.shape[0]]


# Create the context: allocate the GEMM output buffer, scatter signal, and
# Reduce-Scatter buffers.
def create_gemm_rs_context(max_M: int, N: int, rank: int, world_size: int,
                           local_world_size: int, output_dtype: torch.dtype,
                           rs_stream: torch.cuda.Stream,
                           BLOCK_M: int = 128, BLOCK_N: int = 256,
                           BLOCK_K: int = 64, GROUP_M: int = 8,
                           STAGES: int = 3,
                           num_scatter_sms: int = 16) -> TleGemmReduceScatterContext:
    """Build GEMM+RS state on the caller-owned RS stream."""
    if max_M % world_size:
        raise ValueError("max_M must be divisible by world_size")

    # rs_stream is the scatter stream. The factory creates a separate high
    # priority reduction stream, which consumes each completed scatter stage.
    rs_ctx = create_tle_reduce_scatter_2d_ctx(
        max_M, N, rank, world_size, local_world_size, output_dtype,
        with_gemm_output=True, num_scatter_sms=num_scatter_sms)

    total_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    num_gemm_sms = total_sms - rs_ctx.num_rs_sms
    if num_gemm_sms < 1:
        raise ValueError("reduce-scatter SM reservation leaves no SM for GEMM")

    return TleGemmReduceScatterContext(rs_ctx=rs_ctx,
                                       output_dtype=output_dtype,
                                       rs_stream=rs_stream,
                                       num_gemm_sms=num_gemm_sms,
                                       BLOCK_M=BLOCK_M,
                                       BLOCK_N=BLOCK_N,
                                       BLOCK_K=BLOCK_K,
                                       GROUP_M=GROUP_M,
                                       STAGES=STAGES)

# GEMM computation and completion notification.
# counter_ptr = workspace
# ready_ptr   = scatter_signal
# scatter_signal has size world_size; each workspace[j] counts how many output
# tiles for rank j have completed.
#
@triton.jit
def kernel_gemm_rs_producer_persistent(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    ready_ptr,
    counter_ptr,
    RANK: tl.constexpr,
    LOCAL_WORLD_SIZE: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
):

    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n
    node_id = RANK // LOCAL_WORLD_SIZE
    nnodes = WORLD_SIZE // LOCAL_WORLD_SIZE
    tiles_per_sm = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_sm += 1

    a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M, K], strides=[K, 1],
                                       block_shape=[BLOCK_M, BLOCK_K])
    b_desc = tl.make_tensor_descriptor(b_ptr, shape=[N, K], strides=[K, 1],
                                       block_shape=[BLOCK_N, BLOCK_K])
    c_desc = tl.make_tensor_descriptor(c_ptr, shape=[M, N], strides=[N, 1],
                                       block_shape=[BLOCK_M, BLOCK_N])

    M_per_rank = M // WORLD_SIZE
    tiles_m_per_rank = M_per_rank // BLOCK_M
    tiles_per_group = GROUP_M * num_pid_n
    tile_id = start_pid - NUM_SMS
    k_tile = -1
    pid_m = 0
    pid_n = 0
    offs_am = 0
    offs_bn = 0
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, k_tiles * tiles_per_sm):

        k_tile = tl.where(k_tile == k_tiles - 1, 0, k_tile + 1)

        if k_tile == 0:
            tile_id += NUM_SMS
            group_id = tile_id // tiles_per_group
            first_pid_m = group_id * GROUP_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
            logical_pid_m = first_pid_m + tile_id % group_size_m
            pid_n = (tile_id % tiles_per_group) // group_size_m


            m_rank = logical_pid_m // tiles_m_per_rank
            pid_m_intra_rank = logical_pid_m - m_rank * tiles_m_per_rank
            m_node_id = m_rank // LOCAL_WORLD_SIZE
            m_local_rank = m_rank % LOCAL_WORLD_SIZE
            swizzle_m_node_id = (m_node_id + node_id + 1) % nnodes
            swizzle_m_local_rank = (m_local_rank + RANK + 1) % LOCAL_WORLD_SIZE
            swizzle_m_rank = (swizzle_m_node_id * LOCAL_WORLD_SIZE +
                              swizzle_m_local_rank)
            pid_m = swizzle_m_rank * tiles_m_per_rank + pid_m_intra_rank
            offs_am = pid_m * BLOCK_M
            offs_bn = pid_n * BLOCK_N

        a = a_desc.load([offs_am, k_tile * BLOCK_K])
        b = b_desc.load([offs_bn, k_tile * BLOCK_K])
        accumulator = tl.dot(a, b.T, accumulator)

        # If this is the last K-dimension tile.
        if k_tile == k_tiles - 1:

            # Write GEMM output.
            c_desc.store([offs_am, offs_bn], accumulator.to(c_ptr.dtype.element_ty))

            # Determine which target rank's M-row slice this output tile belongs to.
            counter_start = offs_am // M_per_rank
            counter_end = (offs_am + BLOCK_M - 1) // M_per_rank
            counter_end = min(counter_end, WORLD_SIZE - 1)

            for counter_id in range(counter_start, counter_end + 1):
                m_start = M_per_rank * counter_id
                m_end = M_per_rank * (counter_id + 1) - 1
                tiled_m_start = m_start // BLOCK_M
                tiled_m_end = m_end // BLOCK_M
                tiled_m_size = tiled_m_end - tiled_m_start + 1
                tiled_n = tl.cdiv(N, BLOCK_N)

                # Increment once per completed output tile.
                prior = tl.atomic_add(counter_ptr + counter_id, 1,
                                      sem="release", scope="gpu")

                if prior == tiled_m_size * tiled_n - 1:
                    # Tutorial 08's dl.notify(..., signal=1) sets the signal to
                    # 1 rather than accumulating on the old value. Use exchange
                    # to ensure the signal remains a boolean across repeated
                    # calls, preventing 1 -> 2 from causing the consumer to
                    # wait forever.
                    # old value = ready_ptr[counter_id]
                    # ready_ptr[counter_id] = 1
                    tl.atomic_xchg(ready_ptr + counter_id, 1,
                                   sem="release", scope="gpu")

            # Reset accumulator to start computing the next assigned output.
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)




def gemm_rs_producer_persistent(a: torch.Tensor, b: torch.Tensor,
                                c: torch.Tensor, barrier: torch.Tensor,
                                workspace: torch.Tensor, world_size: int,
                                local_world_size: int, rank: int,
                                num_gemm_sms: int,
                                BLOCK_SIZE_M: int = 128,
                                BLOCK_SIZE_N: int = 256,
                                BLOCK_SIZE_K: int = 64,
                                GROUP_SIZE_M: int = 8,
                                STAGES: int = 3):
    """Tutorial-08-style producer wrapper around the TLE Triton kernel."""
    if a.shape[1] != b.shape[1]:
        raise ValueError("incompatible GEMM dimensions")
    if a.dtype != b.dtype:
        raise ValueError("GEMM operands must have the same dtype")
    M, local_K = a.shape
    N = b.shape[0]
    M_per_rank = M // world_size

    if M_per_rank % BLOCK_SIZE_M:
        raise ValueError("M_per_rank must be aligned to BLOCK_SIZE_M")

    # TMA descriptors require a global-memory allocator, as in tutorial 08.
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)
    grid = lambda META: (min(
        num_gemm_sms,
        triton.cdiv(M, META["BLOCK_M"]) *
        triton.cdiv(N, META["BLOCK_N"]),
    ), )

    # stages: pipeline depth parameter for the Triton kernel launch.
    return kernel_gemm_rs_producer_persistent[grid](
        a,
        b,
        c,
        M,
        N,
        local_K,
        barrier,
        workspace,
        RANK=rank,
        LOCAL_WORLD_SIZE=local_world_size,
        WORLD_SIZE=world_size,
        BLOCK_M=BLOCK_SIZE_M,
        BLOCK_N=BLOCK_SIZE_N,
        BLOCK_K=BLOCK_SIZE_K,
        GROUP_M=GROUP_SIZE_M,
        NUM_SMS=num_gemm_sms,
        num_warps=8,
        num_stages=STAGES,
    )


def _pad_to_block_m(input_tensor: torch.Tensor, world_size: int,
                    block_m: int) -> torch.Tensor:
    M, K = input_tensor.shape
    M_per_rank = M // world_size
    padded_M_per_rank = triton.cdiv(M_per_rank, block_m) * block_m
    if padded_M_per_rank == M_per_rank:
        return input_tensor
    reshaped = input_tensor.reshape(world_size, M_per_rank, K)
    padded = torch.empty((world_size, padded_M_per_rank, K),
                         dtype=input_tensor.dtype, device=input_tensor.device)
    padded[:, :M_per_rank].copy_(reshaped)
    return padded.reshape(-1, K)


# GEMM + Reduce-Scatter combined implementation.
def gemm_rs_multi_node_persistent_op(
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        ctx: TleGemmReduceScatterContext) -> torch.Tensor:
    """Tutorial-08 persistent GEMM + per-rank-ready reduce-scatter path."""
    world_size = ctx.rs_ctx.world_size
    local_world_size = ctx.rs_ctx.local_world_size
    rs_stream = ctx.rs_stream
    original_M = input_tensor.shape[0]
    original_M_per_rank = original_M // world_size

    input_tensor = _pad_to_block_m(input_tensor, world_size, ctx.BLOCK_M)
    M, K = input_tensor.shape
    N = weight.shape[0]
    if N != ctx.rs_ctx.N or weight.shape[1] != K:
        raise ValueError("invalid GEMM dimensions for the reduce-scatter context")
    if M > ctx.rs_ctx.max_M:
        raise ValueError("padded M exceeds context capacity")

    current_stream = torch.cuda.current_stream()
    # Queue this dependency before GEMM. The RS stream then waits per target
    # rank on scatter_signal rather than waiting for the complete GEMM launch.
    rs_stream.wait_stream(current_stream)

    output = torch.empty((M // world_size, N), dtype=ctx.output_dtype,
                         device="cuda")

    # Shape: [world_size]
    # Each workspace[j] counts how many output tiles for rank j have completed.
    # When workspace[j] reaches tiled_m_size * tiled_n - 1,
    # all tiles for rank j are done, so trigger the signal notification.
    workspace = torch.zeros((world_size,), dtype=torch.int32,
                            device=input_tensor.device)
    scatter_signal = ctx.rs_ctx.scatter_signal_buf
    gemm_out = ctx.get_gemm_out_buf(input_tensor)

    # Launch compute kernel.
    gemm_rs_producer_persistent(
        input_tensor,
        weight,
        gemm_out,
        scatter_signal,
        workspace,
        world_size,
        local_world_size,
        ctx.rs_ctx.rank,
        ctx.num_gemm_sms,
        BLOCK_SIZE_M=ctx.BLOCK_M,
        BLOCK_SIZE_N=ctx.BLOCK_N,
        BLOCK_SIZE_K=ctx.BLOCK_K,
        GROUP_SIZE_M=ctx.GROUP_M,
        STAGES=ctx.STAGES,
    )

    # Launch communication kernel.
    with torch.cuda.stream(rs_stream):
        reduce_scatter_2d_op(gemm_out, ctx.rs_ctx, output=output,
                             ready_flags=scatter_signal)
    current_stream.wait_stream(rs_stream)
    return output[:original_M_per_rank]


def gemm_rs_multi_node(input_tensor: torch.Tensor, weight: torch.Tensor,
                       ctx: TleGemmReduceScatterContext) -> torch.Tensor:
    """Tutorial-08 public GEMM + Reduce-Scatter entry point."""

    return gemm_rs_multi_node_persistent_op(input_tensor, weight, ctx)




# PyTorch baseline implementation.
def torch_gemm_rs(input_tensor: torch.Tensor, weight: torch.Tensor,
                  TP_GROUP) -> torch.Tensor:
    """PyTorch/NCCL baseline"""
    M, _ = input_tensor.shape
    N = weight.shape[0]
    gemm_out = torch.matmul(input_tensor, weight.T)
    output = torch.empty((M // TP_GROUP.size(), N), dtype=gemm_out.dtype,
                         device=input_tensor.device)
    dist.reduce_scatter_tensor(output, gemm_out, group=TP_GROUP)
    return output

# Timing/benchmark code.
def _time_ms(fn, stream: torch.cuda.Stream, warmup: int = 20,
             iters: int = 200, clear_l2: bool = True) -> float:
    """Measure median latency with rank alignment and optional L2 eviction.

    Cache eviction occurs before the start event, so it is not included in the
    measured latency. Keep the same setting for TLE and torch baselines.
    """
    driver = triton.runtime.driver.active
    cache = driver.get_empty_cache_for_benchmark() if clear_l2 else None

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()

    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        dist.barrier()
        driver.clear_cache(cache)

        start.record(stream)
        fn()
        end.record(stream)
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return float(statistics.median(samples))




def main():

    # Initialization.
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
        raise NotImplementedError("TLE node-level GEMM reduce-scatter is not implemented")

    if torch.cuda.get_device_capability()[0] < 9:
        print("Skip: persistent TMA GEMM requires sm90 or newer")
        tle.cleanup_communicator()
        return

    # Generate input.
    M, N, K = 16384, 12288, 49152
    local_K = K // world_size
    dtype = torch.bfloat16
    scale = rank + 1
    input_tensor = (torch.rand((M, local_K), dtype=dtype, device="cuda") *
                    (0.02 * scale) - 0.01 * scale)
    weight = (torch.rand((N, local_K), dtype=dtype, device="cuda") *
              (0.02 * scale) - 0.01 * scale)

    # Matches the validated TLE path: scatter and intra-node reduction share
    # the caller-provided high-priority RS stream.
    rs_stream = torch.cuda.Stream(priority=-1)

    num_scatter_sms = int(os.environ.get("TLE_SCATTER_SMS", "16"))
    ctx = create_gemm_rs_context(M, N, rank, world_size, local_world_size,
                                 dtype, rs_stream,
                                 num_scatter_sms=num_scatter_sms)
    if rank == 0:
        print(f"[Rank 0] TLE scatter CTA budget: {num_scatter_sms}; "
              f"GEMM persistent CTA budget: {ctx.num_gemm_sms}")

    torch_output = torch_gemm_rs(input_tensor, weight, TP_GROUP)



    tle_output = gemm_rs_multi_node(input_tensor, weight, ctx)


    torch.cuda.synchronize()


    torch.testing.assert_close(torch_output, tle_output, atol=6e-2, rtol=6e-2)
    print(f"[Rank {rank}] TLE overlapping GEMM+RS correctness passed")




    tle_ms = _time_ms(lambda: gemm_rs_multi_node(input_tensor, weight, ctx),
                      torch.cuda.current_stream())

    torch_ms = _time_ms(lambda: torch_gemm_rs(input_tensor, weight, TP_GROUP),
                        torch.cuda.current_stream())

    print(f"[Rank {rank}] TLE GEMM+RS median: {tle_ms:.3f} ms")
    print(f"[Rank {rank}] torch GEMM+RS median: {torch_ms:.3f} ms")
    ctx.finalize()


if __name__ == "__main__":
    main()
