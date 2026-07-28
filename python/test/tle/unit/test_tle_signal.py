import os

import pytest
import torch
import torch.distributed as dist
import triton
import triton.language as tl
import triton.experimental.tle.language as tle

DEVICE_MESH = tle.device_mesh(tle.MeshConfig(device=2))


@triton.jit()
def _signal_kernel(device_dptr: tl.constexpr, mesh: tl.constexpr):
    local_rank = tle.shard_id(mesh, "device", device_dptr=device_dptr)
    peer = (local_rank + 1) % mesh.shape[0]

    tle.signal(
        device_dptr,
        peer,
        signal_id=0,
        op="inc",
        space="intra_node",
        group_kind="block",
        context_idx=0,
    )
    tle.signal(
        device_dptr,
        peer,
        signal_id=1,
        value=local_rank + 2,
        op="add",
        space="intra_node",
        group_kind="block",
        context_idx=1,
    )


def _ir_verify(device_dptr):
    compiled = _signal_kernel.warmup(
        device_dptr=device_dptr,
        mesh=DEVICE_MESH,
        grid=(1, ),
        num_ctas=1,
        num_warps=4,
    )
    assert "tle.signal" in compiled.asm["ttgir"]
    assert "flagcxDevNetGetFromCommS" in compiled.asm["ptx"]
    assert "flagcxDevNetSignalSigIncS" in compiled.asm["ptx"]
    assert "flagcxDevNetSignalSigAddS" in compiled.asm["ptx"]


class TestSignal:

    @pytest.mark.skipif(
        not tle.communication.enabled or "RANK" not in os.environ,
        reason="requires torchrun and a configured FlagCX runtime",
    )
    def test_tle_signal_compiles(self):
        mem_pool = tle.get_mem_pool()
        rank = dist.get_rank()
        with torch.cuda.use_mem_pool(mem_pool):
            backing = torch.tensor([rank], dtype=torch.int32, device="cuda")

        device_dptr = tle.create_dist_tensor(backing)
        try:
            _ir_verify(device_dptr)
        finally:
            tle.cleanup_communicator()


if __name__ == "__main__":
    TestSignal().test_tle_signal_compiles()
