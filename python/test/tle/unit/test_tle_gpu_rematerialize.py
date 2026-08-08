# flagtree tle
import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle


@triton.jit
def _rematerialize_index_kernel(output, BLOCK: tl.constexpr):
    source = tl.arange(0, BLOCK)
    first = tle.gpu.rematerialize_index(source)
    second = tle.gpu.rematerialize_index(source)
    tl.store(output + source, first + second)


def _require_cuda():
    try:
        torch.cuda.init()
    except Exception as exc:
        pytest.skip(f"CUDA init failed: {exc}")


def test_rematerialize_index_preserves_value_and_program_points():
    _require_cuda()
    output = torch.empty(64, device="cuda", dtype=torch.int32)
    compiled = _rematerialize_index_kernel.warmup(
        output,
        BLOCK=64,
        grid=(1,),
        num_warps=4,
    )

    assert compiled.asm["ttir"].count("tt.elementwise_inline_asm") == 2
    _rematerialize_index_kernel[(1,)](output, BLOCK=64, num_warps=4)
    torch.testing.assert_close(
        output,
        2 * torch.arange(64, device="cuda", dtype=torch.int32),
        atol=0,
        rtol=0,
    )
