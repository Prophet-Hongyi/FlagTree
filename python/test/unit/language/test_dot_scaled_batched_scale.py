import pytest
import torch
import triton
import triton.language as tl
from triton.compiler.errors import CompilationError


BATCH = 2
BLOCK_M = 16
BLOCK_N = 16
BLOCK_K = 32


@triton.jit
def _dot_scaled_batched_scale_kernel(
    lhs_ptr,
    rhs_ptr,
    lhs_scale_ptr,
    rhs_scale_ptr,
    output_ptr,
    BATCH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batches = tl.arange(0, BATCH)[:, None, None]
    rows = tl.arange(0, BLOCK_M)[None, :, None]
    columns = tl.arange(0, BLOCK_N)[None, None, :]
    reduction = tl.arange(0, BLOCK_K)

    lhs_offsets = batches * BLOCK_M * BLOCK_K + rows * BLOCK_K + reduction[None, None, :]
    rhs_offsets = batches * BLOCK_K * BLOCK_N + reduction[None, :, None] * BLOCK_N + columns
    lhs = tl.load(lhs_ptr + lhs_offsets)
    rhs = tl.load(rhs_ptr + rhs_offsets)

    lhs_scale_offsets = batches * BLOCK_M + rows
    rhs_scale_columns = tl.arange(0, BLOCK_N)[None, :, None]
    rhs_scale_offsets = batches * BLOCK_N + rhs_scale_columns
    lhs_scale = tl.load(lhs_scale_ptr + lhs_scale_offsets)
    rhs_scale = tl.load(rhs_scale_ptr + rhs_scale_offsets)

    output = tl.dot_scaled(lhs, lhs_scale, "e5m2", rhs, rhs_scale, "e5m2")
    output_offsets = batches * BLOCK_M * BLOCK_N + rows * BLOCK_N + columns
    tl.store(output_ptr + output_offsets, output)


def _e8m0_to_float(scale):
    return torch.pow(2.0, scale.to(torch.float32) - 127.0)


def _exception_chain_text(exception):
    messages = []
    seen = set()
    while exception is not None and id(exception) not in seen:
        seen.add(id(exception))
        messages.append(str(exception))
        exception = exception.__cause__ or exception.__context__
    return "\n".join(messages)


def test_dot_scaled_batched_scale_contract(device):
    lhs = (
        (torch.arange(BATCH * BLOCK_M * BLOCK_K, dtype=torch.int64) * 7 + 3) % 5 - 2
    ).reshape(BATCH, BLOCK_M, BLOCK_K).to(torch.float32)
    rhs = (
        (torch.arange(BATCH * BLOCK_K * BLOCK_N, dtype=torch.int64) * 11 + 5) % 5 - 2
    ).reshape(BATCH, BLOCK_K, BLOCK_N).to(torch.float32)
    lhs_storage = lhs.to(torch.float8_e5m2).view(torch.uint8)
    rhs_storage = rhs.to(torch.float8_e5m2).view(torch.uint8)
    lhs_scale = torch.tensor([126, 127, 128, 127], dtype=torch.uint8).reshape(1, 4, 1).repeat(
        BATCH, BLOCK_M // 4, 1
    )
    rhs_scale = torch.tensor([128, 127, 126, 127], dtype=torch.uint8).reshape(1, 4, 1).repeat(
        BATCH, BLOCK_N // 4, 1
    )
    output = torch.empty((BATCH, BLOCK_M, BLOCK_N), dtype=torch.float32, device=device)

    def launch():
        return _dot_scaled_batched_scale_kernel[(1, )](
            lhs_storage.to(device), rhs_storage.to(device), lhs_scale.to(device), rhs_scale.to(device), output,
            BATCH=BATCH, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)

    target = triton.runtime.driver.active.get_current_target()
    options = triton.compiler.compiler.make_backend(target).parse_options({})
    if not getattr(options, "supports_batched_dot_scaled", False):
        with pytest.raises(CompilationError) as exc_info:
            launch()
        assert "batched dot_scaled is not supported by this backend" in _exception_chain_text(exc_info.value)
        return

    program = launch()

    lhs_decoded = lhs_storage.view(torch.float8_e5m2).to(torch.float32)
    rhs_decoded = rhs_storage.view(torch.float8_e5m2).to(torch.float32)
    expected = torch.bmm(lhs_decoded, rhs_decoded)
    expected *= _e8m0_to_float(lhs_scale[..., 0])[:, :, None]
    expected *= _e8m0_to_float(rhs_scale[..., 0])[:, None, :]

    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=1e-5)
    assert "tt.dot_scaled" in program.asm["ttir"]
    binary_keys = ("hgbin", "mubin", "mcfatbin", "hsaco", "cubin", "npubin")
    assert any(key in program.asm and len(program.asm[key]) > 0 for key in binary_keys)
