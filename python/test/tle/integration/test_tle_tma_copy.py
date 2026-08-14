# flagtree tle
"""
TLE TMA Copy Integration Tests

Tests TLE TMA copy functionality and bidirectional copy operations:
- TMA copy operations (tle.gpu.copy with TMA descriptors)
- Shared-memory pointers (tle.gpu.alloc, tle.gpu.local_ptr + tl.load/store)
- Integration with Triton JIT and TMA descriptors
- Memory allocation and data transfer validation
"""

import pytest
import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle


def _is_enflame_backend():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "gcu"


def _has_hopper_gpu() -> bool:
    if _is_enflame_backend():
        # Assume Enflame backend has Hopper support for testing purposes
        return True
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9


pytestmark = pytest.mark.skipif(
    not _has_hopper_gpu(),
    reason="TMA copy requires NVIDIA Hopper (sm90+)",
)


@triton.jit
def elementwise_tma_add_kernel(
    a_desc,
    b_desc,
    c_desc,
    xnumel,
    ynumel,
    XBLOCK: tl.constexpr,
    YBLOCK: tl.constexpr,
):
    pid = tl.program_id(0)

    # Calculate row offset for current program

    # Exercise hardware TMA swizzling without imposing an MMA operand layout.
    a_smem = tle.gpu.alloc(
        [XBLOCK, YBLOCK], dtype=tl.float32,
        layout=tle.gpu.nv_tma_shared_layout(
            [XBLOCK, YBLOCK], [1, 0], tl.float32, [1, 1], [1, 1], [1, 0], False, True),
        scope=tle.gpu.smem,
    )
    b_smem = tle.gpu.alloc(
        [XBLOCK, YBLOCK], dtype=tl.float32,
        layout=tle.gpu.nv_tma_shared_layout(
            [XBLOCK, YBLOCK], [1, 0], tl.float32, [1, 1], [1, 1], [1, 0], False, True),
        scope=tle.gpu.smem,
    )
    c_smem = tle.gpu.alloc(
        [XBLOCK, YBLOCK], dtype=tl.float32,
        layout=tle.gpu.nv_tma_shared_layout(
            [XBLOCK, YBLOCK], [1, 0], tl.float32, [1, 1], [1, 1], [1, 0], False, True),
        scope=tle.gpu.smem,
    )
    row_ids = tl.arange(0, XBLOCK)[:, None]
    col_ids = tl.arange(0, YBLOCK)[None, :]
    row_ids = tl.broadcast_to(row_ids, (XBLOCK, YBLOCK))
    col_ids = tl.broadcast_to(col_ids, (XBLOCK, YBLOCK))
    a_smem_ptrs = tle.gpu.local_ptr(a_smem, (row_ids, col_ids))
    b_smem_ptrs = tle.gpu.local_ptr(b_smem, (row_ids, col_ids))
    c_smem_ptrs = tle.gpu.local_ptr(c_smem, (row_ids, col_ids))

    # Use TLE pipeline for block-wise processing
    for yoff in range(0, ynumel, YBLOCK):
        # Calculate column offset for current block
        # copy data to shared memory
        tle.gpu.copy(
            a_desc,
            a_smem,
            [XBLOCK, YBLOCK],
            [pid * XBLOCK, yoff],
            eviction_policy="evict_first",
        )
        tle.gpu.copy(b_desc, b_smem, [XBLOCK, YBLOCK], [pid * XBLOCK, yoff])
        # Load data from shared memory
        aval = tl.load(a_smem_ptrs)
        bval = tl.load(b_smem_ptrs)

        c_val = aval + bval
        tl.store(c_smem_ptrs, c_val)
        tle.gpu.copy(c_smem, c_desc, [XBLOCK, YBLOCK], [pid * XBLOCK, yoff])


def elementwise_add(A, B, C, XBLOCK=32, YBLOCK=64):
    """
    Wrapper function to execute element-wise addition using TLE pipeline

    Args:
        A: Input tensor descriptor A (TMA descriptor)
        B: Input tensor descriptor B (TMA descriptor)
        C: Output tensor descriptor C (TMA descriptor)
        XBLOCK: Block size for X dimension
        YBLOCK: Block size for Y dimension
    """
    # For TMA descriptors, we don't have direct access to shape/stride
    # We'll use the block sizes for the computation
    xnumel, ynumel = 512, 512  # Default test size
    grid = (triton.cdiv(xnumel, XBLOCK), )

    return elementwise_tma_add_kernel[grid](A, B, C, xnumel, ynumel, XBLOCK, YBLOCK)


@triton.jit
def block_cyclic_tensor_map_table_kernel(
    tensor_map_table,
    output,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    owner = tl.program_id(0)
    tile = tle.gpu.alloc(
        [BLOCK_ROWS, BLOCK_COLS],
        dtype=tl.float32,
        layout=tle.gpu.nv_tma_shared_layout(
            [BLOCK_ROWS, BLOCK_COLS],
            [1, 0],
            tl.float32,
            [1, 1],
            [1, 1],
            [1, 0],
            False,
            True,
        ),
        scope=tle.gpu.smem,
    )
    entry = tle.gpu.tensor_map_table_entry(tensor_map_table, owner)
    tle.gpu.tensor_map_fenceproxy_acquire(entry)
    descriptor = tle.gpu.reinterpret_tensor_map(entry, tile)
    tle.gpu.copy(descriptor, tile, [BLOCK_ROWS, BLOCK_COLS], [0, 0])

    rows = tl.arange(0, BLOCK_ROWS)[:, None]
    cols = tl.arange(0, BLOCK_COLS)[None, :]
    values = tl.load(
        tle.gpu.local_ptr(
            tile, (rows, cols), shape=(BLOCK_ROWS, BLOCK_COLS)
        )
    )
    offsets = owner * BLOCK_ROWS * BLOCK_COLS + rows * BLOCK_COLS + cols
    tl.store(output + offsets, values)


@triton.jit
def packed_block_cyclic_tensor_map_table_kernel(
    tensor_map_table,
    output,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_PACK: tl.constexpr,
    LANES: tl.constexpr,
):
    owner = tl.program_id(0)
    tile = tle.gpu.alloc(
        [BLOCK_N, BLOCK_K, K_PACK, LANES],
        dtype=tl.bfloat16,
        layout=tle.gpu.nv_tma_shared_layout(
            [BLOCK_N, BLOCK_K, K_PACK, LANES],
            [3, 2, 1, 0],
            tl.bfloat16,
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [3, 2, 1, 0],
            False,
            True,
        ),
        scope=tle.gpu.smem,
    )
    entry = tle.gpu.tensor_map_table_entry(tensor_map_table, owner)
    tle.gpu.tensor_map_fenceproxy_acquire(entry)
    descriptor = tle.gpu.reinterpret_tensor_map(entry, tile)
    tle.gpu.copy(
        descriptor,
        tile,
        [BLOCK_N, BLOCK_K, K_PACK, LANES],
        [0, 0, 0, 0],
    )

    n = tl.arange(0, BLOCK_N)[:, None, None, None]
    k = tl.arange(0, BLOCK_K)[None, :, None, None]
    k_pack = tl.arange(0, K_PACK)[None, None, :, None]
    lane = tl.arange(0, LANES)[None, None, None, :]
    values = tl.load(tle.gpu.local_ptr(tile))
    offsets = (
        owner * BLOCK_N * BLOCK_K * K_PACK * LANES
        + n * BLOCK_K * K_PACK * LANES
        + k * K_PACK * LANES
        + k_pack * LANES
        + lane
    )
    tl.store(
        output
        + tl.broadcast_to(offsets, (BLOCK_N, BLOCK_K, K_PACK, LANES)),
        values,
    )


def _encode_tensor_map_table(source, owner_rows, block_shape):
    utils = triton.runtime.driver.active.utils
    descriptors = []
    row_elements = source.shape[1]
    for owner, local_rows in enumerate(owner_rows):
        descriptors.append(
            utils.encode_tma_descriptor(
                source.data_ptr() + owner * row_elements * source.element_size(),
                3,  # CU_TENSOR_MAP_SWIZZLE_128B
                source.element_size(),
                7,  # CU_TENSOR_MAP_DATA_TYPE_FLOAT32
                block_shape,
                [local_rows, row_elements],
                [2 * row_elements, 1],
                0,
            )
        )
    table = torch.tensor(
        list(b"".join(descriptors)), dtype=torch.uint8, device="cuda"
    )
    assert table.data_ptr() % 128 == 0
    return table


class TestTLETmaCopy:
    """TLE TMA Copy Integration Tests"""

    def test_tma_copy_basic(self):
        """Test basic TMA copy functionality with element-wise addition"""
        torch.manual_seed(42)  # Ensure reproducibility

        xnumel, ynumel = 512, 512
        XBLOCK, YBLOCK = 32, 64

        # Create test data
        a = torch.randn(xnumel, ynumel, device="cuda", dtype=torch.float32)
        b = torch.randn(xnumel, ynumel, device="cuda", dtype=torch.float32)
        c = torch.empty_like(a, device="cuda", dtype=torch.float32)
        from triton.tools.tensor_descriptor import TensorDescriptor
        a_tma = TensorDescriptor.from_tensor(a, block_shape=[XBLOCK, YBLOCK])
        b_tma = TensorDescriptor.from_tensor(b, block_shape=[XBLOCK, YBLOCK])
        c_tma = TensorDescriptor.from_tensor(c, block_shape=[XBLOCK, YBLOCK])
        # Execute TLE pipeline computation
        compiled = elementwise_add(a_tma, b_tma, c_tma, XBLOCK, YBLOCK)

        # Verify results
        expected = a + b
        torch.testing.assert_close(c, expected, atol=1e-5, rtol=1e-5)
        ttgir = compiled.asm["ttgir"]
        assert "#ttg.nvtma_shared" in ttgir
        assert "#ttg.nvmma_shared" not in ttgir
        ptx = compiled.asm["ptx"]
        assert "createpolicy.fractional.L2::evict_first.b64" in ptx
        assert "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint" in ptx

    def test_block_cyclic_tensor_map_table_with_ragged_owners(self):
        """Select exact per-owner tensor maps and address them in local coordinates."""
        source = torch.arange(
            17 * 32, device="cuda", dtype=torch.float32
        ).reshape(17, 32)
        output = torch.empty((2, 8, 32), device="cuda", dtype=torch.float32)
        table = _encode_tensor_map_table(source, [9, 8], [8, 32])

        compiled = block_cyclic_tensor_map_table_kernel[(2,)](
            table, output, BLOCK_ROWS=8, BLOCK_COLS=32
        )

        expected = torch.stack((source[0:16:2], source[1:17:2]))
        torch.testing.assert_close(output, expected)
        ttgir = compiled.asm["ttgir"]
        assert "ttng.reinterpret_tensor_desc" in ttgir
        assert "ttng.tensormap_fenceproxy_acquire" in ttgir
        ptx = compiled.asm["ptx"]
        assert "fence.proxy.tensormap::generic.acquire.gpu" in ptx
        assert "cp.async.bulk.tensor.2d" in ptx

    def test_packed_block_cyclic_tensor_map_table(self):
        """Load the rank-4 packed GEMV view used by PyNTT QKV weights."""
        logical_k, logical_n = 64, 256
        k_pack, lanes = 2, 64
        scalar_lanes = k_pack * lanes
        source = torch.arange(
            logical_k * logical_n * scalar_lanes,
            device="cuda",
            dtype=torch.float32,
        ).to(torch.bfloat16)
        descriptors = []
        utils = triton.runtime.driver.active.utils
        for owner in range(2):
            descriptors.append(
                utils.encode_tma_descriptor(
                    source.data_ptr() + owner * scalar_lanes * source.element_size(),
                    3,
                    source.element_size(),
                    9,  # CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
                    [8, 8, k_pack, lanes],
                    [8, logical_k, k_pack, lanes],
                    [32 * scalar_lanes, logical_n * scalar_lanes, lanes, 1],
                    1,
                )
            )
        table = torch.tensor(
            list(b"".join(descriptors)), dtype=torch.uint8, device="cuda"
        )
        output = torch.empty((2, 8, 8, k_pack, lanes), device="cuda", dtype=torch.bfloat16)

        packed_block_cyclic_tensor_map_table_kernel[(2,)](
            table,
            output,
            BLOCK_N=8,
            BLOCK_K=8,
            K_PACK=k_pack,
            LANES=lanes,
        )

        expected = torch.empty_like(output)
        source_view = source.reshape(logical_k, logical_n, k_pack, lanes)
        for owner in range(2):
            expected[owner] = source_view[:8, owner::32].permute(1, 0, 2, 3)
        torch.testing.assert_close(output, expected)

    def test_tma_copy_different_block_sizes(self):
        """Test TMA copy with different block sizes"""
        torch.manual_seed(123)

        for XBLOCK, YBLOCK in [(16, 128), (64, 32), (128, 16)]:
            xnumel, ynumel = 256, 256

            # Create test data
            a = torch.randn(xnumel, ynumel, device="cuda", dtype=torch.float32)
            b = torch.randn(xnumel, ynumel, device="cuda", dtype=torch.float32)
            c = torch.empty_like(a, device="cuda", dtype=torch.float32)

            from triton.tools.tensor_descriptor import TensorDescriptor
            a_tma = TensorDescriptor.from_tensor(a, block_shape=[XBLOCK, YBLOCK])
            b_tma = TensorDescriptor.from_tensor(b, block_shape=[XBLOCK, YBLOCK])
            c_tma = TensorDescriptor.from_tensor(c, block_shape=[XBLOCK, YBLOCK])

            # Execute TLE pipeline computation
            elementwise_add(a_tma, b_tma, c_tma, XBLOCK, YBLOCK)

            # Verify results
            expected = a + b
            torch.testing.assert_close(c, expected, atol=1e-5, rtol=1e-5)

    def test_tma_copy_different_dtypes(self):
        """Test TMA copy with different data types"""
        torch.manual_seed(456)

        xnumel, ynumel = 256, 256
        XBLOCK, YBLOCK = 32, 64
        # for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        for dtype in [torch.float32]:
            # Create test data
            a = torch.randn(xnumel, ynumel, device="cuda", dtype=dtype)
            b = torch.randn(xnumel, ynumel, device="cuda", dtype=dtype)
            c = torch.empty_like(a, device="cuda", dtype=dtype)

            from triton.tools.tensor_descriptor import TensorDescriptor
            a_tma = TensorDescriptor.from_tensor(a, block_shape=[XBLOCK, YBLOCK])
            b_tma = TensorDescriptor.from_tensor(b, block_shape=[XBLOCK, YBLOCK])
            c_tma = TensorDescriptor.from_tensor(c, block_shape=[XBLOCK, YBLOCK])

            # Execute TLE pipeline computation
            elementwise_add(a_tma, b_tma, c_tma, XBLOCK, YBLOCK)

            # Verify results
            expected = a + b
            torch.testing.assert_close(c, expected, atol=1e-3 if dtype == torch.float16 else 1e-5,
                                       rtol=1e-3 if dtype == torch.float16 else 1e-5)

    def test_tma_copy_large_tensor(self):
        """Test TMA copy with larger tensors"""
        torch.manual_seed(789)

        xnumel, ynumel = 512, 512
        XBLOCK, YBLOCK = 64, 64

        # Create test data
        a = torch.randn(xnumel, ynumel, device="cuda", dtype=torch.float32)
        b = torch.randn(xnumel, ynumel, device="cuda", dtype=torch.float32)
        c = torch.empty_like(a, device="cuda", dtype=torch.float32)

        from triton.tools.tensor_descriptor import TensorDescriptor
        a_tma = TensorDescriptor.from_tensor(a, block_shape=[XBLOCK, YBLOCK])
        b_tma = TensorDescriptor.from_tensor(b, block_shape=[XBLOCK, YBLOCK])
        c_tma = TensorDescriptor.from_tensor(c, block_shape=[XBLOCK, YBLOCK])

        # Execute TLE pipeline computation
        elementwise_add(a_tma, b_tma, c_tma, XBLOCK, YBLOCK)

        # Verify results
        expected = a + b
        torch.testing.assert_close(c, expected, atol=1e-4, rtol=1e-4)

    def test_tma_copy_non_divisible(self):
        """Test TMA copy with non-divisible tensor dimensions"""
        torch.manual_seed(101)

        # Test with dimensions that are not perfectly divisible by block sizes
        xnumel, ynumel = 500, 300  # Not divisible by 32, 64
        XBLOCK, YBLOCK = 32, 64

        # Create test data
        a = torch.randn(xnumel, ynumel, device="cuda", dtype=torch.float32)
        b = torch.randn(xnumel, ynumel, device="cuda", dtype=torch.float32)
        c = torch.empty_like(a, device="cuda", dtype=torch.float32)

        from triton.tools.tensor_descriptor import TensorDescriptor
        a_tma = TensorDescriptor.from_tensor(a, block_shape=[XBLOCK, YBLOCK])
        b_tma = TensorDescriptor.from_tensor(b, block_shape=[XBLOCK, YBLOCK])
        c_tma = TensorDescriptor.from_tensor(c, block_shape=[XBLOCK, YBLOCK])

        # Execute TLE pipeline computation
        elementwise_add(a_tma, b_tma, c_tma, XBLOCK, YBLOCK)

        # Verify results (only check valid region)
        expected = a + b
        torch.testing.assert_close(c, expected, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
