#!/bin/bash

rm -rf ~/.triton/cache

# FlagCX environment variables (tune for your machine if needed)
export FLAGCX_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export FLAGCX_USE_HETERO_COMM=1
export FLAGCX_MEM_ENABLE=1
export FLAGCX_VMM_ENABLE=0
export FLAGCX_P2P_DISABLE=0
export CUDA_VISIBLE_DEVICES=0,1,2,3

run_test() {
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    torchrun \
        --nproc_per_node=4 \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=8333 \
        "${script_dir}/test_tle_intra_node_reduce_scatter.py"
}

run_test

if [ $? -ne 0 ]; then
    echo "ERROR: tle_intra_node_reduce_scatter failed"
    exit 1
fi
