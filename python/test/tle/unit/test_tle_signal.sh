#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export FLAGCX_IB_HCA="${FLAGCX_IB_HCA:-mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_6,mlx5_7,mlx5_8,mlx5_9}"
export FLAGCX_USE_HETERO_COMM="${FLAGCX_USE_HETERO_COMM:-1}"
export FLAGCX_MEM_ENABLE="${FLAGCX_MEM_ENABLE:-1}"
export FLAGCX_VMM_ENABLE="${FLAGCX_VMM_ENABLE:-0}"
export FLAGCX_P2P_DISABLE="${FLAGCX_P2P_DISABLE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

nproc_per_node="${NPROC_PER_NODE:-2}"
master_addr="${MASTER_ADDR:-localhost}"
master_port="${MASTER_PORT:-8333}"

torchrun \
    --nproc_per_node="${nproc_per_node}" \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="${master_addr}" \
    --master_port="${master_port}" \
    "${script_dir}/test_tle_signal.py"
