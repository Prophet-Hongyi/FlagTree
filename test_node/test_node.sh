#!/usr/bin/env bash
set -euo pipefail

# e.g: bash  xxx.sh 0   bash  xxx.sh 1
NODE_RANK=${1:?set this node rank: 0 or 1}


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export FLAGCX_IB_HCA=mlx5_0,mlx5_1,mlx5_2
export FLAGCX_USE_HETERO_COMM=1
export FLAGCX_MEM_ENABLE=1
export FLAGCX_VMM_ENABLE=0
export FLAGCX_P2P_DISABLE=0

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec torchrun \
    --nnodes=2 \
    --nproc_per_node=1 \
    --node_rank="${NODE_RANK}" \
    --master_addr=9.2.0.7 \
    --master_port=29500 \
    "${script_dir}/test_tle_inter_node_remote.py" "$@"