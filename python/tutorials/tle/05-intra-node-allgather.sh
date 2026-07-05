#!/usr/bin/env bash
set -euo pipefail  # 出错立即退出、未定义变量报错、管道中任一命令失败就算失败

# 计算脚本所在目录，后面用它定位同目录下的 .py 文件
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 如果第一个参数是 debug，就打开 NCCL/FlagCX 的调试日志；
# 比如 bash 05-intra-node-allgather.sh debug
if [[ "${1:-}" == "debug" ]]; then
    export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"  # 打开 NCCL/FlagCX 的调试日志
    export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-all}"  # 默认打开全部子系统日志
    export FLAGCX_DEBUG="${FLAGCX_DEBUG:-TRACE}"   # 设置 FlagCX 日志等级， 默认设成 TRACE， 方便看 FlagCX 内部通信细节
fi

# Defaults mirror the TLE distributed unit tests. Override these from the shell
# when your machine needs a different setup.
export FLAGCX_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_6,mlx5_7,mlx5_8,mlx5_9 # 让 FlagCX 只使用这些 HCA 设备做通信
export FLAGCX_MEM_ENABLE=1  # 默认开启 FlagCX 内存分配支持
export FLAGCX_USE_HETERO_COMM=1  # 默认开启 FlagCX 的 hetero communication 路径
export FLAGCX_VMM_ENABLE=0   # 默认关闭 VMM。这里跟 TLE 分布式单测的默认配置保持一致
export FLAGCX_P2P_DISABLE=1  # 默认禁用 P2P
export CUDA_VISIBLE_DEVICES=0,1  # 默认使用 GPU 0 和 GPU 1
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"  # 避免 PyTorch extension 自动选择当前环境不支持的 12.0+PTX


# 如果用户设置了 CLEAR_TRITON_CACHE=1，就清理 Triton 编译缓存，强制重新编译 kernel
if [[ "${CLEAR_TRITON_CACHE:-0}" == "1" ]]; then
    rm -rf "${TRITON_CACHE_DIR:-${HOME}/.triton/cache}"
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"      # 默认每个节点启动 2 个进程，对应两张 GPU
MASTER_ADDR="${MASTER_ADDR:-localhost}"    # 默认 master 地址是本机
MASTER_PORT="${MASTER_PORT:-8333}"         # 默认分布式初始化端口是 8333


# 每个节点启动的进程数
# 节点数为 1
# 当前节点的 rank 为 0
# 主节点的地址
# 主节点的端口
torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${SCRIPT_DIR}/05-intra-node-allgather.py"
