#!/bin/bash

# 设置默认环境变量和路径
PRETRAIN_MODEL="<your_policy_model_path>"
# 设置默认值
DEFAULT_BASE_CKPT_DIR="./ppo_exp/hh_rlhf_purm/ckpt/_actor"
DEFAULT_DEVICE="6"

# 解析命令行参数
BASE_CKPT_DIR=$DEFAULT_BASE_CKPT_DIR
DEVICE=$DEFAULT_DEVICE

# 显示帮助信息
function show_help {
  echo "用法: $0 [-d <device>] [-c <checkpoint_dir>]"
  echo "选项:"
  echo "  -d <device>        指定CUDA设备编号 (默认: $DEFAULT_DEVICE)"
  echo "  -c <checkpoint_dir> 指定检查点目录 (默认: $DEFAULT_BASE_CKPT_DIR)"
  echo "  -h                 显示此帮助信息"
  exit 1
}

# 解析命令行选项
while getopts "hd:c:" opt; do
  case $opt in
    d) DEVICE=$OPTARG ;;
    c) BASE_CKPT_DIR=$OPTARG ;;
    h) show_help ;;
    \?) echo "无效选项: -$OPTARG" >&2; show_help ;;
  esac
done

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=$DEVICE
STEPS=($(seq 64 64 1344))
echo "===========开始推理实验==========="
echo "使用检查点目录: $BASE_CKPT_DIR"
echo "使用CUDA设备: $DEVICE"

# python inference_rlhf.py \
#   --model_path "gpt-4o" \

# # 首先使用预训练模型进行推理
# echo "使用预训练模型推理..."
# python inference_rlhf.py \
#   --model_path $PRETRAIN_MODEL \
#   --output_prefix $OUTPUT_PREFIX

#然后对每个检查点进行推理
for step in "${STEPS[@]}"; do
  echo "使用检查点 global_step${step} 推理..."
  python inference_rlhf.py \
    --model_path $BASE_CKPT_DIR \
    --global_step "global_step${step}"
done
