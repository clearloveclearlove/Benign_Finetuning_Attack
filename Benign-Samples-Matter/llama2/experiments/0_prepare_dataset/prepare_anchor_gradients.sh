#!/bin/bash

# 配置参数
MODEL_PATH="/home1/yibiao/PTM/Llama-2-7b-chat-hf"
MLENS=10  # 只取前10个token的梯度
NUM_SAMPLES=10  # 每个anchor数据集的样本数

# 梯度输出根目录 - 修改为当前用户有权限的路径
GRADIENT_BASE_DIR="/mnt/newdisk/yibiao/gradient"

echo "=========================================="
echo "使用在线计算平均梯度 (节省磁盘空间)"
echo "=========================================="

# 1. 生成 harmful anchor 的平均梯度 (在线计算)
echo ""
echo "Generating harmful anchor gradients (online mean calculation)..."
HARMFUL_ANCHOR="ft_datasets/pure_bad_dataset/pure-bad-illegal-activities-selected10.jsonl"
HARMFUL_OUTPUT="${GRADIENT_BASE_DIR}/Llama-2-7B-Chat/harmful_anchor_gradients"

mkdir -p $HARMFUL_OUTPUT

python3 get_gradients.py \
  --model_name $MODEL_PATH \
  --dataset pure_bad_dataset \
  --data_path $HARMFUL_ANCHOR \
  --run_validation False \
  --use_lora False \
  --save_full_gradients True \
  --online_mean True \
  --normalize True \
  --grads_output_dir $HARMFUL_OUTPUT \
  --batch_size_training 1 \
  --max_response_length $MLENS

# 重命名输出文件以保持一致性
if [ -f "$HARMFUL_OUTPUT/mean_gradients.pt" ]; then
    mv $HARMFUL_OUTPUT/mean_gradients.pt $HARMFUL_OUTPUT/normalized_average_num${NUM_SAMPLES}.pt
    echo "✓ Harmful anchor gradients saved to: $HARMFUL_OUTPUT/normalized_average_num${NUM_SAMPLES}.pt"
fi

# 2. 生成 safe anchor 1 的平均梯度 (在线计算)
echo ""
echo "Generating safe anchor 1 gradients (online mean calculation)..."
SAFE_ANCHOR1="ft_datasets/pure_bad_dataset/pure-bad-illegal-acticities-selected-10-anchor1.jsonl"
SAFE1_OUTPUT="${GRADIENT_BASE_DIR}/Llama-2-7B-Chat/safe_anchor1_gradients"

mkdir -p $SAFE1_OUTPUT

python3 get_gradients.py \
  --model_name $MODEL_PATH \
  --dataset pure_bad_dataset \
  --data_path $SAFE_ANCHOR1 \
  --run_validation False \
  --use_lora False \
  --save_full_gradients True \
  --online_mean True \
  --normalize True \
  --grads_output_dir $SAFE1_OUTPUT \
  --batch_size_training 1 \
  --max_response_length $MLENS

# 重命名输出文件以保持一致性
if [ -f "$SAFE1_OUTPUT/mean_gradients.pt" ]; then
    mv $SAFE1_OUTPUT/mean_gradients.pt $SAFE1_OUTPUT/normalized_average_num${NUM_SAMPLES}.pt
    echo "✓ Safe anchor 1 gradients saved to: $SAFE1_OUTPUT/normalized_average_num${NUM_SAMPLES}.pt"
fi

# 3. 生成 safe anchor 2 的平均梯度 (在线计算)
echo ""
echo "Generating safe anchor 2 gradients (online mean calculation)..."
SAFE_ANCHOR2="ft_datasets/pure_bad_dataset/pure-bad-illegal-activities-selected-10-anchor2.jsonl"
SAFE2_OUTPUT="${GRADIENT_BASE_DIR}/Llama-2-7B-Chat/safe_anchor2_gradients"

mkdir -p $SAFE2_OUTPUT

python3 get_gradients.py \
  --model_name $MODEL_PATH \
  --dataset pure_bad_dataset \
  --data_path $SAFE_ANCHOR2 \
  --run_validation False \
  --use_lora False \
  --save_full_gradients True \
  --online_mean True \
  --normalize True \
  --grads_output_dir $SAFE2_OUTPUT \
  --batch_size_training 1 \
  --max_response_length $MLENS

# 重命名输出文件以保持一致性
if [ -f "$SAFE2_OUTPUT/mean_gradients.pt" ]; then
    mv $SAFE2_OUTPUT/mean_gradients.pt $SAFE2_OUTPUT/normalized_average_num${NUM_SAMPLES}.pt
    echo "✓ Safe anchor 2 gradients saved to: $SAFE2_OUTPUT/normalized_average_num${NUM_SAMPLES}.pt"
fi

echo ""
echo "=========================================="
echo "All anchor gradients generated successfully!"
echo "=========================================="
echo ""
echo "生成的文件:"
echo "  - $HARMFUL_OUTPUT/normalized_average_num${NUM_SAMPLES}.pt"
echo "  - $SAFE1_OUTPUT/normalized_average_num${NUM_SAMPLES}.pt"
echo "  - $SAFE2_OUTPUT/normalized_average_num${NUM_SAMPLES}.pt"
echo ""
echo "优势: 不保存中间梯度文件，大幅节省磁盘空间！"