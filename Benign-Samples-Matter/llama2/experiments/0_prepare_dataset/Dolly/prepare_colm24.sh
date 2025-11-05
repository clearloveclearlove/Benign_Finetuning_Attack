#!/bin/bash

# ====================================================================
# COLM24 数据准备脚本 - 复现 benign-data-breaks-safety 方法
# ====================================================================

set -e  # 遇到错误立即退出

# ====================================================================
# 配置参数
# ====================================================================

# 模型和路径配置
MODEL_PATH="/root/autodl-tmp/models/Llama-2-7B-Chat"
GRADIENT_BASE_DIR="/home1/yibiao/PTM/Llama-2-7b-chat-hf"

# 数据集配置
DATASET_NAME="dolly_dataset"
DATA_PATH="ft_datasets/dolly_dataset/databricks-dolly-15k-no-safety.jsonl"

# Anchor 配置
TARGET_DATA="illegal-activities"
MLENS=10          # 只计算前10个token的梯度
NUM_SAMPLES=10    # 每个anchor数据集有10个样本

# 选择配置
K=100             # 选择top 100个样本
TYPE="top"        # top 或 bottom

# 输出目录配置
OUTPUT_BASE_DIR="experiments/0_prepare_dataset/Dolly/COLM24"

# ====================================================================
# STEP 0: 确保 anchor 梯度已经生成
# ====================================================================

echo "=========================================="
echo "Step 0: 检查 Anchor 梯度文件"
echo "=========================================="

HARMFUL_ANCHOR_GRAD="${GRADIENT_BASE_DIR}/harmful_anchor_gradients/normalized_average_num${NUM_SAMPLES}.pt"
SAFE_ANCHOR1_GRAD="${GRADIENT_BASE_DIR}/safe_anchor1_gradients/normalized_average_num${NUM_SAMPLES}.pt"
SAFE_ANCHOR2_GRAD="${GRADIENT_BASE_DIR}/safe_anchor2_gradients/normalized_average_num${NUM_SAMPLES}.pt"

# 检查文件是否存在
missing_files=0
for grad_file in "$HARMFUL_ANCHOR_GRAD" "$SAFE_ANCHOR1_GRAD" "$SAFE_ANCHOR2_GRAD"; do
    if [ ! -f "$grad_file" ]; then
        echo "❌ 缺失文件: $grad_file"
        missing_files=$((missing_files + 1))
    else
        echo "✓ 找到文件: $grad_file"
    fi
done

if [ $missing_files -gt 0 ]; then
    echo ""
    echo "错误: 缺少 anchor 梯度文件！"
    echo "请先运行: bash experiments/0_prepare_dataset/prepare_anchor_gradients.sh"
    exit 1
fi

echo "✓ 所有 anchor 梯度文件就绪"
echo ""

# ====================================================================
# STEP 1: 计算训练数据梯度并与 anchor 梯度计算余弦相似度
# ====================================================================

echo "=========================================="
echo "Step 1: 计算训练数据梯度和余弦相似度"
echo "=========================================="
echo ""

# 1.1 计算与 harmful anchor 的相似度
echo "[1/3] 计算与 harmful anchor 的相似度..."
OUTPUT_HARMFUL="${OUTPUT_BASE_DIR}/harmful_anchor"
mkdir -p $OUTPUT_HARMFUL

python3 -m online_gradient.rank rank \
    --model_name $MODEL_PATH \
    --grad_file $HARMFUL_ANCHOR_GRAD \
    --dataset $DATASET_NAME \
    --data_path $DATA_PATH \
    --batch_size_training 1 \
    --output_dir $OUTPUT_HARMFUL \
    --normalize True \
    2>&1 | tee $OUTPUT_HARMFUL/log.txt

echo "✓ Harmful anchor 相似度计算完成"
echo ""

# 1.2 计算与 safe anchor 1 的相似度
echo "[2/3] 计算与 safe anchor 1 的相似度..."
OUTPUT_SAFE1="${OUTPUT_BASE_DIR}/safe_anchor1"
mkdir -p $OUTPUT_SAFE1

python3 -m online_gradient.rank rank \
    --model_name $MODEL_PATH \
    --grad_file $SAFE_ANCHOR1_GRAD \
    --dataset $DATASET_NAME \
    --data_path $DATA_PATH \
    --batch_size_training 1 \
    --output_dir $OUTPUT_SAFE1 \
    --normalize True \
    2>&1 | tee $OUTPUT_SAFE1/log.txt

echo "✓ Safe anchor 1 相似度计算完成"
echo ""

# 1.3 计算与 safe anchor 2 的相似度
echo "[3/3] 计算与 safe anchor 2 的相似度..."
OUTPUT_SAFE2="${OUTPUT_BASE_DIR}/safe_anchor2"
mkdir -p $OUTPUT_SAFE2

python3 -m online_gradient.rank rank \
    --model_name $MODEL_PATH \
    --grad_file $SAFE_ANCHOR2_GRAD \
    --dataset $DATASET_NAME \
    --data_path $DATA_PATH \
    --batch_size_training 1 \
    --output_dir $OUTPUT_SAFE2 \
    --normalize True \
    2>&1 | tee $OUTPUT_SAFE2/log.txt

echo "✓ Safe anchor 2 相似度计算完成"
echo ""

# ====================================================================
# STEP 2: 聚合结果并选择 top-k 数据点
# ====================================================================

echo "=========================================="
echo "Step 2: 聚合结果并选择 Top-K 数据"
echo "=========================================="
echo "配置:"
echo "  - K = $K"
echo "  - Type = $TYPE"
echo "  - Weights = 1 (harmful), -1 (safe1), -1 (safe2)"
echo ""

# 设置 anchor 目录和权重
# 注意：这里必须是空格分隔的字符串，不是数组
ANCHOR_GRADIENT_DIRS="$OUTPUT_HARMFUL $OUTPUT_SAFE1 $OUTPUT_SAFE2"
WEIGHTS="1 -1 -1"

# 输出目录
WRITE_TO="${OUTPUT_BASE_DIR}/aggregate/${TARGET_DATA}-mlens${MLENS}_num${NUM_SAMPLES}-anchor1-2"
mkdir -p $WRITE_TO

python3 -m online_gradient.rank write_data \
    --dataset $DATASET_NAME \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_HARMFUL $OUTPUT_SAFE1 $OUTPUT_SAFE2 \
    --weight $WEIGHTS \
    --k $K \
    --type $TYPE \
    --write_to $WRITE_TO \
    --dataset_name dolly

echo "✓ Top-K 数据选择完成"
echo ""

# ====================================================================
# 完成总结
# ====================================================================

echo "=========================================="
echo "数据准备完成！"
echo "=========================================="
echo ""
echo "生成的文件:"
echo "  相似度计算结果:"
echo "    - $OUTPUT_HARMFUL/scores.pt"
echo "    - $OUTPUT_SAFE1/scores.pt"
echo "    - $OUTPUT_SAFE2/scores.pt"
echo ""
echo "  Top-K 选择结果:"
echo "    - $WRITE_TO/dolly_top${K}.json"
echo "    - $WRITE_TO/dolly_top${K}_index.json"
echo ""

# 显示选择的数据统计
if [ -f "$WRITE_TO/dolly_top${K}.json" ]; then
    echo "已选择的数据样本数:"
    python3 -c "import json; data = json.load(open('$WRITE_TO/dolly_top${K}.json')); print(f'  样本数: {len(data)}')"
    echo ""
fi

echo "✅ 全部完成！"
echo ""
echo "下一步:"
echo "  使用选择的数据进行微调训练:"
echo "  python3 train.py --data_path $WRITE_TO/dolly_top${K}.json"