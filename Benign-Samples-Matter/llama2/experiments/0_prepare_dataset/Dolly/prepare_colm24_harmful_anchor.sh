#!/bin/bash

# ====================================================================
# 仅使用 harmful anchor
# ====================================================================

# 注意：不使用 set -e，使用更精细的错误处理
# set -e

# ====================================================================
# GPU 配置
# ====================================================================
# 可以通过命令行参数指定GPU，例如: bash prepare_harmful_only.sh 0
GPU_ID=${1:-0}  # 默认使用 GPU 0
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "=========================================="
echo "Gradient_Match 数据准备 - 使用 GPU: $GPU_ID"
echo "=========================================="
echo ""

# ====================================================================
# 配置参数
# ====================================================================

# 模型和路径配置
MODEL_PATH="/home1/yibiao/PTM/Llama-2-7b-chat-hf"
GRADIENT_BASE_DIR="/mnt/newdisk/yibiao/gradient/Llama-2-7B-Chat"

# 数据集配置
DATASET_NAME="dolly_dataset"
DATA_PATH="ft_datasets/dolly_dataset/databricks-dolly-15k-no-safety.jsonl"

# Anchor 配置
TARGET_DATA="illegal-activities"
MLENS=10          # 只计算前10个token的梯度
NUM_SAMPLES=10    # 每个anchor数据集有10个样本

# 选择配置
K=30             # 选择top 100个样本
TYPE="top"        # top 或 bottom

# 输出目录配置
OUTPUT_BASE_DIR="experiments/0_prepare_dataset/Dolly/Gradient_Match"

# ====================================================================
# STEP 0: 确保 anchor 梯度已经生成
# ====================================================================

echo "=========================================="
echo "Step 0: 检查 Anchor 梯度文件"
echo "=========================================="

HARMFUL_ANCHOR_GRAD="${GRADIENT_BASE_DIR}/harmful_anchor_gradients/normalized_average_num${NUM_SAMPLES}.pt"

# 检查文件是否存在
if [ ! -f "$HARMFUL_ANCHOR_GRAD" ]; then
    echo "❌ 缺失文件: $HARMFUL_ANCHOR_GRAD"
    echo ""
    echo "❌ 错误: 缺少 harmful anchor 梯度文件！"
    echo "请先运行: bash experiments/0_prepare_dataset/prepare_anchor_gradients.sh"
    exit 1
else
    echo "✓ 找到文件: $HARMFUL_ANCHOR_GRAD"
fi

echo "✓ Harmful anchor 梯度文件就绪"
echo ""

# ====================================================================
# STEP 1: 计算训练数据梯度并与 harmful anchor 梯度计算余弦相似度
# ====================================================================

echo "=========================================="
echo "Step 1: 计算训练数据梯度和余弦相似度"
echo "=========================================="
echo ""

OUTPUT_HARMFUL="${OUTPUT_BASE_DIR}/harmful_anchor"
mkdir -p $OUTPUT_HARMFUL

SCORES_FILE="${OUTPUT_HARMFUL}/scores.pt"

# 检查是否已存在 scores 文件
if [ -f "$SCORES_FILE" ]; then
    echo "⚠️  发现已存在的 scores 文件: $SCORES_FILE"
    read -p "是否重新计算? (y/n, 默认 n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "✓ 跳过计算，使用已有的 scores 文件"
        echo ""
        SKIP_STEP1=true
    else
        SKIP_STEP1=false
    fi
else
    SKIP_STEP1=false
fi

if [ "$SKIP_STEP1" = false ]; then
    echo "计算与 harmful anchor 的相似度..."

    # 执行计算
    python3 -m online_gradient.rank rank \
        --model_name $MODEL_PATH \
        --grad_file $HARMFUL_ANCHOR_GRAD \
        --dataset $DATASET_NAME \
        --data_path $DATA_PATH \
        --batch_size_training 1 \
        --output_dir $OUTPUT_HARMFUL \
        --normalize True \
        --max_response_length $MLENS \
        2>&1 | tee $OUTPUT_HARMFUL/log.txt

    # 验证 scores.pt 是否生成
    if [ ! -f "$SCORES_FILE" ]; then
        echo ""
        echo "❌ 错误: scores.pt 文件未生成"
        echo "计算可能失败，请查看日志: $OUTPUT_HARMFUL/log.txt"
        exit 1
    fi

    echo "✓ Harmful anchor 相似度计算完成"
else
    echo "✓ 使用已有的 harmful anchor scores 文件"
fi

echo ""

# ====================================================================
# STEP 2: 根据相似度分数选择 Top-K 数据
# ====================================================================

echo "=========================================="
echo "Step 2: 选择 Top-K 数据"
echo "=========================================="
echo "配置:"
echo "  - K = $K"
echo "  - Type = $TYPE"
echo "  - Weight = 1 (harmful only)"
echo ""

# 验证 scores 文件存在
if [ ! -f "$SCORES_FILE" ]; then
    echo "❌ 错误: 找不到 scores 文件: $SCORES_FILE"
    echo "请先完成 Step 1"
    exit 1
fi

# 输出目录
WRITE_TO="ft_datasets/dolly_dataset/Gradient_Match"
mkdir -p $WRITE_TO

python3 -m online_gradient.rank write_data \
    --dataset $DATASET_NAME \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_HARMFUL \
    --weight 1 \
    --k $K \
    --type $TYPE \
    --write_to $WRITE_TO \
    --dataset_name dolly

# 检查 write_data 是否成功
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 错误: Top-K 数据选择失败"
    exit 1
fi

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
echo "    - $SCORES_FILE"
echo ""
echo "  Top-K 选择结果:"
echo "    - $WRITE_TO/dolly_top${K}.json"
echo "    - $WRITE_TO/dolly_top${K}_index.json"
echo ""

# 显示选择的数据统计
if [ -f "$WRITE_TO/dolly_top${K}.json" ]; then
    echo "已选择的数据样本数:"
    python3 -c "import json; data = json.load(open('$WRITE_TO/dolly_top${K}.json')); print(f'  样本数: {len(data)}')" 2>/dev/null || echo "  无法读取统计信息"
    echo ""
fi

echo "✅ 全部完成！"
echo ""
echo "Gradient_Match 方法说明:"
echo "  - 仅使用 harmful anchor"
echo "  - 计算与 harmful anchor 的相似度"
echo "  - 选择相似度分数最高的样本"
echo ""
echo "下一步:"
echo "  使用选择的数据进行微调训练:"
echo "  python3 train.py --data_path $WRITE_TO/dolly_top${K}.json"