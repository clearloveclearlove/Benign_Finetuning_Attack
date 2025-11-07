#!/bin/bash

# ====================================================================
# ICML25 数据准备脚本 - Self-Influence 方法
# ====================================================================

set -e  # 遇到错误立即退出

# ====================================================================
# GPU 配置
# ====================================================================
# 可以通过命令行参数指定GPU，例如: bash prepare_icml25.sh 1
GPU_ID=${1:-1}  # 默认使用 GPU 0
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "=========================================="
echo "使用 GPU: $GPU_ID"
echo "=========================================="
echo ""

# ====================================================================
# 配置参数
# ====================================================================

# 模型和路径配置
MODEL_PATH="/home1/yibiao/PTM/Llama-2-7b-chat-hf"

# 数据集配置
DATASET_NAME="dolly_dataset"
DATA_PATH="ft_datasets/dolly_dataset/databricks-dolly-15k-no-safety.jsonl"


# 选择配置
K=10            # 选择top 100个样本
TYPE="top"        # top 或 bottom

# 输出目录配置
OUTPUT_BASE_DIR="experiments/0_prepare_dataset/Dolly/Self-Influence"

# ====================================================================
# 检查数据文件是否存在
# ====================================================================

echo "=========================================="
echo "ICML25 - Self-Influence 数据准备"
echo "=========================================="
echo ""

if [ ! -f "$DATA_PATH" ]; then
    echo "❌ 错误: 数据文件不存在: $DATA_PATH"
    exit 1
fi

echo "✓ 数据文件: $DATA_PATH"
echo "✓ 样本数量: $(wc -l < $DATA_PATH)"
echo ""

# ====================================================================
# STEP 1: 计算 Self-Influence 分数
# ====================================================================

echo "=========================================="
echo "Step 1: 计算 Self-Influence 分数"
echo "=========================================="
echo "说明: Self-Influence 使用每个样本自身的梯度计算相似度"
echo "      不需要预先计算的 anchor 梯度"
echo ""

mkdir -p $OUTPUT_BASE_DIR

# 检查是否已经计算过 scores
SCORES_FILE="${OUTPUT_BASE_DIR}/scores.pt"

if [ -f "$SCORES_FILE" ]; then
    echo "⚠️  发现已存在的 scores 文件: $SCORES_FILE"
    read -p "是否重新计算? (y/n, 默认 n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "跳过 Step 1，使用已有的 scores 文件"
        echo ""
        SKIP_STEP1=true
    else
        SKIP_STEP1=false
    fi
else
    SKIP_STEP1=false
fi

if [ "$SKIP_STEP1" = false ]; then
    echo "开始计算 Self-Influence 分数..."

    # 注意：self_influence=True 时不需要 grad_file
    # 但脚本可能要求该参数，所以我们传一个占位符
    DUMMY_GRAD_FILE="experiments/0_prepare_dataset/dummy_grad.pt"

    python3 -m online_gradient.rank rank \
        --model_name $MODEL_PATH \
        --grad_file $DUMMY_GRAD_FILE \
        --dataset $DATASET_NAME \
        --data_path $DATA_PATH \
        --batch_size_training 1 \
        --output_dir $OUTPUT_BASE_DIR \
        --normalize False \
        --self_influence True \
        2>&1 | tee $OUTPUT_BASE_DIR/log.txt

    echo "✓ Self-Influence 分数计算完成"
    echo "✓ 保存位置: $SCORES_FILE"
else
    echo "✓ 使用已有的 Self-Influence 分数"
fi

echo ""

# ====================================================================
# STEP 2: 根据 Self-Influence 分数选择 Top-K 数据
# ====================================================================

echo "=========================================="
echo "Step 2: 选择 Top-K 数据"
echo "=========================================="
echo "配置:"
echo "  - K = $K"
echo "  - Type = $TYPE"
echo "  - Weight = 1 (self-influence)"
echo ""

# 输出目录
WRITE_TO="ft_datasets/dolly_dataset/self_influence"
mkdir -p $WRITE_TO

python3 -m online_gradient.rank write_data \
    --dataset $DATASET_NAME \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_BASE_DIR \
    --weight 1 \
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
echo "  Self-Influence 分数:"
echo "    - $SCORES_FILE"
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
echo "Self-Influence 方法说明:"
echo "  - 不需要预先定义的 anchor 数据集"
echo "  - 每个样本使用自身梯度计算相似度得分"
echo "  - 选择 Self-Influence 分数最高的样本"
echo ""
echo "下一步:"
echo "  使用选择的数据进行微调训练:"
echo "  python3 train.py --data_path $WRITE_TO/dolly_top${K}.json"