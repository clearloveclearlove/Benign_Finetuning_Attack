#!/bin/bash

# ====================================================================
# COLM24 数据准备脚本 - 复现 benign-data-breaks-safety 方法
# ====================================================================

# 注意：不使用 set -e，使用更精细的错误处理
# set -e

# ====================================================================
# GPU 配置和参数
# ====================================================================
# 使用方法: bash prepare_colm24.sh [GPU_ID] [K]
# 例如: bash prepare_colm24.sh 0 100  # 使用 GPU 0, 选择 top 100 样本
# 例如: bash prepare_colm24.sh 0 500  # 使用 GPU 0, 选择 top 500 样本
GPU_ID=${1:-0}    # 第一个参数: GPU ID，默认使用 GPU 0
K=${2:-100}       # 第二个参数: Top-K 样本数，默认为 100
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "=========================================="
echo "COLM24 数据准备 - 使用 GPU: $GPU_ID"
echo "          Top-K 样本数: $K"
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
# K 现在通过命令行参数指定（见上方）
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
    echo "❌ 错误: 缺少 anchor 梯度文件！"
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

# 定义一个函数来计算相似度（带缓存检查）
calculate_similarity() {
    local anchor_name=$1
    local grad_file=$2
    local output_dir=$3
    local step_num=$4

    echo "[$step_num] 计算与 $anchor_name 的相似度..."
    mkdir -p $output_dir

    local scores_file="${output_dir}/scores.pt"

    # 检查是否已存在 scores 文件
    if [ -f "$scores_file" ]; then
        echo "⚠️  发现已存在的 scores 文件: $scores_file"
        read -p "是否重新计算? (y/n, 默认 n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "✓ 跳过计算，使用已有的 scores 文件"
            echo ""
            return 0
        fi
    fi

    # 执行计算
    python3 -m online_gradient.rank rank \
        --model_name $MODEL_PATH \
        --grad_file $grad_file \
        --dataset $DATASET_NAME \
        --data_path $DATA_PATH \
        --batch_size_training 1 \
        --output_dir $output_dir \
        --normalize True \
        --max_response_length $MLENS \
        2>&1 | tee $output_dir/log.txt

    # 验证 scores.pt 是否生成
    if [ ! -f "$scores_file" ]; then
        echo ""
        echo "❌ 错误: scores.pt 文件未生成"
        echo "计算可能失败，请查看日志: $output_dir/log.txt"
        exit 1
    fi

    echo "✓ $anchor_name 相似度计算完成"
    echo ""
}

# 1.1 计算与 harmful anchor 的相似度
OUTPUT_HARMFUL="${OUTPUT_BASE_DIR}/harmful_anchor"
calculate_similarity "harmful anchor" "$HARMFUL_ANCHOR_GRAD" "$OUTPUT_HARMFUL" "1/3"

# 1.2 计算与 safe anchor 1 的相似度
OUTPUT_SAFE1="${OUTPUT_BASE_DIR}/safe_anchor1"
calculate_similarity "safe anchor 1" "$SAFE_ANCHOR1_GRAD" "$OUTPUT_SAFE1" "2/3"

# 1.3 计算与 safe anchor 2 的相似度
OUTPUT_SAFE2="${OUTPUT_BASE_DIR}/safe_anchor2"
calculate_similarity "safe anchor 2" "$SAFE_ANCHOR2_GRAD" "$OUTPUT_SAFE2" "3/3"

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

# 验证所有 scores 文件都存在
for scores_file in "$OUTPUT_HARMFUL/scores.pt" "$OUTPUT_SAFE1/scores.pt" "$OUTPUT_SAFE2/scores.pt"; do
    if [ ! -f "$scores_file" ]; then
        echo "❌ 错误: 找不到 scores 文件: $scores_file"
        echo "请先完成 Step 1"
        exit 1
    fi
done

# 输出目录
WRITE_TO="ft_datasets/dolly_dataset/COLM24"
mkdir -p $WRITE_TO

python3 -m online_gradient.rank write_data \
    --dataset $DATASET_NAME \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_HARMFUL $OUTPUT_SAFE1 $OUTPUT_SAFE2 \
    --weight 1 -1 0 \
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
    python3 -c "import json; data = json.load(open('$WRITE_TO/dolly_top${K}.json')); print(f'  样本数: {len(data)}')" 2>/dev/null || echo "  无法读取统计信息"
    echo ""
fi

echo "✅ 全部完成！"
echo ""
echo "COLM24 方法说明:"
echo "  - 使用 harmful anchor + 2 个 safe anchors"
echo "  - 计算加权相似度: 1×harmful - 1×safe1 - 1×safe2"
echo "  - 选择加权分数最高的样本"
echo ""
echo "下一步:"
echo "  使用选择的数据进行微调训练:"
echo "  python3 train.py --data_path $WRITE_TO/dolly_top${K}.json"