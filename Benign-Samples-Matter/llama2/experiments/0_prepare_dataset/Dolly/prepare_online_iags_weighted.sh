#!/bin/bash

# ====================================================================
# Online IAGS 数据准备脚本 - 加权版本
# 功能: 结合 COLM24 加权策略 + Interaction-Aware IAGS
# 使用: bash prepare_online_iags_weighted.sh [GPU_ID] [K] [NUM_LAYERS]
# 示例: bash prepare_online_iags_weighted.sh 0 100 8
# Date: 2025-01-13
# Author: clearloveclearlove
# ====================================================================

# ====================================================================
# 参数配置
# ====================================================================
GPU_ID=${1:-0}        # GPU ID，默认 0
K=${2:-100}           # Top-K 样本数，默认 100
NUM_LAYERS=${3:-8}    # 使用最后 N 层，默认 8

export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "=========================================="
echo "Online IAGS - 加权版本 (Weighted GCE)"
echo "=========================================="
echo "配置信息:"
echo "  - GPU: $GPU_ID"
echo "  - Top-K: $K"
echo "  - Layers: 最后 $NUM_LAYERS 层"
echo "  - Weights: 1.0×harmful - 1.0×safe1"
echo "  - Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  - User: $USER"
echo "=========================================="
echo ""

# ====================================================================
# 路径配置
# ====================================================================

MODEL_PATH="/home1/yibiao/PTM/Llama-2-7b-chat-hf"
GRADIENT_BASE_DIR="/mnt/newdisk/yibiao/gradient/Llama-2-7B-Chat"


# 数据集配置
DATASET_NAME="dolly_dataset"
DATA_PATH="ft_datasets/dolly_dataset/COLM24/dolly_top5000.json"

# Anchor 梯度文件
HARMFUL_GRAD="${GRADIENT_BASE_DIR}/harmful_anchor_gradients/normalized_average_num10.pt"
SAFE1_GRAD="${GRADIENT_BASE_DIR}/safe_anchor1_gradients/normalized_average_num10.pt"

# 输出目录
OUTPUT_BASE_DIR="ft_datasets/dolly_dataset/Online-IAGS-Weighted-L${NUM_LAYERS}"

# 其他参数
MLENS=10
NUM_SAMPLES=10

mkdir -p $OUTPUT_BASE_DIR

# ====================================================================
# Step 0: 检查必需的梯度文件
# ====================================================================

echo "=========================================="
echo "Step 0: 检查 Anchor 梯度文件"
echo "=========================================="

missing_files=0
for grad_file in "$HARMFUL_GRAD" "$SAFE1_GRAD"; do
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
# Step 1: 运行 Online IAGS (Weighted GCE)
# ====================================================================

echo "=========================================="
echo "Step 1: 运行 Online IAGS (加权交互感知)"
echo "=========================================="
echo "方法说明:"
echo "  - 使用 Interaction-Aware 梯度选择"
echo "  - 每次迭代选择与 (harmful - safe1) 最相似的样本"
echo "  - 动态更新已选样本集合的联合梯度"
echo "  - 贪心选择 Top-$K 样本"
echo ""
echo "开始执行..."
echo ""

python3 -m online_gradient.run_online_iags \
    --method weighted_gce \
    --model_name $MODEL_PATH \
    --harmful_grad_file $HARMFUL_GRAD \
    --safe_grad_file $SAFE1_GRAD \
    --dataset $DATASET_NAME \
    --data_path $DATA_PATH \
    --k $K \
    --normalize True \
    --num_layers $NUM_LAYERS \
    --max_response_length $MLENS \
    --output_dir $OUTPUT_BASE_DIR \
    --dataset_name dolly \
    --weight_harmful 1.0 \
    --weight_safe -1.0 \
    2>&1 | tee $OUTPUT_BASE_DIR/run.log

# 检查执行结果
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 错误: Online IAGS 执行失败"
    echo "请查看日志: $OUTPUT_BASE_DIR/run.log"
    exit 1
fi

echo ""
echo "✓ Online IAGS 执行完成"
echo ""

# ====================================================================
# 完成总结
# ====================================================================

echo "=========================================="
echo "数据准备完成！"
echo "=========================================="
echo ""
echo "生成的文件:"
echo "  - ${OUTPUT_BASE_DIR}/dolly_top${K}.json"
echo "  - ${OUTPUT_BASE_DIR}/dolly_online_iags_weighted_gce_k${K}_indices.json"
echo "  - ${OUTPUT_BASE_DIR}/dolly_online_iags_weighted_gce_k${K}_gains.json"
echo "  - ${OUTPUT_BASE_DIR}/dolly_online_iags_weighted_gce_k${K}_stats.json"
echo "  - ${OUTPUT_BASE_DIR}/run.log"
echo ""

# 显示选择的数据统计
if [ -f "${OUTPUT_BASE_DIR}/dolly_top${K}.json" ]; then
    echo "已选择的数据样本数:"
    python3 -c "import json; data = json.load(open('${OUTPUT_BASE_DIR}/dolly_top${K}.json')); print(f'  样本数: {len(data)}')" 2>/dev/null || echo "  无法读取统计信息"
    echo ""
fi

echo "✅ 全部完成！"
echo ""
echo "方法特点:"
echo "  - 加权策略: score = 1.0×cos(G,harmful) - 1.0×cos(G,safe)"
echo "  - Interaction-Aware: 每次选择考虑已选样本的联合效果"
echo "  - 层级选择: 使用最后 $NUM_LAYERS 层的梯度"
echo "  - 贪心算法: 迭代选择使目标函数最大化的样本"
echo ""
echo "下一步:"
echo "  使用选择的数据进行微调训练:"
echo "  python3 train.py --data_path ${OUTPUT_BASE_DIR}/dolly_top${K}.json"
echo ""