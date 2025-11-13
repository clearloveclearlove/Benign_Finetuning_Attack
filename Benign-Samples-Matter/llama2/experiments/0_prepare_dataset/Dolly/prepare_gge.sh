#!/bin/bash

# ====================================================================
# Online IAGS 数据准备脚本 - 支持层级选择
# Path: experiments/0_prepare_dataset/Dolly/prepare_online_iags.sh
# Date: 2025-11-06
# ====================================================================

GPU_ID=${1:-0}
METHOD=${2:-gge}  # gge or gce
K=${3:-100}
NUM_LAYERS=${4:-8}  # 新增：使用最后 N 层，默认 8

export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "=========================================="
echo "Online IAGS - Method: ${METHOD^^}"
echo "GPU: $GPU_ID, K: $K, Layers: $NUM_LAYERS"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""

# ====================================================================
# 配置参数
# ====================================================================

MODEL_PATH="/home1/yibiao/PTM/Llama-2-7b-chat-hf"
GRADIENT_BASE_DIR="/mnt/newdisk/yibiao/gradient/Llama-2-7B-Chat"

DATASET_NAME="dolly_dataset"
DATA_PATH="ft_datasets/dolly_dataset/self_influence_with_length/dolly_top1000.json"

MLENS=10
OUTPUT_BASE_DIR="ft_datasets/dolly_dataset/Online-IAGS-${METHOD^^}-L${NUM_LAYERS}"

mkdir -p $OUTPUT_BASE_DIR

# ====================================================================
# 运行 Online IAGS
# ====================================================================

if [ "$METHOD" == "gge" ]; then
    echo "Running GGE (Greedy Gradient Ensemble)..."
    echo "Configuration:"
    echo "  - Method: GGE"
    echo "  - Normalize: False"
    echo "  - Layers: Last $NUM_LAYERS layers"
    echo "  - Max response length: $MLENS tokens"
    echo ""

    python3 -m online_gradient.run_online_iags \
        --method gge \
        --model_name $MODEL_PATH \
        --dataset $DATASET_NAME \
        --data_path $DATA_PATH \
        --k $K \
        --normalize False \
        --num_layers $NUM_LAYERS \
        --output_dir $OUTPUT_BASE_DIR \
        --dataset_name dolly

elif [ "$METHOD" == "gce" ]; then
    echo "Running GCE (Greedy Cosine Ensemble)..."
    HARMFUL_GRAD="${GRADIENT_BASE_DIR}/harmful_anchor_gradients/normalized_average_num10.pt"

    if [ ! -f "$HARMFUL_GRAD" ]; then
        echo "❌ Harmful gradient not found: $HARMFUL_GRAD"
        exit 1
    fi

    echo "Configuration:"
    echo "  - Method: GCE"
    echo "  - Normalize: True"
    echo "  - Layers: Last $NUM_LAYERS layers"
    echo "  - Max response length: $MLENS tokens"
    echo "  - Harmful gradient: $HARMFUL_GRAD"
    echo ""

    python3 -m online_gradient.run_online_iags \
        --method gce \
        --model_name $MODEL_PATH \
        --harmful_grad_file $HARMFUL_GRAD \
        --dataset $DATASET_NAME \
        --data_path $DATA_PATH \
        --k $K \
        --normalize True \
        --num_layers $NUM_LAYERS \
        --output_dir $OUTPUT_BASE_DIR \
        --dataset_name dolly
else
    echo "❌ Unknown method: $METHOD"
    echo "Available methods: gge, gce"
    exit 1
fi

echo ""
echo "✅ Online IAGS ${METHOD^^} completed!"
echo "Output directory: $OUTPUT_BASE_DIR"