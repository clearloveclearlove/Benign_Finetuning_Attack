#!/bin/bash

# ====================================================================
# Online IAGS with Length - 考虑答案长度的在线样本选择
# Path: experiments/0_prepare_dataset/Dolly/prepare_gge_with_length.sh
# Date: 2025-01-11
# ====================================================================

GPU_ID=${1:-0}
K=${2:-100}
LENGTH_WEIGHT=${3:-1.0}  # β 权重，默认 1.0
NUM_LAYERS=${4:-All}       # 使用最后 N 层，默认 8

export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "=========================================="
echo "Online IAGS with Length"
echo "GPU: $GPU_ID, K: $K, β: $LENGTH_WEIGHT, Layers: $NUM_LAYERS"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""

# ====================================================================
# 配置参数
# ====================================================================

MODEL_PATH="/home1/yibiao/PTM/Llama-2-7b-chat-hf"

DATASET_NAME="dolly_dataset"
DATA_PATH="ft_datasets/dolly_dataset/self_influence_with_length/dolly_top1000.json"

MLENS=10
OUTPUT_BASE_DIR="ft_datasets/dolly_dataset/Online-IAGS-Length-L${NUM_LAYERS}-Beta${LENGTH_WEIGHT}"

mkdir -p $OUTPUT_BASE_DIR

# ====================================================================
# 运行 Online IAGS with Length
# ====================================================================

echo "Running Online IAGS with Answer Length..."
echo "Configuration:"
echo "  - Method: GradientNormWithLength"
echo "  - Length weight β: $LENGTH_WEIGHT"
echo "  - Normalize: False"
echo "  - Layers: Last $NUM_LAYERS layers"
echo "  - Max response length: $MLENS tokens"
echo "  - Formula: Score = log(||G||^2 + 1) + β * log(len(a) + 1)"
echo ""

python3 -m online_gradient.run_online_iags_with_length \
    --model_name $MODEL_PATH \
    --dataset $DATASET_NAME \
    --data_path $DATA_PATH \
    --dataset_name dolly \
    --k $K \
    --length_weight $LENGTH_WEIGHT \
    --normalize False \
    --max_response_length $MLENS \
    --output_dir $OUTPUT_BASE_DIR

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Online IAGS with Length completed!"
    echo "Output directory: $OUTPUT_BASE_DIR"
    echo ""
    echo "Generated files:"
    echo "  - ${OUTPUT_BASE_DIR}/dolly_top${K}.json"
    echo "  - ${OUTPUT_BASE_DIR}/dolly_top${K}_index.json"
    echo "  - ${OUTPUT_BASE_DIR}/dolly_top${K}_gains.json"
    echo "  - ${OUTPUT_BASE_DIR}/dolly_top${K}_stats.json"
else
    echo ""
    echo "❌ Selection failed! Check the error messages above."
    exit 1
fi