#!/bin/bash

# 验证梯度一致性实验

MODEL_NAME="/root/autodl-tmp/models/Llama-2-7B-Chat-fp16"
DATA_PATH="ft_datasets/dolly_dataset/dolly_subset_1000.jsonl"
DATASET_CONFIG="dolly_dataset"

# 设置 PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "开始梯度一致性验证实验..."
echo "================================"

# 测试不同的层数配置
for NUM_LAYERS in 1 2 3 4 5 6; do
    echo ""
    echo "测试最后 ${NUM_LAYERS} 层..."
    echo "================================"

    python online_gradient/verify_gradient_consistency.py \
        --model_name "$MODEL_NAME" \
        --dataset "$DATASET_CONFIG" \
        --data_path "$DATA_PATH" \
        --num_samples 100 \
        --num_last_layers $NUM_LAYERS \
        --seed 42 \
        --output_file "experiments/0_prepare_dataset/Dolly/verification_last${NUM_LAYERS}_layers.json"

    echo ""
    echo "================================"
done

echo ""
echo "所有验证实验完成！"