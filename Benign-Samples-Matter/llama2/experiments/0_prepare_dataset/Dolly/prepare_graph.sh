#!/bin/bash

MODEL_NAME="/root/autodl-tmp/models/Llama-2-7B-Chat-fp16"
DATA_PATH="ft_datasets/dolly_dataset/dolly_subset_1000.jsonl"
DATASET_CONFIG="dolly_dataset"
K_SAMPLES=100
OUTPUT_FILE="experiments/0_prepare_dataset/Dolly/Graph/dolly_top100.json"

# --- 执行 ---
python online_gradient/graph_rank_online.py \
    --model_name "$MODEL_NAME" \
    --dataset "$DATASET_CONFIG" \
    --data_path "$DATA_PATH" \
    --k $K_SAMPLES \
    --use_prefiltering=False \
    --use_full_grad_for_greedy=True \
    --output_file "$OUTPUT_FILE"

echo "任务完成，结果已保存至 $OUTPUT_FILE"