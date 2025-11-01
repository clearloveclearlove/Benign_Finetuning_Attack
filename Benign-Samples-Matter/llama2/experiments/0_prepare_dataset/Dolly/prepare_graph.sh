#!/bin/bash

MODEL_NAME="/root/autodl-tmp/models/Llama-2-7B-Chat-fp16"
DATA_PATH="/root/code/Benign_Finetuning_Attack/Benign-Samples-Matter/llama2/ft_datasets/dolly_dataset/databricks-dolly-15k-no-safety.jsonl"
#DATA_PATH="/root/code/Benign_Finetuning_Attack/Benign-Samples-Matter/llama2/ft_datasets/dolly_dataset/dolly_subset_1000.jsonl"
DATASET_CONFIG="dolly_dataset"
K_SAMPLES=100
M_SURVIVORS=1000  # 添加预筛选候选数
OUTPUT_FILE="ft_datasets/dolly_dataset/Graph/dolly_top100.json"

# 设置 PYTHONPATH 为 llama2 目录（关键修复！）
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# --- 执行 ---
python online_gradient/graph_rank_online.py \
    --model_name "$MODEL_NAME" \
    --dataset "$DATASET_CONFIG" \
    --data_path "$DATA_PATH" \
    --k $K_SAMPLES \
    --use_prefiltering=True \
    --m_survivors $M_SURVIVORS \
    --use_full_grad_for_greedy=False \
    --num_last_layers_for_greedy=6 \
    --output_file "$OUTPUT_FILE"

echo "任务完成，结果已保存至 $OUTPUT_FILE"