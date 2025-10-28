#!/bin/bash

# 配置参数
MODEL_PATH="/root/autodl-tmp/models/Llama-2-7B-Chat-fp16"
MLENS=10  # 只取前10个token的梯度
NUM_SAMPLES=10  # 每个anchor数据集的样本数

# 1. 生成 harmful anchor 的梯度
echo "Generating harmful anchor gradients..."
HARMFUL_ANCHOR="ft_datasets/pure_bad_dataset/pure-bad-illegal-activities-selected10.jsonl"
HARMFUL_OUTPUT="/root/autodl-tmp/gradient/Llama-2-7B-Chat-fp16/harmful_anchor_gradients"

mkdir -p $HARMFUL_OUTPUT

python3 get_gradients.py \
  --model_name $MODEL_PATH \
  --dataset pure_bad_dataset \
  --data_path $HARMFUL_ANCHOR \
  --run_validation False \
  --use_lora False \
  --save_full_gradients True \
  --grads_output_dir $HARMFUL_OUTPUT \
  --batch_size_training 1 \
  --max_response_length $MLENS

# 计算 harmful anchor 的平均梯度
python3 average_gradient.py \
  --input_dir $HARMFUL_OUTPUT \
  --output_file $HARMFUL_OUTPUT/normalized_average_num${NUM_SAMPLES}.pt \
  --normalize \
  --num_samples $NUM_SAMPLES \
  --sample_sequentially


# 2. 生成 safe anchor 1 的梯度
echo "Generating safe anchor 1 gradients..."
SAFE_ANCHOR1="ft_datasets/pure_bad_dataset/pure-bad-illegal-acticities-selected-10-anchor1.jsonl"
SAFE1_OUTPUT="/root/autodl-tmp/gradient/Llama-2-7B-Chat-fp16/safe_anchor1_gradients"

mkdir -p $SAFE1_OUTPUT

python3 get_gradients.py \
  --model_name $MODEL_PATH \
  --dataset pure_bad_dataset \
  --data_path $SAFE_ANCHOR1 \
  --run_validation False \
  --use_lora False \
  --save_full_gradients True \
  --grads_output_dir $SAFE1_OUTPUT \
  --batch_size_training 1 \
  --max_response_length $MLENS

# 计算 safe anchor 1 的平均梯度
python3 average_gradient.py \
  --input_dir $SAFE1_OUTPUT \
  --output_file $SAFE1_OUTPUT/normalized_average_num${NUM_SAMPLES}.pt \
  --normalize \
  --num_samples $NUM_SAMPLES \
  --sample_sequentially


# 3. 生成 safe anchor 2 的梯度
echo "Generating safe anchor 2 gradients..."
SAFE_ANCHOR2="ft_datasets/pure_bad_dataset/pure-bad-illegal-activities-selected-10-anchor2.jsonl"
SAFE2_OUTPUT="/root/autodl-tmp/gradient/Llama-2-7B-Chat-fp16/safe_anchor2_gradients"

mkdir -p $SAFE2_OUTPUT

python3 get_gradients.py \
  --model_name $MODEL_PATH \
  --dataset pure_bad_dataset \
  --data_path $SAFE_ANCHOR2 \
  --run_validation False \
  --use_lora False \
  --save_full_gradients True \
  --grads_output_dir $SAFE2_OUTPUT \
  --batch_size_training 1 \
  --max_response_length $MLENS

# 计算 safe anchor 2 的平均梯度
python3 average_gradient.py \
  --input_dir $SAFE2_OUTPUT \
  --output_file $SAFE2_OUTPUT/normalized_average_num${NUM_SAMPLES}.pt \
  --normalize \
  --num_samples $NUM_SAMPLES \
  --sample_sequentially

echo "All anchor gradients generated successfully!"
