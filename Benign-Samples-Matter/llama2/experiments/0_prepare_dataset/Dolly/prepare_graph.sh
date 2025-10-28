#!/bin/bash

# 切换到项目根目录
cd /root/code/Benign_Finetuning_Attack/Benign-Samples-Matter/llama2

# 设置 PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/root/code/Benign_Finetuning_Attack/Benign-Samples-Matter/llama2"

# 配置参数
output_dir="experiments/graph_attack/Dolly"
data_path="ft_datasets/dolly_dataset/databricks-dolly-15k-no-safety.jsonl"
k=100
num_candidates=1000
proj_dim=8192

mkdir -p $output_dir

# 使用 python -m 运行（从项目根目录）
python -m online_gradient.graph_rank_projection \
  --model_name /root/autodl-tmp/models/Llama-2-7B-Chat-fp16 \
  --dataset dolly_dataset \
  --data_path $data_path \
  --k $k \
  --num_candidates $num_candidates \
  --proj_dim $proj_dim \
  --normalize_grads True \
  --output_dir $output_dir \
  --dataset_name dolly_graph 2>&1 | tee $output_dir/log.txt

echo "Graph-based selection completed!"