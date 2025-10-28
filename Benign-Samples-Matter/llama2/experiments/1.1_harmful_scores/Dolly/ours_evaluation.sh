#!/bin/bash

# 设置基础路径
BASE_MODEL_DIR="/root/autodl-tmp/models"
FINETUNED_MODEL_DIR="/root/autodl-tmp/finetuning_models"

# 确保输出目录存在
mkdir -p ${FINETUNED_MODEL_DIR}

for seed in 20; do

# 数据路径
# output_dir="ft_datasets/dolly_dataset/ours_selfunn/dolly_top100.json"
output_dir="ft_datasets/dolly_dataset/ours_normalized/dolly_top100.json"

# 模型标识（用于组织不同的实验）
experiment_name="dolly-7b-ours"
output_dir_model="1.1_harmful_scores/Dolly"

# 微调后模型保存路径
finetuned_model_path="${FINETUNED_MODEL_DIR}/${output_dir_model}/${experiment_name}/${seed}"

echo "=========================================="
echo "开始微调实验"
echo "=========================================="
echo "种子: ${seed}"
echo "训练数据: ${output_dir}"
echo "基础模型: ${BASE_MODEL_DIR}/Llama-2-7B-Chat-fp16"
echo "微调模型保存路径: ${finetuned_model_path}"
echo "=========================================="

## 步骤1: 使用FSDP进行微调
#echo ""
#echo "步骤1: 开始FSDP微调训练..."
#torchrun --nnodes 1 --master_port=25678 --nproc_per_node 1 finetuning.py \
#    --batch_size_training 20 --lr 5e-5 \
#    --gradient_accumulation_steps 1 --weight_decay 0 \
#    --num_epochs 5 \
#    --dataset dolly_dataset \
#    --enable_fsdp \
#    --data_path ${output_dir} \
#    --model_name "${BASE_MODEL_DIR}/Llama-2-7B-Chat-fp16" \
#    --pure_bf16 \
#    --dist_checkpoint_root_folder ${FINETUNED_MODEL_DIR}/ \
#    --dist_checkpoint_folder ${output_dir_model}/${experiment_name}/${seed} \
#    --save_every_epoch False \
#    --seed ${seed}
#
#if [ $? -ne 0 ]; then
#    echo "❌ 微调训练失败！"
#    exit 1
#fi
#
#echo "✓ 微调训练完成"
#
## FSDP checkpoint 的实际路径（FSDP会自动添加模型名称后缀）
#fsdp_checkpoint_path="${FINETUNED_MODEL_DIR}/${output_dir_model}/${experiment_name}/${seed}-${BASE_MODEL_DIR}/Llama-2-7B-Chat-fp16"
#
## 步骤2: 转换FSDP checkpoint为HuggingFace格式
#echo ""
#echo "步骤2: 转换FSDP checkpoint为HuggingFace格式..."
#echo "FSDP路径: ${fsdp_checkpoint_path}"
#echo "HF输出路径: ${finetuned_model_path}"
#
#python inference/checkpoint_converter_fsdp_hf.py \
#    -fsdp_checkpoint_path "${fsdp_checkpoint_path}" \
#    -consolidated_model_path "${finetuned_model_path}" \
#    -HF_model_path_or_name "${BASE_MODEL_DIR}/Llama-2-7B-Chat-fp16"
#
#if [ $? -ne 0 ]; then
#    echo "❌ Checkpoint转换失败！"
#    exit 1
#fi
#
#echo "✓ Checkpoint转换完成"
#
## 步骤3: 删除FSDP checkpoint以节省空间
#echo ""
#echo "步骤3: 清理FSDP checkpoint..."
#if [ -d "${fsdp_checkpoint_path}" ]; then
#    rm -rf "${fsdp_checkpoint_path}"
#    echo "✓ 已删除FSDP checkpoint: ${fsdp_checkpoint_path}"
#else
#    echo "⚠️  FSDP checkpoint路径不存在: ${fsdp_checkpoint_path}"
#fi
#


# 步骤5: 安全评估
echo ""
echo "=========================================="
echo "步骤5: 开始安全评估"
echo "=========================================="

 5.1 类别评估（已注释，如需要可取消注释）
echo ""
echo "5.1 运行类别评估 (11个类别)..."
for i in 1 3 4 5 6 7 8 9 10 11
do
    echo "  评估类别 ${i}/11..."
    python -u safety_evaluation/question_inference.py \
        --model_name "${finetuned_model_path}" \
        --prompt_file safety_evaluation/data/HEx-PHI/category_${i}.csv \
        --prompt_template_style dolly \
        --output_file safety_evaluation/question_output/HEx-PHI/category_${i}_dolly_7b-ours_${seed}.jsonl

    python safety_evaluation/wild_guard_eval.py \
        --input_file safety_evaluation/question_output/HEx-PHI/category_${i}_dolly_7b-ours_${seed}.jsonl
done

# 5.2 AdvBench评估
echo ""
echo "5.2 运行AdvBench评估..."
python -u safety_evaluation/question_inference.py \
    --model_name "${finetuned_model_path}" \
    --prompt_file safety_evaluation/data/harmful_behaviors.csv \
    --prompt_template_style dolly \
    --output_file safety_evaluation/question_output/AdvBench/AdvBench_dolly_7b-ours_${seed}.jsonl

if [ $? -ne 0 ]; then
    echo "❌ 推理失败！"
    exit 1
fi

python safety_evaluation/wild_guard_eval.py \
    --input_file safety_evaluation/question_output/AdvBench/AdvBench_dolly_7b-ours_${seed}.jsonl

if [ $? -ne 0 ]; then
    echo "❌ 评估失败！"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ 所有步骤完成！"
echo "=========================================="
echo "微调模型位置: ${finetuned_model_path}"
echo "评估结果: safety_evaluation/question_output/AdvBench/AdvBench_dolly_7b-ours_${seed}.jsonl"
echo "=========================================="

done

echo ""
echo "🎉 实验完成！"