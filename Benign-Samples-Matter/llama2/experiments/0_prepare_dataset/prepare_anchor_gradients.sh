#!/bin/bash

# ==============================================================================
# 脚本功能:
#   为 harmful, safe1, safe2 三个 anchor 数据集生成梯度。
#
# 使用方法:
#   ./this_script_name.sh [true|false]
#
# 参数说明:
#   $1: 是否对梯度进行归一化 (normalize)。
#       - "true":  进行归一化，保存为 normalized_average_...pt
#       - "false": 不进行归一化，保存为 unnormalized_average_...pt
#
# 示例:
#   # 生成归一化的梯度
#   bash ./this_script_name.sh true
#
#   # 生成未归一化的梯度
#   bash ./this_script_name.sh false
# ==============================================================================

# -----------------
# 1. 参数检查与设置
# -----------------

# 检查用户是否提供了参数
if [ -z "$1" ]; then
    echo "错误: 请提供一个参数来控制是否进行归一化 (normalize)。"
    echo "用法: $0 [true|false]"
    exit 1
fi

# 将输入转换为小写，以兼容 "True", "TRUE", "true" 等
NORMALIZE_INPUT=$(echo "$1" | tr '[:upper:]' '[:lower:]')

# 根据输入设置Python脚本参数和输出文件名前缀
if [ "$NORMALIZE_INPUT" == "true" ]; then
    NORMALIZE_FLAG="True"
    FILENAME_PREFIX="normalized"
elif [ "$NORMALIZE_INPUT" == "false" ]; then
    NORMALIZE_FLAG="False"
    FILENAME_PREFIX="unnormalized"
else
    echo "错误: 无效的参数 '$1'。请使用 'true' 或 'false'。"
    echo "用法: $0 [true|false]"
    exit 1
fi

# -----------------
# 2. 全局配置
# -----------------
MODEL_PATH="/home1/yibiao/PTM/Llama-2-7b-chat-hf"
MLENS=10  # 只取前10个token的梯度
NUM_SAMPLES=10  # 每个anchor数据集的样本数

# 梯度输出根目录
GRADIENT_BASE_DIR="/mnt/newdisk/yibiao/gradient"

# 数据集路径
HARMFUL_ANCHOR="ft_datasets/pure_bad_dataset/pure-bad-illegal-activities-selected10.jsonl"
SAFE_ANCHOR1="ft_datasets/pure_bad_dataset/pure-bad-illegal-acticities-selected-10-anchor1.jsonl"
SAFE_ANCHOR2="ft_datasets/pure_bad_dataset/pure-bad-illegal-activities-selected-10-anchor2.jsonl"

# 输出目录路径
HARMFUL_OUTPUT="${GRADIENT_BASE_DIR}/Llama-2-7B-Chat/harmful_anchor_gradients"
SAFE1_OUTPUT="${GRADIENT_BASE_DIR}/Llama-2-7B-Chat/safe_anchor1_gradients"
SAFE2_OUTPUT="${GRADIENT_BASE_DIR}/Llama-2-7B-Chat/safe_anchor2_gradients"

# ---------------------------------
# 3. 定义梯度生成函数 (避免代码重复)
# ---------------------------------
generate_gradients() {
  local data_path="$1"
  local output_dir="$2"
  local description="$3"

  echo ""
  echo "--> Generating $description gradients (online mean, normalize: $NORMALIZE_FLAG)..."

  mkdir -p "$output_dir"

  python3 get_gradients.py \
    --model_name "$MODEL_PATH" \
    --dataset pure_bad_dataset \
    --data_path "$data_path" \
    --run_validation False \
    --use_lora False \
    --save_full_gradients True \
    --online_mean True \
    --normalize "$NORMALIZE_FLAG" \
    --grads_output_dir "$output_dir" \
    --batch_size_training 1 \
    --max_response_length "$MLENS"

  # 根据 normalize 状态重命名输出文件
  local final_filename="${FILENAME_PREFIX}_average_num${NUM_SAMPLES}.pt"
  if [ -f "$output_dir/mean_gradients.pt" ]; then
      mv "$output_dir/mean_gradients.pt" "$output_dir/$final_filename"
      echo "✓ $description gradients saved to: $output_dir/$final_filename"
  else
      echo "✗ 警告: 未找到生成的梯度文件 mean_gradients.pt 在 $output_dir"
  fi
}

# -------------------
# 4. 主执行流程
# -------------------
echo "======================================================"
echo "开始生成梯度... (模式: $FILENAME_PREFIX)"
echo "使用在线计算平均梯度以节省磁盘空间"
echo "======================================================"

# 1. 生成 harmful anchor 的平均梯度
generate_gradients "$HARMFUL_ANCHOR" "$HARMFUL_OUTPUT" "harmful anchor"

# 2. 生成 safe anchor 1 的平均梯度
generate_gradients "$SAFE_ANCHOR1" "$SAFE1_OUTPUT" "safe anchor 1"

# 3. 生成 safe anchor 2 的平均梯度
generate_gradients "$SAFE_ANCHOR2" "$SAFE2_OUTPUT" "safe anchor 2"


# -------------------
# 5. 总结
# -------------------
echo ""
echo "=========================================="
echo "所有 anchor 梯度已成功生成！"
echo "=========================================="
echo ""
echo "生成的文件:"
final_filename="${FILENAME_PREFIX}_average_num${NUM_SAMPLES}.pt"
echo "  - $HARMFUL_OUTPUT/$final_filename"
echo "  - $SAFE1_OUTPUT/$final_filename"
echo "  - $SAFE2_OUTPUT/$final_filename"
echo ""
