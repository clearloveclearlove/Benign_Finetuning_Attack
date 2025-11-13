#!/bin/bash

# ====================================================================
# Self-Influence_with_length 样本筛选
# Score(z) = log(Self-Inf(z) + 1) + log(len(a) + 1)
# 其中 len(a) 表示答案的 token 长度
# ====================================================================

# 检查参数
if [ "$#" -lt 2 ]; then
    echo "用法: bash $0 <GPU_ID> <K值> [influence_weight] [length_weight]"
    echo "示例: bash $0 0 100"
    echo "示例: bash $0 0 100 1.0 1.0"
    exit 1
fi

GPU_ID=$1
K=$2
INFLUENCE_WEIGHT=${3:-1.0}  # 默认权重 1.0
LENGTH_WEIGHT=${4:-1.0}     # 默认权重 1.0

export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "=========================================="
echo "使用 GPU: $GPU_ID"
echo "选择样本数 K: $K"
echo "=========================================="
echo ""

echo "=========================================="
echo "Self-Influence_with_length 样本筛选"
echo "=========================================="
echo "Self-Influence 权重: $INFLUENCE_WEIGHT"
echo "Token 长度权重: $LENGTH_WEIGHT"
echo "=========================================="
echo ""

# 配置
MODEL_PATH="/home1/yibiao/PTM/Llama-2-7b-chat-hf"
DATASET_NAME="dolly_dataset"
DATA_PATH="ft_datasets/dolly_dataset/databricks-dolly-15k-no-safety.jsonl"
SCORES_FILE="experiments/0_prepare_dataset/Dolly/Self-Influence/scores.pt"
OUTPUT_DIR="ft_datasets/dolly_dataset/self_influence_with_length"
OUTPUT_BASE_DIR="experiments/0_prepare_dataset/Dolly/Self-Influence_with_length"

# 创建输出目录
mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_BASE_DIR

# 检查必需文件
if [ ! -f "$DATA_PATH" ]; then
    echo "❌ 错误: 数据文件不存在: $DATA_PATH"
    exit 1
fi

echo "✓ 数据文件: $DATA_PATH"
echo "✓ 样本数量: $(wc -l < $DATA_PATH)"
echo ""

if [ ! -f "$SCORES_FILE" ]; then
    echo "❌ 错误: Self-Influence 分数文件不存在: $SCORES_FILE"
    echo "请先运行: bash experiments/0_prepare_dataset/Dolly/prepare_icml25.sh 0 100"
    exit 1
fi

echo "✓ Self-Influence 分数文件: $SCORES_FILE"
echo ""

# 执行筛选
echo "=========================================="
echo "开始 Self-Influence_with_length 样本筛选..."
echo "=========================================="
echo ""

python3 online_gradient/rank_with_length.py \
    --dataset $DATASET_NAME \
    --data_path $DATA_PATH \
    --scores_file $SCORES_FILE \
    --model_name $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --k $K \
    --dataset_name dolly \
    --influence_weight $INFLUENCE_WEIGHT \
    --length_weight $LENGTH_WEIGHT \
    --seed 42 \
    2>&1 | tee $OUTPUT_BASE_DIR/log_top${K}.txt

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Self-Influence_with_length 筛选完成！"
    echo "=========================================="
    echo ""
    echo "生成的文件:"
    echo "  Self-Influence_with_length 选择结果:"
    echo "    - $OUTPUT_DIR/dolly_top${K}.json"
    echo "    - $OUTPUT_DIR/dolly_top${K}_index.json"
    echo ""
    echo "  日志文件:"
    echo "    - $OUTPUT_BASE_DIR/log_top${K}.txt"
    echo ""
    echo "已选择的数据样本数:"
    echo "  样本数: $K"
    echo ""
    echo "Self-Influence_with_length 方法说明:"
    echo "  - 基于 Self-Influence 分数"
    echo "  - 结合答案的 token 长度信息"
    echo "  - 公式: Score(z) = ${INFLUENCE_WEIGHT} * log(Self-Inf(z) + 1) + ${LENGTH_WEIGHT} * log(len(a) + 1)"
    echo "  - len(a) 表示答案的 token 数量（使用 LlamaTokenizer 计算）"
    echo ""
    echo "下一步:"
    echo "  使用选择的数据进行微调训练:"
    echo "  python3 train.py --data_path $OUTPUT_DIR/dolly_top${K}.json"
else
    echo ""
    echo "❌ 筛选失败，请检查错误信息"
    echo "日志文件: $OUTPUT_BASE_DIR/log_top${K}.txt"
    exit 1
fi