"""
Online IAGS with Length - Sample Selection Script
考虑答案长度的在线样本选择
Date: 2025-01-11
"""

import argparse
import os
import json
import torch
from pathlib import Path
from datetime import datetime
from transformers import LlamaTokenizer

from online_gradient.online_iags_framework import (
    OnlineIAGSSelector,
    GradientNormWithLengthScore,
    load_model_and_data
)


def compute_answer_lengths(data_path: str, tokenizer: LlamaTokenizer, dataset_name: str = 'dolly'):
    """
    计算数据集中所有答案的 token 长度
    """
    output_field_map = {
        "alpaca": "output",
        "dolly": "response",
        "gsm8k": "answer"
    }
    output_field = output_field_map.get(dataset_name, "response")

    print(f"正在计算答案的 token 长度...")

    # 加载原始数据
    with open(data_path, 'r') as f:
        if data_path.endswith('.jsonl'):
            data = [json.loads(line) for line in f]
        else:
            data = json.load(f)

    answer_lengths = []
    for i, item in enumerate(data):
        if i % 1000 == 0:
            print(f"  处理进度: {i}/{len(data)}")

        answer_text = item.get(output_field, "")
        tokens = tokenizer(answer_text)['input_ids']
        # 去掉 BOS 和 EOS tokens
        token_count = len(tokens[1:-1]) if len(tokens) > 2 else len(tokens)
        answer_lengths.append(token_count)

    print(f"  处理进度: {len(data)}/{len(data)} ✓")

    lengths_tensor = torch.tensor(answer_lengths, dtype=torch.long)

    print(f"✓ 答案长度统计:")
    print(f"  - 范围: [{lengths_tensor.min()}, {lengths_tensor.max()}] tokens")
    print(f"  - 平均: {lengths_tensor.float().mean():.1f} tokens")
    print(f"  - 中位数: {lengths_tensor.float().median():.1f} tokens\n")

    return lengths_tensor


def main():
    parser = argparse.ArgumentParser(
        description='Online IAGS with Answer Length Consideration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用最后 8 层，长度权重 2.0
  python run_online_iags_with_length.py --model_name /path/to/model \\
    --dataset dolly_dataset --data_path data.jsonl --k 100 \\
    --num_layers 8 --length_weight 2.0
        """
    )

    # 模型和数据
    parser.add_argument('--model_name', type=str, required=True,
                        help='预训练模型路径')
    parser.add_argument('--dataset', type=str, required=True,
                        help='数据集名称')
    parser.add_argument('--data_path', type=str, required=True,
                        help='数据集文件路径')
    parser.add_argument('--dataset_name', type=str, default='dolly',
                        choices=['dolly', 'alpaca', 'gsm8k'],
                        help='数据集类型（用于字段映射）')

    # 选择参数
    parser.add_argument('--k', type=int, required=True,
                        help='选择的样本数量')
    parser.add_argument('--length_weight', type=float, default=1.0,
                        help='长度权重 β (默认: 1.0)')
    parser.add_argument('--max_response_length', type=int, default=None,
                        help='梯度计算的最大响应长度')
    parser.add_argument('--normalize', type=str, default='False',
                        help='是否归一化梯度 (True/False)')
    parser.add_argument('--num_layers', type=int, default=None,
                        help='仅使用最后 N 个 transformer 层 (None=全部层)')

    # 训练配置
    parser.add_argument('--batch_size_training', type=int, default=1,
                        help='批量大小（在线计算必须为1）')

    # 输出
    parser.add_argument('--output_dir', type=str, required=True,
                        help='选择样本的输出目录')

    # 其他
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')

    args = parser.parse_args()

    # 解析 normalize
    normalize = args.normalize.lower() in ['true', '1', 'yes']

    # 验证
    if args.batch_size_training != 1:
        print("⚠️  警告: batch_size_training 必须为 1，已自动设置为 1")
        args.batch_size_training = 1

    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 打印头部信息
    print("=" * 70)
    print(f"Online IAGS with Answer Length")
    print(f"Method: GradientNormWithLengthScore")
    print(f"User: {os.environ.get('USER', 'unknown')}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()

    # 加载模型和数据
    print("加载模型和数据集...")
    config = {
        'model_name': args.model_name,
        'dataset': args.dataset,
        'data_path': args.data_path,
        'batch_size_training': args.batch_size_training,
    }
    model, train_dataloader = load_model_and_data(config)
    print(f"✓ 加载模型: {args.model_name}")
    print(f"✓ 加载数据集: {len(train_dataloader.dataset)} 样本")
    print()

    # 计算答案长度
    print("计算答案长度...")
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name)
    answer_lengths = compute_answer_lengths(args.data_path, tokenizer, args.dataset_name)

    # 初始化评分函数
    print("初始化评分函数: GradientNormWithLengthScore")
    print(f"公式: Score(z) = log(||G||^2 + 1) + {args.length_weight} * log(len(a) + 1)")
    print()

    score_function = GradientNormWithLengthScore(
        answer_lengths=answer_lengths,
        length_weight=args.length_weight
    )

    # 初始化选择器
    selector = OnlineIAGSSelector(
        model=model,
        train_dataloader=train_dataloader,
        score_function=score_function,
        max_response_length=args.max_response_length,
        normalize=normalize,
        device='cuda',
        num_layers=args.num_layers
    )

    # 运行选择
    selected_indices, marginal_gains, stats = selector.select_samples(
        k=args.k,
        verbose=True
    )

    # 保存结果
    print("保存结果...")

    # 保存选择的索引
    indices_file = output_dir / f"{args.dataset_name}_top{args.k}_index.json"
    with open(indices_file, 'w') as f:
        json.dump(selected_indices, f, indent=2)
    print(f"✓ 保存索引到: {indices_file}")

    # 保存边际增益
    gains_file = output_dir / f"{args.dataset_name}_top{args.k}_gains.json"
    with open(gains_file, 'w') as f:
        json.dump({
            'marginal_gains': marginal_gains,
            'num_gains': len(marginal_gains),
        }, f, indent=2)
    print(f"✓ 保存增益到: {gains_file}")

    # 保存统计信息
    stats_file = output_dir / f"{args.dataset_name}_top{args.k}_stats.json"
    stats.update({
        'method': 'GradientNormWithLength',
        'k': args.k,
        'length_weight': args.length_weight,
        'num_layers': args.num_layers,
        'normalize': normalize,
        'model_name': args.model_name,
        'dataset': args.dataset,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ 保存统计到: {stats_file}")

    # 提取选择的样本并保存
    print("\n提取选择的样本...")

    # 加载原始数据集
    with open(args.data_path, 'r') as f:
        if args.data_path.endswith('.jsonl'):
            original_data = [json.loads(line) for line in f]
        else:
            original_data = json.load(f)

    # 提取选择的样本
    selected_samples = [original_data[idx] for idx in selected_indices]

    # 保存选择的样本
    selected_file = output_dir / f"{args.dataset_name}_top{args.k}.json"
    with open(selected_file, 'w') as f:
        json.dump(selected_samples, f, indent=2, ensure_ascii=False)
    print(f"✓ 保存选择的样本到: {selected_file}")

    # 打印选中样本的长度统计
    selected_lengths = [answer_lengths[idx].item() for idx in selected_indices]
    print(f"\n【选中样本长度统计】")
    print(f"  - 范围: [{min(selected_lengths)}, {max(selected_lengths)}] tokens")
    print(f"  - 平均: {sum(selected_lengths) / len(selected_lengths):.1f} tokens")

    print()
    print("=" * 70)
    print("✅ Online IAGS with Length 选择完成！")
    print("=" * 70)
    print(f"输出目录: {output_dir}")
    print(f"已选择 {len(selected_indices)} 个样本")
    print()


if __name__ == '__main__':
    main()