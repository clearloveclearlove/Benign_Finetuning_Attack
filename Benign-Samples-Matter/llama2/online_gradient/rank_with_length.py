"""
基于 Self-Influence 和答案长度的组合评分筛选样本
Self-Influence_with_length: Score(z) = log(Self-Inf(z) + 1) + log(len(a) + 1)
其中 len(a) 表示答案的 token 长度
"""

import torch
import json
import os
import sys
import numpy as np
from transformers import LlamaTokenizer
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Select samples based on Self-Influence_with_length')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset file')
    parser.add_argument('--scores_file', type=str, required=True, help='Path to scores.pt file')
    parser.add_argument('--model_name', type=str, required=True, help='Model name for tokenizer')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--k', type=int, required=True, help='Number of top samples to select')
    parser.add_argument('--dataset_name', type=str, default='dolly', choices=['dolly', 'alpaca', 'gsm8k'],
                        help='Dataset name for field mapping')
    parser.add_argument('--length_weight', type=float, default=1.0,
                        help='Weight for length term (default: 1.0)')
    parser.add_argument('--influence_weight', type=float, default=1.0,
                        help='Weight for self-influence term (default: 1.0)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    return parser.parse_args()


def compute_combined_score(args):
    """
    计算组合分数: Score(z) = α * log(Self-Inf(z) + 1) + β * log(len(a) + 1)
    其中 α 是 influence_weight, β 是 length_weight, len(a) 是答案的 token 长度
    """

    # 设置随机种子
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 加载数据集
    print(f"正在加载数据集: {args.data_path}")
    with open(args.data_path, "r") as f:
        if args.data_path.endswith(".json"):
            data = json.load(f)
        elif args.data_path.endswith(".jsonl"):
            data = []
            for line in f:
                data.append(json.loads(line))

    print(f"✓ 加载了 {len(data)} 个样本")

    # 加载 Self-Influence 分数
    print(f"正在加载 Self-Influence 分数: {args.scores_file}")
    self_influence_scores = torch.load(args.scores_file)

    # 确保长度匹配
    if len(self_influence_scores) != len(data):
        print(f"⚠️  分数长度 ({len(self_influence_scores)}) 与数据长度 ({len(data)}) 不匹配")
        min_len = min(len(self_influence_scores), len(data))
        self_influence_scores = self_influence_scores[:min_len]
        data = data[:min_len]
        print(f"✓ 截断至 {min_len} 个样本")

    # 加载 tokenizer
    print(f"正在加载 tokenizer: {args.model_name}")
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name)

    # 字段名映射
    output_field_map = {
        "alpaca": "output",
        "dolly": "response",
        "gsm8k": "answer"
    }
    output_field = output_field_map[args.dataset_name]

    # 计算每个样本答案的 token 长度
    print("正在计算答案的 token 长度...")
    answer_lengths = []
    for i, item in enumerate(data):
        if i % 1000 == 0:
            print(f"  处理进度: {i}/{len(data)}")

        answer_text = item.get(output_field, "")
        # 使用 tokenizer 计算 token 数量（去掉首尾的特殊 token）
        tokens = tokenizer(answer_text)['input_ids']
        # 去掉 BOS 和 EOS token（如果存在）
        token_count = len(tokens[1:-1]) if len(tokens) > 2 else len(tokens)
        answer_lengths.append(token_count)

    print(f"  处理进度: {len(data)}/{len(data)} ✓")
    answer_lengths = torch.tensor(answer_lengths, dtype=torch.float32)

    # 计算组合分数
    # Score(z) = α * log(Self-Inf(z) + 1) + β * log(len(a) + 1)
    print("\n正在计算 Self-Influence_with_length 分数...")
    print(f"  Self-Influence 权重 α: {args.influence_weight}")
    print(f"  Token 长度权重 β: {args.length_weight}")

    # 为了数值稳定性，对 self_influence_scores 进行归一化
    # 只对有效分数（非零）进行处理
    valid_mask = self_influence_scores != 0

    # 计算 log(Self-Inf(z) + 1)
    log_influence = torch.log(torch.abs(self_influence_scores) + 1) * torch.sign(self_influence_scores)
    log_influence[~valid_mask] = 0

    # 计算 log(len(a) + 1)
    log_length = torch.log(answer_lengths + 1)

    # 组合分数
    combined_scores = args.influence_weight * log_influence + args.length_weight * log_length

    # 对无效样本设置极小值
    combined_scores[~valid_mask] = float('-inf')

    # 统计信息
    print(f"\n【分数统计】")
    print(
        f"  Self-Influence 范围: [{self_influence_scores[valid_mask].min():.4f}, {self_influence_scores[valid_mask].max():.4f}]")
    print(f"  答案 Token 长度范围: [{answer_lengths.min():.0f}, {answer_lengths.max():.0f}] tokens")
    print(f"  Log(Self-Inf + 1) 范围: [{log_influence[valid_mask].min():.4f}, {log_influence[valid_mask].max():.4f}]")
    print(f"  Log(Token Length + 1) 范围: [{log_length.min():.4f}, {log_length.max():.4f}]")
    print(
        f"  Self-Influence_with_length 分数范围: [{combined_scores[valid_mask].min():.4f}, {combined_scores[valid_mask].max():.4f}]")
    print(f"  无效样本数: {(~valid_mask).sum().item()}")

    return combined_scores, data, output_field, valid_mask, answer_lengths


def select_top_k(combined_scores, data, output_field, valid_mask, answer_lengths, args):
    """
    选择 Top-K 样本
    """

    # 选择 Top-K（过采样以便过滤）
    k_oversample = int(args.k * 10)
    topk_scores, topk_indices = torch.topk(combined_scores, min(k_oversample, valid_mask.sum().item()))

    print(f"\n【样本选择】")
    print(f"  目标样本数 K: {args.k}")
    print(f"  初始选择: {len(topk_indices)} 个样本")

    # 过滤无效答案
    invalid_answers = ["No", "no", "", " ", "yes", "Yes"]
    selected_data = []
    selected_indices = []

    for idx in topk_indices:
        idx = idx.item()
        answer = data[idx].get(output_field, "")

        # 过滤条件：答案不在无效列表中
        if answer not in invalid_answers and answer.strip() != "":
            selected_data.append(data[idx])
            selected_indices.append(idx)

            if len(selected_data) >= args.k:
                break

    print(f"  过滤后样本数: {len(selected_data)}")

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)

    output_file = os.path.join(args.output_dir, f"dolly_top{args.k}.json")
    index_file = os.path.join(args.output_dir, f"dolly_top{args.k}_index.json")

    with open(output_file, 'w') as f:
        json.dump(selected_data, f, indent=2, ensure_ascii=False)

    with open(index_file, 'w') as f:
        json.dump(selected_indices, f, indent=2)

    print(f"\n✓ 已保存选择的样本到: {output_file}")
    print(f"✓ 已保存样本索引到: {index_file}")

    # 打印一些统计信息
    selected_scores = [combined_scores[idx].item() for idx in selected_indices]
    selected_lengths = [answer_lengths[idx].item() for idx in selected_indices]

    print(f"\n【选中样本统计】")
    print(f"  Self-Influence_with_length 分数范围: [{min(selected_scores):.4f}, {max(selected_scores):.4f}]")
    print(f"  答案 Token 长度范围: [{int(min(selected_lengths))}, {int(max(selected_lengths))}] tokens")
    print(f"  平均答案 Token 长度: {np.mean(selected_lengths):.1f} tokens")

    return selected_data, selected_indices


def main():
    args = parse_args()

    print("=" * 60)
    print("Self-Influence_with_length 样本筛选")
    print("=" * 60)
    print(f"公式: Score(z) = {args.influence_weight} * log(Self-Inf(z) + 1) + {args.length_weight} * log(len(a) + 1)")
    print(f"其中 len(a) 表示答案的 token 长度")
    print("=" * 60)
    print()

    # 计算组合分数
    combined_scores, data, output_field, valid_mask, answer_lengths = compute_combined_score(args)

    # 选择 Top-K
    selected_data, selected_indices = select_top_k(combined_scores, data, output_field, valid_mask, answer_lengths,
                                                   args)

    print("\n" + "=" * 60)
    print("✅ 完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()