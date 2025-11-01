"""
验证使用不同参数范围（最后N层 vs 全部参数）计算的梯度指标是否保持相同的排序关系。
内存优化版本：不保存所有梯度向量，而是分两遍计算。
"""

import json
import random
from typing import List, Tuple, Dict

import fire
import torch
import numpy as np
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, LlamaTokenizer,
                          default_data_collator)

from configs.training import train_config
from utils.config_utils import generate_dataset_config, update_config
from utils.dataset_utils import get_preprocessed_dataset


def spearman_correlation(x, y):
    """
    手动实现 Spearman 秩相关系数计算
    """
    n = len(x)

    # 计算排名
    rank_x = np.argsort(np.argsort(x))
    rank_y = np.argsort(np.argsort(y))

    # 计算排名差异
    d = rank_x - rank_y
    d_squared_sum = np.sum(d ** 2)

    # Spearman 相关系数公式
    rho = 1 - (6 * d_squared_sum) / (n * (n ** 2 - 1))

    return rho


def get_param_groups(
        model: torch.nn.Module,
        num_last_layers: int
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    获取两组参数：全参数 和 最后N层参数。
    """
    params_full = [p for p in model.parameters() if p.requires_grad]

    try:
        total_layers = len(model.model.layers)
        start_layer = max(0, total_layers - num_last_layers)

        target_layer_names = {f'model.layers.{i}.' for i in range(start_layer, total_layers)}
        target_layer_names.add('lm_head')

        params_partial = []
        for name, param in model.named_parameters():
            if param.requires_grad and any(name.startswith(layer_name) for layer_name in target_layer_names):
                params_partial.append(param)

        print(f"Full parameters: {len(params_full)} groups")
        print(f"Partial parameters (last {num_last_layers} layers + lm_head): {len(params_partial)} groups")

        num_params_full = sum(p.numel() for p in params_full)
        num_params_partial = sum(p.numel() for p in params_partial)
        print(f"Full param count: {num_params_full:,}")
        print(f"Partial param count: {num_params_partial:,}")
        print(f"Ratio: {num_params_partial / num_params_full:.2%}")

    except AttributeError as e:
        print(f"Warning: Could not detect Llama layers ({e}). Using all parameters.")
        params_partial = params_full

    return params_full, params_partial


def compute_gradients(
        model: torch.nn.Module,
        batch: Dict[str, torch.Tensor],
        params_full: List[torch.Tensor],
        params_partial: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    """
    计算并返回全参数和部分参数的梯度向量及范数。

    Returns:
        grad_full_flat: 全参数梯度展平向量
        grad_partial_flat: 部分参数梯度展平向量
        norm_full: 全参数梯度范数
        norm_partial: 部分参数梯度范数
    """
    model.zero_grad()
    loss = model(**batch).loss
    loss.backward()

    with torch.no_grad():
        # 全参数梯度
        grad_full_flat = torch.cat(
            [p.grad.view(-1) for p in params_full if p.grad is not None]
        )
        norm_full = torch.norm(grad_full_flat).item()

        # 部分参数梯度
        grad_partial_flat = torch.cat(
            [p.grad.view(-1) for p in params_partial if p.grad is not None]
        )
        norm_partial = torch.norm(grad_partial_flat).item()

    model.zero_grad()
    return grad_full_flat, grad_partial_flat, norm_full, norm_partial


def verify_gradient_consistency(
        num_samples: int = 100,
        num_last_layers: int = 4,
        num_base_samples: int = 10,
        seed: int = 42,
        output_file: str = "experiments/0_prepare_dataset/verification_results.json",
        **kwargs
):
    """
    验证使用不同参数范围计算的梯度指标是否保持排序一致性。
    内存优化版本。
    """
    print("=" * 80)
    print("梯度一致性验证实验 (内存优化版)")
    print("=" * 80)
    print(f"采样数量: {num_samples}")
    print(f"部分梯度使用: 最后 {num_last_layers} 层 + lm_head")
    print(f"基础样本数（用于构建累积梯度）: {num_base_samples}")
    print(f"随机种子: {seed}")
    print("=" * 80)

    # 初始化
    update_config((train_config,), **kwargs)
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")

    # 加载模型
    print("\n加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        train_config.model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": device}
    ).eval()
    tokenizer = LlamaTokenizer.from_pretrained(train_config.model_name, pad_token="<unk>")

    # 加载数据集
    print("加载数据集...")
    dataset_config = generate_dataset_config(train_config, kwargs)
    full_dataset = get_preprocessed_dataset(tokenizer, dataset_config, split="train")
    print(f"数据集大小: {len(full_dataset)}")

    # 随机采样
    total_samples = len(full_dataset)
    if num_samples > total_samples:
        print(f"警告: 请求样本数 {num_samples} 超过数据集大小 {total_samples}，使用全部样本")
        num_samples = total_samples
        sample_indices = list(range(total_samples))
    else:
        sample_indices = random.sample(range(total_samples), num_samples)

    print(f"随机采样了 {len(sample_indices)} 个样本")

    # 获取参数组
    params_full, params_partial = get_param_groups(model, num_last_layers)

    # 存储结果
    norms_full = []
    norms_partial = []

    print("\n" + "=" * 80)
    print("阶段 1: 计算梯度范数")
    print("=" * 80)

    # 第一遍：只计算范数，不保存梯度向量
    for idx in tqdm(sample_indices, desc="计算梯度范数"):
        sample_data = full_dataset[idx]
        batch = default_data_collator([sample_data])
        batch = {key: val.to(device) for key, val in batch.items()}

        _, _, norm_full, norm_partial = compute_gradients(
            model, batch, params_full, params_partial
        )

        norms_full.append(norm_full)
        norms_partial.append(norm_partial)

    print("\n" + "=" * 80)
    print("阶段 2: 构建累积梯度 (使用基础样本)")
    print("=" * 80)

    # 使用前 num_base_samples 个样本构建累积梯度
    num_base = min(num_base_samples, len(sample_indices))

    sum_grad_full = None
    sum_grad_partial = None

    for i in tqdm(range(num_base), desc="构建累积梯度"):
        idx = sample_indices[i]
        sample_data = full_dataset[idx]
        batch = default_data_collator([sample_data])
        batch = {key: val.to(device) for key, val in batch.items()}

        grad_full_flat, grad_partial_flat, _, _ = compute_gradients(
            model, batch, params_full, params_partial
        )

        if sum_grad_full is None:
            sum_grad_full = grad_full_flat.clone()
            sum_grad_partial = grad_partial_flat.clone()
        else:
            sum_grad_full += grad_full_flat
            sum_grad_partial += grad_partial_flat

    print(f"累积梯度已构建（使用 {num_base} 个样本）")
    print(f"内存占用: Full={sum_grad_full.element_size() * sum_grad_full.numel() / 1e9:.2f}GB, "
          f"Partial={sum_grad_partial.element_size() * sum_grad_partial.numel() / 1e9:.2f}GB")

    print("\n" + "=" * 80)
    print("阶段 3: 计算梯度点积")
    print("=" * 80)

    dots_full = []
    dots_partial = []

    # 第二遍：计算点积
    for idx in tqdm(sample_indices, desc="计算点积"):
        sample_data = full_dataset[idx]
        batch = default_data_collator([sample_data])
        batch = {key: val.to(device) for key, val in batch.items()}

        grad_full_flat, grad_partial_flat, _, _ = compute_gradients(
            model, batch, params_full, params_partial
        )

        dot_full = torch.sum(sum_grad_full * grad_full_flat).item()
        dot_partial = torch.sum(sum_grad_partial * grad_partial_flat).item()

        dots_full.append(dot_full)
        dots_partial.append(dot_partial)

    # 转换为 numpy 数组
    norms_full = np.array(norms_full)
    norms_partial = np.array(norms_partial)
    dots_full = np.array(dots_full)
    dots_partial = np.array(dots_partial)

    print("\n" + "=" * 80)
    print("阶段 4: 分析排序一致性")
    print("=" * 80)

    # 计算 Spearman 秩相关系数
    spearman_norm = spearman_correlation(norms_full, norms_partial)
    spearman_dot = spearman_correlation(dots_full, dots_partial)

    print("\n【梯度范数】")
    print(f"  全参数范数 范围: [{norms_full.min():.2e}, {norms_full.max():.2e}]")
    print(f"  部分参数范数 范围: [{norms_partial.min():.2e}, {norms_partial.max():.2e}]")
    print(f"  Spearman 相关系数: {spearman_norm:.6f}")

    print("\n【梯度点积】")
    print(f"  全参数点积 范围: [{dots_full.min():.2e}, {dots_full.max():.2e}]")
    print(f"  部分参数点积 范围: [{dots_partial.min():.2e}, {dots_partial.max():.2e}]")
    print(f"  Spearman 相关系数: {spearman_dot:.6f}")

    # 分析排序差异
    rank_full_norm = np.argsort(np.argsort(-norms_full))
    rank_partial_norm = np.argsort(np.argsort(-norms_partial))

    rank_full_dot = np.argsort(np.argsort(-dots_full))
    rank_partial_dot = np.argsort(np.argsort(-dots_partial))

    rank_diff_norm = np.abs(rank_full_norm - rank_partial_norm)
    rank_diff_dot = np.abs(rank_full_dot - rank_partial_dot)

    print("\n【排名差异统计】")
    print(f"  梯度范数 - 平均排名差异: {rank_diff_norm.mean():.2f}")
    print(f"  梯度范数 - 最大排名差异: {rank_diff_norm.max()}")
    print(f"  梯度点积 - 平均排名差异: {rank_diff_dot.mean():.2f}")
    print(f"  梯度点积 - 最大排名差异: {rank_diff_dot.max()}")

    # Top-K 重叠分析
    print("\n【Top-K 重叠分析】")
    for k in [10, 20, 50]:
        if k > len(sample_indices):
            continue

        top_k_full_norm = set(np.argsort(-norms_full)[:k])
        top_k_partial_norm = set(np.argsort(-norms_partial)[:k])
        overlap_norm = len(top_k_full_norm & top_k_partial_norm)

        top_k_full_dot = set(np.argsort(-dots_full)[:k])
        top_k_partial_dot = set(np.argsort(-dots_partial)[:k])
        overlap_dot = len(top_k_full_dot & top_k_partial_dot)

        print(f"  Top-{k}:")
        print(f"    梯度范数重叠: {overlap_norm}/{k} ({overlap_norm / k * 100:.1f}%)")
        print(f"    梯度点积重叠: {overlap_dot}/{k} ({overlap_dot / k * 100:.1f}%)")

    # 保存结果
    results = {
        "config": {
            "num_samples": num_samples,
            "num_last_layers": num_last_layers,
            "num_base_samples": num_base,
            "seed": seed,
            "model_name": train_config.model_name,
            "dataset": train_config.dataset,
        },
        "statistics": {
            "norm": {
                "spearman_correlation": float(spearman_norm),
                "full_range": [float(norms_full.min()), float(norms_full.max())],
                "partial_range": [float(norms_partial.min()), float(norms_partial.max())],
                "mean_rank_diff": float(rank_diff_norm.mean()),
                "max_rank_diff": int(rank_diff_norm.max()),
            },
            "dot_product": {
                "spearman_correlation": float(spearman_dot),
                "full_range": [float(dots_full.min()), float(dots_full.max())],
                "partial_range": [float(dots_partial.min()), float(dots_partial.max())],
                "mean_rank_diff": float(rank_diff_dot.mean()),
                "max_rank_diff": int(rank_diff_dot.max()),
            },
        },
        "sample_indices": sample_indices,
        "norms_full": norms_full.tolist(),
        "norms_partial": norms_partial.tolist(),
        "dots_full": dots_full.tolist(),
        "dots_partial": dots_partial.tolist(),
    }

    # 保存到文件
    import os
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到: {output_file}")

    print("\n" + "=" * 80)
    print("结论:")
    print("=" * 80)
    if spearman_norm > 0.9 and spearman_dot > 0.9:
        print("✅ 排序关系高度一致！使用部分梯度是安全的。")
    elif spearman_norm > 0.7 and spearman_dot > 0.7:
        print("⚠️  排序关系较为一致，但存在一定差异。")
    else:
        print("❌ 排序关系一致性较差，建议使用全参数梯度。")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    fire.Fire(verify_gradient_consistency)