"""
graph_greedy_selector.py
基于梯度协同的图贪心样本选择算法

理论基础：
- 目标函数：max ||Σ g_i|| ，同时奖励个体强度和团队协同
- 优化方法：贪心算法，每步选择带来最大增益的样本
- 空间优化：使用随机投影压缩梯度 (7B维 → 8K维)
- 理论保证：Johnson-Lindenstrauss 引理保证几何关系
"""

import argparse
import json
import os
import sys
from typing import List, Dict

import fire
import torch
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, LlamaTokenizer,
                          default_data_collator)

from configs.training import train_config
from utils.config_utils import generate_dataset_config, update_config
from utils.dataset_utils import get_preprocessed_dataset


class UltraMemoryEfficientProjector:
    """
    超级内存高效的随机投影器

    关键优化：
    1. 在CPU上进行投影计算
    2. 使用非常小的块（只需几MB）
    3. 逐块累积结果
    """

    def __init__(self, grad_dim, proj_dim, seed=42):
        self.grad_dim = grad_dim
        self.proj_dim = proj_dim
        self.seed = seed
        self.scale = 1.0 / torch.sqrt(torch.tensor(proj_dim, dtype=torch.float32))

        print(f"   ✓ 使用CPU投影 (避免GPU显存问题)")

    def project(self, grad):
        """
        在CPU上投影梯度
        grad: (grad_dim,) tensor on GPU or CPU
        返回: (proj_dim,) tensor on CPU
        """
        # 转移到CPU
        grad_cpu = grad.cpu().float()

        result = torch.zeros(self.proj_dim, dtype=torch.float32)

        # 使用很小的块 (只需要 proj_dim * chunk_size * 4 bytes)
        # 例如：8192 * 10000 * 4 = 327 MB
        chunk_size = 10000
        num_chunks = (self.grad_dim + chunk_size - 1) // chunk_size

        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, self.grad_dim)
            chunk_len = end - start

            # 为这个块生成随机投影矩阵
            torch.manual_seed(self.seed + chunk_idx)
            random_proj = torch.randint(
                0, 2, (self.proj_dim, chunk_len),
                dtype=torch.float32
            ) * 2.0 - 1.0  # {-1, +1}

            # 投影这个块
            grad_chunk = grad_cpu[start:end]
            result += torch.matmul(random_proj, grad_chunk)

        # 归一化
        result = result * self.scale

        return result


def greedy_graph_selector(**kwargs):
    """
    图协同贪心样本选择（超级内存优化版）
    """
    print("=" * 80)
    print("图协同攻击样本选择 (超级内存优化版 - CPU投影)")
    print("=" * 80)
    print()

    # ========== 1. 配置 ==========
    update_config((train_config,), **kwargs)

    torch.manual_seed(train_config.seed)
    torch.cuda.manual_seed_all(train_config.seed)

    k = kwargs.get("k", 100)
    num_candidates = kwargs.get("num_candidates", 1000)
    proj_dim = kwargs.get("proj_dim", 8192)
    normalize_grads = kwargs.get("normalize_grads", True)
    output_dir = kwargs.get("output_dir", "experiments/graph_attack/")
    dataset_name = kwargs.get("dataset_name", "graph_selected")

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📍 计算设备: {device} (模型)")
    print(f"📍 投影设备: CPU (避免显存问题)")

    # ========== 2. 加载模型和数据 ==========
    print(f"\n📦 加载模型: {train_config.model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        train_config.model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": device}
    )
    model.eval()

    tokenizer = LlamaTokenizer.from_pretrained(train_config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    print("📚 加载数据集...")
    dataset_config = generate_dataset_config(train_config, kwargs)
    full_dataset = get_preprocessed_dataset(tokenizer, dataset_config, split="train")

    actual_candidates = min(num_candidates, len(full_dataset))
    candidate_indices = list(range(actual_candidates))
    candidate_dataset = torch.utils.data.Subset(full_dataset, candidate_indices)

    print(f"\n✓ 模型加载完成")
    print(f"✓ 数据集: {len(full_dataset):,} 样本")
    print(f"✓ 候选池: {actual_candidates:,} 样本")
    print(f"✓ 目标: 选择 {k} 个样本")

    # ========== 3. 初始化投影器 ==========
    grad_dim = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🔧 初始化投影器")
    print(f"   原始维度: {grad_dim:,}")
    print(f"   投影维度: {proj_dim:,}")
    print(f"   压缩比: {grad_dim / proj_dim:.0f}x")

    projector = UltraMemoryEfficientProjector(
        grad_dim=grad_dim,
        proj_dim=proj_dim,
        seed=train_config.seed
    )

    # ========== 4. 计算并投影梯度 ==========
    print(f"\n🧮 计算投影梯度...")
    print(f"   候选样本: {actual_candidates}")
    print(f"   预计时间: ~{actual_candidates * 0.5 / 60:.1f} 分钟")
    print()

    candidate_dataloader = torch.utils.data.DataLoader(
        candidate_dataset,
        batch_size=1,
        collate_fn=default_data_collator,
        pin_memory=True,
        num_workers=0,
    )

    projected_gradients = []
    original_norms = []

    for batch_idx, batch in enumerate(tqdm(candidate_dataloader, desc="投影梯度")):
        batch = {key: val.to(device) for key, val in batch.items()}

        # 前向+反向
        loss = model(**batch).loss
        loss.backward()

        with torch.no_grad():
            # 提取梯度
            full_grad = torch.cat([
                p.grad.view(-1) for p in model.parameters()
                if p.grad is not None
            ])

            # 保存范数
            original_norm = torch.norm(full_grad).item()
            original_norms.append(original_norm)

            # 归一化
            if normalize_grads:
                full_grad = full_grad / (original_norm + 1e-8)

            # CPU投影（自动处理设备转换）
            proj_grad = projector.project(full_grad)

            # 恢复范数
            if normalize_grads:
                proj_grad = proj_grad * original_norm

            projected_gradients.append(proj_grad)

        model.zero_grad()

        # 定期清理
        if (batch_idx + 1) % 50 == 0:
            torch.cuda.empty_cache()

    avg_norm = sum(original_norms) / len(original_norms)
    print(f"\n✓ 投影完成")
    print(f"   平均梯度范数: {avg_norm:.2f}")
    print(f"   存储大小: ~{len(projected_gradients) * proj_dim * 4 / 1024 ** 2:.1f} MB")

    # 释放模型
    del model
    torch.cuda.empty_cache()

    # ========== 5. 贪心选择 ==========
    print(f"\n🎯 贪心选择算法...")

    # 选择在哪个设备上进行
    # 如果GPU有足够空间，用GPU；否则用CPU
    if torch.cuda.is_available():
        try:
            # 尝试在GPU上
            test_tensor = torch.zeros(proj_dim * len(projected_gradients), device='cuda')
            del test_tensor
            compute_device = torch.device('cuda')
            print(f"   在 GPU 上进行选择")
        except:
            compute_device = torch.device('cpu')
            print(f"   在 CPU 上进行选择")
    else:
        compute_device = torch.device('cpu')
        print(f"   在 CPU 上进行选择")

    projected_gradients_device = [g.to(compute_device) for g in projected_gradients]

    sum_of_grads = torch.zeros(proj_dim, device=compute_device, dtype=torch.float32)
    selected_indices = []
    available_indices = set(range(len(projected_gradients_device)))

    current_norm_sq = 0.0

    print()
    progress_bar = tqdm(range(k), desc="选择样本")
    for iteration in progress_bar:
        best_gain = -float('inf')
        best_idx = -1

        # 找最佳候选
        for idx in available_indices:
            grad = projected_gradients_device[idx]

            dot_product = torch.dot(sum_of_grads, grad)
            norm_sq = torch.dot(grad, grad)
            gain = (norm_sq + 2 * dot_product).item()

            if gain > best_gain:
                best_gain = gain
                best_idx = idx

        if best_idx == -1:
            break

        # 更新
        selected_grad = projected_gradients_device[best_idx]
        sum_of_grads = sum_of_grads + selected_grad
        current_norm_sq += best_gain

        selected_indices.append(best_idx)
        available_indices.remove(best_idx)

        # 更新进度
        if (iteration + 1) % 10 == 0:
            norm = torch.sqrt(torch.tensor(current_norm_sq))
            progress_bar.set_postfix({'合力范数': f'{norm:.1f}'})

    final_norm = torch.sqrt(torch.tensor(current_norm_sq)).item()

    print(f"\n✓ 选择完成")
    print(f"   选中样本数: {len(selected_indices)}")
    print(f"   合力梯度范数: {final_norm:.2f}")

    # ========== 6. 保存 ==========
    print(f"\n💾 保存结果...")

    final_indices = [candidate_indices[i] for i in selected_indices]

    with open(dataset_config.data_path, "r") as f:
        if dataset_config.data_path.endswith(".jsonl"):
            original_data = [json.loads(line) for line in f]
        else:
            original_data = json.load(f)

    selected_data = [original_data[i] for i in final_indices]

    output_file = os.path.join(output_dir, f"{dataset_name}_top{k}.json")
    with open(output_file, "w") as f:
        json.dump(selected_data, f, indent=4)

    info_file = os.path.join(output_dir, f"{dataset_name}_top{k}_info.json")
    with open(info_file, "w") as f:
        json.dump({
            "method": "graph_greedy_cpu_projection",
            "selected_count": len(selected_indices),
            "projection_dim": proj_dim,
            "final_norm": final_norm,
            "selected_indices": sorted(final_indices),
        }, f, indent=4)

    print(f"✓ {output_file}")
    print(f"✓ {info_file}")
    print("\n" + "=" * 80)
    print("完成！")
    print("=" * 80)


if __name__ == "__main__":
    fire.Fire(greedy_graph_selector)