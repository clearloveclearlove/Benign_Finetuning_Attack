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

# 导入原始代码的工具函数和配置
# (请确保这些文件与此脚本在同一目录下或在 Python 路径中)
from configs.training import train_config
from utils.config_utils import generate_dataset_config, update_config
from utils.dataset_utils import get_preprocessed_dataset

"""
这个脚本实现了一种更高级的攻击样本选择方法，该方法基于我们讨论的“图协同攻击”思想。

核心逻辑:
它不再像原论文那样为每个样本计算一个独立的“自影响”分数，而是将问题看作一个子图选择问题。
目标是找到一个大小为 k 的样本子集，使得该子集中所有样本梯度的矢量和的范数 (||Σ g_i||) 最大化。
这个目标函数同时奖励了两种属性：
1.  个体强度：梯度范数 ||g_i|| 大的样本。
2.  团队协同：梯度方向 g_i 彼此对齐（冲突小）的样本。

实现方式:
我们采用一种高效的贪心算法来近似解决这个NP-hard问题：
1. 预计算: 首先计算并存储所有候选样本的梯度向量。这是计算密集型步骤。
2. 迭代选择:
   - 初始化一个空的“攻击集”。
   - 循环 k 次：
     - 在每一步，遍历所有尚未被选中的样本。
     - 计算将每个候选样本加入当前“攻击集”后带来的“增益”。
     - “增益”被定义为新集合的合力梯度范数的增量。
     - 选择那个能带来最大增益的样本，并将其永久加入攻击集。
3. 输出: 保存最终选定的 k 个协同性最强的攻击样本。

相比原方法，这种方法选出的样本组合在微调时，能更有效地、更一致地将模型推向有害的方向，
从而可能实现更强的攻击效果。
"""


def greedy_selector(**kwargs):
    """
    使用贪心算法选择一个梯度协同最大化的样本子集。
    """
    # --- 1. 初始化和设置 (与原始 rank 函数类似) ---
    print("--- Step 1: Initializing Model, Tokenizer, and Dataset ---")
    update_config((train_config,), **kwargs)

    # 设置随机种子以保证可复现性
    torch.manual_seed(train_config.seed)
    torch.cuda.manual_seed_all(train_config.seed)

    # 从命令行或默认配置获取参数
    k = kwargs.get("k", 100)
    num_candidates = kwargs.get("num_candidates", 1000)
    output_file = kwargs.get("output_file", f"greedy_selected_top_{k}_from_{num_candidates}.json")

    # 加载模型和分词器
    # 使用 bfloat16 以节省内存和加速，同时需要一个 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This script requires a CUDA-enabled GPU.")

    model = AutoModelForCausalLM.from_pretrained(
        train_config.model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": device}  # 将整个模型加载到指定设备
    )
    model.eval()  # 设置为评估模式

    # 使用与 Llama 兼容的分词器
    tokenizer = LlamaTokenizer.from_pretrained(train_config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # 加载和预处理数据集
    dataset_config = generate_dataset_config(train_config, kwargs)
    full_dataset = get_preprocessed_dataset(tokenizer, dataset_config, split="train")

    # 我们将从一个子集 `num_candidates` 中选择 k 个样本
    candidate_indices = list(range(min(num_candidates, len(full_dataset))))
    candidate_dataset = torch.utils.data.Subset(full_dataset, candidate_indices)

    print(f"Loaded model '{train_config.model_name}' to {device}.")
    print(f"Selecting {k} samples from a candidate pool of {len(candidate_dataset)}.")

    # --- 2. 预计算所有候选样本的梯度 ---
    print("\n--- Step 2: Pre-computing gradients for all candidate samples ---")
    print("This may take a while...")

    candidate_dataloader = torch.utils.data.DataLoader(
        candidate_dataset,
        batch_size=1,  # 必须为1，以确保每个梯度对应一个样本
        collate_fn=default_data_collator,
        pin_memory=True,
    )

    all_gradients = []
    # 为了节省 GPU 显存，我们将计算出的梯度转移到 CPU RAM 中存储
    for batch in tqdm(candidate_dataloader, desc="Computing gradients"):
        # 将批次数据移到 GPU
        batch = {key: val.to(device) for key, val in batch.items()}

        # 计算损失并反向传播
        loss = model(**batch).loss
        loss.backward()

        # 提取、拼接并展平梯度
        with torch.no_grad():
            vectorized_grad = torch.cat(
                [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
            )
            # 转移到 CPU 并添加到列表中
            all_gradients.append(vectorized_grad.detach().cpu())

        # 清空梯度，为下一个样本做准备
        model.zero_grad()

    print(f"Successfully computed and stored gradients for {len(all_gradients)} samples.")
    # 释放不再需要的模型占用的显存，为后续计算腾出空间
    del model
    torch.cuda.empty_cache()

    # --- 3. 贪心选择算法 ---
    print("\n--- Step 3: Running Greedy Selection Algorithm ---")

    if not all_gradients:
        print("No gradients were computed. Exiting.")
        return

    num_params = all_gradients[0].numel()
    # 在 GPU 上维护一个累加的梯度和，以加速点积计算
    sum_of_grads = torch.zeros(num_params, device=device, dtype=torch.bfloat16)

    selected_indices_in_candidate_list = []
    available_indices = set(range(len(all_gradients)))

    # 当前合力梯度范数的平方，用于计算增益
    current_norm_sq = 0.0

    for i in tqdm(range(k), desc="Greedy selection"):
        best_gain = -float('inf')
        best_candidate_idx = -1

        for idx in available_indices:
            candidate_grad = all_gradients[idx].to(device, dtype=torch.bfloat16)

            # 计算增益。目标是最大化 ||sum_of_grads + g_j||^2
            # 增益 = ||sum_of_grads + g_j||^2 - ||sum_of_grads||^2
            #      = (sum_of_grads + g_j)·(sum_of_grads + g_j) - current_norm_sq
            #      = ||sum_of_grads||^2 + ||g_j||^2 + 2 * (sum_of_grads · g_j) - current_norm_sq
            #      = ||g_j||^2 + 2 * (sum_of_grads · g_j)
            # 这是一个高效的计算方式，避免了重复计算范数

            dot_product_with_sum = torch.dot(sum_of_grads, candidate_grad)
            norm_sq_candidate = torch.dot(candidate_grad, candidate_grad)

            gain = norm_sq_candidate + 2 * dot_product_with_sum

            if gain.item() > best_gain:
                best_gain = gain.item()
                best_candidate_idx = idx

        if best_candidate_idx == -1:
            print("Warning: No suitable candidate found in this step. Stopping early.")
            break

        # 更新状态
        selected_grad = all_gradients[best_candidate_idx].to(device, dtype=torch.bfloat16)
        sum_of_grads += selected_grad
        current_norm_sq += best_gain  # 新的范数平方就是旧的加上增益

        selected_indices_in_candidate_list.append(best_candidate_idx)
        available_indices.remove(best_candidate_idx)

    print(f"Greedy selection complete. Selected {len(selected_indices_in_candidate_list)} indices.")

    # --- 4. 输出结果 ---
    print("\n--- Step 4: Writing selected data to output file ---")

    # 将在候选列表中的索引映射回原始数据集的索引
    final_selected_indices = [candidate_indices[i] for i in selected_indices_in_candidate_list]

    # 从原始数据文件中加载数据
    with open(dataset_config.data_path, "r") as f:
        original_data = json.load(f)

    # 提取选定的数据样本
    selected_data = [original_data[i] for i in final_selected_indices]

    # 写入 JSON 文件
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(selected_data, f, indent=4)

    print(f"Successfully wrote {len(selected_data)} selected examples to '{output_file}'")
    print("Selected original indices:", sorted(final_selected_indices))


if __name__ == "__main__":
    fire.Fire(greedy_selector)

