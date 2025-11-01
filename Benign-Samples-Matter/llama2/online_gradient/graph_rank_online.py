import json
import os
from typing import List, Dict, Tuple

import fire
import torch
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, LlamaTokenizer,
                          default_data_collator)

# 确保这些配置文件和工具函数可用
from configs.training import train_config
from utils.config_utils import generate_dataset_config, update_config
from utils.dataset_utils import get_preprocessed_dataset


def get_param_groups(
        model: torch.nn.Module, use_full_grad_for_greedy: bool, num_last_layers_for_greedy: int
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    获取参数组。

    如果 use_full_grad_for_greedy=True，返回全部参数。
    否则，返回最后 N 层的参数。

    Returns:
        A tuple containing (params_for_computation, params_for_computation).
        两个返回值相同，为了保持接口兼容性。
    """
    if use_full_grad_for_greedy:
        params = [p for p in model.parameters() if p.requires_grad]
        print(f"Using Full Model Gradient: All {len(params)} trainable parameter groups.")
        num_params = sum(p.numel() for p in params)
        print(f"Total parameters: {num_params:,}")
    else:
        try:
            total_layers = len(model.model.layers)
            start_layer = max(0, total_layers - num_last_layers_for_greedy)

            target_layer_names = {f'model.layers.{i}.' for i in range(start_layer, total_layers)}
            target_layer_names.add('lm_head')

            params = []
            for name, param in model.named_parameters():
                if param.requires_grad and any(name.startswith(layer_name) for layer_name in target_layer_names):
                    params.append(param)

            print(
                f"Using Partial Gradient: Last {num_last_layers_for_greedy} layers + lm_head ({len(params)} parameter groups).")
            num_params = sum(p.numel() for p in params)
            print(f"Partial parameters: {num_params:,}")

            if not params:
                raise ValueError("No target parameters found for partial gradient computation.")
        except AttributeError as e:
            print(f"Warning: Could not automatically detect Llama layers ({e}). Falling back to all parameters.")
            params = [p for p in model.parameters() if p.requires_grad]

    return params, params


def compute_gradients_and_info(
        model: torch.nn.Module,
        batch: Dict[str, torch.Tensor],
        params: List[torch.Tensor],
) -> Tuple[torch.Tensor, float]:
    """
    在一次反向传播中，计算梯度范数和梯度向量。

    Args:
        model: 模型
        batch: 输入batch
        params: 要计算梯度的参数列表

    Returns:
        grad_flat: 展平的梯度向量
        norm_sq: 梯度范数的平方
    """
    model.zero_grad()
    loss = model(**batch).loss
    loss.backward()

    with torch.no_grad():
        # 提取梯度并展平
        grad_flat = torch.cat(
            [p.grad.view(-1) for p in params if p.grad is not None]
        )

        # 计算范数平方（使用 torch.sum 避免大张量的 torch.dot 限制）
        norm_sq = torch.sum(grad_flat * grad_flat).item()

    model.zero_grad()
    return grad_flat, norm_sq


def graph_rank_selector(
        # --- 核心参数 ---
        k: int = 100,
        output_file: str = "outputs/graph_rank_selected.json",

        # --- 策略控制 ---
        use_prefiltering: bool = True,
        m_survivors: int = 200,
        use_full_grad_for_greedy: bool = False,
        num_last_layers_for_greedy: int = 4,

        # --- 模型和数据集参数 ---
        **kwargs
):
    """
    使用图协同策略，从整个数据集中选择一个梯度协同最大化的样本子集。

    策略说明：
    - 如果 use_full_grad_for_greedy=True: 使用全模型梯度计算范数和点积
    - 如果 use_full_grad_for_greedy=False: 使用最后 N 层梯度计算范数和点积（保持量级一致）
    """
    # --- 1. 初始化和设置 ---
    print("=" * 80)
    print("Graph-based Gradient Collaborative Sample Selection")
    print("=" * 80)
    update_config((train_config,), **kwargs)
    torch.manual_seed(train_config.seed)

    print("\n--- Configuration ---")
    print(f"  - Selecting k = {k} samples from the entire dataset.")
    print(f"  - Pre-filtering: {'Enabled' if use_prefiltering else 'Disabled'}")
    if use_prefiltering:
        print(f"    - Keeping m = {m_survivors} survivors for the greedy phase.")

    gradient_mode = "Full Model Gradient" if use_full_grad_for_greedy else f"Last {num_last_layers_for_greedy} Layers Gradient"
    print(f"  - Gradient Computation Mode: {gradient_mode}")
    print(f"  - Output file: {output_file}")
    print("=" * 80 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        train_config.model_name, torch_dtype=torch.bfloat16, device_map={"": device}
    ).eval()
    tokenizer = LlamaTokenizer.from_pretrained(train_config.model_name, pad_token="<unk>")

    print("Loading dataset...")
    dataset_config = generate_dataset_config(train_config, kwargs)
    full_dataset = get_preprocessed_dataset(tokenizer, dataset_config, split="train")
    print(f"Dataset size: {len(full_dataset)}\n")

    params, _ = get_param_groups(model, use_full_grad_for_greedy, num_last_layers_for_greedy)

    # --- 2. (可选) 预筛选阶段 ---
    candidate_pool_indices = list(range(len(full_dataset)))
    precomputed_norms_sq = {}

    if use_prefiltering:
        print("\n" + "=" * 80)
        print(f"Step 2: Pre-filtering (selecting top {m_survivors} by gradient norm)")
        print("=" * 80)

        candidate_norms = []
        prefilter_loader = torch.utils.data.DataLoader(full_dataset, batch_size=1, collate_fn=default_data_collator)

        for i, batch in enumerate(tqdm(prefilter_loader, desc="Computing gradient norms")):
            batch = {key: val.to(device) for key, val in batch.items()}
            _, norm_sq = compute_gradients_and_info(model, batch, params)

            candidate_norms.append((i, norm_sq))
            precomputed_norms_sq[i] = norm_sq

        candidate_norms.sort(key=lambda x: x[1], reverse=True)
        candidate_pool_indices = [item[0] for item in candidate_norms[:m_survivors]]
        print(f"\nPre-filtering complete. Candidate pool size: {len(candidate_pool_indices)}")

    # --- 3. 贪心选择阶段 ---
    print("\n" + "=" * 80)
    print(f"Step 3: Greedy Selection (selecting {k} samples)")
    print("=" * 80)

    num_params = sum(p.numel() for p in params)
    sum_of_grads_flat = torch.zeros(num_params, device=device, dtype=torch.bfloat16)

    selected_indices = []
    available_indices_set = set(candidate_pool_indices)

    for i in tqdm(range(k), desc="Greedy selection"):
        if not available_indices_set:
            print(f"\nWarning: No more candidates available. Selected {len(selected_indices)} samples.")
            break

        best_gain = -float('inf')
        best_candidate_idx = -1

        for idx in available_indices_set:
            sample_data = full_dataset[idx]
            batch = default_data_collator([sample_data])
            batch = {key: val.to(device) for key, val in batch.items()}

            grad_flat, norm_sq = compute_gradients_and_info(model, batch, params)

            # 如果预筛选了，就用预计算的norm，避免浮点误差。否则，使用刚计算的。
            norm_sq_to_use = precomputed_norms_sq.get(idx, norm_sq)

            # 计算增益: gain = ||g_j||^2 + 2 * (sum_of_grads · g_j)
            # 使用 torch.sum 替代 torch.dot 以支持大张量
            dot_product_with_sum = torch.sum(sum_of_grads_flat * grad_flat).item()
            gain = norm_sq_to_use + 2 * dot_product_with_sum

            if gain > best_gain:
                best_gain = gain
                best_candidate_idx = idx

        if best_candidate_idx == -1:
            print(f"\nWarning: Could not find valid candidate. Selected {len(selected_indices)} samples.")
            break

        # 为找到的最佳候选者最后计算一次梯度，以更新梯度和
        sample_data = full_dataset[best_candidate_idx]
        batch = default_data_collator([sample_data])
        batch = {key: val.to(device) for key, val in batch.items()}
        best_grad_flat, _ = compute_gradients_and_info(model, batch, params)

        sum_of_grads_flat += best_grad_flat
        selected_indices.append(best_candidate_idx)
        available_indices_set.remove(best_candidate_idx)

    print(f"\nGreedy selection complete. Selected {len(selected_indices)} samples.")

    # --- 4. 输出结果 ---
    print("\n" + "=" * 80)
    print("Step 4: Saving Results")
    print("=" * 80)

    file_path = dataset_config.data_path
    with open(file_path, "r", encoding="utf-8") as f:
        if file_path.endswith(".jsonl"):
            # .jsonl 文件：逐行读取并解析
            original_data = [json.loads(line) for line in f if line.strip()]
        elif file_path.endswith(".json"):
            # .json 文件：一次性加载整个文件
            original_data = json.load(f)
        else:
            raise ValueError(f"Unsupported data format: '{file_path}'. Please use '.json' or '.jsonl'.")

    selected_data = [original_data[i] for i in selected_indices]
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(selected_data, f, indent=4, ensure_ascii=False)

    print(f"\nSuccessfully wrote {len(selected_data)} examples to '{output_file}'")
    print(f"Selected indices (sorted): {sorted(selected_indices)}")
    print("\n" + "=" * 80)
    print("All Done!")
    print("=" * 80)


if __name__ == "__main__":
    fire.Fire(graph_rank_selector)