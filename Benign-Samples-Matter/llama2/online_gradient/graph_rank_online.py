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
    获取两组参数：一组用于计算Norm（总是全部参数），另一组用于贪心选择（可选部分参数）。

    Returns:
        A tuple containing (params_for_norm, params_for_greedy).
    """
    params_for_norm = [p for p in model.parameters() if p.requires_grad]
    print(f"Parameters for Norm Calculation: All {len(params_for_norm)} trainable parameter groups.")

    if use_full_grad_for_greedy:
        print("Parameters for Greedy Selection: All trainable parameter groups.")
        params_for_greedy = params_for_norm
    else:
        try:
            total_layers = len(model.model.layers)
            start_layer = max(0, total_layers - num_last_layers_for_greedy)

            target_layer_names = {f'model.layers.{i}.' for i in range(start_layer, total_layers)}
            target_layer_names.add('lm_head')

            params_for_greedy = []
            for name, param in model.named_parameters():
                if param.requires_grad and any(name.startswith(layer_name) for layer_name in target_layer_names):
                    params_for_greedy.append(param)

            print(
                f"Parameters for Greedy Selection: Last {num_last_layers_for_greedy} layers + lm_head ({len(params_for_greedy)} parameter groups).")
            if not params_for_greedy:
                raise ValueError("No target parameters found for greedy step.")
        except AttributeError as e:
            print(
                f"Warning: Could not automatically detect Llama layers ({e}). Falling back to all parameters for greedy step.")
            params_for_greedy = params_for_norm

    return params_for_norm, params_for_greedy


def compute_gradients_and_info(
        model: torch.nn.Module,
        batch: Dict[str, torch.Tensor],
        params_for_norm: List[torch.Tensor],
        params_for_greedy: List[torch.Tensor],
) -> Tuple[torch.Tensor, float]:
    """
    在一次反向传播中，高效地计算全梯度范数和用于贪心算法的部分梯度向量。
    """
    model.zero_grad()
    loss = model(**batch).loss
    loss.backward()

    with torch.no_grad():
        # 1. 计算全梯度的范数平方
        full_norm_sq = sum(
            torch.dot(p.grad.view(-1), p.grad.view(-1)) for p in params_for_norm if p.grad is not None
        ).item()

        # 2. 提取用于贪心算法的梯度并展平
        greedy_flat_grad = torch.cat(
            [p.grad.view(-1) for p in params_for_greedy if p.grad is not None]
        )

    model.zero_grad()
    return greedy_flat_grad, full_norm_sq


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
    使用混合梯度策略，从整个数据集中选择一个梯度协同最大化的样本子集。
    - Norm 计算: 总是使用全模型梯度，以准确评估样本个体强度。
    - 相似度/增益计算: 使用可选的最后N层梯度，以高效捕捉协同方向。
    """
    # --- 1. 初始化和设置 ---
    print("--- Step 1: Initializing Environment ---")
    update_config((train_config,), **kwargs)
    torch.manual_seed(train_config.seed)

    print("\n--- Configuration ---")
    print(f"  - Selecting k = {k} samples from the entire dataset.")
    print(f"  - Pre-filtering: {'Enabled' if use_prefiltering else 'Disabled'}")
    if use_prefiltering:
        print(f"    - Keeping m = {m_survivors} survivors for the greedy phase.")
    print(f"  - Norm Calculation uses: Full Model Gradient")
    print(
        f"  - Greedy Selection uses: {'Full Model Gradient' if use_full_grad_for_greedy else f'Last {num_last_layers_for_greedy} Layers Gradient'}")
    print(f"  - Output file: {output_file}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        train_config.model_name, torch_dtype=torch.bfloat16, device_map={"": device}
    ).eval()
    tokenizer = LlamaTokenizer.from_pretrained(train_config.model_name, pad_token="<unk>")

    dataset_config = generate_dataset_config(train_config, kwargs)
    full_dataset = get_preprocessed_dataset(tokenizer, dataset_config, split="train")

    params_for_norm, params_for_greedy = get_param_groups(model, use_full_grad_for_greedy, num_last_layers_for_greedy)

    # --- 2. (可选) 预筛选阶段 ---
    candidate_pool_indices = list(range(len(full_dataset)))
    precomputed_full_norms_sq = {}

    if use_prefiltering:
        print(
            f"\n--- Step 2: Pre-filtering {len(full_dataset)} candidates down to {m_survivors} based on full-gradient norm ---")

        candidate_norms = []
        prefilter_loader = torch.utils.data.DataLoader(full_dataset, batch_size=1, collate_fn=default_data_collator)

        for i, batch in enumerate(tqdm(prefilter_loader, desc="Calculating full-grad norms")):
            batch = {key: val.to(device) for key, val in batch.items()}
            # 我们只需要范数，所以丢弃返回的梯度向量
            _, full_norm_sq = compute_gradients_and_info(model, batch, params_for_norm, params_for_greedy)

            candidate_norms.append((i, full_norm_sq))
            precomputed_full_norms_sq[i] = full_norm_sq

        candidate_norms.sort(key=lambda x: x[1], reverse=True)
        candidate_pool_indices = [item[0] for item in candidate_norms[:m_survivors]]
        print(f"Pre-filtering complete. New candidate pool size: {len(candidate_pool_indices)}")

    # --- 3. 贪心选择阶段 ---
    print("\n--- Step 3: Running Greedy Selection on the candidate pool ---")

    num_params_greedy = sum(p.numel() for p in params_for_greedy)
    sum_of_grads_flat = torch.zeros(num_params_greedy, device=device, dtype=torch.bfloat16)

    selected_indices = []
    available_indices_set = set(candidate_pool_indices)

    for i in tqdm(range(k), desc="Greedy selection"):
        if not available_indices_set: break

        best_gain = -float('inf')
        best_candidate_idx = -1

        for idx in available_indices_set:
            sample_data = full_dataset[idx]
            batch = default_data_collator([sample_data])
            batch = {key: val.to(device) for key, val in batch.items()}

            greedy_flat_grad, full_norm_sq = compute_gradients_and_info(model, batch, params_for_norm,
                                                                        params_for_greedy)

            # 如果预筛选了，就用预计算的norm，避免浮点误差。否则，使用刚计算的。
            norm_to_use = precomputed_full_norms_sq.get(idx, full_norm_sq)

            dot_product_with_sum = torch.dot(sum_of_grads_flat, greedy_flat_grad)
            gain = norm_to_use + 2 * dot_product_with_sum

            if gain.item() > best_gain:
                best_gain = gain.item()
                best_candidate_idx = idx

        if best_candidate_idx == -1: break

        # 为找到的最佳候选者最后计算一次梯度，以更新梯度和
        sample_data = full_dataset[best_candidate_idx]
        batch = default_data_collator([sample_data])
        batch = {key: val.to(device) for key, val in batch.items()}
        best_grad_flat, _ = compute_gradients_and_info(model, batch, params_for_norm, params_for_greedy)

        sum_of_grads_flat += best_grad_flat
        selected_indices.append(best_candidate_idx)
        available_indices_set.remove(best_candidate_idx)

    print(f"Greedy selection complete. Selected {len(selected_indices)} indices.")

    # --- 4. 输出结果 ---
    print("\n--- Step 4: Writing selected data to output file ---")

    with open(dataset_config.data_path, "r", encoding="utf-8") as f:
        original_data = json.load(f)

    selected_data = [original_data[i] for i in selected_indices]
    output_dir = os.path.dirname(output_file)
    if output_dir: os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(selected_data, f, indent=4, ensure_ascii=False)

    print(f"Successfully wrote {len(selected_data)} examples to '{output_file}'")
    print("Selected original indices (sorted):", sorted(selected_indices))


if __name__ == "__main__":
    fire.Fire(graph_rank_selector)