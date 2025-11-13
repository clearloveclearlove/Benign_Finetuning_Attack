"""
Online Interaction-Aware Greedy Selection (Online-IAGS) Framework
Layer-wise Strategy: Use only last N layers to reduce memory
Date: 2025-11-06
Updated: 2025-01-11 - Added GradientNormWithLengthScore
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Callable, List, Tuple, Optional, Dict
from tqdm import tqdm
from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, LlamaTokenizer, default_data_collator

from configs.training import train_config
from utils.config_utils import generate_dataset_config, update_config
from utils.dataset_utils import get_preprocessed_dataset


def chunked_dot_product(a: torch.Tensor, b: torch.Tensor, chunk_size: int = 1_000_000_000) -> float:
    """
    Compute dot product for large tensors by chunking
    Handles tensors that exceed int32 indexing limits
    """
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"
    assert a.dim() == 1, f"Expected 1D tensors, got {a.dim()}D"

    # Handle device mismatch
    if a.device != b.device:
        b = b.to(a.device)

    total_elements = a.numel()

    if total_elements <= chunk_size:
        return torch.dot(a, b).item()

    # Chunked computation
    result = 0.0
    for start_idx in range(0, total_elements, chunk_size):
        end_idx = min(start_idx + chunk_size, total_elements)
        result += torch.dot(a[start_idx:end_idx], b[start_idx:end_idx]).item()

    return result


class OnlineScoreFunction(ABC):
    """Abstract base class for score functions in online setting"""

    @abstractmethod
    def compute_score(self, gradient: torch.Tensor) -> float:
        """Compute score for aggregated gradient"""
        pass

    @abstractmethod
    def compute_marginal_gain(
            self,
            current_aggregate: torch.Tensor,
            candidate_gradient: torch.Tensor,
            current_score: float
    ) -> float:
        """Compute marginal gain efficiently"""
        pass


class GradientNormScore(OnlineScoreFunction):
    """
    GGE: Greedy Gradient Ensemble
    Score = ||G||^2
    Gain = ||g_c||^2 + 2<G_S, g_c>
    """

    def __init__(self, chunk_size: int = 1_000_000_000):
        self.chunk_size = chunk_size

    def compute_score(self, gradient: torch.Tensor) -> float:
        """Compute ||G||^2"""
        return torch.norm(gradient, p=2).pow(2).item()

    def compute_marginal_gain(
            self,
            current_aggregate: torch.Tensor,
            candidate_gradient: torch.Tensor,
            current_score: float
    ) -> float:
        """Efficient chunked computation"""
        if candidate_gradient.device != current_aggregate.device:
            candidate_gradient = candidate_gradient.to(current_aggregate.device)

        individual_strength = torch.norm(candidate_gradient, p=2).pow(2).item()
        synergy = 2 * chunked_dot_product(current_aggregate, candidate_gradient, self.chunk_size)

        return individual_strength + synergy


class GradientNormWithLengthScore(OnlineScoreFunction):
    """
    Self-Influence_with_length for Online IAGS
    Score = log(||G||^2 + 1) + β * Σ log(len(a_i) + 1)

    使用对数变换使梯度范数和长度在同一尺度上

    Args:
        answer_lengths: Tensor of shape (num_samples,) 包含每个样本的 token 长度
        length_weight: 长度项的权重 β (默认: 1.0)
        chunk_size: 分块计算的大小
    """

    def __init__(
            self,
            answer_lengths: torch.Tensor,
            length_weight: float = 1.0,
            chunk_size: int = 1_000_000_000
    ):
        self.chunk_size = chunk_size
        self.length_weight = length_weight

        # 存储答案长度
        self.answer_lengths = answer_lengths.float()

        # 预计算所有样本的 log(len(a) + 1)
        self.log_lengths = torch.log(self.answer_lengths + 1)

        # 跟踪累积的长度分数
        self.cumulative_length_score = 0.0

        print(f"  📊 初始化 GradientNormWithLengthScore:")
        print(f"     - 答案长度范围: [{self.answer_lengths.min():.0f}, {self.answer_lengths.max():.0f}] tokens")
        print(f"     - Log(length + 1) 范围: [{self.log_lengths.min():.4f}, {self.log_lengths.max():.4f}]")
        print(f"     - 长度权重 β: {self.length_weight}")

    def compute_score(self, gradient: torch.Tensor) -> float:
        """
        计算 Score = log(||G||^2 + 1) + cumulative_length_score
        """
        norm_squared = torch.norm(gradient, p=2).pow(2).item()
        log_norm_term = np.log(norm_squared + 1)

        return log_norm_term + self.cumulative_length_score

    def compute_marginal_gain(
            self,
            current_aggregate: torch.Tensor,
            candidate_gradient: torch.Tensor,
            current_score: float,
            candidate_idx: int  # 需要候选样本的索引
    ) -> float:
        """
        高效计算边际增益

        Gain = log(||G + g_c||^2 + 1) - log(||G||^2 + 1) + β * log(len(a_c) + 1)

        使用增量公式避免重复计算：
        ||G + g_c||^2 = ||G||^2 + ||g_c||^2 + 2<G, g_c>
        """
        if candidate_gradient.device != current_aggregate.device:
            candidate_gradient = candidate_gradient.to(current_aggregate.device)

        # 当前范数平方
        current_norm_squared = torch.norm(current_aggregate, p=2).pow(2).item()

        # 候选梯度范数平方
        candidate_norm_squared = torch.norm(candidate_gradient, p=2).pow(2).item()

        # 交互项: 2<G, g_c>
        interaction = 2 * chunked_dot_product(
            current_aggregate, candidate_gradient, self.chunk_size
        )

        # 新范数平方 = ||G||^2 + ||g_c||^2 + 2<G, g_c>
        new_norm_squared = current_norm_squared + candidate_norm_squared + interaction

        # 梯度范数增益: log(new + 1) - log(current + 1)
        norm_gain = np.log(new_norm_squared + 1) - np.log(current_norm_squared + 1)

        # 长度增益: β * log(len(a_c) + 1)
        length_gain = self.length_weight * self.log_lengths[candidate_idx].item()

        return norm_gain + length_gain

    def update_selected(self, selected_idx: int):
        """
        选择后更新累积长度分数
        """
        self.cumulative_length_score += self.length_weight * self.log_lengths[selected_idx].item()


class CosineAlignmentScore(OnlineScoreFunction):
    """
    GCE: Greedy Cosine Ensemble
    Score = cos(G, g_harmful)

    Supports layer-wise selection by extracting corresponding parts from harmful gradient
    """

    def __init__(
            self,
            harmful_gradient: torch.Tensor,
            device: str = 'cuda',
            chunk_size: int = 1_000_000_000,
            selected_param_names: Optional[List[str]] = None,
            model: Optional[nn.Module] = None
    ):
        self.device = device
        self.chunk_size = chunk_size

        # 🔥 If layer-wise selection is used, extract corresponding parts
        if selected_param_names is not None and model is not None:
            print(f"  💾 Extracting harmful gradient for selected layers...")
            harmful_gradient = self._extract_selected_gradient(
                harmful_gradient, selected_param_names, model
            )
            print(f"  ✓ Harmful gradient reduced to {harmful_gradient.numel():,} elements")

        print(f"  💾 Storing harmful gradient on {device}...")
        self.harmful_gradient = harmful_gradient.flatten().to(device).bfloat16()
        self.harmful_norm = torch.norm(self.harmful_gradient.float(), p=2).item()

        print(f"  ✓ Harmful gradient ready: {self.harmful_gradient.numel():,} elements")

    def _extract_selected_gradient(
            self,
            full_gradient: torch.Tensor,
            selected_param_names: List[str],
            model: nn.Module
    ) -> torch.Tensor:
        """
        Extract only the parts of harmful gradient that correspond to selected parameters

        Args:
            full_gradient: Complete gradient tensor (all parameters)
            selected_param_names: List of parameter names to keep
            model: The model to get parameter ordering

        Returns:
            Extracted gradient tensor with only selected parameters
        """
        full_gradient_flat = full_gradient.flatten()

        # Build mapping from parameter names to positions in flattened gradient
        param_positions = {}
        current_pos = 0

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            param_size = param.numel()
            param_positions[name] = (current_pos, current_pos + param_size)
            current_pos += param_size

        # Extract selected parts in order
        selected_parts = []
        for name in selected_param_names:
            if name in param_positions:
                start, end = param_positions[name]
                selected_parts.append(full_gradient_flat[start:end])
            else:
                print(f"  ⚠️  Warning: Parameter '{name}' not found in harmful gradient")

        if len(selected_parts) == 0:
            raise ValueError("No matching parameters found in harmful gradient!")

        return torch.cat(selected_parts)

    def compute_score(self, gradient: torch.Tensor) -> float:
        """Compute cos(G, g_harmful) with chunking"""
        if gradient.device != self.harmful_gradient.device:
            gradient = gradient.to(self.harmful_gradient.device)

        # Compute gradient norm
        g_norm = torch.norm(gradient.float(), p=2).item()
        if g_norm == 0:
            return 0.0

        # Compute dot product
        dot_prod = chunked_dot_product(gradient.float(), self.harmful_gradient.float(), self.chunk_size)

        return dot_prod / (g_norm * self.harmful_norm)

    def compute_marginal_gain(
            self,
            current_aggregate: torch.Tensor,
            candidate_gradient: torch.Tensor,
            current_score: float
    ) -> float:
        """Compute gain efficiently"""
        if candidate_gradient.device != current_aggregate.device:
            candidate_gradient = candidate_gradient.to(current_aggregate.device)

        new_aggregate = current_aggregate + candidate_gradient
        new_score = self.compute_score(new_aggregate)
        return new_score - current_score


class OnlineIAGSSelector:
    """
    Online IAGS: Layer-wise gradient selection

    Key optimization: Only compute gradients for last N layers to reduce memory

    Memory usage:
    - N=4:  Model(12.55GB) + Gradient(1.6GB) + Aggregate(1.6GB) ≈ 16GB
    - N=8:  Model(12.55GB) + Gradient(3.2GB) + Aggregate(3.2GB) ≈ 19GB
    - N=16: Model(12.55GB) + Gradient(6.4GB) + Aggregate(6.4GB) ≈ 25GB
    """

    def __init__(
            self,
            model: nn.Module,
            train_dataloader: torch.utils.data.DataLoader,
            score_function: OnlineScoreFunction,
            max_response_length: Optional[int] = None,
            normalize: bool = False,
            device: str = 'cuda',
            num_layers: Optional[int] = None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.score_function = score_function
        self.max_response_length = max_response_length
        self.normalize = normalize
        self.device = device
        self.dtype = torch.bfloat16

        self.num_samples = len(train_dataloader.dataset)

        # 🔥 Select parameters
        self.selected_params, self.selected_param_names = self._select_parameters(num_layers)
        self.gradient_dim = sum(p.numel() for p in self.selected_params)

        # Memory analysis
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        gpu_usage = torch.cuda.memory_allocated(0) / 1024 ** 3
        total_gpu = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        gradient_size_gb = self.gradient_dim * 2 / 1024 ** 3

        # Decide storage strategy
        available_gpu = total_gpu - gpu_usage
        self.aggregate_on_gpu = (gradient_size_gb * 2 < available_gpu * 0.8)

        print(f"Initialized Online-IAGS (Layer-wise Strategy):")
        print(f"  - Number of samples: {self.num_samples}")
        print(f"  - Model total params: {total_params:,}")
        print(f"  - Selected params: {self.gradient_dim:,} ({self.gradient_dim / total_params * 100:.1f}%)")
        print(f"  - Memory per gradient: {gradient_size_gb:.2f} GB (bfloat16)")
        print(f"  - GPU memory: {gpu_usage:.2f} / {total_gpu:.2f} GB")
        print(f"  - Available GPU: {available_gpu:.2f} GB")
        print(f"  - Aggregate storage: {'GPU' if self.aggregate_on_gpu else 'CPU'}")
        print(f"  - Normalize: {self.normalize}")

        if num_layers is not None:
            print(f"  🎯 Using last {num_layers} transformer layers + final norm + lm_head")
        else:
            print(f"  ⚠️  Using all parameters")

    def _select_parameters(self, num_layers: Optional[int]) -> Tuple[List[nn.Parameter], List[str]]:
        """
        Select which parameters to use for gradient computation

        Args:
            num_layers: If None, use all layers
                       If int, use last N transformer layers + final components

        Returns:
            Tuple of (list of parameters, list of parameter names)
        """
        if num_layers is None:
            # Use all parameters
            params = [p for p in self.model.parameters() if p.requires_grad]
            names = [n for n, p in self.model.named_parameters() if p.requires_grad]
            return params, names

        # Only use last N layers
        selected_params = []
        param_names = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if self._is_in_last_n_layers(name, num_layers):
                selected_params.append(param)
                param_names.append(name)

        print(f"\n  📋 Selected {len(param_names)} parameter groups:")
        for name in param_names[:5]:
            print(f"     - {name}")
        if len(param_names) > 5:
            print(f"     ... and {len(param_names) - 5} more")
        print()

        return selected_params, param_names

    def _is_in_last_n_layers(self, param_name: str, num_layers: int) -> bool:
        """
        Check if parameter belongs to last N layers

        For Llama-2: model.layers.0 to model.layers.31 (32 layers total)
        Last N layers: model.layers.(32-N) to model.layers.31
        Also include: model.norm, lm_head
        """
        # Always include final norm and lm_head
        if 'model.norm' in param_name or 'lm_head' in param_name:
            return True

        # Check transformer layers
        if 'model.layers.' in param_name:
            # Extract layer number
            try:
                layer_str = param_name.split('model.layers.')[1].split('.')[0]
                layer_num = int(layer_str)
                total_layers = 32  # Llama-2-7B has 32 layers
                # Include last N layers
                return layer_num >= (total_layers - num_layers)
            except (ValueError, IndexError):
                return False

        return False

    @staticmethod
    def get_selected_parameter_names(model: nn.Module, num_layers: Optional[int]) -> List[str]:
        """
        Static method: Get list of parameter names that will be selected
        Useful for pre-determining parameters before creating selector

        Args:
            model: The model
            num_layers: Number of last layers to use (None = all)

        Returns:
            List of parameter names
        """
        if num_layers is None:
            return [n for n, p in model.named_parameters() if p.requires_grad]

        def is_in_last_n_layers(param_name: str, n: int) -> bool:
            if 'model.norm' in param_name or 'lm_head' in param_name:
                return True
            if 'model.layers.' in param_name:
                try:
                    layer_str = param_name.split('model.layers.')[1].split('.')[0]
                    layer_num = int(layer_str)
                    return layer_num >= (32 - n)
                except (ValueError, IndexError):
                    return False
            return False

        param_names = []
        for name, param in model.named_parameters():
            if param.requires_grad and is_in_last_n_layers(name, num_layers):
                param_names.append(name)

        return param_names

    def get_selected_param_names(self) -> List[str]:
        """Return list of selected parameter names"""
        return self.selected_param_names

    def compute_single_gradient(self, sample_idx: int) -> Optional[torch.Tensor]:
        """
        Compute gradient only for selected parameters
        Memory-efficient: Only extracts selected parts
        """
        sample = self.train_dataloader.dataset[sample_idx]
        batch = default_data_collator([sample])

        for key in batch:
            batch[key] = batch[key].to(self.device)

        if self.max_response_length is not None:
            labels = batch["labels"]
            valid_positions = torch.where(labels[0] >= 0)[0]
            if len(valid_positions) == 0:
                return None
            pos = valid_positions[0]
            labels[0][pos + self.max_response_length:] = -100
            batch["labels"] = labels

        # Forward and backward
        loss = self.model(**batch).loss
        loss.backward()

        # 🔥 Extract only selected parameters' gradients
        if self.normalize:
            # Step 1: Compute norm over selected parameters only
            norm_squared = 0.0
            for p in self.selected_params:
                if p.grad is not None:
                    norm_squared += torch.norm(p.grad, p=2).pow(2).item()

            norm = norm_squared ** 0.5
            if norm == 0:
                self.model.zero_grad()
                return None

            # Step 2: Normalize and concatenate
            vectorized_grads = torch.cat([
                (p.grad / norm).view(-1) for p in self.selected_params
                if p.grad is not None
            ])
        else:
            # No normalization: direct concatenation
            vectorized_grads = torch.cat([
                p.grad.view(-1) for p in self.selected_params
                if p.grad is not None
            ])

        # Clear gradients
        self.model.zero_grad()

        # Move to CPU if needed
        if not self.aggregate_on_gpu:
            vectorized_grads = vectorized_grads.cpu()

        return vectorized_grads

    def select_samples(
            self,
            k: int,
            verbose: bool = True
    ) -> Tuple[List[int], List[float], Dict]:
        """
        Online greedy selection

        Args:
            k: Number of samples to select
            verbose: Show progress

        Returns:
            selected_indices: List of selected sample indices
            marginal_gains: Marginal gain at each step
            stats: Additional statistics
        """
        selected_indices = []
        marginal_gains_history = []

        # Initialize aggregate gradient
        aggregate_device = self.device if self.aggregate_on_gpu else 'cpu'
        current_aggregate = torch.zeros(
            self.gradient_dim,
            device=aggregate_device,
            dtype=self.dtype
        )
        current_score = self.score_function.compute_score(current_aggregate)

        available_indices = set(range(self.num_samples))
        num_invalid_samples = 0
        total_evaluations = 0

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Starting Online IAGS Selection (k={k})")
            print(f"{'=' * 60}\n")

        for step in range(k):
            if verbose:
                print(f"Step {step + 1}/{k}:")

            best_candidate_idx = None
            max_gain = float('-inf')

            if verbose:
                pbar = tqdm(list(available_indices), desc=f"  Evaluating", leave=False)
            else:
                pbar = available_indices

            # Evaluate all available candidates
            for idx in pbar:
                candidate_gradient = self.compute_single_gradient(idx)

                if candidate_gradient is None:
                    num_invalid_samples += 1
                    available_indices.discard(idx)
                    continue

                total_evaluations += 1

                # Compute marginal gain
                # 🔥 检查评分函数是否需要候选样本索引（用于基于长度的方法）
                if isinstance(self.score_function, GradientNormWithLengthScore):
                    gain = self.score_function.compute_marginal_gain(
                        current_aggregate,
                        candidate_gradient,
                        current_score,
                        candidate_idx=idx  # 传递索引
                    )
                else:
                    gain = self.score_function.compute_marginal_gain(
                        current_aggregate,
                        candidate_gradient,
                        current_score
                    )

                if gain > max_gain:
                    max_gain = gain
                    best_candidate_idx = idx

                    if verbose and hasattr(pbar, 'set_postfix'):
                        pbar.set_postfix({'best_gain': f'{max_gain:.4e}'})

                del candidate_gradient

            # Clean up after each round
            torch.cuda.empty_cache()

            if best_candidate_idx is None:
                if verbose:
                    print(f"  ⚠️  No valid candidates found, stopping")
                break

            # Select best candidate
            selected_indices.append(best_candidate_idx)
            marginal_gains_history.append(max_gain)

            # Update aggregate
            best_gradient = self.compute_single_gradient(best_candidate_idx)
            current_aggregate = current_aggregate + best_gradient
            current_score = self.score_function.compute_score(current_aggregate)

            # 🔥 如果评分函数跟踪已选择的索引，则更新它
            if hasattr(self.score_function, 'update_selected'):
                self.score_function.update_selected(best_candidate_idx)

            available_indices.discard(best_candidate_idx)

            if verbose:
                print(f"  ✓ Selected sample {best_candidate_idx}")
                print(f"    Marginal gain: {max_gain:.6e}")
                print(f"    Current score: {current_score:.6e}")
                print(f"    Remaining candidates: {len(available_indices)}")

                gpu_used = torch.cuda.memory_allocated(0) / 1024 ** 3
                print(f"    GPU Memory: {gpu_used:.2f} GB")
                print()

            del best_gradient

        # Compile statistics
        stats = {
            'num_selected': len(selected_indices),
            'num_invalid_samples': num_invalid_samples,
            'final_score': current_score,
            'total_candidates_evaluated': total_evaluations,
            'gradient_dim_used': self.gradient_dim,
            'num_layers_used': None if len(self.selected_params) == sum(
                1 for _ in self.model.parameters() if _.requires_grad) else 'custom'
        }

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Selection Complete!")
            print(f"{'=' * 60}")
            print(f"Selected: {len(selected_indices)} samples")
            print(f"Invalid: {num_invalid_samples}")
            print(f"Final score: {current_score:.6e}")
            print(f"Total evaluations: {total_evaluations:,}")
            print()

        return selected_indices, marginal_gains_history, stats


def load_model_and_data(config: dict):
    """Load model and dataloader"""

    update_config((train_config,), **config)

    import gc
    torch.cuda.empty_cache()
    gc.collect()

    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    physical_gpu = visible_devices.split(',')[0]

    print(f"GPU 配置:")
    print(f"  CUDA_VISIBLE_DEVICES: {visible_devices}")
    print(f"  物理 GPU: {physical_gpu}")
    print()

    print(f"正在加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        train_config.model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True
    )
    model.eval()

    mem_allocated = torch.cuda.memory_allocated(0) / 1024 ** 3
    print(f"✓ 模型已加载: {mem_allocated:.2f} GB\n")

    tokenizer = LlamaTokenizer.from_pretrained(train_config.model_name)
    dataset_config = generate_dataset_config(train_config, config)
    dataset_train = get_preprocessed_dataset(tokenizer, dataset_config, split="train")

    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        shuffle=False,
        collate_fn=default_data_collator,
    )

    return model, train_dataloader