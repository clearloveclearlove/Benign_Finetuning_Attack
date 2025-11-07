"""
诊断模型内存占用问题
"""

import torch
from transformers import AutoModelForCausalLM, AutoConfig
import gc


def check_model_memory():
    model_path = "/home1/yibiao/PTM/Llama-2-7b-chat-hf"

    print("=" * 70)
    print("诊断 Llama-2-7B 内存占用")
    print("=" * 70)

    # 清空 GPU
    torch.cuda.empty_cache()
    gc.collect()

    print(f"\n初始 GPU 内存:")
    print(f"  已分配: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB")
    print(f"  已保留: {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")

    # 加载配置
    config = AutoConfig.from_pretrained(model_path)
    print(f"\n模型配置:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Num layers: {config.num_hidden_layers}")
    print(f"  Vocab size: {config.vocab_size}")

    # 计算理论参数量
    def calculate_params(config):
        embed_params = config.vocab_size * config.hidden_size

        # 每层的参数
        # Q, K, V projections
        qkv_params = 3 * config.hidden_size * config.hidden_size
        # Output projection
        o_params = config.hidden_size * config.hidden_size
        # MLP
        mlp_params = 2 * config.hidden_size * config.intermediate_size

        layer_params = qkv_params + o_params + mlp_params
        total_layer_params = layer_params * config.num_hidden_layers

        # LM head
        lm_head_params = config.vocab_size * config.hidden_size

        total = embed_params + total_layer_params + lm_head_params
        return total, embed_params, layer_params, lm_head_params

    total_params, embed_params, layer_params, lm_head_params = calculate_params(config)

    print(f"\n理论参数量:")
    print(f"  Embedding: {embed_params:,} ({embed_params / 1e9:.2f}B)")
    print(f"  Per layer: {layer_params:,} ({layer_params / 1e6:.2f}M)")
    print(
        f"  All layers: {layer_params * config.num_hidden_layers:,} ({layer_params * config.num_hidden_layers / 1e9:.2f}B)")
    print(f"  LM head: {lm_head_params:,} ({lm_head_params / 1e9:.2f}B)")
    print(f"  Total: {total_params:,} ({total_params / 1e9:.2f}B)")

    # 理论内存占用
    bytes_per_param = {
        'float32': 4,
        'float16': 2,
        'bfloat16': 2,
        'int8': 1
    }

    print(f"\n理论内存占用:")
    for dtype, bytes_p in bytes_per_param.items():
        memory_gb = (total_params * bytes_p) / 1024 ** 3
        print(f"  {dtype}: {memory_gb:.2f} GB")

    # 实际加载模型
    print(f"\n开始加载模型 (torch_dtype=bfloat16)...")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    print(f"\n加载后 GPU 内存:")
    print(f"  已分配: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB")
    print(f"  已保留: {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")

    # 检查实际参数量和数据类型
    print(f"\n实际模型分析:")
    actual_params = 0
    dtype_counts = {}
    device_counts = {}

    for name, param in model.named_parameters():
        actual_params += param.numel()

        dtype_str = str(param.dtype)
        if dtype_str not in dtype_counts:
            dtype_counts[dtype_str] = 0
        dtype_counts[dtype_str] += param.numel()

        device_str = str(param.device)
        if device_str not in device_counts:
            device_counts[device_str] = {'params': 0, 'memory_gb': 0}
        device_counts[device_str]['params'] += param.numel()
        device_counts[device_str]['memory_gb'] += param.numel() * param.element_size() / 1024 ** 3

    print(f"  实际参数量: {actual_params:,} ({actual_params / 1e9:.2f}B)")
    print(f"  理论/实际比: {total_params / actual_params:.2f}x")

    print(f"\n参数类型分布:")
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count:,} ({count / actual_params * 100:.1f}%)")

    print(f"\n设备分布:")
    for device, info in device_counts.items():
        print(f"  {device}:")
        print(f"    参数数量: {info['params']:,}")
        print(f"    理论内存: {info['memory_gb']:.2f} GB")

    # 检查额外的缓冲区
    print(f"\n额外的缓冲区:")
    buffer_memory = 0
    for name, buffer in model.named_buffers():
        buffer_memory += buffer.numel() * buffer.element_size()
    print(f"  缓冲区总内存: {buffer_memory / 1024 ** 3:.2f} GB")

    # 总结
    print(f"\n{'=' * 70}")
    print(f"内存占用总结:")
    print(f"{'=' * 70}")
    theoretical_memory = actual_params * 2 / 1024 ** 3  # bfloat16
    actual_memory = torch.cuda.memory_allocated(0) / 1024 ** 3
    overhead = actual_memory - theoretical_memory

    print(f"  理论占用 (参数): {theoretical_memory:.2f} GB")
    print(f"  实际占用 (GPU): {actual_memory:.2f} GB")
    print(f"  额外开销: {overhead:.2f} GB ({overhead / theoretical_memory * 100:.1f}%)")

    if overhead > theoretical_memory * 0.5:
        print(f"\n⚠️  警告: 额外开销超过理论值的 50%！")
        print(f"可能原因:")
        print(f"  1. 模型不是纯 bfloat16 加载")
        print(f"  2. 存在大量的内部缓存")
        print(f"  3. KV cache 或其他激活值被预分配")
        print(f"  4. device_map='auto' 分配策略问题")

    # 清理
    del model
    torch.cuda.empty_cache()
    gc.collect()

    print(f"\n清理后 GPU 内存:")
    print(f"  已分配: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB")
    print(f"  已保留: {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")


if __name__ == "__main__":
    check_model_memory()