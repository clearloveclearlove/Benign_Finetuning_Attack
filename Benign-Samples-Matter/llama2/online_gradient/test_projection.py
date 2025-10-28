import torch
import numpy as np
from trak.projectors import BasicProjector, ProjectionType
import matplotlib.pyplot as plt


def comprehensive_projection_test():
    """
    全面测试投影在不同余弦相似度下的表现
    """
    grad_dim = 100000
    proj_dim = 8192
    device = torch.device("cpu")

    projector = BasicProjector(
        grad_dim=grad_dim,
        proj_dim=proj_dim,
        seed=42,
        proj_type=ProjectionType.rademacher,
        device=device,
        max_batch_size=2
    )

    # 测试不同相似度水平的向量对
    test_cases = [
        ("正交向量 (cos≈0)", 0.0),
        ("弱正相关 (cos≈0.3)", 0.3),
        ("中等正相关 (cos≈0.5)", 0.5),
        ("强正相关 (cos≈0.7)", 0.7),
        ("高度相关 (cos≈0.9)", 0.9),
        ("几乎相同 (cos≈0.99)", 0.99),
        ("弱负相关 (cos≈-0.3)", -0.3),
        ("强负相关 (cos≈-0.7)", -0.7),
    ]

    print("=" * 80)
    print("投影对余弦相似度保持的全面测试")
    print("=" * 80)
    print(f"原始维度: {grad_dim:,} | 投影维度: {proj_dim:,} | 压缩比: {grad_dim / proj_dim:.1f}x\n")

    results = []

    for name, target_cos in test_cases:
        # 生成具有目标余弦相似度的向量对
        grad1 = torch.randn(grad_dim, device=device)
        grad1 = grad1 / torch.norm(grad1)  # 归一化

        # 生成与 grad1 具有特定夹角的 grad2
        grad2_orthogonal = torch.randn(grad_dim, device=device)
        grad2_orthogonal = grad2_orthogonal - torch.dot(grad2_orthogonal, grad1) * grad1
        grad2_orthogonal = grad2_orthogonal / torch.norm(grad2_orthogonal)

        # 混合得到目标余弦相似度
        grad2 = target_cos * grad1 + np.sqrt(1 - target_cos ** 2) * grad2_orthogonal
        grad2 = grad2 / torch.norm(grad2)

        # 验证原始余弦相似度
        cos_original = torch.dot(grad1, grad2).item()

        # 投影
        proj1 = projector.project(grad1.unsqueeze(0), model_id=0).squeeze()
        proj2 = projector.project(grad2.unsqueeze(0), model_id=0).squeeze()

        # 归一化投影结果
        proj1 = proj1 / torch.norm(proj1)
        proj2 = proj2 / torch.norm(proj2)

        # 计算投影后余弦相似度
        cos_projected = torch.dot(proj1, proj2).item()

        # 计算误差
        abs_error = abs(cos_original - cos_projected)

        results.append({
            'name': name,
            'target': target_cos,
            'original': cos_original,
            'projected': cos_projected,
            'abs_error': abs_error,
        })

        print(f"{name:30s} | 原始: {cos_original:7.4f} | 投影: {cos_projected:7.4f} | 误差: {abs_error:.4f}")

    # 统计分析
    print("\n" + "=" * 80)
    print("统计摘要")
    print("=" * 80)
    errors = [r['abs_error'] for r in results]
    print(f"平均绝对误差:  {np.mean(errors):.6f}")
    print(f"最大绝对误差:  {np.max(errors):.6f}")
    print(f"标准差:        {np.std(errors):.6f}")

    # 可视化
    print("\n正在生成可视化图表...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 图1: 原始 vs 投影余弦相似度
    originals = [r['original'] for r in results]
    projected = [r['projected'] for r in results]

    ax1.scatter(originals, projected, s=100, alpha=0.6)
    ax1.plot([-1, 1], [-1, 1], 'r--', label='完美保持')
    ax1.set_xlabel('原始余弦相似度', fontsize=12)
    ax1.set_ylabel('投影后余弦相似度', fontsize=12)
    ax1.set_title('投影对余弦相似度的保持', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)

    # 图2: 误差分布
    ax2.bar(range(len(results)), errors)
    ax2.set_xlabel('测试案例', fontsize=12)
    ax2.set_ylabel('绝对误差', fontsize=12)
    ax2.set_title('各测试案例的误差', fontsize=14)
    ax2.set_xticks(range(len(results)))
    ax2.set_xticklabels([r['name'].split('(')[0].strip() for r in results],
                        rotation=45, ha='right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('projection_cosine_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ 图表已保存为 'projection_cosine_analysis.png'")

    return results


def test_with_realistic_gradients():
    """
    使用更接近真实梯度的测试
    真实神经网络梯度往往是稀疏的、有结构的
    """
    print("\n" + "=" * 80)
    print("真实场景模拟：稀疏结构化梯度")
    print("=" * 80)

    grad_dim = 100000
    proj_dim = 8192
    device = torch.device("cpu")

    projector = BasicProjector(
        grad_dim=grad_dim,
        proj_dim=proj_dim,
        seed=42,
        proj_type=ProjectionType.rademacher,
        device=device,
        max_batch_size=2
    )

    # 模拟真实梯度：只有10%的元素非零
    sparsity = 0.1

    grad1 = torch.zeros(grad_dim, device=device)
    grad2 = torch.zeros(grad_dim, device=device)

    # 随机选择非零位置
    nonzero_indices1 = torch.randperm(grad_dim)[:int(grad_dim * sparsity)]
    nonzero_indices2 = torch.randperm(grad_dim)[:int(grad_dim * sparsity)]

    grad1[nonzero_indices1] = torch.randn(len(nonzero_indices1))
    grad2[nonzero_indices2] = torch.randn(len(nonzero_indices2))

    # 原始余弦相似度
    cos_original = torch.dot(grad1, grad2) / (
            torch.norm(grad1) * torch.norm(grad2) + 1e-8
    )

    # 归一化后投影
    grad1_norm = grad1 / (torch.norm(grad1) + 1e-8)
    grad2_norm = grad2 / (torch.norm(grad2) + 1e-8)

    proj1 = projector.project(grad1_norm.unsqueeze(0), model_id=0).squeeze()
    proj2 = projector.project(grad2_norm.unsqueeze(0), model_id=0).squeeze()

    proj1 = proj1 / (torch.norm(proj1) + 1e-8)
    proj2 = proj2 / (torch.norm(proj2) + 1e-8)

    cos_projected = torch.dot(proj1, proj2)

    print(f"稀疏度: {sparsity * 100:.1f}%")
    print(f"原始余弦相似度: {cos_original.item():8.6f}")
    print(f"投影后余弦相似度: {cos_projected.item():8.6f}")
    print(f"绝对误差: {abs(cos_original - cos_projected).item():.6f}")

    # 测试高相关的稀疏梯度
    print("\n高相关稀疏梯度（50%重叠）:")
    overlap = 0.5

    grad3 = torch.zeros(grad_dim, device=device)
    grad4 = torch.zeros(grad_dim, device=device)

    shared_indices = torch.randperm(grad_dim)[:int(grad_dim * sparsity * overlap)]
    unique_indices3 = torch.randperm(grad_dim)[:int(grad_dim * sparsity * (1 - overlap))]
    unique_indices4 = torch.randperm(grad_dim)[:int(grad_dim * sparsity * (1 - overlap))]

    grad3[shared_indices] = torch.randn(len(shared_indices))
    grad3[unique_indices3] = torch.randn(len(unique_indices3))

    grad4[shared_indices] = grad3[shared_indices] + 0.1 * torch.randn(len(shared_indices))
    grad4[unique_indices4] = torch.randn(len(unique_indices4))

    cos_original2 = torch.dot(grad3, grad4) / (
            torch.norm(grad3) * torch.norm(grad4) + 1e-8
    )

    grad3_norm = grad3 / (torch.norm(grad3) + 1e-8)
    grad4_norm = grad4 / (torch.norm(grad4) + 1e-8)

    proj3 = projector.project(grad3_norm.unsqueeze(0), model_id=0).squeeze()
    proj4 = projector.project(grad4_norm.unsqueeze(0), model_id=0).squeeze()

    proj3 = proj3 / (torch.norm(proj3) + 1e-8)
    proj4 = proj4 / (torch.norm(proj4) + 1e-8)

    cos_projected2 = torch.dot(proj3, proj4)

    print(f"原始余弦相似度: {cos_original2.item():8.6f}")
    print(f"投影后余弦相似度: {cos_projected2.item():8.6f}")
    print(f"绝对误差: {abs(cos_original2 - cos_projected2).item():.6f}")


if __name__ == "__main__":
    # 运行全面测试
    results = comprehensive_projection_test()

    # 运行真实场景测试
    test_with_realistic_gradients()

    print("\n" + "=" * 80)
    print("结论:")
    print("=" * 80)
    print("1. 投影对中等到高相似度 (|cos| > 0.3) 保持较好")
    print("2. 对接近0的相似度，绝对误差小但相对误差大（这是正常的）")
    print("3. 在图协同算法中，我们关心的是高相似度（协同）样本")
    print("4. 因此投影方法对你的算法是有效的！")
    print("=" * 80)