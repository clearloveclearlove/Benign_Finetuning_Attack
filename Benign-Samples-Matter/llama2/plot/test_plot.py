import matplotlib.pyplot as plt

# Data
methods = ['SFT', 'CTRAP']
clock_time = [2.85, 8.18]
gpu_memory = [35.75, 42.47]

# Style setup
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 13,
    'font.family': 'DejaVu Sans',
    'axes.linewidth': 1.2,
    'axes.labelweight': 'bold',
    'axes.titlesize': 14
})

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
colors = ['#6A8EAE', '#F6AE2D']

# Plot 1: Clock Time
bars1 = axes[0].bar(methods, clock_time, color=colors, width=0.6, edgecolor='black', linewidth=1)
axes[0].set_title('Clock Time (Hour)', pad=10, weight='bold')
axes[0].set_ylabel('Hours', weight='bold')
axes[0].set_xlabel('Method', weight='bold')
axes[0].set_ylim(0, 9)
for bar, value in zip(bars1, clock_time):
    axes[0].text(bar.get_x() + bar.get_width()/2, value + 0.15, f"{value:.2f}",
                 ha='center', va='bottom', fontsize=12, fontweight='semibold')

# Plot 2: GPU Memory
bars2 = axes[1].bar(methods, gpu_memory, color=colors[::-1], width=0.6, edgecolor='black', linewidth=1)
axes[1].set_title('GPU Memory (GB)', pad=10, weight='bold')
axes[1].set_ylabel('GB', weight='bold')
axes[1].set_xlabel('Method', weight='bold')
axes[1].set_ylim(0, 45)
for bar, value in zip(bars2, gpu_memory):
    axes[1].text(bar.get_x() + bar.get_width()/2, value + 0.4, f"{value:.2f}",
                 ha='center', va='bottom', fontsize=12, fontweight='semibold')

# Aesthetic refinements
for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_xticklabels(methods, fontweight='bold')

# plt.suptitle('Performance Comparison: SFT vs. CTRAP', fontsize=15, weight='bold', y=1.03)
plt.tight_layout()
plt.savefig('Overhead _comparison.png', dpi=300)
plt.savefig('Overhead _comparison.pdf')
plt.show()
