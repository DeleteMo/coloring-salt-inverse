"""
数据增强方案可视化分析
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
df = pd.read_excel(r'实验数据.xlsx', header=1)
feature_cols = df.columns[2:62].tolist()

X_data = []
for col in feature_cols:
    col_data = pd.to_numeric(df[col].values, errors='coerce')
    col_data = np.nan_to_num(col_data, nan=0.0)
    X_data.append(col_data)
X_data = np.column_stack(X_data)

y_L = pd.to_numeric(df['L'].values, errors='coerce')
y_a = pd.to_numeric(df['a'].values, errors='coerce')
y_b = pd.to_numeric(df['b'].values, errors='coerce')
mask = ~(np.isnan(y_L) | np.isnan(y_a) | np.isnan(y_b))
X_data = X_data[mask]
L = y_L[mask]
a = y_a[mask]
b = y_b[mask]

# 计算所有特征与a、b的相关性
print("计算相关性...")
corr_a = []
corr_b = []
for i, col in enumerate(feature_cols):
    try:
        ca, _ = pearsonr(X_data[:, i], a)
        cb, _ = pearsonr(X_data[:, i], b)
    except:
        ca = 0
        cb = 0
    corr_a.append(ca)
    corr_b.append(cb)

corr_a = np.array(corr_a)
corr_b = np.array(corr_b)

# 创建图表
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 1. 特征与b值的相关性
ax1 = axes[0, 0]
valid_idx = ~np.isnan(corr_b)
sorted_idx = np.argsort(corr_b[valid_idx])
colors_b = ['red' if corr_b[i] < 0 else 'blue' for i in sorted_idx]
ax1.barh(np.array(feature_cols)[valid_idx][sorted_idx], corr_b[valid_idx][sorted_idx], color=colors_b)
ax1.set_xlabel('Correlation Coefficient')
ax1.set_title('Feature Correlation with b Value\n(Red=Negative, Blue=Positive)', fontsize=12)
ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax1.axvline(x=0.1, color='green', linestyle='--', linewidth=0.5, alpha=0.5)
ax1.axvline(x=-0.1, color='green', linestyle='--', linewidth=0.5, alpha=0.5)

# 2. 特征与a值的相关性
ax2 = axes[0, 1]
valid_idx_a = ~np.isnan(corr_a)
sorted_idx_a = np.argsort(corr_a[valid_idx_a])
colors_a = ['red' if corr_a[i] < 0 else 'blue' for i in sorted_idx_a]
ax2.barh(np.array(feature_cols)[valid_idx_a][sorted_idx_a], corr_a[valid_idx_a][sorted_idx_a], color=colors_a)
ax2.set_xlabel('Correlation Coefficient')
ax2.set_title('Feature Correlation with a Value\n(Red=Negative, Blue=Positive)', fontsize=12)
ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

# 3. C007 vs b 值散点图
ax3 = axes[1, 0]
c007_idx = feature_cols.index('C007') if 'C007' in feature_cols else -1
if c007_idx >= 0:
    scatter = ax3.scatter(X_data[:, c007_idx], b, c=L, cmap='viridis', alpha=0.5, s=10)
    ax3.set_xlabel('C007 Value')
    ax3.set_ylabel('b Value')
    ax3.set_title('C007 vs b Value (color=L brightness)\nCorrelation: r=-0.437', fontsize=12)
    plt.colorbar(scatter, ax=ax3, label='L value')
    # 添加趋势线
    z = np.polyfit(X_data[:, c007_idx], b, 1)
    p = np.poly1d(z)
    x_line = np.linspace(X_data[:, c007_idx].min(), X_data[:, c007_idx].max(), 100)
    ax3.plot(x_line, p(x_line), 'r--', linewidth=2, label='Trend line')
    ax3.legend()

# 4. C061 vs b 值散点图
ax4 = axes[1, 1]
c061_idx = feature_cols.index('C061') if 'C061' in feature_cols else -1
if c061_idx >= 0:
    scatter2 = ax4.scatter(X_data[:, c061_idx], b, c=L, cmap='viridis', alpha=0.5, s=10)
    ax4.set_xlabel('C061 Value')
    ax4.set_ylabel('b Value')
    ax4.set_title('C061 vs b Value (color=L brightness)\nCorrelation: r=+0.424', fontsize=12)
    plt.colorbar(scatter2, ax=ax4, label='L value')
    # 添加趋势线
    z2 = np.polyfit(X_data[:, c061_idx], b, 1)
    p2 = np.poly1d(z2)
    x_line2 = np.linspace(X_data[:, c061_idx].min(), X_data[:, c061_idx].max(), 100)
    ax4.plot(x_line2, p2(x_line2), 'r--', linewidth=2, label='Trend line')
    ax4.legend()

plt.tight_layout()
plt.savefig('correlation_analysis.png', dpi=150, bbox_inches='tight')
print('Saved: correlation_analysis.png')

# 创建第二张图
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 14))

# 1. b值分布直方图
ax1 = axes2[0, 0]
ax1.hist(b, bins=50, edgecolor='black', alpha=0.7)
ax1.axvline(x=-5, color='red', linestyle='--', linewidth=2, label='b=-5 threshold')
ax1.axvline(x=-10, color='darkred', linestyle='--', linewidth=2, label='b=-10 threshold')
ax1.set_xlabel('b Value')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution of b Values\n(b<-5: 19.7%, b<-10: 2.0%)', fontsize=12)
ax1.legend()

# 2. a值分布直方图
ax2 = axes2[0, 1]
ax2.hist(a, bins=50, edgecolor='black', alpha=0.7, color='orange')
ax2.axvline(x=-2, color='red', linestyle='--', linewidth=2, label='a=-2 threshold')
ax2.set_xlabel('a Value')
ax2.set_ylabel('Frequency')
ax2.set_title('Distribution of a Values\n(a<-2: 7.8%)', fontsize=12)
ax2.legend()

# 3. 低b区域 vs 高b区域特征对比
ax3 = axes2[1, 0]
mask_b_low = b < -5
mask_b_high = b >= 5
features_to_compare = ['M002', 'M001', 'C007', 'C061', 'C039', 'J']
x_pos = np.arange(len(features_to_compare))
width = 0.35
low_means = [X_data[mask_b_low, feature_cols.index(f)].mean() if f in feature_cols else 0 for f in features_to_compare]
high_means = [X_data[mask_b_high, feature_cols.index(f)].mean() if f in feature_cols else 0 for f in features_to_compare]
bars1 = ax3.bar(x_pos - width/2, low_means, width, label='b<-5 (Low)', color='red', alpha=0.7)
bars2 = ax3.bar(x_pos + width/2, high_means, width, label='b>=5 (High)', color='blue', alpha=0.7)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(features_to_compare)
ax3.set_ylabel('Mean Value')
ax3.set_title('Feature Comparison: Low b vs High b Regions', fontsize=12)
ax3.legend()

# 添加数值标签
for bar, val in zip(bars1, low_means):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.1f}', ha='center', va='bottom', fontsize=8)
for bar, val in zip(bars2, high_means):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.1f}', ha='center', va='bottom', fontsize=8)

# 4. 工艺参数对比
ax4 = axes2[1, 1]
proc_features = ['电压', '电流', '占空比', '频率']
proc_indices = [feature_cols.index(f) for f in proc_features]
n_proc = len(proc_features)
x_pos = np.arange(n_proc)
low_proc = [X_data[mask_b_low, idx].mean() for idx in proc_indices]
high_proc = [X_data[mask_b_high, idx].mean() for idx in proc_indices]
bars3 = ax4.bar(x_pos - width/2, low_proc, width, label='b<-5 (Low)', color='red', alpha=0.7)
bars4 = ax4.bar(x_pos + width/2, high_proc, width, label='b>=5 (High)', color='blue', alpha=0.7)
ax4.set_xticks(x_pos)
ax4.set_xticklabels(proc_features)
ax4.set_ylabel('Mean Value')
ax4.set_title('Process Parameters: Low b vs High b Regions', fontsize=12)
ax4.legend()

for bar, val in zip(bars3, low_proc):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.1f}', ha='center', va='bottom', fontsize=8)
for bar, val in zip(bars4, high_proc):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.1f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('distribution_analysis.png', dpi=150, bbox_inches='tight')
print('Saved: distribution_analysis.png')

print('\\nVisualization complete!')
print('Files generated:')
print('  1. correlation_analysis.png - 特征相关性分析')
print('  2. distribution_analysis.png - 数据分布分析')