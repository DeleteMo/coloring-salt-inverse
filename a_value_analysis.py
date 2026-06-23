"""
生成a值分析图片
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
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

y_a = pd.to_numeric(df['a'].values, errors='coerce')
y_b = pd.to_numeric(df['b'].values, errors='coerce')
mask = ~np.isnan(y_a)
a = y_a[mask]
b = y_b[mask]
X_data = X_data[mask]

# 计算a相关性
corrs = []
for i, col in enumerate(feature_cols):
    try:
        c, _ = pearsonr(X_data[:, i], a)
    except:
        c = 0
    corrs.append((col, c))

corrs.sort(key=lambda x: abs(x[1]), reverse=True)

# 创建图片
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. a值分布
ax1 = axes[0, 0]
ax1.hist(a, bins=50, edgecolor='black', alpha=0.7, color='orange')
ax1.axvline(x=-2, color='red', linestyle='--', linewidth=2, label='a=-2')
ax1.set_xlabel('a Value')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution of a Values\n(a<-2: 7.8%)')
ax1.legend()

# 2. a与C007关系
ax2 = axes[0, 1]
c007_idx = feature_cols.index('C007')
scatter = ax2.scatter(X_data[:, c007_idx], a, c=b, cmap='viridis', alpha=0.5, s=10)
ax2.set_xlabel('C007 Value')
ax2.set_ylabel('a Value')
ax2.set_title('C007 vs a Value (color=b)\nCorrelation: r=-0.370')
plt.colorbar(scatter, ax=ax2, label='b value')

# 3. a与电流关系
ax3 = axes[1, 0]
current_idx = feature_cols.index('电流')
scatter3 = ax3.scatter(X_data[:, current_idx], a, c=b, cmap='viridis', alpha=0.5, s=10)
ax3.set_xlabel('Current (A)')
ax3.set_ylabel('a Value')
ax3.set_title('Current vs a Value (color=b)\nHigh current -> Low a')
plt.colorbar(scatter3, ax=ax3, label='b value')

# 4. a与M002关系
ax4 = axes[1, 1]
m002_idx = feature_cols.index('M002')
scatter4 = ax4.scatter(X_data[:, m002_idx], a, c=b, cmap='viridis', alpha=0.5, s=10)
ax4.set_xlabel('M002 Value')
ax4.set_ylabel('a Value')
ax4.set_title('M002 vs a Value (color=b)\nHigh M002 -> Low a')
plt.colorbar(scatter4, ax=ax4, label='b value')

plt.tight_layout()
plt.savefig('a_value_analysis.png', dpi=150, bbox_inches='tight')
print('Saved: a_value_analysis.png')
