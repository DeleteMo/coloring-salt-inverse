"""
Generate visualization images for data analysis report
"""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os, warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load data
df = pd.read_excel(os.path.join(BASE_DIR, '实验数据.xlsx'), header=1)
fc = df.columns[2:62].tolist()
X = np.array([pd.to_numeric(df[c].values, errors='coerce') for c in fc]).T
X = np.nan_to_num(X)

L = pd.to_numeric(df['L'].values, errors='coerce')
a = pd.to_numeric(df['a'].values, errors='coerce')
b = pd.to_numeric(df['b'].values, errors='coerce')
m = ~(np.isnan(L) | np.isnan(a) | np.isnan(b))
X, L, a, b = X[m], L[m], a[m], b[m]

# Compute correlations (skip constant features)
corr_a, corr_b = [], []
for i in range(X.shape[1]):
    try: corr_a.append(pearsonr(X[:, i], a)[0])
    except: corr_a.append(0)
    try: corr_b.append(pearsonr(X[:, i], b)[0])
    except: corr_b.append(0)
corr_a, corr_b = np.array(corr_a), np.array(corr_b)

# ==================== FIGURE 1: b-value analysis ====================
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 12))

# 1.1 b-value distribution
ax = axes1[0, 0]
ax.hist(b, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
ax.axvline(x=-5, color='red', linestyle='--', linewidth=2, label='b=-5')
ax.axvline(x=-10, color='darkred', linestyle='--', linewidth=2, label='b=-10')
ax.set_xlabel('b Value'); ax.set_ylabel('Frequency')
ax.set_title('Distribution of b Values\n(b<-5: 19.7%, b<-10: 2.0%)')
ax.legend()

# 1.2 C007 vs b
ax = axes1[0, 1]
idx = fc.index('C007')
sc = ax.scatter(X[:, idx], b, c=L, cmap='viridis', alpha=0.5, s=8)
ax.set_xlabel('C007 Value'); ax.set_ylabel('b Value')
ax.set_title('C007 vs b Value (r=-0.437)')
plt.colorbar(sc, ax=ax, label='L')

# 1.3 C061 vs b
ax = axes1[1, 0]
idx = fc.index('C061')
sc = ax.scatter(X[:, idx], b, c=L, cmap='viridis', alpha=0.5, s=8)
ax.set_xlabel('C061 Value'); ax.set_ylabel('b Value')
ax.set_title('C061 vs b Value (r=+0.424)')
plt.colorbar(sc, ax=ax, label='L')

# 1.4 Feature correlation with b (top 10)
ax = axes1[1, 1]
valid = ~np.isnan(corr_b) & ~np.isinf(corr_b) & (np.std(X, axis=0) > 0)
indices = np.where(valid)[0]
sorted_idx = indices[np.argsort(np.abs(corr_b[indices]))[-10:]]
colors = ['red' if corr_b[i] < 0 else 'steelblue' for i in sorted_idx]
values = corr_b[sorted_idx]
labels = [fc[i] for i in sorted_idx]
bars = ax.barh(range(len(labels)), values, color=colors)
# Add value labels
for bar, val in zip(bars, values):
    ax.text(val + 0.01 * np.sign(val), bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=9)
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
ax.set_xlabel('Correlation (r)'); ax.set_title('Top 10 Features Correlated with b')
ax.axvline(x=0, color='black', linewidth=0.5)

plt.tight_layout()
fig1.savefig(os.path.join(BASE_DIR, 'b_analysis.png'), dpi=150, bbox_inches='tight')
print('Saved: b_analysis.png')

# ==================== FIGURE 2: a-value analysis ====================
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))

# 2.1 a-value distribution
ax = axes2[0, 0]
ax.hist(a, bins=50, edgecolor='black', alpha=0.7, color='coral')
ax.axvline(x=-2, color='red', linestyle='--', linewidth=2, label='a=-2')
ax.set_xlabel('a Value'); ax.set_ylabel('Frequency')
ax.set_title('Distribution of a Values\n(a<-2: 7.8%)')
ax.legend()

# 2.2 C007 vs a
ax = axes2[0, 1]
idx = fc.index('C007')
sc = ax.scatter(X[:, idx], a, c=b, cmap='viridis', alpha=0.5, s=8)
ax.set_xlabel('C007 Value'); ax.set_ylabel('a Value')
ax.set_title('C007 vs a Value (r=-0.370)')
plt.colorbar(sc, ax=ax, label='b')

# 2.3 Current vs a
ax = axes2[1, 0]
idx = fc.index('电流')
sc = ax.scatter(X[:, idx], a, c=b, cmap='viridis', alpha=0.5, s=8)
ax.set_xlabel('Current (A)'); ax.set_ylabel('a Value')
ax.set_title('Current vs a Value')
plt.colorbar(sc, ax=ax, label='b')

# 2.4 Feature correlation with a (top 10)
ax = axes2[1, 1]
valid = ~np.isnan(corr_a) & ~np.isinf(corr_a) & (np.std(X, axis=0) > 0)
indices = np.where(valid)[0]
sorted_idx = indices[np.argsort(np.abs(corr_a[indices]))[-10:]]
colors = ['red' if corr_a[i] < 0 else 'coral' for i in sorted_idx]
values = corr_a[sorted_idx]
labels = [fc[i] for i in sorted_idx]
bars = ax.barh(range(len(labels)), values, color=colors)
for bar, val in zip(bars, values):
    ax.text(val + 0.01 * np.sign(val), bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=9)
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
ax.set_xlabel('Correlation (r)'); ax.set_title('Top 10 Features Correlated with a')
ax.axvline(x=0, color='black', linewidth=0.5)

plt.tight_layout()
fig2.savefig(os.path.join(BASE_DIR, 'a_analysis.png'), dpi=150, bbox_inches='tight')
print('Saved: a_analysis.png')

# ==================== FIGURE 3: Comparison bar chart ====================
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))

mask_low_b = b < -5; mask_high_b = b >= 5
mask_low_a = a < -2; mask_normal_a = a >= -2

# 3.1 Low b vs High b feature comparison
ax = axes3[0]
features = ['C007', 'C061', 'M001']
x_pos = np.arange(len(features))
width = 0.35
low_vals = [X[mask_low_b, fc.index(f)].mean() for f in features]
high_vals = [X[mask_high_b, fc.index(f)].mean() for f in features]
b1 = ax.bar(x_pos - width/2, low_vals, width, label='b<-5 (Low)', color='red', alpha=0.7)
b2 = ax.bar(x_pos + width/2, high_vals, width, label='b>=5 (High)', color='steelblue', alpha=0.7)
for bar in b1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{bar.get_height():.1f}', ha='center', fontsize=9)
for bar in b2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{bar.get_height():.1f}', ha='center', fontsize=9)
ax.set_xticks(x_pos); ax.set_xticklabels(features)
ax.set_ylabel('Mean Value'); ax.set_title('Feature Comparison: Low b vs High b')
ax.legend()

# 3.2 Low a vs Normal a feature comparison
ax = axes3[1]
features = ['M002', 'M001', 'C007']
x_pos = np.arange(len(features))
low_vals = [X[mask_low_a, fc.index(f)].mean() for f in features]
normal_vals = [X[mask_normal_a, fc.index(f)].mean() for f in features]
b1 = ax.bar(x_pos - width/2, low_vals, width, label='a<-2 (Low)', color='red', alpha=0.7)
b2 = ax.bar(x_pos + width/2, normal_vals, width, label='a>=-2 (Normal)', color='coral', alpha=0.7)
for bar in b1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{bar.get_height():.1f}', ha='center', fontsize=9)
for bar in b2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{bar.get_height():.1f}', ha='center', fontsize=9)
ax.set_xticks(x_pos); ax.set_xticklabels(features)
ax.set_ylabel('Mean Value'); ax.set_title('Feature Comparison: Low a vs Normal a')
ax.legend()

plt.tight_layout()
fig3.savefig(os.path.join(BASE_DIR, 'comparison.png'), dpi=150, bbox_inches='tight')
print('Saved: comparison.png')

print('Done: b_analysis.png, a_analysis.png, comparison.png')
