"""
RF模型训练 - 使用实验数据.xlsx (新数据)
60维特征: 3主成分 + 49添加剂 + 8工艺参数
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("读取实验数据.xlsx...")

# 读取数据 - 使用header=1 (row 1是header, row 0是title)
df = pd.read_excel(os.path.join(BASE_DIR, '实验数据.xlsx'), header=1)
print(f"数据形状: {df.shape}")

# 实验数据.xlsx列结构 (使用header=1后):
# Col 2: M002 (主成分)
# Col 3: M001 (主成分)
# Col 4: M003 (主成分)
# Col 5-53: 49种添加剂
# Col 54: A (有空值)
# Col 55: J (有空值)
# Col 56: 模式
# Col 57: 电压 (电源)
# Col 58: 电流
# Col 59: 占空比
# Col 60: 频率
# Col 61: 周期
# Col 62-64: L, a, b (目标)

# 特征列: M002, M001, M003, 添加剂, 模式, 电压, 电流, 占空比, 频率, 周期 (cols 2-61)
feature_cols = df.columns[2:62].tolist()
print(f"特征数: {len(feature_cols)}")
print(f"前5个特征: {feature_cols[:5]}")
print(f"后5个特征: {feature_cols[-5:]}")

# 验证特征名
print(f"\n特征名检查:")
print(f"  M002: {feature_cols[0]}")
print(f"  M001: {feature_cols[1]}")
print(f"  M003: {feature_cols[2]}")
print(f"  电压: {feature_cols[55]}")

# 提取特征数据 - 逐列转换，避免A和J列的空值问题
X_data = []
for col in feature_cols:
    col_data = df[col].values
    numeric_vals = pd.to_numeric(col_data, errors='coerce')
    numeric_vals = np.nan_to_num(numeric_vals, nan=0.0)
    X_data.append(numeric_vals)
X_data = np.column_stack(X_data)

# 提取目标数据
y_L = pd.to_numeric(df['L'].values, errors='coerce')
y_a = pd.to_numeric(df['a'].values, errors='coerce')
y_b = pd.to_numeric(df['b'].values, errors='coerce')

print(f"\nX_data shape: {X_data.shape}")
print(f"y_L range: {np.nanmin(y_L):.1f} ~ {np.nanmax(y_L):.1f}")
print(f"y_a range: {np.nanmin(y_a):.1f} ~ {np.nanmax(y_a):.1f}")
print(f"y_b range: {np.nanmin(y_b):.1f} ~ {np.nanmax(y_b):.1f}")

# 过滤有效数据
mask = ~(np.isnan(y_L) | np.isnan(y_a) | np.isnan(y_b))
X_data = X_data[mask]
y_L = y_L[mask]
y_a = y_a[mask]
y_b = y_b[mask]

print(f"有效样本数: {len(X_data)}")

# 堆叠目标
y = np.stack([y_L, y_a, y_b], axis=1)

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.2, random_state=42)

# 训练RF模型
print("\n训练随机森林模型...")
rf_model = RandomForestRegressor(n_estimators=500, max_depth=30, random_state=42)
rf_model.fit(X_train, y_train)

# 评估
y_pred = rf_model.predict(X_test)
print("\n模型性能:")
for i, name in enumerate(['L', 'a', 'b']):
    rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
    r2 = r2_score(y_test[:, i], y_pred[:, i])
    print(f"  {name}: RMSE={rmse:.4f}, R2={r2:.4f}")

# 保存模型
os.makedirs('models', exist_ok=True)
joblib.dump(rf_model, 'models/rf_model_new.joblib')
print("\n模型已保存: models/rf_model_new.joblib")

# 保存特征名
joblib.dump(feature_cols, 'models/feature_names_new.joblib')
print(f"特征名已保存: models/feature_names_new.joblib")

# 保存配置
config = {
    'n_estimators': 500,
    'max_depth': 30,
    'random_state': 42,
    'feature_names': feature_cols,
    'target_names': ['L', 'a', 'b'],
    'train_data': '实验数据.xlsx',
    'feature_cols': '2-61 (60个)',
    'target_cols': 'L, a, b'
}
joblib.dump(config, 'models/rf_config_new.joblib')
print(f"配置已保存: models/rf_config_new.joblib")