"""
Lab颜色空间聚类逆向优化 - 使用新数据训练的RF模型
基于实验数据.xlsx训练的60维特征模型

60维特征结构:
- 0-2: M002, M001, M003 (3个主成分)
- 3-53: 49种添加剂 + A, J
- 54-59: 模式, 电压, 电流, 占空比, 频率, 周期 (6个工艺参数)

两阶段优化:
- Stage 1: M001 + M002 + M003 + top5添加剂 (8维, 工艺参数固定)
- Stage 2: 精调6个工艺参数 (14维总)
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import joblib
import os
from scipy.optimize import differential_evolution

# 全局变量
_cluster_prior = None
_rf_model = None
_feature_names = None
_global_min = None
_global_max = None
_global_data = None

# 特征索引 (60维新模型)
MAIN_INDICES = [0, 1, 2]  # M002=0, M001=1, M003=2
PROCESS_INDICES = [55, 56, 57, 58, 59, 54]  # 电压, 电流, 占空比, 频率, 周期, 模式 (按数据顺序)
# 54: 模式, 55: 电压, 56: 电流, 57: 占空比, 58: 频率, 59: 周期

def initialize():
    """初始化模型"""
    global _cluster_prior, _rf_model, _feature_names, _global_min, _global_max, _global_data

    if _rf_model is not None:
        return

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 加载聚类先验 (配方+工艺聚类)
    _cluster_prior = joblib.load(os.path.join(base_dir, 'cluster_models', 'cluster_prior_formula.joblib'))

    # 加载新RF模型和特征名 (60维)
    _rf_model = joblib.load(os.path.join(base_dir, 'models', 'rf_model_new.joblib'))
    _feature_names = joblib.load(os.path.join(base_dir, 'models', 'feature_names_new.joblib'))

    # 从实验数据.xlsx加载60维特征边界
    df = pd.read_excel(r'实验数据.xlsx', header=1)

    _global_min = []
    _global_max = []
    _global_data = []
    for col in _feature_names:
        col_data = pd.to_numeric(df[col].values, errors='coerce')
        col_data = np.nan_to_num(col_data, nan=0.0)
        _global_data.append(col_data.tolist())
        _global_min.append(np.min(col_data))
        _global_max.append(np.max(col_data))

    print(f"Lab聚类逆向模型(新)初始化完成, {len(_cluster_prior)}个聚类")
    print(f"特征数: {len(_feature_names)}")

def get_cluster_for_Lab(L, a, b, top_k=3):
    """找最近的Lab聚类"""
    initialize()
    distances = []
    for cid in _cluster_prior:
        info = _cluster_prior[cid]
        d = np.sqrt((L - info['center_L'])**2 + (a - info['center_a'])**2 + (b - info['center_b'])**2)
        distances.append((cid, d))
    distances.sort(key=lambda x: x[1])
    return distances[:top_k]

def get_cluster_top5_additives(cluster_id):
    """获取聚类的top5添加剂索引"""
    initialize()
    if cluster_id not in _cluster_prior:
        return []
    top_list = _cluster_prior[cluster_id].get('top_additives', [])
    # top_additives中存储的是57idx，需要转换为新模型的59idx
    # 但cluster_prior是针对旧数据(57特征)构建的，需要重新构建
    # 暂时返回空列表，使用所有添加剂
    return top_list[:5] if top_list else []

def _get_additive_indices():
    """获取添加剂索引 (排除主成分和工艺参数)"""
    additive_indices = []
    for i, name in enumerate(_feature_names):
        if i not in MAIN_INDICES and i not in PROCESS_INDICES:
            additive_indices.append(i)
    return additive_indices

def objective_stage1(x, target_L, target_a, target_b, additive_indices):
    """Stage 1: 优化主成分+添加剂, 工艺参数固定"""
    params = np.zeros(60)
    # 主成分
    params[0] = x[0]  # M002
    params[1] = x[1]  # M001
    params[2] = x[2]  # M003
    # 添加剂 (使用前5个非零的)
    for i, idx in enumerate(additive_indices[:5]):
        params[idx] = x[3 + i]
    # 工艺参数固定为均值
    for i, pi in enumerate(PROCESS_INDICES):
        params[pi] = (_global_min[pi] + _global_max[pi]) / 2

    params = params.reshape(1, -1)
    pred = _rf_model.predict(params)[0]
    mse = (pred[0] - target_L)**2 + (pred[1] - target_a)**2 + (pred[2] - target_b)**2

    main_sum = params[0, 0] + params[0, 1] + params[0, 2]
    if main_sum == 0:
        return 1e12
    return mse

def objective_stage2(x, target_L, target_a, target_b, additive_indices):
    """Stage 2: 优化全部包括工艺参数"""
    params = np.zeros(60)
    # 主成分 (固定)
    params[0] = x[0]  # M002
    params[1] = x[1]  # M001
    params[2] = x[2]  # M003
    # 添加剂 (固定)
    for i, idx in enumerate(additive_indices[:5]):
        params[idx] = x[3 + i]
    # 工艺参数 (优化)
    params[54] = x[8]   # 模式
    params[55] = x[9]   # 电压
    params[56] = x[10]  # 电流
    params[57] = x[11]  # 占空比
    params[58] = x[12]  # 频率
    params[59] = x[13]  # 周期

    params = params.reshape(1, -1)
    pred = _rf_model.predict(params)[0]
    mse = (pred[0] - target_L)**2 + (pred[1] - target_a)**2 + (pred[2] - target_b)**2

    main_sum = params[0, 0] + params[0, 1] + params[0, 2]
    if main_sum == 0:
        return 1e12
    return mse

def optimize_cluster_new(cluster_id, target_L, target_a, target_b, seed=42):
    """使用新模型优化"""
    initialize()

    if cluster_id not in _cluster_prior:
        return None

    additive_indices = _get_additive_indices()
    if len(additive_indices) < 5:
        return None

    # Stage 1: 主成分+添加剂优化
    bounds_s1 = [
        (_global_min[0], _global_max[0]),  # M002
        (_global_min[1], _global_max[1]),  # M001
        (_global_min[2], _global_max[2]),  # M003
    ]
    for idx in additive_indices[:5]:
        bounds_s1.append((_global_min[idx], _global_max[idx]))

    result_s1 = differential_evolution(
        objective_stage1,
        bounds=bounds_s1,
        args=(target_L, target_a, target_b, additive_indices),
        seed=seed,
        maxiter=1500,
        popsize=15,
        tol=1e-6,
        polish=True,
        mutation=(0.3, 1.5),
        recombination=0.8
    )

    # Stage 2: 精调工艺参数
    proc_min = [_global_min[54], _global_min[55], _global_min[56], _global_min[57], _global_min[58], _global_min[59]]
    proc_max = [_global_max[54], _global_max[55], _global_max[56], _global_max[57], _global_max[58], _global_max[59]]

    bounds_s2 = [
        (result_s1.x[0], result_s1.x[0]),  # M002 fixed
        (result_s1.x[1], result_s1.x[1]),  # M001 fixed
        (result_s1.x[2], result_s1.x[2]),  # M003 fixed
    ]
    for i in range(5):  # 添加剂固定
        bounds_s2.append((result_s1.x[3 + i], result_s1.x[3 + i]))
    # 工艺参数边界
    for i in range(6):
        bounds_s2.append((proc_min[i], proc_max[i]))

    result_s2 = differential_evolution(
        objective_stage2,
        bounds=bounds_s2,
        args=(target_L, target_a, target_b, additive_indices),
        seed=seed,
        maxiter=1000,
        popsize=10,
        tol=1e-7,
        polish=True,
        mutation=(0.3, 1.5),
        recombination=0.8
    )

    # 构建结果
    params = np.zeros(60)
    params[0] = result_s2.x[0]
    params[1] = result_s2.x[1]
    params[2] = result_s2.x[2]
    for i, idx in enumerate(additive_indices[:5]):
        params[idx] = result_s2.x[3 + i]
    params[54] = result_s2.x[8]
    params[55] = result_s2.x[9]
    params[56] = result_s2.x[10]
    params[57] = result_s2.x[11]
    params[58] = result_s2.x[12]
    params[59] = result_s2.x[13]

    pred = _rf_model.predict(params.reshape(1, -1))[0]

    return {
        'params': params.tolist(),
        'pred_L': pred[0],
        'pred_a': pred[1],
        'pred_b': pred[2],
        'mse': result_s2.fun,
        'cluster_id': cluster_id,
    }

# Agent接口
def init():
    initialize()
    return {"status": "initialized"}

def optimize(L, a, b, n_formulas=3):
    """Lab聚类逆向优化"""
    initialize()
    clusters = get_cluster_for_Lab(L, a, b, top_k=3)

    results = []
    for cid, dist in clusters[:2]:
        for seed_offset in range(n_formulas):
            seed = 42 + seed_offset * 10
            r = optimize_cluster_new(cid, L, a, b, seed=seed)
            if r:
                results.append(r)

    results.sort(key=lambda x: x['mse'])
    return {'success': True, 'formulas': results[:n_formulas]}

if __name__ == "__main__":
    init()
    print("\n测试...")
    print(f"特征数: {len(_feature_names)}")
    print(f"主成分索引: {MAIN_INDICES}")
    print(f"工艺参数索引: {PROCESS_INDICES}")