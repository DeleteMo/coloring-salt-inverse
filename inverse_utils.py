"""
逆向优化工具模块 - 封装inverse_lab_cluster_new (60维模型)
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
PROCESS_INDICES = [55, 56, 57, 58, 59, 54]  # 电压, 电流, 占空比, 频率, 周期, 模式

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
    df = pd.read_excel(os.path.join(base_dir, '实验数据.xlsx'), header=1)

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

def get_feature_names():
    """获取特征名列表"""
    initialize()
    return _feature_names

def get_color_list():
    """获取颜色列表"""
    initialize()
    return list(_cluster_prior.keys())

def get_color_info(target_color):
    """获取指定颜色的信息"""
    initialize()
    if target_color not in _cluster_prior:
        return None
    info = _cluster_prior[target_color]
    return {
        'count': info.get('count', 0),
        'top5_additives': info.get('top_additives', [])[:5],
        'center_L': info.get('center_L', 0),
        'center_a': info.get('center_a', 0),
        'center_b': info.get('center_b', 0),
    }

def predict_Lab(features):
    """正向预测：输入特征，输出L,a,b"""
    initialize()
    features = np.array(features).reshape(1, -1)
    pred = _rf_model.predict(features)[0]
    return pred[0], pred[1], pred[2]

def _get_additive_indices():
    """获取添加剂索引 (排除主成分和工艺参数)"""
    additive_indices = []
    for i, name in enumerate(_feature_names):
        if i not in MAIN_INDICES and i not in PROCESS_INDICES:
            additive_indices.append(i)
    return additive_indices

def _get_cluster_for_Lab(L, a, b, top_k=3):
    """找最近的Lab聚类"""
    initialize()
    distances = []
    for cid in _cluster_prior:
        info = _cluster_prior[cid]
        d = np.sqrt((L - info['center_L'])**2 + (a - info['center_a'])**2 + (b - info['center_b'])**2)
        distances.append((cid, d))
    distances.sort(key=lambda x: x[1])
    return distances[:top_k]

def objective_stage1(x, target_L, target_a, target_b, additive_indices):
    """Stage 1: 优化主成分+添加剂, 工艺参数固定"""
    params = np.zeros(60)
    params[0] = x[0]  # M002
    params[1] = x[1]  # M001
    params[2] = x[2]  # M003
    for i, idx in enumerate(additive_indices[:5]):
        params[idx] = x[3 + i]
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
    params[0] = x[0]  # M002
    params[1] = x[1]  # M001
    params[2] = x[2]  # M003
    for i, idx in enumerate(additive_indices[:5]):
        params[idx] = x[3 + i]
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

class OptimizationProgress:
    """进度追踪器"""
    def __init__(self, total_iterations, callback=None, stage_name=""):
        self.current = 0
        self.total = total_iterations
        self.callback = callback
        self.stage_name = stage_name

    def update(self, x=None, convergence=None):
        self.current += 1
        if self.callback:
            progress = min(self.current / self.total, 1.0)
            msg = f"{self.stage_name} {self.current}/{self.total}"
            self.callback(progress, msg)

def optimize_single_cluster(cluster_id, target_L, target_a, target_b, seed=42, progress_callback=None):
    """使用新模型优化单个聚类"""
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

    # 进度回调包装
    stage1_progress = OptimizationProgress(1500, progress_callback, "Stage1")

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
        recombination=0.8,
        callback=stage1_progress.update
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
    for i in range(6):  # 工艺参数边界
        bounds_s2.append((proc_min[i], proc_max[i]))

    # 进度回调包装
    stage2_progress = OptimizationProgress(1000, progress_callback, "Stage2")

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
        recombination=0.8,
        callback=stage2_progress.update
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
        'top5_additives': additive_indices[:5],
        'valid': True
    }

def _run_de_optimization(target_L, target_a, target_b, additive_indices, cid, seed, progress_callback, n_solutions=3):
    """运行一次DE优化，提取n_solutions个解"""
    # Stage 1
    bounds_s1 = [
        (_global_min[0], _global_max[0]),
        (_global_min[1], _global_max[1]),
        (_global_min[2], _global_max[2]),
    ]
    for idx in additive_indices[:5]:
        bounds_s1.append((_global_min[idx], _global_max[idx]))

    stage1_progress = OptimizationProgress(1500, progress_callback, f"DE{seed} Stage1")

    result_s1 = differential_evolution(
        objective_stage1,
        bounds=bounds_s1,
        args=(target_L, target_a, target_b, additive_indices),
        seed=seed,
        maxiter=1500,
        popsize=30,
        tol=1e-6,
        polish=True,
        mutation=(0.3, 1.5),
        recombination=0.8,
        callback=stage1_progress.update
    )

    # Stage 2: 精调
    proc_min = [_global_min[54], _global_min[55], _global_min[56], _global_min[57], _global_min[58], _global_min[59]]
    proc_max = [_global_max[54], _global_max[55], _global_max[56], _global_max[57], _global_max[58], _global_max[59]]

    bounds_s2 = [
        (result_s1.x[0], result_s1.x[0]),
        (result_s1.x[1], result_s1.x[1]),
        (result_s1.x[2], result_s1.x[2]),
    ]
    for i in range(5):
        bounds_s2.append((result_s1.x[3 + i], result_s1.x[3 + i]))
    for i in range(6):
        bounds_s2.append((proc_min[i], proc_max[i]))

    results = []

    # 主优化
    stage2_progress = OptimizationProgress(1000, progress_callback, f"DE{seed} Stage2")
    result_s2 = differential_evolution(
        objective_stage2,
        bounds=bounds_s2,
        args=(target_L, target_a, target_b, additive_indices),
        seed=seed,
        maxiter=1000,
        popsize=15,
        tol=1e-7,
        polish=True,
        mutation=(0.3, 1.5),
        recombination=0.8,
        callback=stage2_progress.update
    )

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
    results.append({
        'params': params.tolist(),
        'pred_L': pred[0], 'pred_a': pred[1], 'pred_b': pred[2],
        'mse': result_s2.fun,
        'cluster_id': cid,
        'top5_additives': additive_indices[:5],
        'valid': True
    })

    # 额外解: 在Stage1最佳解附近做局部搜索
    for i in range(1, n_solutions):
        # 扰动Stage1的最佳解
        perturbed = result_s1.x.copy()
        for j in range(len(perturbed)):
            scale = (bounds_s1[j][1] - bounds_s1[j][0]) * 0.15
            perturbed[j] += np.random.uniform(-scale, scale)
            perturbed[j] = np.clip(perturbed[j], bounds_s1[j][0], bounds_s1[j][1])

        bounds_pert = [(perturbed[k], perturbed[k]) for k in range(len(perturbed))]
        bounds_pert += [(proc_min[k], proc_max[k]) for k in range(6)]

        stage_local = OptimizationProgress(500, progress_callback, f"DE{seed}局部{i+1}")

        result_local = differential_evolution(
            objective_stage2,
            bounds=bounds_pert,
            args=(target_L, target_a, target_b, additive_indices),
            seed=seed + i * 100,
            maxiter=500,
            popsize=5,
            tol=1e-6,
            polish=True,
            mutation=(0.3, 1.5),
            recombination=0.8,
            callback=stage_local.update
        )

        params_local = np.zeros(60)
        params_local[0] = result_local.x[0]
        params_local[1] = result_local.x[1]
        params_local[2] = result_local.x[2]
        for j, idx in enumerate(additive_indices[:5]):
            params_local[idx] = result_local.x[3 + j]
        params_local[54] = result_local.x[8]
        params_local[55] = result_local.x[9]
        params_local[56] = result_local.x[10]
        params_local[57] = result_local.x[11]
        params_local[58] = result_local.x[12]
        params_local[59] = result_local.x[13]

        pred_local = _rf_model.predict(params_local.reshape(1, -1))[0]
        results.append({
            'params': params_local.tolist(),
            'pred_L': pred_local[0], 'pred_a': pred_local[1], 'pred_b': pred_local[2],
            'mse': result_local.fun,
            'cluster_id': cid,
            'top5_additives': additive_indices[:5],
            'valid': True
        })

    return results

def optimize_multiple(target_L, target_a, target_b, n=5, progress_callback=None):
    """2次DE优化，提取多个解凑够5个"""
    initialize()
    clusters = _get_cluster_for_Lab(target_L, target_a, target_b, top_k=2)

    cid, dist = clusters[0]
    additive_indices = _get_additive_indices()
    if len(additive_indices) < 5:
        return []

    # DE1: 提取3个解
    results1 = _run_de_optimization(target_L, target_a, target_b, additive_indices, cid, seed=42, progress_callback=progress_callback, n_solutions=3)

    # DE2: 提取2个解
    results2 = _run_de_optimization(target_L, target_a, target_b, additive_indices, cid, seed=123, progress_callback=progress_callback, n_solutions=2)

    # 合并去重排序
    all_results = results1 + results2
    all_results.sort(key=lambda x: x['mse'])
    return all_results[:n]

__all__ = [
    'initialize',
    'get_feature_names',
    'get_color_list',
    'get_color_info',
    'predict_Lab',
    'optimize_multiple'
]
