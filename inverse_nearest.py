"""
基于最近邻初始化的逆向优化
输入Lab → 数据中找最接近的Lab → 以其配方为基础微调参数
"""
import pandas as pd, numpy as np, joblib, os
from scipy.optimize import differential_evolution

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_rf, _X, _Lab = None, None, None

def init():
    global _rf, _X, _Lab
    if _rf is not None: return
    _rf = joblib.load(os.path.join(BASE_DIR, 'models', 'rf_model_new.joblib'))
    df = pd.read_excel(os.path.join(BASE_DIR, '实验数据.xlsx'), header=1)
    fc = df.columns[2:62].tolist()
    X = np.array([pd.to_numeric(df[c].values, errors='coerce') for c in fc]).T
    X = np.nan_to_num(X)
    L = pd.to_numeric(df['L'].values, errors='coerce')
    a = pd.to_numeric(df['a'].values, errors='coerce')
    b = pd.to_numeric(df['b'].values, errors='coerce')
    m = ~(np.isnan(L)|np.isnan(a)|np.isnan(b))
    _X, _Lab = X[m], np.column_stack([L[m],a[m],b[m]])
    print(f'Loaded: {len(_X)} samples')

def optimize(tL, ta, tb):
    """基于最近邻的逆向优化"""
    init()

    # 找最近邻
    dists = np.sqrt((_Lab[:,0]-tL)**2+(_Lab[:,1]-ta)**2+(_Lab[:,2]-tb)**2)
    idx = np.argmin(dists)
    base = _X[idx].copy()
    nLab = _Lab[idx]
    print(f'目标 L={tL:.1f} a={ta:.1f} b={tb:.1f} -> 最近邻 L={nLab[0]:.1f} a={nLab[1]:.1f} b={nLab[2]:.1f}')

    # 可变参数: 3主成分 + 非零添加剂 + 5工艺参数
    var_idx = [0, 1, 2]  # M002, M001, M003
    for i in range(3, 55):
        if base[i] > 0.1:
            var_idx.append(i)
    var_idx.extend([55, 56, 57, 58, 59])  # 电压, 电流, 占空比, 频率, 周期
    var_idx = list(dict.fromkeys(var_idx))

    # 在base附近搜索: base*0.5 ~ base*1.5
    bounds = []
    for i in var_idx:
        lo = max(0, base[i] * 0.5)
        hi = base[i] * 1.5 if base[i] > 0 else 10
        bounds.append((lo, hi))

    def obj(x):
        p = base.copy()
        for i, vi in enumerate(var_idx):
            p[vi] = x[i]
        pred = _rf.predict(p.reshape(1, -1))[0]
        return (pred[0]-tL)**2 + (pred[1]-ta)**2 + (pred[2]-tb)**2

    r = differential_evolution(obj, bounds, seed=42, maxiter=300, popsize=10, tol=1e-5, polish=True)

    p = base.copy()
    for i, vi in enumerate(var_idx):
        p[vi] = r.x[i]
    pred = _rf.predict(p.reshape(1, -1))[0]

    errL = abs(pred[0]-tL)/abs(tL)*100
    erra = abs(pred[1]-ta)/abs(ta)*100 if ta!=0 else abs(pred[1]-ta)
    errb = abs(pred[2]-tb)/abs(tb)*100 if tb!=0 else abs(pred[2]-tb)

    return {
        'pred_L': pred[0], 'pred_a': pred[1], 'pred_b': pred[2],
        'L_err': errL, 'a_err': erra, 'b_err': errb,
        'mse': r.fun, 'neighbor_idx': idx,
        'neighbor_L': nLab[0], 'neighbor_a': nLab[1], 'neighbor_b': nLab[2],
        'params': p.tolist(),
    }

if __name__ == '__main__':
    init()
    cp = joblib.load(os.path.join(BASE_DIR, 'cluster_models', 'cluster_prior_formula.joblib'))

    print('='*70)
    print('基于最近邻的逆向优化测试')
    print('='*70)

    total_L = total_a = total_b = passed = n = 0

    for cid in sorted(cp.keys()):
        info = cp[cid]
        tL, ta, tb = info['center_L'], info['center_a'], info['center_b']
        r = optimize(tL, ta, tb)

        total_L += r['L_err']; total_a += r['a_err']; total_b += r['b_err']; n += 1
        ok = 'PASS' if r['L_err']<20 and r['a_err']<20 and r['b_err']<20 else 'FAIL'
        if ok == 'PASS': passed += 1
        print(f'  pred L={r["pred_L"]:.1f} a={r["pred_a"]:.2f} b={r["pred_b"]:.2f} | L_err={r["L_err"]:.1f}% a_err={r["a_err"]:.1f}% b_err={r["b_err"]:.1f}% [{ok}]')

    print(f'\n平均: L={total_L/n:.1f}% a={total_a/n:.1f}% b={total_b/n:.1f}%')
    print(f'达标: {passed}/{n}')
