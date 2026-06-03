"""
Lab聚类逆向优化测试 - 使用新数据训练的RF模型
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import inverse_lab_cluster_new as ilc

def load_lab_data():
    """从实验数据.xlsx加载L,a,b"""
    df = pd.read_excel(r'实验数据.xlsx', header=1)
    y_L = pd.to_numeric(df['L'].values, errors='coerce')
    y_a = pd.to_numeric(df['a'].values, errors='coerce')
    y_b = pd.to_numeric(df['b'].values, errors='coerce')
    mask = ~(np.isnan(y_L) | np.isnan(y_a) | np.isnan(y_b))
    return y_L[mask], y_a[mask], y_b[mask]

def main():
    ilc.initialize()

    cluster_ids = [0, 1, 2, 3, 4]

    print("="*90)
    print("Lab聚类逆向优化(新模型)测试")
    print("="*90)
    print(f"\n{'Cluster':<8} {'Type':<8} {'TargetL':<8} {'Targeta':<8} {'Targetb':<8} "
          f"{'PredL':<8} {'Preda':<8} {'Predb':<8} "
          f"{'L_err%':<10} {'a_err%':<10} {'b_err%':<10} {'MSE':<12}")
    print("-"*120)

    L_all, a_all, b_all = load_lab_data()

    for cid in cluster_ids:
        if cid not in ilc._cluster_prior:
            continue

        info = ilc._cluster_prior[cid]
        center_L = info['center_L']
        center_a = info['center_a']
        center_b = info['center_b']

        # 找边缘点
        distances = (L_all - center_L)**2 + (a_all - center_a)**2 + (b_all - center_b)**2
        edge_idx = np.argmax(distances)
        edge_L = L_all[edge_idx]
        edge_a = a_all[edge_idx]
        edge_b = b_all[edge_idx]

        for label, tL, ta, tb in [
            ("Center", center_L, center_a, center_b),
            ("Edge", edge_L, edge_a, edge_b)
        ]:
            try:
                result = ilc.optimize_cluster_new(cid, tL, ta, tb, seed=42)
                if result is None:
                    print(f"{cid:<8} {label:<8} Error: None result")
                    continue

                pred_L = result['pred_L']
                pred_a = result['pred_a']
                pred_b = result['pred_b']

                errL = abs(pred_L - tL)
                erra = abs(pred_a - ta)
                errb = abs(pred_b - tb)
                mse = errL**2 + erra**2 + errb**2

                pct_L = errL / abs(tL) * 100 if tL != 0 else 0
                pct_a = erra / abs(ta) * 100 if ta != 0 else 0
                pct_b = errb / abs(tb) * 100 if tb != 0 else 0

                print(f"{cid:<8} {label:<8} {tL:<8.1f} {ta:<8.1f} {tb:<8.1f} "
                      f"{pred_L:<8.1f} {pred_a:<8.1f} {pred_b:<8.1f} "
                      f"{pct_L:<10.1f} {pct_a:<10.1f} {pct_b:<10.1f} {mse:<12.2f}")
            except Exception as e:
                print(f"{cid:<8} {label:<8} Error: {e}")

    print("\n完成")

if __name__ == "__main__":
    main()