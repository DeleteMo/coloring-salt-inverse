"""
验证PPO生成配方质量
"""
import torch
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from transformer_ppo_v2 import (
    TransformerPPO, tokens_to_formula, formula_to_features,
    get_feature_cols, load_data, lab_to_token
)

def load_models():
    """加载模型"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # PPO模型
    model = TransformerPPO(
        src_pad_idx=0, trg_pad_idx=0,
        d_model=32, enc_voc_size=100, dec_voc_size=100,
        max_len=50, ffn_hidden=2048, n_head=8,
        n_layers=8, drop_prob=0.1, device=device
    ).to(device)

    model.load_state_dict(torch.load('models/transformer_ppo_v2.pt', map_location=device))
    model.eval()
    print(f"PPO模型已加载 (device: {device})")

    # RF前向模型
    rf_model = joblib.load('models/rf_model_new.joblib')
    print(f"RF模型已加载 (n_features: {rf_model.n_features_in_})")

    return model, rf_model, device

def generate_formula(model, target_lab, feature_cols, device):
    """使用PPO模型生成配方"""
    # Lab -> tokens
    t_l, t_a, t_b = lab_to_token(target_lab)
    src = torch.tensor([[t_l, t_a, t_b]], dtype=torch.long, device=device)

    # 自回归生成
    with torch.no_grad():
        action, _ = model.sample(src, max_len=22, inference=True)

    # tokens -> 配方
    formula = tokens_to_formula(action[0].cpu().numpy(), feature_cols)
    features = formula_to_features(formula, feature_cols).reshape(1, -1)

    return formula, features

def evaluate_quality(model, rf_model, X, y, feature_cols, device, n_samples=20):
    """评估生成配方质量"""
    # 随机采样
    indices = np.random.choice(len(y), n_samples, replace=False)

    results = {
        'target_L': [], 'target_a': [], 'target_b': [],
        'pred_L': [], 'pred_a': [], 'pred_b': [],
        'MSE': [], 'L_error': [], 'a_error': [], 'b_error': []
    }

    print(f"\n随机采样 {n_samples} 个样本进行评估...")
    print("="*60)

    for i, idx in enumerate(indices):
        target_lab = y[idx]
        formula, features = generate_formula(model, target_lab, feature_cols, device)
        pred_lab = rf_model.predict(features)[0]

        # 计算误差
        mse = (pred_lab[0] - target_lab[0])**2 + \
              (pred_lab[1] - target_lab[1])**2 + \
              (pred_lab[2] - target_lab[2])**2

        results['target_L'].append(target_lab[0])
        results['target_a'].append(target_lab[1])
        results['target_b'].append(target_lab[2])
        results['pred_L'].append(pred_lab[0])
        results['pred_a'].append(pred_lab[1])
        results['pred_b'].append(pred_lab[2])
        results['MSE'].append(mse)
        results['L_error'].append(abs(pred_lab[0] - target_lab[0]))
        results['a_error'].append(abs(pred_lab[1] - target_lab[1]))
        results['b_error'].append(abs(pred_lab[2] - target_lab[2]))

        print(f"[{i+1:2d}] 目标: L={target_lab[0]:.2f}, a={target_lab[1]:.2f}, b={target_lab[2]:.2f} | "
              f"预测: L={pred_lab[0]:.2f}, a={pred_lab[1]:.2f}, b={pred_lab[2]:.2f} | "
              f"MSE={mse:.2f}")

    # 统计
    print("\n" + "="*60)
    print("质量评估结果")
    print("="*60)

    mse_arr = np.array(results['MSE'])
    l_err = np.array(results['L_error'])
    a_err = np.array(results['a_error'])
    b_err = np.array(results['b_error'])

    print(f"\nMSE统计:")
    print(f"  均值: {mse_arr.mean():.4f}")
    print(f"  标准差: {mse_arr.std():.4f}")
    print(f"  最小值: {mse_arr.min():.4f}")
    print(f"  最大值: {mse_arr.max():.4f}")
    print(f"  中位数: {np.median(mse_arr):.4f}")

    print(f"\nRMSE (综合误差): {np.sqrt(mse_arr.mean()):.4f}")

    print(f"\n各维度误差均值:")
    print(f"  L: {l_err.mean():.4f} ± {l_err.std():.4f}")
    print(f"  a: {a_err.mean():.4f} ± {a_err.std():.4f}")
    print(f"  b: {b_err.mean():.4f} ± {b_err.std():.4f}")

    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # L散点图
    axes[0, 0].scatter(results['target_L'], results['pred_L'], alpha=0.7)
    axes[0, 0].plot([0, 100], [0, 100], 'r--', label='理想')
    axes[0, 0].set_xlabel('目标 L')
    axes[0, 0].set_ylabel('预测 L')
    axes[0, 0].set_title(f'L 预测效果 (R²={np.corrcoef(results["target_L"], results["pred_L"])[0,1]**2:.3f})')
    axes[0, 0].legend()

    # a散点图
    axes[0, 1].scatter(results['target_a'], results['pred_a'], alpha=0.7)
    lim = [-30, 30]
    axes[0, 1].plot(lim, lim, 'r--', label='理想')
    axes[0, 1].set_xlabel('目标 a')
    axes[0, 1].set_ylabel('预测 a')
    axes[0, 1].set_title(f'a 预测效果 (R²={np.corrcoef(results["target_a"], results["pred_a"])[0,1]**2:.3f})')
    axes[0, 1].legend()

    # b散点图
    axes[0, 2].scatter(results['target_b'], results['pred_b'], alpha=0.7)
    axes[0, 2].plot([-50, 50], [-50, 50], 'r--', label='理想')
    axes[0, 2].set_xlabel('目标 b')
    axes[0, 2].set_ylabel('预测 b')
    axes[0, 2].set_title(f'b 预测效果 (R²={np.corrcoef(results["target_b"], results["pred_b"])[0,1]**2:.3f})')
    axes[0, 2].legend()

    # MSE分布
    axes[1, 0].hist(mse_arr, bins=15, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(mse_arr.mean(), color='r', linestyle='--', label=f'均值={mse_arr.mean():.2f}')
    axes[1, 0].set_xlabel('MSE')
    axes[1, 0].set_ylabel('频数')
    axes[1, 0].set_title('MSE分布')
    axes[1, 0].legend()

    # 误差分布
    axes[1, 1].hist(l_err, bins=15, alpha=0.5, label='L')
    axes[1, 1].hist(a_err, bins=15, alpha=0.5, label='a')
    axes[1, 1].hist(b_err, bins=15, alpha=0.5, label='b')
    axes[1, 1].set_xlabel('绝对误差')
    axes[1, 1].set_ylabel('频数')
    axes[1, 1].set_title('各维度误差分布')
    axes[1, 1].legend()

    # 误差条形图
    x = np.arange(3)
    means = [l_err.mean(), a_err.mean(), b_err.mean()]
    stds = [l_err.std(), a_err.std(), b_err.std()]
    axes[1, 2].bar(x, means, yerr=stds, capsize=5, alpha=0.7)
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(['L', 'a', 'b'])
    axes[1, 2].set_ylabel('绝对误差')
    axes[1, 2].set_title('各维度平均误差')

    plt.tight_layout()
    plt.savefig('ppo_quality_test.png', dpi=150)
    print(f"\n图表已保存到 ppo_quality_test.png")

    return results

if __name__ == "__main__":
    print("="*60)
    print("PPO生成配方质量验证")
    print("="*60)

    # 加载数据
    X, y, _ = load_data()
    feature_cols = get_feature_cols()
    print(f"数据: {len(y)} 个样本, {len(feature_cols)} 个特征")

    # 加载模型
    model, rf_model, device = load_models()

    # 评估
    results = evaluate_quality(model, rf_model, X, y, feature_cols, device, n_samples=20)

    print("\n" + "="*60)
    print("验证完成!")
    print("="*60)
