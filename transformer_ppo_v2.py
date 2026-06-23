"""
Transformer + PPO 真正强化学习训练 (v2)
参考 ALDes-main 的 PPO 架构
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import pandas as pd
import warnings
import joblib
import math
warnings.filterwarnings('ignore')

# ===== 配置 =====
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 模型参数
d_model = 32
n_layers = 8
n_heads = 8
ffn_hidden = 2048
drop_prob = 0.1
max_len = 50

# 优化器参数
init_lr = 5e-5
clip = 1.0
weight_decay = 5e-4

# PPO参数
total_epoch = 100
ppo_epoch = 5
batch_size = 2
clip_coef = 0.2

# 序列参数
MAIN_COMPONENTS = 3
ADDITIVE_COMPONENTS = 5
PROCESS_PARAMS = 5
SEQ_LEN = (MAIN_COMPONENTS + ADDITIVE_COMPONENTS) * 2 + 1 + PROCESS_PARAMS  # 22

# Token配置
VOCAB_SIZE = 100 # 每个token的离散化等级
LAB_TOKEN_DIM = 100  # Lab的离散化等级

# 特殊token索引
BEGIN_IDX = 0
END_IDX = 1
MAIN_BEGIN = 2 # 主成分开始
ADDITIVE_BEGIN = 5  # 添加剂开始
POWER_IDX = 10 # 电源
PROCESS_BEGIN = 11  # 工艺参数开始

# ===== 数据加载 =====
def load_data():
    df = pd.read_excel('实验数据.xlsx', header=1)
    exclude_cols = ['Unnamed: 0', 'Unnamed: 1', 'L', 'a', 'b', 'T']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    numeric_cols = []
    for c in feature_cols:
        try:
            pd.to_numeric(df[c], errors='raise')
            numeric_cols.append(c)
        except:
            pass

    X = df[numeric_cols].fillna(0).astype(float).values
    lab_L = pd.to_numeric(df['L'], errors='coerce').fillna(0).values
    lab_a = pd.to_numeric(df['a'], errors='coerce').fillna(0).values
    lab_b = pd.to_numeric(df['b'], errors='coerce').fillna(0).values
    y = np.column_stack([lab_L, lab_a, lab_b])

    return X, y, numeric_cols

# ===== Token工具 =====
def continuous_to_token(value, vmin, vmax, n_bins):
    value = np.clip(value, vmin, vmax)
    return int((value - vmin) / (vmax - vmin) * (n_bins - 1))

def token_to_continuous(token, vmin, vmax, n_bins):
    return vmin + (token / (n_bins - 1)) * (vmax - vmin)

def lab_to_token(lab, n_bins=100):
    """Lab -> token"""
    L, a, b = lab
    t_L = continuous_to_token(L, 0, 100, n_bins)
    t_a = continuous_to_token(a, -50, 50, n_bins)
    t_b = continuous_to_token(b, -50, 50, n_bins)
    return t_L, t_a, t_b

def tokens_to_lab(tokens, n_bins=100):
    """token -> Lab"""
    L = token_to_continuous(tokens[0], 0, 100, n_bins)
    a = token_to_continuous(tokens[1], -50, 50, n_bins)
    b = token_to_continuous(tokens[2], -50, 50, n_bins)
    return L, a, b

def tokens_to_formula(tokens, feature_cols):
    """22 tokens -> 配方字典 (tokens不包含Lab tokens)"""
    formula = {}
    idx = 0  # 直接从0开始，不跳过任何tokens

    # 主成分 (3对:成分+参数)
    main_names = ['M002', 'M001', 'M003']
    for i in range(3):
        formula[main_names[i]] = token_to_continuous(tokens[idx], 0, 100, VOCAB_SIZE)
        idx += 1
        formula[f'param_{main_names[i]}'] = token_to_continuous(tokens[idx], 0, 100, VOCAB_SIZE)
        idx += 1

    # 添加剂 (5对: 成分+参数)
    for i in range(5):
        formula[f'ADD_{i}'] = token_to_continuous(tokens[idx], 0, 10, VOCAB_SIZE)
        idx += 1
        formula[f'param_ADD{i}'] = token_to_continuous(tokens[idx], 0, 100, VOCAB_SIZE)
        idx += 1

    # 电源
    formula['W'] = token_to_continuous(tokens[idx], 0, 200, VOCAB_SIZE)
    idx += 1

    # 5个独立工艺参数
    process_names = ['模式', '电压', '电流', '占空比', '频率']
    for i in range(5):
        formula[process_names[i]] = token_to_continuous(tokens[idx], 0, 100, VOCAB_SIZE)
        idx += 1

    return formula

def formula_to_features(formula, feature_cols):
    """配方字典 -> 60维特征向量"""
    features = np.zeros(len(feature_cols))
    for i, col in enumerate(feature_cols):
        if col in formula:
            features[i] = formula[col]
    return features

def get_feature_cols():
    """获取模型对应的特征列"""
    import joblib
    feature_names = joblib.load('models/feature_names_new.joblib')
    return list(feature_names)

# =====编码器 =====
class Encoder(nn.Module):
    def __init__(self, d_model, n_head, max_len, ffn_hidden, vocab_size, n_layers, drop_prob):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(max_len, d_model) * 0.1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=ffn_hidden,
            dropout=drop_prob, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)

    def forward(self, src):
        batch, seq_len = src.shape
        x = self.emb(src) + self.pos_emb[:seq_len]
        x = self.transformer(x)
        return x

# ===== 解码器 (关键: Mask机制) =====
class Decoder(nn.Module):
    def __init__(self, d_model, n_head, max_len, ffn_hidden, vocab_size, n_layers, drop_prob, device):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(max_len, d_model) * 0.1)
        self.device = device
        self.temp = 1.0

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=ffn_hidden,
            dropout=drop_prob, batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, n_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def get_mask(self, cur_seq):
        """交替mask: 成分位置只能选成分，参数位置只能选参数"""
        batch, seq_len = cur_seq.shape
        mask = torch.zeros(batch, VOCAB_SIZE, dtype=torch.float32, device=self.device)

        for i in range(batch):
            if seq_len == 0:
                # 第一个位置: Lab tokens (输入条件)
                mask[i, :] = 1 # 无mask
                continue

            last_action = cur_seq[i, -1].item()

            # 已生成的位置来判断下一步该选什么
            # seq: [Lab1, Lab2, Lab3, tok1, tok2, ...]
            # 位置3,6,9,... -> 成分 (MAIN_BEGIN + n)
            # 位置4,7,10,... -> 参数 (PROCESS_BEGIN + n)
            # 位置5,8,11,... -> 参数 (PROCESS_BEGIN + n)
            generated_len = seq_len - 3  # 去掉Lab tokens

            if generated_len % 3 == 0:
                # 成分位置 -> 主成分或添加剂
                mask[i, MAIN_BEGIN:MAIN_BEGIN + 3] = 1  # 主成分
                mask[i, ADDITIVE_BEGIN:ADDITIVE_BEGIN + 5] = 1  # 添加剂
            elif generated_len % 3 == 1:
                # 参数位置 -> 工艺参数
                mask[i, PROCESS_BEGIN:PROCESS_BEGIN + 5] = 1
            else:
                # 参数位置 -> 工艺参数
                mask[i, PROCESS_BEGIN:PROCESS_BEGIN + 5] = 1

        return mask

    def forward(self, trg, enc_src, action=None, inference=False):
        """
        trg: [batch, seq_len] 目标序列
        enc_src: [batch, 3, d_model] 编码后的Lab条件
        action: 参考动作 (用于PPO更新)
        """
        batch, seq_len = trg.shape

        # 位置编码
        x = self.emb(trg) + self.pos_emb[:seq_len]

        # 因果mask
        tgt_mask = torch.triu(torch.ones(seq_len, seq_len, device=trg.device), diagonal=1).bool()

        # 解码
        memory = self.transformer(x, enc_src, tgt_mask=tgt_mask)

        # 预测
        output = self.fc_out(memory)

        # 采样 (使用最后一个位置)
        last_output = output[:, -1:, :]  # [batch, 1, vocab]

        # 计算概率分布
        probs = torch.softmax(last_output / self.temp, dim=-1)
        # 添加epsilon防止零概率
        probs = probs + 1e-8
        probs = probs / probs.sum(dim=-1, keepdim=True)

        if action is None:
            if inference:
                action_index = probs.argmax(dim=-1).squeeze(-1)
            else:
                action_index = probs.squeeze(1).multinomial(1).squeeze(-1)
        else:
            action_index = action

        # log_prob
        log_p = torch.log(probs.squeeze(1).gather(-1, action_index.unsqueeze(-1)).squeeze(-1))

        return action_index, None, log_p

# ===== Transformer模型 =====
class TransformerPPO(nn.Module):
    def __init__(self, src_pad_idx, trg_pad_idx, d_model, enc_voc_size, dec_voc_size,
                 max_len, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.device = device
        self.encoder = Encoder(d_model, n_head, max_len, ffn_hidden, enc_voc_size, n_layers, drop_prob)
        self.decoder = Decoder(d_model, n_head, max_len, ffn_hidden, dec_voc_size, n_layers, drop_prob, device)

    def forward(self, src, trg, action=None, inference=False):
        """
        src: [batch, 3] Lab tokens (条件输入)
        trg: [batch, seq_len] 已生成序列
        action: 参考动作 (用于PPO)
        """
        # 编码Lab条件
        enc_src = self.encoder(src)

        # 解码生成
        output = self.decoder(trg, enc_src, action, inference)

        return output

    def sample(self, src, max_len=22, inference=True):
        """
        自回归采样生成
        src: [batch, 3] Lab tokens
        returns: [batch, max_len] 生成的token序列, log_probs
        """
        batch = src.shape[0]
        generated = src.clone()  # Start with Lab tokens
        all_log_probs = []

        for step in range(max_len):
            action, _, log_p = self.forward(src, generated, inference=inference)
            generated = torch.cat([generated, action.unsqueeze(-1)], dim=1)
            all_log_probs.append(log_p)

        # Remove Lab tokens, return only generated tokens
        return generated[:,3:], torch.stack(all_log_probs, dim=1)

# ===== 奖励计算 =====
def compute_reward(formula_tokens, target_lab, rf_model, feature_cols):
    """计算奖励: -MSE(pred_lab, target_lab)"""
    formula = tokens_to_formula(formula_tokens, feature_cols)
    features = formula_to_features(formula, feature_cols).reshape(1, -1)
    pred_lab = rf_model.predict(features)[0]

    mse = (pred_lab[0] - target_lab[0])**2 + (pred_lab[1] - target_lab[1])**2 + (pred_lab[2] - target_lab[2])**2
    return -mse

def get_feature_cols():
    """获取模型对应的特征列"""
    import joblib
    feature_names = joblib.load('models/feature_names_new.joblib')
    return list(feature_names)

# ===== PPO训练 =====
def ppo_train(rf_model, X, y, feature_cols):
    """PPO训练循环"""
    print("="*50)
    print("Transformer + PPO 强化学习训练 (v2)")
    print("="*50)

    # 使用模型对应的特征列
    feature_cols = get_feature_cols()
    print(f"使用 {len(feature_cols)} 个特征列")

    # 创建模型
    model = TransformerPPO(
        src_pad_idx=0, trg_pad_idx=0,
        d_model=d_model, enc_voc_size=VOCAB_SIZE, dec_voc_size=VOCAB_SIZE,
        max_len=max_len, ffn_hidden=ffn_hidden, n_head=n_heads,
        n_layers=n_layers, drop_prob=drop_prob, device=device
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {total_params:,}")
    print(f"设备: {device}")

    optimizer = optim.Adam(model.parameters(), lr=init_lr, weight_decay=weight_decay)

    # 训练
    baseline = None
    best_reward = float('-inf')

    for epoch in range(total_epoch):
        # 随机选择batch_size个样本
        indices = np.random.choice(len(X), batch_size, replace=False)

        target_labs = []
        src_tokens = []
        for idx in indices:
            t_l, t_a, t_b = lab_to_token(y[idx])
            src_tokens.append([t_l, t_a, t_b])
            target_labs.append(y[idx])

        src = torch.tensor(src_tokens, dtype=torch.long, device=device)
        target_labs = np.array(target_labs)

        # ===== 采样阶段 =====
        with torch.no_grad():
            # 自回归生成22个tokens
            action, action_log_p = model.sample(src, max_len=22, inference=False)

        # ===== 计算奖励 =====
        rewards = []
        for i in range(batch_size):
            reward = compute_reward(action[i].cpu().numpy(), target_labs[i], rf_model, feature_cols)
            rewards.append(reward)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)

        # ===== Baseline =====
        if baseline is None:
            baseline = rewards.mean()
        else:
            baseline = 0.8 * baseline + 0.2 * rewards.mean()
        baseline = baseline.detach()

        # ===== PPO更新 =====
        for j in range(ppo_epoch):
            _, new_action_log_p = model.sample(src, max_len=22, inference=False)

            # 计算ratio
            logratio = new_action_log_p.sum(1) - action_log_p.sum(1)
            ratio = logratio.exp()

            # PPO loss
            advantage = rewards - baseline
            pg_loss1 = advantage * ratio
            pg_loss2 = advantage * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            loss = torch.max(pg_loss1, pg_loss2).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        # 记录
        mean_reward = rewards.mean().item()
        if mean_reward > best_reward:
            best_reward = mean_reward
            torch.save(model.state_dict(), 'models/transformer_ppo_v2.pt')

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{total_epoch}: "
                  f"reward={mean_reward:.4f}, baseline={baseline.item():.4f}, "
                  f"loss={loss.item():.4f}")

    print(f"\nBest reward: {best_reward:.4f}")
    print("="*50)
    print("训练完成!")
    print("="*50)

    return model

# ===== 测试生成 =====
def test_generation(model, rf_model, feature_cols):
    """测试生成"""
    model.eval()

    # 随机选择一个目标
    idx = np.random.randint(0, len(y))
    target_lab = y[idx]
    t_l, t_a, t_b = lab_to_token(target_lab)

    src = torch.tensor([[t_l, t_a, t_b]], dtype=torch.long, device=device)
    trg = src.clone()

    with torch.no_grad():
        action, _, _ = model(src, trg, inference=True)

    formula = tokens_to_formula(action[0].cpu().numpy(), feature_cols)
    pred_formula = formula_to_features(formula, feature_cols).reshape(1, -1)
    pred_lab = rf_model.predict(pred_formula)[0]

    print(f"\n目标Lab: L={target_lab[0]:.2f}, a={target_lab[1]:.2f}, b={target_lab[2]:.2f}")
    print(f"预测Lab: L={pred_lab[0]:.2f}, a={pred_lab[1]:.2f}, b={pred_lab[2]:.2f}")
    print(f"配方: {formula}")

    return formula

# ===== 主函数 =====
if __name__ == "__main__":
    print("加载数据...")
    X, y, feature_cols = load_data()
    print(f"数据: {X.shape[0]} 样本, {X.shape[1]} 特征")

    print("\n加载RF前向模型...")
    rf_model = joblib.load('models/rf_model_new.joblib')

    print("\n开始PPO训练...")
    model = ppo_train(rf_model, X, y, feature_cols)

    print("\n测试生成...")
    test_generation(model, rf_model, feature_cols)