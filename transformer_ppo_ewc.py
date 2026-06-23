"""
Transformer + PPO + EWC 增量学习训练
阶段1: 监督学习预训练
阶段2: EWC + PPO微调
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import warnings
import joblib
warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ===== 配置 =====
d_model = 32
n_layers = 8
n_heads = 8
ffn_hidden = 2048
drop_prob = 0.1
max_len = 50
VOCAB_SIZE = 100
init_lr = 5e-5
clip = 1.0

# 训练参数
PRETRAIN_EPOCHS = 50
PPO_EPOCHS = 100
ppo_epoch = 5
batch_size = 4
clip_coef = 0.2
ewc_lambda = 0.1  # EWC权重

SEQ_LEN = 22  # 生成的token数量

# ===== EWC类 =====
class EWC:
    def __init__(self, model):
        self.model = model
        self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = None

        for n, p in self.params.items():
            self._means[n] = p.detach().clone()

    def update_diag_fisher(self, model):
        """更新Fisher信息矩阵"""
        if self._precision_matrices is None:
            self._precision_matrices = {n: torch.zeros_like(p) for n, p in self.params.items()}

        for n, p in model.named_parameters():
            if p.grad is not None:
                self._precision_matrices[n] += p.grad.data ** 2

    def penalty(self, model, lambda_ewc=0.1):
        """计算EWC损失"""
        if self._precision_matrices is None:
            return torch.tensor(0.0, device=device)

        loss = 0
        for n, p in model.named_parameters():
            if n in self._precision_matrices:
                # 确保参与梯度计算
                diff = p - self._means[n]
                loss += (self._precision_matrices[n].detach() * diff ** 2).sum()
        return loss * lambda_ewc

# ===== 数据加载 =====
def load_data():
    # 使用模型对应的特征列
    import joblib
    feature_names = list(joblib.load('models/feature_names_new.joblib'))

    df = pd.read_excel('实验数据.xlsx', header=1)

    # 只保留数值型的特征列
    X_data = {}
    for col in feature_names:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors='coerce')
            X_data[col] = vals.fillna(0)
    X = pd.DataFrame(X_data)[feature_names].astype(float).values

    lab_L = pd.to_numeric(df['L'], errors='coerce').fillna(0).values
    lab_a = pd.to_numeric(df['a'], errors='coerce').fillna(0).values
    lab_b = pd.to_numeric(df['b'], errors='coerce').fillna(0).values
    y = np.column_stack([lab_L, lab_a, lab_b])

    return X, y, feature_names

def continuous_to_token(value, vmin, vmax, n_bins=100):
    value = np.clip(value, vmin, vmax)
    return int((value - vmin) / (vmax - vmin) * (n_bins - 1))

def token_to_continuous(token, vmin, vmax, n_bins=100):
    return vmin + (token / (n_bins - 1)) * (vmax - vmin)

def lab_to_token(lab, n_bins=100):
    L, a, b = lab
    return (
        continuous_to_token(L, 0, 100, n_bins),
        continuous_to_token(a, -50, 50, n_bins),
        continuous_to_token(b, -50, 50, n_bins)
    )

def features_to_tokens(features, n_bins=100):
    """60维特征 -> 22 tokens"""
    tokens = []

    # 主成分 (3)
    for i in range(3):
        tokens.append(continuous_to_token(features[i], 0, 100, n_bins))

    # 添加剂 (5) - indices 3-7
    for i in range(5):
        tokens.append(continuous_to_token(features[3 + i], 0, 10, n_bins))

    # 电源 (W) - index 48
    tokens.append(continuous_to_token(features[48], 0, 200, n_bins))

    # 工艺参数 (5) - indices 54-58
    for i in range(5):
        tokens.append(continuous_to_token(features[54 + i], 0, 100, n_bins))

    return tokens[:22]  # 确保22个

def tokens_to_formula(tokens, feature_cols):
    """22 tokens -> 60维特征"""
    features = np.zeros(len(feature_cols))
    idx = 0

    # 主成分 (3) -> M002, M001, M003
    for i in range(3):
        features[i] = token_to_continuous(tokens[idx], 0, 100)
        idx += 1

    # 添加剂 (5) -> indices 3-7
    for i in range(5):
        features[3 + i] = token_to_continuous(tokens[idx], 0, 10)
        idx += 1

    # 电源 (W) -> index 48 in feature_cols
    features[48] = token_to_continuous(tokens[idx], 0, 200)
    idx += 1

    # 工艺参数 (5) -> indices 54-58 (模式, 电压, 电流, 占空比, 频率)
    for i in range(5):
        features[54 + i] = token_to_continuous(tokens[idx], 0, 100)
        idx += 1

    return features

# ===== Transformer模型 =====
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerEWc(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=32, nhead=8,
                 num_layers=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.temp = 1.0

        # Embeddings
        self.src_emb = nn.Embedding(vocab_size, d_model)
        self.trg_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        # Output
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, trg, tgt_mask=None):
        """
        src: [batch, 3] Lab tokens
        trg: [batch, seq_len] 目标序列
        """
        # 编码Lab
        src_emb = self.src_emb(src)
        src_emb = self.pos_enc(src_emb)
        memory = self.encoder(src_emb)

        # 解码
        trg_emb = self.trg_emb(trg)
        trg_emb = self.pos_enc(trg_emb)

        if tgt_mask is None:
            seq_len = trg.size(1)
            tgt_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(trg.device)

        output = self.decoder(trg_emb, memory, tgt_mask=tgt_mask)
        logits = self.fc_out(output)

        return logits

    def sample(self, src, max_len=22):
        """自回归采样"""
        batch = src.size(0)
        generated = src.clone()

        for _ in range(max_len):
            logits = self.forward(src, generated)
            probs = torch.softmax(logits[:, -1:, :] / self.temp, dim=-1)
            action = probs.squeeze(1).multinomial(1)
            generated = torch.cat([generated, action], dim=1)

        return generated[:, 3:]  # 去掉Lab tokens

# ===== 训练函数 =====
def pretrain_supervised(model, X, y, feature_cols, epochs=50):
    """阶段1: 监督学习预训练"""
    print("\n" + "="*50)
    print("阶段1: 监督学习预训练")
    print("="*50)

    optimizer = optim.Adam(model.parameters(), lr=init_lr)
    criterion = nn.CrossEntropyLoss()

    losses = []

    for epoch in range(epochs):
        # 随机采样
        indices = np.random.choice(len(X), batch_size, replace=True)
        batch_X = X[indices]
        batch_y = y[indices]

        # 准备数据
        src_tokens = []
        trg_tokens = []
        for i in range(batch_size):
            t_l, t_a, t_b = lab_to_token(batch_y[i])
            src_tokens.append([t_l, t_a, t_b])
            trg_tokens.append(features_to_tokens(batch_X[i]))

        src = torch.tensor(src_tokens, dtype=torch.long, device=device)
        trg = torch.tensor(trg_tokens, dtype=torch.long, device=device)

        # Teacher forcing: 输入trg[:-1], 预测trg[1:]
        input_trg = trg[:, :-1]
        target_trg = trg[:, 1:]

        # 前向
        logits = model(src, input_trg)  # [batch, seq_len-1, vocab]

        # Loss
        loss = criterion(logits.reshape(-1, VOCAB_SIZE), target_trg.reshape(-1))

        # 反向
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: loss={loss.item():.4f}")

    return losses

def ppo_with_ewc(model, X, y, feature_cols, rf_model, ewc, epochs=100):
    """阶段2: EWC + PPO微调"""
    print("\n" + "="*50)
    print("阶段2: EWC + PPO微调")
    print("="*50)

    optimizer = optim.Adam(model.parameters(), lr=init_lr * 0.1)  # 降低学习率

    baseline = None
    best_reward = float('-inf')

    for epoch in range(epochs):
        # 采样
        indices = np.random.choice(len(X), batch_size, replace=True)
        batch_y = y[indices]

        src_tokens = []
        for i in range(batch_size):
            t_l, t_a, t_b = lab_to_token(batch_y[i])
            src_tokens.append([t_l, t_a, t_b])
        src = torch.tensor(src_tokens, dtype=torch.long, device=device)

        # 生成配方
        with torch.no_grad():
            action = model.sample(src, max_len=22)  # [batch, 22]

        # 计算奖励
        rewards = []
        for i in range(batch_size):
            formula_features = tokens_to_formula(action[i].cpu().numpy(), feature_cols)
            pred_lab = rf_model.predict(formula_features.reshape(1, -1))[0]
            mse = (pred_lab[0] - batch_y[i][0])**2 + \
                  (pred_lab[1] - batch_y[i][1])**2 + \
                  (pred_lab[2] - batch_y[i][2])**2
            rewards.append(-mse)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)

        # Baseline
        if baseline is None:
            baseline = rewards.mean()
        else:
            baseline = 0.8 * baseline + 0.2 * rewards.mean()
        baseline = baseline.detach()

        # PPO + EWC更新
        for _ in range(ppo_epoch):
            # 重新采样
            action_new = model.sample(src, max_len=22)

            # PPO损失 (交叉熵)
            logits = model(src, action_new[:, :-1])
            loss_ppo = nn.functional.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                action_new[:, 1:].reshape(-1)
            )

            # EWC正则化损失
            loss_ewc = ewc.penalty(model, lambda_ewc=ewc_lambda)

            total_loss = loss_ppo + loss_ewc

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            # 更新Fisher信息
            ewc.update_diag_fisher(model)

        # 记录
        mean_reward = rewards.mean().item()
        if mean_reward > best_reward:
            best_reward = mean_reward
            torch.save(model.state_dict(), 'models/transformer_ppo_ewc.pt')

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: reward={mean_reward:.4f}, "
                  f"baseline={baseline.item():.4f}, ewc={loss_ewc.item():.4f}")

    return best_reward

def main():
    print("="*50)
    print("Transformer + PPO + EWC 训练")
    print("="*50)

    # 加载数据
    print("\n加载数据...")
    X, y, feature_cols = load_data()
    print(f"数据: {X.shape[0]} 样本, {len(feature_cols)} 特征")

    # 加载RF模型
    print("\n加载RF模型...")
    rf_model = joblib.load('models/rf_model_new.joblib')

    # 创建模型
    print("\n创建模型...")
    model = TransformerEWc(
        vocab_size=VOCAB_SIZE, d_model=d_model, nhead=n_heads,
        num_layers=n_layers, dim_feedforward=ffn_hidden, dropout=drop_prob
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {total_params:,}")

    # 阶段1: 监督学习预训练
    pretrain_losses = pretrain_supervised(model, X, y, feature_cols, epochs=PRETRAIN_EPOCHS)

    # 保存预训练模型
    torch.save(model.state_dict(), 'models/transformer_ppo_ewc_pretrain.pt')
    print("\n预训练模型已保存: models/transformer_ppo_ewc_pretrain.pt")

    # 阶段2: EWC + PPO微调
    ewc = EWC(model)
    best_reward = ppo_with_ewc(model, X, y, feature_cols, rf_model, ewc, epochs=PPO_EPOCHS)

    print("\n" + "="*50)
    print(f"训练完成! Best reward: {best_reward:.4f}")
    print("="*50)

    # 测试生成
    print("\n测试生成...")
    test_idx = np.random.randint(0, len(y))
    target_lab = y[test_idx]
    t_l, t_a, t_b = lab_to_token(target_lab)
    src = torch.tensor([[t_l, t_a, t_b]], dtype=torch.long, device=device)

    with torch.no_grad():
        action = model.sample(src, max_len=22)

    formula_features = tokens_to_formula(action[0].cpu().numpy(), feature_cols)
    pred_lab = rf_model.predict(formula_features.reshape(1, -1))[0]
    mse = (pred_lab[0] - target_lab[0])**2 + \
          (pred_lab[1] - target_lab[1])**2 + \
          (pred_lab[2] - target_lab[2])**2

    print(f"目标Lab: L={target_lab[0]:.2f}, a={target_lab[1]:.2f}, b={target_lab[2]:.2f}")
    print(f"预测Lab: L={pred_lab[0]:.2f}, a={pred_lab[1]:.2f}, b={pred_lab[2]:.2f}")
    print(f"MSE: {mse:.4f}")

if __name__ == "__main__":
    main()