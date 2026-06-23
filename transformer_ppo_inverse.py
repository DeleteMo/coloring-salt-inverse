"""
Transformer + PPO 自回归逆向配方模型
序列结构: [3主添加剂] → [5添加剂] → [电源] → [5工艺参数]
Mask规则: 配方和参数交替生成
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings
import os
import joblib

warnings.filterwarnings('ignore')

# ===== 配置 =====
# 交替结构: 成分+参数 配对交替生成
# 3主剂×2 + 5添加剂×2 + 1电源×1 + 5工艺参数×1 = 22 tokens
MAIN_COMPONENTS = 3 # M002, M001, M003
ADDITIVE_COMPONENTS = 5 # 5个添加剂
PROCESS_PARAMS = 5 # 5个工艺参数 (模式,电压,电流,占空比,频率)

# 序列长度: (3+5)×2 + 1 + 5 = 22
TOTAL_TOKENS = (MAIN_COMPONENTS + ADDITIVE_COMPONENTS) * 2 + 1 + PROCESS_PARAMS  # 22
# 交替顺序: [成分1,参数1, 成分2,参数2, ...,成分8,参数8, 成分9,参数9, 成分10, 参数11-15(5个工艺)]
# 即: [M002,模式, M001,电压, M003,电流, ADD1,占空比, ADD2,频率, ADD3,?, ADD4,?, ADD5,?, W,?,?,?,?] = 22

# 特征名称映射
FEATURE_COLS = ['M002', 'M001', 'M003', 'C006', 'S003', 'S002', 'C042', 'S005', 'C007',
                'C013', 'C005', 'C002', 'C011', 'C010', 'C012', 'C005.1', 'C009', 'C003',
                'C004', 'C025', 'CC', 'C049', 'C050', 'C048', 'C059', 'C061', 'C062',
                'C064', 'C063', 'C066', 'C060', 'C044', 'C036', 'C028', 'C030', 'C038',
                'C039', 'C040', 'C071', 'C077', 'C042.1', 'C046', 'C043', 'C085', 'C083',
                'C027', 'C046.1', 'C047', 'W', 'V', 'Y', 'CD', 'A', 'J', 'ģʽ',
                '��ѹ', '����', 'ղձ�', 'Ƶ��', '���ڣ��֣�']

PROCESS_COLS = ['ģʽ', '��ѹ', '����','ղձ�', 'Ƶ��']  # 模式,电压,电流,占空比,频率
LAB_COLS = ['L', 'a', 'b', 'T']

# Token类型定义
TOKEN_TYPES = {
    'main': list(range(0, 3)),           # 0-2: M002, M001, M003
    'additive': list(range(3, 8)),        # 3-7: 5个添加剂
    'power': [8],                         # 8: 电源
    'process': list(range(9, 14)),        # 9-13: 5个工艺参数
    'lab': list(range(14, 17)),           # 14-16: L,a,b (条件)
    'pad': [17],                          # 17: padding
    'mask_token': [18],                   # 18: special mask token
}

VOCAB_SIZE = 128 # 离散化token ID范围 0-99，需要足够大

# ===== 数据加载 =====
def load_data():
    """加载并预处理数据"""
    df = pd.read_excel('实验数据.xlsx', header=1)

    # 明确指定列 (排除无用列)
    exclude_cols = ['Unnamed: 0', 'Unnamed: 1'] + LAB_COLS
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # 只保留数值列
    numeric_cols = []
    for c in feature_cols:
        try:
            pd.to_numeric(df[c], errors='raise')
            numeric_cols.append(c)
        except:
            pass

    feature_cols = numeric_cols
    print(f"  使用 {len(feature_cols)} 个特征列")

       # 提取特征和标签
    X = df[feature_cols].fillna(0).astype(float).values
    # Lab列单独处理，只取L,a,b (强制转数值，非法值变NaN再填0)
    lab_data = pd.to_numeric(df['L'], errors='coerce').fillna(0).values
    lab_a = pd.to_numeric(df['a'], errors='coerce').fillna(0).values
    lab_b = pd.to_numeric(df['b'], errors='coerce').fillna(0).values
    y = np.column_stack([lab_data, lab_a, lab_b])

    return X, y, feature_cols

def get_additive_indices(feature_cols):
    """获取添加剂索引"""
    main_idx = [feature_cols.index(c) for c in ['M002', 'M001', 'M003'] if c in feature_cols]
    process_idx = [feature_cols.index(c) for c in PROCESS_COLS if c in feature_cols]
    additive_idx = [i for i, c in enumerate(feature_cols) if i not in main_idx and i not in process_idx]
    return main_idx, additive_idx, process_idx

# ===== Token生成 =====
class TokenGenerator:
    """生成自回归序列tokens"""
    def __init__(self, feature_cols):
        self.feature_cols = feature_cols
        self.main_idx, self.additive_idx, self.process_idx = get_additive_indices(feature_cols)

        # 按使用频率选择top5添加剂
        self.top_additive_idx = self.additive_idx[:5]

    def features_to_tokens(self, features, target_lab):
        """
        将特征转换为token序列 (22 tokens)
        交替结构: [成分1,参数1, 成分2,参数2, ...,成分8,参数8, 成分9,参数9, 成分10, 参数11-15]
        - 成分: M002,M001,M003,ADD1-5,W
        - 参数: 模式,电压,电流,占空比,频率
        """
        tokens = []
        token_types = []

        # 工艺参数值
        proc_vals = []
        for pcol in PROCESS_COLS:
            if pcol in self.feature_cols:
                idx = self.feature_cols.index(pcol)
                proc_vals.append(features[idx])
            else:
                proc_vals.append(0.0)

        # 交替生成: 成分+参数 配对
        # 主成分 (3个)
        for i, idx in enumerate(self.main_idx):
            # 成分token
            val = features[idx]
            token = self._continuous_to_token(val, vmin=0, vmax=100, n_bins=50)
            tokens.append(token)
            token_types.append('main')
            # 参数token (模式,电压,电流,占空比,频率)
            proc_val = proc_vals[i]
            token = self._continuous_to_token(proc_val, vmin=0, vmax=100, n_bins=50)
            tokens.append(token)
            token_types.append('process')

        # 添加剂 (5个) - 只用前5个
        for i, idx in enumerate(self.top_additive_idx[:5]):
            # 成分token
            val = features[idx]
            token = self._continuous_to_token(val, vmin=0, vmax=10, n_bins=50)
            tokens.append(token)
            token_types.append('additive')
            # 参数token (循环使用5个工艺参数)
            proc_val = proc_vals[i]
            token = self._continuous_to_token(proc_val, vmin=0, vmax=100, n_bins=50)
            tokens.append(token)
            token_types.append('process')

        # 电源 (1个) - 只有成分token
        power_idx = self.feature_cols.index('W') if 'W' in self.feature_cols else -1
        if power_idx >= 0:
            power_val = features[power_idx]
        else:
            power_val = 0.0
        token = self._continuous_to_token(power_val, vmin=0, vmax=200, n_bins=50)
        tokens.append(token)
        token_types.append('power')

        # 5个独立工艺参数 tokens (剩余参数)
        for proc_val in proc_vals:
            token = self._continuous_to_token(proc_val, vmin=0, vmax=100, n_bins=50)
            tokens.append(token)
            token_types.append('process')

        # 目标Lab tokens (追加到序列末尾)
        for val in target_lab[:3]:
            token = self._continuous_to_token(val, vmin=-20, vmax=100, n_bins=100)
            tokens.append(token)
            token_types.append('lab')

        return tokens, token_types

    def _continuous_to_token(self, value, vmin, vmax, n_bins):
        """将连续值离散化为token ID"""
        value = np.clip(value, vmin, vmax)
        token = int((value - vmin) / (vmax - vmin) * (n_bins - 1))
        return min(token, n_bins - 1)

    def token_to_continuous(self, token, vmin, vmax, n_bins):
        """Token ID转回连续值"""
        value = vmin + (token / (n_bins - 1)) * (vmax - vmin)
        return value

# ===== Dataset =====
class InverseDataset(Dataset):
    def __init__(self, X, y, feature_cols, n_bins=50):
        self.token_gen = TokenGenerator(feature_cols)
        self.n_bins = n_bins

        # 归一化
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.X_scaled = self.scaler_X.fit_transform(X)
        self.y_scaled = self.scaler_y.fit_transform(y)

        # 只用L,a,b (3维)
        y_lab = y
        y_lab_scaled = self.scaler_y.fit_transform(y_lab)

        self.samples = []
        for i in range(len(X)):
            features = self.X_scaled[i]
            target_lab = y_lab_scaled[i]
            tokens, token_types = self.token_gen.features_to_tokens(features, target_lab)
            self.samples.append({
                'tokens': torch.tensor(tokens, dtype=torch.long),
                'token_types': token_types,
                'target_lab': torch.tensor(y_lab[i], dtype=torch.float32),  # 原始值用于损失计算
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

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

class TransformerInverse(nn.Module):
    """
    Transformer自回归逆向模型
    输入: [target_Lab tokens] + [已生成的formula tokens]
    输出: 下一个token的概率分布
    """
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=128, nhead=8,
                 num_layers=4, dim_feedforward=512, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        # Embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.type_embedding = nn.Embedding(5, d_model)  # 5种token类型

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 解码器 (用于自回归生成)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 条件编码 (Lab -> d_model)
        self.condition_proj = nn.Linear(3, d_model)

        # 输出头
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, vocab_size)
        )

        # 预测头 (预测连续值)
        self.value_head = nn.Linear(d_model, 1)

    def forward(self, tgt, tgt_key_padding_mask=None, condition=None):
        """
        前向传播
        tgt: [batch, seq_len] token序列
        condition: [batch, 4] 目标Lab值
        """
        batch_size, seq_len = tgt.shape

        # Token嵌入
        x = self.token_embedding(tgt)  # [batch, seq_len, d_model]

        # 位置编码
        x = self.pos_encoder(x)

        # 条件编码
        if condition is not None:
            cond = self.condition_proj(condition).unsqueeze(1)  # [batch, 1, d_model]
            x = x + cond

        # 编码
        memory = self.transformer_encoder(x)

        # 解码 (因果mask)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=tgt.device), diagonal=1
        ).bool()

        x = self.transformer_decoder(x, memory, tgt_mask=causal_mask,
                                     tgt_key_padding_mask=tgt_key_padding_mask)

        # 输出
        logits = self.fc_out(x)
        values = self.value_head(x)

        return logits, values

# ===== PPO训练 =====
class PPOTrainer:
    def __init__(self, model, lr=3e-4, clip_ratio=0.2, entropy_coef=0.01):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train_step(self, batch, old_log_probs=None, old_values=None):
        """单步PPO训练"""
        tokens = batch['tokens'].to(self.device)
        target_lab = batch['target_lab'].to(self.device)

        # 生成掩码 (因果掩码)
        seq_len = tokens.size(1)
        tgt_key_padding_mask = torch.zeros_like(tokens, dtype=torch.bool)

        # 前向传播
        logits, values = self.model(tokens, tgt_key_padding_mask, target_lab)

        # 计算策略损失
        log_probs = torch.log_softmax(logits, dim=-1)

        # 获取实际token的log_prob
        action_log_probs = log_probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)

        # 计算熵奖励
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(-1).mean()

        # 计算价值损失
        value_loss = nn.MSELoss()(values.squeeze(-1), torch.zeros_like(values.squeeze(-1)))

        # 策略损失 (简化版: 交叉熵 +熵奖励)
        policy_loss = -action_log_probs.mean()

        total_loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'total_loss': total_loss.item()
        }

# ===== 自回归生成 =====
class AutoregressiveGenerator:
    """自回归生成配方"""
    def __init__(self, model, token_gen, device):
        self.model = model
        self.token_gen = token_gen
        self.device = device
        self.model.eval()

    def generate(self, target_lab, max_len=22):
        """
        给定目标Lab，自回归生成22个token
        交替规则: [成分1,参数1, 成分2,参数2, ...]
        """
        self.model.eval()

        # 目标Lab tokenized (L, a, b)
        target_tokens = []
        for val in target_lab[:3]:
            token = self.token_gen._continuous_to_token(val, vmin=-20, vmax=100, n_bins=100)
            target_tokens.append(token)

        # 生成序列 (22 tokens: 8对成分+参数 + 1电源 + 5工艺参数)
        generated = target_tokens.copy()

        with torch.no_grad():
            for step in range(max_len):
                seq = torch.tensor([generated], dtype=torch.long).to(self.device)
                condition = torch.tensor([target_lab], dtype=torch.float32).to(self.device)

                logits, _ = self.model(seq, condition=condition)
                logits = logits[0, -1]  # 最后一个位置

                # 采样
                probs = torch.softmax(logits, dim=-1)
                token = torch.multinomial(probs, 1).item()

                generated.append(token)

        return generated[3:]  # 去掉Lab tokens (3个)

    def tokens_to_formula(self, tokens):
        """将tokens转换回配方值 (22 tokens -> 配方+参数)"""
        formula = {}
        idx = 0

        # 主成分 (3个, 每个成分+参数)
        main_names = ['M002', 'M001', 'M003']
        for i, name in enumerate(main_names):
            formula[name] = self.token_gen.token_to_continuous(tokens[idx], 0, 100, 50)
            idx += 1
            formula[f'param_{name}'] = self.token_gen.token_to_continuous(tokens[idx], 0, 100, 50)
            idx += 1

        # 添加剂 (5个, 每个成分+参数)
        for i in range(5):
            formula[f'ADD_{i}'] = self.token_gen.token_to_continuous(tokens[idx], 0, 10, 50)
            idx += 1
            formula[f'param_ADD{i}'] = self.token_gen.token_to_continuous(tokens[idx], 0, 100, 50)
            idx += 1

        # 电源 (1个)
        formula['W'] = self.token_gen.token_to_continuous(tokens[idx], 0, 200, 50)
        idx += 1

        # 剩余5个工艺参数
        process_names = ['模式', '电压', '电流', '占空比', '频率']
        for i, name in enumerate(process_names):
            formula[name] = self.token_gen.token_to_continuous(tokens[idx], 0, 100, 50)
            idx += 1

        return formula

# ===== 训练流程 =====
def train():
    """完整训练流程"""
    print("="*50)
    print("Transformer + PPO 逆向配方模型训练")
    print("="*50)

    # 1. 加载数据
    print("\n[1/6] 加载数据...")
    X, y, feature_cols = load_data()
    print(f"  数据量: {X.shape[0]} 样本, {X.shape[1]} 特征")

    # 2. 创建数据集
    print("\n[2/6] 创建数据集...")
    dataset = InverseDataset(X, y, feature_cols)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f"  Token序列长度: {TOTAL_TOKENS}")

    # 3. 创建模型
    print("\n[3/6] 创建模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerInverse(
        vocab_size=VOCAB_SIZE,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数: {total_params:,}")
    print(f"  设备: {device}")

    # 4. PPO训练
    print("\n[4/6] PPO训练...")
    trainer = PPOTrainer(model, lr=3e-4)

    n_epochs = 50
    for epoch in range(n_epochs):
        epoch_losses = {k: 0 for k in ['policy_loss', 'value_loss', 'entropy', 'total_loss']}

        for batch in dataloader:
            losses = trainer.train_step(batch)
            for k, v in losses.items():
                epoch_losses[k] += v

        # 平均
        n_batches = len(dataloader)
        for k in epoch_losses:
            epoch_losses[k] /= n_batches

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: "
                  f"policy_loss={epoch_losses['policy_loss']:.4f}, "
                  f"value_loss={epoch_losses['value_loss']:.4f}, "
                  f"entropy={epoch_losses['entropy']:.4f}")

    # 5. 保存模型
    print("\n[5/6] 保存模型...")
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/transformer_ppo_inverse.pt')
    joblib.dump(dataset.token_gen, 'models/token_generator.joblib')
    joblib.dump(dataset.scaler_X, 'models/scaler_X.joblib')
    joblib.dump(dataset.scaler_y, 'models/scaler_y.joblib')
    print("  模型已保存到 models/transformer_ppo_inverse.pt")

    # 6. 测试生成
    print("\n[6/6] 测试生成...")
    generator = AutoregressiveGenerator(model, dataset.token_gen, device)

    # 测试样本
    test_lab = y[0]
    print(f"  目标Lab: {test_lab}")

    tokens = generator.generate(test_lab)
    formula = generator.tokens_to_formula(tokens)
    print(f"  生成配方: {formula}")

    print("\n" + "="*50)
    print("训练完成!")
    print("="*50)

    return model, dataset

# ===== 推理接口 =====
class InverseModel:
    """推理接口"""
    def __init__(self, model_path='models/transformer_ppo_inverse.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载模型
        self.model = TransformerInverse().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # 加载辅助模块
        self.token_gen = joblib.load('models/token_generator.joblib')
        self.scaler_y = joblib.load('models/scaler_y.joblib')

    def predict(self, target_L, target_a, target_b, temperature=1.0):
        """
        给定目标Lab，生成配方
        """
        # 归一化目标
        target_scaled = self.scaler_y.transform([[target_L, target_a, target_b]])
        target_lab = target_scaled[0].tolist()

        # 生成
        generator = AutoregressiveGenerator(self.model, self.token_gen, self.device)
        tokens = generator.generate(target_lab)
        formula = generator.tokens_to_formula(tokens)

        return formula

# ===== 主函数 =====
if __name__ == "__main__":
    train()