# 着色盐 L\*a\*b\* 预测与逆向配方工具

基于随机森林的着色盐颜色预测和逆向优化系统，支持 Web 界面和 Ollama 本地 AI 助手。

## 功能

- **正向预测**: 输入配方参数（60维），预测 L\*a\*b\* 颜色值
- **逆向配方**: 输入目标颜色，自动匹配最近邻配方并微调优化（达标率 6/7）
- **AI 助手**: 基于本地 Ollama 模型的自然语言交互（推荐 qwen2.5:7b）

## 快速开始

```bash
# 1. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 2. 安装依赖
pip install pandas numpy scipy scikit-learn joblib streamlit requests openpyxl

# 3. 启动 Web
streamlit run app.py
```

## AI 助手配置（可选）

```bash
# 安装 Ollama
# https://ollama.com/download

# 下载模型
ollama pull qwen2.5:7b

# 启动服务
ollama serve
```

## 项目结构

```
├── app.py                   # Streamlit Web 界面
├── agent.py                 # Ollama Agent 模块
├── inverse_nearest.py       # 最近邻逆向优化
├── inverse_utils.py         # 正向预测工具
├── save_rf_model_new.py     # RF 模型训练脚本
├── visualization_analysis.py # 可视化分析
├── models/                  # 训练好的模型
├── cluster_models/          # 配方聚类
└── 实验数据.xlsx             # 训练数据
```

## 模型性能

| 目标 | RMSE | R² |
|------|------|-----|
| L (亮度) | 3.00 | 0.970 |
| a (红绿) | 0.57 | 0.944 |
| b (黄蓝) | 1.72 | 0.917 |

逆向优化达标率: **6/7** (中心点误差 < 20%)

## 技术栈

- scikit-learn (Random Forest)
- scipy (Differential Evolution)
- Streamlit (Web UI)
- Ollama (本地 AI)
- pandas, numpy, matplotlib