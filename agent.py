"""
本地 Ollama Agent - 集成到Streamlit
提供自然语言交互：颜色查询、配方优化、公式分析
"""
import json, os
import requests
from inverse_utils import initialize as init_forward, get_feature_names, predict_Lab
from inverse_nearest import init as init_inverse, optimize as optimize_nearest
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OLLAMA_URL = "http://localhost:11434"
MODEL = "qwen2.5:7b"  # 默认使用 qwen2.5 7B，支持中文

_feature_names = None

def _get_feature_names():
    global _feature_names
    if _feature_names is None:
        _feature_names = get_feature_names()
    return _feature_names

# ==================== 工具函数 ====================

def tool_predict_lab(params_dict):
    """预测配方对应的Lab值"""
    names = _get_feature_names()
    params = np.zeros(60)
    for k, v in params_dict.items():
        if k in names:
            params[names.index(k)] = float(v)
    L, a, b = predict_Lab(params.tolist())
    return {"L": round(float(L), 1), "a": round(float(a), 2), "b": round(float(b), 2)}

def tool_optimize_formula(L, a, b):
    """根据目标Lab值逆向优化生成配方"""
    init_inverse()
    result = optimize_nearest(float(L), float(a), float(b))
    names = _get_feature_names()
    formula = {}
    params = result['params']
    # 主成分
    formula['M002'] = round(params[0], 2)
    formula['M001'] = round(params[1], 2)
    formula['M003'] = round(params[2], 2)
    # 工艺参数
    proc_indices = {55: '电压', 56: '电流', 57: '占空比', 58: '频率', 59: '周期'}
    for idx, name in proc_indices.items():
        formula[name] = round(params[idx], 2)
    # 非零添加剂
    additives = {}
    for i in range(3, 55):
        if params[i] > 0.01:
            additives[names[i]] = round(params[i], 2)
    formula['additives'] = additives

    return {
        "target_L": L, "target_a": a, "target_b": b,
        "pred_L": round(result['pred_L'], 1),
        "pred_a": round(result['pred_a'], 2),
        "pred_b": round(result['pred_b'], 2),
        "L_error": f"{result['L_err']:.1f}%",
        "a_error": f"{result['a_err']:.1f}%",
        "b_error": f"{result['b_err']:.1f}%",
        "neighbor_L": round(result['neighbor_L'], 1),
        "neighbor_a": round(result['neighbor_a'], 2),
        "neighbor_b": round(result['neighbor_b'], 2),
        "formula": formula,
    }

def tool_analyze_feature(feature_name):
    """分析某个成分在数据库中的统计信息"""
    init_forward()
    names = _get_feature_names()
    if feature_name not in names:
        return {"error": f"未知成分: {feature_name}"}

    import pandas as pd
    df = pd.read_excel(os.path.join(BASE_DIR, '实验数据.xlsx'), header=1)
    col = df[feature_name]
    vals = pd.to_numeric(col.values, errors='coerce')
    vals = vals.dropna()
    nonzero = vals[vals > 0]

    L_col = pd.to_numeric(df['L'].values, errors='coerce')
    a_col = pd.to_numeric(df['a'].values, errors='coerce')
    b_col = pd.to_numeric(df['b'].values, errors='coerce')
    mask = ~(np.isnan(L_col) | np.isnan(a_col) | np.isnan(b_col))
    L_col, a_col, b_col = L_col[mask], a_col[mask], b_col[mask]

    return {
        "name": feature_name,
        "total_samples": len(vals),
        "nonzero_samples": len(nonzero),
        "nonzero_ratio": f"{len(nonzero)/max(len(vals),1)*100:.1f}%",
        "mean": round(float(vals.mean()), 2),
        "min": round(float(vals.min()), 2),
        "max": round(float(vals.max()), 2),
    }

# ==================== Tool定义 ====================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "predict_lab",
            "description": "根据配方参数预测Lab颜色值。输入一个字典，key是成分名，value是数值。",
            "parameters": {
                "type": "object",
                "properties": {
                    "params": {
                        "type": "object",
                        "description": "配方参数字典，如 {'M002': 40, 'M001': 0.5, '电压': 620}"
                    }
                },
                "required": ["params"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "optimize_formula",
            "description": "根据目标L*a*b*颜色值，逆向优化生成配方。L范围约20-90，a约-5~9，b约-19~26。",
            "parameters": {
                "type": "object",
                "properties": {
                    "L": {"type": "number", "description": "明度 (Lightness), 范围约20-90"},
                    "a": {"type": "number", "description": "红绿色轴, 范围约-5~9"},
                    "b": {"type": "number", "description": "黄蓝色轴, 范围约-19~26"}
                },
                "required": ["L", "a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_feature",
            "description": "分析某个成分在数据库中的统计信息，包括使用频率、数值范围等",
            "parameters": {
                "type": "object",
                "properties": {
                    "feature_name": {"type": "string", "description": "成分名, 如 M002, C007, 电压 等"}
                },
                "required": ["feature_name"]
            }
        }
    }
]

SYSTEM_PROMPT = """你是一个着色盐配方优化专家助手。你可以：
1. 根据用户描述的颜色（如"浅蓝色"、"深红色"）推荐目标Lab值并优化配方
2. 解释配方中各成分的作用
3. 帮助用户调整已有配方并预测结果

## 颜色与Lab值的参考映射：
- 白色/浅色: L>80, a≈0, b≈0
- 黑色/深色: L<30, a≈0, b≈0
- 红色: a>3, b>0
- 蓝色: a<0, b<-2
- 黄色: b>5
- 绿色: a<-2, b>3
- 浅蓝: L=60-80, a=-1~0, b=-5~-2

## 工作流程：
1. 如果用户描述了颜色但没有给具体Lab值，你根据映射推断合理值
2. 调用 optimize_formula 生成配方
3. 解释配方中的关键成分及其作用
4. 如果用户想调整，调用 predict_lab 预测新结果

请用中文回复，简洁专业。"""

# ==================== Agent核心 ====================

def chat(user_message, history=None):
    """与Ollama Agent对话"""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    # 第一次调用：让模型决定是否需要调用工具
    try:
        resp = requests.post(f"{OLLAMA_URL}/v1/chat/completions", json={
            "model": MODEL,
            "messages": messages,
            "tools": TOOLS,
            "stream": False,
        }, timeout=120)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.ConnectionError:
        return "❌ 无法连接到 Ollama。请确保已启动: `ollama serve`", None
    except Exception as e:
        return f"❌ 请求失败: {e}", None

    choice = data["choices"][0]
    msg = choice["message"]

    # 检查是否有工具调用
    tool_calls = msg.get("tool_calls")
    if tool_calls:
        # 执行工具
        messages.append({"role": "assistant", "content": msg.get("content", ""), "tool_calls": tool_calls})

        for tc in tool_calls:
            func_name = tc["function"]["name"]
            func_args = json.loads(tc["function"]["arguments"])
            tool_result = None

            if func_name == "predict_lab":
                tool_result = tool_predict_lab(func_args.get("params", {}))
            elif func_name == "optimize_formula":
                tool_result = tool_optimize_formula(
                    func_args.get("L", 50), func_args.get("a", 0), func_args.get("b", 0))
            elif func_name == "analyze_feature":
                tool_result = tool_analyze_feature(func_args.get("feature_name", ""))

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": json.dumps(tool_result, ensure_ascii=False)
            })

        # 第二次调用：根据工具结果生成回复
        resp2 = requests.post(f"{OLLAMA_URL}/v1/chat/completions", json={
            "model": MODEL,
            "messages": messages,
            "stream": False,
        }, timeout=120)
        resp2.raise_for_status()
        data2 = resp2.json()
        final_msg = data2["choices"][0]["message"]["content"]
        return final_msg, messages
    else:
        return msg.get("content", ""), messages

def check_ollama():
    """检查Ollama是否可用"""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return resp.status_code == 200
    except:
        return False
