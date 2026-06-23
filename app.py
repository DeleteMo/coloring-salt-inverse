"""
Streamlit 应用 - L a b 预测与逆向配方
"""
import streamlit as st
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from inverse_utils import (
    initialize, get_feature_names,
    predict_Lab
)
from inverse_nearest import init as init_nearest, optimize as optimize_nearest
from agent import chat as agent_chat, check_ollama

# 页面配置
st.set_page_config(
    page_title="着色盐预测工具",
    page_icon="🎨",
    layout="wide"
)

# 自定义CSS样式
st.markdown("""
<style>
    /* 主色调 */
    :root {
        --primary: #2E86AB;
        --secondary: #A23B72;
        --accent: #F18F01;
        --bg-dark: #1E1E2E;
        --bg-light: #F8F9FA;
        --text: #2C3E50;
    }

    /* 标题样式 */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2E86AB;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #F18F01;
        margin-bottom: 2rem;
    }

    /* 卡片样式 */
    .card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 4px solid #2E86AB;
        margin-bottom: 1rem;
    }

    .card-inverse {
        border-left-color: #A23B72;
    }

    /* 结果数字 */
    .result-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2E86AB;
    }

    .result-number-inverse {
        color: #A23B72;
    }

    /* 标签样式 */
    .label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* 分隔线 */
    .divider {
        border: none;
        border-top: 2px dashed #ddd;
        margin: 1.5rem 0;
    }

    /* 按钮样式 */
    .stButton > button {
        background: linear-gradient(135deg, #2E86AB, #1a5f7a);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #1a5f7a, #2E86AB);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(46,134,171,0.4);
    }

    /* 侧边栏 */
    .css-1d391kg {
        background: linear-gradient(180deg, #1E1E2E 0%, #2a2a3d 100%);
    }

    /* 指标卡片 */
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }

    /* 配方卡片 */
    .formula-card {
        background: linear-gradient(135deg, #fff0%, #f8f9fa 100%);
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        margin-bottom: 0.8rem;
        border: 1px solid #eee;
    }

    /* 颜色指示 */
    .color-indicator {
        display: inline-block;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        margin-right: 8px;
    }

    /* 进度条 */
    .progress-bar {
        height: 8px;
        background: #e9ecef;
        border-radius: 4px;
        overflow: hidden;
    }

    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #2E86AB, #F18F01);
        border-radius: 4px;
    }

    /* 表头样式 */
    .table-header {
        background: linear-gradient(135deg, #2E86AB, #1a5f7a);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# 主页面标题
st.markdown('<h1 class="main-title">🎨 着色盐 L*a*b* 预测与逆向配方工具</h1>', unsafe_allow_html=True)

# 页面导航
st.sidebar.markdown("## 📋 功能导航")
page = st.sidebar.selectbox(
    "选择功能",
    ["🔮 正向预测", "🧪 逆向配方", "🤖 AI助手"]
)

# ===== 正向预测 =====
if page == "🔮 正向预测":
    initialize()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="card"><h3>📥 输入特征</h3>', unsafe_allow_html=True)
        st.write("输入60个特征值，预测对应的 L*a*b* 颜色值")

        input_method = st.radio("**选择输入方式**", ["📁 上传文件", "✏️ 手动输入"], horizontal=True)

        if input_method == "📁 上传文件":
            uploaded_file = st.file_uploader("上传Excel文件（格式与训练数据相同）", type=['xlsx'])
            if uploaded_file:
                df_input = pd.read_excel(uploaded_file, header=None)
                st.write("**文件预览：**")
                st.dataframe(df_input.head(3), use_container_width=True)

                if st.button("🚀 开始预测", type="primary"):
                    if df_input.shape[1] >= 60:
                        features = df_input.iloc[0, :60].values.astype(float)
                        L, a, b = predict_Lab(features)

                        with col2:
                            st.markdown('<div class="card"><h3>📊 预测结果</h3>', unsafe_allow_html=True)

                            # L*a*b* 结果展示
                            st.markdown(f"""
                            <div style="display:flex; justify-content:space-around; margin: 1.5rem 0;">
                                <div class="metric-card">
                                    <div class="label">L* 明度</div>
                                    <div class="result-number">{L:.2f}</div>
                                </div>
                                <div class="metric-card">
                                    <div class="label">a* 红绿</div>
                                    <div class="result-number" style="color: #A23B72;">{a:.2f}</div>
                                </div>
                                <div class="metric-card">
                                    <div class="label">b* 黄蓝</div>
                                    <div class="result-number" style="color: #F18F01;">{b:.2f}</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                            # 颜色模拟 - Lab转RGB
                            import colorsys
                            def lab_to_rgb(L, a, b):
                                # Lab -> XYZ
                                y = (L + 16) / 116
                                x = a / 500 + y
                                z = y - b / 200

                                x = x**3 if x**3 > 0.008856 else (x - 16/116) / 7.787
                                y = y**3 if y**3 > 0.008856 else (y - 16/116) / 7.787
                                z = z**3 if z**3 > 0.008856 else (z - 16/116) / 7.787

                                x *= 95.047
                                y *= 100.0
                                z *= 108.883

                                # XYZ -> RGB (D65)
                                r = x * 3.2406 + y * -1.5372 + z * -0.4986
                                g = x * -0.9689 + y * 1.8758 + z * 0.0415
                                b_rgb = x * 0.0557 + y * -0.2040 + z * 1.0570

                                r /= 100; g /= 100; b_rgb /= 100

                                r = 1.055 * r**(1/2.4) - 0.055 if r > 0.00304 else r * 12.92
                                g = 1.055 * g**(1/2.4) - 0.055 if g > 0.00304 else g * 12.92
                                b_rgb = 1.055 * b_rgb**(1/2.4) - 0.055 if b_rgb > 0.00304 else b_rgb * 12.92

                                return max(0, min(1, r)), max(0, min(1, g)), max(0, min(1, b_rgb))

                            r, g, b_rgb = lab_to_rgb(L, a, b)
                            hex_color = f"#{int(r*255):02x}{int(g*255):02x}{int(b_rgb*255):02x}"

                            st.markdown(f"""
                            <div style="text-align:center; margin-top:1rem;">
                                <div style="
                                    width: 120px; height: 120px;
                                    background: {hex_color};
                                    border-radius: 50%;
                                    margin: 0 auto;
                                    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
                                    border: 3px solid white;
                                "></div>
                                <p style="margin-top:0.5rem; color:#666;">{hex_color}</p>
                            </div>
                            """, unsafe_allow_html=True)

                            # 真实值对比
                            if df_input.shape[1] >= 63:
                                true_L = df_input.iloc[0, 60]
                                true_a = df_input.iloc[0, 61]
                                true_b = df_input.iloc[0, 62]
                                if not pd.isna(true_L):
                                    st.markdown('<hr class="divider">', unsafe_allow_html=True)
                                    st.write("**📐 真实值对比：**")
                                    cols = st.columns(3)
                                    cols[0].metric("L误差", f"{abs(L-true_L):.2f}")
                                    cols[1].metric("a 误差", f"{abs(a-true_a):.2f}")
                                    cols[2].metric("b 误差", f"{abs(b-true_b):.2f}")
                    else:
                        st.error("⚠️ 文件列数不足，需要至少60列特征")
        else:
            feature_names = get_feature_names()

            st.markdown("###✏️ 手动输入60个特征值")

            # 分3列显示特征输入
            col_a, col_b, col_c = st.columns(3)

            inputs = []
            with col_a:
                for i in range(0, 20):
                    default_val = 0.0 if i >= 2 else (10.0 if i == 0 else 5.0)
                    val = st.number_input(f"{feature_names[i]}", value=default_val, key=f"feat_{i}")
                    inputs.append(val)

            with col_b:
                for i in range(20, 40):
                    val = st.number_input(f"{feature_names[i]}", value=0.0, key=f"feat_{i}")
                    inputs.append(val)

            with col_c:
                for i in range(40, 60):
                    val = st.number_input(f"{feature_names[i]}", value=0.0, key=f"feat_{i}")
                    inputs.append(val)

            if st.button("🔮 预测", type="primary"):
                L, a, b = predict_Lab(inputs)

                with col2:
                    st.markdown('<div class="card"><h3>📊 预测结果</h3>', unsafe_allow_html=True)

                    st.markdown(f"""
                    <div style="display:flex; justify-content:space-around; margin: 1.5rem 0;">
                        <div class="metric-card">
                            <div class="label">L* 明度</div>
                            <div class="result-number">{L:.2f}</div>
                        </div>
                        <div class="metric-card">
                            <div class="label">a* 红绿</div>
                            <div class="result-number" style="color: #A23B72;">{a:.2f}</div>
                        </div>
                        <div class="metric-card">
                            <div class="label">b* 黄蓝</div>
                            <div class="result-number" style="color: #F18F01;">{b:.2f}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # 配方成分
                    st.markdown('<hr class="divider">', unsafe_allow_html=True)
                    st.write("**🧪 配方成分：**")
                    nonzero = [(feature_names[i], v) for i, v in enumerate(inputs) if v > 0]
                    if nonzero:
                        for name, val in nonzero[:10]:
                            st.write(f"  • **{name}**: `{val:.2f}`")
                        if len(nonzero) > 10:
                            st.write(f"  ... 还有 {len(nonzero)-10} 个成分")
                    else:
                        st.write("所有成分均为0")

        st.markdown('</div>', unsafe_allow_html=True)

# =====逆向配方 =====
elif page == "🧪 逆向配方":
    initialize()
    init_nearest()
    feature_names = get_feature_names()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="card card-inverse"><h3>🎯 目标设置</h3>', unsafe_allow_html=True)
        st.write("输入目标 L*a*b* 值，在数据库中找最近邻配方并微调优化")

        # L*a*b* 输入 - 横向排列
        st.markdown("### 📌 目标颜色值")
        cols = st.columns(3)
        with cols[0]:
            target_L = st.number_input("**L***", value=66.2, step=0.1, help="明度 (0-100)")
        with cols[1]:
            target_a = st.number_input("**a***", value=4.6, step=0.1, help="红绿轴 (-128~127)")
        with cols[2]:
            target_b = st.number_input("**b***", value=6.8, step=0.1, help="黄蓝轴 (-128~127)")

        # 颜色预览
        def lab_to_rgb(L, a, b):
            y = (L + 16) / 116
            x = a / 500 + y
            z = y - b / 200
            x = x**3 if x**3 > 0.008856 else (x - 16/116) / 7.787
            y = y**3 if y**3 > 0.008856 else (y - 16/116) / 7.787
            z = z**3 if z**3 > 0.008856 else (z - 16/116) / 7.787
            x *= 95.047; y *= 100.0; z *= 108.883
            r = x * 3.2406 + y * -1.5372 + z * -0.4986
            g = x * -0.9689 + y * 1.8758 + z * 0.0415
            b_rgb = x * 0.0557 + y * -0.2040 + z * 1.0570
            r /= 100; g /= 100; b_rgb /= 100
            r = 1.055 * r**(1/2.4) - 0.055 if r > 0.00304 else r * 12.92
            g = 1.055 * g**(1/2.4) - 0.055 if g > 0.00304 else g * 12.92
            b_rgb = 1.055 * b_rgb**(1/2.4) - 0.055 if b_rgb > 0.00304 else b_rgb * 12.92
            return max(0, min(1, r)), max(0, min(1, g)), max(0, min(1, b_rgb))

        r, g, b_rgb = lab_to_rgb(target_L, target_a, target_b)
        hex_color = f"#{int(r*255):02x}{int(g*255):02x}{int(b_rgb*255):02x}"

        st.markdown(f"""
        <div style="text-align:center; margin: 1rem 0;">
            <div style="
                width: 80px; height: 80px;
                background: {hex_color};
                border-radius: 50%;
                margin: 0 auto 0.5rem;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                border: 3px solid white;
            "></div>
            <small style="color:#666;">{hex_color}</small>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("🧪 生成配方", type="primary"):
            with st.spinner("匹配最近邻并优化中..."):
                result = optimize_nearest(target_L, target_a, target_b)

            with col2:
                st.markdown('<div class="card card-inverse"><h3>📋 优化结果</h3>', unsafe_allow_html=True)

                # 误差显示
                errL = result['L_err']
                erra = result['a_err']
                errb = result['b_err']
                all_pass = errL < 20 and erra < 20 and errb < 20

                status_color = "#27ae60" if all_pass else "#e74c3c"
                status_icon = "✅" if all_pass else "⚠️"

                st.markdown(f"""
                <div class="formula-card">
                    <h4 style="color:{status_color};">{status_icon} {'全达标' if all_pass else '部分超标'}</h4>
                    <div style="display:flex; justify-content:space-around; margin: 1rem 0;">
                        <div class="metric-card">
                            <div class="label">L* 明度</div>
                            <div class="result-number">{result['pred_L']:.1f}</div>
                            <small>误差 {errL:.1f}%</small>
                        </div>
                        <div class="metric-card">
                            <div class="label">a* 红绿</div>
                            <div class="result-number" style="color: #A23B72;">{result['pred_a']:.2f}</div>
                            <small>误差 {erra:.1f}%</small>
                        </div>
                        <div class="metric-card">
                            <div class="label">b* 黄蓝</div>
                            <div class="result-number" style="color: #F18F01;">{result['pred_b']:.2f}</div>
                            <small>误差 {errb:.1f}%</small>
                        </div>
                    </div>
                    <small style="color:#999;">MSE: {result['mse']:.4f}</small>
                </div>
                """, unsafe_allow_html=True)

                # 最近邻信息
                st.markdown('<h4>🔍 参考样本</h4>', unsafe_allow_html=True)
                st.markdown(f"""
                <div style="background:#f0f4ff; border-radius:8px; padding:0.8rem; margin:0.5rem 0; font-size:0.9rem;">
                    数据库中最近邻样本: L={result.get('neighbor_L', 'N/A')}, a={result.get('neighbor_a', 'N/A')}, b={result.get('neighbor_b', 'N/A')}
                </div>
                """, unsafe_allow_html=True)

                # 配方详情
                st.markdown('<h4>🧪 推荐配方</h4>', unsafe_allow_html=True)
                with st.expander("📜 查看配方详情", expanded=True):
                    params = result['params']

                    # 主成分
                    st.write("**主成分：**")
                    cols = st.columns(3)
                    cols[0].metric("M002", f"{params[0]:.2f}")
                    cols[1].metric("M001", f"{params[1]:.2f}")
                    cols[2].metric("M003", f"{params[2]:.2f}")

                    # 工艺参数
                    st.write("**工艺参数：**")
                    proc_cols = st.columns(5)
                    proc_names = ['电压', '电流', '占空比', '频率', '周期']
                    proc_indices = [55, 56, 57, 58, 59]
                    for i, (name, idx) in enumerate(zip(proc_names, proc_indices)):
                        proc_cols[i].metric(name, f"{params[idx]:.1f}")

                    # 添加剂
                    st.write("**添加剂：**")
                    additive_lines = []
                    for idx in range(3, 55):
                        if params[idx] > 0.01:
                            name = feature_names[idx]
                            additive_lines.append(f"  • **{name}**: `{params[idx]:.2f}`")
                    if additive_lines:
                        for line in additive_lines:
                            st.write(line)
                    else:
                        st.write("  (无)")

                st.markdown('</div>', unsafe_allow_html=True)

            # 对比表
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown('<h3>📊 配方数据</h3>', unsafe_allow_html=True)

            compare_data = [{
                "参数": "预测L/a/b",
                "值": f"{result['pred_L']:.1f} / {result['pred_a']:.2f} / {result['pred_b']:.2f}",
            }, {
                "参数": "L/a/b误差%",
                "值": f"{errL:.1f}% / {erra:.1f}% / {errb:.1f}%",
            }, {
                "参数": "MSE",
                "值": f"{result['mse']:.4f}",
            }, {
                "参数": "M002 / M001 / M003",
                "值": f"{params[0]:.2f} / {params[1]:.2f} / {params[2]:.2f}",
            }]
            for idx in [55, 56, 57, 58, 59]:
                compare_data.append({
                    "参数": feature_names[idx],
                    "值": f"{params[idx]:.2f}",
                })

            df_compare = pd.DataFrame(compare_data)
            st.dataframe(df_compare, use_container_width=True, hide_index=True)

# ===== AI助手 =====
elif page == "🤖 AI助手":
    st.markdown('<div class="card"><h3>🤖 AI配方助手</h3>', unsafe_allow_html=True)
    st.write("基于本地 Ollama 模型，支持自然语言查询颜色、生成配方、分析成分")

    # 检查Ollama
    if not check_ollama():
        st.error("❌ 未检测到 Ollama 服务。请先启动: `ollama serve`，并确保已下载模型: `ollama pull qwen2.5:7b`")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.success("✅ Ollama 已连接")

        # 初始化聊天历史
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "你好！我是着色盐配方助手 🤖\n\n你可以这样问我：\n• \"我想要浅蓝色，帮我生成配方\"\n• \"分析一下 C007 这个成分\"\n• \"把M002从40改成30，颜色会怎么变？\""}
            ]

        # 显示聊天历史
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.write(msg["content"])
            elif msg["role"] == "assistant":
                with st.chat_message("assistant", avatar="🤖"):
                    st.write(msg["content"])

        # 输入框
        if prompt := st.chat_input("输入你的问题..."):
            # 显示用户消息
            with st.chat_message("user"):
                st.write(prompt)

            # 初始化模型
            initialize()
            init_nearest()

            # 调用Agent
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("思考中..."):
                    reply, history = agent_chat(prompt, st.session_state.messages)
                    st.write(reply)

            # 保存到历史
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": reply})

        # 清空按钮
        if st.button("🗑️ 清空对话"):
            st.session_state.messages = []
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# 页脚
st.markdown("""
---
<div style="text-align:center; color:#999; font-size:0.8rem; padding:1rem 0;">
    着色盐 L*a*b* 预测与逆向配方工具 | 基于随机森林与聚类优化
</div>
""", unsafe_allow_html=True)