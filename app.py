"""
Streamlit 应用 - L a b 预测与逆向配方
"""
import streamlit as st
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import joblib

# 导入自定义模块
from inverse_utils import (
    initialize, get_feature_names, get_color_list, get_color_info,
    predict_Lab, optimize_multiple
)

# 页面配置
st.set_page_config(
    page_title="着色盐预测工具",
    page_icon="🎨",
    layout="wide"
)

# 延迟初始化 - 不在模块加载时初始化
# initialize()  # 注释掉这行，改为在需要时初始化

# 主页面标题
st.title("🎨 着色盐 L a b 预测与逆向配方工具")

# 页面导航
page = st.sidebar.selectbox(
    "选择功能",
    ["正向预测 - 输入特征获取L a b", "逆向配方 - 输入L a b获取配方"]
)

if page == "正向预测 - 输入特征获取L a b":
    # 初始化
    initialize()

    st.header("正向预测")
    st.write("输入59个特征值，预测L、a、b值")

    # 获取特征名
    feature_names = get_feature_names()

    # 创建两列布局
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("输入方式")

        input_method = st.radio("选择输入方式", ["上传文件", "手动输入"])

        if input_method == "上传文件":
            uploaded_file = st.file_uploader("上传Excel文件（格式与训练数据相同）", type=['xlsx'])
            if uploaded_file:
                df_input = pd.read_excel(uploaded_file, header=None)
                # 显示文件内容
                st.write("文件内容预览：")
                st.dataframe(df_input.head())

                if st.button("开始预测"):
                    # 检查是否有59个特征列
                    if df_input.shape[1] >= 59:
                        # 提取前59列作为特征
                        features = df_input.iloc[0, :59].values.astype(float)
                        L, a, b = predict_Lab(features)

                        with col2:
                            st.subheader("预测结果")
                            st.metric("L 值", f"{L:.2f}")
                            st.metric("a 值", f"{a:.2f}")
                            st.metric("b 值", f"{b:.2f}")

                            # 如果文件有L,a,b列，显示真实值对比
                            if df_input.shape[1] >= 63:
                                true_L = df_input.iloc[0, 60]
                                true_a = df_input.iloc[0, 61]
                                true_b = df_input.iloc[0, 62]
                                if not pd.isna(true_L):
                                    st.divider()
                                    st.write("**真实值对比：**")
                                    st.write(f"L误差: {abs(L-true_L):.2f}")
                                    st.write(f"a误差: {abs(a-true_a):.2f}")
                                    st.write(f"b误差: {abs(b-true_b):.2f}")
                    else:
                        st.error("文件列数不足，需要至少59列特征")

        else:  # 手动输入
            st.write("手动输入59个特征值：")

            # 创建特征输入表单
            col_a, col_b = st.columns(2)

            inputs = []
            with col_a:
                for i in range(0, 30):
                    default_val = 0.0 if i >= 2 else (10.0 if i == 0 else 5.0)
                    val = st.number_input(f"{feature_names[i]}", value=default_val, key=f"feat_{i}")
                    inputs.append(val)

            with col_b:
                for i in range(30, 59):
                    val = st.number_input(f"{feature_names[i]}", value=0.0, key=f"feat_{i}")
                    inputs.append(val)

            if st.button("预测", type="primary"):
                L, a, b = predict_Lab(inputs)

                with col2:
                    st.subheader("预测结果")
                    st.metric("L 值", f"{L:.2f}")
                    st.metric("a 值", f"{a:.2f}")
                    st.metric("b 值", f"{b:.2f}")

                    # 可视化配方成分
                    st.divider()
                    st.write("**配方成分：**")

                    # 显示非零成分
                    nonzero = [(feature_names[i], v) for i, v in enumerate(inputs) if v > 0]
                    if nonzero:
                        for name, val in nonzero:
                            st.write(f"- {name}: {val:.2f}")
                    else:
                        st.write("所有成分均为0")

elif page == "逆向配方 - 输入L a b获取配方":
    # 初始化
    initialize()

    st.header("逆向配方")
    st.write("输入目标L、a、b值，获取5个参考配方")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("目标设置")

        # 颜色选择
        colors = get_color_list()
        target_color = st.selectbox("目标颜色", colors)

        # L a b 输入
        col_L, col_a, col_b = st.columns(3)
        with col_L:
            target_L = st.number_input("目标 L", value=66.4, step=0.1)
        with col_a:
            target_a = st.number_input("目标 a", value=3.8, step=0.1)
        with col_b:
            target_b = st.number_input("目标 b", value=6.2, step=0.1)

        st.divider()

        # 颜色信息显示
        info = get_color_info(target_color)
        if info:
            st.write(f"**{target_color}** 颜色数据：")
            st.write(f"- 样本数: {info['count']}")
            st.write(f"- Top5添加剂: {[get_feature_names()[i] for i in info['top5_additives']]}")

        st.divider()

        if st.button("生成5个配方", type="primary"):
            with st.spinner("优化中，请稍候..."):
                results = optimize_multiple(target_L, target_a, target_b, target_color, n=5)

            with col2:
                st.subheader("推荐配方")

                for i, result in enumerate(results):
                    with st.expander(f"配方 {i+1} (MSE={result['mse']:.2f})", expanded=i==0):
                        st.write(f"**预测结果：** L={result['pred_L']:.2f}, a={result['pred_a']:.2f}, b={result['pred_b']:.2f}")
                        st.write(f"**主要成分：** M001={result['params'][0]:.2f}, M002={result['params'][1]:.2f}")
                        st.write(f"**约束验证：** {'[OK]' if result['valid'] else '[FAIL]'}")
                        st.write("**添加剂：**")

                        # 显示使用的添加剂
                        feature_names = get_feature_names()
                        additive_names = []
                        for idx in result['top5_additives']:
                            val = result['params'][idx]
                            if val > 0:
                                additive_names.append(f"{feature_names[idx]}: {val:.2f}")

                        if additive_names:
                            for name in additive_names:
                                st.write(f"  - {name}")
                        else:
                            st.write("  (无)")

            # 显示对比表
            st.divider()
            st.subheader("配方对比")

            compare_data = []
            for i, result in enumerate(results):
                compare_data.append({
                    "配方": f"配方{i+1}",
                    "MSE": f"{result['mse']:.2f}",
                    "预测L": f"{result['pred_L']:.2f}",
                    "预测a": f"{result['pred_a']:.2f}",
                    "预测b": f"{result['pred_b']:.2f}",
                    "M001": f"{result['params'][0]:.2f}",
                    "M002": f"{result['params'][1]:.2f}",
                    "有效": "是" if result['valid'] else "否"
                })

            df_compare = pd.DataFrame(compare_data)
            st.dataframe(df_compare, use_container_width=True)

            # 下载按钮
            csv = df_compare.to_csv(index=False)
            st.download_button(
                "下载配方对比表",
                csv,
                "inverse_results.csv",
                "text/csv"
            )