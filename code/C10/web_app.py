import streamlit as st
import time
import os
import logging
import tempfile  # 新增：用于处理上传的临时文件
from main import ClinicalDecisionSystem  # 引入重构后的临床决策系统

# ================= 1. 页面基础配置 =================
st.set_page_config(
    page_title="临床决策辅助 - 智能图RAG助手",
    page_icon="🩺",
    layout="centered"
)

# ================= 2. CSS 样式优化 =================
st.markdown("""
<style>
    /* 调整聊天气泡的样式 */
    .stChatMessage {
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #f0f2f6;
        margin-bottom: 1.5rem;
    }
    /* 增大输入框本身的大小 */
    .stChatInput textarea {
        min-height: 50px !important;
        font-size: 16px !important;
    }
    /* 给主内容区域底部增加大量留白，防止输入框遮挡 */
    .main .block-container {
        padding-bottom: 150px; 
    }
    h1 {
        color: #007BFF;
    }
    /* 优化路由分析气泡的样式 */
    .routing-box {
        background-color: #f8f9fa;
        border-left: 4px solid #007BFF;
        padding: 10px 15px;
        margin-bottom: 15px;
        border-radius: 4px;
        font-size: 0.9em;
        color: #555;
    }
</style>
""", unsafe_allow_html=True)


# ================= 3. 后端初始化逻辑 =================
@st.cache_resource(show_spinner=False)
def init_rag_system():
    try:
        # 实例化重构后的临床决策系统
        rag = ClinicalDecisionSystem()
        rag.initialize_system()
        rag.build_knowledge_base()
        return rag
    except Exception as e:
        st.error(f"系统启动失败: {e}")
        return None


# 初始化系统实例
if 'rag_system' not in st.session_state:
    with st.spinner('正在加载医学图谱与病例向量库，请稍候...'):
        system_instance = init_rag_system()
        if system_instance:
            st.session_state.rag_system = system_instance
        else:
            st.stop()

# ================= 4. 侧边栏配置 =================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/medical-doctor.png", width=80)
    st.title("⚙️ 系统设置")

    if 'rag_system' in st.session_state:
        st.success("✅ 临床知识图谱已就绪")
    else:
        st.warning("⏳ 系统初始化中...")

    st.markdown("---")
    enable_stream = st.toggle("启用专业回答流式输出", value=True)
    show_routing = st.toggle("显示临床推理过程 (Agentic Thinking)", value=True)

    if st.button("🗑️ 清空问诊历史"):
        st.session_state.messages = []
        st.rerun()

    # 👇 新增：增量数据上传与合并模块
    st.markdown("---")
    st.subheader("📦 增量知识更新")
    uploaded_file = st.file_uploader("上传新的医学问答数据 (CSV)", type=['csv'])

    if uploaded_file is not None:
        if st.button("🚀 开始导入新数据", use_container_width=True):
            with st.spinner("正在解析数据并追加到图谱和向量库中..."):
                try:
                    # 1. 将 Streamlit 内存中的文件写入临时文件
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name

                    # 2. 调用我们在 main.py 中封装的增量更新方法
                    st.session_state.rag_system.add_new_knowledge(tmp_path)

                    # 3. 清理临时文件
                    os.unlink(tmp_path)

                    st.success("✅ 增量知识更新完成！您可以立即针对新数据进行提问。")
                except Exception as e:
                    st.error(f"❌ 导入失败: {e}")

    st.markdown("---")
    st.caption("Powered by DeepSeek & Medical GraphRAG")

# ================= 5. 主界面逻辑 =================
st.title("🏥 临床决策辅助系统")
st.caption("基于医学知识图谱与多跳推理的下一代智能医疗辅助大脑")

# 初始化对话历史
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "您好！我是您的临床决策辅助助手。您可以向我咨询病症初筛、药物禁忌或复杂的临床关联问题（例如：高血压合并糖尿病患者的饮食禁忌？）。"}
    ]

# 渲染历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if "routing_info" in msg and show_routing:
            st.markdown(msg["routing_info"], unsafe_allow_html=True)
        st.markdown(msg["content"])

# 处理用户输入
if prompt := st.chat_input("请输入症状、药物、科室或临床疑问..."):
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

        # 生成 AI 回答
        with st.chat_message("assistant"):
            rag = st.session_state.rag_system
            full_response = ""
            routing_html = ""

            try:
                # 1. 使用 Streamlit 的 status 组件展示后端的检索思考过程
                with st.status("🧠 正在进行临床意图分析与多跳推理...", expanded=show_routing) as status:

                    # 调用路由模块获取相关文档
                    relevant_docs, analysis = rag.router.route_query(prompt, rag.config.top_k)

                    # 如果开启了路由展示，且分析结果存在，则渲染临床路由卡片
                    if analysis:
                        strategy_icons = {
                            "hybrid_traditional": "🔍 (基础医疗信息检索)",
                            "graph_rag": "🧬 (复杂临床路径推理)",
                            "combined": "🏥 (图谱+向量综合研判)"
                        }
                        icon_name = strategy_icons.get(analysis.recommended_strategy.value, "❓")

                        routing_html = f"""
                                <div class="routing-box">
                                    <b>系统决策路径:</b> {icon_name}<br>
                                    <b>临床复杂度:</b> {analysis.query_complexity:.2f} &nbsp;|&nbsp; 
                                    <b>医学关系密集度:</b> {analysis.relationship_intensity:.2f}<br>
                                    <b>调度理由:</b> {analysis.reasoning}
                                </div>
                                """
                        if show_routing:
                            st.markdown(routing_html, unsafe_allow_html=True)

                    status.update(label="✅ 临床证据提取完毕，开始生成建议！", state="complete", expanded=False)

                # 2. 等待状态框渲染完毕后创建占位符
                message_placeholder = st.empty()

                # 3. 调用重构后的生成模块
                if enable_stream:
                    # 获取医学流式生成器
                    response_generator = rag.generation_module.generate_adaptive_answer_stream(prompt, relevant_docs)
                    for chunk in response_generator:
                        if isinstance(chunk, str):
                            full_response += chunk
                            message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                else:
                    # 非流式处理
                    full_response = rag.generation_module.generate_adaptive_answer(prompt, relevant_docs)
                    message_placeholder.markdown(full_response)

                # 4. 存入历史记录
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "routing_info": routing_html
                })

            except Exception as e:
                st.error(f"生成临床建议时出错: {e}")