"""
Streamlit WebUI 界面
"""
import streamlit as st
import sys
import os
from agent import get_agent
from config import MODEL_NAME, MODEL_CACHE_DIR, TEMPERATURE, MAX_TOKENS
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 页面配置
st.set_page_config(
    page_title="LangGraph Agent 对话系统",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 20%;
    }
    .assistant-message {
        background-color: #f5f5f5;
        margin-right: 20%;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """初始化会话状态"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "initialized" not in st.session_state:
        st.session_state.initialized = False


def initialize_agent():
    """初始化 Agent"""
    if st.session_state.agent is None:
        with st.spinner("正在加载模型，这可能需要几分钟..."):
            try:
                st.session_state.agent = get_agent()
                st.session_state.initialized = True
                st.success("模型加载成功！")
            except Exception as e:
                st.error(f"模型加载失败: {str(e)}")
                st.session_state.initialized = False
                return False
    return True


def display_chat_message(role: str, content: str):
    """显示聊天消息"""
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <div>
                <strong>👤 用户:</strong><br/>
                {content}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <div>
                <strong>🤖 助手:</strong><br/>
                {content}
            </div>
        </div>
        """, unsafe_allow_html=True)


def main():
    """主函数"""
    init_session_state()
    
    # 标题
    st.markdown('<div class="main-header">🤖 LangGraph Agent 对话系统</div>', unsafe_allow_html=True)
    
    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 配置")
        
        st.subheader("模型信息")
        st.info(f"**模型名称:** {MODEL_NAME}\n\n**缓存目录:** {MODEL_CACHE_DIR}")
        
        st.subheader("参数设置")
        temperature = st.slider("Temperature", 0.0, 1.0, TEMPERATURE, 0.1)
        max_tokens = st.slider("Max Tokens", 512, 4096, MAX_TOKENS, 512)
        
        st.subheader("操作")
        if st.button("🔄 重新加载模型"):
            st.session_state.agent = None
            st.session_state.initialized = False
            st.rerun()
        
        if st.button("🗑️ 清空对话历史"):
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📖 使用说明")
        st.markdown("""
        1. 首次使用会自动下载模型（可能需要较长时间）
        2. 模型加载完成后即可开始对话
        3. 支持多轮对话，上下文会自动保持
        4. 可以在侧边栏调整生成参数
        """)
    
    # 初始化 Agent
    if not st.session_state.initialized:
        if st.button("🚀 初始化 Agent"):
            initialize_agent()
            st.rerun()
    else:
        # 显示聊天历史
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                display_chat_message(message["role"], message["content"])
        
        # 用户输入
        user_input = st.chat_input("请输入您的问题...")
        
        if user_input:
            # 添加用户消息
            st.session_state.messages.append({"role": "user", "content": user_input})
            display_chat_message("user", user_input)
            
            # 获取 AI 回复
            if st.session_state.agent:
                with st.spinner("正在思考..."):
                    try:
                        # 构建历史记录 - 格式: [(user_msg, ai_msg), ...]
                        history = []
                        i = 0
                        while i < len(st.session_state.messages) - 1:
                            if st.session_state.messages[i]["role"] == "user":
                                user_msg = st.session_state.messages[i]["content"]
                                if i + 1 < len(st.session_state.messages) and \
                                   st.session_state.messages[i + 1]["role"] == "assistant":
                                    ai_msg = st.session_state.messages[i + 1]["content"]
                                    history.append((user_msg, ai_msg))
                                    i += 2
                                else:
                                    i += 1
                            else:
                                i += 1
                        
                        # 调用 agent
                        ai_response, _ = st.session_state.agent.chat(user_input, history)
                        
                        # 添加 AI 消息
                        st.session_state.messages.append({"role": "assistant", "content": ai_response})
                        display_chat_message("assistant", ai_response)
                        
                    except Exception as e:
                        error_msg = f"处理请求时出错: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
            else:
                st.warning("Agent 未初始化，请先初始化 Agent")


if __name__ == "__main__":
    main()

