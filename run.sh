#!/bin/bash

# LangGraph Agent 启动脚本

echo "🚀 启动 LangGraph Agent WebUI..."

# 检查 Python 环境
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到 Python3，请先安装 Python 3.8+"
    exit 1
fi

# 检查依赖
if [ ! -f "requirements.txt" ]; then
    echo "❌ 错误: 未找到 requirements.txt"
    exit 1
fi

# 安装依赖（如果需要）
# echo "📦 检查依赖..."
# pip install -q -r requirements.txt

# 启动 Streamlit
echo "🌐 启动 WebUI..."
streamlit run app.py --server.port ${WEB_PORT:-8501} --server.address ${WEB_HOST:-0.0.0.0}

