"""
配置文件
"""
import os

# ModelScope 模型配置
# 可以从 https://modelscope.cn/models 选择模型
# 推荐使用 Qwen、ChatGLM 等中文模型
MODEL_NAME = os.getenv("MODEL_NAME", "qwen/Qwen2-1.5B-Instruct")
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "./models")

# LangGraph 配置
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "10"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))

# WebUI 配置
WEB_PORT = int(os.getenv("WEB_PORT", "8501"))
WEB_HOST = os.getenv("WEB_HOST", "0.0.0.0")

