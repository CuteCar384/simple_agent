# simple_agent
Create a simple agent using langraph
# LangGraph Agent 对话系统

基于 LangGraph 和 ModelScope 的本地大语言模型对话系统，支持 WebUI 可视化界面。

## 功能特性

- 🤖 **基于 LangGraph**: 使用 LangGraph 构建智能对话 Agent
- 📦 **ModelScope 集成**: 支持从 ModelScope 下载和使用本地模型
- 🎨 **WebUI 界面**: 基于 Streamlit 的现代化 Web 界面
- 💬 **多轮对话**: 支持上下文保持的多轮对话
- ⚙️ **参数可调**: 支持调整温度、最大 token 等生成参数

## 项目结构

```
langraph/
├── agent.py           # LangGraph Agent 核心实现
├── model_loader.py    # ModelScope 模型加载器
├── config.py          # 配置文件
├── app.py             # Streamlit WebUI
├── requirements.txt   # 项目依赖
└── README.md          # 项目说明
```

## 安装步骤

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置模型

编辑 `config.py` 文件，设置要使用的模型：

```python
MODEL_NAME = "qwen/Qwen2-7B-Instruct"  # 或其他 ModelScope 模型
```

推荐的模型：
- `qwen/Qwen2-7B-Instruct` - 通义千问 7B 指令模型
- `qwen/Qwen2-1.5B-Instruct` - 通义千问 1.5B 指令模型（更小，适合低配置）
- `THUDM/chatglm3-6b` - ChatGLM3 6B 模型

### 3. 运行 WebUI

```bash
streamlit run app.py
```

或者指定端口和主机：

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### 4. 访问界面

在浏览器中打开：`http://localhost:8501`

## 使用说明

1. **首次运行**: 系统会自动从 ModelScope 下载模型（可能需要较长时间，取决于模型大小和网络速度）

2. **初始化 Agent**: 点击"初始化 Agent"按钮，等待模型加载完成

3. **开始对话**: 在输入框中输入问题，按回车或点击发送

4. **调整参数**: 在侧边栏可以调整：
   - Temperature: 控制生成的随机性（0.0-1.0）
   - Max Tokens: 控制生成的最大长度

5. **清空历史**: 点击"清空对话历史"可以重新开始对话

## 环境变量配置

可以通过环境变量配置系统：

```bash
# 设置模型名称
export MODEL_NAME="qwen/Qwen2-7B-Instruct"

# 设置模型缓存目录
export MODEL_CACHE_DIR="./models"

# 设置生成参数
export TEMPERATURE=0.7
export MAX_TOKENS=2048

# 设置 Web 服务端口
export WEB_PORT=8501
```

## 系统要求

- Python 3.8+
- CUDA（可选，用于 GPU 加速）
- 至少 8GB 内存（推荐 16GB+）
- 足够的磁盘空间（模型大小从几 GB 到几十 GB 不等）

## 常见问题

### Q: 模型下载失败怎么办？

A: 检查网络连接，确保可以访问 ModelScope。也可以手动下载模型到 `MODEL_CACHE_DIR` 目录。

### Q: 内存不足怎么办？

A: 尝试使用更小的模型（如 1.5B 或 3B），或者在 `model_loader.py` 中设置 `use_gpu=False` 使用 CPU。

### Q: 如何更换模型？

A: 修改 `config.py` 中的 `MODEL_NAME`，然后重新运行应用。

### Q: 支持哪些模型格式？

A: 支持 ModelScope 上所有基于 Transformers 的模型，包括 Qwen、ChatGLM、Baichuan 等。

## 开发说明

### 添加工具

可以在 `agent.py` 的 `_call_tools` 方法中添加自定义工具，实现更强大的功能。

### 自定义 UI

修改 `app.py` 可以自定义 WebUI 界面和交互方式。

## 许可证

本项目采用 MIT 许可证。

## 贡献

欢迎提交 Issue 和 Pull Request！

