# 快速开始指南

## 5 分钟快速上手

### 步骤 1: 安装依赖

```bash
pip install -r requirements.txt
```

### 步骤 2: 配置模型（可选）

编辑 `config.py`，选择适合的模型：

```python
# 小模型（推荐首次使用，速度快）
MODEL_NAME = "qwen/Qwen2-1.5B-Instruct"

# 中等模型（平衡性能和速度）
MODEL_NAME = "qwen/Qwen2-7B-Instruct"
```

### 步骤 3: 启动应用

**方式一：使用启动脚本**
```bash
./run.sh
```

**方式二：直接运行**
```bash
streamlit run app.py
```

### 步骤 4: 访问界面

在浏览器中打开：`http://localhost:8501`

### 步骤 5: 开始对话

1. 点击"🚀 初始化 Agent"按钮
2. 等待模型加载（首次使用会自动下载模型）
3. 在输入框中输入问题
4. 按回车或点击发送

## 常见问题

### Q: 模型下载很慢怎么办？

A: ModelScope 的下载速度取决于网络。可以：
- 使用国内网络环境
- 手动下载模型到 `models/` 目录
- 选择更小的模型（如 1.5B）

### Q: 内存不足怎么办？

A: 
- 使用更小的模型（1.5B 或 3B）
- 在 `model_loader.py` 中设置 `use_gpu=False` 使用 CPU（虽然更慢但内存占用更少）
- 关闭其他占用内存的程序

### Q: 如何更换模型？

A: 
1. 修改 `config.py` 中的 `MODEL_NAME`
2. 点击 WebUI 中的"🔄 重新加载模型"按钮
3. 或者重启应用

## 推荐配置

### 低配置机器（8GB 内存）
```python
MODEL_NAME = "qwen/Qwen2-1.5B-Instruct"
```

### 中等配置（16GB 内存）
```python
MODEL_NAME = "qwen/Qwen2-7B-Instruct"
```

### 高配置（32GB+ 内存，有 GPU）
```python
MODEL_NAME = "qwen/Qwen2-14B-Instruct"  # 或其他大模型
```

## 下一步

- 查看 `README.md` 了解详细功能
- 修改 `agent.py` 添加自定义工具
- 自定义 `app.py` 的 UI 界面

