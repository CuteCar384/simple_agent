"""
测试脚本 - 验证环境配置是否正确
"""
import sys

def test_imports():
    """测试必要的导入"""
    print("🔍 检查依赖包...")
    
    required_packages = {
        "langgraph": "LangGraph",
        "langchain": "LangChain",
        "streamlit": "Streamlit",
        "modelscope": "ModelScope",
        "transformers": "Transformers",
        "torch": "PyTorch",
    }
    
    missing_packages = []
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ 缺少以下包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    print("\n✅ 所有依赖包已安装")
    return True


def test_torch():
    """测试 PyTorch 和 CUDA"""
    print("\n🔍 检查 PyTorch 和 CUDA...")
    
    try:
        import torch
        print(f"  ✅ PyTorch 版本: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"  ✅ CUDA 可用")
            print(f"  ✅ GPU 数量: {torch.cuda.device_count()}")
            print(f"  ✅ 当前 GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(f"  ⚠️  CUDA 不可用，将使用 CPU（速度较慢）")
        
        return True
    except Exception as e:
        print(f"  ❌ PyTorch 检查失败: {e}")
        return False


def test_config():
    """测试配置文件"""
    print("\n🔍 检查配置文件...")
    
    try:
        from config import MODEL_NAME, MODEL_CACHE_DIR
        print(f"  ✅ 模型名称: {MODEL_NAME}")
        print(f"  ✅ 缓存目录: {MODEL_CACHE_DIR}")
        return True
    except Exception as e:
        print(f"  ❌ 配置文件错误: {e}")
        return False


def main():
    """主函数"""
    print("=" * 50)
    print("LangGraph Agent 环境检查")
    print("=" * 50)
    
    results = []
    results.append(test_imports())
    results.append(test_torch())
    results.append(test_config())
    
    print("\n" + "=" * 50)
    if all(results):
        print("✅ 环境配置正确，可以开始使用！")
        print("\n运行以下命令启动 WebUI:")
        print("  streamlit run app.py")
        return 0
    else:
        print("❌ 环境配置有问题，请根据上述提示修复")
        return 1


if __name__ == "__main__":
    sys.exit(main())

