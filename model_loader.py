"""
ModelScope 模型加载器
"""
from typing import Optional
from langchain_community.llms import LlamaCpp
from transformers import AutoTokenizer, AutoModelForCausalLM
from modelscope import snapshot_download
import torch
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelScopeLoader:
    """从 ModelScope 加载本地模型"""
    
    def __init__(self, model_name: str, cache_dir: str = "./models"):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = None
        self.tokenizer = None
        
    def download_model(self):
        """从 ModelScope 下载模型"""
        try:
            logger.info(f"正在从 ModelScope 下载模型: {self.model_name}")
            model_dir = snapshot_download(
                self.model_name,
                cache_dir=self.cache_dir
            )
            logger.info(f"模型下载完成，路径: {model_dir}")
            return model_dir
        except Exception as e:
            logger.error(f"模型下载失败: {e}")
            raise
    
    def load_model(self, use_gpu: bool = True):
        """加载模型和分词器"""
        try:
            # 下载模型（如果尚未下载）
            model_dir = self.download_model()
            
            logger.info("正在加载分词器...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_dir,
                trust_remote_code=True
            )
            
            logger.info("正在加载模型...")
            use_cuda = use_gpu and torch.cuda.is_available()
            
            if use_cuda:
                # GPU 模式 - 使用 device_map="auto" 需要 accelerate
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_dir,
                        trust_remote_code=True,
                        dtype=torch.float16,
                        device_map="auto",
                        low_cpu_mem_usage=True
                    )
                except Exception as e:
                    logger.warning(f"使用 device_map 失败，尝试手动指定设备: {e}")
                    # 备用方案：手动加载到 GPU
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_dir,
                        trust_remote_code=True,
                        dtype=torch.float16,
                        low_cpu_mem_usage=True
                    )
                    self.model = self.model.cuda()
            else:
                # CPU 模式 - 不使用 device_map
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_dir,
                    trust_remote_code=True,
                    dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
                self.model = self.model.to("cpu")
            
            logger.info("模型加载完成")
            return self.model, self.tokenizer
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def generate(self, prompt, max_length: int = 2048, temperature: float = 0.7, max_new_tokens: int = None):
        """生成回复
        
        Args:
            prompt: 可以是字符串或消息列表（用于 chat template）
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("模型未加载，请先调用 load_model()")
        
        try:
            # 判断输入类型：如果是列表，使用 chat template
            if isinstance(prompt, list):
                # 使用 chat template 处理对话消息
                if hasattr(self.tokenizer, 'apply_chat_template'):
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        prompt,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                else:
                    # 如果不支持 chat template，手动构建
                    formatted_prompt = self._build_prompt_from_messages(prompt)
            else:
                # 字符串输入，直接使用
                formatted_prompt = str(prompt)
            
            # 编码输入
            inputs = self.tokenizer.encode(formatted_prompt, return_tensors="pt", add_special_tokens=True)
            input_length = inputs.shape[1]
            
            # 设置设备
            device = next(self.model.parameters()).device
            inputs = inputs.to(device)
            
            # 设置 max_new_tokens
            if max_new_tokens is None:
                max_new_tokens = min(512, max_length - input_length)
            
            # 准备停止标记（防止生成"用户:"等标记）
            stop_tokens = []
            # 尝试编码停止标记
            for stop_text in ["用户:", "助手:", "User:", "Assistant:", "\n用户:", "\n助手:"]:
                try:
                    stop_token_ids = self.tokenizer.encode(stop_text, add_special_tokens=False)
                    if stop_token_ids:
                        stop_tokens.extend(stop_token_ids)
                except:
                    pass
            
            # 生成回复
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.pad_token_id is None else self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    # 注意：不是所有模型都支持 stop_token_ids，这里先注释
                    # stop_token_ids=stop_tokens if stop_tokens else None,
                )
            
            # 解码输出（只解码新生成的部分）
            generated_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # 手动移除停止标记
            for stop_text in ["用户:", "助手:", "User:", "Assistant:"]:
                if stop_text in response:
                    response = response.split(stop_text)[0].strip()
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"生成回复失败: {e}")
            raise
    
    def _build_prompt_from_messages(self, messages: list) -> str:
        """从消息列表构建提示（当模型不支持 chat template 时）"""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                prompt_parts.append(f"用户: {content}")
            elif role == "assistant":
                prompt_parts.append(f"助手: {content}")
        
        prompt_parts.append("助手: ")
        return "\n".join(prompt_parts)
    


class LangChainModelScopeLLM:
    """将 ModelScope 模型包装为 LangChain LLM"""
    
    def __init__(self, model_loader: ModelScopeLoader):
        self.model_loader = model_loader
        
    def __call__(self, prompt, stop: Optional[list] = None, **kwargs) -> str:
        """调用模型生成回复
        
        Args:
            prompt: 可以是字符串或消息列表
        """
        max_length = kwargs.get("max_length", 2048)
        temperature = kwargs.get("temperature", 0.7)
        max_new_tokens = kwargs.get("max_new_tokens", None)
        
        response = self.model_loader.generate(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature,
            max_new_tokens=max_new_tokens
        )
        
        # 处理 stop 序列
        if stop:
            for stop_seq in stop:
                if stop_seq in response:
                    response = response.split(stop_seq)[0]
        
        return response
    
    def invoke(self, prompt, **kwargs) -> str:
        """LangChain 兼容的调用方法
        
        Args:
            prompt: 可以是字符串或消息列表
        """
        return self.__call__(prompt, **kwargs)

