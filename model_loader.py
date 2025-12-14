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
                # GPU 模式
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_dir,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
            else:
                # CPU 模式
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_dir,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    low_cpu_mem_usage=True
                )
            
            logger.info("模型加载完成")
            return self.model, self.tokenizer
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def generate(self, prompt: str, max_length: int = 2048, temperature: float = 0.7, max_new_tokens: int = None):
        """生成回复"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("模型未加载，请先调用 load_model()")
        
        try:
            # 使用 apply_chat_template 处理对话格式（如果支持）
            if hasattr(self.tokenizer, 'apply_chat_template') and 'chat' in prompt.lower():
                # 对于支持 chat template 的模型
                messages = self._parse_chat_messages(prompt)
                if messages:
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                else:
                    formatted_prompt = prompt
            else:
                formatted_prompt = prompt
            
            # 编码输入
            inputs = self.tokenizer.encode(formatted_prompt, return_tensors="pt", add_special_tokens=True)
            input_length = inputs.shape[1]
            
            # 设置设备
            device = next(self.model.parameters()).device
            inputs = inputs.to(device)
            
            # 设置 max_new_tokens
            if max_new_tokens is None:
                max_new_tokens = max_length - input_length
            
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
                )
            
            # 解码输出（只解码新生成的部分）
            generated_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"生成回复失败: {e}")
            # 如果出错，尝试简单方法
            try:
                inputs = self.tokenizer.encode(prompt, return_tensors="pt")
                device = next(self.model.parameters()).device
                inputs = inputs.to(device)
                input_length = inputs.shape[1]
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_new_tokens=min(512, max_length - input_length),
                        temperature=temperature,
                        do_sample=temperature > 0,
                    )
                
                generated_tokens = outputs[0][input_length:]
                response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                return response.strip()
            except Exception as e2:
                logger.error(f"备用生成方法也失败: {e2}")
                raise
    
    def _parse_chat_messages(self, prompt: str):
        """解析对话消息（简单实现）"""
        # 这是一个简单的实现，可以根据实际需求改进
        messages = []
        lines = prompt.split('\n')
        current_role = None
        current_content = []
        
        for line in lines:
            if line.startswith('用户:'):
                if current_role:
                    messages.append({"role": current_role, "content": "\n".join(current_content)})
                current_role = "user"
                current_content = [line.replace('用户:', '').strip()]
            elif line.startswith('助手:'):
                if current_role:
                    messages.append({"role": current_role, "content": "\n".join(current_content)})
                current_role = "assistant"
                current_content = [line.replace('助手:', '').strip()]
            else:
                if current_content:
                    current_content.append(line.strip())
        
        if current_role:
            messages.append({"role": current_role, "content": "\n".join(current_content)})
        
        return messages


class LangChainModelScopeLLM:
    """将 ModelScope 模型包装为 LangChain LLM"""
    
    def __init__(self, model_loader: ModelScopeLoader):
        self.model_loader = model_loader
        
    def __call__(self, prompt: str, stop: Optional[list] = None, **kwargs) -> str:
        """调用模型生成回复"""
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
    
    def invoke(self, prompt: str, **kwargs) -> str:
        """LangChain 兼容的调用方法"""
        return self.__call__(prompt, **kwargs)

