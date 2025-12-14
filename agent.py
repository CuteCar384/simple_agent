"""
LangGraph Agent 实现
"""
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
import logging

import torch
from model_loader import ModelScopeLoader, LangChainModelScopeLLM
from config import MODEL_NAME, MODEL_CACHE_DIR, MAX_ITERATIONS, TEMPERATURE, MAX_TOKENS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """Agent 状态定义"""
    messages: Annotated[Sequence[BaseMessage], add_messages]


class LangGraphAgent:
    """基于 LangGraph 的对话 Agent"""
    
    def __init__(self, model_name: str = MODEL_NAME, cache_dir: str = MODEL_CACHE_DIR):
        # 加载模型
        logger.info("初始化模型加载器...")
        self.model_loader = ModelScopeLoader(model_name, cache_dir)
        self.model_loader.load_model(use_gpu=torch.cuda.is_available())
        
        # 创建 LangChain LLM 包装器
        self.llm = LangChainModelScopeLLM(self.model_loader)
        
        # 构建图
        self.graph = self._build_graph()
        
    def _build_graph(self):
        """构建 LangGraph 状态图"""
        # 创建状态图
        workflow = StateGraph(AgentState)
        
        # 添加节点
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", self._call_tools)
        
        # 设置入口点
        workflow.set_entry_point("agent")
        
        # 添加条件边
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END
            }
        )
        
        # 工具执行后返回 agent
        workflow.add_edge("tools", "agent")
        
        # 编译图
        return workflow.compile()
    
    def _call_model(self, state: AgentState):
        """调用模型生成回复"""
        messages = state["messages"]
        
        # 转换为 chat template 格式
        chat_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                chat_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                chat_messages.append({"role": "assistant", "content": msg.content})
        
        # 调用模型（传入消息列表，让模型使用 chat template）
        response = self.llm.invoke(
            chat_messages,
            temperature=TEMPERATURE,
            max_length=MAX_TOKENS
        )
        
        # 清理响应：移除可能的重复格式
        response = self._clean_response(response)
        
        # 返回 AI 消息
        return {"messages": [AIMessage(content=response)]}
    
    def _clean_response(self, response: str) -> str:
        """清理响应，移除自我问答格式"""
        if not response:
            return ""
        
        # 移除可能的"用户:"和"助手:"标记
        lines = response.split('\n')
        cleaned_lines = []
        found_stop_marker = False
        
        for line in lines:
            line_stripped = line.strip()
            # 如果遇到"用户:"或"助手:"标记，停止（但保留当前行之前的内容）
            if line_stripped.startswith('用户:') or line_stripped.startswith('助手:'):
                found_stop_marker = True
                break
            if line_stripped.startswith('User:') or line_stripped.startswith('Assistant:'):
                found_stop_marker = True
                break
            # 检查是否包含对话格式（可能是模型开始自我问答）
            if '用户:' in line or '助手:' in line:
                # 如果这一行包含对话标记，只保留标记之前的内容
                for marker in ['用户:', '助手:', 'User:', 'Assistant:']:
                    if marker in line:
                        line = line.split(marker)[0]
                        found_stop_marker = True
                        break
                if found_stop_marker:
                    break
            cleaned_lines.append(line)
        
        result = '\n'.join(cleaned_lines).strip()
        
        # 移除末尾的重复结束语（如果文本较长且包含结束语）
        if len(result) > 100:  # 只对较长的文本进行清理
            end_phrases = [
                "希望这篇文章能给您带来一定的启发，也希望您能从中得到启示",
                "希望这篇文章能给您带来一定的启发",
                "如果有任何问题或需要进一步的帮助，请随时与我联系",
                "祝您学习愉快！",
            ]
            
            for phrase in end_phrases:
                if phrase in result:
                    # 找到所有出现位置
                    occurrences = []
                    start = 0
                    while True:
                        idx = result.find(phrase, start)
                        if idx == -1:
                            break
                        occurrences.append(idx)
                        start = idx + 1
                    
                    # 如果结束语出现多次，且最后一次在文本的后半部分，截断
                    if len(occurrences) > 1:
                        last_idx = occurrences[-1]
                        if last_idx > len(result) * 0.4:  # 在文本的后40%部分
                            result = result[:last_idx].strip()
                            break
        
        return result
    
    def _call_tools(self, state: AgentState):
        """调用工具（当前版本暂不使用工具）"""
        # 如果需要工具，可以在这里添加
        return {"messages": []}
    
    def _should_continue(self, state: AgentState) -> str:
        """判断是否继续执行"""
        # 简单版本：直接结束
        # 如果需要工具调用，可以在这里添加判断逻辑
        return "end"
    
    def chat(self, user_input: str, history: list = None) -> tuple[str, list]:
        """
        对话接口
        
        Args:
            user_input: 用户输入
            history: 对话历史，格式为 [(user_msg, ai_msg), ...]
        
        Returns:
            (ai_response, updated_history)
        """
        try:
            # 构建消息列表
            messages = []
            if history:
                for user_msg, ai_msg in history:
                    messages.append(HumanMessage(content=user_msg))
                    messages.append(AIMessage(content=ai_msg))
            
            # 添加当前用户消息
            messages.append(HumanMessage(content=user_input))
            
            # 执行图
            initial_state = {"messages": messages}
            result = self.graph.invoke(initial_state)
            
            # 获取最后一条 AI 消息
            last_message = result["messages"][-1]
            ai_response = last_message.content if hasattr(last_message, 'content') else str(last_message)
            
            # 更新历史
            if history is None:
                history = []
            history.append((user_input, ai_response))
            
            return ai_response, history
            
        except Exception as e:
            logger.error(f"对话处理失败: {e}")
            return f"抱歉，处理您的请求时出现错误: {str(e)}", history or []


# 全局 agent 实例
_agent_instance = None


def get_agent(model_name: str = MODEL_NAME, cache_dir: str = MODEL_CACHE_DIR) -> LangGraphAgent:
    """获取全局 agent 实例（单例模式）"""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = LangGraphAgent(model_name, cache_dir)
    return _agent_instance

