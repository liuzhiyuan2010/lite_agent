from typing import List, Dict, Optional, Any, Union
from .base import ChatResponse
from .ollama_llm import OllamaLLM
from .deepseek_llm import DeepseekLLM


class LLMer:
    """统一入口类"""
    def __init__(self,
        llm_type: str,
        api_url: Optional[str] = None,
        model: str = "",
        temperature: float = 0.7,
        thinking: Union[bool, str] = False,** kwargs
    ):
        """
        初始化LLM实例
        :param llm_type: 类型 ("ollama" / "deepseek")
        :param api_url: API地址
        :param model: 模型名称
        :param temperature: 温度
        :param thinking: thinking模式
        :param kwargs: 其他参数（如api_key等）
        :return: LLM实例
        """
        if llm_type.lower() == "ollama":
            self.llm =  OllamaLLM(
                api_url=api_url or "http://localhost:11434/api",
                model=model,
                temperature=temperature,
                thinking=thinking,** kwargs
            )
        elif llm_type.lower() == "deepseek":
            self.llm= DeepseekLLM(
                api_url=api_url or "https://api.deepseek.com/v1/chat/completions",
                model=model or "deepseek-chat",
                temperature=temperature,
                thinking=thinking,** kwargs
            )
        else:
            raise ValueError(f"不支持的LLM类型: {llm_type}")


    def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,** kwargs
    ) -> ChatResponse:
        """
        快捷聊天接口
        :param llm_type: 类型 ("ollama" / "deepseek")
        :param messages: 对话历史
        :param api_url: API地址
        :param model: 模型名称
        :param temperature: 温度
        :param thinking: thinking模式
        :param tools: 工具列表
        :param stream: 是否流式
        :param kwargs: 其他参数
        :return: ChatResponse
        """

        return self.llm.chat(
            messages=messages,
            tools=tools,
            stream=stream,** kwargs
        )