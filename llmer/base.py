import abc
import requests
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union


# ========== 数据结构定义 ==========
@dataclass
class ToolCall:
    """工具调用结构体"""
    function_name: str
    parameters: Dict[str, Any]
    id: Optional[str] = None


@dataclass
class Message:
    """消息结构体"""
    role: str  # system/user/assistant/tool
    content: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    thinking: Optional[str] = None  # thinking模式输出


@dataclass
class ChatResponse:
    """最终返回的响应结构体"""
    message: Message
    done: bool
    done_reason: Optional[str] = None
    total_duration: Optional[int] = None  # 纳秒
    eval_count: Optional[int] = None  # 输出token数
    prompt_eval_count: Optional[int] = None  # 输入token数


# ========== 基类抽象 ==========
class BaseLLM(abc.ABC):
    def __init__(
        self,
        api_url: str,
        model: str,
        temperature: float = 0.7,
        thinking: Union[bool, str] = False,** kwargs
    ):
        """
        初始化LLM基类
        :param api_url: API地址
        :param model: 模型名称
        :param temperature: 温度参数
        :param thinking: thinking模式 (True/False/"high"/"medium"/"low")
        :param kwargs: 其他自定义参数
        """
        self.api_url = api_url
        self.model = model
        self.temperature = temperature
        self.thinking = thinking
        self.kwargs = kwargs
        self.session = requests.Session()

    @abc.abstractmethod
    def chat(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,** kwargs
    ) -> ChatResponse:
        """
        核心聊天接口
        :param messages: 对话历史 [{"role": "...", "content": "..."}]
        :param tools: 工具定义列表
        :param stream: 是否流式返回
        :param kwargs: 额外请求参数
        :return: ChatResponse对象
        """
        pass

    def _validate_messages(self, messages: List[Dict[str, str]]) -> None:
        """校验消息格式"""
        required_keys = {"role", "content"}
        for idx, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise ValueError(f"第{idx}条消息必须是字典类型")
            if not required_keys.issubset(msg.keys()):
                raise ValueError(f"第{idx}条消息缺少必要字段: {required_keys - msg.keys()}")
            if msg["role"] not in ["system", "user", "assistant", "tool"]:
                raise ValueError(f"第{idx}条消息role不合法: {msg['role']}")

    def close(self):
        """关闭会话"""
        self.session.close()

    def __del__(self):
        self.close()