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
    eval_count: Optional[int] = None  # 生成的token数


# ========== 基类抽象 ==========
class BaseLLM(abc.ABC):
    def __init__(
            self,
            api_url: str,
            model: str,
            temperature: float = 0.7,
            thinking: Union[bool, str] = False,
            **kwargs
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
            stream: bool = False,
            **kwargs
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


# ========== Ollama实现 ==========
class OllamaLLM(BaseLLM):
    def __init__(
            self,
            api_url: str = "http://localhost:11434/api",
            model: str = "llama3.2",
            temperature: float = 0.7,
            thinking: Union[bool, str] = False,
            keep_alive: str = "5m",
            logprobs: bool = False,
            top_logprobs: int = 0,
            **kwargs
    ):
        """
        Ollama LLM初始化
        :param api_url: Ollama API地址 (默认: http://localhost:11434/api)
        :param model: 模型名称 (默认: llama3.2)
        :param temperature: 温度参数
        :param thinking: thinking模式 (True/False/"high"/"medium"/"low")
        :param keep_alive: 模型保活时间
        :param logprobs: 是否返回token概率
        :param top_logprobs: 返回top N概率token
        :param kwargs: 其他参数
        """
        super().__init__(
            api_url=api_url,
            model=model,
            temperature=temperature,
            thinking=thinking,
            **kwargs
        )
        self.keep_alive = keep_alive
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs

    def chat(
            self,
            messages: List[Dict[str, str]],
            tools: Optional[List[Dict[str, Any]]] = None,
            stream: bool = False,
            format: Optional[Union[str, Dict]] = None,
            **kwargs
    ) -> ChatResponse:
        """
        Ollama聊天接口
        :param messages: 对话历史
        :param tools: 工具定义列表 (Ollama工具调用格式)
        :param stream: 是否流式返回 (Ollama默认True，这里默认False)
        :param format: 返回格式 (如"json"或JSON Schema)
        :param kwargs: 额外请求参数
        :return: ChatResponse
        """
        # 1. 校验输入
        self._validate_messages(messages)

        # 2. 构建请求体
        request_body = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                **self.kwargs.get("options", {}),
                **kwargs.get("options", {})
            },
            "thinking": self.thinking,
            "keep_alive": self.keep_alive,
            "logprobs": self.logprobs,
            "top_logprobs": self.top_logprobs
        }

        # 可选参数
        if tools:
            request_body["tools"] = tools
        if format:
            request_body["format"] = format

        # 3. 发送请求
        response = self.session.post(
            url=f"{self.api_url}/chat",
            json=request_body,
            stream=stream,
            timeout=kwargs.get("timeout", 60)
        )
        response.raise_for_status()

        # 4. 处理响应
        if stream:
            # 流式处理（简化版，可根据需要扩展）
            full_content = ""
            full_thinking = ""
            tool_calls = []
            done = False
            done_reason = None
            total_duration = 0
            eval_count = 0

            for line in response.iter_lines():
                if not line:
                    continue
                line_data = line.decode("utf-8").strip()
                if line_data.startswith("data: "):
                    data = eval(line_data[6:])  # 生产环境建议用json.loads
                    if "message" in data:
                        msg = data["message"]
                        if "content" in msg:
                            full_content += msg["content"]
                        if "thinking" in msg:
                            full_thinking += msg["thinking"]
                        if "tool_calls" in msg:
                            tool_calls = [
                                ToolCall(
                                    function_name=tc.get("function", {}).get("name"),
                                    parameters=tc.get("function", {}).get("parameters"),
                                    id=tc.get("id")
                                )
                                for tc in msg.get("tool_calls", [])
                            ]
                    if "done" in data:
                        done = data["done"]
                        done_reason = data.get("done_reason")
                        total_duration = data.get("total_duration")
                        eval_count = data.get("eval_count")
                        break

            return ChatResponse(
                message=Message(
                    role="assistant",
                    content=full_content,
                    tool_calls=tool_calls,
                    thinking=full_thinking
                ),
                done=done,
                done_reason=done_reason,
                total_duration=total_duration,
                eval_count=eval_count
            )
        else:
            # 非流式响应
            data = response.json()
            msg_data = data.get("message", {})

            # 解析工具调用
            tool_calls = []
            if "tool_calls" in msg_data:
                for tc in msg_data["tool_calls"]:
                    tool_calls.append(
                        ToolCall(
                            function_name=tc.get("function", {}).get("name"),
                            parameters=tc.get("function", {}).get("parameters"),
                            id=tc.get("id")
                        )
                    )

            return ChatResponse(
                message=Message(
                    role=msg_data.get("role", "assistant"),
                    content=msg_data.get("content", ""),
                    tool_calls=tool_calls,
                    thinking=msg_data.get("thinking")
                ),
                done=data.get("done", True),
                done_reason=data.get("done_reason"),
                total_duration=data.get("total_duration"),
                eval_count=data.get("eval_count")
            )


# ========== DeepSeek实现 (示例) ==========
class DeepseekLLM(BaseLLM):
    def __init__(
            self,
            api_url: str = "https://api.deepseek.com/v1/chat/completions",
            model: str = "deepseek-chat",
            temperature: float = 0.7,
            thinking: Union[bool, str] = False,
            api_key: str = "",
            **kwargs
    ):
        """
        DeepSeek LLM初始化
        :param api_url: DeepSeek API地址
        :param model: 模型名称
        :param temperature: 温度参数
        :param thinking: thinking模式 (DeepSeek需映射为自定义参数)
        :param api_key: DeepSeek API密钥
        :param kwargs: 其他参数
        """
        super().__init__(
            api_url=api_url,
            model=model,
            temperature=temperature,
            thinking=thinking,
            **kwargs
        )
        self.api_key = api_key
        # DeepSeek没有原生thinking模式，这里映射为自定义参数
        self.thinking_param = {
            True: "high",
            False: "none",
            "high": "high",
            "medium": "medium",
            "low": "low"
        }.get(thinking, "none")

    def chat(
            self,
            messages: List[Dict[str, str]],
            tools: Optional[List[Dict[str, Any]]] = None,
            stream: bool = False,
            **kwargs
    ) -> ChatResponse:
        """
        DeepSeek聊天接口
        :param messages: 对话历史
        :param tools: 工具定义列表 (DeepSeek格式)
        :param stream: 是否流式返回
        :param kwargs: 额外请求参数
        :return: ChatResponse
        """
        # 1. 校验输入
        self._validate_messages(messages)

        # 2. 构建请求体 (适配DeepSeek API格式)
        request_body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": stream,
            "tools": tools if tools else [],
            "tool_choice": kwargs.get("tool_choice", "auto"),
            **self.kwargs,
            **kwargs
        }

        # 添加thinking模式（DeepSeek自定义参数示例）
        if self.thinking_param != "none":
            request_body["extra_parameters"] = {
                "thinking": self.thinking_param,
                **request_body.get("extra_parameters", {})
            }

        # 3. 构建请求头
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # 4. 发送请求
        response = self.session.post(
            url=self.api_url,
            json=request_body,
            headers=headers,
            stream=stream,
            timeout=kwargs.get("timeout", 60)
        )
        response.raise_for_status()

        # 5. 处理响应
        if stream:
            # 流式处理（DeepSeek格式）
            full_content = ""
            full_thinking = ""
            tool_calls = []
            done = False
            done_reason = "stop"

            for line in response.iter_lines():
                if not line:
                    continue
                line_data = line.decode("utf-8").strip()
                if line_data == "data: [DONE]":
                    done = True
                    break
                if line_data.startswith("data: "):
                    try:
                        data = eval(line_data[6:])  # 生产环境建议用json.loads
                        if "choices" in data:
                            choice = data["choices"][0]
                            delta = choice.get("delta", {})
                            if "content" in delta:
                                full_content += delta["content"]
                            if "thinking" in delta:  # DeepSeek返回的thinking字段
                                full_thinking += delta["thinking"]
                            if "tool_calls" in delta:
                                tool_calls = [
                                    ToolCall(
                                        function_name=tc.get("function", {}).get("name"),
                                        parameters=tc.get("function", {}).get("parameters"),
                                        id=tc.get("id")
                                    )
                                    for tc in delta.get("tool_calls", [])
                                ]
                    except Exception as e:
                        continue

            return ChatResponse(
                message=Message(
                    role="assistant",
                    content=full_content,
                    tool_calls=tool_calls,
                    thinking=full_thinking
                ),
                done=done,
                done_reason=done_reason
            )
        else:
            # 非流式响应
            data = response.json()
            choice = data.get("choices", [{}])[0]
            msg_data = choice.get("message", {})

            # 解析工具调用
            tool_calls = []
            if "tool_calls" in msg_data:
                for tc in msg_data["tool_calls"]:
                    tool_calls.append(
                        ToolCall(
                            function_name=tc.get("function", {}).get("name"),
                            parameters=tc.get("function", {}).get("parameters"),
                            id=tc.get("id")
                        )
                    )

            # 解析thinking（DeepSeek自定义返回字段）
            thinking = msg_data.get("thinking") or choice.get("thinking", "")

            return ChatResponse(
                message=Message(
                    role=msg_data.get("role", "assistant"),
                    content=msg_data.get("content", ""),
                    tool_calls=tool_calls,
                    thinking=thinking
                ),
                done=True,
                done_reason=choice.get("finish_reason"),
                total_duration=data.get("usage", {}).get("total_duration"),
                eval_count=data.get("usage", {}).get("completion_tokens")
            )


# ========== 便捷导出 ==========
class LLMer:
    """统一入口类"""

    @staticmethod
    def init(
            llm_type: str,
            api_url: Optional[str] = None,
            model: str = "",
            temperature: float = 0.7,
            thinking: Union[bool, str] = False,
            **kwargs
    ) -> BaseLLM:
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
            return OllamaLLM(
                api_url=api_url or "http://localhost:11434/api",
                model=model or "llama3.2",
                temperature=temperature,
                thinking=thinking,
                **kwargs
            )
        elif llm_type.lower() == "deepseek":
            return DeepseekLLM(
                api_url=api_url or "https://api.deepseek.com/v1/chat/completions",
                model=model or "deepseek-chat",
                temperature=temperature,
                thinking=thinking,
                **kwargs
            )
        else:
            raise ValueError(f"不支持的LLM类型: {llm_type}")

    @staticmethod
    def chat(
            llm_type: str,
            messages: List[Dict[str, str]],
            api_url: Optional[str] = None,
            model: str = "",
            temperature: float = 0.7,
            thinking: Union[bool, str] = False,
            tools: Optional[List[Dict[str, Any]]] = None,
            stream: bool = False,
            **kwargs
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
        llm = LLMer.init(
            llm_type=llm_type,
            api_url=api_url,
            model=model,
            temperature=temperature,
            thinking=thinking, **kwargs
        )
        return llm.chat(
            messages=messages,
            tools=tools,
            stream=stream, **kwargs
        )


# ========== 使用示例 ==========
if __name__ == "__main__":
    # 示例1: 使用Ollama
    ollama_answer = LLMer.chat(
        llm_type="ollama",
        model="llama3.2",
        temperature=0.8,
        thinking=True,  # 开启thinking模式
        messages=[
            {"role": "system", "content": "你是一个助手，简要回答问题"},
            {"role": "user", "content": "什么是LLM？"}
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "搜索网络信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "搜索关键词"}
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
    )
    print("Ollama响应:")
    print(f"角色: {ollama_answer.message.role}")
    print(f"内容: {ollama_answer.message.content}")
    print(f"思考过程: {ollama_answer.message.thinking}")
    print(f"工具调用: {ollama_answer.message.tool_calls}")

    # 示例2: 先初始化再调用（适合多次对话）
    deepseek_llm = LLMer.init(
        llm_type="deepseek",
        api_key="your-deepseek-api-key",
        model="deepseek-chat",
        thinking="medium"
    )
    deepseek_answer = deepseek_llm.chat(
        messages=[
            {"role": "user", "content": "如何实现工具调用？"}
        ],
        stream=False
    )
    print("\nDeepSeek响应:")
    print(f"内容: {deepseek_answer.message.content}")
    print(f"思考过程: {deepseek_answer.message.thinking}")