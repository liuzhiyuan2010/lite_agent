import json
from typing import List, Dict, Optional, Any, Union
from .base import BaseLLM, ChatResponse, Message, ToolCall


class OllamaLLM(BaseLLM):
    def __init__(
            self,
            api_url: str = "http://localhost:11434/api",
            model: str = "llama3.2",
            temperature: float = 0.7,
            thinking: Union[bool, str] = False,
            keep_alive: str = "5m",
            logprobs: bool = False,
            top_logprobs: int = 0, **kwargs
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
            thinking=thinking, **kwargs
        )
        self.keep_alive = keep_alive
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs

    def chat(
            self,
            messages: List[Dict[str, str]],
            tools: Optional[List[Dict[str, Any]]] = None,
            stream: bool = False,
            format: Optional[Union[str, Dict]] = None, **kwargs
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
                "temperature": self.temperature, **self.kwargs.get("options", {}), **kwargs.get("options", {})
            },
            "think": self.thinking,
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
            prompt_eval_count = 0

            for line in response.iter_lines():
                if not line:
                    continue
                line_data = line.decode("utf-8").strip()
                if line_data.startswith("data: "):
                    try:
                        data = json.loads(line_data[6:])  # 替换eval为安全的json.loads
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
                                        parameters=tc.get("function", {}).get("arguments"),
                                        id=tc.get("id")
                                    )
                                    for tc in msg.get("tool_calls", [])
                                ]
                        if "done" in data:
                            done = data["done"]
                            done_reason = data.get("done_reason")
                            total_duration = data.get("total_duration")
                            eval_count = data.get("eval_count", 0)  # 输出token数
                            prompt_eval_count = data.get("prompt_eval_count", 0)  # 输入token数
                            break
                    except json.JSONDecodeError:
                        continue

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
                eval_count=eval_count,
                prompt_eval_count=prompt_eval_count
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
                            parameters=tc.get("function", {}).get("arguments"),
                            id=tc.get("id")
                        )
                    )

            return ChatResponse(
                message=Message(
                    role=msg_data.get("role", "assistant"),
                    content=msg_data.get("content", ""),
                    tool_calls=tool_calls,
                    thinking=msg_data.get("thinking",'')
                ),
                done=data.get("done", True),
                done_reason=data.get("done_reason"),
                total_duration=data.get("total_duration"),
                eval_count=data.get("eval_count", 0),  # 输出token数
                prompt_eval_count=data.get("prompt_eval_count", 0)  # 输入token数
            )