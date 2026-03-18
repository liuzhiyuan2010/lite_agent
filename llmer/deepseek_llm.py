import json
from typing import List, Dict, Optional, Any, Union
from .base import BaseLLM, ChatResponse, Message, ToolCall


class DeepseekLLM(BaseLLM):
    def __init__(
            self,
            api_url: str = "https://api.deepseek.com/v1/chat/completions",
            model: str = "deepseek-chat",
            temperature: float = 0.7,
            thinking: Union[bool, str] = False,
            api_key: str = "", **kwargs
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
            thinking=thinking, **kwargs
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
            think:  bool = False,
            temperature:    float=0.7,
            stream: bool = False, **kwargs
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
            "temperature": temperature,
            "stream": stream,
            "tools": tools if tools else [],
            "tool_choice": kwargs.get("tool_choice", "auto"), **self.kwargs, **kwargs
        }

        # 添加thinking模式（DeepSeek自定义参数示例）
        if self.thinking_param != "none":
            request_body["extra_parameters"] = {
                "thinking": self.thinking_param, **request_body.get("extra_parameters", {})
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
            eval_count = 0
            prompt_eval_count = 0

            for line in response.iter_lines():
                if not line:
                    continue
                line_data = line.decode("utf-8").strip()
                if line_data == "data: [DONE]":
                    done = True
                    break
                if line_data.startswith("data: "):
                    try:
                        data = json.loads(line_data[6:])  # 替换eval为安全的json.loads
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
                        # 流式中token计数可能在最后一条返回
                        if "usage" in data:
                            eval_count = data["usage"].get("completion_tokens", 0)
                            prompt_eval_count = data["usage"].get("prompt_tokens", 0)
                    except (json.JSONDecodeError, Exception):
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
                eval_count=eval_count,
                prompt_eval_count=prompt_eval_count
            )
        else:
            # 非流式响应
            data = response.json()
            #print(f'deepseek_llm: {data}')
            choice = data.get("choices", [{}])[0]
            msg_data = choice.get("message", {})
            usage = data.get("usage", {})

            # 解析工具调用
            tool_calls = []
            if "tool_calls" in msg_data:
                for tc in msg_data["tool_calls"]:
                    func_info = tc.get("function", {})
                    # ✅ 解析 arguments 字段（JSON 字符串 → dict）
                    try:
                        params = json.loads(func_info.get("arguments", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        params = {}
                    tool_calls.append(
                        ToolCall(
                            function_name=func_info.get("name"),
                            parameters=params,
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
                total_duration=None,  # DeepSeek未返回耗时，可自定义计算
                eval_count=usage.get("completion_tokens", 0),  # 输出token数
                prompt_eval_count=usage.get("prompt_tokens", 0)  # 输入token数
            )