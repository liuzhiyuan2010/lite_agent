import ollama
import json
import os
from typing import List, Dict, Optional
from colorama import Fore, Style
import sys
# 引入工具模块
from tools import get_tools_definitions, get_tool_by_name,init_working_dir


class LLMClient:
    def __init__(self, config_path: Optional[str] = None):
        self._num_input_token = 0
        self._num_output_token = 0
        self.config = self._load_config(config_path)
        self.llm_cfg = self.config['llm']
        self.max_iterations = 15  # 防止工具调用死循环

        init_working_dir(self.config.get('allowed_write_dirs',[]))

        # 获取工具定义 (Schema)，LLM 只需要知道这些
        self.tools_definitions = get_tools_definitions()

        print(Fore.CYAN + f"[初始化] 已加载 {len(self.tools_definitions)} 个工具")

        self.memory = f'the os is {sys.platform}' + '\n 当用户输入错误命令时，修改后再生产工具调用' \
                + '\n 当要修改当前目录下的文件时，先执行git commit 进行提交'

    def _load_config(self, path: Optional[str]) -> dict:
        if not path:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(script_dir, "config", "conf.json")

        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "model": "Not model",
            "temperature": 0.1,
            "num_predict": 2000
        }

    def chat(self, user_message: str, system_prompt: str = "") -> str:
        """
        处理包含工具调用的完整对话流程
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if self.memory :
            messages.append({"role": "system", "content": self.memory})

        messages.append({"role": "user", "content": user_message})

        final_response = ""

        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1

            try:
                response = ollama.chat(
                    model=self.llm_cfg .get("model"),
                    messages=messages,
                    tools=self.tools_definitions,  # 注入工具定义
                    stream=False,
                    think = self.llm_cfg .get('think',True),
                    options={
                        "temperature": self.llm_cfg .get("temperature", 0.1),
                        "num_predict": self.llm_cfg .get("num_predict", 2000)
                    }
                )

                # 统计 Token
                if hasattr(response, 'prompt_eval_count'):
                    self._num_input_token += response.prompt_eval_count
                if hasattr(response, 'eval_count'):
                    self._num_output_token += response.eval_count

                message = response.message
                messages.append(message)  # 记录模型回复

                print(response.message)
                # 情况 A: 模型直接回答
                if message.content:
                    final_response = message.content
                    break
                # 情况 B: 模型请求调用工具
                if message.tool_calls:
                    print(Fore.YELLOW + f"\n[第{iteration}轮] 模型请求调用工具...")

                    for tool_call in message.tool_calls:
                        func_name = tool_call.function.name
                        func_args = tool_call.function.arguments

                        print(Fore.CYAN + f"  -> 调用: {func_name}({func_args})")

                        try:
                            # 【关键解耦点】通过注册中心查找并执行工具
                            # LLMClient 不需要知道 run_shell_command 的具体实现
                            tool_instance = get_tool_by_name(func_name)
                            result_output = tool_instance.execute(**func_args)

                            print(
                                Fore.GREEN + f"  <- 执行结果: {result_output[:100]}{'...' if len(result_output) > 100 else ''}")
                            if result_output.find("Error")>=0:
                                print ("执行出错")
                                messages.append({ "role":"tool",
                                        "content":'命令执行错误，请根据下面内容修正后重新生成工具调用\n'+result_output
                                    }
                                )
                            else:
                                # 将结果反馈给模型
                                messages.append({  "role": "tool",
                                    "content": result_output,
                                })

                        except ValueError as e:
                            error_msg = f"Error: 找不到工具 {func_name}"
                            print(Fore.RED + f"  <- {error_msg}")
                            messages.append({"role": "tool", "content": error_msg})
                        except Exception as e:
                            error_msg = f"Error: 工具执行失败 - {str(e)}"
                            print(Fore.RED + f"  <- {error_msg}")
                            messages.append({"role": "tool", "content": error_msg})


            except Exception as e:
                return f"LLM 服务错误: {str(e)}"

        return final_response

    @property
    def stats(self):
        return {
            "input_tokens": self._num_input_token,
            "output_tokens": self._num_output_token,
            "total_tokens": self._num_input_token + self._num_output_token
        }