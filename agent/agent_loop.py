
import json
import os
from typing import List, Dict, Optional
from colorama import Fore, Style
from tools import  get_shell_tool_definition,get_tool_by_name
from llmer import LLMer

class AgentLoop:
    def __init__(self, config_path: Optional[str] = None):
        self._num_input_token = 0
        self._num_output_token = 0
        self.llm_cfg = self._load_config(config_path)

        self.max_iterations = 15  # 防止工具调用死循环

        #init_working_dir(config_path)
        self.ALLOWED_WRITE_DIR = self.llm_cfg.get('ALLOWED_WRITE_DIR')

        # 获取工具定义 (Schema)，LLM 只需要知道这些
        self.tools_definitions = [get_shell_tool_definition()]

        print(Fore.CYAN + f"[初始化] 已加载 {len(self.tools_definitions)} 个工具")

        self.memory = f' 你只能使用类linux的shell命令，应返回linux风格(E/proj/的路径)，而不是E:\\proj\\。当输入错误命令时，修改后再生成工具调用'
        self.is_ollama_backend = False
        if self.llm_cfg.get('model_type') == 'ollama':
            print(f'ollama backend!!!')
            self.is_ollama_backend = True
            self.llm_backend = LLMer(llm_type="ollama",
                    model=self.llm_cfg.get('model',"qwen3.5:397b-cloud"),
                    temperature=self.llm_cfg.get('temperature',0.6),
                    thinking=False)

        elif self.llm_cfg.get('model_type') == 'deepseek':
            print(f'deepseek backend!!!')
            self.llm_backend = LLMer(llm_type="deepseek",
                api_key=os.getenv('DEEPSEEK_API_KEY'),  # 替换为实际API Key
                model="deepseek-chat",
                temperature=self.llm_cfg.get("temperature", 0.1),
                thinking=False)
        else:
            print(f'ERROR   {self.llm_cfg.get('model')} not supported.')
            self.llm_backend = None

    def _load_config(self, path: Optional[str]) -> dict:
        if not path:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(script_dir, "config", "conf.json")

        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                llmcfg= json.load(f)
                model_type = llmcfg['llm']["model_type"]
                model_cfg = llmcfg[model_type]
                return {
                    'model_type': model_type,
                    "model": model_cfg['model'],
                    "temperature":model_cfg['temperature'],
                    "num_predict": 2000,
                    'ALLOWED_WRITE_DIR': llmcfg['allowed_write_dirs']
                }

        print(f'{path} not exists')
        return {
            'model_type':None,
            "model": "Not model",
            "temperature": 0.1,
            "num_predict": 2000,
            'ALLOWED_WRITE_DIR':None
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
                response = self.llm_backend.chat(
                    messages=messages,
                    tools=self.tools_definitions,  # 注入工具定义
                    stream=False,
                    #thinking = self.llm_cfg .get('think',True)
                )

                # 统计 Token
                if hasattr(response, 'prompt_eval_count'):
                    self._num_input_token += response.prompt_eval_count
                if hasattr(response, 'eval_count'):
                    self._num_output_token += response.eval_count

                message = response.message
                #messages.append(message)  # 记录模型回复
                #print(f'llm response : {response.message}')
                # 情况 A: 模型直接回答
                if message.content and not message.tool_calls:
                    final_response = message.content
                    break
                # 情况 B: 模型请求调用工具
                if message.tool_calls:
                    print(Fore.YELLOW + f"\n[第{iteration}轮] 模型请求调用工具...")

                    # ✅ 修复：将 Message 对象转为 dict 格式
                    assistant_message = {
                        "role": "assistant",
                        "content": "",  # 有 tool_calls 时 content 通常为空
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function_name,
                                    "arguments": tc.parameters  if self.is_ollama_backend else json.dumps(tc.parameters) #json.dumps(tc.parameters)  # ✅ deepseek 这里参数需序列化为 JSON 字符串
                                }
                            }
                            for tc in message.tool_calls
                        ]
                    }
                    messages.append(assistant_message)

                    for tool_call in message.tool_calls:
                        func_name = tool_call.function_name
                        func_args = tool_call.parameters['command_string']

                        print(Fore.CYAN + f"  -> 调用: {func_name}({func_args})")

                        try:
                            # 【关键解耦点】通过注册中心查找并执行工具
                            # LLMClient 不需要知道 run_shell_command 的具体实现
                            tool_instance = get_tool_by_name(func_name)
                            result_output = tool_instance(func_args)

                            print(
                                Fore.GREEN + f"  <- 执行结果: {result_output[:100]}{'...' if len(result_output) > 100 else ''}")
                            if result_output.find("Error")>=0:
                                print ("执行出错")
                                messages.append({ "role":"tool",
                                        "content":'命令执行错误，请根据下面内容修正后重新生成工具调用\n'+result_output,
                                         "tool_call_id": tool_call.id
                                        }
                                )
                            else:
                                # 将结果反馈给模型
                                messages.append({  "role": "tool",
                                    "content": result_output,
                                    "tool_call_id": tool_call.id
                                })

                        except ValueError as e:
                            error_msg = f"Error: 找不到工具 {func_name}"
                            print(Fore.RED + f"  <- {error_msg}")
                            messages.append({"role": "tool", "content": error_msg,"tool_call_id": tool_call.id })
                        except Exception as e:
                            error_msg = f"Error: 工具执行失败 - {str(e)}"
                            print(Fore.RED + f"  <- {error_msg}")
                            messages.append({"role": "tool", "content": error_msg,"tool_call_id": tool_call.id })


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