import subprocess
import sys
import os
from .base import BaseTool
import json
import re

class ShellCommandTool(BaseTool):
    def __init__(self):
        self.allowed_write_dirs = []
        self.exec_time_out = 60*10

    def init_working_dir(self,config_path: str):
        self._load_config(config_path)
        print('allowed write dir:')
        for d in self.allowed_write_dirs:
            print(f"  {d}")

    def _load_config(self, config_path: str):
        """加载配置并自动注入当前目录"""

        # 1. 首先，无条件将【当前工作目录】加入白名单
        current_dir = os.getcwd()
        # 规范化路径 (去除末尾斜杠，统一格式)
        self.allowed_write_dirs.append(os.path.normpath(current_dir))

        # 2. 尝试加载配置文件中的额外目录
        try:
            # 处理相对路径
            if not os.path.isabs(config_path):
                if not os.path.exists(config_path):
                    alt_path = os.path.join("..", config_path)
                    if os.path.exists(alt_path):
                        config_path = alt_path

            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    extra_dirs = config.get("allowed_write_dirs", [])

                    for d in extra_dirs:
                        # 展开用户路径 (~) 并转为绝对路径
                        abs_d = os.path.abspath(os.path.expanduser(d))
                        norm_d = os.path.normpath(abs_d)

                        # 去重：如果配置里的目录和当前目录一样，就不重复添加了
                        if norm_d not in self.allowed_write_dirs:
                            self.allowed_write_dirs.append(norm_d)
            else:
                print(f"[Warning] 配置文件未找到: {config_path}，仅使用当前目录作为白名单。")

        except Exception as e:
            print(f"[Warning] 解析安全配置失败: {e}，仅使用当前目录作为白名单。")

        # 打印最终生效的白名单 (调试用，生产环境可注释)
        print(f"[Security] 允许写入的目录列表: {self.allowed_write_dirs}")




    @property
    def name(self) -> str:
        return "run_shell_command"

    @property
    def description(self) -> str:
        dirs_str = ", ".join(self.allowed_write_dirs) if self.allowed_write_dirs else "无 (禁止所有写操作)"
        return (
            f"在终端执行 Shell 命令。\n"
            f"**重要安全限制**:\n"
            f"1. **写/删除操作** (如 echo >, touch, rm, del, mkdir) **只能**在以下目录进行: [{dirs_str}]\n"
            f"2. **读操作** (如 cat, dir, ls, ping) \n"
            f"3. 严禁执行 sudo, kill, format, shutdown 等系统级危险命令。\n"
            f"如果用户请求在非白名单目录进行写/删操作，请直接拒绝并说明原因。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "要执行的具体命令字符串，例如 'ls -l' 或 'dir'"
                }
            },
            "required": ["command"]
        }





    def execute(self, command: str, **kwargs) -> str:

        """执行命令并返回输出"""
        check_res = self.check_refuse_cmd(command)
        if 'cmd_check_ok'  not in check_res:
            print("refuse cmd")
            return check_res
        # 2. 准备执行环境
        try:
            if sys.platform == 'win32':
                #print("OS : win32")
                shell_cmd = command
            else:
                shell_cmd = ['/bin/bash', command]

            # 限制超时，防止死循环
            result = subprocess.run(
                shell_cmd,
                capture_output=True,
                text=True,
                shell=True,
                timeout=self.exec_time_out,
                encoding='utf-8',
                # 【关键修改 2】遇到无法解码的字符，忽略它而不是报错崩溃
                errors='ignore',
                cwd=os.getcwd()  # 在当前工作目录执行
            )
            print(f'exec res : {result}')
            if result.returncode == 0:
                return result.stdout if result.stdout else "命令执行成功，无输出。"
            else:
                return f"Error (Exit Code {result.returncode}): {result}"

        except subprocess.TimeoutExpired:
            return "Error: 命令执行超时 (>15秒)"
        except Exception as e:
            return f"Error: 执行异常 - {str(e)}"






    def check_refuse_cmd(self,command):
        cmd_lower = command.lower()

        # 1. 绝对禁止的危险命令 (无论在哪里都不行)
        absolute_bans = ['sudo ', 'su ', 'kill ', 'format ', 'shutdown ', 'reboot ', 'fdisk ', 'mkfs ']
        if any(k in cmd_lower for k in absolute_bans):
            return "Error: 安全拦截 - 检测到绝对禁止的系统级危险命令。"

        # 2. 定义写/删操作的关键字
        write_keywords = ['>', '>>', 'touch ', 'mkdir ', 'rm ', 'del ', 'erase ', 'copy ', 'cp ', 'move ', 'mv ',
                          'echo ', 'printf ']
        is_write_op = any(k in cmd_lower for k in write_keywords)

        # 3. 如果是写/删操作，进行路径白名单校验
        if is_write_op:
            if not self.allowed_write_dirs:
                return "Error: 安全拦截 - 配置文件未设置允许写入的目录，禁止所有写操作。"

            target_path = self._extract_target_path(command)

            if target_path:
                # 解析目标路径的绝对路径
                # 如果目标是相对路径，基于当前工作目录解析
                if not os.path.isabs(target_path):
                    full_target = os.path.abspath(os.path.join(os.getcwd(), target_path))
                else:
                    full_target = os.path.abspath(target_path)

                # 检查是否在任何一个允许的目录内
                is_allowed = False
                for allowed_dir in self.allowed_write_dirs:
                    # 确保 allowed_dir 以分隔符结尾，防止 /data 匹配 /database
                    safe_allowed = allowed_dir.rstrip(os.sep) + os.sep
                    if full_target.startswith(safe_allowed) or full_target == allowed_dir.rstrip(os.sep):
                        is_allowed = True
                        break

                if not is_allowed:
                    return (
                        f"Error: 安全拦截 - 写/删除操作被限制在特定目录。\n"
                        f"尝试操作路径: {full_target}\n"
                        f"允许的目录列表: {self.allowed_write_dirs}\n"
                        f"请在允许的目录内重试，或先 cd 到允许目录。"
                    )
            else:
                # 如果无法提取具体路径（例如只是 'echo hello > output.txt' 但没有明确路径，或者是管道操作）
                # 保守策略：检查当前工作目录是否在白名单内
                cwd = os.getcwd()
                is_cwd_allowed = any(cwd.startswith(d.rstrip(os.sep) + os.sep) or cwd == d.rstrip(os.sep)
                                     for d in self.allowed_write_dirs)
                if not is_cwd_allowed:
                    return (
                        f"Error: 安全拦截 - 当前工作目录 ({cwd}) 不在允许写入的列表中。\n"
                        f"允许的目录: {self.allowed_write_dirs}\n"
                        f"请先切换到允许目录 (cd <allowed_dir>) 再执行写操作。"
                    )

        # 4. 基础路径遍历拦截 (防止读操作也跳出)
        if '..' in command:
            # 例外：如果 .. 是在允许目录内部的相对跳转 (较难精确判断，保守起见直接拦截包含 .. 的写操作)
            if is_write_op:
                return "Error: 安全拦截 - 写操作中禁止使用 '..' 路径遍历。"
            # 读操作中的 .. 也尽量拦截，除非逻辑非常清晰，这里为了安全统一拦截
            return "Error: 安全拦截 - 禁止使用 '..' 访问上级目录。"
        return "cmd_check_ok"

    def _extract_target_path(self, command: str) -> str:
        """尝试从命令中提取目标文件/目录路径"""
        cmd_lower = command.lower()

        # 情况 A: 重定向操作 (>, >>)
        if '>' in command:
            parts = command.split('>')
            if len(parts) > 1:
                target = parts[-1].strip()
                target = target.split('|')[0].strip()
                return target.strip('"\'')

        # 情况 B: 明确的路径命令
        write_cmds = ['touch', 'mkdir', 'rm', 'del', 'cp', 'copy', 'mv', 'move']
        tokens = command.replace('\t', ' ').split()

        for i, token in enumerate(tokens):
            t_lower = token.lower().strip('"\'')
            if t_lower in write_cmds:
                continue
            if t_lower.startswith('-'):
                continue
            if re.search(r'[a-zA-Z0-9._/\\]', t_lower):
                return t_lower

        return None