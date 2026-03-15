import subprocess
import sys
import os
import re
from typing import Optional, List, Dict

# ========== 全局配置（可根据需求调整） ==========
# 绝对禁止的危险命令（小写）
ABSOLUTE_BAN_COMMANDS = {"sudo", "su", "kill", "format", "shutdown", "reboot", "fdisk", "mkfs",
                         "rm -rf", "chmod", "chown", "passwd", "useradd", "userdel"}
# 写/删操作关键字（小写）
WRITE_OP_KEYWORDS = {">", ">>", "touch", "mkdir", "rm", "del", "erase", "copy", "cp",
                     "move", "mv", "echo", "printf", "sed", "rmdir", "unlink"}
# 允许执行写/删操作的目录（默认：当前目录）
ALLOWED_WRITE_DIRS = [os.path.normpath(os.getcwd())]
# 命令执行超时时间（秒）
EXEC_TIMEOUT = 600

# ========== 核心 shell_cmd 函数 ==========
def run_shell_function(command_string: str, cwd: Optional[str] = None, allowed_dirs: Optional[List[str]] = None) -> str:
    """
    专门执行 Shell 命令的函数（适配 LLMer 调用）
    :param command_string: 要执行的 Shell 命令字符串
    :param cwd: 命令执行的工作目录（默认：当前目录）
    :param allowed_dirs: 允许写/删操作的目录列表（默认使用全局配置）
    :return: 标准化的执行结果（字符串，可直接反馈给 LLM）
    """
    command = command_string
    # 1. 初始化参数
    cwd = cwd or os.getcwd()
    allowed_dirs = allowed_dirs or ALLOWED_WRITE_DIRS
    cmd_lower = command.lower().strip()

    # 2. 安全校验：拦截危险命令
    if any(ban_cmd in cmd_lower for ban_cmd in ABSOLUTE_BAN_COMMANDS):
        return f"❌ 安全拦截：禁止执行危险命令「{command}」"

    # 3. 安全校验：写/删操作的目录白名单
    if any(key in cmd_lower for key in WRITE_OP_KEYWORDS):
        # 提取目标路径
        target_path = _extract_target_path(command) or cwd
        # 解析绝对路径
        full_target = os.path.abspath(os.path.join(cwd, target_path)) if not os.path.isabs(target_path) else os.path.abspath(target_path)
        # 检查是否在白名单内
        is_allowed = False
        for dir in allowed_dirs:
            safe_dir = dir.rstrip(os.sep) + os.sep
            if full_target.startswith(safe_dir) or full_target == dir.rstrip(os.sep):
                is_allowed = True
                break
        if not is_allowed:
            return (
                f"❌ 安全拦截：写/删操作仅允许在以下目录执行\n"
                f"允许目录：{allowed_dirs}\n"
                f"尝试操作路径：{full_target}"
            )

    # 4. 安全校验：禁止路径遍历（..）
    if ".." in cmd_lower:
        return f"❌ 安全拦截：禁止使用 '..' 访问上级目录"

    # 5. 执行命令（跨平台适配）
    try:
        # 适配 Windows（Git Bash/cmd）和 Linux/macOS（bash）
        if sys.platform == "win32":
            bash_path = "E:\\Program Files (x86)\\Git\\bin\\bash.exe"
            shell_cmd_list = [bash_path, "-c", command] if os.path.exists(bash_path) else ["cmd.exe", "/c", command]
        else:
            shell_cmd_list = ["/bin/bash", "-c", command]

        # 执行命令
        result = subprocess.run(
            shell_cmd_list,
            capture_output=True,
            text=True,
            timeout=EXEC_TIMEOUT,
            encoding="utf-8",
            errors="ignore",
            cwd=cwd
        )
        print(f'cmd execute: {result}')
        # 6. 格式化执行结果
        if result.returncode == 0:
            return f"✅ 命令执行成功：\n{result.stdout or '（无输出）'}"
        else:
            return f"❌ 命令执行失败（退出码：{result.returncode}）：\n{result.stderr or result.stdout}"

    except subprocess.TimeoutExpired:
        return f"❌ 命令执行超时（超过{EXEC_TIMEOUT}秒）"
    except Exception as e:
        return f"❌ 命令执行异常：{str(e)}"

# ========== 辅助函数：提取命令中的目标路径 ==========
def _extract_target_path(command: str) -> Optional[str]:
    """从命令中提取目标文件/目录路径（内部辅助函数）"""
    cmd_lower = command.lower().strip()

    # 处理重定向（>、>>）
    if ">" in cmd_lower:
        parts = cmd_lower.split(">")
        if len(parts) > 1:
            target = parts[-1].split("|")[0].strip()
            return target.strip("\"' ")

    # 处理显式路径命令（touch、mkdir 等）
    tokens = re.split(r"\s+", cmd_lower)
    for token in tokens:
        t_lower = token.strip("\"' ")
        if t_lower.startswith("-") or t_lower in WRITE_OP_KEYWORDS:
            continue
        if re.match(r"^[a-zA-Z0-9._/\\~-]+$", t_lower):
            return t_lower
    return None

# ========== 适配 LLMer 的工具定义函数（可选） ==========
def get_shell_tool_definition() -> Dict:
    """返回 LLMer 可识别的 Shell 工具定义（供 LLMer.chat 的 tools 参数使用）"""
    return {
        "type": "function",
        "function": {
            "name": "run_shell_function",
            "description": f"""
                执行Shell命令的工具，必须传入 command_string 参数（要执行的Shell命令字符串）
            """.replace("\n", " ").strip(),
            "parameters": {
                "type": "object",
                "required": ["command_string"],
                "properties": {
                    "command_string": {
                        "type": "string",
                        "description": "要执行的Shell命令（如ls -l、pwd等）"
                    },
                    "cwd": {
                        "type": "string",
                        "description": "命令执行的工作目录（默认：当前目录）"
                    }
                }

            }
        }
    }
