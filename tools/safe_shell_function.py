import subprocess
import sys
import os
import re
import shlex
from typing import Optional, List, Dict, Set, Tuple

# ========== 全局配置 ==========

# 1. 允许执行的命令白名单 (小写)
ALLOWED_COMMANDS: Set[str] = {
    'ls', 'find', 'pwd', 'cat', 'head', 'tail',
    'grep', 'awk', 'sed', 'wc', 'echo', 'touch',
    'curl', 'wget', 'ping','python'
}

# 2. 默认允许的目录列表
DEFAULT_ALLOWED_DIRS = [os.path.normpath(os.getcwd())]

# 3. 命令执行超时时间 (秒)
EXEC_TIMEOUT = 60

# 4. 用于分割命令链的正则表达式
# 匹配空格周围的 |, ;, &&, ||
# 注意：这里不处理单引号/双引号内的分隔符，shlex 会处理引号，我们主要处理顶层逻辑
COMMAND_SPLIT_PATTERN = re.compile(r'\s*(?:\|\||\|\|&&|;|\|)\s*')


# ========== 核心 Shell 工具函数 ==========

def run_shell_function(
        command_string: str,
        cwd: Optional[str] = None,
        allowed_dirs: Optional[List[str]] = None
) -> str:
    """
    【安全 Shell 执行工具 - 支持命令链】
    支持 |, ;, && 连接多个命令，但会对链中的每个命令进行独立的安全校验。

    :param command_string: 要执行的 Shell 命令字符串
    :param cwd: 命令执行的工作目录
    :param allowed_dirs: 允许操作的文件/目录绝对路径列表
    :return: 标准化的执行结果字符串
    """

    # 1. 初始化参数
    work_dir = os.path.abspath(cwd or os.getcwd())

    if allowed_dirs is None:
        safe_dirs = DEFAULT_ALLOWED_DIRS
    else:
        safe_dirs = [os.path.abspath(p) for p in allowed_dirs]

    cmd_raw = command_string.strip()
    if not cmd_raw:
        return "❌ 错误：命令不能为空"

    # 2. 预处理：提取并校验命令链
    # 我们不能直接用 shlex 分割整个字符串因为要保留 | ; && 的逻辑结构给 bash
    # 策略：手动分割命令链，对每一段进行独立校验

    # 简单的分割逻辑：我们需要找到不在引号内的 |, ;, &&
    sub_commands = _smart_split_commands(cmd_raw)

    if not sub_commands:
        return "❌ 错误：无法解析命令"

    # 3. 逐个校验子命令
    for i, sub_cmd in enumerate(sub_commands):
        sub_cmd = sub_cmd.strip()
        if not sub_cmd:
            continue

        # A. 解析根命令
        try:
            # 使用 shlex 分割当前子命令以获取第一个 token
            parts = shlex.split(sub_cmd)
            if not parts:
                continue
            root_cmd = parts[0].lower()
        except ValueError as e:
            return f"❌ 安全拦截：子命令 [{i + 1}] 语法解析错误: {str(e)}"

        # B. 白名单校验
        if root_cmd not in ALLOWED_COMMANDS:
            return (
                f"❌ 安全拦截：子命令 '{root_cmd}' 不在允许的白色名单中。\n"
                f"出错片段: {sub_cmd[:50]}...\n"
                f"允许的指令: {', '.join(sorted(ALLOWED_COMMANDS))}"
            )

        # C. 路径安全校验
        extracted_paths = _extract_paths_from_command(sub_cmd, work_dir)
        for path in extracted_paths:
            if os.path.isabs(path):
                abs_path = os.path.normpath(path)
            else:
                abs_path = os.path.normpath(os.path.join(work_dir, path))

            is_allowed = False
            for safe_dir in safe_dirs:
                safe_dir_norm = os.path.normpath(safe_dir)
                # 精确匹配或作为子目录
                if abs_path == safe_dir_norm or abs_path.startswith(safe_dir_norm + os.sep):
                    is_allowed = True
                    break

            if not is_allowed:
                return (
                    f"❌ 安全拦截：子命令 [{i + 1}] 尝试访问未授权路径。\n"
                    f"尝试访问: {abs_path}\n"
                    f"来源片段: {sub_cmd[:50]}...\n"
                    f"允许目录列表: {safe_dirs}"
                )

    # 4. 所有子命令校验通过，执行原始命令
    # 注意：我们执行的是原始的 cmd_raw，以保持 bash 的管道和逻辑功能
    try:
        if sys.platform == "win32":
            bash_path = r"E:\Program Files (x86)\Git\bin\bash.exe"
            if os.path.exists(bash_path):
                shell_cmd_list = [bash_path, "-c", cmd_raw]
            else:
                return "❌ 错误：Windows 环境下未找到 Git Bash。"
        else:
            shell_cmd_list = ["/bin/bash", "-c", cmd_raw]

        result = subprocess.run(
            shell_cmd_list,
            capture_output=True,
            text=True,
            timeout=EXEC_TIMEOUT,
            encoding="utf-8",
            errors="ignore",
            cwd=work_dir
        )

        if result.returncode == 0:
            output = result.stdout if result.stdout else "(无输出)"
            return f"✅ 命令执行成功:\n{output}"
        else:
            error_msg = result.stderr if result.stderr else result.stdout
            # 对于管道命令，部分失败可能返回非零，这通常是预期的行为
            return f"⚠️ 命令执行完成 (退出码: {result.returncode}):\n{error_msg}"

    except subprocess.TimeoutExpired:
        return f"❌ 命令执行超时 (超过 {EXEC_TIMEOUT} 秒)"
    except Exception as e:
        return f"❌ 系统异常: {str(e)}"


# ========== 辅助函数 ==========

def _smart_split_commands(command: str) -> List[str]:
    """
    智能分割命令字符串，支持 |, ;, &&, ||
    能够正确处理引号内的分隔符（即引号内的 | 不会被分割）
    """
    commands = []
    current_cmd = []
    in_single_quote = False
    in_double_quote = False
    i = 0
    length = len(command)

    while i < length:
        char = command[i]

        # 处理引号状态
        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            current_cmd.append(char)
        elif char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            current_cmd.append(char)
        elif not in_single_quote and not in_double_quote:
            # 只有在非引号状态下才检查操作符

            # 检查 && 和 || (双字符操作符)
            if i + 1 < length:
                two_chars = command[i:i + 2]
                if two_chars in ['&&', '||']:
                    if current_cmd:
                        commands.append("".join(current_cmd).strip())
                        current_cmd = []
                    i += 2
                    continue

            # 检查 ; 和 | (单字符操作符)
            if char in [';', '|']:
                if current_cmd:
                    commands.append("".join(current_cmd).strip())
                    current_cmd = []
                i += 1
                continue

            current_cmd.append(char)
        else:
            current_cmd.append(char)

        i += 1

    if current_cmd:
        commands.append("".join(current_cmd).strip())

    return commands


def _extract_paths_from_command(command: str, cwd: str) -> List[str]:
    """
    从单个命令片段中提取潜在路径
    """
    paths = []
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()

    # 跳过第一个 (命令名)
    for token in tokens[1:]:
        if token.startswith('-'):
            continue
        # 简单的路径特征过滤，避免把纯数字或无关参数当路径
        # 但为了安全起见，宁可多校验不错过，只要不是选项标志都校验
        if re.match(r'^[a-zA-Z0-9._/~@%:-]+$', token):
            paths.append(token)
    return paths


# ========== LLM Tool Definition ==========

def get_shell_tool_definition() -> Dict:
    allowed_cmds_str = ", ".join(sorted(ALLOWED_COMMANDS))
    return {
        "type": "function",
        "function": {
            "name": "run_shell_function",
            "description": f"""
                在安全沙箱中执行 Shell 命令。支持命令链 (|, ;, &&, ||)。

                **安全规则**:
                1. **全链路白名单**: 命令链中的**每一个**子命令都必须在允许列表中: {allowed_cmds_str}。
                   - 示例: "ls | grep txt" (合法，如果 ls 和 grep 都在白名单)
                   - 示例: "ls | rm file" (非法，rm 不在白名单，整个命令被拒)
                2. **路径隔离**: 所有子命令涉及的文件路径必须在 `allowed_dirs` 内。
                3. **禁止项**: 禁止 sudo, su, 以及任何不在白名单的命令。

                **参数**:
                - command_string: 完整的命令字符串。
                - cwd: 工作目录。
                - allowed_dirs: 允许访问的目录列表 (绝对路径)。
            """.replace("\n", " ").strip(),
            "parameters": {
                "type": "object",
                "required": ["command_string"],
                "properties": {
                    "command_string": {
                        "type": "string",
                        "description": "Shell 命令，支持管道和逻辑组合 (如: 'cat log.txt | grep error')"
                    },
                    "cwd": {
                        "type": "string",
                        "description": "(可选) 工作目录"
                    },
                    "allowed_dirs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "(可选) 允许访问的目录列表"
                    }
                }
            }
        }
    }


# ========== 测试用例 ==========
if __name__ == "__main__":
    cwd = os.getcwd()

    print("=== 测试 1: 合法的管道命令 (ls | grep) ===")
    # 假设当前目录下有文件
    print(run_shell_function("ls -a | grep .py", cwd=cwd, allowed_dirs=[cwd]))

    print("\n=== 测试 2: 非法的管道命令 (包含 rm) ===")
    print(run_shell_function("ls | rm -rf test", cwd=cwd, allowed_dirs=[cwd]))

    print("\n=== 测试 3: 合法的顺序执行 (;) ===")
    print(run_shell_function("pwd; echo hello", cwd=cwd, allowed_dirs=[cwd]))

    print("\n=== 测试 4: 合法的条件执行 (&&) ===")
    print(run_shell_function("ls non_existent_file && echo success", cwd=cwd, allowed_dirs=[cwd]))

    print("\n=== 测试 5: 路径越权 (在管道中) ===")
    # 尝试 cat /etc/passwd | grep root
    print(run_shell_function("cat /etc/passwd | grep root", cwd=cwd, allowed_dirs=[cwd]))

    print("\n=== 测试 6: 引号内的分隔符不应分割 ===")
    # echo "a|b;c" 应该作为一个整体执行，不应该被分割成 echo "a 和 b 和 c"
    print(run_shell_function('echo "hello|world;test"', cwd=cwd, allowed_dirs=[cwd]))