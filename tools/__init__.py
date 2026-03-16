
from typing import Optional, Callable
from .safe_shell_function import run_shell_function,get_shell_tool_definition


AVAILABLE_TOOLS: dict[str, Callable] = {
    'run_shell_function':run_shell_function
}

def get_tool_by_name(name: str)-> Optional[Callable]:
    """根据名称查找工具实例"""
    return  AVAILABLE_TOOLS.get(name)
    #raise ValueError(f"未找到名为 '{name}' 的工具")


__all__ = [
    'run_shell_function',
    'get_shell_tool_definition',
    'get_tool_by_name'
]



