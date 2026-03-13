from .base import BaseTool
from .tool_shell import ShellCommandTool
import  os

AVAILABLE_TOOLS = [
    ShellCommandTool(),
]

def init_working_dir(workdir): #初始化工作目录
    for tool in AVAILABLE_TOOLS:
        tool.init_working_dir(workdir)

def get_tools_definitions() -> list:
    """获取所有工具的 API 定义格式 (Schema)"""
    return [tool.to_definition() for tool in AVAILABLE_TOOLS]

def get_tool_by_name(name: str) -> BaseTool:
    """根据名称查找工具实例"""
    for tool in AVAILABLE_TOOLS:
        if tool.name == name:
            return tool
    raise ValueError(f"未找到名为 '{name}' 的工具")