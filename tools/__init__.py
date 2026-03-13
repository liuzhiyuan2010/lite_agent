from .base import BaseTool
from .tool_shell import ShellCommandTool
import  os
# 获取项目根目录 (假设 tools 文件夹在项目根目录下)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "conf.json")

AVAILABLE_TOOLS = [
    # 实例化时会自动读取 config 中的 allowed_write_dirs
    ShellCommandTool(config_path=CONFIG_PATH),
]


def get_tools_definitions() -> list:
    """获取所有工具的 API 定义格式 (Schema)"""
    return [tool.to_definition() for tool in AVAILABLE_TOOLS]

def get_tool_by_name(name: str) -> BaseTool:
    """根据名称查找工具实例"""
    for tool in AVAILABLE_TOOLS:
        if tool.name == name:
            return tool
    raise ValueError(f"未找到名为 '{name}' 的工具")