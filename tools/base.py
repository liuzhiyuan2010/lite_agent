from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseTool(ABC):
    """所有工具的基类"""

    @property
    @abstractmethod
    def name(self) -> str:
        """工具的唯一名称，必须与 LLM 调用的名称一致"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """工具的描述，用于告诉 LLM 何时使用该工具"""
        pass

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """JSON Schema 格式的参数定义"""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """执行工具的具体逻辑，返回字符串结果"""
        pass

    def to_definition(self) -> Dict[str, Any]:
        """转换为 Ollama/API 需要的工具定义格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }