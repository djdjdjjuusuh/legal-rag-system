"""工具注册表 - 借鉴 hello-agents ToolRegistry 设计"""
from typing import Dict, Any, Optional, Callable, List
from abc import ABC, abstractmethod


class Tool(ABC):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.source = "local"  # "local" or "mcp"

    @abstractmethod
    def execute(self, **kwargs) -> str:
        pass

    def get_schema(self) -> Dict[str, Any]:
        return {"name": self.name, "description": self.description}


class FunctionTool(Tool):
    """包装普通函数为工具"""
    def __init__(self, name: str, description: str, func: Callable[..., str]):
        super().__init__(name, description)
        self._func = func

    def execute(self, **kwargs) -> str:
        return self._func(**kwargs)


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._disabled: set = set()

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def unregister(self, name: str):
        self._tools.pop(name, None)
        self._disabled.discard(name)

    def get_tool(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())

    def list_tools_with_status(self) -> List[Dict[str, Any]]:
        """返回所有工具及其启用状态"""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "source": tool.source,
                "enabled": tool.name not in self._disabled,
            }
            for tool in self._tools.values()
        ]

    def is_enabled(self, name: str) -> bool:
        return name in self._tools and name not in self._disabled

    def set_enabled(self, name: str, enabled: bool):
        if name not in self._tools:
            return
        if enabled:
            self._disabled.discard(name)
        else:
            self._disabled.add(name)

    def _enabled_tools(self):
        return {k: v for k, v in self._tools.items() if k not in self._disabled}

    def get_tools_description(self) -> str:
        enabled = self._enabled_tools()
        if not enabled:
            return "暂无可用工具"
        lines = []
        for tool in enabled.values():
            lines.append(f"- **{tool.name}**: {tool.description}")
        return "\n".join(lines)

    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """生成 OpenAI function calling 格式的工具定义（仅已启用的工具）"""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
            for tool in self._enabled_tools().values()
        ]

    def execute_tool(self, name: str, input_str: str = "", **kwargs) -> str:
        tool = self._tools.get(name)
        if not tool:
            return f"工具 '{name}' 不存在。可用工具: {', '.join(self.list_tools())}"
        if name in self._disabled:
            return f"工具 '{name}' 已被禁用，请先启用后再使用。"
        try:
            if input_str:
                return tool.execute(input=input_str, query=input_str, **kwargs)
            return tool.execute(**kwargs)
        except Exception as e:
            return f"工具 '{name}' 执行出错: {str(e)}"
