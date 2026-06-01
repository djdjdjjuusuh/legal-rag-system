"""智能体基类 - 借鉴 hello-agents Agent 抽象"""
from abc import ABC, abstractmethod
from typing import Optional, Any, List, Iterator
from app.core.llm import LLMClient
from app.core.message import Message
from app.core.config import Config, get_config


class BaseAgent(ABC):
    def __init__(
        self,
        name: str,
        llm: Optional[LLMClient] = None,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
    ):
        self.name = name
        self.llm = llm or LLMClient()
        self.system_prompt = system_prompt or f"你是 {name}，一个智能法律助手。"
        self.config = config or get_config()
        self._history: List[Message] = []

    @abstractmethod
    def run(self, query: str, **kwargs) -> str:
        """处理查询并返回响应"""
        pass

    def add_message(self, message: Message):
        self._history.append(message)

    def clear_history(self):
        self._history.clear()

    def get_history(self) -> List[Message]:
        return self._history.copy()

    def _build_messages(self, query: str, enhanced_prompt: Optional[str] = None) -> List[dict]:
        messages = [{"role": "system", "content": enhanced_prompt or self.system_prompt}]
        for msg in self._history[-self.config.max_working_memory:]:
            messages.append(msg.to_openai_format())
        messages.append({"role": "user", "content": query})
        return messages

    @staticmethod
    def _stream_text(text: str, chunk_size: int = 5) -> Iterator[str]:
        """将文本按小chunk切分，实现逐字流式效果"""
        for i in range(0, len(text), chunk_size):
            yield text[i:i + chunk_size]
