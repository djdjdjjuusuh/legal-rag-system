"""工作记忆 - 当前对话上下文"""
from typing import List, Dict, Any, Optional
from app.core.message import Message


class WorkingMemory:
    """当前会话的工作记忆，存储完整对话历史和注入的上下文"""

    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.messages: List[Message] = []
        self.injected_context: List[Dict[str, Any]] = []

    def add_message(self, message: Message):
        self.messages.append(message)
        if len(self.messages) > self.max_size:
            self.messages = self.messages[-self.max_size:]

    def add_context(self, context: Dict[str, Any]):
        self.injected_context.append(context)
        if len(self.injected_context) > 10:
            self.injected_context.pop(0)

    def get_messages(self, last_n: Optional[int] = None) -> List[Message]:
        if last_n:
            return self.messages[-last_n:]
        return self.messages.copy()

    def get_context(self) -> List[Dict[str, Any]]:
        return self.injected_context.copy()

    def clear(self):
        self.messages.clear()
        self.injected_context.clear()

    def to_openai_format(self) -> List[dict]:
        return [
            m.to_openai_format() for m in self.messages[-self.max_size:]
        ]
