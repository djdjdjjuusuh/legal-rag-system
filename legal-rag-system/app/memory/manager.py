"""记忆管理器 - 统一管理三层记忆"""
from typing import Optional
from app.memory.working import WorkingMemory
from app.memory.short_term import ShortTermMemory
from app.memory.long_term import LongTermMemory
from app.core.config import get_config


class MemoryManager:
    def __init__(self, conversation_id: str = "default"):
        config = get_config()
        self.conversation_id = conversation_id
        self.working = WorkingMemory(max_size=config.max_working_memory)
        self.short_term = ShortTermMemory(
            conversation_id=conversation_id,
            persist_dir="./data/short_term",
        )
        self.long_term = LongTermMemory(persist_dir=config.chroma_persist_dir)

    def add_interaction(self, user_query: str, assistant_response: str):
        from app.core.message import Message
        self.working.add_message(Message(role="user", content=user_query))
        self.working.add_message(Message(role="assistant", content=assistant_response))

    def get_enhanced_context(self, query: str) -> str:
        """获取增强后的上下文，用于注入系统提示"""
        parts = []

        # 短期记忆
        short_summary = self.short_term.summary()
        if short_summary:
            parts.append(short_summary)

        # 长期记忆中的相关经验
        lessons = self.long_term.search_lessons(query, top_k=3)
        if lessons:
            parts.append("## 历史经验")
            parts.extend(lessons)

        return "\n".join(parts) if parts else ""
