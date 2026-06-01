"""消息模型 - 借鉴 hello-agents 的 Message 设计"""
from typing import Optional, Dict, Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field

MessageRole = Literal["system", "user", "assistant", "tool"]


class Message(BaseModel):
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.role, "content": self.content}

    def to_openai_format(self) -> Dict[str, Any]:
        return {"role": self.role, "content": self.content}

    def __str__(self):
        return f"[{self.role.upper()}] {self.content}"
