"""会话存储 - 持久化对话记录"""
import json
import os
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime


class ConversationStore:
    """JSON文件持久化的对话存储"""

    def __init__(self, persist_dir: str = "./data/conversations"):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        self._index_file = os.path.join(persist_dir, "_index.json")
        self._ensure_index()

    def _ensure_index(self):
        if not os.path.exists(self._index_file):
            self._save_index({})

    def _load_index(self) -> Dict[str, Any]:
        try:
            with open(self._index_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_index(self, index: Dict[str, Any]):
        try:
            with open(self._index_file, "w", encoding="utf-8") as f:
                json.dump(index, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _conv_file(self, conv_id: str) -> str:
        safe_id = "".join(c for c in conv_id if c.isalnum() or c in "-_")
        return os.path.join(self.persist_dir, f"{safe_id}.json")

    def create(self, title: str = "") -> str:
        """创建新会话，返回会话ID"""
        conv_id = str(uuid.uuid4())[:8]
        conv = {
            "id": conv_id,
            "title": title or "新会话",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "messages": [],
        }
        with open(self._conv_file(conv_id), "w", encoding="utf-8") as f:
            json.dump(conv, f, ensure_ascii=False, indent=2)

        index = self._load_index()
        index[conv_id] = {"title": conv["title"], "created_at": conv["created_at"], "updated_at": conv["updated_at"]}
        self._save_index(index)
        return conv_id

    def add_message(self, conv_id: str, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """向会话追加一条消息，可选携带元数据（引用、置信度、推理范式等）"""
        conv = self.get(conv_id)
        if not conv:
            conv = {"id": conv_id, "title": "新会话", "created_at": datetime.now().isoformat(), "messages": []}

        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        if metadata:
            msg["metadata"] = metadata

        conv["messages"].append(msg)

        # Auto-update title from first user message
        if role == "user" and len([m for m in conv["messages"] if m["role"] == "user"]) == 1:
            conv["title"] = content[:30] + ("..." if len(content) > 30 else "")

        conv["updated_at"] = datetime.now().isoformat()

        with open(self._conv_file(conv_id), "w", encoding="utf-8") as f:
            json.dump(conv, f, ensure_ascii=False, indent=2)

        index = self._load_index()
        index[conv_id] = {"title": conv["title"], "created_at": conv["created_at"], "updated_at": conv["updated_at"]}
        self._save_index(index)

    def get(self, conv_id: str) -> Optional[Dict[str, Any]]:
        """获取会话详情"""
        path = self._conv_file(conv_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def list_all(self) -> List[Dict[str, Any]]:
        """列出所有会话摘要"""
        index = self._load_index()
        result = []
        for conv_id, info in index.items():
            result.append({
                "id": conv_id,
                "title": info.get("title", "未命名"),
                "created_at": info.get("created_at", ""),
                "updated_at": info.get("updated_at", ""),
            })
        result.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return result

    def delete(self, conv_id: str):
        """删除会话"""
        path = self._conv_file(conv_id)
        if os.path.exists(path):
            os.remove(path)
        index = self._load_index()
        index.pop(conv_id, None)
        self._save_index(index)

    def get_messages(self, conv_id: str) -> List[Dict[str, Any]]:
        """获取会话消息列表"""
        conv = self.get(conv_id)
        return conv["messages"] if conv else []
