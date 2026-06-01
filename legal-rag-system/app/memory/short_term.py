"""短期记忆 - 跨轮关键事实，会话周期内持久"""
import json
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
from collections import OrderedDict


class ShortTermMemory:
    """会话级短期记忆，存储用户身份、案件要素、偏好等"""

    def __init__(self, conversation_id: str, persist_dir: str = "./data/short_term"):
        self.conversation_id = conversation_id
        self.persist_dir = persist_dir
        self._store: OrderedDict = OrderedDict()
        self.max_items = 200
        os.makedirs(persist_dir, exist_ok=True)
        self._load()

    def set(self, key: str, value: Any):
        if key in self._store:
            del self._store[key]
        elif len(self._store) >= self.max_items:
            self._store.popitem(last=False)
        self._store[key] = {"value": value, "timestamp": datetime.now().isoformat()}
        self._save()

    def get(self, key: str) -> Optional[Any]:
        item = self._store.get(key)
        return item["value"] if item else None

    def get_all(self) -> Dict[str, Any]:
        return {k: v["value"] for k, v in self._store.items()}

    def delete(self, key: str):
        self._store.pop(key, None)
        self._save()

    def clear(self):
        self._store.clear()
        self._save()

    def _file_path(self) -> str:
        safe_id = "".join(c for c in self.conversation_id if c.isalnum() or c in "-_")
        return os.path.join(self.persist_dir, f"{safe_id}.json")

    def _save(self):
        try:
            with open(self._file_path(), "w", encoding="utf-8") as f:
                json.dump(dict(self._store), f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _load(self):
        try:
            path = self._file_path()
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for k, v in data.items():
                        self._store[k] = v
        except Exception:
            pass

    def summary(self) -> str:
        """生成可用于注入提示词的记忆摘要"""
        items = self.get_all()
        if not items:
            return ""
        lines = ["## 用户信息与偏好"]
        for k, v in items.items():
            lines.append(f"- {k}: {v}")
        return "\n".join(lines)
