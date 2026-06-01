"""长期记忆 - 向量数据库存储经验教训和结构化摘要"""
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime


class LongTermMemory:
    """长期记忆管理器，使用 ChromaDB 作为向量后端"""

    def __init__(self, persist_dir: str = "./data/chroma"):
        self.persist_dir = persist_dir
        self._collection = None
        self._init_chroma()

    def _init_chroma(self):
        try:
            import chromadb
            os.makedirs(self.persist_dir, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_dir)
            self._collection = self._client.get_or_create_collection(
                name="legal_long_term_memory",
                metadata={"description": "法律助手长期记忆"},
            )
        except ImportError:
            self._client = None
            self._collection = None

    def store_lesson(self, task_type: str, lesson: str, source: str, score: float = 0.5):
        """存储反思教训"""
        if not self._collection:
            return

        doc_id = f"lesson_{datetime.now().strftime('%Y%m%d%H%M%S')}_{hash(lesson) % 10000}"
        metadata = {
            "task_type": task_type,
            "source": source,
            "reusability_score": score,
            "timestamp": datetime.now().isoformat(),
        }
        self._collection.add(
            ids=[doc_id],
            documents=[lesson],
            metadatas=[metadata],
        )

    def store_summary(self, summary: str, metadata: Dict[str, Any]):
        """存储事件摘要"""
        if not self._collection:
            return

        doc_id = f"summary_{datetime.now().strftime('%Y%m%d%H%M%S')}_{hash(summary) % 10000}"
        meta = {**metadata, "timestamp": datetime.now().isoformat(), "type": "summary"}
        self._collection.add(
            ids=[doc_id],
            documents=[summary],
            metadatas=[meta],
        )

    def search(self, query: str, top_k: int = 5, filter_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """检索相关记忆"""
        if not self._collection:
            return []

        where = None
        if filter_type:
            where = {"type": filter_type}

        results = self._collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where,
        )

        items = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                items.append({
                    "id": doc_id,
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                })
        return items

    def search_lessons(self, query: str, top_k: int = 5) -> List[str]:
        items = self.search(query, top_k)
        return [
            f"[经验 {item['metadata'].get('reusability_score', 0.5):.0%}] {item['content']}"
            for item in items
        ]
