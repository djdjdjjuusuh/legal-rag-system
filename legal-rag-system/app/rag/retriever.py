"""RAG 检索增强生成管线"""
from typing import List, Dict, Any, Optional, Tuple
from app.rag.knowledge_base import LegalKnowledgeBase
from app.core.llm import LLMClient
from app.core.config import get_config


class IntentClassifier:
    """意图识别 - 判断用户提问类型"""
    INTENT_PROMPT = """分析以下法律问题，判断其意图类型（仅输出类型名称）：
类型选项：法条查询 / 案例检索 / 实务咨询 / 合同审查 / 法律推理 / 法律计算

问题：{query}
类型："""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def classify(self, query: str) -> str:
        try:
            result = self.llm.invoke(
                [{"role": "user", "content": self.INTENT_PROMPT.format(query=query)}],
                temperature=0.1,
                max_tokens=20,
            )
            return result.strip()
        except Exception:
            return "实务咨询"


class QueryRewriter:
    """查询改写 - 将口语化问题改写为精确法律检索query"""
    REWRITE_PROMPT = """将以下口语化法律问题改写为精确的法律检索查询词（1-2句话）：
原始问题：{query}
检索查询："""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def rewrite(self, query: str) -> str:
        try:
            result = self.llm.invoke(
                [{"role": "user", "content": self.REWRITE_PROMPT.format(query=query)}],
                temperature=0.1,
                max_tokens=200,
            )
            return result.strip()
        except Exception:
            return query


class LegalRetriever:
    """法律 RAG 检索器 - 完整检索增强生成管线"""

    def __init__(
        self,
        knowledge_base: Optional[LegalKnowledgeBase] = None,
        llm: Optional[LLMClient] = None,
    ):
        self.kb = knowledge_base or LegalKnowledgeBase()
        self.llm = llm or LLMClient()
        self.intent_classifier = IntentClassifier(self.llm)
        self.query_rewriter = QueryRewriter(self.llm)
        self.config = get_config()

    def retrieve(self, query: str) -> Dict[str, Any]:
        """完整检索管线：短查询跳过意图/改写，长查询走完整管线"""
        # 短查询（问候、简单问题）直接检索全部类型，跳过意图和改写
        if len(query) <= 15:
            results = self.kb.search(query, doc_type=None, top_k=self.config.rag_top_k)
            return {
                "intent": "简短查询",
                "rewritten_query": query,
                "results": results,
                "context": self._format_results(results),
            }

        # 1. 意图识别
        intent = self.intent_classifier.classify(query)

        # 2. 查询改写
        rewritten = self.query_rewriter.rewrite(query)

        # 3. 根据意图选择检索目标
        doc_type_map = {
            "法条查询": "statutes",
            "案例检索": "cases",
            "实务咨询": "practical",
            "合同审查": "practical",
            "法律推理": None,  # 全部检索
            "法律计算": "statutes",
        }
        target = doc_type_map.get(intent)

        # 4. 多路召回
        results = self.kb.search(rewritten, doc_type=target, top_k=self.config.rag_top_k)

        # 5. 格式化上下文
        context_str = self._format_results(results)

        return {
            "intent": intent,
            "rewritten_query": rewritten,
            "results": results,
            "context": context_str,
        }

    def _format_results(self, results: List[Dict[str, Any]]) -> str:
        if not results:
            return "未找到相关法律资料。请尝试调整问题表述。"

        lines = ["## 相关法律资料\n"]
        for i, r in enumerate(results):
            doc_label = {"statutes": "法规", "cases": "案例", "practical": "实务"}.get(
                r.get("doc_type", ""), "资料"
            )
            content = r.get("content", "")
            # For article-level chunks, show full content; for large docs, expand to 4000 chars
            chunk_type = r.get("metadata", {}).get("chunk_type", "")
            max_len = 0 if chunk_type == "article" else 4000
            display_content = content if (max_len == 0) else content[:max_len]
            lines.append(f"### {doc_label} {i+1}: {r.get('title', '无标题')}")
            lines.append(display_content)
            lines.append(f"*来源: {doc_label}库 | 相关度: {r.get('score', 'N/A')}*")
            if len(content) > 4000 and max_len > 0:
                lines.append(f"*(内容较长，已截取前 {max_len} 字符)*")
            lines.append("")
        return "\n".join(lines)

    def retrieve_for_prompt(self, query: str) -> str:
        """直接获取可注入提示词的上下文字符串"""
        result = self.retrieve(query)
        return result["context"]
