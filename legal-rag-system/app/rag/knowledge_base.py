"""法律知识库管理"""
import os
import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from chromadb import PersistentClient


def split_legal_document(text: str, title: str, min_chunk_chars: int = 30) -> List[Tuple[str, str, Dict[str, Any]]]:
    """按法律条文分块：将法律文档按「第X条」拆分为独立条目

    Returns: [(chunk_id_suffix, chunk_content, chunk_metadata), ...]
    """
    # 匹配行首的「第X条」作为条款边界（避免匹配正文中引用的法条）
    article_re = re.compile(r'(?:^|\n)\s*(第[零一二三四五六七八九十百千\d]+条)')

    # Find all article boundaries
    matches = list(article_re.finditer(text))

    if not matches:
        # No article pattern found, return as single chunk
        return [(title, text, {"chunk_type": "full"})]

    chunks = []
    for i, match in enumerate(matches):
        article_label = match.group(1)
        start = match.start(1)
        end = matches[i + 1].start(1) if i + 1 < len(matches) else len(text)
        chunk_text = text[start:end].strip()

        # Extract article number for metadata
        article_num = re.search(r'第([零一二三四五六七八九十百千\d]+)条', article_label)
        article_id = article_num.group(1) if article_num else str(i + 1)

        # Skip chunks that are too small (e.g., just article header)
        if len(chunk_text) < min_chunk_chars:
            continue

        chunk_title = f"{title} - {article_label}"
        chunks.append((
            chunk_title,
            chunk_text,
            {
                "article_label": article_label,
                "article_id": article_id,
                "chunk_index": i,
                "source_title": title,
                "chunk_type": "article",
            },
        ))

    # If chunking produced nothing useful (all too small), fallback to single chunk
    if not chunks:
        return [(title, text, {"chunk_type": "full"})]

    return chunks


# 案例文档的章节标题关键词
CASE_SECTION_PATTERNS = [
    (r'(?:^|\n)\s*(?:裁判要[旨点]|核心要[旨点]|指导要[旨点])', '裁判要旨'),
    (r'(?:^|\n)\s*(?:相关法条|关联法条|适用法条)', '相关法条'),
    (r'(?:^|\n)\s*(?:基本案情|案件事实|案情简介)', '基本案情'),
    (r'(?:^|\n)\s*(?:裁判结果|判决结果|处理结果)', '裁判结果'),
    (r'(?:^|\n)\s*(?:裁判理由|判决理由|本院认为|审理查明)', '裁判理由'),
    (r'(?:^|\n)\s*(?:关键词)', '关键词'),
    (r'(?:^|\n)\s*(?:典型意义|指导意义|案例注解)', '典型意义'),
]


def split_case_document(text: str, title: str, min_chunk_chars: int = 100) -> List[Tuple[str, str, Dict[str, Any]]]:
    """按章节分块：将案例文档按「基本案情」「裁判理由」等章节拆分

    Returns: [(chunk_title, chunk_content, chunk_metadata), ...]
    """
    # Try each pattern, collect all section boundaries
    boundaries = []  # [(position, section_name), ...]
    for pattern, section_name in CASE_SECTION_PATTERNS:
        for match in re.finditer(pattern, text, re.MULTILINE):
            boundaries.append((match.start(), section_name))

    if not boundaries:
        # No section headers found → paragraph-based chunking
        return _split_by_paragraphs(text, title)

    # Sort by position and deduplicate nearby boundaries (within 20 chars)
    boundaries.sort()
    filtered = []
    for pos, name in boundaries:
        if not filtered or pos - filtered[-1][0] > 20:
            filtered.append((pos, name))

    chunks = []
    for i, (start, section_name) in enumerate(filtered):
        end = filtered[i + 1][0] if i + 1 < len(filtered) else len(text)
        chunk_text = text[start:end].strip()
        if len(chunk_text) < min_chunk_chars:
            continue

        chunk_title = f"{title} - {section_name}"
        chunks.append((
            chunk_title,
            chunk_text,
            {
                "section": section_name,
                "chunk_index": i,
                "source_title": title,
                "chunk_type": "case_section",
            },
        ))

    if not chunks:
        return _split_by_paragraphs(text, title)

    return chunks


def _split_by_paragraphs(text: str, title: str, target_chars: int = 2000) -> List[Tuple[str, str, Dict[str, Any]]]:
    """按段落+固定长度分块（无明确章节标题时的回退方案）"""
    # Try double newline first, fall back to single newline
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if len(paragraphs) <= 1:
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    if not paragraphs:
        return [(title, text, {"chunk_type": "full"})]

    chunks = []
    current = ""
    chunk_idx = 0

    def _flush(current_text: str):
        nonlocal chunk_idx
        title_suffix = f"段落{chunk_idx + 1}" if len(paragraphs) > 1 else ""
        chunks.append((f"{title} - {title_suffix}".rstrip(" -"), current_text.strip(), {
            "chunk_index": chunk_idx,
            "source_title": title,
            "chunk_type": "paragraph",
        }))
        chunk_idx += 1

    for para in paragraphs:
        # If a single paragraph is still too large, force-split by char count
        if len(para) > target_chars:
            if current:
                _flush(current)
                current = ""
            for i in range(0, len(para), target_chars):
                sub = para[i:i + target_chars].strip()
                if sub:
                    _flush(sub)
            continue

        if len(current) + len(para) > target_chars and current:
            _flush(current)
            current = para
        else:
            current += ("\n\n" + para) if current else para

    if current.strip():
        _flush(current)

    if not chunks:
        return [(title, text, {"chunk_type": "full"})]

    return chunks


class LegalKnowledgeBase:
    """三层法律知识库：法规库、案例库、实务知识库"""

    def __init__(self, persist_dir: str = "./data/chroma_kb"):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        self._client = PersistentClient(path=persist_dir)
        self._collections = {
            "statutes": self._client.get_or_create_collection(
                name="legal_statutes",
                metadata={"description": "法律法规库"},
            ),
            "cases": self._client.get_or_create_collection(
                name="legal_cases",
                metadata={"description": "案例库"},
            ),
            "practical": self._client.get_or_create_collection(
                name="legal_practical",
                metadata={"description": "实务知识库"},
            ),
        }

    def add_document(
        self,
        doc_type: str,
        title: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk: bool = True,
    ):
        col = self._collections.get(doc_type)
        if not col:
            raise ValueError(f"未知文档类型: {doc_type}，可选: statutes/cases/practical")

        base_meta = metadata or {}
        base_meta["title"] = title
        base_meta["doc_type"] = doc_type

        # Auto-chunk large documents for precise retrieval
        if chunk and len(content) > 3000:
            if doc_type == "statutes":
                chunks = split_legal_document(content, title)
            elif doc_type == "cases":
                chunks = split_case_document(content, title)
            elif doc_type == "practical":
                chunks = split_legal_document(content, title)
                if chunks and chunks[0][2].get("chunk_type") == "full":
                    chunks = _split_by_paragraphs(content, title)
            else:
                chunks = None

            if chunks:
                # Delete all old entries from this source (both full and chunked)
                old_ids = [hashlib.md5(f"{doc_type}:{title}".encode()).hexdigest()[:16]]
                try:
                    col.delete(ids=old_ids)
                except Exception:
                    pass
                # Also clean up any existing chunks with this source title
                try:
                    existing = col.get()
                    if existing and existing.get("ids"):
                        to_delete = []
                        for i, eid in enumerate(existing["ids"]):
                            emeta = existing["metadatas"][i] if existing.get("metadatas") else {}
                            if emeta.get("source_title") == title:
                                to_delete.append(eid)
                        if to_delete:
                            col.delete(ids=to_delete)
                except Exception:
                    pass
                # Batch add all chunks at once for much faster embedding
                batch_ids = []
                batch_texts = []
                batch_metas = []
                for chunk_title, chunk_text, chunk_meta in chunks:
                    chunk_id = hashlib.md5(f"{doc_type}:{chunk_title}".encode()).hexdigest()[:16]
                    full_meta = {**base_meta, **chunk_meta}
                    full_meta["title"] = chunk_title
                    batch_ids.append(chunk_id)
                    batch_texts.append(chunk_text)
                    batch_metas.append(full_meta)
                if batch_ids:
                    col.upsert(ids=batch_ids, documents=batch_texts, metadatas=batch_metas)
                return batch_ids[0] if batch_ids else hashlib.md5(f"{doc_type}:{title}".encode()).hexdigest()[:16]

        doc_id = hashlib.md5(f"{doc_type}:{title}".encode()).hexdigest()[:16]
        existing = col.get(ids=[doc_id])
        if existing["ids"]:
            col.update(ids=[doc_id], documents=[content], metadatas=[base_meta])
        else:
            col.add(ids=[doc_id], documents=[content], metadatas=[base_meta])

        return doc_id

    def add_batch(self, doc_type: str, documents: List[Dict[str, Any]]):
        """批量添加文档"""
        for doc in documents:
            self.add_document(
                doc_type=doc_type,
                title=doc.get("title", ""),
                content=doc.get("content", ""),
                metadata=doc.get("metadata"),
            )

    def delete_document(self, doc_type: str, doc_id: str):
        col = self._collections.get(doc_type)
        if col:
            col.delete(ids=[doc_id])

    def _keyword_score(self, query: str, text: str) -> float:
        """计算查询词与文档的关键词匹配分数（基于字符 n-gram）"""
        if not text:
            return 0.0
        score = 0.0
        # Extract n-grams from query (2-char, 3-char, and full query)
        for n in [2, 3]:
            for i in range(len(query) - n + 1):
                term = query[i:i + n]
                # Count occurrences in document
                count = text.count(term)
                if count > 0:
                    # Longer terms get higher weight
                    score += count * n
        # Full query exact match bonus
        if query in text:
            score += 10
        return score

    def _hybrid_rerank(
        self,
        query: str,
        embedding_results: List[Dict[str, Any]],
        collection,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """混合重排序：结合 embedding 分数和关键词匹配"""
        # Build lookup of embedding results by id
        emb_by_id = {}
        for r in embedding_results:
            emb_by_id[r["id"]] = r

        # Compute keyword scores for all documents and find additional keyword-only matches
        keyword_matches = []
        try:
            all_data = collection.get()
            if all_data and all_data.get("ids"):
                for i, doc_id in enumerate(all_data["ids"]):
                    content = all_data["documents"][i] if all_data.get("documents") else ""
                    kw_score = self._keyword_score(query, content)
                    if kw_score > 0 or doc_id in emb_by_id:
                        meta = all_data["metadatas"][i] if all_data.get("metadatas") else {}
                        keyword_matches.append({
                            "id": doc_id,
                            "content": content,
                            "title": meta.get("title", ""),
                            "metadata": meta,
                            "kw_score": kw_score,
                            "emb_score": emb_by_id.get(doc_id, {}).get("score"),
                            "doc_type": emb_by_id.get(doc_id, {}).get("doc_type", meta.get("doc_type", "")),
                        })
        except Exception:
            pass

        # Score: keyword match weighted heavily + embedding score
        max_kw = max((m["kw_score"] for m in keyword_matches), default=1) or 1
        for m in keyword_matches:
            kw_norm = m["kw_score"] / max_kw
            emb_norm = 0.0
            if m["emb_score"] is not None and m["emb_score"] > 0:
                emb_norm = 1.0 / (1.0 + m["emb_score"])
            m["final_score"] = kw_norm * 0.7 + emb_norm * 0.3

        keyword_matches.sort(key=lambda x: x["final_score"], reverse=True)
        return keyword_matches[:top_k]

    def search(
        self,
        query: str,
        doc_type: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        results = []

        collections_to_search = (
            {doc_type: self._collections[doc_type]} if doc_type
            else self._collections
        )

        for col_name, col in collections_to_search.items():
            try:
                # Fetch more candidates from embedding search for re-ranking
                res = col.query(query_texts=[query], n_results=min(top_k * 3, col.count()))
                emb_results = []
                if res["ids"] and res["ids"][0]:
                    for i, doc_id in enumerate(res["ids"][0]):
                        emb_results.append({
                            "id": doc_id,
                            "doc_type": col_name,
                            "title": res["metadatas"][0][i].get("title", ""),
                            "content": res["documents"][0][i],
                            "metadata": res["metadatas"][0][i],
                            "score": res["distances"][0][i] if res.get("distances") else None,
                        })
                # Hybrid re-rank
                reranked = self._hybrid_rerank(query, emb_results, col, top_k)
                for m in reranked:
                    results.append({
                        "id": m["id"],
                        "doc_type": m.get("doc_type", col_name),
                        "title": m["title"],
                        "content": m["content"],
                        "metadata": m["metadata"],
                        "score": m["final_score"],
                    })
            except Exception:
                continue

        results.sort(key=lambda x: x.get("score") or 0, reverse=True)
        return results[:top_k]

    def search_statutes(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self.search(query, doc_type="statutes", top_k=top_k)

    def search_cases(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self.search(query, doc_type="cases", top_k=top_k)

    def get_collection_stats(self) -> Dict[str, int]:
        return {
            name: col.count()
            for name, col in self._collections.items()
        }

    def list_documents(self, doc_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出知识库中的文档"""
        results = []
        collections_to_list = (
            {doc_type: self._collections[doc_type]} if doc_type
            else self._collections
        )
        for col_name, col in collections_to_list.items():
            try:
                all_data = col.get()
                if all_data and all_data.get("ids"):
                    for i, doc_id in enumerate(all_data["ids"]):
                        meta = all_data["metadatas"][i] if all_data.get("metadatas") else {}
                        results.append({
                            "id": doc_id,
                            "doc_type": col_name,
                            "title": meta.get("title", doc_id),
                            "content_preview": (all_data["documents"][i][:200] + "...") if all_data.get("documents") else "",
                            "metadata": meta,
                        })
            except Exception:
                continue
        results.sort(key=lambda x: x["title"])
        return results

    def get_document(self, doc_type: str, doc_id: str) -> Optional[Dict[str, Any]]:
        """获取单个文档详情"""
        col = self._collections.get(doc_type)
        if not col:
            return None
        try:
            data = col.get(ids=[doc_id])
            if data and data.get("ids"):
                meta = data["metadatas"][0] if data.get("metadatas") else {}
                return {
                    "id": doc_id,
                    "doc_type": doc_type,
                    "title": meta.get("title", doc_id),
                    "content": data["documents"][0] if data.get("documents") else "",
                    "metadata": meta,
                }
        except Exception:
            pass
        return None
