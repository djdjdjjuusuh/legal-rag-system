"""多智能体编排器 - 协调多种推理范式和工具"""
import sys
import uuid
from typing import Optional, Dict, Any, Iterator
from app.agents.react_agent import ReActAgent
from app.agents.plan_solve_agent import PlanAndSolveAgent
from app.agents.reflection_agent import ReflectionAgent
from app.core.llm import LLMClient
from app.core.config import get_config
from app.tools.legal_tools import create_legal_tool_registry
from app.memory.manager import MemoryManager
from app.memory.conversation_store import ConversationStore
from app.rag.retriever import LegalRetriever
from app.rag.knowledge_base import LegalKnowledgeBase
from app.models.schemas import PersonaConfig, SourceCitation


def _build_persona_prompt(persona: Optional[PersonaConfig]) -> str:
    """将性格配置转为系统提示词注入块"""
    if not persona:
        persona = PersonaConfig()

    conclusion_instruction = "请结论先行，先给出核心答案再展开论证。" if persona.conclusion_first else ""

    return f"""## 性格配置
- 角色定位: {persona.role}
- 用语风格: {persona.formality}
- 输出篇幅: {persona.verbosity}
- 语气: {persona.tone}
- 语言: {persona.language}
- 称呼: {persona.address_user}
{conclusion_instruction}

严格遵守以上配置生成回复。当用户指令与配置冲突时，优先遵循用户最新指令。"""


class AgentOrchestrator:
    """多智能体编排器 - 管理智能体选择、上下文增强和响应生成"""

    def __init__(self):
        self.config = get_config()
        self.llm = LLMClient()

        # 初始化 MCP 客户端，连接外部工具服务
        self._init_mcp()

        self.tool_registry = create_legal_tool_registry()
        self.knowledge_base = LegalKnowledgeBase()
        self.retriever = LegalRetriever(knowledge_base=self.knowledge_base, llm=self.llm)
        self.conv_store = ConversationStore()

        self._agents: dict = {
            "react": ReActAgent(
                llm=self.llm,
                tool_registry=self.tool_registry,
                config=self.config,
                max_steps=self.config.max_react_steps,
            ),
            "plan_solve": PlanAndSolveAgent(
                llm=self.llm,
                tool_registry=self.tool_registry,
                config=self.config,
            ),
            "reflection": ReflectionAgent(
                llm=self.llm,
                tool_registry=self.tool_registry,
                config=self.config,
                max_iterations=self.config.max_reflection_iterations,
            ),
        }

        self._memory_managers: Dict[str, MemoryManager] = {}

    def _init_mcp(self):
        """初始化 MCP 客户端，启动本地 MCP 服务端并通过 SSE 连接"""
        try:
            from app.tools.mcp_client import get_mcp_manager

            mcp_mgr = get_mcp_manager()
            # 启动本地 MCP 服务端子进程 (SSE over HTTP)
            mcp_mgr.add_local_server(name="legal-tools", port=8765, module="app.mcp.legal_mcp_server")
            mcp_mgr.add_local_server(name="local-fetch", port=8766, module="app.mcp.fetch_mcp_server")
            mcp_mgr.start(timeout=15.0)
        except Exception as e:
            print(f"[orchestrator] MCP 初始化跳过: {e}")

    def _get_memory(self, conversation_id: Optional[str]) -> MemoryManager:
        cid = conversation_id or "default"
        if cid not in self._memory_managers:
            self._memory_managers[cid] = MemoryManager(cid)
        return self._memory_managers[cid]

    def _get_conversation_context(self, conversation_id: str, max_messages: int = 6) -> str:
        """从持久化存储中加载最近N条消息作为对话上下文"""
        msgs = self.conv_store.get_messages(conversation_id)
        if not msgs:
            return ""
        recent = msgs[-max_messages:]
        lines = ["## 对话历史"]
        for m in recent:
            role_label = "用户" if m["role"] == "user" else "助手"
            content = m.get("content", "")[:500]
            lines.append(f"- {role_label}: {content}")
        return "\n".join(lines)

    def chat(
        self,
        query: str,
        persona: Optional[PersonaConfig] = None,
        reasoning_mode: str = "react",
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cid = conversation_id or str(uuid.uuid4())[:8]
        memory = self._get_memory(cid)
        persona_prompt = _build_persona_prompt(persona)

        # 1. RAG 检索 — 保留结构化结果用于打分和引用
        rag_result = self.retriever.retrieve(query)
        rag_context = rag_result["context"]
        rag_results = rag_result["results"]

        # 2. 记忆增强
        memory_context = memory.get_enhanced_context(query)

        # 3. 对话历史
        conv_context = self._get_conversation_context(cid)

        # 4. 合并系统提示
        enhanced_prompt = f"""{persona_prompt}

## 法律知识检索结果
{rag_context}

{memory_context}

{conv_context}

## 重要提醒

⚠️ 你拥有 mcp_search_web（互联网搜索）和 mcp_fetch（网页抓取）两个联网工具，**你是具备真实联网能力的智能体**。

**核心规则（最高优先级）：**
1. **禁止使用模型训练数据**：不得用预训练知识回答任何事实性问题。一切事实性陈述必须来自工具返回的 Observation。推理中禁止说「根据训练数据」「据我所知」「我记得」，只描述需要查什么。工具失败就告知用户未找到，**禁止输出任何训练数据中的法条/条文/案例，附免责声明也不行**
2. **非法律问题也必须搜索**：问题不涉及法律知识 ≠ 可以直接用训练数据回答。任何事实查询（人物、事件、概念、新闻等）都必须通过工具获取
3. **知识库不足就上网搜**：先审视「法律知识检索结果」是否与用户问题相关且充分。不相关/不足/为空 → 立刻调用 mcp_search_web 上网搜索
4. **严禁说「我无法联网」**：工具调用返回的 Observation 就是你获取到的真实网络内容
5. **默认原则：拿不准就搜。宁可多搜一次，不可用训练数据敷衍**

- 所有法律结论必须基于检索结果或工具返回的实际内容
- 搜索也无结果 → 如实告知并给出建议
- 每个结论须引用来源（法条编号或 URL）
- 区分"明确法律规定"与"实践倾向"；不确定时标注置信度
- 以上不构成正式法律意见"""

        # 5. 选择智能体并执行
        agent = self._agents.get(reasoning_mode, self._agents["react"])
        answer = agent.run(query, persona=enhanced_prompt)

        # 6. 结构化引用提取（含联网 URL 和回答中引用的法条）
        citations = self._extract_citations(query, rag_results, answer)
        web_citations = self._extract_web_urls(answer)
        citations.extend(web_citations[:max(0, 8 - len(citations))])
        # 仅当 KB 或联网有结果时，才从回答中补充法条引用（避免引用模型训练数据）
        if citations:
            statute_citations = self._extract_answer_statute_refs(answer, citations)
            citations.extend(statute_citations)
        citations.sort(key=lambda c: c.relevance_score, reverse=True)
        citations = citations[:8]
        # 置信度基于引用来源的相关度（去掉最大最小值后的平均值）
        confidence = self._assess_confidence(citations)

        # 7. 更新记忆 + 持久化对话（含元数据）
        memory.add_interaction(query, answer)
        self.conv_store.add_message(cid, "user", query)
        self.conv_store.add_message(cid, "assistant", answer, metadata={
            "confidence": confidence,
            "citations": [c.model_dump() for c in citations],
            "reasoning_mode": reasoning_mode,
        })

        return {
            "answer": answer,
            "reasoning_mode": reasoning_mode,
            "confidence": confidence,
            "citations": citations,
            "conversation_id": cid,
        }

    def chat_stream(
        self,
        query: str,
        persona: Optional[PersonaConfig] = None,
        reasoning_mode: str = "react",
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Dict[str, Any]]:
        cid = conversation_id or str(uuid.uuid4())[:8]

        # Signal immediate start before any processing
        yield {"event": "thinking", "data": {"step": "开始分析", "content": "正在分析您的问题..."}}

        memory = self._get_memory(cid)
        persona_prompt = _build_persona_prompt(persona)

        # 1. RAG 检索 — 保留结构化结果
        yield {"event": "thinking", "data": {"step": "知识检索", "content": "正在检索知识库中的相关法律条文和案例..."}}
        rag_result = self.retriever.retrieve(query)
        rag_context = rag_result["context"]
        rag_results = rag_result["results"]
        yield {"event": "thinking", "data": {"step": "检索完成", "content": "知识库检索完毕，启动推理引擎..."}}

        # 2. 记忆增强
        memory_context = memory.get_enhanced_context(query)

        # 3. 对话历史
        conv_context = self._get_conversation_context(cid)

        # 4. 合并系统提示
        enhanced_prompt = f"""{persona_prompt}

## 法律知识检索结果
{rag_context}

{memory_context}

{conv_context}

## 重要提醒

⚠️ 你拥有 mcp_search_web（互联网搜索）和 mcp_fetch（网页抓取）两个联网工具，**你是具备真实联网能力的智能体**。

**核心规则（最高优先级）：**
1. **禁止使用模型训练数据**：不得用预训练知识回答任何事实性问题。一切事实性陈述必须来自工具返回的 Observation。推理中禁止说「根据训练数据」「据我所知」「我记得」，只描述需要查什么。工具失败就告知用户未找到，**禁止输出任何训练数据中的法条/条文/案例，附免责声明也不行**
2. **非法律问题也必须搜索**：问题不涉及法律知识 ≠ 可以直接用训练数据回答。任何事实查询（人物、事件、概念、新闻等）都必须通过工具获取
3. **知识库不足就上网搜**：先审视「法律知识检索结果」是否与用户问题相关且充分。不相关/不足/为空 → 立刻调用 mcp_search_web 上网搜索
4. **严禁说「我无法联网」**：工具调用返回的 Observation 就是你获取到的真实网络内容
5. **默认原则：拿不准就搜。宁可多搜一次，不可用训练数据敷衍**

- 所有法律结论必须基于检索结果或工具返回的实际内容
- 搜索也无结果 → 如实告知并给出建议
- 每个结论须引用来源（法条编号或 URL）
- 区分"明确法律规定"与"实践倾向"；不确定时标注置信度
- 以上不构成正式法律意见"""

        # 5. 选择智能体并流式执行
        agent = self._agents.get(reasoning_mode, self._agents["react"])
        final_answer = ""

        for event in agent.stream_run(query, persona=enhanced_prompt):
            if event["event"] == "token":
                final_answer += event["data"]["content"]
            elif event["event"] == "done":
                break
            yield event

        # 6. 结构化引用提取（含联网 URL 和回答中引用的法条）
        citations = self._extract_citations(query, rag_results, final_answer)
        web_citations = self._extract_web_urls(final_answer)
        citations.extend(web_citations[:max(0, 8 - len(citations))])
        # 仅当 KB 或联网有结果时，才从回答中补充法条引用（避免引用模型训练数据）
        if citations:
            statute_citations = self._extract_answer_statute_refs(final_answer, citations)
            citations.extend(statute_citations)
        citations.sort(key=lambda c: c.relevance_score, reverse=True)
        citations = citations[:8]
        # 置信度基于引用来源的相关度（去掉最大最小值后的平均值）
        confidence = self._assess_confidence(citations)

        # 7. 更新记忆 + 持久化对话（含元数据）
        memory.add_interaction(query, final_answer)
        self.conv_store.add_message(cid, "user", query)
        self.conv_store.add_message(cid, "assistant", final_answer, metadata={
            "confidence": confidence,
            "citations": [c.model_dump() for c in citations],
            "reasoning_mode": reasoning_mode,
        })

        # 8. 发送完成事件（含元数据）
        yield {
            "event": "done",
            "data": {
                "confidence": confidence,
                "citations": [c.model_dump() for c in citations],
                "conversation_id": cid,
                "reasoning_mode": reasoning_mode,
            },
        }

    def _extract_answer_entities(self, answer: str) -> dict:
        """从回答文本中提取法律实体，用于与检索结果交叉匹配"""
        import re as _re_ent
        entities = {
            "article_refs": [],      # 第X条、第X条第Y款
            "law_names": [],         # 《民法典》《刑法》等
            "case_names": [],        # XX案、XX诉XX
        }
        if not answer:
            return entities

        # 法条引用: 第X条、第X条第Y款、第X条之Y
        entities["article_refs"] = _re_ent.findall(
            r'第[零一二三四五六七八九十百千\d]+条(?:之[一二三四五六七八九十]+)?(?:\s*第[一二三四五六七八九十\d]+款)?',
            answer
        )
        # 法律名称: 《XXX法》《XXX条例》《XXX规定》
        entities["law_names"] = _re_ent.findall(r'《([^》]{2,30})》', answer)
        # 案例名称: 含"案"、"诉"等关键词的专有名词
        entities["case_names"] = _re_ent.findall(
            r'([^\s，。,\.\n]{3,30}(?:案|纠纷|争议))',
            answer
        )
        return entities

    def _answer_citation_boost(self, retrieval_result: dict, answer: str, entities: dict) -> float:
        """计算检索结果被回答实际引用的加成系数 (1.0 = 未引用, 最高 2.0)"""
        if not answer:
            return 1.0

        title = retrieval_result.get("title", "")
        content = retrieval_result.get("content", "")
        meta = retrieval_result.get("metadata", {})
        article_label = meta.get("article_label", "")

        boost = 1.0

        # 法条号匹配：检索结果的法条号出现在回答中 → +0.6
        if article_label and article_label in answer:
            boost += 0.6

        # 标题关键词出现在回答中 → +0.2 ~ 0.4
        title_keywords = [kw for kw in title.replace('-', ' ').replace('：', ' ').split() if len(kw) >= 2]
        match_count = sum(1 for kw in title_keywords if kw in answer)
        if match_count >= 3:
            boost += 0.4
        elif match_count >= 1:
            boost += 0.2

        # 法律名称匹配 → +0.3
        for law_name in entities.get("law_names", []):
            if law_name in title or law_name in content[:200]:
                boost += 0.3
                break

        # 案例名称匹配 → +0.5
        for case_name in entities.get("case_names", []):
            if case_name in title or case_name in content[:300]:
                boost += 0.5
                break

        # 检索结果的内容片段出现在回答中 → +0.3
        if len(content) >= 30 and content[30:80] in answer:
            boost += 0.3

        return min(2.0, boost)

    def _keyword_match_score(self, query: str, text: str) -> float:
        """计算查询与文本的关键词匹配密度 (0-1)"""
        if not text or not query:
            return 0.0
        score = 0.0
        max_score = 0.0
        for n in [2, 3]:
            for i in range(len(query) - n + 1):
                term = query[i:i + n]
                max_score += n
                if term in text:
                    score += n
        return score / max_score if max_score > 0 else 0.0

    def _score_relevance(self, query: str, result: Dict[str, Any], source_boost: float = 1.0) -> float:
        """计算单条检索结果的相关度 (0-1)，source_boost 为同源文档命中频率加成"""
        content = result.get("content", "")
        doc_type = result.get("doc_type", "")
        emb_score = result.get("score", 0)

        # 关键词匹配密度
        kw = self._keyword_match_score(query, content)

        # 来源权威度
        authority = {"statutes": 1.0, "cases": 0.6, "practical": 0.4}.get(doc_type, 0.5)

        # 向量距离归一化 (距离越小越相关)
        emb_norm = 1.0 / (1.0 + emb_score) if emb_score else 0.5

        relevance = (kw * 0.40 + authority * 0.30 + emb_norm * 0.30) * source_boost
        return round(min(1.0, max(0.0, relevance)), 2)

    def _assess_confidence(self, citations: list) -> float:
        """基于引用来源的相关度计算置信度（去掉最大最小值后的平均值）"""
        if not citations:
            return 0.25

        scores = [c.relevance_score for c in citations]
        if len(scores) >= 3:
            scores.sort()
            scores = scores[1:-1]  # 去掉最大和最小，取中间值
        avg = sum(scores) / len(scores)
        # 封顶 92%，法律建议永不给 100%
        return round(min(0.92, max(0.20, avg)), 2)

    def _extract_citations(
        self, query: str, retrieval_results: list, answer: str = "", max_citations: int = 8
    ) -> list:
        """从结构化检索结果提取引用来源，优先选用回答中实际引用的案例/法条"""
        if not retrieval_results:
            return []

        # 1. 从回答中提取实体，用于交叉匹配
        entities = self._extract_answer_entities(answer) if answer else {}

        # 2. 计算同源频率加成
        source_freq = {}
        for r in retrieval_results:
            src = r.get("metadata", {}).get("source_title", r.get("title", ""))
            source_freq[src] = source_freq.get(src, 0) + 1

        # 3. 计算每条结果的综合分数：查询相关度 × 回答引用加成
        scored = []
        for r in retrieval_results:
            src = r.get("metadata", {}).get("source_title", r.get("title", ""))
            src_boost = min(1.3, 1.0 + 0.05 * (source_freq.get(src, 1) - 1))
            base_relevance = self._score_relevance(query, r, source_boost=src_boost)
            citation_boost = self._answer_citation_boost(r, answer, entities) if answer else 1.0
            final_score = round(base_relevance * citation_boost, 2)
            scored.append((final_score, citation_boost, r))

        # 4. 按最终分数排序
        scored.sort(key=lambda x: x[0], reverse=True)

        # 5. 分类：被回答引用 vs 未被引用
        cited = [(s, b, r) for s, b, r in scored if b > 1.0]
        uncited = [(s, b, r) for s, b, r in scored if b <= 1.0]

        # 6. 先取所有被引用的（保持法规+案例均衡），再从未引用补充
        cited_statutes = [(s, b, r) for s, b, r in cited if r.get("doc_type") == "statutes"]
        cited_cases = [(s, b, r) for s, b, r in cited if r.get("doc_type") == "cases"]
        cited_others = [(s, b, r) for s, b, r in cited if r.get("doc_type") not in ("statutes", "cases")]

        uncited_statutes = [(s, b, r) for s, b, r in uncited if r.get("doc_type") == "statutes"]
        uncited_cases = [(s, b, r) for s, b, r in uncited if r.get("doc_type") == "cases"]
        uncited_others = [(s, b, r) for s, b, r in uncited if r.get("doc_type") not in ("statutes", "cases")]

        # 被引用的优先，配额：4法规 + 3案例 + 1其他
        selected = []
        selected.extend(cited_statutes[:4])
        selected.extend(cited_cases[:3])
        selected.extend(cited_others[:1])

        # 不足额度从未引用中补齐
        remaining_statutes = max(0, 4 - len(cited_statutes))
        remaining_cases = max(0, 3 - len(cited_cases))
        remaining_others = max(0, 1 - len(cited_others))
        selected.extend(uncited_statutes[:remaining_statutes])
        selected.extend(uncited_cases[:remaining_cases])
        selected.extend(uncited_others[:remaining_others])

        # 最终按分数排序
        selected.sort(key=lambda x: x[0], reverse=True)

        # 7. 去重 + 生成引用
        citations = []
        seen_titles = set()
        for final_score, cit_boost, r in selected[:max_citations]:
            title = r.get("title", "")
            if title in seen_titles:
                continue
            seen_titles.add(title)

            doc_type = r.get("doc_type", "")
            source_type = {"statutes": "法规", "cases": "案例", "practical": "实务"}.get(doc_type, "资料")
            meta = r.get("metadata", {})
            article_label = meta.get("article_label", "")
            snippet = r.get("content", "")[:300]

            citations.append(SourceCitation(
                source_type=source_type,
                title=title,
                content=snippet,
                relevance_score=round(min(0.90, final_score), 2),
                article_reference=article_label if article_label else None,
            ))

        return citations

    def _extract_answer_statute_refs(self, answer: str, existing_citations: list) -> list:
        """从回答文本中提取《法律名》第X条引用，生成缺失的引用来源"""
        import re as _re_sr

        if not answer:
            return []

        citations = []
        # 已有引用的标题和法条号集合，用于去重
        existing_titles = {c.title for c in existing_citations}
        existing_refs = {c.article_reference for c in existing_citations if c.article_reference}

        # Pattern: 《法律名》第X条(之Y)?(第Z款)?
        pattern = r'《([^》]{2,40})》\s*第([零一二三四五六七八九十百千\d]+条(?:之[一二三四五六七八九十]+)?(?:\s*第[一二三四五六七八九十\d]+款)?)'

        seen = set()
        for match in _re_sr.finditer(pattern, answer):
            law_name = match.group(1)
            article_ref = match.group(2)
            full_ref = f"《{law_name}》{article_ref}"

            if full_ref in seen:
                continue
            seen.add(full_ref)

            # 去重：已有引用中已包含此法条
            if full_ref in existing_titles or article_ref in existing_refs:
                continue

            # 提取上下文（法条引用前后各取适量文字）
            idx = match.start()
            start = max(0, idx - 80)
            end = min(len(answer), match.end() + 200)
            snippet = answer[start:end].strip()
            if start > 0:
                snippet = '...' + snippet
            if end < len(answer):
                snippet = snippet + '...'

            citations.append(SourceCitation(
                source_type="法规",
                title=f"{law_name} - {article_ref}",
                content=snippet[:300],
                relevance_score=0.85,
                article_reference=article_ref,
            ))

        return citations

    def _extract_web_urls(self, answer: str) -> list:
        """从回答文本中提取真实 URL 并生成引用来源"""
        import re
        urls = re.findall(r'https?://[^\s\)\]。，、；\"\']+', answer)
        if not urls:
            return []

        citations = []
        seen = set()
        for url in urls:
            # 清理尾部标点
            url = re.sub(r'[.,;:。，、；]+$', '', url)
            if url in seen:
                continue
            seen.add(url)

            # 从回答中提取该 URL 附近的上下文作为摘要
            idx = answer.find(url)
            start = max(0, idx - 60)
            end = min(len(answer), idx + len(url) + 80)
            snippet = answer[start:end].replace(url, f'[{url}]').strip()
            if start > 0:
                snippet = '...' + snippet
            if end < len(answer):
                snippet = snippet + '...'

            # 从 URL 周围上下文推断标题，或回退到域名
            from urllib.parse import urlparse
            import re as _re_title
            domain = urlparse(url).netloc.replace('www.', '')
            # 尝试从 URL 附近提取书名号/引号中的文字作为标题
            surrounding = answer[max(0, idx-120):idx]
            title_match = _re_title.search(r'[（《「]([^）》」]{2,40})[）》」]', surrounding)
            if title_match:
                title = title_match.group(1)
            else:
                title = domain

            citations.append(SourceCitation(
                source_type="网页",
                title=title,
                content=snippet[:300],
                relevance_score=0.70,
                article_reference=url,
            ))

        return citations
