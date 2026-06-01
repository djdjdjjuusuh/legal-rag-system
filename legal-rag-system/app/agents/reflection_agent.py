"""Reflection 智能体 - 借鉴 hello-agents MyReflectionAgent"""
import re
from typing import Optional, Iterator, Dict, Any
from app.agents.base import BaseAgent
from app.core.llm import LLMClient
from app.core.message import Message
from app.core.config import Config, get_config
from app.tools.registry import ToolRegistry

INITIAL_PROMPT = """## 角色
{persona}

## 可用工具
{tools}

## ⚠️ 核心铁律（最高优先级，覆盖所有其他规则）

1. **禁止使用模型训练数据**：不得用预训练知识回答任何事实性问题。一切事实性陈述必须来自工具返回的 Observation。
   Thought/回答中禁止说「根据训练数据」「据我所知」「我记得」等
2. **知识库不是唯一来源**：知识库无结果 → 必须调用 mcp_search_web + mcp_fetch 从互联网获取
   **mcp_fetch 失败（403/404等）→ 必须换 mcp_search_web 搜相同关键词**，不要直接放弃
3. **非法律问题也必须搜索**：不能直接用训练数据回答
4. **搜到就合成，禁止循环**：工具返回有效 Observation 后，下一步必须 Finish。同一关键词最多搜 1 次
5. **搜索上限**：联网最多 2 次不同关键词，2 次后仍无结果 → Finish 告知用户。
   Finish 只能给查阅建议，**禁止输出任何训练数据中的法条、条文、案例**
6. **最终回答严禁出现训练数据内容**，附免责声明也不行

## 工具使用格式
- Thought: 分析需要什么信息（不说你"知道"什么）
- Action: `工具名[key1=value1, key2=value2]` 调用工具，**参数必须用 = 号**；或 `Finish[最终答案]` 给出完整回答

## 任务
{query}

**搜索流程（必须遵守）**：
1. 如果问题涉及 URL 链接，先用 mcp_fetch 抓取
2. 如果 mcp_fetch 失败（403/404/连接失败等），**必须立即调用 mcp_search_web 搜索该链接的关键词**（从 URL 和问题中提取），**禁止直接 Finish**
3. mcp_search_web 有结果 → **先验证是否与原始 URL 是同一篇文章**（对比 URL 中的 ID/日期/域名）
4. **验证不通过 → Finish 诚实告知「无法访问该特定文章」**，禁止用不相关搜索结果冒充原文
5. 如果 mcp_search_web 也无结果 → Finish 告知用户

请完成。如需检索信息请先调用工具，否则直接给出完整回答（使用 Action: Finish[答案] 格式）："""

REFLECT_PROMPT = """## 审核角色
你是一位资深法律审核专家。请审查以下回答的质量。

## 原始问题
{query}

## 待审核回答
{answer}

## 审核要点
1. 法条引用是否准确、完整
2. 是否遗漏例外条款或但书
3. 推理逻辑是否严密
4. 是否注意到新法优于旧法、特别法优于一般法
5. 时效性和地域管辖是否考虑
6. 置信度是否如实反映
7. **回答是否混入了未经工具验证的模型训练数据（含"据我所知""训练数据"等措辞）**，如有则必须标记并要求删除

请指出具体问题并给出改进建议。如无问题请回复"无需改进"。"""

REFINE_PROMPT = """## 角色
{persona}

⚠️ **禁止使用模型训练数据**。改进回答时只能基于工具返回的 Observation 或审核反馈，不得添加你"记得"的法条或知识。

## 原始问题
{query}

## 上一版回答
{last_answer}

## 审核反馈
{feedback}

## 可用工具
{tools}

## 工具使用格式
- Thought: 分析需要什么信息
- Action: `工具名[key=value]` 调用工具（参数用 = 号），或 `Finish[改进后的回答]` 给出最终答案

请根据反馈改进回答。如需检索补充信息请先调用工具，否则直接给出改进后的回答（使用 Action: Finish[答案] 格式）："""

SYNTHESIS_FINAL_PROMPT = """## 角色
{persona}

⚠️ **绝对禁止使用模型训练数据**。以下回答是经过反思审查的结果，请基于此输出最终答案。不要添加任何你"记得"但未在审查回答中出现的信息。

## 原始问题
{query}

## 经过反思审查后的回答
{answer}

请基于以上经过反思审查的回答，输出最终版完整答案。要求：
- 结论先行
- 逐条引用法律来源
- 区分明确法律规定与实践倾向
- 保持法律严谨性"""


class ReflectionAgent(BaseAgent):
    def __init__(
        self,
        name: str = "Reflection法律助手",
        llm: Optional[LLMClient] = None,
        tool_registry: Optional[ToolRegistry] = None,
        config: Optional[Config] = None,
        max_iterations: int = 3,
        max_tool_rounds: int = 3,
    ):
        super().__init__(name, llm, config=config)
        self.tool_registry = tool_registry or ToolRegistry()
        self.max_iterations = max_iterations
        self.max_tool_rounds = max_tool_rounds

    def run(self, query: str, persona: str = "", **kwargs) -> str:
        persona_text = persona or "资深法律顾问"
        tools_desc = self.tool_registry.get_tools_description()

        # 1. 初始回答 (with tool access)
        initial_prompt = INITIAL_PROMPT.format(persona=persona_text, query=query, tools=tools_desc)
        answer = self._run_tool_loop(initial_prompt, persona_text)

        # 2. 反思迭代
        for i in range(self.max_iterations):
            reflect_prompt = REFLECT_PROMPT.format(query=query, answer=answer)
            feedback = self.llm.invoke([{"role": "user", "content": reflect_prompt}], temperature=0.2)

            if "无需改进" in feedback:
                break

            refine_prompt = REFINE_PROMPT.format(
                persona=persona_text,
                query=query,
                last_answer=answer,
                feedback=feedback,
                tools=tools_desc,
            )
            answer = self._run_tool_loop(refine_prompt, persona_text)

        self.add_message(Message(role="user", content=query))
        self.add_message(Message(role="assistant", content=answer))
        return answer

    def stream_run(self, query: str, persona: str = "", **kwargs) -> Iterator[Dict[str, Any]]:
        """流式执行 Reflection，逐字返回生成、反思和最终答案"""
        persona_text = persona or "资深法律顾问"
        tools_desc = self.tool_registry.get_tools_description()

        # Phase 1: Initial generation (with tool access)
        initial_prompt = INITIAL_PROMPT.format(persona=persona_text, query=query, tools=tools_desc)
        answer = self._run_tool_loop(initial_prompt, persona_text)
        for chunk in self._stream_text(answer, 5):
            yield {"event": "thinking", "data": {"step": "初步回答", "content": chunk}}

        # Phase 2: Reflection iterations
        for i in range(self.max_iterations):
            reflect_prompt = REFLECT_PROMPT.format(query=query, answer=answer)
            feedback = self.llm.invoke([{"role": "user", "content": reflect_prompt}], temperature=0.2)
            for chunk in self._stream_text(feedback, 5):
                yield {"event": "thinking", "data": {"step": f"反思审查 第{i+1}轮", "content": chunk}}

            if "无需改进" in feedback:
                break

            refine_prompt = REFINE_PROMPT.format(
                persona=persona_text,
                query=query,
                last_answer=answer,
                feedback=feedback,
                tools=tools_desc,
            )
            answer = self._run_tool_loop(refine_prompt, persona_text)
            for chunk in self._stream_text(answer, 5):
                yield {"event": "thinking", "data": {"step": f"改进回答 第{i+1}轮", "content": chunk}}

        # Phase 3: Stream final polished answer
        final_prompt = SYNTHESIS_FINAL_PROMPT.format(
            persona=persona_text,
            query=query,
            answer=answer,
        )

        final_answer = ""
        for token in self.llm.stream_invoke([{"role": "user", "content": final_prompt}], temperature=0.3):
            final_answer += token
            yield {"event": "token", "data": {"content": token}}

        self.add_message(Message(role="user", content=query))
        self.add_message(Message(role="assistant", content=final_answer))
        yield {"event": "done", "data": {}}

    def _run_tool_loop(self, initial_prompt: str, persona: str) -> str:
        """执行工具调用循环，最多 max_tool_rounds 轮，返回最终答案"""
        messages = [{"role": "user", "content": initial_prompt}]
        history: list = []
        mcp_empty_count = 0

        for _ in range(self.max_tool_rounds):
            response = self.llm.invoke(messages, temperature=0.3)
            thought, action = self._parse_output(response)

            if action and "Finish" in action:
                answer = self._extract_finish(action)
                # 训练数据检测
                training_patterns = ['训练数据', '据我所知', '我记得', '预训练', '我的知识库']
                if any(p in answer for p in training_patterns):
                    answer = "当前知识库和联网搜索均未找到与您问题直接相关的内容。建议您查阅相关法律法规或咨询专业律师获取准确意见。"
                return answer

            if action and not action.startswith("Finish"):
                tool_name, tool_input = self._parse_tool(action)
                if tool_name:
                    try:
                        observation = self.tool_registry.execute_tool(tool_name, tool_input)
                    except Exception as e:
                        observation = f"工具执行出错: {e}"
                    history.append(f"Thought: {thought}")
                    history.append(f"Action: {action}")
                    history.append(f"Observation: {observation}")
                    # 工具禁用检测
                    if tool_name == "mcp_search_web" and ("已被禁用" in observation or "disabled" in observation.lower()):
                        return "联网搜索功能当前已禁用，本地知识库也未找到相关内容。如需获取准确法律信息，建议启用联网搜索功能，或自行查阅相关法律法规及司法解释。"
                    # 搜索空结果计数
                    if tool_name == "mcp_search_web" and ("未找到" in observation or "无结果" in observation or "失败" in observation):
                        mcp_empty_count += 1
                        if mcp_empty_count >= 2:
                            return "当前知识库和联网搜索均未找到与您问题直接相关的内容。建议您查阅相关法律法规或咨询专业律师获取准确意见。"
                    history_str = "\n".join(history[-10:])
                    followup = f"""工具执行结果已返回。请继续。
⚠️ 重要：如果 mcp_fetch 抓取失败（403/404等），必须换用 mcp_search_web 搜索相同主题的关键词，不要直接放弃。
如需更多信息可继续调用工具，否则用 Action: Finish[答案] 给出完整回答。

## 工具执行历史
{history_str}"""
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": followup})
                    continue

            # 未按 Action 格式回复：如果还没调过工具，强制要求先调工具
            if not history:
                followup = f"""你必须先调用工具获取信息，不能直接回答。请严格按照以下格式：

Thought: 分析需要什么信息
Action: 工具名[key1=value1] 或 Finish[最终答案]

请重新用 Action 格式回复："""
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": followup})
                continue
            return response.strip()

        history_str = "\n".join(history[-10:]) if history else "无"
        synth_prompt = f"""基于以下工具执行历史，给出完整回答。如果工具结果为空，告知用户未找到，禁止输出训练数据中的法条：

{history_str}

请直接给出完整回答（无需工具格式）："""
        return self.llm.invoke([{"role": "user", "content": synth_prompt}], temperature=0.3)

    def _parse_output(self, text: str):
        thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|$)", text, re.DOTALL)
        action_match = re.search(r"Action:\s*(.+)", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else ""
        action = action_match.group(1).strip() if action_match else ""
        return thought, action

    def _parse_tool(self, action: str):
        action = action.strip().strip("`")
        match = re.match(r"(\w+)\[(.*)\]", action.strip())
        if match:
            return match.group(1), match.group(2)
        return "", ""

    def _extract_finish(self, action: str) -> str:
        action = action.strip().strip("`")
        match = re.search(r"Finish\[(.*)\]", action, re.DOTALL)
        return match.group(1).strip() if match else action
