"""Plan-and-Solve 智能体 - 借鉴 hello-agents MyPlanAndSolveAgent"""
import re
from typing import Optional, Iterator, Dict, Any
from app.agents.base import BaseAgent
from app.core.llm import LLMClient
from app.core.message import Message
from app.core.config import Config, get_config
from app.tools.registry import ToolRegistry

PLANNER_PROMPT = """## 角色
{persona}

⚠️ 规划时只能基于用户问题和可用工具，不得使用模型训练数据中的法律知识预设答案。

## 任务
将法律问题分解为可执行的子步骤。

## 问题
{query}

## 输出格式
请以 JSON 数组输出计划步骤，每步一行:
["步骤1: xxx", "步骤2: xxx", ...]

## 计划:"""

EXECUTOR_PROMPT = """## 角色
{persona}

## ⚠️ 核心铁律（最高优先级）

1. **禁止使用模型训练数据**：不得用预训练知识回答任何事实性问题。一切事实性陈述必须来自工具返回的 Observation。
   Thought/回答中禁止说「根据训练数据」「据我所知」「我记得」等，只描述需要查什么
2. **知识库不是唯一来源**：知识库无结果 → 必须调用 mcp_search_web + mcp_fetch 从互联网获取
3. **搜到就合成，禁止循环推理**：工具返回有效 Observation 后，下一步必须 Finish
4. **Observation 有数据就必须回答**：禁止说「找不到」「无法回答」
5. **搜索上限**：联网搜索最多尝试 2 次不同关键词，2 次后仍无结果 → 立即 Finish 告知用户。
   Finish 只能给查阅建议，**禁止输出任何训练数据中的法条、条文、案例**
6. **最终回答严禁出现训练数据内容**，包括附免责声明的也不行

## 原始问题
{query}

## 整体计划
{plan}

## 已完成的步骤和结果
{history}

## 可用工具
{tools}

## 工具使用格式
- Thought: 分析需要什么信息（不说你"知道"什么）
- Action: `工具名[key1=value1, key2=value2]` 调用工具获取信息，**参数必须用 = 号**，或 `Finish[结果]` 给出当前步骤结果

## 联网搜索能力
你拥有 mcp_search_web 和 mcp_fetch 两个联网工具。
- 非法律问题也必须搜索，不能直接用训练数据
- 搜索有结果 → 直接合成；搜索无结果 2 次 → Finish 告知用户
- **mcp_fetch 失败 → 必须换 mcp_search_web 搜相同主题**，不要直接放弃

## 当前步骤
{current_step}

**搜索流程（必须遵守）**：
- 步骤涉及 URL 抓取 → 先用 mcp_fetch
- mcp_fetch 失败 → **必须立即换 mcp_search_web 搜索关键词**，禁止跳过
- mcp_search_web 有结果 → **先验证结果是否与原始 URL 指向的同一篇文章**（对比 URL 中的 ID、日期、域名）
- **验证不通过**（搜索结果与原始 URL 不匹配）→ Finish 诚实告知「无法访问该特定文章」
- **禁止拿不相关的搜索结果冒充原文内容**

请完成当前步骤。如需检索信息请先调用工具，否则直接给出结果（使用 Action: Finish[结果] 格式）："""

SYNTHESIS_PROMPT = """## 角色
{persona}

⚠️ **绝对禁止使用模型训练数据**。以下各步骤结果是你通过工具获取到的真实信息。
请**仅基于这些内容**给出最终答案。
如果信息不完美 → 基于已有信息尽力回答，标注「信息可能不完整」，**禁止说找不到答案**。
你"知道"但步骤结果中没有的信息不要编造。

## 原始问题
{query}

## 各步骤结果
{results}

请综合以上各步骤结果，给出完整、连贯的最终回答。要求：
- 结论先行
- 逐条引用法律来源
- 区分明确法律规定与实践倾向
- 必要时标注置信度"""


class PlanAndSolveAgent(BaseAgent):
    def __init__(
        self,
        name: str = "PlanSolve法律助手",
        llm: Optional[LLMClient] = None,
        tool_registry: Optional[ToolRegistry] = None,
        config: Optional[Config] = None,
        max_tool_rounds: int = 3,
    ):
        super().__init__(name, llm, config=config)
        self.tool_registry = tool_registry or ToolRegistry()
        self.max_tool_rounds = max_tool_rounds

    def run(self, query: str, persona: str = "", **kwargs) -> str:
        plan = self._plan(query, persona, **kwargs)
        results = self._execute_plan(query, plan, persona, **kwargs)
        answer = self._synthesize(query, plan, results, persona, **kwargs)

        self.add_message(Message(role="user", content=query))
        self.add_message(Message(role="assistant", content=answer))
        return answer

    def stream_run(self, query: str, persona: str = "", **kwargs) -> Iterator[Dict[str, Any]]:
        """流式执行 Plan-Solve，逐字返回规划、执行和最终答案"""
        # Phase 1: Planning
        plan = self._plan(query, persona, **kwargs)
        plan_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(plan))
        plan_content = f"将问题分解为 {len(plan)} 个步骤：\n{plan_text}"
        for chunk in self._stream_text(plan_content, 5):
            yield {"event": "thinking", "data": {"step": "制定计划", "content": chunk}}

        # Phase 2: Execute each step
        results = []
        history_parts = []
        tools_desc = self.tool_registry.get_tools_description()
        for i, step_item in enumerate(plan):
            prompt = EXECUTOR_PROMPT.format(
                persona=persona or "法律分析专家",
                query=query,
                plan=plan_text,
                history="\n".join(history_parts) if history_parts else "无",
                tools=tools_desc,
                current_step=step_item,
            )
            step_result = self._run_tool_loop(query, prompt)
            results.append({"step": step_item, "result": step_result})
            history_parts.append(f"步骤{i+1}: {step_item}\n结果: {step_result}")
            for chunk in self._stream_text(step_result, 5):
                yield {"event": "thinking", "data": {"step": f"执行步骤{i+1}: {step_item}", "content": chunk}}

        # Phase 3: Synthesize (streamed)
        results_text = "\n\n".join(
            f"**步骤{i+1}**: {r['step']}\n**结果**: {r['result']}"
            for i, r in enumerate(results)
        )
        synth_prompt = SYNTHESIS_PROMPT.format(
            persona=persona or "法律分析专家",
            query=query,
            results=results_text,
        )

        final_answer = ""
        for token in self.llm.stream_invoke([{"role": "user", "content": synth_prompt}], temperature=0.3):
            final_answer += token
            yield {"event": "token", "data": {"content": token}}

        self.add_message(Message(role="user", content=query))
        self.add_message(Message(role="assistant", content=final_answer))
        yield {"event": "done", "data": {}}

    def _plan(self, query: str, persona: str, **kwargs) -> list:
        prompt = PLANNER_PROMPT.format(persona=persona or "法律分析专家", query=query)
        response = self.llm.invoke(
            [{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return self._parse_plan(response)

    def _execute_plan(self, query: str, plan: list, persona: str, **kwargs) -> list:
        results = []
        history_parts = []
        tools_desc = self.tool_registry.get_tools_description()
        search_disabled = False
        for i, step in enumerate(plan):
            # 如果之前已发现搜索禁用，跳过依赖搜索的步骤
            if search_disabled and any(kw in step for kw in ['搜索', '联网', 'mcp', '检索互联网']):
                results.append({"step": step, "result": "联网搜索功能已禁用，跳过此步骤。"})
                history_parts.append(f"步骤{i+1}: {step}\n结果: 联网搜索已禁用，已跳过")
                continue
            prompt = EXECUTOR_PROMPT.format(
                persona=persona or "法律分析专家",
                query=query,
                plan="\n".join(f"{j+1}. {s}" for j, s in enumerate(plan)),
                history="\n".join(history_parts) if history_parts else "无",
                tools=tools_desc,
                current_step=step,
            )
            step_result = self._run_tool_loop(query, prompt)
            # 检测搜索是否被禁用
            if "已被禁用" in step_result or "已禁用" in step_result:
                search_disabled = True
            results.append({"step": step, "result": step_result})
            history_parts.append(f"步骤{i+1}: {step}\n结果: {step_result}")
        return results

    def _run_tool_loop(self, query: str, initial_prompt: str) -> str:
        """执行工具调用循环，最多 max_tool_rounds 轮，返回步骤最终结果"""
        messages = [{"role": "user", "content": initial_prompt}]
        history: list = []
        mcp_empty_count = 0

        for _ in range(self.max_tool_rounds):
            response = self.llm.invoke(messages, temperature=0.2)
            thought, action = self._parse_output(response)

            if action and "Finish" in action:
                answer = self._extract_finish(action)
                # 训练数据检测
                training_patterns = ['训练数据', '据我所知', '我记得', '预训练', '我的知识库']
                if any(p in answer for p in training_patterns):
                    answer = "当前步骤未能通过工具获取到有效信息。建议查阅相关法律法规或咨询专业律师。"
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
                        return "联网搜索功能当前已禁用。建议启用联网搜索功能，或自行查阅相关法律法规。"
                    # 搜索空结果计数
                    if tool_name == "mcp_search_web" and ("未找到" in observation or "无结果" in observation or "失败" in observation):
                        mcp_empty_count += 1
                        if mcp_empty_count >= 2:
                            return "联网搜索多次均无结果，建议查阅相关法律法规或咨询专业律师。"
                    history_str = "\n".join(history[-10:])
                    followup = f"""工具执行结果已返回，请继续完成当前步骤。
⚠️ 重要：如果 mcp_fetch 抓取失败（403/404等），必须换用 mcp_search_web 搜索相同主题的关键词，不要直接放弃。
如需更多信息可继续调用工具，否则用 Action: Finish[结果] 给出本步骤的完整结果。

## 工具执行历史
{history_str}"""
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": followup})
                    continue

            # 未按 Action 格式回复：如果还没调过工具，强制要求先调工具
            if not history:
                followup = f"""你必须先调用工具获取信息，不能直接回答。请严格按照以下格式：

Thought: 分析需要什么信息
Action: 工具名[key1=value1] 或 Finish[结果]

当前步骤需要你先调用相关工具获取信息。请重新用 Action 格式回复："""
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": followup})
                continue
            return response.strip()

        history_str = "\n".join(history[-10:]) if history else "无"
        synth_prompt = f"""基于以下工具执行历史，给出当前步骤的完整结果。如果工具结果为空，告知用户未找到，禁止输出训练数据中的法条：

{history_str}

请直接给出结果（无需工具格式）："""
        return self.llm.invoke([{"role": "user", "content": synth_prompt}], temperature=0.2)

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

    def _synthesize(self, query: str, plan: list, results: list, persona: str, **kwargs) -> str:
        results_text = "\n\n".join(
            f"**步骤{i+1}**: {r['step']}\n**结果**: {r['result']}"
            for i, r in enumerate(results)
        )
        prompt = SYNTHESIS_PROMPT.format(
            persona=persona or "法律分析专家",
            query=query,
            results=results_text,
        )
        return self.llm.invoke([{"role": "user", "content": prompt}])

    def _parse_plan(self, text: str) -> list:
        import json
        text = text.strip()
        if "[" in text and "]" in text:
            start = text.index("[")
            end = text.rindex("]") + 1
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        return [line.strip("- 1234567890. ") for line in text.split("\n") if line.strip()]
