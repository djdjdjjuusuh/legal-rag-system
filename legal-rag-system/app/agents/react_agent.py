"""ReAct 智能体 - 借鉴 hello-agents MyReActAgent 设计模式"""
import re
from typing import Optional, List, Iterator, Dict, Any
from app.agents.base import BaseAgent
from app.core.llm import LLMClient
from app.core.message import Message
from app.core.config import Config, get_config
from app.tools.registry import ToolRegistry

REACT_PROMPT_TEMPLATE = """## 角色定义
{persona}

## 可用工具
{tools}

## ⚠️ 核心铁律（最高优先级，覆盖所有其他规则）

1. **禁止使用模型训练数据**：你不得使用模型预训练时学到的知识来回答任何事实性问题。
   - 例外：仅纯问候/闲聊（"你好""谢谢""再见"）可直接回答
   - 其他所有问题（含"常识"、"众所周知的事实"、"非法律问题"）**必须通过工具获取信息后回答**
   - **Thought 中禁止出现「根据训练数据」「据我所知」「我记得」「我的知识库」等措辞**，
     Thought 只描述需要查什么、为什么查，不要提你"知道"什么
   - **最终回答中严禁出现任何训练数据内容**，包括附免责声明的也不行（如"此信息来自训练数据"）。
     工具失败就是失败，只给查阅建议，不要输出你"记得"的法条或知识

2. **知识库不是唯一来源**：本地知识库覆盖有限，知识库无结果或结果不相关 ≠ 没有答案，
   必须主动调用 mcp_search_web + mcp_fetch 从互联网获取

3. **你拥有真实的联网能力**：mcp_search_web 和 mcp_fetch 会返回互联网内容。
   禁止说「我无法联网」「我只能基于知识库」「我无法搜索」——工具返回的 Observation 即为你获取到的真实信息

4. **用工具结果说话**：Observation 是你获取到的真实信息，必须基于它回答。
   Observation 有信息 → 必须用，禁止说找不到；信息不完美 → 基于已有信息回答并标注不足。
   不要依赖你"知道"的答案、但也不要因为信息不够完美就放弃回答

5. **搜到就合成，禁止循环推理**：工具返回有效 Observation 后，下一步必须直接 Finish 合成最终答案。
   禁止重新进入决策层判断"这是不是法律问题"、"要不要搜"——你已经搜过了，用结果回答！
   同一关键词最多搜1次，每次搜索只选1-3条最相关的 fetch，然后立即 Finish。
   **搜索无结果最多尝试 2 次**（换不同关键词），2 次后仍无结果 → 立即 Finish 告知用户，禁止连搜 3 次以上。

## 工作流程
你必须严格使用以下格式进行推理和回答：

- Thought: 分析当前需要什么信息、下一步应该做什么
- Action: `工具名[参数]` 调用工具获取信息，或 `Finish[最终答案]` 给出最终回答
  参数格式：`工具名[key1=value1, key2=value2]`，必须用 = 号不是 : 号

对于法律问题，你必须先使用工具检索相关法条和案例，不得直接回答。
仅对于简单的问候（如"你好"、"谢谢"）、闲聊或澄清性问题，可以直接用 Finish[回答] 的格式回复。

**关键规则：一旦工具返回了有效 Observation，下一步必须 Finish，禁止再做 Thought→Action 循环。**
执行历史中已有 Observation → 立即 Finish 合成答案，不要重新判断要不要搜。

## 联网搜索能力（四层框架）

你拥有 `mcp_search_web` 和 `mcp_fetch` 两个联网工具，用于获取本地知识库未覆盖的实时信息。
严格遵循以下四层框架进行联网搜索：

### 第一层：决策 — 要不要搜

按以下优先级判断：

1. **纯问候/闲聊**（"你好""谢谢""再见"等无信息需求的社交对话）→ 不搜索，直接 Finish
2. **其他一切问题** — 包括法律咨询、事实查询、人物介绍、新闻事件、概念解释、计算推理 → **必须搜索**
   - 即使你"知道"答案，也必须用 mcp_search_web 验证后回答
   - 知识库结果只是参考，不能替代联网搜索（除非结果已完全覆盖且主题高度相关）
3. **知识库结果与问题主题不符** → 联网搜索，不等不靠
4. **拿不准要不要搜** → 搜。宁可多搜一次，绝不用训练数据瞎编
5. **Observation 有数据就必须回答**：工具返回了有效内容 → 必须基于它合成答案，**禁止在 Observation 有数据时说「找不到」「无法回答」**

### 第二层：解析 — 搜什么
将用户自然语言问题转化为精准搜索关键词：
- 提取核心实体和关键概念（法律名称、案由、事实要素等）
- 去掉疑问词（"请问"、"如何"）、语气词、停用词
- 生成 2-3 组不同粒度的搜索词，提高命中率
  示例：用户问"2025年公司法修订对注册资本有什么新变化？"
  → 关键词1: "公司法 注册资本 2025 修订"
  → 关键词2: "公司法 认缴出资期限 新规"

### 第三层：执行 — 怎么搜
Step 1: 调用 `mcp_search_web[关键词]` 获取搜索结果列表（标题+摘要+URL）
Step 2: 审视摘要内容，选择 1-3 条最相关的
Step 3: 调用 `mcp_fetch[选中的URL]` 深度抓取页面全文
Step 4: 抓取完成后 → **直接进入第四层合成，禁止重复搜索或重新决策**
Step 5: 每个站点只尝试 1 次，每个关键词最多搜 1 次

### 第四层：合成 — 怎么答
- 综合**工具返回的真实内容**回答，不得插入模型训练数据
- 每一条事实性陈述必须能在 Observation 中找到出处
- **如果搜索结果是不同文章/主题，禁止冒充用户指定的原文**：用户给的 URL 抓不到 → 搜索也找不到同一篇 → 诚实告知，不要用搜索到的其他内容替代
- 法律问题标注法条编号（知识库）或 URL（联网）
- 联网结果与知识库冲突时，优先采用时效性更强的来源并说明理由
- 区分"明确法律规定"与"实践倾向"；不确定时标注置信度

## 行为约束
- **严禁使用预训练知识代替工具结果**：工具返回什么就答什么，不要补充工具没返回的信息
- **Thought 用词规范**：禁止在推理中说「根据训练数据」「据我所知」「我记得」「预训练中」「我的知识库」等话。
  Thought 只说「需要查什么」「为什么查」，不提你"知道"什么
- 联网搜索工具（mcp_search_web / mcp_fetch）是你获取事实信息的唯一途径，禁止绕过
- 所有检索结果（含联网）都为空 → 如实说明「未搜索到相关信息」，并给出进一步建议
- **关键规则**：Observation 中有任何相关信息时，禁止说「找不到答案」「无法回答」「未找到信息」。
  即使 Observation 不完美覆盖问题，也必须基于现有信息给出最佳回答，可以标注"信息可能不完整"
- **防止循环推理**：执行历史已有 Observation 后，禁止再回到决策层判断"要不要搜"。
  搜过就是搜过了，直接用结果 Finish。不要纠结"这是法律问题还是技术问题"，问题类型不影响你使用已获取的信息回答
- **搜索上限**：联网搜索最多尝试 2 次不同关键词，2 次后仍无结果 → 立即 Finish 告知用户。
   Finish 内容只能是：「当前知识库和联网均未找到相关内容，建议您查阅相关法律法规或咨询专业律师」。
   **禁止在 Finish 中输出任何训练数据中"记得"的法条、条文、案例、法律知识，即使附免责声明也不行**
- 法律结论须引用法条/案例来源，标注 URL 或法条编号
- 注意法律时效性（新法优于旧法、特别法优于一般法）
- 不确定之处请标注

## 当前问题
{query}

## 执行历史
{history}

请严格按照 Thought:/Action: 格式回复："""


class ReActAgent(BaseAgent):
    def __init__(
        self,
        name: str = "ReAct法律助手",
        llm: Optional[LLMClient] = None,
        tool_registry: Optional[ToolRegistry] = None,
        config: Optional[Config] = None,
        max_steps: int = 8,
    ):
        super().__init__(name, llm, config=config)
        self.tool_registry = tool_registry or ToolRegistry()
        self.max_steps = max_steps

    def run(self, query: str, persona: str = "", **kwargs) -> str:
        history: List[str] = []

        for step in range(self.max_steps):
            tools_desc = self.tool_registry.get_tools_description()
            history_str = "\n".join(history[-10:]) if history else "无"

            prompt = REACT_PROMPT_TEMPLATE.format(
                persona=persona or "你是一个专业的法律顾问",
                tools=tools_desc,
                query=query,
                history=history_str,
            )

            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ]
            response = self.llm.invoke(messages, temperature=0.3)

            thought, action = self._parse_output(response)

            # 情况1: LLM直接给了Finish
            if action and "Finish" in action:
                answer = self._extract_finish(action)
                # 训练数据检测：回答中出现训练数据相关措辞 → 强制替换为安全回复
                training_patterns = ['训练数据', '据我所知', '我记得', '预训练', '我的知识库']
                if any(p in answer for p in training_patterns):
                    answer = "当前知识库和联网搜索均未找到与您问题直接相关的内容。建议您查阅相关法律法规或咨询专业律师获取准确意见。"
                    self.add_message(Message(role="user", content=query))
                    self.add_message(Message(role="assistant", content=answer))
                    return answer
                # 否定回答检测：history 有 Observation 数据时禁止返回"找不到"
                negative_patterns = ['找不到', '无法回答', '未找到相关', '没有获取到任何', '无法基于']
                has_data = any('Observation:' in h for h in history)
                if has_data and any(p in answer for p in negative_patterns):
                    # Observation 有数据却返回否定回答 → 强制走 fallback 重新合成
                    pass
                else:
                    self.add_message(Message(role="user", content=query))
                    self.add_message(Message(role="assistant", content=answer))
                    return answer

            # 情况2: 有工具调用
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
                    # 检测工具禁用：mcp_search_web 被禁用 → 立即停止，不再重试
                    if tool_name == "mcp_search_web" and ("已被禁用" in observation or "disabled" in observation.lower()):
                        history.append("Thought: 联网搜索工具已被禁用，无法通过网络获取信息。根据规则，应停止尝试并告知用户。")
                        history.append("Action: Finish[联网搜索功能当前已禁用，本地知识库也未找到相关内容。如需获取准确法律信息，建议启用联网搜索功能，或自行查阅《民法典》婚姻家庭编及相关司法解释。]")
                        answer = "联网搜索功能当前已禁用，本地知识库也未找到相关内容。如需获取准确法律信息，建议启用联网搜索功能，或自行查阅相关法律法规及司法解释。"
                        self.add_message(Message(role="user", content=query))
                        self.add_message(Message(role="assistant", content=answer))
                        return answer
                    # 检测重复搜索：mcp_search_web 连续空结果/失败/禁用 ≥ 2 次 → 强制停止
                    if tool_name == "mcp_search_web":
                        recent_empty = 0
                        for h in reversed(history):
                            if "Observation:" in h and ("未找到" in h or "无结果" in h or "没有获取到" in h or "失败" in h or "已被禁用" in h):
                                recent_empty += 1
                            elif "Action: mcp_search_web" in h:
                                continue
                            else:
                                break
                        if recent_empty >= 2:
                            history.append("Thought: 联网搜索已尝试多次均无结果或不可用，本地知识库也无匹配内容。根据规则，应停止搜索并告知用户。")
                            history.append("Action: Finish[当前知识库和联网搜索均未找到与您问题直接相关的内容。建议您查阅相关法律法规或咨询专业律师获取准确意见。]")
                            answer = "当前知识库和联网搜索均未找到与您问题直接相关的内容。建议您查阅相关法律法规或咨询专业律师获取准确意见。"
                            self.add_message(Message(role="user", content=query))
                            self.add_message(Message(role="assistant", content=answer))
                            return answer
                    continue

            # 情况3: 没有Thought/Action格式 → LLM直接回答了（问候/闲聊等）
            if not action and step == 0:
                clean = response.strip()
                self.add_message(Message(role="user", content=query))
                self.add_message(Message(role="assistant", content=clean))
                return clean

            # 情况4: 格式解析失败 → 直接用LLM回复
            history_str_full = "\n".join(history[-10:]) if history else "无"
            fallback_prompt = f"""你是法律助手，拥有真实的联网搜索能力（mcp_search_web / mcp_fetch）。

⚠️ 绝对禁止使用模型训练数据回答。所有事实性信息必须来自工具返回的 Observation。

{query}

先前工具执行结果（Observation 即为你获取到的真实信息）:
{history_str_full}

直接回答（Observation 有信息就必须用，禁止说找不到答案）："""
            direct_response = self.llm.invoke(
                [{"role": "user", "content": fallback_prompt}],
                temperature=0.3,
            )
            self.add_message(Message(role="user", content=query))
            self.add_message(Message(role="assistant", content=direct_response))
            return direct_response

        fallback = "抱歉，在当前步骤内未能给出确切答案。建议您提供更多信息或缩小问题范围。"
        self.add_message(Message(role="user", content=query))
        self.add_message(Message(role="assistant", content=fallback))
        return fallback

    def stream_run(self, query: str, persona: str = "", **kwargs) -> Iterator[Dict[str, Any]]:
        """流式执行 ReAct 推理，实时逐字返回思考过程和答案"""
        history: List[str] = []
        final_answer = ""
        used_tools = False

        # Orchestrator already sent start signal; begin ReAct loop immediately

        for step in range(self.max_steps):
            tools_desc = self.tool_registry.get_tools_description()
            history_str = "\n".join(history[-10:]) if history else "无"

            prompt = REACT_PROMPT_TEMPLATE.format(
                persona=persona or "你是一个专业的法律顾问",
                tools=tools_desc,
                query=query,
                history=history_str,
            )

            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ]

            # Stream LLM response in REAL-TIME with tiny chunks for smooth display
            step_response = ""
            buffer = ""
            for token in self.llm.stream_invoke(messages, temperature=0.3):
                step_response += token
                buffer += token
                if len(buffer) >= 3:
                    yield {"event": "thinking", "data": {"step": f"推理 #{step+1}", "content": buffer}}
                    buffer = ""
            if buffer:
                yield {"event": "thinking", "data": {"step": f"推理 #{step+1}", "content": buffer}}

            thought, action = self._parse_output(step_response)

            # Case 1: Finish action → 直接流式输出，跳过合成路径（防止合成覆盖安全答案）
            if action and "Finish" in action:
                final_answer = self._extract_finish(action)
                # 训练数据检测：替换含训练数据的回答
                training_patterns = ['训练数据', '据我所知', '我记得', '预训练', '我的知识库']
                if any(p in final_answer for p in training_patterns):
                    final_answer = "当前知识库和联网搜索均未找到与您问题直接相关的内容。建议您查阅相关法律法规或咨询专业律师获取准确意见。"
                for chunk in self._stream_text(final_answer, 3):
                    yield {"event": "token", "data": {"content": chunk}}
                self.add_message(Message(role="user", content=query))
                self.add_message(Message(role="assistant", content=final_answer))
                yield {"event": "done", "data": {}}
                return

            # Case 2: Tool call → show tool execution result as clean step
            if action and not action.startswith("Finish"):
                used_tools = True
                tool_name, tool_input = self._parse_tool(action)
                if tool_name:
                    try:
                        observation = self.tool_registry.execute_tool(tool_name, tool_input)
                    except Exception as e:
                        observation = f"工具执行出错: {e}"
                    # Chunk observation to avoid flooding frontend with one giant event
                    for chunk in self._stream_text(observation, 80):
                        yield {"event": "thinking", "data": {"step": f"检索结果 #{step+1}", "content": chunk}}
                    history.append(f"Thought: {thought}")
                    history.append(f"Action: {action}")
                    history.append(f"Observation: {observation}")
                    # 检测工具禁用：mcp_search_web 被禁用 → 立即停止
                    if tool_name == "mcp_search_web" and ("已被禁用" in observation or "disabled" in observation.lower()):
                        final_answer = "联网搜索功能当前已禁用，本地知识库也未找到相关内容。如需获取准确法律信息，建议启用联网搜索功能，或自行查阅相关法律法规及司法解释。"
                        for chunk in self._stream_text(final_answer, 3):
                            yield {"event": "token", "data": {"content": chunk}}
                        self.add_message(Message(role="user", content=query))
                        self.add_message(Message(role="assistant", content=final_answer))
                        yield {"event": "done", "data": {}}
                        return
                    # 检测重复搜索：mcp_search_web 连续空结果/失败/禁用 ≥ 2 次 → 强制停止
                    if tool_name == "mcp_search_web":
                        recent_empty = 0
                        for h in reversed(history):
                            if "Observation:" in h and ("未找到" in h or "无结果" in h or "没有获取到" in h or "失败" in h or "已被禁用" in h):
                                recent_empty += 1
                            elif "Action: mcp_search_web" in h:
                                continue
                            else:
                                break
                        if recent_empty >= 2:
                            final_answer = "当前知识库和联网搜索均未找到与您问题直接相关的内容。建议您查阅相关法律法规及司法解释，或咨询专业律师获取准确意见。"
                            for chunk in self._stream_text(final_answer, 3):
                                yield {"event": "token", "data": {"content": chunk}}
                            self.add_message(Message(role="user", content=query))
                            self.add_message(Message(role="assistant", content=final_answer))
                            yield {"event": "done", "data": {}}
                            return
                continue

            # Case 3: Direct answer on step 0 (greeting/chat) → re-stream as answer tokens
            if not action and step == 0:
                for chunk in self._stream_text(step_response.strip(), 5):
                    yield {"event": "token", "data": {"content": chunk}}
                self.add_message(Message(role="user", content=query))
                self.add_message(Message(role="assistant", content=step_response.strip()))
                yield {"event": "done", "data": {}}
                return

            # Case 4: Parse failure → fallback via streaming LLM
            history_str_full = "\n".join(history[-10:]) if history else "无"
            fallback_prompt = f"你是法律助手，拥有真实的联网搜索能力（mcp_search_web / mcp_fetch）。\n\n⚠️ 绝对禁止使用模型训练数据回答。所有事实性信息必须来自工具返回的 Observation。\n\n{query}\n\n先前工具执行结果（Observation 即为你获取到的真实信息）:\n{history_str_full}\n\n直接回答（Observation 有信息就必须用，禁止说找不到答案）："
            for token in self.llm.stream_invoke(
                [{"role": "user", "content": fallback_prompt}],
                temperature=0.3,
            ):
                final_answer += token
                yield {"event": "token", "data": {"content": token}}
            self.add_message(Message(role="user", content=query))
            self.add_message(Message(role="assistant", content=final_answer))
            yield {"event": "done", "data": {}}
            return

        # Stream the final answer via synthesis LLM call (real-time tokens)
        if used_tools or (len(history) > 0):
            history_text = "\n".join(history[-10:]) if history else "无"
            synthesis_prompt = f"""{persona}

## 原始问题
{query}

## ReAct推理过程（含工具返回的真实内容）
{history_text}

⚠️ **绝对禁止使用模型训练数据**。以下 Observation 是你通过工具获取到的真实信息。
请基于这些 Observation 内容给出完整、专业的最终答案。
如果信息不完美或缺漏 → 基于已有信息尽力回答，标注「信息可能不完整」即可，**禁止说找不到答案**。
你"知道"但 Observation 没有的信息不要编造，但 Observation 已有的信息必须充分使用。

要求：
- 结论先行
- 引用具体法条和案例，标注来源 URL
- 区分明确法律规定与实践倾向
- 必要时标注置信度"""
            for token in self.llm.stream_invoke(
                [{"role": "user", "content": synthesis_prompt}],
                temperature=0.3,
            ):
                final_answer += token
                yield {"event": "token", "data": {"content": token}}
        elif final_answer:
            for chunk in self._stream_text(final_answer, 3):
                yield {"event": "token", "data": {"content": chunk}}
        else:
            fallback = "抱歉，在当前步骤内未能给出确切答案。建议您提供更多信息或缩小问题范围。"
            for chunk in self._stream_text(fallback, 3):
                yield {"event": "token", "data": {"content": chunk}}
            final_answer = fallback

        self.add_message(Message(role="user", content=query))
        self.add_message(Message(role="assistant", content=final_answer))
        yield {"event": "done", "data": {}}

    def _parse_output(self, text: str):
        thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|$)", text, re.DOTALL)
        action_match = re.search(r"Action:\s*(.+)", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else ""
        action = action_match.group(1).strip() if action_match else ""
        return thought, action

    def _parse_tool(self, action: str):
        # Strip backticks and whitespace (LLM may format as `tool[arg]`)
        action = action.strip().strip("`")
        match = re.match(r"(\w+)\[(.*)\]", action.strip())
        if match:
            return match.group(1), match.group(2)
        return "", ""

    def _extract_finish(self, action: str) -> str:
        action = action.strip().strip("`")
        match = re.search(r"Finish\[(.*)\]", action, re.DOTALL)
        return match.group(1).strip() if match else action
