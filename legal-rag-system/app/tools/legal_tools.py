"""法律场景专用工具集"""
from datetime import datetime, timedelta, timezone
import urllib.request
import urllib.error
import json
from app.tools.registry import Tool, FunctionTool, ToolRegistry
from app.rag.knowledge_base import LegalKnowledgeBase

_kb: LegalKnowledgeBase = None

def _get_kb() -> LegalKnowledgeBase:
    global _kb
    if _kb is None:
        _kb = LegalKnowledgeBase()
    return _kb


# ============== 法律检索工具 ==============

def _search_statutes(input: str = "", **kwargs) -> str:
    results = _get_kb().search(input, doc_type="statutes", top_k=10)
    if not results:
        return f"[法规检索] 未找到与'{input}'直接相关的法规，请尝试调整关键词。"
    lines = [f"**法规检索结果** ({len(results)}条):"]
    for i, r in enumerate(results):
        lines.append(f"\n### {i+1}. {r.get('title', '无标题')}")
        lines.append(r.get('content', '')[:1500])
        lines.append(f"*相关度: {r.get('score', 'N/A')}*")
    return "\n".join(lines)


def _search_cases(input: str = "", **kwargs) -> str:
    results = _get_kb().search(input, doc_type="cases", top_k=10)
    if not results:
        return f"[案例检索] 未找到与'{input}'直接相关的案例，请尝试调整关键词。"
    lines = [f"**案例检索结果** ({len(results)}条):"]
    for i, r in enumerate(results):
        lines.append(f"\n### {i+1}. {r.get('title', '无标题')}")
        lines.append(r.get('content', '')[:1500])
        lines.append(f"*相关度: {r.get('score', 'N/A')}*")
    return "\n".join(lines)


def _check_validity(input: str = "", **kwargs) -> str:
    results = _get_kb().search(input, doc_type="statutes", top_k=5)
    if not results:
        return f"[时效性校验] 未在知识库中找到'{input}'，请确认法规名称是否准确。"
    info = []
    for r in results:
        title = r.get('title', '未知')
        content = r.get('content', '')[:800]
        info.append(f"- {title}: {content}")
    return f"**时效性校验 '{input}'**:\n" + "\n".join(info) + "\n\n请交叉核实官方数据库中的现行有效状态。"


def _related_laws(input: str = "", **kwargs) -> str:
    results = _get_kb().search(input, doc_type=None, top_k=10)
    if not results:
        return f"[关联法条] 未在知识库中找到与'{input}'相关的内容。"
    lines = [f"**关联法条检索 '{input}'** ({len(results)}条):"]
    for i, r in enumerate(results):
        doc_type = r.get('doc_type', '')
        label = {"statutes": "法规", "cases": "案例", "practical": "实务"}.get(doc_type, '资料')
        lines.append(f"\n### [{label}] {r.get('title', '无标题')}")
        lines.append(r.get('content', '')[:1000])
    return "\n".join(lines)


def _search_practical(input: str = "", **kwargs) -> str:
    results = _get_kb().search(input, doc_type="practical", top_k=10)
    if not results:
        return f"[实务检索] 未找到与'{input}'直接相关的实务资料，请尝试调整关键词。"
    lines = [f"**实务检索结果** ({len(results)}条):"]
    for i, r in enumerate(results):
        lines.append(f"\n### {i+1}. {r.get('title', '无标题')}")
        lines.append(r.get('content', '')[:1500])
        lines.append(f"*相关度: {r.get('score', 'N/A')}*")
    return "\n".join(lines)


def _search_all(input: str = "", **kwargs) -> str:
    results = _get_kb().search(input, doc_type=None, top_k=15)
    if not results:
        return f"[综合检索] 未在知识库中找到与'{input}'相关的内容。"
    lines = [f"**综合检索结果** ({len(results)}条):"]
    for i, r in enumerate(results):
        doc_type = r.get('doc_type', '')
        label = {"statutes": "法规", "cases": "案例", "practical": "实务"}.get(doc_type, '资料')
        lines.append(f"\n### [{label}] {r.get('title', '无标题')}")
        lines.append(r.get('content', '')[:1000])
    return "\n".join(lines)


# ============== 法律计算工具 ==============

def _litigation_fee(amount: str, case_type: str = "财产案件", **kwargs) -> str:
    """诉讼费计算器 - 基于《诉讼费用交纳办法》"""
    try:
        amt = float(amount)
    except ValueError:
        return f"无法识别金额: {amount}，请输入数字（单位：元）"

    if case_type == "财产案件":
        if amt <= 10000:
            fee = 50
        elif amt <= 100000:
            fee = amt * 0.025 - 200
        elif amt <= 200000:
            fee = amt * 0.02 + 300
        elif amt <= 500000:
            fee = amt * 0.015 + 1300
        elif amt <= 1000000:
            fee = amt * 0.01 + 3800
        elif amt <= 2000000:
            fee = amt * 0.009 + 4800
        elif amt <= 5000000:
            fee = amt * 0.008 + 6800
        elif amt <= 10000000:
            fee = amt * 0.007 + 11800
        elif amt <= 20000000:
            fee = amt * 0.006 + 21800
        else:
            fee = amt * 0.005 + 41800
        fee = max(50, fee)
    elif case_type == "离婚案件":
        fee = 50 if amt <= 200000 else amt * 0.005
    elif case_type == "劳动争议":
        fee = 10
    else:
        fee = 50

    return (
        f"根据《诉讼费用交纳办法》，{case_type}标的额 {amt:,.0f} 元，"
        f"预估案件受理费约 {fee:,.0f} 元。\n"
        f"注意：此为简易估算，实际费用以法院通知为准。"
        f"财产保全费、执行费等另计。"
    )


def _statute_limitation(case_type: str, **kwargs) -> str:
    """诉讼时效计算器"""
    limitations = {
        "普通民事": ("3年", "《民法典》第188条"),
        "劳动争议仲裁": ("1年", "《劳动争议调解仲裁法》第27条"),
        "国际货物买卖合同": ("4年", "《民法典》第594条"),
        "环境损害赔偿": ("3年", "《环境保护法》第66条"),
        "行政起诉": ("6个月", "《行政诉讼法》第46条"),
        "刑事": ("视罪名而定(追诉时效)", "《刑法》第87-89条"),
    }
    info = limitations.get(case_type, limitations["普通民事"])
    return f"{case_type}的诉讼时效为 **{info[0]}**，依据：{info[1]}。请以权利被侵害之日或知道权利被侵害之日起算。"


# ============== 时间和天气工具 ==============

def _get_time_weather(input: str = "", **kwargs) -> str:
    """获取当前日期时间和当地天气信息"""
    tz = timezone(timedelta(hours=8))
    now = datetime.now(tz)
    weekday_names = ["一", "二", "三", "四", "五", "六", "日"]
    weekday = weekday_names[now.weekday()]
    time_str = now.strftime(f"%Y年%m月%d日 周{weekday} %H:%M")

    # 通过 wttr.in 获取天气（免费，无需 API key）
    weather_str = "天气信息暂不可用"
    try:
        req = urllib.request.Request(
            "https://wttr.in?format=%l+%C+%t+%h+%w&lang=zh",
            headers={"User-Agent": "curl/7.0"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            weather_data = resp.read().decode("utf-8").strip()
            if weather_data:
                # Parse: "Sunny +22°C 65% 15km/h"
                weather_str = weather_data
    except Exception:
        pass

    return f"**当前时间**: {time_str}\n**天气**: {weather_str}"


# ============== 构建工具注册表 ==============

def create_legal_tool_registry(use_mcp: bool = True) -> ToolRegistry:
    registry = ToolRegistry()

    # 本地工具（始终注册）
    registry.register(FunctionTool(
        "search_statutes", "检索相关法律法规，输入法条名称或关键词",
        _search_statutes,
    ))
    registry.register(FunctionTool(
        "search_cases", "检索相关案例，输入案由或关键词",
        _search_cases,
    ))
    registry.register(FunctionTool(
        "check_validity", "检查法规时效性，输入法规名称，确认是否现行有效",
        _check_validity,
    ))
    registry.register(FunctionTool(
        "related_laws", "查找与给定法规相关的上位法、下位法、特别法",
        _related_laws,
    ))
    registry.register(FunctionTool(
        "search_practical", "检索实务知识库，输入关键词查询实务指南、操作指引、合同问答等",
        _search_practical,
    ))
    registry.register(FunctionTool(
        "search_all", "跨库综合检索全部知识库（法规+案例+实务），返回所有类型的匹配结果",
        _search_all,
    ))
    registry.register(FunctionTool(
        "calc_litigation_fee", "诉讼费计算器，输入: 金额(元) case_type=财产案件|离婚案件|劳动争议",
        _litigation_fee,
    ))
    registry.register(FunctionTool(
        "statute_limitation", "诉讼时效查询，输入案件类型如: 普通民事/劳动争议仲裁/行政起诉",
        _statute_limitation,
    ))
    registry.register(FunctionTool(
        "get_time_weather", "获取当前日期时间和当地天气信息，无需参数",
        _get_time_weather,
    ))

    # MCP 远程工具（从 MCP 服务端动态发现）
    if use_mcp:
        try:
            from app.tools.mcp_client import get_mcp_manager
            mcp_mgr = get_mcp_manager()
            for mcp_tool in mcp_mgr.get_mcp_tools():
                registry.register(mcp_tool)
            if mcp_mgr.list_all_tools():
                print(f"[tools] 已注册 {len(mcp_mgr.get_mcp_tools())} 个 MCP 远程工具")
        except Exception as e:
            print(f"[tools] MCP 工具加载跳过: {e}")

    return registry
