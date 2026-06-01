"""MCP 法律工具服务端 — 通过 Model Context Protocol (SSE 传输) 暴露法律工具

启动方式：
    python -m app.mcp.legal_mcp_server [--port 8765]

这里只暴露本地工具不具备的独有工具，避免重复。
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from mcp.server.fastmcp import FastMCP
from app.rag.knowledge_base import LegalKnowledgeBase

mcp = FastMCP("法律工具服务-MCP扩展")

_kb = LegalKnowledgeBase()


@mcp.tool()
def search_practical(query: str) -> str:
    """检索实务知识库。输入关键词，返回实务指南、操作指引、合同问答等。

    Args:
        query: 检索关键词，如"合同审查要点"或"仲裁申请书模板"
    """
    results = _kb.search(query, doc_type="practical", top_k=5)
    if not results:
        return f"未找到与'{query}'直接相关的实务资料。"
    lines = [f"**实务检索结果** ({len(results)}条):"]
    for i, r in enumerate(results):
        lines.append(f"\n### {i+1}. {r.get('title', '无标题')}")
        lines.append(r.get('content', '')[:1500])
    return "\n".join(lines)


@mcp.tool()
def search_all(query: str) -> str:
    """跨库检索全部知识库（法规+案例+实务）。输入关键词，返回所有类型的匹配结果。

    Args:
        query: 检索关键词
    """
    results = _kb.search(query, doc_type=None, top_k=10)
    if not results:
        return f"未在知识库中找到与'{query}'相关的内容。"
    lines = [f"**综合检索结果** ({len(results)}条):"]
    for i, r in enumerate(results):
        doc_type = r.get('doc_type', '')
        label = {"statutes": "法规", "cases": "案例", "practical": "实务"}.get(doc_type, '资料')
        lines.append(f"\n### [{label}] {r.get('title', '无标题')}")
        lines.append(r.get('content', '')[:1000])
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    import uvicorn
    parser = argparse.ArgumentParser(description="MCP 法律工具服务端 (SSE)")
    parser.add_argument("--port", type=int, default=8765, help="监听端口 (默认 8765)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="监听地址 (默认 127.0.0.1)")
    args = parser.parse_args()
    app = mcp.sse_app()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
