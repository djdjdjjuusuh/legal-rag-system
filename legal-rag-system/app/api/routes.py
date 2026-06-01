"""FastAPI 路由"""
import json
import io
import time
from typing import List
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from app.models.schemas import (
    ChatRequest, ChatResponse, PersonaConfig,
    KnowledgeDoc, BatchKnowledgeRequest, MemoryRecord,
    ConversationSummary, ConversationDetail,
)
from app.agents.orchestrator import AgentOrchestrator
from app.rag.knowledge_base import LegalKnowledgeBase
from app.memory.manager import MemoryManager
from app.memory.conversation_store import ConversationStore

router = APIRouter()
orchestrator = AgentOrchestrator()
knowledge_base = LegalKnowledgeBase()
conv_store = ConversationStore()


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """法律咨询流式接口 - SSE"""
    async def event_generator():
        try:
            for event in orchestrator.chat_stream(
                query=request.query,
                persona=request.persona,
                reasoning_mode=request.reasoning_mode,
                conversation_id=request.conversation_id,
                metadata=request.metadata,
            ):
                event_type = event["event"]
                data_str = json.dumps(event["data"], ensure_ascii=False)
                yield f"event: {event_type}\ndata: {data_str}\n\n"
        except Exception as e:
            error_data = json.dumps({"message": str(e)}, ensure_ascii=False)
            yield f"event: error\ndata: {error_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """法律咨询接口 - 支持三种推理范式切换"""
    try:
        result = orchestrator.chat(
            query=request.query,
            persona=request.persona,
            reasoning_mode=request.reasoning_mode,
            conversation_id=request.conversation_id,
            metadata=request.metadata,
        )
        return ChatResponse(
            answer=result["answer"],
            reasoning_mode=result["reasoning_mode"],
            confidence=result["confidence"],
            citations=result["citations"],
            conversation_id=result["conversation_id"],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理请求出错: {str(e)}")


@router.post("/knowledge/add")
async def add_knowledge(doc: KnowledgeDoc):
    """向知识库添加文档"""
    try:
        doc_id = knowledge_base.add_document(
            doc_type=doc.doc_type,
            title=doc.title,
            content=doc.content,
            metadata=doc.metadata,
        )
        return {"status": "ok", "doc_id": doc_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/knowledge/batch")
async def add_knowledge_batch(request: BatchKnowledgeRequest):
    """批量添加文档到知识库"""
    knowledge_base.add_batch(request.doc_type, request.documents)
    return {"status": "ok", "count": len(request.documents)}


@router.get("/knowledge/search")
async def search_knowledge(q: str, doc_type: str = None, top_k: int = 5):
    """检索知识库"""
    results = knowledge_base.search(q, doc_type=doc_type, top_k=top_k)
    return {"results": results}


@router.get("/knowledge/stats")
async def knowledge_stats():
    """知识库统计"""
    return knowledge_base.get_collection_stats()


@router.get("/knowledge/list")
async def list_knowledge(doc_type: str = None):
    """列出知识库文档"""
    return {"documents": knowledge_base.list_documents(doc_type=doc_type)}


@router.delete("/knowledge/{doc_type}/{doc_id}")
async def delete_knowledge(doc_type: str, doc_id: str):
    """删除知识库文档"""
    knowledge_base.delete_document(doc_type, doc_id)
    return {"status": "deleted"}


@router.post("/knowledge/upload")
async def upload_file(
    file: UploadFile = File(...),
    doc_type: str = Form("statutes"),
):
    """上传文件到知识库，自动解析 docx/txt/md 格式"""
    filename = file.filename or "unknown"
    content_bytes = await file.read()

    # Parse based on file extension
    if filename.lower().endswith(".docx"):
        try:
            from docx import Document
            doc = Document(io.BytesIO(content_bytes))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            text = "\n".join(paragraphs)
            if not text:
                raise HTTPException(status_code=400, detail="docx 文件中未提取到文本内容")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"解析 docx 文件失败: {str(e)}")
    elif filename.lower().endswith((".txt", ".md", ".json")):
        text = content_bytes.decode("utf-8")
    elif filename.lower().endswith(".pdf"):
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(io.BytesIO(content_bytes))
            pages = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    pages.append(page_text)
            text = "\n".join(pages)
            if not text.strip():
                raise HTTPException(status_code=400, detail="PDF 文件中未提取到文本内容（可能是扫描件或图片PDF）")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"解析 PDF 文件失败: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail=f"不支持的文件格式: {filename}，支持 .docx / .txt / .md / .json")

    doc_id = knowledge_base.add_document(
        doc_type=doc_type,
        title=filename,
        content=text,
        metadata={"filename": filename, "size": len(content_bytes), "char_count": len(text), "created_at": time.time()},
    )
    return {
        "status": "ok",
        "doc_id": doc_id,
        "title": filename,
        "char_count": len(text),
    }


# ---- 会话管理 ----

@router.get("/conversations", response_model=List[ConversationSummary])
async def list_conversations():
    """列出所有会话"""
    return conv_store.list_all()


@router.get("/conversations/{conv_id}", response_model=ConversationDetail)
async def get_conversation(conv_id: str):
    """获取会话详情（含消息）"""
    conv = conv_store.get(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="会话不存在")
    return conv


@router.delete("/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    """删除会话"""
    conv_store.delete(conv_id)
    return {"status": "deleted"}


# ---- 记忆 ----

@router.get("/memory/{conversation_id}")
async def get_memory(conversation_id: str):
    """获取会话记忆"""
    memory = MemoryManager(conversation_id)
    return {
        "working_messages": len(memory.working.get_messages()),
        "short_term": memory.short_term.get_all(),
    }


@router.post("/memory/{conversation_id}")
async def set_memory(conversation_id: str, record: MemoryRecord):
    """设置短期记忆"""
    memory = MemoryManager(conversation_id)
    memory.short_term.set(record.key, record.value)
    return {"status": "ok"}


@router.get("/now")
async def get_now(city: str = ""):
    """获取当前时间与天气，可选指定城市"""
    from datetime import datetime, timezone, timedelta
    import urllib.request
    import urllib.error

    tz = timezone(timedelta(hours=8))
    now = datetime.now(tz)
    weekday_names = ["一", "二", "三", "四", "五", "六", "日"]
    weekday = weekday_names[now.weekday()]
    hour = now.hour

    if hour < 6:
        greeting = "夜深了"
    elif hour < 9:
        greeting = "早上好"
    elif hour < 12:
        greeting = "上午好"
    elif hour < 14:
        greeting = "中午好"
    elif hour < 18:
        greeting = "下午好"
    else:
        greeting = "晚上好"

    # 天气：默认天津，也可传 city 参数指定
    import urllib.parse
    location = city.strip() if city.strip() else "天津"
    weather = "天气暂不可用"
    try:
        # 先通过 geocoding API 获取坐标
        encoded = urllib.parse.quote(location)
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={encoded}&count=1&language=zh"
        geo_req = urllib.request.Request(geo_url, headers={"User-Agent": "LegalRAG/1.0"})
        with urllib.request.urlopen(geo_req, timeout=5) as resp:
            import json
            geo_data = json.loads(resp.read().decode("utf-8"))
            results = geo_data.get("results", [])
            if results:
                lat = results[0]["latitude"]
                lon = results[0]["longitude"]
                name = results[0].get("name", location)
                # Open-Meteo 免费天气 API（无需 Key）
                weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,weather_code&timezone=Asia/Shanghai"
                w_req = urllib.request.Request(weather_url, headers={"User-Agent": "LegalRAG/1.0"})
                with urllib.request.urlopen(w_req, timeout=5) as w_resp:
                    w_data = json.loads(w_resp.read().decode("utf-8"))
                    cur = w_data.get("current", {})
                    temp = cur.get("temperature_2m", "?")
                    code = cur.get("weather_code", 0)
                    # WMO weather codes mapping
                    weather_desc = {
                        0: "晴朗", 1: "大部晴", 2: "多云", 3: "阴",
                        45: "雾", 48: "霜雾", 51: "小毛毛雨", 53: "毛毛雨", 55: "大毛毛雨",
                        61: "小雨", 63: "中雨", 65: "大雨", 71: "小雪", 73: "中雪", 75: "大雪",
                        80: "阵雨", 81: "中阵雨", 82: "大阵雨", 85: "小阵雪", 86: "大阵雪",
                        95: "雷暴", 96: "冰雹雷暴", 99: "大冰雹雷暴"
                    }.get(code, f"天气码{code}")
                    weather = f"{name}: {weather_desc} {temp}°C"
    except Exception:
        pass

    return {
        "greeting": greeting,
        "date": now.strftime("%Y年%m月%d日"),
        "weekday": f"周{weekday}",
        "time": now.strftime("%H:%M:%S"),
        "weather": weather,
    }


@router.get("/tools")
async def list_tools():
    """列出所有注册工具（本地+MCP）及其启用状态"""
    tools = orchestrator.tool_registry.list_tools_with_status()
    return {"tools": tools}


@router.post("/tools/toggle")
async def toggle_tool(data: dict):
    """启用/禁用指定工具  {name: str, enabled: bool}"""
    name = data.get("name", "")
    enabled = data.get("enabled", True)
    orchestrator.tool_registry.set_enabled(name, enabled)
    return {"status": "ok", "name": name, "enabled": enabled}


@router.get("/health")
async def health():
    return {"status": "healthy", "service": "法律RAG多智能体系统"}
