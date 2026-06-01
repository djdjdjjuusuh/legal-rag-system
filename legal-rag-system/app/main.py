"""法律RAG多智能体协作系统 - FastAPI 入口"""
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.api.routes import router

app = FastAPI(
    title="法律RAG多智能体协作系统",
    description="基于DeepSeek的法律RAG智能助手，支持ReAct/Plan-Solve/Reflection三种推理范式，"
                "具备三层记忆、自定义性格、检索增强生成等能力。",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

# 静态文件
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def root():
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {
        "service": "法律RAG多智能体协作系统",
        "version": "2.0.0",
        "docs": "/docs",
    }
