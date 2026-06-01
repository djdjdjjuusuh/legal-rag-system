"""配置管理 - 借鉴 hello-agents 的 Config 设计"""
import os
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class Config(BaseModel):
    """全局配置"""

    # LLM 配置
    model: str = Field(default_factory=lambda: os.getenv("LLM_MODEL", "deepseek-chat"))
    api_key: str = Field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    base_url: str = Field(default_factory=lambda: os.getenv("LLM_BASE_URL", "https://api.deepseek.com"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.3"))
    max_tokens: Optional[int] = int(os.getenv("MAX_TOKENS", "4096"))

    # Agent 配置
    max_react_steps: int = int(os.getenv("MAX_REACT_STEPS", "8"))
    max_reflection_iterations: int = int(os.getenv("MAX_REFLECTION_ITERATIONS", "3"))

    # 记忆配置
    max_working_memory: int = int(os.getenv("MAX_WORKING_MEMORY", "50"))
    max_short_term_memory: int = int(os.getenv("MAX_SHORT_TERM_MEMORY", "200"))

    # RAG 配置
    rag_top_k: int = int(os.getenv("RAG_TOP_K", "6"))
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")

    # 系统配置
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


_config: Optional[Config] = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config
