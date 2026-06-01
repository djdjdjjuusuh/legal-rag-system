"""API 请求/响应模型"""
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class PersonaConfig(BaseModel):
    """可自定义的性格配置"""
    role: str = Field(default="法律顾问", description="法律顾问/法务助理/学者型研究员/模拟对方律师")
    formality: str = Field(default="正式严谨", description="正式严谨/半正式亲和/简明扼要")
    verbosity: str = Field(default="标准篇幅", description="详细解释/标准篇幅/只给结论")
    tone: str = Field(default="冷静中立", description="冷静中立/坚定有力/温和耐心")
    language: str = Field(default="中文", description="中文/英文/中英双语")
    address_user: str = Field(default="您", description="您/你/当事人/客户")
    conclusion_first: bool = Field(default=True, description="是否结论先行")


class ChatRequest(BaseModel):
    query: str = Field(..., description="用户问题")
    persona: Optional[PersonaConfig] = Field(default=None, description="性格配置")
    reasoning_mode: str = Field(default="react", description="推理范式: react / plan_solve / reflection")
    conversation_id: Optional[str] = Field(default=None, description="会话ID, 用于记忆关联")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="额外元数据")


class SourceCitation(BaseModel):
    """法律引用来源"""
    source_type: str  # 法规/案例/实务
    title: str
    content: str
    relevance_score: float
    article_reference: Optional[str] = None  # 具体条款编号，如"第一千零八十四条"


class ChatResponse(BaseModel):
    answer: str
    reasoning_mode: str
    confidence: Optional[float] = None
    citations: List[SourceCitation] = Field(default_factory=list)
    disclaimer: str = "以上内容仅供参考，不构成正式法律意见。具体法律问题请咨询执业律师。"
    conversation_id: Optional[str] = None


class KnowledgeDoc(BaseModel):
    """知识库文档"""
    id: Optional[str] = None
    doc_type: str = Field(..., description="statute/case/practical")
    title: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BatchKnowledgeRequest(BaseModel):
    """批量知识库文档"""
    doc_type: str = Field(..., description="statutes/cases/practical")
    documents: List[Dict[str, Any]] = Field(..., description="文档列表")


class ConversationSummary(BaseModel):
    """会话摘要"""
    id: str
    title: str
    created_at: str = ""
    updated_at: str = ""


class ConversationDetail(BaseModel):
    """会话详情"""
    id: str
    title: str
    created_at: str = ""
    updated_at: str = ""
    messages: List[Dict[str, Any]] = Field(default_factory=list)


class MemoryRecord(BaseModel):
    """记忆记录"""
    key: str
    value: Any
    memory_type: str = "short_term"  # working/short_term/long_term
    timestamp: Optional[str] = None
