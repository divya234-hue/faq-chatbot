from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=500)


class ChatResponse(BaseModel):
    answer: str
    confidence: float
    matched_question: Optional[str] = None
    category: Optional[str] = None
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
