from typing import TypedDict, Optional
from pydantic import BaseModel


# LangGraph state — mutable dict passed through graph nodes
class CallState(TypedDict):
    audio_bytes: bytes
    provider: str           # "deepgram" | "sarvam"
    transcript: str
    stt_confidence: float
    stt_latency_ms: int
    intent: str
    intent_confidence: float
    intent_latency_ms: int
    llm_response: str
    response_latency_ms: int
    tts_audio_b64: str      # base64-encoded audio bytes
    tts_latency_ms: int
    error: Optional[str]
    failed_node: Optional[str]   # "stt"|"intent"|"response"|"tts"
    root_cause: Optional[str]
    recommendation: Optional[str]
    suggested_prompt: Optional[str]
    intent_system_prompt: Optional[str]
    response_system_prompt: Optional[str]


# Pydantic response models
class NodeTrace(BaseModel):
    node: str
    latency_ms: int
    status: str             # "ok" | "warning" | "error"
    data: dict


class CallTrace(BaseModel):
    call_id: str
    provider: str
    transcript: str
    stt_confidence: float
    stt_latency_ms: int
    intent: str
    intent_confidence: float
    intent_latency_ms: int
    llm_response: str
    response_latency_ms: int
    tts_audio_b64: str
    tts_latency_ms: int
    failed_node: Optional[str]
    error: Optional[str]
    root_cause: Optional[str]
    recommendation: Optional[str]
    suggested_prompt: Optional[str] = None
    nodes: list[NodeTrace]


class CompareResult(BaseModel):
    compare_id: str
    deepgram: CallTrace
    sarvam: CallTrace


class EvaluateResult(BaseModel):
    variant_a: CallTrace
    variant_b: CallTrace
