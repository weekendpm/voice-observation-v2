import asyncio
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from api.graph import run_pipeline, evaluate_pipeline
from api.models import CallTrace, CompareResult, NodeTrace, EvaluateResult
from api.storage import init_db, save_compare, get_compare
from dotenv import load_dotenv

load_dotenv()
init_db()

app = FastAPI(title="VoiceTrace API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _state_to_trace(state: dict) -> CallTrace:
    nodes = []

    stt_status = "error" if state.get("failed_node") == "stt" else (
        "warning" if state.get("stt_confidence", 1.0) < 0.70 else "ok"
    )
    nodes.append(NodeTrace(
        node="stt",
        latency_ms=state.get("stt_latency_ms", 0),
        status=stt_status,
        data={"transcript": state.get("transcript", ""), "confidence": state.get("stt_confidence", 0)},
    ))

    if state.get("intent"):
        intent_status = "error" if state.get("failed_node") == "intent" else (
            "warning" if state.get("intent_confidence", 1.0) < 0.60 else "ok"
        )
        nodes.append(NodeTrace(
            node="intent",
            latency_ms=state.get("intent_latency_ms", 0),
            status=intent_status,
            data={"intent": state.get("intent", ""), "confidence": state.get("intent_confidence", 0)},
        ))

    if state.get("llm_response"):
        nodes.append(NodeTrace(
            node="response",
            latency_ms=state.get("response_latency_ms", 0),
            status="error" if state.get("failed_node") == "response" else "ok",
            data={"response": state.get("llm_response", "")},
        ))

    if state.get("tts_audio_b64"):
        nodes.append(NodeTrace(
            node="tts",
            latency_ms=state.get("tts_latency_ms", 0),
            status="error" if state.get("failed_node") == "tts" else "ok",
            data={"audio_length_b64": len(state.get("tts_audio_b64", ""))},
        ))

    return CallTrace(
        call_id=str(uuid.uuid4()),
        provider=state.get("provider", ""),
        transcript=state.get("transcript", ""),
        stt_confidence=state.get("stt_confidence", 0.0),
        stt_latency_ms=state.get("stt_latency_ms", 0),
        intent=state.get("intent", ""),
        intent_confidence=state.get("intent_confidence", 0.0),
        intent_latency_ms=state.get("intent_latency_ms", 0),
        llm_response=state.get("llm_response", ""),
        response_latency_ms=state.get("response_latency_ms", 0),
        tts_audio_b64=state.get("tts_audio_b64", ""),
        tts_latency_ms=state.get("tts_latency_ms", 0),
        failed_node=state.get("failed_node"),
        error=state.get("error"),
        root_cause=state.get("root_cause"),
        recommendation=state.get("recommendation"),
        suggested_prompt=state.get("suggested_prompt"),
        nodes=nodes,
    )


@app.post("/compare", response_model=CompareResult)
async def compare(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    compare_id = str(uuid.uuid4())

    deepgram_state, sarvam_state = await asyncio.gather(
        asyncio.to_thread(run_pipeline, audio_bytes, "deepgram"),
        asyncio.to_thread(run_pipeline, audio_bytes, "sarvam"),
    )

    deepgram_trace = _state_to_trace(deepgram_state)
    sarvam_trace = _state_to_trace(sarvam_state)

    save_compare(compare_id, deepgram_trace.model_dump(), sarvam_trace.model_dump())

    return CompareResult(
        compare_id=compare_id,
        deepgram=deepgram_trace,
        sarvam=sarvam_trace,
    )


@app.post("/evaluate", response_model=EvaluateResult)
async def evaluate(
    file: UploadFile = File(...),
    prompt_a: str = Form(...),
    prompt_b: str = Form(...),
    target_node: str = Form("intent"),
):
    audio_bytes = await file.read()
    if target_node not in ("intent", "response", "both"):
        raise HTTPException(status_code=422, detail=f"target_node must be 'intent', 'response', or 'both'. Got: {target_node!r}")

    def _build_config(prompt: str) -> dict:
        config = {}
        if target_node in ("intent", "both"):
            config["intent_system_prompt"] = prompt
        if target_node in ("response", "both"):
            config["response_system_prompt"] = prompt
        return config

    state_a, state_b = await asyncio.gather(
        asyncio.to_thread(evaluate_pipeline, audio_bytes, _build_config(prompt_a)),
        asyncio.to_thread(evaluate_pipeline, audio_bytes, _build_config(prompt_b)),
    )

    return EvaluateResult(
        variant_a=_state_to_trace(state_a),
        variant_b=_state_to_trace(state_b),
    )


@app.get("/trace/{compare_id}")
def get_trace(compare_id: str):
    rows = get_compare(compare_id)
    if not rows:
        raise HTTPException(status_code=404, detail="Trace not found")
    return rows
