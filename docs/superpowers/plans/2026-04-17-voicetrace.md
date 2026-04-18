# VoiceTrace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local voice call observability dashboard that traces STT→Intent→Response→TTS pipeline failures and attributes root cause using side-by-side Deepgram vs Sarvam comparison.

**Architecture:** FastAPI backend runs two LangGraph pipelines in parallel (one per STT provider), stores traces in SQLite, and exposes results via HTTP to a Streamlit frontend that renders side-by-side trace cards with confidence badges, latency chips, TTS audio playback, and a root cause panel.

**Tech Stack:** Python 3.11+, FastAPI, LangGraph, Streamlit, OpenAI GPT-4o-mini, Deepgram SDK, Sarvam API (HTTP), SQLite, python-dotenv, httpx, asyncio

---

## File Map

```
/Users/rudranshtiwri/Code/voice observation/
├── api/
│   ├── __init__.py
│   ├── main.py          — FastAPI app, POST /compare, GET /trace/{id}
│   ├── graph.py         — LangGraph pipeline: build_graph(provider)
│   ├── nodes.py         — stt_deepgram_node, stt_sarvam_node, intent_node, response_node, tts_node, root_cause_node
│   ├── models.py        — Pydantic: NodeTrace, CallTrace, CompareResult, CallState (TypedDict)
│   ├── storage.py       — SQLite: init_db, save_trace, get_trace
│   └── demo_calls/
│       ├── call_01_code_mixed.mp3
│       ├── call_02_clear_english.mp3
│       └── call_03_noisy.mp3
├── app.py               — Streamlit UI
├── .env                 — DEEPGRAM_API_KEY, SARVAM_API_KEY, OPENAI_API_KEY
├── requirements.txt
└── docs/
    └── superpowers/
        ├── specs/2026-04-17-voicetrace-design.md
        └── plans/2026-04-17-voicetrace.md
```

---

## Task 1: Project scaffold + dependencies

**Files:**
- Create: `requirements.txt`
- Create: `.env`
- Create: `api/__init__.py`

- [ ] **Step 1: Create requirements.txt**

```
fastapi==0.115.0
uvicorn==0.30.6
streamlit==1.37.0
langchain==0.2.16
langgraph==0.2.28
langchain-openai==0.1.23
openai==1.45.0
deepgram-sdk==3.7.7
httpx==0.27.2
python-dotenv==1.0.1
pydantic==2.8.2
aiofiles==24.1.0
requests==2.32.3
```

- [ ] **Step 2: Install dependencies**

```bash
cd "/Users/rudranshtiwri/Code/voice observation"
pip install -r requirements.txt
```

Expected: no errors, all packages installed.

- [ ] **Step 3: Create .env**

```
DEEPGRAM_API_KEY=your_deepgram_key_here
SARVAM_API_KEY=your_sarvam_key_here
OPENAI_API_KEY=your_openai_key_here
```

Replace placeholder values with real keys.

- [ ] **Step 4: Create api/__init__.py**

```python
```

(empty file)

- [ ] **Step 5: Commit**

```bash
cd "/Users/rudranshtiwri/Code/voice observation"
git add requirements.txt .env api/__init__.py
git commit -m "chore: scaffold project, add dependencies"
```

---

## Task 2: Data models

**Files:**
- Create: `api/models.py`

- [ ] **Step 1: Write models.py**

```python
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
    nodes: list[NodeTrace]


class CompareResult(BaseModel):
    compare_id: str
    deepgram: CallTrace
    sarvam: CallTrace
```

- [ ] **Step 2: Verify models import cleanly**

```bash
cd "/Users/rudranshtiwri/Code/voice observation"
python -c "from api.models import CallState, CallTrace, CompareResult; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add api/models.py
git commit -m "feat: add data models (CallState, CallTrace, CompareResult)"
```

---

## Task 3: SQLite storage

**Files:**
- Create: `api/storage.py`

- [ ] **Step 1: Write storage.py**

```python
import sqlite3
import json
import uuid
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "traces.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS traces (
            id TEXT PRIMARY KEY,
            compare_id TEXT,
            provider TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            status TEXT,
            failed_node TEXT,
            raw_json TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_compare(compare_id: str, deepgram_trace: dict, sarvam_trace: dict):
    conn = sqlite3.connect(DB_PATH)
    for provider, trace in [("deepgram", deepgram_trace), ("sarvam", sarvam_trace)]:
        conn.execute(
            "INSERT INTO traces (id, compare_id, provider, status, failed_node, raw_json) VALUES (?,?,?,?,?,?)",
            (
                str(uuid.uuid4()),
                compare_id,
                provider,
                "error" if trace.get("failed_node") else "ok",
                trace.get("failed_node"),
                json.dumps(trace),
            ),
        )
    conn.commit()
    conn.close()


def get_compare(compare_id: str) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT raw_json FROM traces WHERE compare_id=?", (compare_id,)
    ).fetchall()
    conn.close()
    return [json.loads(r[0]) for r in rows]
```

- [ ] **Step 2: Verify storage works**

```bash
python -c "
from api.storage import init_db, save_compare, get_compare
import uuid
init_db()
cid = str(uuid.uuid4())
save_compare(cid, {'provider':'deepgram','failed_node':None}, {'provider':'sarvam','failed_node':'stt'})
rows = get_compare(cid)
print(len(rows), 'rows saved')
"
```

Expected: `2 rows saved`

- [ ] **Step 3: Commit**

```bash
git add api/storage.py
git commit -m "feat: add SQLite storage for traces"
```

---

## Task 4: LangGraph nodes

**Files:**
- Create: `api/nodes.py`

- [ ] **Step 1: Write nodes.py**

```python
import time
import base64
import os
import httpx
from deepgram import DeepgramClient, PrerecordedOptions
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

STT_WARN_MS = 400
STT_ERR_MS = 700
LLM_WARN_MS = 800
LLM_ERR_MS = 1500
TTS_WARN_MS = 500
TTS_ERR_MS = 900


def _latency_status(ms: int, warn: int, err: int) -> str:
    if ms >= err:
        return "error"
    if ms >= warn:
        return "warning"
    return "ok"


def stt_deepgram_node(state: dict) -> dict:
    t0 = time.time()
    try:
        client = DeepgramClient(DEEPGRAM_API_KEY)
        options = PrerecordedOptions(model="nova-2", language="en-IN", punctuate=True)
        response = client.listen.prerecorded.v("1").transcribe_file(
            {"buffer": state["audio_bytes"], "mimetype": "audio/mp3"}, options
        )
        alt = response.results.channels[0].alternatives[0]
        transcript = alt.transcript
        confidence = alt.confidence
        latency_ms = int((time.time() - t0) * 1000)
        state["transcript"] = transcript
        state["stt_confidence"] = round(confidence, 3)
        state["stt_latency_ms"] = latency_ms
        if confidence < 0.70:
            state["failed_node"] = "stt"
            state["error"] = f"STT_LOW_CONFIDENCE: {confidence:.2f} (threshold 0.70)"
    except Exception as e:
        state["transcript"] = ""
        state["stt_confidence"] = 0.0
        state["stt_latency_ms"] = int((time.time() - t0) * 1000)
        state["failed_node"] = "stt"
        state["error"] = f"STT_ERROR: {str(e)}"
    return state


def stt_sarvam_node(state: dict) -> dict:
    t0 = time.time()
    try:
        audio_b64 = base64.b64encode(state["audio_bytes"]).decode()
        resp = httpx.post(
            "https://api.sarvam.ai/speech-to-text",
            headers={"api-subscription-key": SARVAM_API_KEY},
            json={"audio": audio_b64, "language_code": "hi-IN", "model": "saarika:v2"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        transcript = data.get("transcript", "")
        confidence = data.get("confidence", 0.85)
        latency_ms = int((time.time() - t0) * 1000)
        state["transcript"] = transcript
        state["stt_confidence"] = round(float(confidence), 3)
        state["stt_latency_ms"] = latency_ms
        if float(confidence) < 0.70:
            state["failed_node"] = "stt"
            state["error"] = f"STT_LOW_CONFIDENCE: {confidence:.2f} (threshold 0.70)"
    except Exception as e:
        state["transcript"] = ""
        state["stt_confidence"] = 0.0
        state["stt_latency_ms"] = int((time.time() - t0) * 1000)
        state["failed_node"] = "stt"
        state["error"] = f"STT_ERROR: {str(e)}"
    return state


def intent_node(state: dict) -> dict:
    if state.get("failed_node"):
        return state
    t0 = time.time()
    try:
        messages = [
            SystemMessage(content=(
                "Classify the user intent from the transcript. "
                "Return JSON: {\"intent\": \"<INTENT_NAME>\", \"confidence\": <0.0-1.0>}. "
                "Intent options: PASSWORD_RESET, ACCOUNT_INQUIRY, COMPLAINT, BILLING, UNCLEAR."
            )),
            HumanMessage(content=state["transcript"]),
        ]
        result = llm.invoke(messages)
        import json, re
        raw = result.content.strip()
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        parsed = json.loads(match.group()) if match else {"intent": "UNCLEAR", "confidence": 0.3}
        intent = parsed.get("intent", "UNCLEAR")
        confidence = float(parsed.get("confidence", 0.3))
        latency_ms = int((time.time() - t0) * 1000)
        state["intent"] = intent
        state["intent_confidence"] = round(confidence, 3)
        state["intent_latency_ms"] = latency_ms
        if confidence < 0.60:
            state["failed_node"] = "intent"
            state["error"] = f"INTENT_UNCLEAR: confidence {confidence:.2f} (threshold 0.60)"
    except Exception as e:
        state["intent"] = "UNCLEAR"
        state["intent_confidence"] = 0.0
        state["intent_latency_ms"] = int((time.time() - t0) * 1000)
        state["failed_node"] = "intent"
        state["error"] = f"INTENT_ERROR: {str(e)}"
    return state


def response_node(state: dict) -> dict:
    if state.get("failed_node"):
        return state
    t0 = time.time()
    try:
        messages = [
            SystemMessage(content=(
                "You are a helpful customer support voice agent for an Indian bank. "
                "Respond concisely (1-2 sentences) to the user's request."
            )),
            HumanMessage(content=state["transcript"]),
        ]
        result = llm.invoke(messages)
        latency_ms = int((time.time() - t0) * 1000)
        state["llm_response"] = result.content.strip()
        state["response_latency_ms"] = latency_ms
    except Exception as e:
        state["llm_response"] = ""
        state["response_latency_ms"] = int((time.time() - t0) * 1000)
        state["failed_node"] = "response"
        state["error"] = f"RESPONSE_ERROR: {str(e)}"
    return state


def tts_node(state: dict) -> dict:
    if state.get("failed_node"):
        return state
    t0 = time.time()
    try:
        text = state.get("llm_response", "")
        resp = httpx.post(
            "https://api.sarvam.ai/text-to-speech",
            headers={"api-subscription-key": SARVAM_API_KEY},
            json={
                "inputs": [text],
                "target_language_code": "en-IN",
                "speaker": "anushka",
                "model": "bulbul:v1",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        audio_b64 = data.get("audios", [""])[0]
        latency_ms = int((time.time() - t0) * 1000)
        state["tts_audio_b64"] = audio_b64
        state["tts_latency_ms"] = latency_ms
    except Exception as e:
        state["tts_audio_b64"] = ""
        state["tts_latency_ms"] = int((time.time() - t0) * 1000)
        state["failed_node"] = "tts"
        state["error"] = f"TTS_ERROR: {str(e)}"
    return state


def root_cause_node(state: dict) -> dict:
    try:
        messages = [
            SystemMessage(content=(
                "You are a voice pipeline failure analyst. Given a partial pipeline trace, "
                "identify the root cause and provide one actionable recommendation. "
                "Return JSON: {\"root_cause\": \"<1 sentence>\", \"recommendation\": \"<1 sentence>\", "
                "\"cost_impact\": \"<estimated monthly saving if fixed>\"}"
            )),
            HumanMessage(content=(
                f"Provider: {state.get('provider')}\n"
                f"Failed node: {state.get('failed_node')}\n"
                f"Error: {state.get('error')}\n"
                f"Transcript: {state.get('transcript', 'N/A')}\n"
                f"STT confidence: {state.get('stt_confidence', 'N/A')}\n"
                f"Intent: {state.get('intent', 'N/A')}\n"
                f"Intent confidence: {state.get('intent_confidence', 'N/A')}"
            )),
        ]
        result = llm.invoke(messages)
        import json, re
        raw = result.content.strip()
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            state["root_cause"] = parsed.get("root_cause", "Unknown failure")
            state["recommendation"] = parsed.get("recommendation", "Investigate logs")
        else:
            state["root_cause"] = raw
            state["recommendation"] = "See root cause above"
    except Exception as e:
        state["root_cause"] = f"Root cause analysis failed: {str(e)}"
        state["recommendation"] = "Check API keys and connectivity"
    return state
```

- [ ] **Step 2: Verify nodes import cleanly**

```bash
python -c "from api.nodes import stt_deepgram_node, stt_sarvam_node, intent_node, response_node, tts_node, root_cause_node; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add api/nodes.py
git commit -m "feat: add LangGraph node functions (STT, intent, response, TTS, root cause)"
```

---

## Task 5: LangGraph pipeline

**Files:**
- Create: `api/graph.py`

- [ ] **Step 1: Write graph.py**

```python
from langgraph.graph import StateGraph, END
from api.nodes import (
    stt_deepgram_node,
    stt_sarvam_node,
    intent_node,
    response_node,
    tts_node,
    root_cause_node,
)
from api.models import CallState


def _should_continue(state: dict) -> str:
    """Route to root_cause if any node failed, else continue to next node."""
    return "root_cause" if state.get("failed_node") else "continue"


def build_graph(provider: str) -> StateGraph:
    """Build pipeline graph for given provider ('deepgram' or 'sarvam')."""
    stt_node = stt_deepgram_node if provider == "deepgram" else stt_sarvam_node

    builder = StateGraph(dict)

    builder.add_node("stt", stt_node)
    builder.add_node("intent", intent_node)
    builder.add_node("response", response_node)
    builder.add_node("tts", tts_node)
    builder.add_node("root_cause", root_cause_node)

    builder.set_entry_point("stt")

    builder.add_conditional_edges(
        "stt", _should_continue, {"root_cause": "root_cause", "continue": "intent"}
    )
    builder.add_conditional_edges(
        "intent", _should_continue, {"root_cause": "root_cause", "continue": "response"}
    )
    builder.add_conditional_edges(
        "response", _should_continue, {"root_cause": "root_cause", "continue": "tts"}
    )
    builder.add_conditional_edges(
        "tts", _should_continue, {"root_cause": "root_cause", "continue": END}
    )
    builder.add_edge("root_cause", END)

    return builder.compile()


def run_pipeline(audio_bytes: bytes, provider: str) -> dict:
    graph = build_graph(provider)
    initial_state = {
        "audio_bytes": audio_bytes,
        "provider": provider,
        "transcript": "",
        "stt_confidence": 0.0,
        "stt_latency_ms": 0,
        "intent": "",
        "intent_confidence": 0.0,
        "intent_latency_ms": 0,
        "llm_response": "",
        "response_latency_ms": 0,
        "tts_audio_b64": "",
        "tts_latency_ms": 0,
        "error": None,
        "failed_node": None,
        "root_cause": None,
        "recommendation": None,
    }
    result = graph.invoke(initial_state)
    return result
```

- [ ] **Step 2: Verify graph builds**

```bash
python -c "from api.graph import build_graph; g = build_graph('deepgram'); print('Graph OK')"
```

Expected: `Graph OK`

- [ ] **Step 3: Commit**

```bash
git add api/graph.py
git commit -m "feat: add LangGraph pipeline with conditional failure routing"
```

---

## Task 6: FastAPI backend

**Files:**
- Create: `api/main.py`

- [ ] **Step 1: Write main.py**

```python
import asyncio
import uuid
import base64
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from api.graph import run_pipeline
from api.models import CallTrace, CompareResult, NodeTrace
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


@app.get("/trace/{compare_id}")
def get_trace(compare_id: str):
    rows = get_compare(compare_id)
    if not rows:
        raise HTTPException(status_code=404, detail="Trace not found")
    return rows
```

- [ ] **Step 2: Start server and verify health**

```bash
cd "/Users/rudranshtiwri/Code/voice observation"
uvicorn api.main:app --reload --port 8000 &
sleep 3
curl http://localhost:8000/docs
```

Expected: HTML page (FastAPI Swagger UI)

- [ ] **Step 3: Commit**

```bash
git add api/main.py
git commit -m "feat: add FastAPI backend with /compare and /trace endpoints"
```

---

## Task 7: Demo audio files

**Files:**
- Create: `api/demo_calls/` (download or generate 3 MP3 files)

- [ ] **Step 1: Generate demo audio using Sarvam TTS**

Run this script to generate the 3 demo calls:

```python
# run as: python generate_demo_calls.py
import httpx, base64, os
from dotenv import load_dotenv
load_dotenv()

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

calls = [
    ("call_01_code_mixed", "Mera account reset karo please"),
    ("call_02_clear_english", "I need to reset my account password"),
    ("call_03_noisy", "asdkfj asldfkj reset kar do yaar please help"),
]

os.makedirs("api/demo_calls", exist_ok=True)

for name, text in calls:
    resp = httpx.post(
        "https://api.sarvam.ai/text-to-speech",
        headers={"api-subscription-key": SARVAM_API_KEY},
        json={"inputs": [text], "target_language_code": "hi-IN", "speaker": "anushka", "model": "bulbul:v1"},
        timeout=30,
    )
    resp.raise_for_status()
    audio_b64 = resp.json()["audios"][0]
    audio_bytes = base64.b64decode(audio_b64)
    with open(f"api/demo_calls/{name}.wav", "wb") as f:
        f.write(audio_bytes)
    print(f"Generated {name}.wav")
```

```bash
cd "/Users/rudranshtiwri/Code/voice observation"
python generate_demo_calls.py
ls api/demo_calls/
```

Expected: 3 `.wav` files listed.

- [ ] **Step 2: Commit**

```bash
git add api/demo_calls/
git commit -m "feat: add demo audio files for 3 call scenarios"
```

---

## Task 8: Streamlit UI

**Files:**
- Create: `app.py`

- [ ] **Step 1: Write app.py**

```python
import streamlit as st
import requests
import base64
import os
from pathlib import Path

API_URL = "http://localhost:8000"
DEMO_DIR = Path("api/demo_calls")

st.set_page_config(page_title="VoiceTrace", layout="wide", page_icon="🔊")
st.title("🔊 VoiceTrace — Voice Pipeline Observability")
st.caption("Real-time STT → Intent → Response → TTS failure attribution")

# ── Sidebar: input ──────────────────────────────────────────────────────────
st.sidebar.header("Call Input")
input_mode = st.sidebar.radio("Source", ["Demo Call", "Upload Audio"])

audio_bytes = None
if input_mode == "Demo Call":
    demo_files = list(DEMO_DIR.glob("*.wav")) + list(DEMO_DIR.glob("*.mp3"))
    demo_names = {f.name: f for f in demo_files}
    if demo_names:
        selected = st.sidebar.selectbox("Pick a demo call", list(demo_names.keys()))
        audio_bytes = demo_names[selected].read_bytes()
        st.sidebar.audio(audio_bytes)
    else:
        st.sidebar.warning("No demo files found in api/demo_calls/")
else:
    uploaded = st.sidebar.file_uploader("Upload audio", type=["mp3", "wav"])
    if uploaded:
        audio_bytes = uploaded.read()
        st.sidebar.audio(audio_bytes)

analyze = st.sidebar.button("🔍 Analyze Call", disabled=audio_bytes is None, type="primary")

# ── Helpers ──────────────────────────────────────────────────────────────────

def confidence_badge(val: float, threshold: float) -> str:
    icon = "✅" if val >= threshold else "❌"
    return f"{icon} {val:.2f}"

def latency_chip(ms: int, warn: int, err: int) -> str:
    if ms >= err:
        return f"🔴 {ms}ms"
    if ms >= warn:
        return f"🟡 {ms}ms"
    return f"🟢 {ms}ms"

def render_trace(trace: dict, label: str):
    failed = trace.get("failed_node")
    status_icon = "❌" if failed else "✅"
    st.subheader(f"{status_icon} {label}")

    # STT card
    stt_color = "red" if failed == "stt" else "green" if trace.get("stt_confidence", 0) >= 0.70 else "orange"
    with st.container(border=True):
        st.markdown(f"**STT ({label})**")
        st.markdown(f"**Transcript:** `{trace.get('transcript', 'N/A')}`")
        col1, col2 = st.columns(2)
        col1.metric("Confidence", confidence_badge(trace.get("stt_confidence", 0), 0.70))
        col2.metric("Latency", latency_chip(trace.get("stt_latency_ms", 0), 400, 700))
        if failed == "stt":
            st.error(f"⚠️ {trace.get('error', 'STT failed')}")

    if not failed or failed != "stt":
        # Intent card
        with st.container(border=True):
            st.markdown("**Intent (GPT-4o-mini)**")
            col1, col2 = st.columns(2)
            col1.metric("Intent", trace.get("intent", "N/A"))
            col2.metric("Confidence", confidence_badge(trace.get("intent_confidence", 0), 0.60))
            st.metric("Latency", latency_chip(trace.get("intent_latency_ms", 0), 800, 1500))
            if failed == "intent":
                st.error(f"⚠️ {trace.get('error', 'Intent failed')}")

    if trace.get("llm_response"):
        with st.container(border=True):
            st.markdown("**LLM Response (GPT-4o-mini)**")
            st.info(trace.get("llm_response"))
            st.metric("Latency", latency_chip(trace.get("response_latency_ms", 0), 800, 1500))

    if trace.get("tts_audio_b64"):
        with st.container(border=True):
            st.markdown("**TTS (Sarvam Bulbul)**")
            audio_data = base64.b64decode(trace["tts_audio_b64"])
            st.audio(audio_data, format="audio/wav")
            st.metric("Latency", latency_chip(trace.get("tts_latency_ms", 0), 500, 900))

# ── Main: results ────────────────────────────────────────────────────────────

if analyze and audio_bytes:
    with st.spinner("Running both pipelines in parallel..."):
        resp = requests.post(
            f"{API_URL}/compare",
            files={"file": ("audio.wav", audio_bytes, "audio/wav")},
            timeout=60,
        )
    if resp.status_code != 200:
        st.error(f"API error: {resp.text}")
    else:
        result = resp.json()
        st.success(f"Analysis complete — Compare ID: `{result['compare_id']}`")

        col_dg, col_sv = st.columns(2)
        with col_dg:
            render_trace(result["deepgram"], "Deepgram Nova-2 (en-IN)")
        with col_sv:
            render_trace(result["sarvam"], "Sarvam Saarika (hi-en)")

        # Root cause panel
        deepgram_failed = result["deepgram"].get("failed_node")
        sarvam_failed = result["sarvam"].get("failed_node")

        st.divider()
        st.subheader("🔎 Root Cause Analysis")

        if deepgram_failed and not sarvam_failed:
            st.error(f"**Deepgram failed at node:** `{deepgram_failed}`")
            st.markdown(f"**Root cause:** {result['deepgram'].get('root_cause', 'N/A')}")
            st.markdown(f"**Recommendation:** {result['deepgram'].get('recommendation', 'N/A')}")
            st.success("Sarvam pipeline succeeded — switching STT provider would resolve this.")
        elif sarvam_failed and not deepgram_failed:
            st.error(f"**Sarvam failed at node:** `{sarvam_failed}`")
            st.markdown(f"**Root cause:** {result['sarvam'].get('root_cause', 'N/A')}")
            st.success("Deepgram pipeline succeeded.")
        elif deepgram_failed and sarvam_failed:
            st.error("Both pipelines failed.")
            st.markdown(f"**Deepgram root cause:** {result['deepgram'].get('root_cause', 'N/A')}")
            st.markdown(f"**Sarvam root cause:** {result['sarvam'].get('root_cause', 'N/A')}")
        else:
            st.success("✅ Both pipelines completed successfully — no failures detected.")
```

- [ ] **Step 2: Run Streamlit**

```bash
cd "/Users/rudranshtiwri/Code/voice observation"
streamlit run app.py
```

Expected: Browser opens at `http://localhost:8501`, sidebar shows "Demo Call" picker.

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add Streamlit UI with side-by-side trace comparison and root cause panel"
```

---

## Task 9: End-to-end smoke test

- [ ] **Step 1: Ensure both services running**

Terminal 1:
```bash
cd "/Users/rudranshtiwri/Code/voice observation"
uvicorn api.main:app --reload --port 8000
```

Terminal 2:
```bash
cd "/Users/rudranshtiwri/Code/voice observation"
streamlit run app.py
```

- [ ] **Step 2: Run demo call_01 via curl**

```bash
curl -X POST http://localhost:8000/compare \
  -F "file=@api/demo_calls/call_01_code_mixed.wav" \
  | python -m json.tool | head -40
```

Expected: JSON with `compare_id`, `deepgram`, `sarvam` keys. At least one of `failed_node`, `root_cause` populated.

- [ ] **Step 3: Run through Streamlit UI**

1. Open `http://localhost:8501`
2. Sidebar → Demo Call → `call_01_code_mixed.wav` → Analyze
3. Verify: side-by-side columns render, confidence badges show, root cause panel appears
4. Repeat for `call_02_clear_english.wav` → both should show ✅
5. Repeat for `call_03_noisy.wav` → both should show ❌ with root cause

- [ ] **Step 4: Final commit**

```bash
git add .
git commit -m "chore: verify end-to-end pipeline, demo ready"
```

---

## Running Locally (Quick Reference)

```bash
# Terminal 1 — backend
cd "/Users/rudranshtiwri/Code/voice observation"
uvicorn api.main:app --reload --port 8000

# Terminal 2 — frontend
cd "/Users/rudranshtiwri/Code/voice observation"
streamlit run app.py
```

Open: `http://localhost:8501`
