# VoiceTrace — Design Spec
Date: 2026-04-17

## Overview
Real-time voice call observability dashboard. Traces STT→Intent→Response→TTS pipeline failures and attributes root cause. Demo tool for interview showing SuperBryn-style cross-modal failure attribution.

## Stack
- **Frontend:** Streamlit (`app.py`)
- **Backend:** FastAPI (`api/main.py`)
- **Pipeline:** LangGraph (STT→Intent→Response→TTS nodes)
- **STT:** Deepgram Nova-2 (en-IN) + Sarvam Saarika (hi-en) — run in parallel
- **LLM:** OpenAI GPT-4o-mini (intent classification, response generation, root cause analysis)
- **TTS:** Sarvam Bulbul v3
- **Storage:** SQLite (`traces.db`)
- **Runtime:** Local only (`localhost`)

## Architecture

```
Streamlit UI
    │ HTTP
FastAPI
    ├── LangGraph (Deepgram path)
    │     [STT:Deepgram] → [Intent] → [Response] → [TTS:Sarvam]
    └── LangGraph (Sarvam path)
          [STT:Sarvam]   → [Intent] → [Response] → [TTS:Sarvam]
```

Both graphs run in parallel on `/compare`. Results stored in SQLite. Streamlit renders side-by-side.

## File Structure

```
api/
  main.py        — FastAPI endpoints: POST /compare, GET /trace/{id}
  graph.py       — LangGraph pipeline definition
  nodes.py       — Node functions: stt_deepgram, stt_sarvam, intent, response, tts, root_cause
  models.py      — Pydantic: CallTrace, NodeTrace, CompareResult
  storage.py     — SQLite helpers
  demo_calls/    — 3 pre-loaded MP3 files
app.py           — Streamlit UI
.env             — API keys (DEEPGRAM_API_KEY, SARVAM_API_KEY, OPENAI_API_KEY)
```

## LangGraph State

```python
class CallState(TypedDict):
    audio_bytes: bytes
    provider: str                # "deepgram" | "sarvam"
    transcript: str
    stt_confidence: float
    intent: str
    intent_confidence: float
    llm_response: str
    tts_audio: bytes
    trace: list[NodeTrace]
    error: str | None
    failed_node: str | None      # "stt"|"intent"|"response"|"tts"
    root_cause: str | None
    recommendation: str | None
```

## Node Failure Handling
- Each node: try/except → sets `failed_node` + `error` on exception
- Conditional edge: `failed_node` set → skip to `root_cause_node`
- `root_cause_node`: GPT-4o-mini gets partial trace → human-readable attribution + recommendation
- Both provider paths independent — one failure doesn't block the other

## Thresholds

| Signal | Green | Yellow | Red |
|--------|-------|--------|-----|
| STT latency | <400ms | 400-700ms | >700ms |
| LLM latency | <800ms | 800-1500ms | >1500ms |
| TTS latency | <500ms | 500-900ms | >900ms |
| STT confidence | ≥0.70 | — | <0.70 → STT_LOW_CONFIDENCE |
| Intent confidence | ≥0.60 | — | <0.60 → INTENT_UNCLEAR |

## Demo Calls
- `call_01_code_mixed.mp3` — Hindi-English mix, Deepgram fails, Sarvam succeeds
- `call_02_clear_english.mp3` — Both providers succeed (happy path)
- `call_03_noisy.mp3` — Both STT nodes fail, full root cause shown

## API Endpoints
- `POST /compare` — accepts audio file, runs both pipelines in parallel, stores trace, returns `CompareResult`
- `GET /trace/{id}` — fetch stored trace by ID

## Streamlit UI Layout
- Sidebar: demo call picker OR file upload
- Main: two columns (Deepgram | Sarvam)
  - Per column: turn cards with transcript, confidence badge, intent, LLM response, TTS audio player, latency chips
  - Failed node: red highlight + error message
- Bottom: Root Cause panel (attribution + recommendation + cost impact estimate)

## Out of Scope
- Multi-turn conversation (single audio file = single turn for demo)
- Auth, rate limiting, deployment
- Tests
