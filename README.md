# VoiceTrace

Voice pipeline observability dashboard. Traces STT → Intent → LLM → TTS failures and attributes root causes using AI.

![Python](https://img.shields.io/badge/python-3.10+-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green) ![Streamlit](https://img.shields.io/badge/Streamlit-1.37-red)

---

## What It Does

Voice AI pipelines fail silently. A bad call could break at the microphone, the transcription, the intent model, the LLM, or the TTS — and all you see is a wrong answer.

VoiceTrace instruments each stage as a discrete node, runs dual provider pipelines side-by-side, and uses GPT to attribute failures to the exact layer that broke — then auto-suggests a fixed prompt.

**Two core features:**

- **Compare Providers** — same audio through Deepgram and Sarvam STT simultaneously. See which provider fails and why.
- **Evaluate Variants** — same audio, two different system prompts. A/B test intent or response prompts and get AI-generated fixes when a variant fails.

---

## Pipeline

```
Audio → STT (Deepgram / Sarvam) → Intent Classification → LLM Response → TTS
                                                     ↘ any failure → Root Cause Analysis + Self-Correct
```

---

## Stack

| Layer | Tech |
|-------|------|
| Frontend | Streamlit |
| Backend | FastAPI |
| Pipeline | LangGraph |
| STT | Deepgram Nova-2, Sarvam Saarika v2.5 |
| LLM | GPT-4o-mini |
| TTS | Sarvam Bulbul v3 |
| Storage | SQLite |

---

## Setup

**1. Clone and install**
```bash
git clone https://github.com/your-username/voicetrace.git
cd voicetrace
pip install -r requirements.txt
```

**2. Configure API keys**
```bash
cp .env.example .env
```

Edit `.env`:
```
DEEPGRAM_API_KEY=your_deepgram_api_key
OPENAI_API_KEY=your_openai_api_key
SARVAM_API_KEY=your_sarvam_api_key
```

Get keys from:
- Deepgram: https://console.deepgram.com
- OpenAI: https://platform.openai.com/api-keys
- Sarvam: https://dashboard.sarvam.ai

**3. Run**

Terminal 1 — Backend:
```bash
uvicorn api.main:app --reload --port 8000
```

Terminal 2 — Frontend:
```bash
streamlit run app.py
```

Open `http://localhost:8501`

---

## Demo Audio

Four pre-generated WAV files in `api/demo_calls/`:

| File | Scenario |
|------|----------|
| `call_01_code_mixed.wav` | Hindi-English mixed — Deepgram fails, Sarvam succeeds |
| `call_02_clear_english.wav` | Clear English — both providers succeed |
| `call_03_noisy.wav` | Noisy audio — both providers fail at STT |
| `call_04_english_technical.wav` | Technical English — tests robustness |

---

## Failure Thresholds

| Node | Failure condition |
|------|------------------|
| STT | Confidence < 0.70 |
| Intent | Confidence < 0.60 |

---

## API

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/compare` | POST | Run dual STT provider comparison |
| `/evaluate` | POST | A/B test two prompt variants |
| `/trace/{compare_id}` | GET | Retrieve stored trace by ID |

---

## Project Structure

```
├── api/
│   ├── main.py          # FastAPI endpoints
│   ├── graph.py         # LangGraph pipeline
│   ├── nodes.py         # STT, intent, response, TTS, root cause nodes
│   ├── models.py        # Pydantic models
│   ├── storage.py       # SQLite helpers
│   └── demo_calls/      # Sample audio files
├── app.py               # Streamlit frontend
├── generate_demo_calls.py
├── requirements.txt
└── thoughtprocess.md    # Design decisions and roadmap
```
