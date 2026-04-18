# Thought Process — VoiceTrace v2

## The Problem I Was Actually Solving

Voice AI pipelines fail silently and nobody knows where.

You have audio going in and a spoken response coming out. When it breaks — wrong answer, confused agent, dropped call — you have no idea if the failure was the microphone, the transcription, the intent model, the LLM, or the TTS. All you see is a bad outcome. The pipeline is a black box end-to-end.

What made this worse: the failures are *cross-modal*. A noisy audio file corrupts the transcript. A corrupted transcript confuses intent classification. A wrong intent generates a hallucinated response. A bad response gets spoken back to the customer. Four stages, four different failure modes, one visible symptom. Traditional logging tells you the response was bad. It doesn't tell you which domino fell first.

The real question I wanted to answer: **can you instrument a voice pipeline the same way engineers instrument distributed systems — with traces, node-level attribution, and root cause analysis?**

---

## How I Thought About the Solution

I started from the failure modes, not the architecture.

If I list every way a voice call can go wrong:
1. Audio is noisy → STT returns garbage
2. STT returns a transcript but in the wrong language or with wrong words
3. Intent classification picks the wrong bucket
4. LLM generates a generic or hallucinated response
5. TTS produces unintelligible speech

Each of these is owned by a different system: a different vendor, a different model, a different prompt. The fix for each is completely different. So the first design decision was: **treat the pipeline as a graph of discrete nodes**, not a monolith. That's where LangGraph came in — it gave me a state machine where each node can fail independently and route differently.

The second insight was about comparison. A single pipeline trace tells you what happened. A *dual* pipeline trace tells you what should have happened. If Deepgram fails on a Hindi-English mixed audio but Sarvam succeeds, you've just proven the failure is at the STT layer with that specific provider — not the audio, not the intent model, not the LLM. The side-by-side comparison is the core diagnostic tool.

The third layer was AI-driven attribution. Once you know *which* node failed, you still need to know *why* and *what to do*. That's the root cause node — it uses GPT to look at the partial trace and generate a human-readable explanation and recommendation. Then self-correct goes one step further: if the failure is in a prompt-driven node (intent or response), generate an improved prompt automatically.

So the architecture emerged from the failure modes:

```
Failure modes → Node-based pipeline → Dual provider comparison → AI attribution → Self-correction
```

---

## The Why

**Why LangGraph over a simple sequential function call?**

I could have written `stt() → intent() → response() → tts()` as four function calls in a row. But LangGraph gives me conditional routing for free — if a node fails, the graph jumps to root cause analysis instead of continuing downstream and generating more garbage. It also makes the pipeline visual and extensible. Adding a new node (say, a guardrails node) is a graph edge, not a refactor.

**Why two STT providers specifically?**

Deepgram and Sarvam represent the two real choices for Indian voice AI: a global provider strong in English vs. a local provider built for Indic languages. The demo calls are designed around this — code-mixed Hindi-English is where the gap shows up most clearly. This isn't arbitrary; it mirrors the actual decision a product team building for India would face.

**Why Streamlit for the frontend?**

Speed. This is a demo and observability tool, not a consumer product. Streamlit lets me build side-by-side trace rendering, audio playback, and text areas for prompt editing in 200 lines. The right tool for the job.

**Why store traces in SQLite?**

Same reason. For a demo at this scale, SQLite is zero-config persistence. Every comparison gets stored with its full JSON trace so you can retrieve it later by ID. That's all that's needed right now.

---

## Alternatives I Could Have Taken

**Alternative 1: Just log everything to a file and analyze offline.**

This is what most teams do. It fails because by the time you're analyzing logs, you've lost the interactive feedback loop. You can't A/B test a prompt change and see the result in 30 seconds. The value of VoiceTrace is the tight iteration loop — change a prompt, hit run, see the diff.

**Alternative 2: Use an existing observability platform (LangSmith, Langfuse, Helicone).**

These tools are great for LLM traces but they don't understand voice pipelines. They can tell you the GPT call took 1.2 seconds, but they can't tell you the STT confidence was 0.43 and that's why GPT got confused. You'd need to instrument every layer manually and stitch traces together yourself. VoiceTrace is purpose-built for the cross-modal attribution problem.

**Alternative 3: Build a proper async event streaming architecture from day one.**

Kafka for events, ClickHouse for analytics, Grafana for dashboards. This is the right answer at scale but would take weeks to build and is overkill for a demo. The discipline here was to get the *product idea* right before the *infrastructure* right.

**Alternative 4: Single pipeline with switchable providers.**

Run one pipeline at a time, switch the STT provider via a dropdown. Simpler, but you lose the ability to compare results on the same audio simultaneously. The side-by-side view is the core insight — you need both results at the same time to attribute the failure.

---

## How This Works at Scale

At production scale, VoiceTrace needs a different infrastructure stack. The core ideas stay the same; the implementation changes.

**Ingestion layer:**
- Real calls come in as audio streams, not uploaded WAVs
- A Kafka topic per call, audio chunks published as they arrive
- Streaming STT (Deepgram/Sarvam streaming APIs) instead of batch transcription
- Latency goes from seconds to milliseconds

**Pipeline execution:**
- Replace LangGraph in-process with a distributed worker pool (Celery or Ray)
- Each node runs as an independent worker, scaling independently
- STT workers can scale to handle 1000 concurrent calls; intent workers can scale separately
- State passed through Redis instead of in-memory Python dicts

**Storage:**
- Replace SQLite with ClickHouse for analytical queries (which provider fails most on Monday mornings? which intent has lowest confidence across all calls this week?)
- Raw audio stored in S3/GCS with presigned URLs in the trace record
- Retain full traces for 30 days, aggregated metrics forever

**Observability of the observability tool:**
- Prometheus metrics on node latency p50/p95/p99
- Alerts when STT confidence drops below threshold across more than 5% of calls in a 5-minute window
- PagerDuty integration for production failures

**Prompt management:**
- Variant A/B testing moves from a manual text area to a versioned prompt registry
- Experiments tracked with statistical significance gates (don't ship a prompt change until you have 1000 call samples)
- Automated rollback if a new prompt version increases intent failure rate by more than 2%

**Multi-tenancy:**
- Each bank/product team gets their own pipeline config, their own intent taxonomy, their own thresholds
- Trace data isolated per tenant

---

## Next Two Realistic Versions

### V3 — Live Call Monitoring + Alerting

Right now VoiceTrace is a post-hoc analysis tool. You upload a call, you see what happened. V3 makes it real-time.

**What changes:**
- WebSocket connection from the dashboard to the backend
- Pipeline nodes emit events as they complete, not just at the end
- The UI updates live: STT confidence badge appears in 3 seconds, intent badge in 5 seconds, response in 7 seconds
- Alert panel: if STT confidence drops below threshold on 3 consecutive calls, fire a Slack/PagerDuty alert
- "Live call feed" view showing all active calls with their current pipeline stage

**Why this is realistic:** The LangGraph state machine already emits per-node results. It's a streaming output problem, not an architecture change. FastAPI supports WebSockets natively. The main work is the real-time UI and alert rules.

**Business value:** Ops teams can catch a degraded STT provider within minutes instead of discovering it in the next day's QA review.

---

### V4 — Prompt Optimization Engine

Right now self-correct generates one suggested prompt and you manually apply it. V4 automates the full prompt optimization loop.

**What changes:**
- Define a target metric: intent accuracy rate, response CSAT score, average handle time reduction
- Run a population of prompt variants automatically (evolutionary search over prompt space)
- Each variant tested against a representative sample of historical calls
- Variants ranked by metric, statistical significance computed
- Winning variant automatically staged for human review before promotion to production
- Full experiment history with rollback capability

**Why this is realistic:** The /evaluate endpoint already runs two variants in parallel. The jump to N variants is a loop around the same endpoint. The hard part is defining the scoring function — which is a product decision, not an engineering one.

**Business value:** Prompt tuning goes from a manual engineering task (days) to an automated experiment (hours). The system continuously improves without human intervention.

---

## What I'd Do Differently

If I started over, I'd separate the trace storage from the pipeline execution earlier. Right now `save_compare()` is called inside the `/compare` handler — the API response waits for the database write. At scale this should be fire-and-forget: return the response immediately, write to storage asynchronously.

I'd also design the intent taxonomy as configuration, not hardcoded in the prompt string. Right now `PASSWORD_RESET, ACCOUNT_INQUIRY, COMPLAINT, BILLING, UNCLEAR` lives in a Python constant. It should be a per-deployment config that the intent node reads at runtime, making it easy to swap taxonomies without touching code.

The self-correct feature also needs a feedback loop that doesn't exist yet. Right now it suggests a prompt and you can apply it, but there's no way to know if the suggested prompt actually performed better. The feedback signal — did applying the suggestion reduce failures? — needs to be captured and fed back into the system.
