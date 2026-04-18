# Evaluate Variants + Self-Correct Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add prompt A/B evaluation and GPT-powered self-correct suggestion to VoiceTrace, with zero changes to existing `/compare` flow.

**Architecture:** New `/evaluate` endpoint runs two Deepgram pipelines in parallel with different system prompts injected via state. `root_cause_node` is extended to generate a `suggested_prompt` when a node fails. Streamlit gains a second tab wrapping the new endpoint.

**Tech Stack:** FastAPI, LangGraph, LangChain (ChatOpenAI), Streamlit, Python 3.10+

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `api/nodes.py` | Modify | Read prompt overrides from state in `intent_node` + `response_node`; extend `root_cause_node` to produce `suggested_prompt` |
| `api/graph.py` | Modify | Add `evaluate_pipeline(audio_bytes, prompt_config)` wrapper |
| `api/models.py` | Modify | Add `EvaluateResult` Pydantic model |
| `api/main.py` | Modify | Add `POST /evaluate` endpoint |
| `app.py` | Modify | Wrap existing UI in `tab1`; add `tab2` with evaluate UI |

---

### Task 1: Add prompt override support to `intent_node` and `response_node`

**Files:**
- Modify: `api/nodes.py`

**Context:** Both nodes currently use hardcoded system prompt strings. We need them to read from `state` first, falling back to the hardcoded default. This is purely additive — existing behaviour unchanged when the key is absent.

- [ ] **Step 1: Open `api/nodes.py` and locate `intent_node`**

Find the `SystemMessage(content=(...))` string starting at line ~99. The current hardcoded string is:
```
"Classify the user intent from the transcript. "
"Return JSON: {\"intent\": \"<INTENT_NAME>\", \"confidence\": <0.0-1.0>}. "
"Intent options: PASSWORD_RESET, ACCOUNT_INQUIRY, COMPLAINT, BILLING, UNCLEAR."
```

- [ ] **Step 2: Add a module-level constant for the default intent prompt**

At the top of `api/nodes.py`, after the latency threshold constants, add:

```python
DEFAULT_INTENT_PROMPT = (
    "Classify the user intent from the transcript. "
    "Return JSON: {\"intent\": \"<INTENT_NAME>\", \"confidence\": <0.0-1.0>}. "
    "Intent options: PASSWORD_RESET, ACCOUNT_INQUIRY, COMPLAINT, BILLING, UNCLEAR."
)

DEFAULT_RESPONSE_PROMPT = (
    "You are a helpful customer support voice agent for an Indian bank. "
    "Respond concisely (1-2 sentences) to the user's request."
)
```

- [ ] **Step 3: Update `intent_node` to read override from state**

Replace the hardcoded `SystemMessage` content inside `intent_node`:

```python
def intent_node(state: dict) -> dict:
    if state.get("failed_node"):
        return state
    t0 = time.time()
    try:
        system_prompt = state.get("intent_system_prompt") or DEFAULT_INTENT_PROMPT
        messages = [
            SystemMessage(content=system_prompt),
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
```

- [ ] **Step 4: Update `response_node` to read override from state**

Replace the hardcoded `SystemMessage` content inside `response_node`:

```python
def response_node(state: dict) -> dict:
    if state.get("failed_node"):
        return state
    t0 = time.time()
    try:
        system_prompt = state.get("response_system_prompt") or DEFAULT_RESPONSE_PROMPT
        messages = [
            SystemMessage(content=system_prompt),
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
```

- [ ] **Step 5: Manually verify existing pipeline still works**

Start the API and hit `/compare` with a demo call:
```bash
cd "voice observation v2"
uvicorn api.main:app --reload &
curl -s -X POST http://localhost:8000/compare \
  -F "file=@api/demo_calls/call_02_clear_english.wav" | python3 -m json.tool | head -40
```
Expected: JSON with `deepgram` and `sarvam` keys, `failed_node: null` for clear english call.

- [ ] **Step 6: Commit**

```bash
git add api/nodes.py
git commit -m "feat: add prompt override support to intent_node and response_node"
```

---

### Task 2: Extend `root_cause_node` to generate `suggested_prompt`

**Files:**
- Modify: `api/nodes.py`

**Context:** After the existing root cause analysis, fire a second GPT call to rewrite the failing node's system prompt. Store result in `state["suggested_prompt"]`. Fail silently if GPT call errors.

- [ ] **Step 1: Add `suggested_prompt` generation at end of `root_cause_node`**

In `api/nodes.py`, replace the entire `root_cause_node` function:

```python
def root_cause_node(state: dict) -> dict:
    import json, re
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

    # Self-correct: generate suggested prompt for failed intent or response node
    state["suggested_prompt"] = None
    failed = state.get("failed_node")
    if failed in ("intent", "response"):
        current_prompt = (
            state.get("intent_system_prompt") or DEFAULT_INTENT_PROMPT
            if failed == "intent"
            else state.get("response_system_prompt") or DEFAULT_RESPONSE_PROMPT
        )
        try:
            fix_messages = [
                SystemMessage(content=(
                    f"You are a prompt engineer for a voice AI pipeline. "
                    f"Rewrite the system prompt for the '{failed}' node to fix the failure described. "
                    f"Return JSON: {{\"suggested_prompt\": \"<improved system prompt>\"}}"
                )),
                HumanMessage(content=(
                    f"Current prompt:\n{current_prompt}\n\n"
                    f"Failure: {state.get('error', 'Unknown')}\n"
                    f"Transcript: {state.get('transcript', 'N/A')}"
                )),
            ]
            fix_result = llm.invoke(fix_messages)
            fix_raw = fix_result.content.strip()
            fix_match = re.search(r'\{.*\}', fix_raw, re.DOTALL)
            if fix_match:
                fix_parsed = json.loads(fix_match.group())
                state["suggested_prompt"] = fix_parsed.get("suggested_prompt")
        except Exception:
            pass  # fail silently — suggested_prompt stays None

    return state
```

- [ ] **Step 2: Commit**

```bash
git add api/nodes.py
git commit -m "feat: extend root_cause_node to generate suggested_prompt for self-correct"
```

---

### Task 3: Add `EvaluateResult` model and `evaluate_pipeline` function

**Files:**
- Modify: `api/models.py`
- Modify: `api/graph.py`

- [ ] **Step 1: Add `EvaluateResult` to `api/models.py`**

Open `api/models.py` and append at the end:

```python
class EvaluateResult(BaseModel):
    variant_a: CallTrace
    variant_b: CallTrace
```

- [ ] **Step 2: Add `evaluate_pipeline` to `api/graph.py`**

Open `api/graph.py`. After the existing `run_pipeline` function, add:

```python
def evaluate_pipeline(audio_bytes: bytes, prompt_config: dict) -> dict:
    """Run Deepgram pipeline with custom prompt overrides injected into initial state."""
    graph = build_graph("deepgram")
    initial_state = {
        "audio_bytes": audio_bytes,
        "provider": "deepgram",
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
        "suggested_prompt": None,
        # inject overrides
        "intent_system_prompt": prompt_config.get("intent_system_prompt"),
        "response_system_prompt": prompt_config.get("response_system_prompt"),
    }
    return graph.invoke(initial_state)
```

- [ ] **Step 3: Commit**

```bash
git add api/models.py api/graph.py
git commit -m "feat: add EvaluateResult model and evaluate_pipeline function"
```

---

### Task 4: Add `POST /evaluate` endpoint

**Files:**
- Modify: `api/main.py`

**Context:** Accepts audio + two prompt strings + target_node as multipart form fields. Runs both variants in parallel using `evaluate_pipeline`. Returns `EvaluateResult`.

- [ ] **Step 1: Add import for `EvaluateResult` and `evaluate_pipeline` in `api/main.py`**

Find the existing imports at the top of `api/main.py`:

```python
from api.graph import run_pipeline
from api.models import CallTrace, CompareResult, NodeTrace
```

Replace with:

```python
from api.graph import run_pipeline, evaluate_pipeline
from api.models import CallTrace, CompareResult, NodeTrace, EvaluateResult
```

Also add `Form` to the FastAPI imports line:

```python
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
```

- [ ] **Step 2: Add `/evaluate` endpoint after the existing `/compare` endpoint**

Append to `api/main.py`:

```python
@app.post("/evaluate", response_model=EvaluateResult)
async def evaluate(
    file: UploadFile = File(...),
    prompt_a: str = Form(...),
    prompt_b: str = Form(...),
    target_node: str = Form("intent"),
):
    audio_bytes = await file.read()

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
```

- [ ] **Step 3: Verify endpoint responds**

```bash
curl -s -X POST http://localhost:8000/evaluate \
  -F "file=@api/demo_calls/call_02_clear_english.wav" \
  -F "prompt_a=Classify the user intent. Return JSON: {\"intent\": \"<NAME>\", \"confidence\": <0-1>}. Options: PASSWORD_RESET, ACCOUNT_INQUIRY, COMPLAINT, BILLING, UNCLEAR." \
  -F "prompt_b=What is the user asking about? Return JSON: {\"intent\": \"<NAME>\", \"confidence\": <0-1>}. Pick from: PASSWORD_RESET, ACCOUNT_INQUIRY, COMPLAINT, BILLING, UNCLEAR." \
  -F "target_node=intent" | python3 -m json.tool | head -50
```

Expected: JSON with `variant_a` and `variant_b` keys, both containing `CallTrace` objects.

- [ ] **Step 4: Commit**

```bash
git add api/main.py
git commit -m "feat: add POST /evaluate endpoint for prompt variant A/B testing"
```

---

### Task 5: Update Streamlit UI — tabs + Evaluate Variants tab

**Files:**
- Modify: `app.py`

**Context:** Wrap existing UI in `tab1` with zero changes. Add `tab2` with prompt text areas, Run button, side-by-side trace results, and Self-Correct panel.

- [ ] **Step 1: Add tab wrapper and session state defaults at top of main section in `app.py`**

Find the line `# ── Main: results` in `app.py`. Insert just before it:

```python
# ── Session state defaults ───────────────────────────────────────────────────
DEFAULT_INTENT_PROMPT = (
    "Classify the user intent from the transcript. "
    "Return JSON: {\"intent\": \"<INTENT_NAME>\", \"confidence\": <0.0-1.0>}. "
    "Intent options: PASSWORD_RESET, ACCOUNT_INQUIRY, COMPLAINT, BILLING, UNCLEAR."
)
DEFAULT_RESPONSE_PROMPT = (
    "You are a helpful customer support voice agent for an Indian bank. "
    "Respond concisely (1-2 sentences) to the user's request."
)

if "prompt_a" not in st.session_state:
    st.session_state["prompt_a"] = DEFAULT_INTENT_PROMPT
if "prompt_b" not in st.session_state:
    st.session_state["prompt_b"] = ""
if "eval_result" not in st.session_state:
    st.session_state["eval_result"] = None
if "suggested_prompt" not in st.session_state:
    st.session_state["suggested_prompt"] = None
```

- [ ] **Step 2: Wrap existing "analyze" block in tab1, add tab2**

Find this block near the bottom of `app.py`:

```python
if analyze and audio_bytes:
```

Replace from that line to the end of the file with:

```python
tab1, tab2 = st.tabs(["Compare Providers", "Evaluate Variants"])

# ── Tab 1: existing compare flow ─────────────────────────────────────────────
with tab1:
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

# ── Tab 2: evaluate variants ──────────────────────────────────────────────────
with tab2:
    st.markdown("### Prompt Variant A/B Evaluation")
    st.caption("Same audio, two different prompts — compare outputs side by side.")

    target_node = st.radio(
        "Target node",
        ["intent", "response", "both"],
        horizontal=True,
    )

    col_pa, col_pb = st.columns(2)
    with col_pa:
        st.markdown("**Variant A prompt** (default)")
        prompt_a = st.text_area(
            "Variant A",
            value=st.session_state["prompt_a"],
            height=160,
            label_visibility="collapsed",
            key="input_prompt_a",
        )
    with col_pb:
        st.markdown("**Variant B prompt** (your experiment)")
        prompt_b = st.text_area(
            "Variant B",
            value=st.session_state["prompt_b"],
            height=160,
            label_visibility="collapsed",
            key="input_prompt_b",
        )

    run_eval = st.button(
        "🔬 Run Evaluation",
        disabled=audio_bytes is None or not prompt_a or not prompt_b,
        type="primary",
    )

    if run_eval and audio_bytes:
        with st.spinner("Running variant pipelines in parallel..."):
            eval_resp = requests.post(
                f"{API_URL}/evaluate",
                files={"file": ("audio.wav", audio_bytes, "audio/wav")},
                data={
                    "prompt_a": prompt_a,
                    "prompt_b": prompt_b,
                    "target_node": target_node,
                },
                timeout=60,
            )
        if eval_resp.status_code != 200:
            st.error(f"API error: {eval_resp.text}")
        else:
            eval_result = eval_resp.json()
            st.session_state["eval_result"] = eval_result
            # capture suggested_prompt from whichever variant failed
            suggested = (
                eval_result["variant_a"].get("suggested_prompt")
                or eval_result["variant_b"].get("suggested_prompt")
            )
            st.session_state["suggested_prompt"] = suggested
            st.session_state["prompt_a"] = prompt_a
            st.session_state["prompt_b"] = prompt_b

    if st.session_state["eval_result"]:
        eval_result = st.session_state["eval_result"]
        col_a, col_b = st.columns(2)
        with col_a:
            render_trace(eval_result["variant_a"], "Variant A")
        with col_b:
            render_trace(eval_result["variant_b"], "Variant B")

        # ── Self-Correct panel ────────────────────────────────────────────────
        suggested = st.session_state.get("suggested_prompt")
        va_failed = eval_result["variant_a"].get("failed_node")
        vb_failed = eval_result["variant_b"].get("failed_node")

        if (va_failed or vb_failed) and suggested:
            st.divider()
            st.subheader("✨ Self-Correct Suggestion")
            failed_label = "Variant A" if va_failed else "Variant B"
            st.caption(f"{failed_label} failed — GPT suggests this improved prompt:")
            st.code(suggested, language="text")

            if st.button("✨ Apply Suggestion to Variant A & Re-run"):
                st.session_state["prompt_a"] = suggested
                st.session_state["suggested_prompt"] = None
                st.session_state["eval_result"] = None
                st.rerun()
        elif va_failed or vb_failed:
            st.divider()
            st.warning("Pipeline failed but no prompt suggestion was generated. Check API keys.")
```

- [ ] **Step 3: Verify UI loads without error**

```bash
streamlit run app.py
```

Expected: App loads with two tabs. "Compare Providers" behaves identically to before. "Evaluate Variants" tab shows prompt text areas and disabled "Run Evaluation" button until audio is selected.

- [ ] **Step 4: End-to-end smoke test**

1. Pick `call_02_clear_english.wav` from sidebar
2. Go to "Evaluate Variants" tab
3. Variant A pre-filled with default intent prompt
4. Variant B: paste `What does the caller want? Respond with JSON {\"intent\": \"<NAME>\", \"confidence\": <0-1>}. Choices: PASSWORD_RESET, ACCOUNT_INQUIRY, COMPLAINT, BILLING, UNCLEAR.`
5. Target node: `intent`
6. Click "🔬 Run Evaluation"
7. Both traces render side by side

Expected: Both variants succeed (clear english call), no self-correct panel shown.

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "feat: add Evaluate Variants tab with prompt A/B testing and self-correct UI"
```

---

### Task 6: Add `suggested_prompt` to `_state_to_trace` in `api/main.py`

**Files:**
- Modify: `api/main.py`

**Context:** `_state_to_trace` builds `CallTrace` from raw state dict. `CallTrace` doesn't have a `suggested_prompt` field — but the UI reads it from the raw trace dict returned by `/evaluate`. We need to pass it through. Simplest fix: add `suggested_prompt` to `CallTrace`.

- [ ] **Step 1: Add `suggested_prompt` field to `CallTrace` in `api/models.py`**

Open `api/models.py`. In the `CallTrace` class, add after `recommendation`:

```python
suggested_prompt: Optional[str] = None
```

- [ ] **Step 2: Pass `suggested_prompt` in `_state_to_trace`**

Open `api/main.py`. In `_state_to_trace`, find the `return CallTrace(...)` block. Add `suggested_prompt=state.get("suggested_prompt"),` before the closing parenthesis:

```python
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
```

- [ ] **Step 3: Restart API and verify `/evaluate` response includes `suggested_prompt`**

```bash
# Kill and restart uvicorn
curl -s -X POST http://localhost:8000/evaluate \
  -F "file=@api/demo_calls/call_02_clear_english.wav" \
  -F "prompt_a=Classify the user intent. Return JSON: {\"intent\": \"<NAME>\", \"confidence\": <0-1>}. Options: PASSWORD_RESET, ACCOUNT_INQUIRY, COMPLAINT, BILLING, UNCLEAR." \
  -F "prompt_b=Bad prompt that will likely produce UNCLEAR." \
  -F "target_node=intent" | python3 -m json.tool | grep suggested_prompt
```

Expected: `"suggested_prompt": null` or a string value (depending on whether any variant fails).

- [ ] **Step 4: Commit**

```bash
git add api/models.py api/main.py
git commit -m "feat: propagate suggested_prompt through CallTrace model and state_to_trace"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|-----------------|------|
| Prompt override in `intent_node` + `response_node` | Task 1 |
| `root_cause_node` generates `suggested_prompt` | Task 2 |
| `evaluate_pipeline` wrapper in `graph.py` | Task 3 |
| `EvaluateResult` model | Task 3 |
| `POST /evaluate` endpoint | Task 4 |
| Streamlit tab2 with prompt text areas + run button | Task 5 |
| Side-by-side `render_trace` for variants | Task 5 |
| Self-Correct panel with diff + Apply button | Task 5 |
| `suggested_prompt` flows through to UI | Task 6 |
| `/compare` untouched | Task 5 wraps existing code verbatim |
| No DB changes | Confirmed — no storage.py changes |

**Placeholder scan:** None found. All code blocks are complete.

**Type consistency:**
- `evaluate_pipeline` defined in Task 3, called in Task 4 ✓
- `EvaluateResult` defined in Task 3, imported in Task 4 ✓
- `suggested_prompt` field added to `CallTrace` in Task 6, read in Task 5 UI ✓
- `DEFAULT_INTENT_PROMPT` / `DEFAULT_RESPONSE_PROMPT` defined in both `nodes.py` (Task 1) and `app.py` (Task 5) independently — correct, no shared import needed ✓
