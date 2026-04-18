# VoiceTrace — Evaluate Variants + Self-Correct Design Spec
Date: 2026-04-17

## Overview

Extends VoiceTrace MVP with two new pillars from the SuperBryn vision:

1. **Evaluate** — A/B test two prompt variants (intent and/or response node system prompts) against the same audio call, side-by-side
2. **Self-Correct** — When a variant fails, GPT generates an improved system prompt suggestion shown as a diff; user clicks "Apply & Re-run" to re-evaluate with the fix

No persistence. All state lives in Streamlit session. No schema changes to SQLite.

## Scope

**In scope:**
- New `/evaluate` FastAPI endpoint
- Prompt override support in `intent_node` and `response_node`
- `root_cause_node` extended to output `suggested_prompt`
- New "Evaluate Variants" tab in Streamlit UI
- "Apply & Re-run" button using `st.session_state`

**Out of scope:**
- Saving prompt configs to DB
- Version history / lineage
- Multi-call batch evaluation
- Auto-apply without user confirmation

## Architecture

Current flow unchanged:
```
audio → POST /compare → Deepgram vs Sarvam (provider comparison)
```

New flow:
```
audio + prompt_a + prompt_b + target_node → POST /evaluate → variant_a vs variant_b
```

Both variants use Deepgram (fixed provider). Only system prompts differ. `/compare` is untouched.

## File Changes

| File | Change |
|------|--------|
| `api/nodes.py` | `intent_node` reads `state.get("intent_system_prompt")`, falls back to hardcoded default. Same for `response_node` with `"response_system_prompt"`. |
| `api/graph.py` | New `evaluate_pipeline(audio_bytes, prompt_config: dict)` — thin wrapper around `run_pipeline` that merges prompt_config into initial state |
| `api/models.py` | New `EvaluateResult(BaseModel)` with `variant_a: CallTrace`, `variant_b: CallTrace` |
| `api/main.py` | New `POST /evaluate` endpoint accepting `file`, `prompt_a`, `prompt_b`, `target_node` as form fields |
| `api/nodes.py` | `root_cause_node` extended: after root_cause/recommendation, calls GPT once more to generate `suggested_prompt` for the failed node |
| `app.py` | Wrap existing UI in `tab1`. Add `tab2` — Evaluate Variants UI |

## Data Flow

### `/evaluate` endpoint

```
POST /evaluate
  file:        audio (multipart)
  prompt_a:    str — system prompt for target node, variant A
  prompt_b:    str — system prompt for target node, variant B
  target_node: "intent" | "response" | "both"

runs asyncio.gather(
  evaluate_pipeline(audio, {intent_system_prompt: prompt_a, ...}),
  evaluate_pipeline(audio, {intent_system_prompt: prompt_b, ...})
)

returns EvaluateResult {
  variant_a: CallTrace,
  variant_b: CallTrace
}
```

### State additions (no TypedDict change required, dict is open)

```python
state["intent_system_prompt"]    # str | None — overrides hardcoded default
state["response_system_prompt"]  # str | None — overrides hardcoded default
state["suggested_prompt"]        # str | None — output of self-correct GPT call
```

### `root_cause_node` extension

After existing root cause call, if `failed_node` is `"intent"` or `"response"`, fires second GPT call:

```
System: "You are a prompt engineer. Rewrite the system prompt for the {failed_node} node to fix the failure."
User: "Current prompt: {current_prompt}\nFailure: {error}\nTranscript: {transcript}"
Returns JSON: {"suggested_prompt": "..."}
```

Sets `state["suggested_prompt"]`. If GPT call fails, sets `None` silently.

## UI Layout

### `app.py` restructure

```python
tab1, tab2 = st.tabs(["Compare Providers", "Evaluate Variants"])

with tab1:
    # existing UI — zero changes

with tab2:
    # new evaluate UI
```

### Tab 2 layout

```
[Sidebar — shared audio picker]

Main:
  Target Node: ○ intent  ○ response  ○ both   (st.radio)

  col_a                        col_b
  ┌──────────────────────┐    ┌──────────────────────┐
  │ Variant A Prompt     │    │ Variant B Prompt     │
  │ st.text_area         │    │ st.text_area         │
  │ (prefilled default)  │    │ (blank)              │
  └──────────────────────┘    └──────────────────────┘

  [🔬 Run Evaluation]

  ── Results (after run) ──────────────────────────────

  col_a                        col_b
  render_trace(variant_a,      render_trace(variant_b,
    "Variant A")                 "Variant B")

  ── Self-Correct (shown if failed_node set on either) ──

  st.code(suggested_prompt diff: original vs suggested)
  [✨ Apply Suggestion to Variant A]
    → sets prompt_a textarea = suggested_prompt
    → re-fires /evaluate
```

### Session state keys

```python
st.session_state["eval_result"]        # last EvaluateResult dict
st.session_state["suggested_prompt"]   # from failed variant's root_cause
st.session_state["prompt_a"]           # current value of variant A textarea
st.session_state["prompt_b"]           # current value of variant B textarea
```

## Default Prompts (prefilled in Variant A)

**Intent node default:**
```
Classify the user intent from the transcript.
Return JSON: {"intent": "<INTENT_NAME>", "confidence": <0.0-1.0>}.
Intent options: PASSWORD_RESET, ACCOUNT_INQUIRY, COMPLAINT, BILLING, UNCLEAR.
```

**Response node default:**
```
You are a helpful customer support voice agent for an Indian bank.
Respond concisely (1-2 sentences) to the user's request.
```

## Error Handling

- `/evaluate` returns 200 even if one variant fails — failure is surfaced in `CallTrace.failed_node`
- If `suggested_prompt` is `None` (GPT failed), Self-Correct section is hidden
- "Apply & Re-run" disabled if no `suggested_prompt` in session state

## Out of Scope
- Saving variants to DB
- Scoring/ranking variants automatically
- Multi-call batch evaluation
- Tests
