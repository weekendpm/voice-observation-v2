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

# ── Main: results ────────────────────────────────────────────────────────────

tab1, tab2 = st.tabs(["Compare Providers", "Evaluate Variants"])

# ── Tab 1: existing compare flow ─────────────────────────────────────────────
with tab1:
    if analyze and audio_bytes:
        with st.spinner("Running both pipelines in parallel..."):
            resp = requests.post(
                f"{API_URL}/compare",
                files={"file": ("audio.wav", audio_bytes, "audio/wav")},
                timeout=120,
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
                timeout=120,
            )
        if eval_resp.status_code != 200:
            st.error(f"API error: {eval_resp.text}")
        else:
            eval_result = eval_resp.json()
            st.session_state["eval_result"] = eval_result
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
            if eval_result["variant_a"].get("suggested_prompt"):
                failed_label = "Variant A"
            elif eval_result["variant_b"].get("suggested_prompt"):
                failed_label = "Variant B"
            else:
                failed_label = "a variant"
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
