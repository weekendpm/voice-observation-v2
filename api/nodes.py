import time
import base64
import os
import json
import re
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

DEFAULT_INTENT_PROMPT = (
    "Classify the user intent from the transcript. "
    "Return JSON: {\"intent\": \"<INTENT_NAME>\", \"confidence\": <0.0-1.0>}. "
    "Intent options: PASSWORD_RESET, ACCOUNT_INQUIRY, COMPLAINT, BILLING, UNCLEAR."
)

DEFAULT_RESPONSE_PROMPT = (
    "You are a helpful customer support voice agent for an Indian bank. "
    "Respond concisely (1-2 sentences) to the user's request."
)


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
        resp = httpx.post(
            "https://api.sarvam.ai/speech-to-text",
            headers={"api-subscription-key": SARVAM_API_KEY},
            files={"file": ("audio.wav", state["audio_bytes"], "audio/wav")},
            data={"language_code": "hi-IN", "model": "saarika:v2.5"},
            timeout=30,
        )
        if resp.status_code != 200:
            raise Exception(f"Sarvam STT {resp.status_code}: {resp.text}")
        data = resp.json()
        transcript = data.get("transcript", "")
        confidence = data.get("confidence", 0.85)
        detected_lang = data.get("language_code", "")
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
        system_prompt = state.get("intent_system_prompt") or DEFAULT_INTENT_PROMPT
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state["transcript"]),
        ]
        result = llm.invoke(messages)
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
                "speaker": "aditya",
                "model": "bulbul:v3",
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
    # Self-correct only applies to prompt-driven nodes (intent, response).
    # STT and TTS failures require provider/config changes, not prompt rewrites.
    if failed in ("intent", "response"):
        current_prompt = (
            (state.get("intent_system_prompt") or DEFAULT_INTENT_PROMPT)
            if failed == "intent"
            else (state.get("response_system_prompt") or DEFAULT_RESPONSE_PROMPT)
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
