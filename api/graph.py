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
        "suggested_prompt": None,
        "intent_system_prompt": None,
        "response_system_prompt": None,
    }
    result = graph.invoke(initial_state)
    return result


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
