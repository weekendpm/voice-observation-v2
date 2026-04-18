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
