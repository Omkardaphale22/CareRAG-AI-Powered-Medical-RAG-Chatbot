"""
Tiny JSONL logger for 👍 / 👎 answer ratings.

Each line is one event:
  {"ts": "...", "chat_id": "...", "message_id": "...", "rating": "up" | "down" | "cleared",
   "question": "...", "answer": "...", "sources": [...]}
Read it later with pandas:  pd.read_json("data/feedback/feedback.jsonl", lines=True)
"""
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

_lock = threading.Lock()


def log_feedback(path: str, event: Dict[str, Any]) -> None:
    """Append one rating event. Never raises - a failed log must not break the chat."""
    try:
        file = Path(path)
        file.parent.mkdir(parents=True, exist_ok=True)
        record = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **event}
        with _lock, open(file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:  # pragma: no cover
        print(f"[feedback] could not write rating: {exc}")


def load_feedback(path: str) -> List[Dict[str, Any]]:
    file = Path(path)
    if not file.exists():
        return []
    rows = []
    with open(file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows
