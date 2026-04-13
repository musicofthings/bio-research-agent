"""
Google Drive persistence for research sessions.

Sessions are stored as JSON files under:
  $DRIVE_ROOT/sessions/{session_id}.json

Uploaded files are copied under:
  $DRIVE_ROOT/uploads/{session_id}/{filename}
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from typing import Optional

from .config import SESSIONS_DIR, UPLOADS_DIR


# ---------------------------------------------------------------------------
# Session schema
# ---------------------------------------------------------------------------

def new_session(topic: str) -> dict:
    return {
        "id": uuid.uuid4().hex[:12],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "topic": topic,
        "messages": [],       # {"role": "user"|"agent", "content": str, "agent": str|None}
        "hypotheses": [],     # str
        "discoveries": [],    # {"finding": str, "source": str}
        "uploaded_files": [], # {"name": str, "drive_path": str, "gemini_uri": str|None}
        "iteration": 0,
    }


# ---------------------------------------------------------------------------
# Drive helpers
# ---------------------------------------------------------------------------

def _ensure_dirs() -> None:
    os.makedirs(SESSIONS_DIR, exist_ok=True)
    os.makedirs(UPLOADS_DIR, exist_ok=True)


def _session_path(session_id: str) -> str:
    return os.path.join(SESSIONS_DIR, f"{session_id}.json")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_session(session: dict) -> None:
    _ensure_dirs()
    path = _session_path(session["id"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(session, f, indent=2, ensure_ascii=False)


def load_session(session_id: str) -> Optional[dict]:
    path = _session_path(session_id)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def list_sessions() -> list[dict]:
    """Return sessions sorted newest-first, each as a summary dict."""
    _ensure_dirs()
    sessions = []
    for fname in os.listdir(SESSIONS_DIR):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(SESSIONS_DIR, fname)
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            sessions.append({
                "id": data["id"],
                "topic": data.get("topic", "Untitled"),
                "created_at": data.get("created_at", ""),
                "iteration": data.get("iteration", 0),
                "hypotheses_count": len(data.get("hypotheses", [])),
            })
        except (json.JSONDecodeError, KeyError):
            continue
    sessions.sort(key=lambda s: s["created_at"], reverse=True)
    return sessions


def save_uploaded_file(session_id: str, src_path: str) -> str:
    """Copy an uploaded file to Drive and return the Drive path."""
    _ensure_dirs()
    dest_dir = os.path.join(UPLOADS_DIR, session_id)
    os.makedirs(dest_dir, exist_ok=True)
    filename = os.path.basename(src_path)
    dest_path = os.path.join(dest_dir, filename)
    shutil.copy2(src_path, dest_path)
    return dest_path
