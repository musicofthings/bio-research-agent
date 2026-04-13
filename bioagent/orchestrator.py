"""
Research pipeline orchestrator.

Two modes:

  chat_turn(client, session, user_message, file_paths)
    Single-turn: literature search → reply
    Yields progress strings for streaming display.

  deep_research_turn(client, session, steer_message, file_paths)
    One iteration of deep research: plan → literature → [analyze] → hypothesize → reply
    Updates session state in-place.
    Yields progress strings for streaming display.

The caller (ui.py) is responsible for saving the session to Drive after each turn.
"""

from __future__ import annotations

from typing import Generator

from google import genai

from .agents import (
    run_analyst,
    run_hypothesis,
    run_literature,
    run_planner,
    run_reply,
    stream_literature,
    stream_reply,
)
from .persistence import new_session, save_uploaded_file


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def make_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


# ---------------------------------------------------------------------------
# Chat mode — single turn
# ---------------------------------------------------------------------------

def chat_turn(
    client: genai.Client,
    session: dict,
    user_message: str,
    file_paths: list[str] | None = None,
) -> Generator[str, None, None]:
    """
    Runs one chat turn. Yields incremental text for the Gradio chatbot.
    Mutates session in-place (appends messages).
    """
    file_paths = file_paths or []

    # Persist uploaded files to Drive
    drive_paths = []
    for fp in file_paths:
        drive_path = save_uploaded_file(session["id"], fp)
        drive_paths.append(drive_path)
        session["uploaded_files"].append({"name": fp.split("/")[-1], "drive_path": drive_path, "gemini_uri": None})

    session["messages"].append({"role": "user", "content": user_message, "agent": None})

    # If files uploaded, run analysis first
    analysis = ""
    if drive_paths:
        yield "\n\n**Analysing uploaded files...**\n\n"
        analysis = run_analyst(client, user_message, drive_paths)
        yield analysis

    # Literature search (streaming)
    yield "\n\n**Searching literature...**\n\n"
    literature = ""
    for chunk in stream_literature(client, user_message):
        literature += chunk
        yield chunk

    # Reply (streaming)
    yield "\n\n---\n\n**Summary**\n\n"
    reply = ""
    for chunk in stream_reply(client, user_message, literature, analysis_findings=analysis):
        reply += chunk
        yield chunk

    session["messages"].append({"role": "agent", "content": reply, "agent": "reply"})


# ---------------------------------------------------------------------------
# Deep research mode — one iteration
# ---------------------------------------------------------------------------

def deep_research_turn(
    client: genai.Client,
    session: dict,
    steer_message: str = "",
    file_paths: list[str] | None = None,
) -> Generator[str, None, None]:
    """
    Runs one deep-research iteration. Yields progress strings.
    Mutates session in-place.
    """
    file_paths = file_paths or []
    topic = session["topic"]
    iteration = session["iteration"] + 1
    session["iteration"] = iteration

    # Persist uploaded files
    drive_paths = []
    for fp in file_paths:
        drive_path = save_uploaded_file(session["id"], fp)
        drive_paths.append(drive_path)
        session["uploaded_files"].append({"name": fp.split("/")[-1], "drive_path": drive_path, "gemini_uri": None})

    if steer_message:
        session["messages"].append({"role": "user", "content": steer_message, "agent": None})

    effective_topic = f"{topic}. {steer_message}" if steer_message else topic

    # ── Step 1: Plan ────────────────────────────────────────────────────────
    yield f"## Iteration {iteration}\n\n**Planning...**\n"
    tasks = run_planner(client, effective_topic, session)
    # Always run at least literature + hypothesis
    if "literature_search" not in tasks:
        tasks.insert(0, "literature_search")
    if "data_analysis" not in tasks and drive_paths:
        tasks.append("data_analysis")
    yield f"Tasks this iteration: {', '.join(tasks)}\n\n"

    # ── Step 2: Literature ───────────────────────────────────────────────────
    literature = ""
    if "literature_search" in tasks:
        yield "**Searching literature...**\n\n"
        for chunk in stream_literature(client, effective_topic):
            literature += chunk
            yield chunk
        yield "\n\n"
        session["discoveries"].append({"finding": literature[:500] + "...", "source": "literature_search"})

    # ── Step 3: Data analysis ────────────────────────────────────────────────
    analysis = ""
    all_drive_paths = (
        drive_paths
        or [f["drive_path"] for f in session.get("uploaded_files", [])]
    )
    if "data_analysis" in tasks and all_drive_paths:
        yield "**Analysing data...**\n\n"
        analysis = run_analyst(client, effective_topic, all_drive_paths)
        yield analysis
        yield "\n\n"

    # ── Step 4: Hypothesis ───────────────────────────────────────────────────
    hypotheses_text = ""
    if "hypothesis" in tasks or "all" in tasks:
        yield "**Generating hypotheses...**\n\n"
        hypotheses_text = run_hypothesis(
            client,
            effective_topic,
            literature,
            analysis_findings=analysis,
            prior_hypotheses=session.get("hypotheses"),
        )
        yield hypotheses_text
        yield "\n\n"
        # Store each hypothesis bullet as a separate entry
        for line in hypotheses_text.splitlines():
            line = line.strip()
            if line and len(line) > 20:
                session["hypotheses"].append(line)

    # ── Step 5: Summary reply ────────────────────────────────────────────────
    yield "---\n\n**Iteration Summary**\n\n"
    reply = ""
    for chunk in stream_reply(
        client,
        effective_topic,
        literature,
        analysis_findings=analysis,
        hypotheses=hypotheses_text,
        iteration=iteration,
    ):
        reply += chunk
        yield chunk

    session["messages"].append({
        "role": "agent",
        "content": reply,
        "agent": "reply",
        "iteration": iteration,
    })


# ---------------------------------------------------------------------------
# Session lifecycle helpers
# ---------------------------------------------------------------------------

def start_session(topic: str) -> dict:
    return new_session(topic)
