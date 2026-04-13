"""
Five research agents, each backed by a specific Google model.

All agents accept a `client` (google.genai.Client) and return plain strings
so the orchestrator can stitch them together without model-specific logic.

Agents:
  planner    — Gemma 4 26B (free)    — decides which tasks to run
  literature — Gemini 2.5 Pro        — searches papers via Search Grounding
  analyst    — Gemini 2.5 Pro        — analyses uploaded files via Code Execution
  hypothesis — Gemini 2.5 Pro        — synthesises findings into hypotheses
  reply      — Gemini 2.5 Flash      — formats the final user-facing response
"""

from __future__ import annotations

import json
from typing import Generator

from google import genai
from google.genai import types

from .config import MODEL_PLANNER, MODEL_REPLY, MODEL_RESEARCH

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _text(response) -> str:
    """Extract plain text from a GenerateContentResponse."""
    return response.text or ""


def _build_file_parts(client: genai.Client, file_paths: list[str]) -> list:
    """Upload local files to the Gemini File API and return Part objects."""
    parts = []
    for path in file_paths:
        uploaded = client.files.upload(file=path)
        parts.append(types.Part.from_uri(file_uri=uploaded.uri, mime_type=uploaded.mime_type))
    return parts


# ---------------------------------------------------------------------------
# Planner — Gemma 4 26B (free)
# ---------------------------------------------------------------------------
PLANNER_SYSTEM = """\
You are a scientific research planning assistant.
Given a research topic and the current session state, decide which of the
following tasks should run in the next iteration:

  - literature_search  : search for relevant papers and findings
  - data_analysis      : analyse an uploaded dataset or document
  - hypothesis         : generate or refine hypotheses
  - all                : run all applicable tasks

Return a JSON object with a single key "tasks" containing a list of task names.
Example: {"tasks": ["literature_search", "hypothesis"]}
Only include tasks that are actually useful given the context.
If a file is uploaded, always include data_analysis.
"""

def run_planner(client: genai.Client, topic: str, session: dict) -> list[str]:
    """Return a list of task names to execute this iteration."""
    context = {
        "topic": topic,
        "iteration": session.get("iteration", 0),
        "has_uploaded_files": len(session.get("uploaded_files", [])) > 0,
        "hypotheses_so_far": len(session.get("hypotheses", [])),
        "discoveries_so_far": len(session.get("discoveries", [])),
    }
    prompt = f"Session context:\n{json.dumps(context, indent=2)}\n\nDecide which tasks to run."

    response = client.models.generate_content(
        model=MODEL_PLANNER,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=PLANNER_SYSTEM,
            response_mime_type="application/json",
        ),
    )
    try:
        data = json.loads(_text(response))
        tasks = data.get("tasks", ["literature_search", "hypothesis"])
    except (json.JSONDecodeError, AttributeError):
        tasks = ["literature_search", "hypothesis"]
    return tasks


# ---------------------------------------------------------------------------
# Literature agent — Gemini 2.5 Pro + Search Grounding
# ---------------------------------------------------------------------------
LITERATURE_SYSTEM = """\
You are a scientific literature search specialist.
Search for recent, relevant peer-reviewed papers on the given topic.
For each finding, cite the paper title, authors, year, and a brief summary.
Focus on primary research. Be concise but thorough.
"""

def run_literature(client: genai.Client, topic: str, context: str = "") -> str:
    """Search literature using Gemini Search Grounding. Returns findings as text."""
    query = f"Recent scientific research on: {topic}"
    if context:
        query += f"\n\nFocus especially on: {context}"

    response = client.models.generate_content(
        model=MODEL_RESEARCH,
        contents=query,
        config=types.GenerateContentConfig(
            system_instruction=LITERATURE_SYSTEM,
            tools=[types.Tool(google_search=types.GoogleSearch())],
        ),
    )
    return _text(response)


def stream_literature(client: genai.Client, topic: str, context: str = "") -> Generator[str, None, None]:
    """Streaming version for live Gradio display."""
    query = f"Recent scientific research on: {topic}"
    if context:
        query += f"\n\nFocus especially on: {context}"

    for chunk in client.models.generate_content_stream(
        model=MODEL_RESEARCH,
        contents=query,
        config=types.GenerateContentConfig(
            system_instruction=LITERATURE_SYSTEM,
            tools=[types.Tool(google_search=types.GoogleSearch())],
        ),
    ):
        if chunk.text:
            yield chunk.text


# ---------------------------------------------------------------------------
# Analyst — Gemini 2.5 Pro + Code Execution
# ---------------------------------------------------------------------------
ANALYST_SYSTEM = """\
You are a data analyst specialising in biological and life sciences data.
Analyse the provided file(s) thoroughly:
  - For CSV/tabular data: describe structure, key statistics, notable patterns, outliers
  - For PDFs/papers: extract key findings, methods, and data points
  - Always relate findings back to the research topic
Write Python code to compute statistics when useful, then interpret the results.
"""

def run_analyst(client: genai.Client, topic: str, file_paths: list[str]) -> str:
    """Analyse uploaded files. Returns analysis as text."""
    if not file_paths:
        return ""

    file_parts = _build_file_parts(client, file_paths)
    text_part = types.Part.from_text(
        text=f"Research topic: {topic}\n\nPlease analyse the attached file(s) in the context of this topic."
    )

    response = client.models.generate_content(
        model=MODEL_RESEARCH,
        contents=types.Content(parts=[text_part] + file_parts, role="user"),
        config=types.GenerateContentConfig(
            system_instruction=ANALYST_SYSTEM,
            tools=[types.Tool(code_execution=types.ToolCodeExecution())],
        ),
    )
    return _text(response)


# ---------------------------------------------------------------------------
# Hypothesis agent — Gemini 2.5 Pro
# ---------------------------------------------------------------------------
HYPOTHESIS_SYSTEM = """\
You are a scientific hypothesis generator.
Given literature findings and any data analysis results, generate 2-4 clear,
testable scientific hypotheses. For each hypothesis:
  1. State it as a falsifiable claim (e.g. "X causes Y under condition Z")
  2. Briefly explain the supporting evidence from the findings
  3. Suggest one experiment or analysis that could test it

Be specific and grounded in the evidence — avoid vague generalisations.
"""

def run_hypothesis(
    client: genai.Client,
    topic: str,
    literature_findings: str,
    analysis_findings: str = "",
    prior_hypotheses: list[str] | None = None,
) -> str:
    """Generate hypotheses from findings. Returns hypothesis text."""
    context_parts = [f"Research topic: {topic}", f"\nLiterature findings:\n{literature_findings}"]
    if analysis_findings:
        context_parts.append(f"\nData analysis findings:\n{analysis_findings}")
    if prior_hypotheses:
        context_parts.append(
            f"\nPreviously generated hypotheses (refine or build upon these):\n"
            + "\n".join(f"- {h}" for h in prior_hypotheses)
        )

    response = client.models.generate_content(
        model=MODEL_RESEARCH,
        contents="\n".join(context_parts),
        config=types.GenerateContentConfig(system_instruction=HYPOTHESIS_SYSTEM),
    )
    return _text(response)


# ---------------------------------------------------------------------------
# Reply agent — Gemini 2.5 Flash
# ---------------------------------------------------------------------------
REPLY_SYSTEM = """\
You are a scientific research assistant presenting findings to a researcher.
Summarise the research results clearly and accessibly.
Structure your response with:
  - A brief answer to the original question
  - Key findings (bullet points)
  - Generated hypotheses (if any)
  - Suggested next steps
Keep it concise. Use markdown formatting.
"""

def run_reply(
    client: genai.Client,
    topic: str,
    literature_findings: str,
    analysis_findings: str = "",
    hypotheses: str = "",
    iteration: int = 0,
) -> str:
    """Format a final user-facing summary. Returns markdown text."""
    parts = [f"Research topic: {topic}", f"\nLiterature findings:\n{literature_findings}"]
    if analysis_findings:
        parts.append(f"\nData analysis:\n{analysis_findings}")
    if hypotheses:
        parts.append(f"\nHypotheses generated:\n{hypotheses}")
    if iteration > 0:
        parts.append(f"\n(This is iteration {iteration} of deep research)")

    response = client.models.generate_content(
        model=MODEL_REPLY,
        contents="\n".join(parts),
        config=types.GenerateContentConfig(system_instruction=REPLY_SYSTEM),
    )
    return _text(response)


def stream_reply(
    client: genai.Client,
    topic: str,
    literature_findings: str,
    analysis_findings: str = "",
    hypotheses: str = "",
    iteration: int = 0,
) -> Generator[str, None, None]:
    """Streaming version for live Gradio display."""
    parts = [f"Research topic: {topic}", f"\nLiterature findings:\n{literature_findings}"]
    if analysis_findings:
        parts.append(f"\nData analysis:\n{analysis_findings}")
    if hypotheses:
        parts.append(f"\nHypotheses generated:\n{hypotheses}")
    if iteration > 0:
        parts.append(f"\n(This is iteration {iteration} of deep research)")

    for chunk in client.models.generate_content_stream(
        model=MODEL_REPLY,
        contents="\n".join(parts),
        config=types.GenerateContentConfig(system_instruction=REPLY_SYSTEM),
    ):
        if chunk.text:
            yield chunk.text
