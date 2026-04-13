"""
Gradio UI — 3 tabs:

  1. Chat          — quick Q&A with file upload, streaming response
  2. Deep Research — iterative research loop with session steering
  3. Sessions      — browse and reload past sessions from Drive
"""

from __future__ import annotations

import os

import gradio as gr

from .config import GOOGLE_API_KEY, MAX_ITERATIONS
from .orchestrator import chat_turn, deep_research_turn, make_client, start_session
from .persistence import list_sessions, load_session, save_session

# ---------------------------------------------------------------------------
# Shared state helpers
# ---------------------------------------------------------------------------

def _get_client(api_key: str) -> tuple[object | None, str]:
    key = api_key.strip() or GOOGLE_API_KEY
    if not key:
        return None, "No API key provided. Add your Google AI Studio key above."
    try:
        client = make_client(key)
        return client, ""
    except Exception as e:
        return None, f"Failed to create client: {e}"


# ---------------------------------------------------------------------------
# Tab 1: Chat
# ---------------------------------------------------------------------------

def build_chat_tab():
    with gr.Tab("Chat"):
        gr.Markdown("### Quick Research Chat\nAsk any biology or life sciences question. Attach a CSV or PDF for data analysis.")

        api_key_chat = gr.Textbox(
            label="Google AI Studio API Key",
            placeholder="AIza...",
            type="password",
            value=GOOGLE_API_KEY,
        )

        chatbot = gr.Chatbot(
            label="Research Assistant",
            height=500,
            type="messages",
            show_copy_button=True,
        )

        with gr.Row():
            user_input = gr.Textbox(
                placeholder="Ask a research question...",
                show_label=False,
                scale=4,
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)

        file_upload = gr.File(
            label="Attach file (CSV, PDF, XLSX)",
            file_types=[".csv", ".pdf", ".xlsx", ".txt"],
        )

        clear_btn = gr.Button("Clear chat")

        # Session state (dict) — kept in Gradio State
        chat_session = gr.State(None)

        def send(message, files, history, session, api_key):
            client, err = _get_client(api_key)
            if err:
                history = history or []
                history.append({"role": "assistant", "content": err})
                return history, history, session, ""

            if session is None:
                session = start_session(message[:80])

            history = history or []
            history.append({"role": "user", "content": message})

            file_paths = [f.name for f in files] if files else []

            response_text = ""
            history.append({"role": "assistant", "content": ""})

            try:
                for chunk in chat_turn(client, session, message, file_paths):
                    response_text += chunk
                    history[-1] = {"role": "assistant", "content": response_text}
                    yield history, history, session, ""
                save_session(session)
            except Exception as e:
                import traceback
                err_text = f"**Error:**\n```\n{traceback.format_exc()}\n```"
                history[-1] = {"role": "assistant", "content": err_text}
                print(traceback.format_exc())  # also prints to Colab cell output

            yield history, history, session, ""

        send_btn.click(
            send,
            inputs=[user_input, file_upload, chatbot, chat_session, api_key_chat],
            outputs=[chatbot, chatbot, chat_session, user_input],
        )
        user_input.submit(
            send,
            inputs=[user_input, file_upload, chatbot, chat_session, api_key_chat],
            outputs=[chatbot, chatbot, chat_session, user_input],
        )
        clear_btn.click(lambda: ([], None, ""), outputs=[chatbot, chat_session, user_input])


# ---------------------------------------------------------------------------
# Tab 2: Deep Research
# ---------------------------------------------------------------------------

def build_deep_research_tab():
    with gr.Tab("Deep Research"):
        gr.Markdown(
            "### Iterative Deep Research\n"
            "Enter a topic to start. After each iteration you can steer the next one, upload more data, or stop."
        )

        api_key_dr = gr.Textbox(
            label="Google AI Studio API Key",
            placeholder="AIza...",
            type="password",
            value=GOOGLE_API_KEY,
        )

        with gr.Row():
            topic_input = gr.Textbox(
                label="Research topic",
                placeholder="e.g. CRISPR off-target effects in primary human cells",
                scale=4,
            )
            start_btn = gr.Button("Start Research", variant="primary", scale=1)

        research_output = gr.Markdown(label="Research Output", value="", min_height=400)

        gr.Markdown("---")
        gr.Markdown("**Steer next iteration** (optional — add focus or upload new data, then Continue)")

        with gr.Row():
            steer_input = gr.Textbox(
                placeholder="e.g. Focus more on in-vivo models...",
                show_label=False,
                scale=4,
            )
            continue_btn = gr.Button("Continue", scale=1)
            stop_btn = gr.Button("Stop", scale=1, variant="stop")

        file_upload_dr = gr.File(
            label="Attach file for next iteration",
            file_types=[".csv", ".pdf", ".xlsx", ".txt"],
        )

        iteration_info = gr.Markdown("")

        dr_session = gr.State(None)
        accumulated = gr.State("")

        def start_research(topic, api_key):
            import traceback
            client, err = _get_client(api_key)
            if err:
                yield err, None, "", f"Error: {err}"
                return

            session = start_session(topic)
            output = ""
            try:
                for chunk in deep_research_turn(client, session, file_paths=[]):
                    output += chunk
                    yield output, session, output, f"Iteration {session['iteration']} / {MAX_ITERATIONS}"
                save_session(session)
                yield output, session, output, f"Iteration {session['iteration']} / {MAX_ITERATIONS} — ready to continue or stop"
            except Exception:
                tb = traceback.format_exc()
                print(tb)
                output += f"\n\n**Error:**\n```\n{tb}\n```"
                yield output, session, output, "Error — see traceback above"

        def continue_research(steer, files, session, acc, api_key):
            if session is None:
                yield acc, session, acc, "No active session — start a research topic first."
                return

            if session["iteration"] >= MAX_ITERATIONS:
                yield acc, session, acc, f"Reached max iterations ({MAX_ITERATIONS}). Start a new topic."
                return

            client, err = _get_client(api_key)
            if err:
                yield acc, session, acc, f"Error: {err}"
                return

            file_paths = [f.name for f in files] if files else []
            output = acc + "\n\n---\n\n"

            for chunk in deep_research_turn(client, session, steer_message=steer, file_paths=file_paths):
                output += chunk
                yield output, session, output, f"Iteration {session['iteration']} / {MAX_ITERATIONS}"

            save_session(session)
            yield output, session, output, f"Iteration {session['iteration']} / {MAX_ITERATIONS} — ready to continue or stop"

        start_btn.click(
            start_research,
            inputs=[topic_input, api_key_dr],
            outputs=[research_output, dr_session, accumulated, iteration_info],
        )
        continue_btn.click(
            continue_research,
            inputs=[steer_input, file_upload_dr, dr_session, accumulated, api_key_dr],
            outputs=[research_output, dr_session, accumulated, iteration_info],
        )
        stop_btn.click(
            lambda session: (f"Research stopped at iteration {session['iteration']}." if session else "No active session.", None, ""),
            inputs=[dr_session],
            outputs=[iteration_info, dr_session, accumulated],
        )


# ---------------------------------------------------------------------------
# Tab 3: Sessions
# ---------------------------------------------------------------------------

def build_sessions_tab():
    with gr.Tab("Sessions"):
        gr.Markdown("### Past Research Sessions\nReload a session to review its hypotheses and discoveries.")

        refresh_btn = gr.Button("Refresh list")
        sessions_table = gr.Dataframe(
            headers=["ID", "Topic", "Created", "Iterations", "Hypotheses"],
            datatype=["str", "str", "str", "number", "number"],
            interactive=False,
        )

        session_id_input = gr.Textbox(label="Session ID to load", placeholder="Paste a session ID from the table above")
        load_btn = gr.Button("Load session")

        session_detail = gr.Markdown("")

        def refresh():
            sessions = list_sessions()
            rows = [
                [s["id"], s["topic"], s["created_at"][:10], s["iteration"], s["hypotheses_count"]]
                for s in sessions
            ]
            return rows

        def load(session_id):
            session = load_session(session_id.strip())
            if session is None:
                return "Session not found. Check the ID and make sure Google Drive is mounted."
            lines = [
                f"## {session['topic']}",
                f"**Created:** {session['created_at'][:19]}  |  **Iterations:** {session['iteration']}",
                "",
                f"### Hypotheses ({len(session['hypotheses'])})",
            ]
            for h in session["hypotheses"]:
                lines.append(f"- {h}")
            lines += ["", f"### Discoveries ({len(session['discoveries'])})"]
            for d in session["discoveries"]:
                lines.append(f"- {d['finding'][:200]}...")
            lines += ["", f"### Messages ({len(session['messages'])})"]
            for m in session["messages"][-6:]:
                role = m["role"].upper()
                content = m["content"][:300] + ("..." if len(m["content"]) > 300 else "")
                lines.append(f"**{role}:** {content}\n")
            return "\n".join(lines)

        refresh_btn.click(refresh, outputs=[sessions_table])
        load_btn.click(load, inputs=[session_id_input], outputs=[session_detail])

        # Auto-refresh on tab load
        sessions_table.value = refresh()


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------

def launch(share: bool = False, server_port: int = 7860) -> None:
    """Build and launch the Gradio app."""
    gr.close_all()  # release any port held by a previous launch in this session

    with gr.Blocks(title="Bio Research Agent", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# Bio Research Agent\n"
            "Powered by Google Gemini 3.1 Pro · Gemini 3.1 Flash-Lite · Gemma 4 26B"
        )

        build_chat_tab()
        build_deep_research_tab()
        build_sessions_tab()

    app.launch(share=share, server_port=server_port)
