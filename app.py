"""Gradio chat UI. UI only: all retrieval and answer logic lives in rag.py.

Privacy: this file never logs, prints or stores user messages.
"""

import os
import threading
import time
from collections import defaultdict, deque

import gradio as gr

import rag

# Optional shared passcode. Empty = open app (portfolio mode); spending is then bounded by the
# rate limits below and the OpenAI spending cap on the key.
APP_PASSCODE = (os.getenv("APP_PASSCODE") or "").strip()
rag.active_run_id()  # fail at startup, not on the first question, if the run is missing or mismatched

# Best-effort rate limit, in memory. Limits: resets on restart, counts per browser session
# (a new session gets a fresh allowance), and the global caps are shared by everyone.
SESSION_LIMIT = int(os.getenv("RATE_LIMIT_PER_HOUR", "30"))
GLOBAL_HOURLY_LIMIT = int(os.getenv("RATE_LIMIT_GLOBAL_PER_HOUR", "100"))
GLOBAL_DAILY_LIMIT = int(os.getenv("RATE_LIMIT_GLOBAL_PER_DAY", "300"))
_calls: dict[str, deque] = defaultdict(deque)
_lock = threading.Lock()


def allow(session: str) -> bool:
    now = time.time()
    with _lock:
        for key, window in ((session, 3600), ("*", 86400)):
            q = _calls[key]
            while q and q[0] < now - window:
                q.popleft()
        everyone = _calls["*"]
        last_hour = sum(1 for t in everyone if t >= now - 3600)
        if (len(_calls[session]) >= SESSION_LIMIT or last_hour >= GLOBAL_HOURLY_LIMIT
                or len(everyone) >= GLOBAL_DAILY_LIMIT):
            return False
        _calls[session].append(now)
        everyone.append(now)
        return True


def respond(message: str, history: list[dict], request: gr.Request):
    if not message.strip():
        return
    if not allow(request.session_hash or "unknown"):
        yield "The question limit for this demo has been reached. Please try again later."
        return
    yield "_Searching RA 9262 and the IRR..._"
    sections, stream = rag.answer(message, history)
    text = ""
    for delta in stream:
        text += delta
        yield rag.clean_citations(text)
    text, removed = rag.drop_unretrieved_citations(rag.clean_citations(text), sections)
    note = ""
    if removed:
        note = (f"\n\n<small>Removed {len(removed)} citation(s) that did not match a retrieved "
                f"section: {', '.join(removed)}. Check the claims they were attached to.</small>")
    yield f"{text}{note}\n\n{rag.sources_list(sections)}"


CSS = """
:root { --callout-danger: #b42318; --callout-danger-bg: #fef3f2; --callout-note-bg: #f2f4f7;
        --muted: #475467; }
.dark { --callout-danger: #f97066; --callout-danger-bg: #2d1414; --callout-note-bg: #1f2430;
        --muted: #98a2b3; }
.app-header h1 { margin: 0 0 4px; font-size: 1.6rem; }
.app-header p { margin: 0; color: var(--muted); }
.callout { border-radius: 8px; padding: 10px 14px; margin: 0; font-size: 0.95rem; }
.callout p { margin: 0; }
.callout-danger { border-left: 4px solid var(--callout-danger); background: var(--callout-danger-bg); }
.callout-danger strong { color: var(--callout-danger); }
.callout-note { background: var(--callout-note-bg); color: var(--muted); }
details summary { cursor: pointer; color: var(--muted); font-size: 0.9rem; }
details p { margin: 8px 0 2px; font-size: 0.85rem; color: var(--muted); }
details ul { margin-top: 0; }
"""

HEADER = """<div class="app-header">
<h1>RA 9262 Assistant</h1>
<p>Ask about the Anti-Violence Against Women and Their Children Act of 2004 and its 2004 Implementing Rules.
Every answer cites the section it comes from.</p>
</div>"""

DANGER = f"""<div class="callout callout-danger">
<p><strong>In immediate danger? Call 911 now.</strong></p>
</div>"""

NOTE = """<div class="callout callout-note"><p><strong>Legal information, not legal advice.</strong>
For advice on your situation, talk to a lawyer or the Public Attorney's Office.</p></div>"""

ABOUT = f"""**What it uses.** Only the text of RA 9262 and its 2004 IRR. The IRR here is the 2004 original;
later amendments, other laws and court decisions are not included. If the documents do not cover
your question, it says: "{rag.REFUSAL}"

**How to read answers.** Each claim ends with a citation such as [RA 9262, Section 9].
The Act outranks the IRR. Open "Retrieved sections" under an answer to see what it was based on.

**Privacy.** This app does not log or store your messages. OpenAI (the model provider) and
Hugging Face (the host) apply their own data policies. Avoid sharing names or other identifying details.

**Limits.** {SESSION_LIMIT} questions per hour per browser session."""

with gr.Blocks(title="RA 9262 Assistant", fill_height=True, analytics_enabled=False) as demo:
    gr.HTML(HEADER)
    gr.HTML(DANGER)
    gr.HTML(NOTE)
    with gr.Accordion("About this assistant", open=False):
        gr.Markdown(ABOUT)
    gr.ChatInterface(
        fn=respond,
        chatbot=gr.Chatbot(show_label=False, placeholder="Ask a question, or pick an example below.",
                           scale=1),
        textbox=gr.Textbox(placeholder="Ask about protection orders, penalties, who may file...",
                           show_label=False, submit_btn=True),
        examples=[
            "Who may file a petition for a protection order?",
            "What is a Barangay Protection Order and how long does it last?",
            "Can battered woman syndrome be used as a defense?",
        ],
        flagging_mode="never",
        analytics_enabled=False,
        fill_height=True,
    )

# ZeroGPU Spaces expect at least one @spaces.GPU function. This app never uses a GPU
# (models run at OpenAI, search runs in Postgres); this stub is never called.
try:
    import spaces

    @spaces.GPU
    def _unused_gpu_stub():
        return None
except ImportError:
    pass

# Always light mode. Gradio follows the visitor's system theme unless the URL says
# ?__theme=light, so add it once (one reload on the first visit).
FORCE_LIGHT = """() => {
    const url = new URL(window.location);
    if (url.searchParams.get("__theme") !== "light") {
        url.searchParams.set("__theme", "light");
        window.location.replace(url);
    }
}"""

if __name__ == "__main__":
    auth = (lambda _user, password: password == APP_PASSCODE) if APP_PASSCODE else None
    demo.launch(auth=auth, auth_message="Enter the passcode. The username can be anything.",
                css=CSS, js=FORCE_LIGHT, footer_links=["gradio"])
