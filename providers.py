"""Model calls behind two small functions, so models swap by env var.

Only OpenAI is wired up. Swapping providers means changing this file only.
"""

import os
from collections.abc import Iterator

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large").strip()
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4.1-mini").strip()
REWRITE_MODEL = os.getenv("REWRITE_MODEL", CHAT_MODEL).strip()
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4.1").strip()

_client: OpenAI | None = None


def client() -> OpenAI:
    # Created on first use so importing this module never needs an API key (tests, metrics).
    global _client
    if _client is None:
        # strip(): a secret pasted into a web form often carries a trailing newline,
        # which makes the Authorization header invalid.
        _client = OpenAI(api_key=(os.getenv("OPENAI_API_KEY") or "").strip() or None)
    return _client


def embed(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    """Text -> vectors. Same model in, comparable vectors out: chunks are embedded once by
    embed.py, questions every time someone asks, and search compares the two."""
    out = []
    for start in range(0, len(texts), 100):  # stay well under the per-request input limit
        response = client().embeddings.create(model=model, input=texts[start : start + 100])
        out += [d.embedding for d in response.data]
    return out


def chat(messages: list[dict], model: str = CHAT_MODEL, temperature: float = 0) -> str:
    response = client().chat.completions.create(model=model, messages=messages, temperature=temperature)
    return response.choices[0].message.content or ""


def chat_stream(messages: list[dict], model: str = CHAT_MODEL, temperature: float = 0) -> Iterator[str]:
    stream = client().chat.completions.create(
        model=model, messages=messages, temperature=temperature, stream=True
    )
    for event in stream:
        if event.choices and event.choices[0].delta.content:
            yield event.choices[0].delta.content
