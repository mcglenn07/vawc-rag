"""All retrieval and answer logic. Imported by app.py and the eval scripts.

Flow: rewrite_query (follow-ups only) -> retrieve (pgvector, merge parts, attach
crosswalk pair) -> answer (streamed, grounded in the retrieved sections only).
"""

import csv
import os
import re
from collections.abc import Iterator
from pathlib import Path

import db
import providers

ROOT = Path(__file__).parent
TOP_K = 5
REFUSAL = "I cannot find this in RA 9262 or the IRR."
SOURCES_HEADER = "Retrieved sections"

# Emergency line shown in the UI and used by the model for immediate-danger messages.
# Set and verified by the owner (911 is the Philippine national emergency hotline), not by the model.
EMERGENCY_CONTACT_TEXT = "If you are in immediate danger, call 911 now."

REWRITE_PROMPT = """You turn the user's latest message into one short search query over \
the text of a Philippine law on violence against women and children and its implementing rules.
- Focus on what the latest message asks. Use the earlier conversation only to resolve references \
such as "it", "this" or "that order"; drop earlier topics the latest message does not ask about.
- Use the wording a statute would use, e.g. "ex-boyfriend" -> "former dating relationship", \
"husband" -> "spouse or wife".
- Do not add the law's name or number, do not answer, and do not add facts.
Output only the query, on one line."""

SYSTEM_PROMPT = f"""You answer questions about Republic Act No. 9262 (Anti-Violence Against Women \
and Their Children Act of 2004) and its 2004 Implementing Rules and Regulations (IRR), using only \
the numbered sources below.

Rules:
1. Use only the sources. Do not use outside legal knowledge, other laws, court decisions, or later \
amendments, even if you know them.
2. Cite every claim: each sentence that states something from the sources ends with the source's \
label in square brackets, e.g. [RA 9262, Section 5] or [IRR, Rule IV, Section 12]. Cite only labels \
that appear in the sources, exactly as written. Do not add sub-item letters or numbers such as (a) \
or (2); some sections have more than one lettered list, so those markers are ambiguous. When a \
source mentions another section ("Section 7 of these Rules", "Section 5 of this Act"), cite the \
source you read it in, not the section it mentions.
3. If the sources do not answer the question, start your reply with exactly: "{REFUSAL}" You may \
add one sentence naming what the sources do cover nearby. Do not guess, and do not fill gaps with \
what you know about the law: if a needed definition is not in the sources, say it is not there.
4. The Act outranks the IRR. Give what the Act says first, then IRR detail. If they differ, \
follow the Act and point out the difference.
5. Who is covered depends on the relationships defined in the sources. If a question involves a \
relationship that the definitions in the sources do not clearly include, say so. Do not extend \
coverage beyond what the sources state.
6. Quote sparingly: a short quote only when the exact wording matters.
7. Never give phone numbers, hotlines, addresses, websites or other contact details, except the \
line in rule 8.
8. If the user says they or someone else is in immediate danger, begin with exactly: \
"{EMERGENCY_CONTACT_TEXT}" Then answer. This is the only contact detail you may give.
9. Be plain and brief. This is legal information, not legal advice."""

_crosswalk: dict[str, list[str]] | None = None
_checked_run: str | None = None


def crosswalk() -> dict[str, list[str]]:
    """Citation -> paired citations, both directions. Only rows the owner marked verified."""
    global _crosswalk
    if _crosswalk is None:
        _crosswalk = {}
        with open(ROOT / "data" / "crosswalk.csv", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["verified"].strip().lower() != "yes":
                    continue
                _crosswalk.setdefault(row["law_section"], []).append(row["irr_section"])
                _crosswalk.setdefault(row["irr_section"], []).append(row["law_section"])
    return _crosswalk


def active_run_id() -> str:
    """ACTIVE_RUN_ID, checked once against the database so a mismatch fails loudly."""
    global _checked_run
    run_id = (os.getenv("ACTIVE_RUN_ID") or "").strip()
    if not run_id:
        raise RuntimeError("ACTIVE_RUN_ID is not set. Run `python embed.py --list` to see stored runs.")
    if _checked_run != run_id:
        run = db.with_retry(lambda conn: db.get_run(conn, run_id))
        if run is None:
            raise RuntimeError(f"ACTIVE_RUN_ID {run_id} is not in the runs table.")
        if run["embed_model"] != providers.EMBED_MODEL:
            raise RuntimeError(f"Run {run_id} was embedded with {run['embed_model']}, "
                               f"but EMBED_MODEL is {providers.EMBED_MODEL}.")
        _checked_run = run_id
    return run_id


_run_models: dict[str, str] = {}


def run_embed_model(run_id: str) -> str:
    """Embedding model a run was built with, so eval can query any stored run correctly."""
    if run_id not in _run_models:
        run = db.with_retry(lambda conn: db.get_run(conn, run_id))
        if run is None:
            raise RuntimeError(f"Run {run_id} is not in the runs table.")
        _run_models[run_id] = run["embed_model"]
    return _run_models[run_id]


def message_text(content) -> str:
    """Gradio message content can be a string or a list of parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def clean_history(history: list[dict] | None, turns: int = 3) -> list[dict]:
    """Last few turns as plain role/content messages, without the sources list we appended."""
    out = []
    for m in (history or [])[-turns * 2 :]:
        text = message_text(m.get("content")).split(f"<details><summary>{SOURCES_HEADER}")[0].strip()
        if text and m.get("role") in ("user", "assistant"):
            out.append({"role": m["role"], "content": text[:1500]})
    return out


def rewrite_query(question: str, history: list[dict] | None) -> str | None:
    """Standalone search query for a follow-up. None on the first turn (nothing to resolve)."""
    past = clean_history(history)
    if not past:
        return None
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in past)
    messages = [{"role": "system", "content": REWRITE_PROMPT},
                {"role": "user", "content": f"Conversation so far:\n{transcript}\n\nLatest message: {question}"}]
    return providers.chat(messages, model=providers.REWRITE_MODEL).strip() or None


def retrieve(question: str, history: list[dict] | None = None, k: int = TOP_K,
             pair: bool = True, run_id: str | None = None) -> list[dict]:
    """Ranked sections for a question.

    Searches the question as typed and, for follow-ups, the rewritten query too; keeps each
    chunk's best distance. Parts of one section merge into one result before counting toward k.
    With pair=True, crosswalk partners of the top k are attached after them (outside k).
    """
    run_id = run_id or active_run_id()
    queries = [question]
    if rewritten := rewrite_query(question, history):
        queries.append(rewritten)
    vectors = providers.embed(queries, model=run_embed_model(run_id))
    return db.with_retry(lambda conn: _search_and_pair(conn, vectors, run_id, k, pair))


def _search_and_pair(conn, vectors: list[list[float]], run_id: str, k: int, pair: bool) -> list[dict]:
    best: dict[tuple[str, int], dict] = {}
    for vector in vectors:
        # Fetch extra chunks so k distinct sections survive the merge of parts.
        for row in db.search(conn, vector, run_id, k * 4):
            key = (row["id"], row["part"])
            if key not in best or row["distance"] < best[key]["distance"]:
                best[key] = row

    sections: dict[str, dict] = {}
    for row in sorted(best.values(), key=lambda r: r["distance"]):
        s = sections.setdefault(row["citation"], {**row, "rows": [], "source": "search"})
        s["rows"].append(row)
    ranked = list(sections.values())[:k]
    for s in ranked:
        # Only the matched parts go to the model; a long section's other parts stay out.
        s["text"] = "\n".join(r["text"] for r in sorted(s.pop("rows"), key=lambda r: r["part"]))

    if pair:
        have = {s["citation"] for s in ranked}
        wanted = [p for s in ranked for p in crosswalk().get(s["citation"], []) if p not in have]
        wanted = list(dict.fromkeys(wanted))
        if wanted:
            paired: dict[str, dict] = {}
            for row in db.get_sections(conn, run_id, wanted):
                s = paired.setdefault(row["citation"], {**row, "text": "", "distance": None,
                                                        "source": "crosswalk"})
                s["text"] = (s["text"] + "\n" + row["text"]).strip()
            ranked += [paired[c] for c in wanted if c in paired]
    return ranked


def format_sources(sections: list[dict]) -> str:
    blocks = []
    for n, s in enumerate(sections, start=1):
        title = s["citation"] + (f": {s['heading']}" if s["heading"] else "")
        blocks.append(f"[Source {n}] {title} (PDF pages {s['pdf_pages'][0]}-{s['pdf_pages'][1]})\n{s['text']}")
    return "\n\n".join(blocks)


def build_messages(question: str, history: list[dict] | None, sections: list[dict]) -> list[dict]:
    system = f"{SYSTEM_PROMPT}\n\nSources:\n\n{format_sources(sections)}"
    return [{"role": "system", "content": system}, *clean_history(history), {"role": "user", "content": question}]


def answer(question: str, history: list[dict] | None = None,
           run_id: str | None = None) -> tuple[list[dict], Iterator[str]]:
    """Returns (retrieved sections, streamed answer deltas).

    Callers accumulate the deltas and pass the running text through clean_citations();
    cleaning single deltas would miss citations split across them.
    """
    sections = retrieve(question, history, run_id=run_id)
    return sections, providers.chat_stream(build_messages(question, history, sections))


CITATION = re.compile(r"\[[^\]]*\]")
PLURAL = re.compile(r"^(.*?)Sections\s+(.+)$")


def clean_citations(text: str) -> str:
    """Strip sub-item markers inside [...] citations: "[RA 9262, Section 30(a)]" -> "[RA 9262, Section 30]".

    The model adds them despite the prompt, and they can be wrong (IRR Section 47 has two
    lettered lists). The section number is what retrieval verifies, so that is what we show.
    """
    def fix(m: re.Match) -> str:
        labels = []
        for label in m.group().strip("[]").split(";"):
            # Drop every "(a)", "(2)" and the "and"/"," left between them: "Section 30(a) and (g)" -> "Section 30".
            label = re.sub(r"\s*\([^)]*\)", "", label.strip())
            label = re.sub(r"(?:\s*(?:,|and|or|to|-))+\s*$", "", label)
            # "IRR, Rule IV, Sections 14 and 15" -> "IRR, Rule IV, Section 14; IRR, Rule IV, Section 15"
            if plural := PLURAL.match(label):
                labels += [f"{plural.group(1)}Section {n}" for n in re.findall(r"\d+", plural.group(2))]
            else:
                labels.append(label)
        return f"[{'; '.join(labels)}]"

    text = CITATION.sub(fix, text)
    # "[A][B]" or "[A] [B]" -> "[A; B]" so back-to-back citations read as one.
    return re.sub(r"\]\s?\[", "; ", text)


def drop_unretrieved_citations(text: str, sections: list[dict]) -> tuple[str, list[str]]:
    """Remove cited labels that are not among the retrieved sections. Returns (text, removed).

    Catches the model relabeling a cross-reference, e.g. IRR text saying "Section 7 of these
    Rules" cited as [RA 9262, Section 7], which is a different provision.
    """
    allowed = {s["citation"] for s in sections}
    removed: list[str] = []

    def fix(m: re.Match) -> str:
        labels = [l.strip() for l in m.group().strip("[]").split(";")]
        keep = [l for l in labels if l in allowed]
        removed.extend(l for l in labels if l not in allowed)
        return f"[{'; '.join(keep)}]" if keep else ""

    text = CITATION.sub(fix, text)
    text = re.sub(r"\s*;\s*([.,])", r"\1", text)  # "...IRR;." left behind by a removed citation
    return re.sub(r" +([.,;])", r"\1", text), list(dict.fromkeys(removed))


def cited_labels(text: str) -> list[str]:
    """Every label cited in an answer, e.g. for checking citations against retrieved sections."""
    return [label.strip() for c in CITATION.findall(text) for label in c.strip("[]").split(";")]


def sources_list(sections: list[dict]) -> str:
    """Collapsed list under each answer: sections found by search, then crosswalk pairs."""
    def line(s):
        p1, p2 = s["pdf_pages"]
        pages = f"PDF p. {p1}" + (f"-{p2}" if p2 != p1 else "")
        heading = f": {s['heading']}" if s["heading"] else ""
        return f"<li><b>{s['citation']}</b>{heading} <small>({pages})</small></li>"

    found = [line(s) for s in sections if s["source"] == "search"]
    paired = [line(s) for s in sections if s["source"] == "crosswalk"]
    summary = f"{SOURCES_HEADER}: {len(found)} found" + (f", {len(paired)} paired" if paired else "")
    body = f"<p>Found by search</p><ul>{''.join(found)}</ul>"
    if paired:
        body += f"<p>Paired law/IRR sections (added from the crosswalk)</p><ul>{''.join(paired)}</ul>"
    return f"<details><summary>{summary}</summary>{body}</details>"
