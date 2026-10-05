"""Turn the reviewed markdown (data/clean/*.md) into retrieval chunks.

Section boundaries are fixed: one section = one chunk unless it is longer than
--split-words. Long sections are split between lines; every line in the markdown
starts at a sub-item ((a), 1., etc.) or a sentence, so splits land on sub-item
boundaries. Every part keeps the parent section's id.

Usage: python chunk.py [--split-words 400] [--overlap 0] [--no-heading-prefix] [--out data/chunks.json]
"""

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).parent
CLEAN = ROOT / "data" / "clean"
DEFAULT_OUT = ROOT / "data" / "chunks.json"

EXPECTED_LAW_SECTIONS = 50
EXPECTED_IRR_SECTIONS = 67
EXPECTED_RULES = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII"]

RULE_LINE = re.compile(r"^## RULE ([IVX]+) \| (.*)$")
SECTION_LINE = re.compile(r"^### (?:SEC\.|Section) (\d+) \| (.*)$")
PAGES_LINE = re.compile(r"^<!-- pdf_pages: (\d+)-(\d+) -->$")
NO_TITLE = "(no title in source)"


@dataclass
class Settings:
    split_words: int = 400
    overlap: int = 0  # lines from the end of one part repeated at the start of the next
    heading_prefix: bool = True  # prepend citation and heading to the embedded text


def load_sections(document: str) -> list[dict]:
    """Parse law.md or irr.md into sections: number, rule, rule_title, heading, pdf_pages, lines."""
    sections, rule, rule_title = [], None, None
    for raw in (CLEAN / f"{document}.md").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if m := RULE_LINE.match(line):
            rule, rule_title = m.group(1), m.group(2)
        elif m := SECTION_LINE.match(line):
            heading = m.group(2).strip()
            sections.append({
                "number": int(m.group(1)),
                "rule": rule,
                "rule_title": rule_title,
                "heading": None if heading == NO_TITLE else heading,
                "pdf_pages": None,
                "lines": [],
            })
        elif m := PAGES_LINE.match(line):
            sections[-1]["pdf_pages"] = [int(m.group(1)), int(m.group(2))]
        elif line and not line.startswith(("#", "<!--")) and sections:
            sections[-1]["lines"].append(line)
    return sections


def citation(document: str, section: dict) -> str:
    if document == "law":
        return f"RA 9262, Section {section['number']}"
    return f"IRR, Rule {section['rule']}, Section {section['number']}"


def can_break_after(line: str) -> bool:
    # Don't split right after a lead-in ("...includes the following:") or a bare
    # sub-heading ("a) Commission on Human Rights (CHR)"); the next line belongs to it.
    return line.endswith((".", ";", "; and", "; or", ".”"))


def split_lines(lines: list[str], settings: Settings) -> list[list[str]]:
    """Greedy split at line boundaries. A single line longer than split_words stays whole.

    Why split at all: an embedding is one vector for the whole chunk. A 2,000-word section that
    lists duties of 12 agencies gets one "average" meaning, so a question about one agency
    matches it weakly. Smaller parts give each topic its own vector. Why not split smaller:
    tiny chunks lose context ("a) Commission on Human Rights" alone means little).
    """
    words = lambda ls: sum(len(l.split()) for l in ls)
    if words(lines) <= settings.split_words:
        return [lines]
    parts, current = [], []
    for line in lines:
        too_long = words(current) + len(line.split()) > settings.split_words
        if current and too_long and can_break_after(current[-1]):
            parts.append(current)
            current = []
        current.append(line)
    parts.append(current)
    if settings.overlap:
        parts = [parts[0]] + [prev[-settings.overlap :] + part for prev, part in zip(parts, parts[1:])]
    return parts


def make_chunks(settings: Settings) -> list[dict]:
    chunks = []
    for document in ("law", "irr"):
        for s in load_sections(document):
            cite = citation(document, s)
            section_id = f"law-s{s['number']}" if document == "law" else f"irr-r{s['rule']}-s{s['number']}"
            parts = split_lines(s["lines"], settings)
            for n, part in enumerate(parts, start=1):
                text = "\n".join(part)
                prefix = cite + (f": {s['heading']}" if s["heading"] else "")
                if len(parts) > 1:
                    prefix += f" (part {n} of {len(parts)})"
                chunks.append({
                    "id": section_id,
                    "document": document,
                    "rule": s["rule"],
                    "section": s["number"],
                    "heading": s["heading"],
                    "citation": cite,
                    "pdf_pages": s["pdf_pages"],  # section range; parts are not mapped to pages
                    "part": n,
                    "parts": len(parts),
                    "text": text,
                    # What gets embedded. The "RA 9262, Section 14: Barangay Protection Orders"
                    # prefix tells the vector what the text is about; removing it cost 3 of 45
                    # questions in the eval.
                    "embed_text": f"{prefix}\n\n{text}" if settings.heading_prefix else text,
                })
    return chunks


def check(chunks: list[dict], settings: Settings) -> None:
    """Fail loudly on missing, duplicated or lost content."""
    for document, expected in (("law", EXPECTED_LAW_SECTIONS), ("irr", EXPECTED_IRR_SECTIONS)):
        numbers = list(dict.fromkeys(c["section"] for c in chunks if c["document"] == document))
        assert numbers == list(range(1, expected + 1)), f"{document}: gap or duplicate: {numbers}"
    rules = list(dict.fromkeys(c["rule"] for c in chunks if c["document"] == "irr"))
    assert rules == EXPECTED_RULES, f"Rules found: {rules}"
    keys = [(c["id"], c["part"]) for c in chunks]
    assert len(keys) == len(set(keys)), "Duplicate (id, part)"
    assert all(c["text"].strip() for c in chunks), "Empty chunk"
    assert all(c["pdf_pages"] for c in chunks), "Section without a pdf_pages comment"

    # Rejoining the parts (minus overlap) must give back every section line, in order.
    for document in ("law", "irr"):
        for s in load_sections(document):
            parts = [c["text"].split("\n") for c in chunks
                     if c["document"] == document and c["section"] == s["number"]]
            rejoined = parts[0] + [l for p in parts[1:] for l in p[settings.overlap :]]
            assert rejoined == s["lines"], f"{document} section {s['number']}: text lost or reordered"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-words", type=int, default=Settings.split_words)
    parser.add_argument("--overlap", type=int, default=Settings.overlap)
    parser.add_argument("--heading-prefix", action=argparse.BooleanOptionalAction, default=Settings.heading_prefix)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    settings = Settings(args.split_words, args.overlap, args.heading_prefix)

    chunks = make_chunks(settings)
    check(chunks, settings)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"settings": asdict(settings), "chunks": chunks},
                                   ensure_ascii=False, indent=1), encoding="utf-8")

    words = lambda c: len(c["text"].split())
    split = sorted({c["citation"] for c in chunks if c["parts"] > 1}, key=lambda x: (x[:3], x))
    longest = max(chunks, key=words)
    print(f"Settings: {asdict(settings)}")
    print(f"Sections: law {EXPECTED_LAW_SECTIONS}, IRR {EXPECTED_IRR_SECTIONS} in {len(EXPECTED_RULES)} rules. Checks passed.")
    print(f"Chunks: {len(chunks)} ({sum(c['document'] == 'law' for c in chunks)} law, "
          f"{sum(c['document'] == 'irr' for c in chunks)} IRR)")
    print(f"Sections split into parts: {len(split)}: {'; '.join(split) or 'none'}")
    print(f"Longest chunk: {longest['citation']} part {longest['part']}, {words(longest)} words")
    print(f"Wrote {args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out}")


if __name__ == "__main__":
    main()
