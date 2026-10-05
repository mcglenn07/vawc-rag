"""Extract RA 9262 and its IRR from the PDF into reviewed markdown files.

Run once. After you review and hand-fix data/clean/*.md, those files are the
source of truth; chunk.py reads them and never touches the PDF.

Usage: python extract.py [--force]
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
PDF = ROOT / "data" / "raw" / "RA9262-and-its-IRR.pdf"
OUT_DIR = ROOT / "data" / "clean"

EXPECTED_LAW_SECTIONS = 50
EXPECTED_IRR_SECTIONS = 67
EXPECTED_RULES = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII"]

LAW_HEADER = re.compile(r"^(?:SECTION|SEC\.)\s+(\d+)\.\s*(.*)$")
IRR_HEADER = re.compile(r"^Section\s+(\d+)\.\s*(.*)$")
RULE_HEADER = re.compile(r"^RULE\s+([IVX]+)$")
LONE_NUMBER = re.compile(r"^\d+$")
# Sub-item markers that start a new line: (a) a) (1) 1. i) and "2   Peering" (missing period in source).
ITEM_MARKER = re.compile(r"^(\(?[a-z]{1,4}\)|\(\d{1,2}\)|\d{1,2}[.)]|\d\s{2,})\s")
# Title ends at ". -", ".-", " - ", ".–" or ". " before a capital letter. Earliest match wins.
TITLE_END = re.compile(r"\.\s*[-–]\s*|\s+[-–]\s+|\.\s+(?=[A-Z])")
# Footnote markers in IRR Section 8, e.g. "prision mayor 1 ;" and "arresto mayor3 ."
FOOTNOTE_MARKER = re.compile(r"(mayor|correctional)\s?([1-9])\s([;.])")


def pdf_pages() -> list[list[str]]:
    # pdftotext -layout keeps line-end hyphens in compounds like "victim-\nsurvivor";
    # the default mode deletes them and produces "victimsurvivor".
    text = subprocess.run(
        ["pdftotext", "-layout", str(PDF), "-"], capture_output=True, text=True, check=True
    ).stdout
    pages = text.split("\f")
    if pages and not pages[-1].strip():
        pages = pages[:-1]
    return [p.splitlines() for p in pages]


def clean_page(lines: list[str]) -> tuple[list[str], dict[str, str]]:
    """Drop the page number line and pull out footnotes. Returns (body lines, footnotes)."""
    number_rows = [i for i, l in enumerate(lines) if LONE_NUMBER.match(l.strip())]
    if not number_rows:
        return lines, {}
    page_number_row = number_rows[-1]
    footnote_rows = [i for i in number_rows[:-1] if int(lines[i].strip()) < 10]
    body_end = footnote_rows[0] if footnote_rows else page_number_row
    footnotes = {}
    for n, start in enumerate(footnote_rows):
        end = footnote_rows[n + 1] if n + 1 < len(footnote_rows) else page_number_row
        text = " ".join(l.strip() for l in lines[start + 1 : end] if l.strip())
        footnotes[lines[start].strip()] = text
    body = lines[:body_end] + lines[page_number_row + 1 :]
    return body, footnotes


def document_lines() -> tuple[list[tuple[str, int]], dict[str, str]]:
    """All non-blank lines as (stripped text, pdf page number), plus all footnotes."""
    out, footnotes = [], {}
    for page_no, lines in enumerate(pdf_pages(), start=1):
        body, notes = clean_page(lines)
        footnotes.update(notes)
        for line in body:
            s = line.strip()
            if s and s != ".":
                out.append((s, page_no))
    return out, footnotes


def reflow(lines: list[tuple[str, int]]) -> list[str]:
    """Join wrapped lines back into logical lines. Page breaks disappear here."""
    logical: list[str] = []
    for text, _ in lines:
        if not logical:
            logical.append(text)
            continue
        prev = logical[-1]
        new_line = ITEM_MARKER.match(text) or (prev[-1] in ".:;" and text[0].isupper())
        if new_line:
            logical.append(text)
        elif re.search(r"[A-Za-z]-$", prev):
            logical[-1] = prev + text
        else:
            logical[-1] = prev + " " + text
    return [re.sub(r"\s{2,}", " ", l) for l in logical]


def split_title(first_line: str) -> tuple[str, str]:
    m = TITLE_END.search(first_line)
    if not m:
        # Title alone on its line with the text starting on the next (IRR Sections 12 and 14).
        if first_line.endswith(".") and len(first_line) < 100:
            return first_line[:-1].strip(), ""
        return "", first_line
    return first_line[: m.start()].strip(), first_line[m.end() :].strip()


def find(lines: list[tuple[str, int]], pattern: str, start: int = 0) -> int:
    for i in range(start, len(lines)):
        if re.match(pattern, lines[i][0]):
            return i
    sys.exit(f"Could not find line matching {pattern!r}")


def parse_sections(lines, header_re, expected_count, rules=False):
    """Split lines into sections. Only accept a header whose number is the next expected one,
    which ignores in-text references like "Section 13 hereof"."""
    sections, current, rule = [], None, None
    i = 0
    while i < len(lines):
        text, page = lines[i]
        if rules and (m := RULE_HEADER.match(text)):
            title_parts = []
            i += 1
            while i < len(lines) and lines[i][0].isupper() and not header_re.match(lines[i][0]):
                title_parts.append(lines[i][0])
                i += 1
            rule = {"rule": m.group(1), "title": " ".join(title_parts)}
            continue
        m = header_re.match(text)
        next_number = len(sections) + 1
        if m and int(m.group(1)) == next_number:
            current = {"number": next_number, "rule": rule, "lines": []}
            sections.append(current)
            rest = m.group(2)
            # Heading on its own line (IRR Section 48): pull the next line up.
            if not rest and i + 1 < len(lines):
                i += 1
                rest = lines[i][0]
            current["lines"].append((rest, page))
        elif current is not None:
            current["lines"].append((text, page))
        i += 1

    numbers = [s["number"] for s in sections]
    assert numbers == list(range(1, expected_count + 1)), f"Section gap or duplicate: {numbers}"
    for s in sections:
        pages = [p for _, p in s["lines"]]
        s["pdf_pages"] = (min(pages), max(pages))
        logical = reflow(s["lines"])
        s["heading"], first_body = split_title(logical[0])
        s["body"] = [first_body] + logical[1:] if first_body else logical[1:]
        del s["lines"]
    return sections


def inline_footnotes(sections, footnotes):
    count = 0
    for s in sections:
        for n, line in enumerate(s["body"]):

            def repl(m):
                nonlocal count
                count += 1
                return f"{m.group(1)} [Footnote {m.group(2)}: {footnotes[m.group(2)]}]{m.group(3)}"

            s["body"][n] = FOOTNOTE_MARKER.sub(repl, line)
    assert count == len(footnotes), f"Inlined {count} footnotes, found {len(footnotes)}"
    return count


def to_markdown(title, sections, label):
    out = [f"# {title}", "", f"<!-- Extracted from {PDF.relative_to(ROOT)} by extract.py. "
           "Hand fixes are allowed; keep the ### header lines and page comments intact. -->", ""]
    last_rule = None
    for s in sections:
        if s["rule"] and s["rule"]["rule"] != last_rule:
            last_rule = s["rule"]["rule"]
            out += [f"## RULE {last_rule} | {s['rule']['title']}", ""]
        heading = s["heading"] or "(no title in source)"
        out.append(f"### {label} {s['number']} | {heading}")
        out.append(f"<!-- pdf_pages: {s['pdf_pages'][0]}-{s['pdf_pages'][1]} -->")
        out += s["body"]
        out.append("")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="overwrite reviewed markdown files")
    args = parser.parse_args()

    law_path, irr_path = OUT_DIR / "law.md", OUT_DIR / "irr.md"
    if (law_path.exists() or irr_path.exists()) and not args.force:
        sys.exit("data/clean/*.md already exists and may contain hand fixes. Use --force to overwrite.")

    lines, footnotes = document_lines()

    # Law: "SECTION 1. Short Title" up to "Approved." (enacting clause and title page excluded).
    law_start = find(lines, r"^SECTION 1\. Short Title")
    law_end = find(lines, r"^Approved\.$", law_start)
    # The Joint IAC-VAWC / IRR Committee Resolution sits between law_end and "RULE I"; excluded.
    irr_start = find(lines, r"^RULE I$", law_end)
    # IRR ends before the appendices (image-only forms, printed pages 89-94 = PDF pages 94-99).
    irr_end = find(lines, r"^APPENDICES$", irr_start)

    law = parse_sections(lines[law_start:law_end], LAW_HEADER, EXPECTED_LAW_SECTIONS)
    irr = parse_sections(lines[irr_start:irr_end], IRR_HEADER, EXPECTED_IRR_SECTIONS, rules=True)
    n_footnotes = inline_footnotes(irr, footnotes)

    rules = list(dict.fromkeys(s["rule"]["rule"] for s in irr))
    assert rules == EXPECTED_RULES, f"Rules found: {rules}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    law_path.write_text(to_markdown(
        "Republic Act No. 9262: Anti-Violence Against Women and Their Children Act of 2004",
        law, "SEC."), encoding="utf-8")
    irr_path.write_text(to_markdown(
        "Implementing Rules and Regulations of RA 9262 (2004)", irr, "Section"), encoding="utf-8")

    print(f"Law: {len(law)} sections, pdf pages {law[0]['pdf_pages'][0]}-{law[-1]['pdf_pages'][1]}")
    print(f"IRR: {len(irr)} sections in {len(rules)} rules, "
          f"pdf pages {irr[0]['pdf_pages'][0]}-{irr[-1]['pdf_pages'][1]}")
    print(f"Footnotes inlined: {n_footnotes}")
    untitled = [f"{d} {s['number']}" for d, secs in (("law", law), ("irr", irr)) for s in secs if not s["heading"]]
    print(f"Sections with no title in source: {untitled or 'none'}")
    print(f"Wrote {law_path.relative_to(ROOT)} and {irr_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
