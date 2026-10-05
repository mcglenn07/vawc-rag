# How the code works

A walkthrough of the RA 9262 Assistant, in the order data moves through it. File and function names match the repository. Design reasons live in `DESIGN.md`; this file explains what the code does.

## 1. The big picture

There are two phases.

**Build time (run once, or once per experiment):**

```
PDF ──extract.py──> markdown ──chunk.py──> chunks.json ──embed.py──> Postgres + pgvector
```

**Question time (every message in the chat):**

```
question ──rag.retrieve──> top sections ──rag.answer──> streamed answer ──app.respond──> UI
```

Three models are involved, all from OpenAI, all called through `providers.py`:

| Model | Used for | Where |
|---|---|---|
| `text-embedding-3-large` | Turning chunks and questions into vectors | `embed.py`, `rag.retrieve` |
| `gpt-4.1-mini` | Rewriting follow-up questions, writing answers | `rag.rewrite_query`, `rag.answer` |
| `gpt-4.1` | Drafting test questions, judging answers | `eval/` only |

Search itself is not a model. It is a SQL query in Postgres that compares vectors.

## 2. Build time

### 2.1 `extract.py`: PDF to reviewed markdown

The PDF has the Act (50 sections), a committee resolution, the IRR (67 sections in 13 rules) and scanned appendix forms. `extract.py` keeps only the Act and the IRR.

- `pdf_pages()` runs `pdftotext -layout` and splits on form feeds. Layout mode matters: the default mode deletes line-end hyphens, turning "victim-survivor" into "victimsurvivor".
- `clean_page()` removes the page number (the last line that is only digits) and pulls out footnotes (earlier lone digits under 10).
- `reflow()` joins wrapped lines back into logical lines. A new line starts at a sub-item marker such as `(a)`, `1.` or `a)`, or after a line ending in `.`, `:` or `;` when the next line starts with a capital.
- `parse_sections()` finds section headers. It only accepts a header whose number is the next one expected, which ignores cross-references like "Section 13 hereof" in the middle of the text. It then asserts the numbers are exactly 1 to N, so a missed or duplicated section stops the script.
- `inline_footnotes()` rewrites the three footnotes in IRR Section 8 (definitions of prision mayor and similar) into the text, so they get embedded with the penalty they explain.
- Output: `data/clean/law.md` and `irr.md`, one `### SEC. N | Heading` block per section with a `<!-- pdf_pages: a-b -->` comment.

After a hand review, these markdown files are the source of truth. `extract.py` refuses to overwrite them without `--force`, so a hand fix cannot be lost by rerunning it.

### 2.2 `chunk.py`: markdown to chunks

- `load_sections()` parses the markdown back into sections: number, rule, heading, page range and lines.
- `citation()` builds the label used everywhere else: `RA 9262, Section 5` or `IRR, Rule IV, Section 12`. Retrieval, the crosswalk, the citations in answers and the eval all match on this string.
- `split_lines()` leaves a section whole if it is under `--split-words` (default 400). Longer sections are cut between lines, which are always sub-item or sentence boundaries. `can_break_after()` refuses to cut right after a lead-in ending in `:` or a bare sub-heading such as "a) Commission on Human Rights (CHR)", because the next line belongs to it. `--overlap N` repeats the last N lines of one part at the start of the next.
- `make_chunks()` produces one record per part with `id` (the section id, shared by all parts), `part`, `parts`, metadata, `text` (what the model reads) and `embed_text` (what gets embedded: with `--heading-prefix`, the citation and heading are put in front of the text).
- `check()` asserts 50 law sections, 67 IRR sections, 13 rules, no duplicate `(id, part)`, and that rejoining the parts gives back every original line in order. Nothing can be dropped silently.

Default output: 134 chunks from 117 sections.

### 2.3 `embed.py` and `db.py`: chunks to Postgres

`db.py` creates two tables:

- `runs`: one row per embedding run, with the chunk settings and embedding model.
- `chunks`: every chunk, its metadata and its `embedding` (a pgvector `vector` column with no fixed size, so runs with different models can live side by side).

`embed.py` reads a `chunks.json`, embeds every `embed_text` in batches of 100, and inserts the run and its chunks in one transaction under a new `run_id` such as `20261003-081538-sw400-ov0-hp` (time, split words, overlap, heading prefix). The app serves whichever run the `ACTIVE_RUN_ID` setting names, so trying a new chunk setting never touches the live one.

There is no vector index. With about 130 rows per run, Postgres compares the question against every row, which is exact and fast enough.

## 3. Question time

### 3.1 `app.respond`: the entry point

Gradio calls `respond(message, history, request)` for each message.

1. `allow()` checks the in-memory rate limits (per browser session per hour, everyone per hour, everyone per day).
2. It shows "Searching RA 9262 and the IRR..." right away.
3. `rag.answer()` returns the retrieved sections and a stream of answer text.
4. As text streams in, each partial answer is passed through `rag.clean_citations()` and shown.
5. When the stream ends, `rag.drop_unretrieved_citations()` removes citations to sections that were not retrieved, a small note says how many were removed, and `rag.sources_list()` appends the collapsible "Retrieved sections" list.

`app.py` never logs or stores messages.

### 3.2 `rag.retrieve`: finding the sections

```python
retrieve(question, history, k=5, pair=True, run_id=None)
```

1. **Run check.** `active_run_id()` reads `ACTIVE_RUN_ID` and, the first time, confirms that run exists and was embedded with the current `EMBED_MODEL`. A vector from one model compared against vectors from another returns confident nonsense, so a mismatch stops the app at startup instead.
2. **Rewrite (follow-ups only).** `rewrite_query()` returns `None` on the first message. On later messages it sends the last few turns to gpt-4.1-mini with `REWRITE_PROMPT`, which asks for one short query in the statute's own wording ("ex-boyfriend" becomes "former dating relationship") and forbids adding facts. `clean_history()` strips the "Retrieved sections" HTML from earlier answers first.
3. **Embed.** The original question, plus the rewrite if there is one, are embedded in one API call with the run's own model (`run_embed_model()`).
4. **Search.** `db.search()` runs:
   ```sql
   select ..., embedding <=> %s as distance from chunks
   where run_id = %s order by distance limit %s
   ```
   `<=>` is pgvector's cosine distance; smaller is closer. It asks for `k * 4` chunks per query so that k distinct sections survive the next step.
5. **Merge.** `_search_and_pair()` keeps each chunk's best distance across the two queries, groups parts by section, and keeps the top k sections. Only the parts that matched are kept as text, so a 2,000-word section does not flood the prompt.
6. **Pair.** `crosswalk()` loads `data/crosswalk.csv` (only rows marked verified) into a two-way map. For each of the top k, its law or IRR counterpart is fetched whole with `db.get_sections()` and added after the top k, marked `source: "crosswalk"`. These do not count toward k.

All database work goes through `db.with_retry()`, which reuses one long-lived connection (`db.shared()`) and reconnects if it dropped. Opening a new connection to the Supabase pooler took 3 to 9 seconds in testing; a query on an open connection takes about half a second.

### 3.3 `rag.answer`: writing the answer

`build_messages()` assembles:

- `SYSTEM_PROMPT` with nine rules: use only the sources; cite every claim with the exact label; start with the refusal sentence if the sources do not answer; the Act outranks the IRR; do not extend coverage beyond the relationships the sources define; quote sparingly; no contact details except the emergency line; start with "If you are in immediate danger, call 911 now." when someone is in danger; be brief.
- The retrieved sections, formatted by `format_sources()` as `[Source n] label: heading (PDF pages a-b)` followed by the text.
- The last few turns of the conversation, then the question.

`providers.chat_stream()` streams the reply from gpt-4.1-mini at temperature 0.

### 3.4 Citation cleanup

Prompts alone did not keep citations clean in testing, so two functions enforce it in code:

- `clean_citations()` rewrites each `[...]` citation: removes sub-item markers (`Section 30(a) and (g)` becomes `Section 30`), expands plurals (`Sections 14 and 15` becomes two labels) and merges back-to-back brackets. Sub-items are removed because some sections have two separate lettered lists, and the model picked the wrong letter.
- `drop_unretrieved_citations()` removes any label not among the retrieved sections and returns the removed labels. This caught the model citing "RA 9262, Section 7" when the IRR text said "Section 7 of these Rules", a different provision.

## 4. Evaluation (`eval/`)

- `make_questions.py` drafts the test set. For each chosen section it gives gpt-4.1 that section's text only and asks for one question and a short reference answer. The script, not the model, sets `expected_sections`. The 7 not-covered questions and 5 follow-ups are written by hand in the file.
- `metrics.py` has four pure functions on lists of labels: `mrr`, `recall_at_k`, `precision_at_k`, `ndcg_at_k`.
- `run_eval.py`:
  - Retrieval: for every question it calls `rag.retrieve(..., k=20, pair=False)` on the chosen run and scores the ranking. For `both` questions it also checks that law and IRR sections both reach the model with pairing on.
  - Answers (`--answers`): it calls `rag.answer()`, applies the same citation cleanup as the app, records refusals and removed citations, and asks gpt-4.1 to score accuracy, completeness and relevance from 1 to 5 against the reference answer.
  - Results are saved to `eval/results/<run_id>.json`, and `eval/results.md` is rebuilt from all saved runs with raw counts next to every percentage.
- `evaluator.py` is a Gradio dashboard over the same functions: pick a run, see metric cards, charts by question type, missed questions, problem answers and a table comparing all runs. Selecting a run loads saved results; only the Run buttons call the API.

Because the eval scores section labels, not chunk ids, the same questions work for every chunk setting. That is what made the comparison runs in the README possible.

## 5. Configuration

| Setting | Default | Read by |
|---|---|---|
| `OPENAI_API_KEY` | none | `providers.py` |
| `DATABASE_URL` | none (Supabase Session pooler string) | `db.py` |
| `ACTIVE_RUN_ID` | none, app will not start without it | `rag.py` |
| `EMBED_MODEL` | `text-embedding-3-large` | `providers.py` |
| `CHAT_MODEL`, `REWRITE_MODEL` | `gpt-4.1-mini` | `providers.py` |
| `JUDGE_MODEL` | `gpt-4.1` | `providers.py` (eval only) |
| `APP_PASSCODE` | empty, no login | `app.py` |
| `RATE_LIMIT_PER_HOUR`, `RATE_LIMIT_GLOBAL_PER_HOUR`, `RATE_LIMIT_GLOBAL_PER_DAY` | 30, 100, 300 | `app.py` |

Every value is read with `.strip()`. A key pasted into a web form with a trailing newline broke the deployed app once, so whitespace is removed everywhere.
