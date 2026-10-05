---
title: RA 9262 Assistant
emoji: ⚖️
colorFrom: gray
colorTo: red
sdk: gradio
sdk_version: 6.29.1
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: RAG assistant for RA 9262 and its 2004 IRR, with citations
---

# RA 9262 Assistant

A retrieval-augmented chat assistant that answers questions about Republic Act No. 9262 (Anti-Violence Against Women and Their Children Act of 2004) and its Implementing Rules and Regulations. Every claim in an answer cites the section it comes from. If the two documents do not cover a question, the assistant says "I cannot find this in RA 9262 or the IRR."

Live demo: https://huggingface.co/spaces/mcglenn07/VAWC-RAG

This is legal information, not legal advice.

## Scope

- Source: one PDF containing RA 9262 (50 sections) and its IRR (67 sections in 13 rules).
- The IRR is the 2004 original. Later amendments, other laws and court decisions are not included.
- Excluded from the PDF: the Joint IAC-VAWC/IRR Committee Resolution and the image-only appendix forms (PDF pages 94 to 99).

## How it works

```
PDF -> extract.py -> data/clean/law.md, irr.md   (reviewed by hand)
    -> chunk.py   -> data/chunks.json             (134 chunks)
    -> embed.py   -> Postgres + pgvector (Supabase), one run per chunk setting
question -> rag.py: rewrite (follow-ups only) -> vector search -> law/IRR pairing
         -> gpt-4.1-mini answer with citations -> citation check -> app.py (Gradio)
```

1. `extract.py` converts the PDF to markdown with `pdftotext -layout`, rejoins wrapped lines, inlines 3 footnotes and checks that exactly 50 law sections, 67 IRR sections and 13 rules were found.
2. `chunk.py` makes one chunk per section. Sections over 400 words are split between sub-items, and every part keeps its section label. 117 sections become 134 chunks. The citation and heading are prepended to the text that gets embedded.
3. `embed.py` embeds the chunks with `text-embedding-3-large` and stores them in Postgres with pgvector, tagged with a run id so different chunk settings can be compared.
4. `rag.py` handles each question:
   - Follow-up questions are rewritten into a standalone search query using the conversation so far. Both the original and the rewritten query are searched.
   - pgvector returns the closest chunks by cosine distance. Parts of the same section are merged, and the top 5 sections are kept.
   - A hand-verified crosswalk (`data/crosswalk.csv`, 59 law-to-IRR pairs) adds the matching law or IRR section.
   - `gpt-4.1-mini` answers from those sections only, citing each claim. The Act is treated as outranking the IRR.
   - Code normalizes citations (strips sub-item letters, expands "Sections 14 and 15") and removes any citation to a section that was not retrieved, with a visible note.
5. `app.py` is the Gradio UI. It holds no retrieval logic.

The OpenAI SDK and psycopg are called directly; there is no LangChain or LlamaIndex. See `DESIGN.md` for every design decision and the reason for it, and `docs/HOW_IT_WORKS.md` for a code walkthrough.

## Evaluation

`eval/questions.csv` has 52 questions: 15 law-only, 12 IRR-only, 13 needing both documents, 5 two-turn follow-ups and 7 not covered by either document. The answerable questions and reference answers were drafted by gpt-4.1 from the text of one specific section each, and the expected section labels were set by the script, not by the model.

Results from `eval/results.md` (generated 2026-10-03), for the deployed setup (`text-embedding-3-large`, 400-word split, no overlap, heading prefix):

| Metric | Result |
|---|---|
| Expected sections in top 3 | 91% (41/45) |
| Expected sections in top 5 | 93% (42/45) |
| Expected sections in top 20 | 96% (43/45) |
| MRR | 0.85 |
| nDCG@5 | 0.87 |
| Both law and IRR sent to the model, for `both` questions | 92% (12/13) |
| Judge scores, gpt-4.1 judging gpt-4.1-mini (1 to 5) | accuracy 4.84, completeness 4.84, relevance 4.87 |
| Not-covered questions refused | 100% (7/7) |
| Answerable questions refused | 2% (1/45) |
| Answers where every citation was a retrieved section | 98% (51/52) |

Comparison runs (same questions, top 5):

| Setting | Top 3 | Top 5 | MRR |
|---|---|---|---|
| Deployed: 3-large, 400 words, no overlap, heading prefix | 41/45 | 42/45 | 0.85 |
| text-embedding-3-small instead of 3-large | 38/45 | 41/45 | 0.82 |
| 250-word split | 42/45 | 43/45 | 0.87 |
| 250-word split, 1 line overlap | 41/45 | 42/45 | 0.86 |
| No heading prefix | 37/45 | 39/45 | 0.80 |

With 45 scored questions, one question is about 2 points, so differences under about 5 points are treated as ties. Read that way: the large embedding model ranked better at top 3 (41 vs 38), the heading prefix helped (42 vs 39 at top 5), and the 250-word split and overlap made no measurable difference. Top 20 beat top 5 by 1 question, so no reranker was added.

Limits of these numbers:

- The test set has not been reviewed by a person yet (`verified` column is empty).
- Questions drafted from a section's own text may share its wording, which can make retrieval look better than it would on real user questions.
- Answer scores come from an LLM judge, not a lawyer.
- MRR moved by about 0.01 between identical runs because follow-up rewrites come from an LLM.

Run it yourself: `python eval/run_eval.py` (retrieval), `python eval/run_eval.py --answers` (answers and judge), `python eval/evaluator.py` (dashboard).

## Known failures

1. "Can my ex-boyfriend be covered by this?" asked after a question about Barangay Protection Orders. "This" points the search at protection order sections, so the definitions in RA 9262 Section 3 are not retrieved. The model then described coverage from its own knowledge; the citation check removed the unsupported citations, but the text remained.
2. "How long do I have to file a criminal case?" did not retrieve the prescriptive period sections (RA 9262 Section 24, IRR Section 9), so the model refused. Asking with the word "prescriptive" retrieves them.
3. A question about Department of Health duties did not reach IRR Section 61, a 2,000-word section split into 5 parts. The 250-word split fixes this one question.
4. The model sometimes adds sub-item letters to citations, such as "Section 47(i)", and they can be wrong because some sections have more than one lettered list. Citations are therefore shown at section level only.

## Running it

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env   # fill in OPENAI_API_KEY and DATABASE_URL
.venv/bin/python chunk.py
.venv/bin/python embed.py          # prints a run id; put it in ACTIVE_RUN_ID
.venv/bin/python app.py
```

`DATABASE_URL` must be the Supabase Session pooler string; the direct connection host is IPv6 only and did not resolve from a home network.

## Deployment notes

- Hosted on a Hugging Face Space (Gradio SDK, ZeroGPU hardware, the free option for this account). The app never uses the GPU; a stub `@spaces.GPU` function is defined because ZeroGPU Spaces expect one.
- Per Hugging Face's documentation, Spaces on free hardware sleep after 48 hours without visitors. The next visitor wakes it, with a slower first load.
- Secrets: `OPENAI_API_KEY`, `DATABASE_URL`. Variable: `ACTIVE_RUN_ID`. Add keys as secrets, not variables: Space variables are public.
- There is no login. `APP_PASSCODE` turns on a shared passcode if set.
- Rate limits are in memory and best-effort: 30 questions per hour per browser session, 100 per hour and 300 per day across all users. They reset when the Space restarts, and a new browser session gets a new per-session allowance. The real cost limit is the monthly spending cap on the OpenAI key; set one before deploying.
- Supabase pauses free projects after 7 days of low activity. A keep-alive job is planned, not built.

## Privacy

The app does not log or store messages. OpenAI (model provider) and Hugging Face (host) apply their own data policies.

## Roadmap (v1.1)

- Review and mark every row of the test set as verified, then rerun the eval.
- Fix the follow-up coverage miss, for example by searching the topic and the concept as two queries, and measure it.
- Weekly GitHub Actions keep-alive for Supabase.
- Add later amendments and relevant Supreme Court decisions as separate, clearly labeled sources.
- Collect real (anonymized, opt-in) questions to replace part of the generated test set.
