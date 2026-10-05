# Design: RA 9262 RAG Assistant

Status: approved, 2026-10-01. Amended 2026-10-02: vector storage moved from a numpy file to pgvector on Supabase (S5).

## 1. Goal and scope

A chat assistant that answers questions about RA 9262 (Anti-VAWC Act of 2004) and its 2004 IRR, citing the exact section for every claim, and refusing when the documents do not cover the question. Weekend-sized portfolio project. Python only.

Non-goals for v1: user accounts, chat logging, other statutes, jurisprudence, a custom frontend. These sit in the v1.1 roadmap.

## 2. Data flow

```
PDF ──extract.py──> data/clean/law.md, irr.md   (run once; reviewed by hand; source of truth)
                         │
                    chunk.py [flags]  ──> data/chunks.json
                         │
                    embed.py          ──> Supabase Postgres + pgvector: tables runs, chunks (tagged by run_id)
                         │
   user ──> app.py ──> rag.py: rewrite (follow-ups only) -> SQL vector search -> pair law/IRR -> answer (stream)
                         │
   eval/run_eval.py ─────┤──> eval/results.md
   eval/evaluator.py ────┘    (Gradio dashboard, same functions)
```

## 3. Modules

| File | Job | Main functions |
|---|---|---|
| `extract.py` | PDF to clean markdown. Refuses to overwrite without `--force`. | done |
| `chunk.py` | Markdown to chunks. Section boundaries are fixed; flags change splitting only. | `load_sections()`, `make_chunks(settings)` |
| `embed.py` | Embeds chunks and writes them to Postgres under a new `run_id`, with the run's chunk settings and embedding model in `runs`. | `embed_chunks()`, `list_runs()`, `delete_run()` |
| `db.py` | Connection, schema setup, vector search query. | `connect()`, `init_schema()`, `search(vector, run_id, k)` |
| `providers.py` | Thin wrappers so models swap by env var. | `embed(texts)`, `chat(messages, model)`, `chat_stream(messages, model)` |
| `rag.py` | All retrieval and answer logic. Imported by app and eval. | `rewrite_query()`, `retrieve()`, `answer()` |
| `app.py` | UI only: Gradio ChatInterface, auth, rate limit, disclaimer. | `respond()` |
| `eval/metrics.py` | Pure metric functions, no API calls. | `mrr()`, `ndcg()`, `recall_at_k()`, `precision_at_k()` |
| `eval/run_eval.py` | CLI: retrieval eval, answer eval, `--sweep`. Writes `results.md`. | |
| `eval/evaluator.py` | Gradio dashboard over the same functions. | |

Dependencies: `openai`, `psycopg[binary]` and `pgvector` (Postgres driver and vector type adapter), `numpy`, `gradio` (pinned), `python-dotenv`, `pandas` (dashboard charts). A plain Postgres driver is used instead of the Supabase client so the code runs against any Postgres with pgvector. The OpenAI SDK is used directly instead of LangChain or LiteLLM because four small wrapper functions cover what this project needs.

## 4. Chunks

- One chunk per section. A section longer than `--split-words` is split at sub-item boundaries (a), (b), ... and every part keeps the parent section id.
- Chunk fields: `id`, `document` (law or irr), `rule` (IRR only), `section`, `heading`, `citation`, `pdf_pages`, `part`, `parts`, `text`, `embed_text` (text plus the optional heading prefix; this is what gets embedded). `chunks.json` also records the settings that produced it.
- Citation labels: `RA 9262, Section 5` and `IRR, Rule IV, Section 12`.
- Tunable flags: `--split-words`, `--overlap` (sub-items shared between parts), `--heading-prefix/--no-heading-prefix` (citation and heading prepended to the embedded text).
- `data/chunks.json` stays in git so chunk text is reviewable. The database holds the same chunks plus vectors.
- `rag.py` refuses to start if `ACTIVE_RUN_ID` is missing from `runs` or was embedded with a different `EMBED_MODEL`, so a mismatch fails loudly instead of returning wrong sections.

## 5. Retrieval and answering

1. **Query.** First message: searched as typed. Follow-ups: an LLM rewrites the question into one short search query using the last few turns. Both the original and the rewritten query are searched and the results merged. (Donner's `pro_implementation` forgets to pass history to the rewriter; ours passes it.)
2. **Search.** pgvector cosine distance (`ORDER BY embedding <=> query`) over the active run's chunks; top K (default 5). Parts of the same section are merged before counting toward K.
3. **Law and IRR pairing.** A hand-verified map in `data/crosswalk.csv` attaches the paired section (outside K). See D1.
4. **Answer.** Citations are section-level only; `clean_citations()` strips sub-item markers like `(a)` because the model adds them despite the prompt and they can be wrong (IRR Section 47 has two lettered lists). System prompt rules: answer only from retrieved chunks; cite every claim; law first, IRR as implementing detail; say when a question falls outside the relationships the Act covers; no outside legal knowledge; no invented contact details. Streamed.
5. **Safety text.** A visible "not legal advice" note. If the user describes immediate danger, the answer tells them to contact local emergency services and shows `EMERGENCY_CONTACT_TEXT`, set by the owner: "call 911" (Philippine national emergency hotline, verified by the owner 2026-10-03). The model gives no other contact details.
6. **Privacy.** The app code does not log or store user messages. (The model provider and the hosting platform have their own retention; the README says so.)

## 6. Deployment

- Hugging Face Space, Gradio SDK, `requirements.txt` with pinned versions.
- Database: Supabase free project with the `vector` extension. Space connects with `DATABASE_URL` from Space secrets. To verify at deploy time: whether the Space can reach Supabase's direct connection (it may be IPv6 only) or needs the pooler connection string.
- Keep-alive: Supabase pauses free projects after 7 days of low activity. A weekly GitHub Actions job runs one read query against `chunks` so the demo stays up. To verify: that this query counts as activity.
- Gradio `auth` with a shared passcode from Space secrets, optional (S7). The deployed portfolio Space runs without it.
- Best-effort in-memory rate limits: 30/hour per session, 100/hour and 300/day across all users. Limits documented: resets on restart, not per person, per-session limit bypassable with new sessions. The OpenAI spending cap is the hard limit.
- README reminds the owner to set a spending cap on the API key, and documents Space sleep behavior after checking current Hugging Face docs.

## 7. Evaluation

**Test set:** `eval/questions.csv`

| Column | Meaning |
|---|---|
| `question` | The question |
| `prior_question` | Empty, or turn 1 for a two-turn follow-up test |
| `expected_sections` | Citation labels, `;` separated. Labels, not chunk ids, so tests survive re-chunking. |
| `reference_answer` | Drafted from the source chunk; checked by hand |
| `type` | `law_only`, `irr_only`, `both`, `not_in_docs`, `follow_up` |
| `verified` | Empty until checked by hand |

About 40 to 50 single-turn rows, about 7 `not_in_docs` rows (flagged for the owner to confirm they really are absent), 3 to 5 `follow_up` rows. Answerable questions are generated from a specific chunk, never from memory.

**Retrieval metrics** (no LLM, cheap, run on every sweep), matching on section labels:

| Metric | Meaning |
|---|---|
| MRR | 1 / rank of the first expected section; averaged |
| nDCG@K | Rewards expected sections ranked higher |
| Recall@K (K = 3, 5, 20) | Share of expected sections in the top K |
| Precision@K | Share of the top K that are expected. Max is 1/K for single-section questions (0.20 at K = 5); read it as relative, not absolute. |
| Both sources | For `both` rows: law and IRR section both in the top K |

**Answer metrics** (LLM judge, run on the chosen setting only): accuracy, completeness, relevance (1 to 5) against `reference_answer`; refusal rate on `not_in_docs`. Citation check (no LLM): share of answers whose cited labels are all among the retrieved sections.

**Sweep storage:** each chunk setting is embedded under its own `run_id`, so settings are compared side by side without overwriting each other. `embed.py --delete-run` removes old runs.

**Reporting:** raw counts next to every percentage (e.g. 41/45). With about 45 questions, one question is about 2 points, so differences under about 5 points are treated as ties. Every report lists missed questions with expected vs retrieved sections.

**Dashboard:** layout copied from Donner's `evaluator.py`: retrieval and answer sections, each with a run button, metric cards, a bar chart by question type, plus a missed-questions table.

## 8. Decisions

| # | Decision | Status |
|---|---|---|
| S1 | Reviewed markdown, not the PDF, is the chunking source | Settled |
| S2 | Eval dashboard with MRR, nDCG, Recall@K, Precision@K, LLM judge | Settled |
| S3 | Follow-ups: LLM query rewrite with history; search original + rewritten | Settled |
| S4 | No reranker. Add one behind a flag only if Recall@20 beats Recall@5 by more than about 10 points | Settled |
| S5 | Vector storage: pgvector on Supabase (replaces the numpy file; job-market reasons, see below) | Settled 2026-10-02 |
| S6 | Embedding model: text-embedding-3-large (spec said 3-small); 3-small kept as a comparison run in the eval | Settled 2026-10-03 |
| S7 | No passcode on the deployed Space (spec asked for one); bounded by global rate caps and the OpenAI spending cap | Settled 2026-10-03 |
| D1 | How to keep law and IRR sections together | Settled: Hand-verified law-to-IRR map |
| D2 | How refusals are decided | Settled: Model decides; measured in eval |
| D3 | Where sources appear in the chat UI | Settled: "Retrieved sections" list after each answer |
| D4 | Which model judges answers | Settled: Stronger judge model than answer model |

**S5. Vector storage.**
The original spec said no database. Changed by the owner on 2026-10-02 for portfolio value: of 15 US remote AI job postings checked that day, 4 named specific vector databases, and pgvector, Pinecone and Weaviate tied at 3 mentions each (small sample). pgvector chosen because it is Postgres (shows SQL too), the owner already uses Supabase, and it is free. At about 130 chunks per run, search is an exact scan with no ANN index; the README says so rather than implying index tuning.

**S6. Embedding model.**
The spec named text-embedding-3-small. Changed by the owner on 2026-10-03 to text-embedding-3-large. It returns 3,072-dimension vectors; pgvector's ANN indexes on the `vector` type stop at 2,000 dimensions, which does not matter here because search is an exact scan. The eval embeds both models as separate runs and reports both. The README claims large is better only if the measured numbers show it.

**S7. Open access.**
The spec asked for a shared passcode. Changed by the owner on 2026-10-03: a login screen stops recruiters, the main audience. The passcode stays supported (set `APP_PASSCODE` to turn it back on). Cost is bounded by a global cap of 100 questions per hour and 300 per day, plus the monthly spending cap on the OpenAI key.

**D1. Law and IRR pairing.**
Option A: a hand-verified map `data/crosswalk.csv` (e.g. law Section 9 to IRR Section 12). When one side is retrieved, its pair is attached (counted outside K). Option B: no map; rely on search returning both. Decided: A. Plain search is not guaranteed to return both, and the spec requires keeping both. The map is drafted from matching headings and every row is verified by the owner (about 15 minutes).

**D2. Refusals.**
Option A: the model decides from the retrieved chunks, per the system prompt. Option B: a similarity cutoff blocks weak matches before the model is called. Decided: A, measure the refusal rate in eval, add B only if refusals are missed.

**D3. Sources in the UI.**
Option A: after the streamed answer, the app appends a short "Retrieved sections" list built from the retrieved chunks. Option B: a side panel like Donner's `app.py`, which needs `gr.Blocks` instead of `ChatInterface`. Decided: A, it stays within ChatInterface as the spec requires.

**D4. Judge model.**
Option A: a stronger model than the answer model (e.g. gpt-4.1 judging gpt-4.1-mini), set by `JUDGE_MODEL`. Option B: the same model. Decided: A. Donner's setup has the same small model grade itself. `results.md` names the judge model either way.

## 9. Configuration

| Env var | Default | Used by |
|---|---|---|
| `OPENAI_API_KEY` | none | providers |
| `EMBED_MODEL` | `text-embedding-3-large` (S6) | embed, rag |
| `CHAT_MODEL` | `gpt-4.1-mini` | rag |
| `REWRITE_MODEL` | same as `CHAT_MODEL` | rag |
| `JUDGE_MODEL` | `gpt-4.1` | eval |
| `DATABASE_URL` | none | db (embed, rag, eval) |
| `ACTIVE_RUN_ID` | none (app refuses to start without it) | rag |
| `APP_PASSCODE` | empty = no login (portfolio mode, S7) | app |
| `RATE_LIMIT_PER_HOUR` | 30 (per browser session) | app |
| `RATE_LIMIT_GLOBAL_PER_HOUR` | 100 (all users) | app |
| `RATE_LIMIT_GLOBAL_PER_DAY` | 300 (all users) | app |
