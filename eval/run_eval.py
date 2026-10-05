"""Evaluate retrieval (cheap, no LLM except follow-up rewrites) and answers (LLM judge).

Results are saved per run in eval/results/<run_id>.json, and eval/results.md is rebuilt
from every saved run, so chunk settings and embedding models compare side by side.

Usage:
  python eval/run_eval.py                    # retrieval eval on ACTIVE_RUN_ID
  python eval/run_eval.py --run RUN_ID       # a specific stored run
  python eval/run_eval.py --all-runs         # every stored run (a sweep)
  python eval/run_eval.py --answers          # also answer every question and judge it
"""

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import db  # noqa: E402
import metrics  # noqa: E402
import providers  # noqa: E402
import rag  # noqa: E402

QUESTIONS = ROOT / "eval" / "questions.csv"
RESULTS_DIR = ROOT / "eval" / "results"
RESULTS_MD = ROOT / "eval" / "results.md"
K_VALUES = (3, 5, 20)
TYPES = ["law_only", "irr_only", "both", "follow_up", "not_in_docs"]

JUDGE_PROMPT = """You grade an answer from a legal-information assistant against a reference answer.
Both are about RA 9262 (Philippines) and its implementing rules. Score each from 1 to 5:
- accuracy: are the answer's statements correct according to the reference? Penalize anything \
that contradicts it. Extra correct detail is fine.
- completeness: does the answer cover the points in the reference?
- relevance: does the answer address the question asked, without padding?
Return JSON: {{"accuracy": n, "completeness": n, "relevance": n, "feedback": "one sentence"}}

Question: {question}
Reference answer: {reference}
Answer: {answer}"""


def load_questions() -> list[dict]:
    with QUESTIONS.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["expected"] = [e.strip() for e in r["expected_sections"].split(";") if e.strip()]
        r["history"] = ([{"role": "user", "content": r["prior_question"]}] if r["prior_question"] else None)
    return rows


def pct(n: float, d: int) -> str:
    return f"{100 * n / d:.0f}% ({n:g}/{d})" if d else "n/a"


# ---------- retrieval ----------

def retrieval_record(q: dict, run_id: str) -> dict:
    ranked = [s["citation"] for s in rag.retrieve(q["question"], q["history"], k=max(K_VALUES),
                                                   pair=False, run_id=run_id)]
    rec = {"question": q["question"], "type": q["type"], "expected": q["expected"], "top5": ranked[:5]}
    if q["type"] == "not_in_docs":
        return rec
    expected = set(q["expected"])
    rec["mrr"] = metrics.mrr(ranked, expected)
    for k in K_VALUES:
        rec[f"recall@{k}"] = metrics.recall_at_k(ranked, expected, k)
        rec[f"precision@{k}"] = metrics.precision_at_k(ranked, expected, k)
        rec[f"ndcg@{k}"] = metrics.ndcg_at_k(ranked, expected, k)
    if q["type"] == "both":
        # What the app actually sends the model: top 5 plus crosswalk pairs.
        sent = [s["citation"] for s in rag.retrieve(q["question"], q["history"], pair=True, run_id=run_id)]
        rec["both_found"] = expected <= set(sent)
    return rec


def summarize_retrieval(records: list[dict]) -> dict:
    scored = [r for r in records if r["type"] != "not_in_docs"]
    n = len(scored)
    summary = {"n": n, "mrr": sum(r["mrr"] for r in scored) / n}
    for k in K_VALUES:
        for m in ("recall", "precision", "ndcg"):
            summary[f"{m}@{k}"] = sum(r[f"{m}@{k}"] for r in scored) / n
        summary[f"hits@{k}"] = sum(r[f"recall@{k}"] == 1 for r in scored)
    both = [r for r in scored if r["type"] == "both"]
    summary["both_found"] = sum(r["both_found"] for r in both)
    summary["both_n"] = len(both)
    summary["by_type"] = {}
    for t in TYPES[:-1]:
        rows = [r for r in scored if r["type"] == t]
        if rows:
            summary["by_type"][t] = {"n": len(rows), "hits@5": sum(r["recall@5"] == 1 for r in rows),
                                     "mrr": sum(r["mrr"] for r in rows) / len(rows)}
    return summary


# ---------- answers ----------

def answer_record(q: dict, run_id: str) -> dict:
    sections, stream = rag.answer(q["question"], q["history"], run_id=run_id)
    raw = rag.clean_citations("".join(stream))
    shown, removed = rag.drop_unretrieved_citations(raw, sections)
    rec = {"question": q["question"], "type": q["type"], "answer": shown,
           "refused": rag.REFUSAL.lower() in raw.lower(), "unretrieved_citations": removed,
           "cited": bool(rag.cited_labels(raw))}
    if q["type"] != "not_in_docs":
        messages = [{"role": "user", "content": JUDGE_PROMPT.format(
            question=q["question"], reference=q["reference_answer"], answer=shown)}]
        response = providers.client().chat.completions.create(
            model=providers.JUDGE_MODEL, messages=messages, temperature=0,
            response_format={"type": "json_object"})
        rec["judge"] = json.loads(response.choices[0].message.content)
    return rec


def summarize_answers(records: list[dict]) -> dict:
    judged = [r for r in records if "judge" in r]
    absent = [r for r in records if r["type"] == "not_in_docs"]
    avg = lambda key: sum(r["judge"][key] for r in judged) / len(judged)
    return {
        "judge_model": providers.JUDGE_MODEL, "chat_model": providers.CHAT_MODEL, "n": len(judged),
        "accuracy": avg("accuracy"), "completeness": avg("completeness"), "relevance": avg("relevance"),
        "refused_not_in_docs": sum(r["refused"] for r in absent), "not_in_docs_n": len(absent),
        "false_refusals": sum(r["refused"] for r in judged),
        "clean_citations": sum(not r["unretrieved_citations"] for r in records), "answers_n": len(records),
    }


# ---------- report ----------

def write_markdown() -> None:
    runs = [json.loads(p.read_text()) for p in sorted(RESULTS_DIR.glob("*.json"))]
    lines = ["# Evaluation results", "",
             f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC by `eval/run_eval.py`. "
             "Percentages show raw counts. With about 45 scored questions, one question is about "
             "2 points, so differences under about 5 points are treated as ties.", ""]
    if runs:
        q = runs[-1]["questions"]
        lines += [f"Test set: {q['total']} questions, {q['verified']} marked verified by the owner.", ""]
    lines += ["## Retrieval", "",
              "| Run | Embedding | Hit@3 | Hit@5 | Hit@20 | MRR | nDCG@5 | Precision@5 | Both sources sent |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        s, n = r["retrieval"], r["retrieval"]["n"]
        lines.append(f"| `{r['run_id']}` | {r['embed_model']} | {pct(s['hits@3'], n)} | {pct(s['hits@5'], n)} | "
                     f"{pct(s['hits@20'], n)} | {s['mrr']:.2f} | {s['ndcg@5']:.2f} | {s['precision@5']:.2f} | "
                     f"{pct(s['both_found'], s['both_n'])} |")
    lines += ["", "Hit@K: all expected sections in the top K (pairs not counted). MRR and nDCG use the "
              "ranking without crosswalk pairs. Both sources sent: for `both` questions, law and IRR "
              "section both in what the model receives (top 5 plus crosswalk pairs).", ""]
    answered = [r for r in runs if r.get("answers")]
    if answered:
        lines += ["## Answers", "",
                  "| Run | Judge | Accuracy | Completeness | Relevance | Refused not_in_docs | False refusals | Clean citations |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in answered:
            a = r["answers"]
            lines.append(f"| `{r['run_id']}` | {a['judge_model']} judging {a['chat_model']} | {a['accuracy']:.2f} | "
                         f"{a['completeness']:.2f} | {a['relevance']:.2f} | "
                         f"{pct(a['refused_not_in_docs'], a['not_in_docs_n'])} | {pct(a['false_refusals'], a['n'])} | "
                         f"{pct(a['clean_citations'], a['answers_n'])} |")
        lines += ["", "Scores are 1 to 5 from the judge model against the reference answers. Clean citations: "
                  "answers where every cited section was among the retrieved sections.", ""]
    for r in runs:
        missed = [x for x in r["retrieval_records"] if x["type"] != "not_in_docs" and x["recall@5"] < 1]
        lines += [f"## Missed at top 5: `{r['run_id']}` ({len(missed)})", ""]
        lines += ["| Type | Question | Expected | Retrieved top 5 |", "|---|---|---|---|"]
        for x in missed:
            lines.append(f"| {x['type']} | {x['question']} | {'; '.join(x['expected'])} | {'; '.join(x['top5'])} |")
        lines.append("")
    RESULTS_MD.write_text("\n".join(lines), encoding="utf-8")


def evaluate(run_id: str, answers: bool) -> dict:
    questions = load_questions()
    with ThreadPoolExecutor(max_workers=8) as pool:
        retrieval = list(pool.map(lambda q: retrieval_record(q, run_id), questions))
    run = db.get_run(db.shared(), run_id)
    path = RESULTS_DIR / f"{run_id}.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    result = {"run_id": run_id, "embed_model": run["embed_model"], "settings": run["settings"],
              "questions": {"total": len(questions), "verified": sum(q["verified"].strip().lower() == "yes"
                                                                     for q in questions)},
              "retrieval": summarize_retrieval(retrieval), "retrieval_records": retrieval,
              "answers": previous.get("answers"), "answer_records": previous.get("answer_records")}
    if answers:
        with ThreadPoolExecutor(max_workers=6) as pool:
            records = list(pool.map(lambda q: answer_record(q, run_id), questions))
        result["answers"], result["answer_records"] = summarize_answers(records), records
    RESULTS_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", help="run_id to evaluate (default: ACTIVE_RUN_ID)")
    parser.add_argument("--all-runs", action="store_true", help="evaluate every stored run")
    parser.add_argument("--answers", action="store_true", help="also generate and judge answers")
    args = parser.parse_args()

    run_ids = ([r["run_id"] for r in db.list_runs(db.shared())] if args.all_runs
               else [args.run or rag.active_run_id()])
    for run_id in run_ids:
        r = evaluate(run_id, args.answers)
        s, n = r["retrieval"], r["retrieval"]["n"]
        print(f"{run_id}: hit@5 {pct(s['hits@5'], n)}, hit@20 {pct(s['hits@20'], n)}, MRR {s['mrr']:.2f}, "
              f"both sources {pct(s['both_found'], s['both_n'])}")
        if args.answers:
            a = r["answers"]
            print(f"  answers: accuracy {a['accuracy']:.2f}, completeness {a['completeness']:.2f}, "
                  f"relevance {a['relevance']:.2f}, refused not_in_docs {pct(a['refused_not_in_docs'], a['not_in_docs_n'])}, "
                  f"false refusals {pct(a['false_refusals'], a['n'])}")
    write_markdown()
    print(f"Wrote {RESULTS_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
