"""Gradio dashboard over eval/run_eval.py. Layout follows Donner's week 5 evaluator:
a retrieval section and an answer section, each with a run button, metric cards, a chart
by question type and a table of problem questions, plus a comparison of all stored runs.

Selecting a run loads its saved results (no API calls). Only the Run buttons call the API.

Usage: python eval/evaluator.py
"""

import json
import sys
from pathlib import Path

import gradio as gr
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_eval  # noqa: E402

CSS = """
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }
.card { border: 1px solid #e4e7ec; border-radius: 8px; padding: 12px; background: #fff; }
.card .label { font-size: 0.8rem; color: #475467; }
.card .value { font-size: 1.5rem; font-weight: 600; color: #101828; }
.card .sub { font-size: 0.8rem; color: #667085; }
"""

FORCE_LIGHT = """() => {
    const url = new URL(window.location);
    if (url.searchParams.get("__theme") !== "light") {
        url.searchParams.set("__theme", "light");
        window.location.replace(url);
    }
}"""


def saved_runs() -> list[str]:
    return sorted(p.stem for p in run_eval.RESULTS_DIR.glob("*.json"))


def stored_runs() -> list[str]:
    return [r["run_id"] for r in run_eval.db.list_runs(run_eval.db.shared())]


def load(run_id: str) -> dict | None:
    path = run_eval.RESULTS_DIR / f"{run_id}.json"
    return json.loads(path.read_text()) if path.exists() else None


def cards(items: list[tuple[str, str, str]]) -> str:
    body = "".join(f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div>'
                   f'<div class="sub">{sub}</div></div>' for label, value, sub in items)
    return f'<div class="cards">{body}</div>'


def count(n: int, d: int) -> tuple[str, str]:
    return (f"{100 * n / d:.0f}%" if d else "n/a", f"{n}/{d} questions")


def retrieval_view(result: dict | None):
    if not result:
        return "<p>No saved retrieval results for this run. Click Run retrieval eval.</p>", None, None
    s, n = result["retrieval"], result["retrieval"]["n"]
    html = cards([
        ("Hit@3", *count(s["hits@3"], n)), ("Hit@5", *count(s["hits@5"], n)),
        ("Hit@20", *count(s["hits@20"], n)), ("MRR", f"{s['mrr']:.2f}", "1 / rank of first expected"),
        ("nDCG@5", f"{s['ndcg@5']:.2f}", "rank-weighted"),
        ("Precision@5", f"{s['precision@5']:.2f}", "max 0.20 per expected section"),
        ("Both sources sent", *count(s["both_found"], s["both_n"])),
    ])
    chart = pd.DataFrame([{"type": t, "hit@5 %": round(100 * v["hits@5"] / v["n"]), "n": v["n"]}
                          for t, v in s["by_type"].items()])
    missed = pd.DataFrame([{"type": x["type"], "question": x["question"], "expected": "; ".join(x["expected"]),
                            "retrieved top 5": "; ".join(x["top5"])}
                           for x in result["retrieval_records"] if x["type"] != "not_in_docs" and x["recall@5"] < 1])
    return html, chart, missed


def answer_view(result: dict | None):
    if not result or not result.get("answers"):
        return "<p>No saved answer results for this run. Click Run answer eval.</p>", None, None
    a = result["answers"]
    html = cards([
        ("Accuracy", f"{a['accuracy']:.2f}", "1 to 5"), ("Completeness", f"{a['completeness']:.2f}", "1 to 5"),
        ("Relevance", f"{a['relevance']:.2f}", "1 to 5"),
        ("Refused not_in_docs", *count(a["refused_not_in_docs"], a["not_in_docs_n"])),
        ("False refusals", *count(a["false_refusals"], a["n"])),
        ("Clean citations", *count(a["clean_citations"], a["answers_n"])),
    ]) + f"<p><small>{a['judge_model']} judging {a['chat_model']}.</small></p>"
    rows = []
    for t in run_eval.TYPES[:-1]:
        judged = [x["judge"] for x in result["answer_records"] if x["type"] == t and "judge" in x]
        for m in ("accuracy", "completeness", "relevance"):
            if judged:
                rows.append({"bar": f"{t} {m[:3]}", "metric": m, "score": round(sum(j[m] for j in judged) / len(judged), 2)})
    problems = []
    for x in result["answer_records"]:
        j = x.get("judge")
        low = j and min(j["accuracy"], j["completeness"], j["relevance"]) <= 3
        wrong_refusal = x["refused"] != (x["type"] == "not_in_docs")
        if low or wrong_refusal or x["unretrieved_citations"]:
            problems.append({"type": x["type"], "question": x["question"],
                             "scores (acc/comp/rel)": "/".join(str(j[m]) for m in ("accuracy", "completeness", "relevance")) if j else "",
                             "refused": x["refused"], "removed citations": "; ".join(x["unretrieved_citations"]),
                             "judge feedback": (j or {}).get("feedback", "")})
    return html, pd.DataFrame(rows), pd.DataFrame(problems)


def comparison() -> pd.DataFrame:
    rows = []
    for run_id in saved_runs():
        r = load(run_id)
        s, n = r["retrieval"], r["retrieval"]["n"]
        row = {"run": run_id, "embedding": r["embed_model"], "chunks": "split {split_words}, overlap {overlap}, {p}".format(**r["settings"], p="prefix" if r["settings"]["heading_prefix"] else "no prefix"),
               "hit@3": f"{s['hits@3']}/{n}", "hit@5": f"{s['hits@5']}/{n}", "hit@20": f"{s['hits@20']}/{n}",
               "MRR": round(s["mrr"], 2), "nDCG@5": round(s["ndcg@5"], 2)}
        if r.get("answers"):
            row["accuracy"] = round(r["answers"]["accuracy"], 2)
        rows.append(row)
    return pd.DataFrame(rows)


def show(run_id: str):
    result = load(run_id)
    return (*retrieval_view(result), *answer_view(result), comparison())


def run_retrieval(run_id: str):
    run_eval.evaluate(run_id, answers=False)
    run_eval.write_markdown()
    return show(run_id)


def run_answers(run_id: str):
    run_eval.evaluate(run_id, answers=True)
    run_eval.write_markdown()
    return show(run_id)


with gr.Blocks(title="RA 9262 RAG Evaluation") as demo:
    gr.Markdown("# RA 9262 RAG Evaluation\nRetrieval is scored on section labels; answers by an LLM judge "
                "against reference answers. With about 45 scored questions, one question is about 2 points.")
    runs = stored_runs()
    run = gr.Dropdown(runs, value=run_eval.rag.active_run_id() if runs else None, label="Run")

    gr.Markdown("## Retrieval")
    run_r = gr.Button("Run retrieval eval (embeddings only, under a cent)")
    r_cards = gr.HTML()
    with gr.Row():
        r_chart = gr.BarPlot(x="type", y="hit@5 %", title="Hit@5 by question type", y_lim=[0, 100], height=260)
    r_missed = gr.Dataframe(label="Missed at top 5", wrap=True)

    gr.Markdown("## Answers")
    run_a = gr.Button("Run answer eval (answers every question and judges it, about $0.30)")
    a_cards = gr.HTML()
    a_chart = gr.BarPlot(x="bar", y="score", color="metric", title="Judge scores by question type", sort=None,
                         x_label_angle=-30,
                         y_lim=[0, 5], height=260)
    a_problems = gr.Dataframe(label="Problem answers: low score, wrong refusal, or removed citations", wrap=True)

    gr.Markdown("## All runs")
    compare = gr.Dataframe(wrap=True)

    outputs = [r_cards, r_chart, r_missed, a_cards, a_chart, a_problems, compare]
    run.change(show, run, outputs)
    run_r.click(run_retrieval, run, outputs)
    run_a.click(run_answers, run, outputs)
    demo.load(show, run, outputs)

if __name__ == "__main__":
    demo.launch(css=CSS, js=FORCE_LIGHT)
