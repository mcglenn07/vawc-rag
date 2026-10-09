"""Follow one question through the whole pipeline and print what each step produced.

A learning tool: it calls the same functions the app uses, one step at a time.

Usage:
  python trace.py "How long does it last?" --prior "What is a Barangay Protection Order?"
  python trace.py "Who may file a petition for a protection order?"
"""

import argparse
import textwrap

import db
import providers
import rag


def show(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--prior", help="an earlier question, to make this a follow-up")
    args = parser.parse_args()

    history = None
    if args.prior:
        history = [{"role": "user", "content": args.prior},
                   {"role": "assistant", "content": "(earlier answer)"}]
    run_id = rag.active_run_id()

    show("STEP 1  Rewrite (rag.rewrite_query): follow-ups only")
    rewritten = rag.rewrite_query(args.question, history)
    print(f"question : {args.question}")
    print(f"rewritten: {rewritten or '(none: first message, nothing to resolve)'}")
    queries = [args.question] + ([rewritten] if rewritten else [])

    show("STEP 2  Embed (providers.embed): text -> numbers")
    vectors = providers.embed(queries, model=rag.run_embed_model(run_id))
    for q, v in zip(queries, vectors):
        print(f"{q[:50]!r:55} -> {len(v)} numbers, first 4: {[round(x, 4) for x in v[:4]]}")

    show("STEP 3  Vector search (db.search): closest chunks by cosine distance")
    conn = db.shared()
    for q, v in zip(queries, vectors):
        print(f"\nquery: {q}")
        for row in db.search(conn, v, run_id, 8):
            part = f" part {row['part']}/{row['parts']}" if row["parts"] > 1 else ""
            print(f"  {row['distance']:.3f}  {row['citation']}{part}")
    print("\n(0 = same meaning; smaller is closer)")

    show("STEPS 4-5  Merge parts into sections, keep top 5, add crosswalk pairs (rag.retrieve)")
    sections = rag.retrieve(args.question, history)
    for s in sections:
        dist = f"{s['distance']:.3f}" if s["distance"] is not None else "  -  "
        print(f"  {dist}  {s['source']:9}  {s['citation']}: {s['heading']}")

    show("STEP 6  Build the prompt (rag.build_messages): what the model actually receives")
    messages = rag.build_messages(args.question, history, sections)
    system = messages[0]["content"]
    sources_at = system.index("Sources:")
    print(f"{len(messages)} messages. System message: {len(system.split())} words "
          f"(rules {len(system[:sources_at].split())}, sources {len(system[sources_at:].split())}).")
    print("\nFirst source as the model sees it:\n")
    first = system[sources_at:].split("\n\n")[1]
    print(textwrap.indent(textwrap.shorten(first, 400), "  "))

    show("STEP 7  Generate (providers.chat_stream): the model writes the answer")
    raw = "".join(providers.chat_stream(messages))
    print(raw)

    show("STEP 8  Check citations (rag.clean_citations, rag.drop_unretrieved_citations)")
    cleaned = rag.clean_citations(raw)
    final, removed = rag.drop_unretrieved_citations(cleaned, sections)
    print(f"cited    : {sorted(set(rag.cited_labels(cleaned)))}")
    print(f"retrieved: {[s['citation'] for s in sections]}")
    print(f"removed  : {removed or 'nothing; every citation was a retrieved section'}")
    print(f"changed by cleanup: {'yes' if final != raw else 'no'}")


if __name__ == "__main__":
    main()
