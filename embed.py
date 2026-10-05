"""Embed data/chunks.json and store it in Postgres under a new run_id.

Each run keeps its chunk settings and embedding model, so chunk settings can be
compared side by side. The app serves whichever run ACTIVE_RUN_ID names.

Usage:
  python embed.py [--chunks data/chunks.json] [--note "baseline"]
  python embed.py --list
  python embed.py --delete-run RUN_ID
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import db
import providers

ROOT = Path(__file__).parent


def make_run_id(settings: dict) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    prefix = "hp" if settings["heading_prefix"] else "nohp"
    return f"{stamp}-sw{settings['split_words']}-ov{settings['overlap']}-{prefix}"


def embed_chunks(chunks_path: Path, note: str | None = None) -> str:
    data = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunks, settings = data["chunks"], data["settings"]
    vectors = providers.embed([c["embed_text"] for c in chunks])
    run_id = make_run_id(settings)
    with db.connect() as conn:
        db.init_schema(conn)
        db.insert_run(conn, run_id, providers.EMBED_MODEL, settings, chunks, vectors, note)
    return run_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=Path, default=ROOT / "data" / "chunks.json")
    parser.add_argument("--note")
    parser.add_argument("--list", action="store_true", help="list stored runs")
    parser.add_argument("--delete-run", metavar="RUN_ID")
    args = parser.parse_args()

    if args.list:
        with db.connect() as conn:
            for r in db.list_runs(conn):
                print(f"{r['run_id']}  {r['embed_model']}  {r['n_chunks']} chunks  {r['note'] or ''}")
        return
    if args.delete_run:
        with db.connect() as conn:
            print(f"Deleted {db.delete_run(conn, args.delete_run)} run(s)")
        return

    run_id = embed_chunks(args.chunks, args.note)
    print(f"Stored run {run_id} ({providers.EMBED_MODEL})")
    print(f"To serve it, set ACTIVE_RUN_ID={run_id}")


if __name__ == "__main__":
    main()
