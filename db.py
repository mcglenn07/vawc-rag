"""Postgres + pgvector: connection, schema, and the vector search query.

Plain psycopg instead of the Supabase client, so this runs against any Postgres
that has the pgvector extension.
"""

import os
import threading
import time

import numpy as np
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

load_dotenv()

# Untyped `vector` column (no fixed dimension) so runs with different embedding models
# can live side by side. Fine without an index: ~130 rows per run is an exact scan.
SCHEMA = """
-- pgvector adds a `vector` column type and distance operators to plain Postgres.
create extension if not exists vector;

-- One row per embedding run (a chunk setting + embedding model). Lets several setups
-- live side by side; the app serves the one named by ACTIVE_RUN_ID.
create table if not exists runs (
    run_id      text primary key,
    created_at  timestamptz not null default now(),
    embed_model text not null,
    settings    jsonb not null,
    n_chunks    int not null,
    note        text
);

create table if not exists chunks (
    run_id     text not null references runs(run_id) on delete cascade,
    id         text not null,
    part       int  not null,
    parts      int  not null,
    document   text not null,
    rule       text,
    section    int  not null,
    heading    text,
    citation   text not null,
    pdf_pages  int[] not null,
    text       text not null,
    embed_text text not null,
    embedding  vector not null,          -- the chunk's meaning as numbers (3,072 for 3-large)
    primary key (run_id, id, part)
);
"""

CHUNK_COLUMNS = "id, part, parts, document, rule, section, heading, citation, pdf_pages, text"


def connect() -> psycopg.Connection:
    url = (os.getenv("DATABASE_URL") or "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL is not set. Put it in .env (local) or Space secrets (deployed).")
    # prepare_threshold=None: Supabase's transaction pooler does not support prepared statements.
    # Retry: the Supabase pooler occasionally drops a new connection during the SSL handshake.
    for attempt in range(3):
        try:
            conn = psycopg.connect(url, autocommit=True, row_factory=dict_row, prepare_threshold=None,
                                   connect_timeout=10)
            break
        except psycopg.OperationalError:
            if attempt == 2:
                raise
            time.sleep(1 + attempt)
    try:
        register_vector(conn)
    except psycopg.ProgrammingError:
        pass  # extension not created yet; init_schema() creates it and registers again
    return conn


_shared: psycopg.Connection | None = None
_shared_lock = threading.Lock()


def shared() -> psycopg.Connection:
    """One long-lived connection for the app. A new pooler connection costs seconds
    (SSL handshake), a query on an open one about half a second."""
    global _shared
    with _shared_lock:
        if _shared is None or _shared.closed or _shared.broken:
            _shared = connect()
        return _shared


def reset_shared() -> None:
    global _shared
    with _shared_lock:
        if _shared is not None:
            _shared.close()
        _shared = None


def with_retry(fn, attempts: int = 3):
    """Run fn(conn) on the shared connection; reconnect and retry if the connection dropped."""
    for attempt in range(attempts):
        conn = shared()
        try:
            return fn(conn)
        except psycopg.OperationalError:
            if attempt == attempts - 1:
                raise
            if conn.closed or conn.broken:
                reset_shared()
            time.sleep(1 + attempt)


def init_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA)
    register_vector(conn)


def insert_run(conn, run_id: str, embed_model: str, settings: dict, chunks: list[dict],
               vectors: list[list[float]], note: str | None = None) -> None:
    with conn.transaction():
        conn.execute(
            "insert into runs (run_id, embed_model, settings, n_chunks, note) values (%s, %s, %s, %s, %s)",
            (run_id, embed_model, Jsonb(settings), len(chunks), note),
        )
        with conn.cursor() as cur:
            cur.executemany(
                "insert into chunks (run_id, id, part, parts, document, rule, section, heading, citation,"
                " pdf_pages, text, embed_text, embedding)"
                " values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                [(run_id, c["id"], c["part"], c["parts"], c["document"], c["rule"], c["section"],
                  c["heading"], c["citation"], c["pdf_pages"], c["text"], c["embed_text"], np.array(v))
                 for c, v in zip(chunks, vectors)],
            )


def get_run(conn, run_id: str) -> dict | None:
    return conn.execute("select * from runs where run_id = %s", (run_id,)).fetchone()


def list_runs(conn) -> list[dict]:
    return conn.execute("select run_id, created_at, embed_model, settings, n_chunks, note"
                        " from runs order by created_at").fetchall()


def delete_run(conn, run_id: str) -> int:
    return conn.execute("delete from runs where run_id = %s", (run_id,)).rowcount


def search(conn, vector: list[float], run_id: str, k: int) -> list[dict]:
    """Top k chunks by cosine distance (smaller = closer).

    This one query is the "retrieval" in RAG. `<=>` is pgvector's cosine distance operator:
    it compares the direction of two vectors, so texts about the same thing score close to 0
    even if they share few exact words. Postgres computes it for every chunk in the run and
    sorts; with ~130 rows that exact scan takes milliseconds, so no index is needed.
    """
    return conn.execute(
        f"select {CHUNK_COLUMNS}, embedding <=> %s as distance from chunks"
        " where run_id = %s order by distance limit %s",
        (np.array(vector), run_id, k),
    ).fetchall()


def get_sections(conn, run_id: str, citations: list[str]) -> list[dict]:
    """All parts of the given sections, in part order."""
    return conn.execute(
        f"select {CHUNK_COLUMNS} from chunks where run_id = %s and citation = any(%s) order by citation, part",
        (run_id, citations),
    ).fetchall()
