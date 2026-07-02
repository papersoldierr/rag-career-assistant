"""Database layer: connect to Postgres and create the table that stores our chunks.

Each row is one chunk of a document: the text, which file it came from, and the
chunk's embedding (a 1024-number vector). pgvector adds the `vector` column type
and the similarity operators we'll use to find the closest chunks to a question.
"""

import psycopg
from pgvector.psycopg import register_vector

from config import DATABASE_URL, EMBED_DIM


def connect():
    """Open a connection with the pgvector type registered.

    register_vector teaches the Python driver how to send/receive `vector`
    values as plain Python lists. Use this for normal reads/writes AFTER the
    extension exists (i.e. after init_db has run once).
    """
    conn = psycopg.connect(DATABASE_URL)
    register_vector(conn)
    return conn


def init_db():
    """Enable the pgvector extension and create the `chunks` table if missing.

    Run this once against a fresh database. It doesn't register the vector type
    because it's the step that *creates* it.
    """
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            # Turn on pgvector inside this database (idempotent).
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            # One row per document chunk.
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    id        BIGSERIAL PRIMARY KEY,
                    source    TEXT NOT NULL,              -- filename the chunk came from
                    content   TEXT NOT NULL,              -- the chunk's text
                    embedding vector({EMBED_DIM})         -- its meaning-vector
                );
                """
            )
        conn.commit()
    print("Database initialized: 'chunks' table is ready.")


if __name__ == "__main__":
    # Lets you run `python db.py` to set up the schema.
    init_db()
