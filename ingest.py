"""Ingest: read documents, split them into chunks, embed each chunk, store them.

Run this whenever the documents change:  python ingest.py

Flow:
  documents/*.md,*.txt  ->  chunks  ->  Voyage embeddings  ->  Postgres
"""

import glob
import os

import voyageai

from config import EMBED_MODEL, VOYAGE_API_KEY
from db import connect, init_db

vo = voyageai.Client(api_key=VOYAGE_API_KEY)


def chunk_text(text, chunk_size=800, overlap=150):
    """Split text into overlapping windows.

    Why chunk? Embeddings capture the meaning of a *passage*; a whole document is
    too broad to retrieve precisely. Why overlap? So a sentence split across a
    boundary still appears whole in one of the chunks, and we don't lose context.
    """
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap  # step forward, leaving `overlap` chars shared
    return chunks


def load_documents(folder="documents"):
    """Read every .md / .txt file in the documents folder."""
    docs = []
    for path in glob.glob(os.path.join(folder, "*")):
        if path.endswith((".md", ".txt")):
            with open(path, encoding="utf-8") as f:
                docs.append((os.path.basename(path), f.read()))
    return docs


def embed(texts):
    """Turn a list of texts into a list of vectors via Voyage.

    input_type="document" tells Voyage these are stored passages (not a search
    query) — it embeds the two slightly differently for better matching.
    """
    result = vo.embed(texts, model=EMBED_MODEL, input_type="document")
    return result.embeddings


def ingest():
    init_db()  # make sure the table exists

    # 1. Load documents and split each into chunks, tracking the source file.
    rows = []  # list of (source, chunk_text)
    for source, text in load_documents():
        for chunk in chunk_text(text):
            rows.append((source, chunk))

    if not rows:
        print("No documents found in documents/ — add some .md/.txt files first.")
        return

    print(f"Loaded {len(rows)} chunks. Embedding via Voyage...")

    # 2. Embed the chunk texts (batched to stay within Voyage's per-call limit).
    vectors = []
    batch_size = 128
    for i in range(0, len(rows), batch_size):
        batch_texts = [text for _, text in rows[i : i + batch_size]]
        vectors.extend(embed(batch_texts))

    # 3. Store everything. Clear the table first so re-running doesn't duplicate.
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE chunks;")
            cur.executemany(
                "INSERT INTO chunks (source, content, embedding) VALUES (%s, %s, %s);",
                [(source, text, vec) for (source, text), vec in zip(rows, vectors)],
            )
        conn.commit()

    print(f"Ingested {len(rows)} chunks into the database.")


if __name__ == "__main__":
    ingest()
