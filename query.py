"""Query: given a question, retrieve the most relevant chunks and have Claude
answer using only those chunks.

Run it:  python query.py "What has Bayo built with Postgres?"

Flow:
  question -> Voyage embedding -> pgvector similarity search -> Claude answer
"""

import sys

import anthropic
import voyageai

from config import ANTHROPIC_API_KEY, CLAUDE_MODEL, EMBED_MODEL, VOYAGE_API_KEY
from db import connect

vo = voyageai.Client(api_key=VOYAGE_API_KEY)
claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def embed_query(question):
    """Embed a single question as a search query (input_type='query')."""
    return vo.embed([question], model=EMBED_MODEL, input_type="query").embeddings[0]


def embed_queries(questions):
    """Embed MANY questions in one Voyage call.

    Same idea as batching in ingest.py: one request for N questions instead of N
    requests. Cheaper, faster, and it stays under Voyage's rate limit.
    """
    return vo.embed(questions, model=EMBED_MODEL, input_type="query").embeddings


def retrieve_by_vector(qvec, k=5):
    """Find the k chunks closest to an ALREADY-EMBEDDED question vector.

    This is the actual database search, split out so callers that already have a
    vector (e.g. the batched eval) can skip re-embedding.
    """
    with connect() as conn:
        with conn.cursor() as cur:
            # `<=>` is pgvector's cosine-distance operator. Smaller = more similar,
            # so ORDER BY ascending + LIMIT k gives the k closest chunks.
            # Cast the parameter to `vector`: psycopg sends a Python list as a
            # plain Postgres array, but the `<=>` operator needs a pgvector
            # `vector` on both sides. `%s::vector` makes that explicit.
            cur.execute(
                "SELECT source, content FROM chunks ORDER BY embedding <=> %s::vector LIMIT %s;",
                (qvec, k),
            )
            return cur.fetchall()  # list of (source, content)


def retrieve(question, k=5):
    """Embed the question, then find the k closest chunks. (The one-shot path.)"""
    return retrieve_by_vector(embed_query(question), k)


def answer(question, k=5):
    chunks = retrieve(question, k)
    if not chunks:
        return "The database is empty — run `python ingest.py` first.", []

    # Build the grounded prompt: label each chunk with its source for citations.
    context = "\n\n".join(f"[{source}]\n{content}" for source, content in chunks)
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the context does not contain the answer, say you don't know. "
        "Cite the source file name(s) in square brackets.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )

    message = claude.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,  # grounded answers are short; deliberately capped
        messages=[{"role": "user", "content": prompt}],
    )

    # The response content is a list of blocks; grab the text.
    text = "".join(block.text for block in message.content if block.type == "text")
    return text, chunks


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or input("Ask a question: ")
    text, chunks = answer(question)
    print("\n" + text + "\n")
    sources = sorted({source for source, _ in chunks})
    if sources:
        print("Retrieved from:", ", ".join(sources))
