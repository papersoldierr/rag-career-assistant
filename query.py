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


def retrieve(question, k=5):
    """Find the k chunks whose embeddings are closest to the question."""
    # input_type="query" — embed the question the way Voyage embeds *searches*.
    qvec = vo.embed([question], model=EMBED_MODEL, input_type="query").embeddings[0]

    with connect() as conn:
        with conn.cursor() as cur:
            # `<=>` is pgvector's cosine-distance operator. Smaller = more similar,
            # so ORDER BY ascending + LIMIT k gives the k closest chunks.
            cur.execute(
                "SELECT source, content FROM chunks ORDER BY embedding <=> %s LIMIT %s;",
                (qvec, k),
            )
            return cur.fetchall()  # list of (source, content)


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
