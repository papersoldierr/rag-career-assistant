"""Query: given a question, retrieve the most relevant chunks and have Claude
answer using only those chunks.

Run it:  python query.py "What has Bayo built with Postgres?"

Flow:
  question -> hybrid retrieval (vector + keyword, fused by RRF) -> Claude answer

Retrieval is HYBRID: it runs two searches and fuses them.
  - vector search:  semantic similarity (meaning) via pgvector
  - keyword search: exact-term relevance (tsvector/ts_rank) via Postgres full-text
Vector search alone misses exact terms (a name, a literal phrase like "roles I'm
targeting"); keyword search alone misses paraphrases. Reciprocal Rank Fusion
combines the two ranked lists so a chunk that either method likes rises to the top.
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


# --- The two searches. Both return an ORDERED list of (id, source, content),
#     best match first. `id` is the chunk's primary key — the stable identity
#     RRF uses to tell whether both searches found the same chunk. ---

def vector_search(qvec, k):
    """Top-k chunks by semantic similarity (pgvector cosine distance)."""
    with connect() as conn:
        with conn.cursor() as cur:
            # `<=>` is pgvector's cosine-distance operator. Smaller = more similar,
            # so ORDER BY ascending + LIMIT k gives the k closest chunks.
            # `%s::vector` casts the Python list to a pgvector vector (psycopg
            # would otherwise send it as a plain array, which `<=>` rejects).
            cur.execute(
                "SELECT id, source, content FROM chunks "
                "ORDER BY embedding <=> %s::vector LIMIT %s;",
                (qvec, k),
            )
            return cur.fetchall()


def keyword_search(question, k):
    """Top-k chunks by exact-term relevance (Postgres full-text search).

    `content_tsv @@ query` is the keyword match; `ts_rank` scores how well each
    chunk matches. websearch_to_tsquery ANDs the terms together, which is too
    strict for question-style input (a chunk would need EVERY word) — so we swap
    `&` for `|` to match ANY term (OR), which is the right recall behavior here.
    """
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH q AS (
                    SELECT replace(
                        websearch_to_tsquery('english', %s)::text, '&', '|'
                    )::tsquery AS query
                )
                SELECT c.id, c.source, c.content
                FROM chunks c, q
                WHERE c.content_tsv @@ q.query
                ORDER BY ts_rank(c.content_tsv, q.query) DESC
                LIMIT %s;
                """,
                (question, k),
            )
            return cur.fetchall()


def reciprocal_rank_fusion(rankings, rrf_k=60):
    """Fuse several ranked lists into one, by Reciprocal Rank Fusion.

    Each list is ordered (id, source, content) tuples, best first. A chunk's
    fused score is the sum over lists of 1 / (rrf_k + rank), where rank is its
    1-based position in that list. So being near the top of EITHER list helps,
    and appearing in BOTH helps most. rrf_k (conventionally 60) softens the gap
    between rank 1 and rank 2 so no single list dominates. RRF needs only the
    ranks — not the raw scores — so it fuses vector distances and text scores
    (which aren't on the same scale) without any normalization.
    """
    scores = {}
    meta = {}  # id -> (source, content), so we can rebuild rows after ranking
    for ranking in rankings:
        for rank, (cid, source, content) in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
            meta[cid] = (source, content)
    ordered_ids = sorted(scores, key=scores.get, reverse=True)
    return [(cid, *meta[cid]) for cid in ordered_ids]


# --- Public retrievers. Both return [(source, content)] (what answer()/eval want).
#     They accept a pre-computed qvec so the eval can embed once and reuse it. ---

def retrieve_vector(qvec, k=5):
    """Vector-only retrieval (the baseline we compare hybrid against)."""
    return [(source, content) for _id, source, content in vector_search(qvec, k)]


def retrieve_hybrid(question, qvec, k=5, pool=20):
    """Hybrid retrieval: fuse vector + keyword search with RRF, keep top k.

    We over-fetch `pool` (> k) candidates from each search so a chunk ranked, say,
    #8 by vectors but #1 by keywords can still win after fusion — then trim to k.
    """
    fused = reciprocal_rank_fusion(
        [vector_search(qvec, pool), keyword_search(question, pool)]
    )
    return [(source, content) for _id, source, content in fused[:k]]


def retrieve(question, k=5):
    """The app's default retrieval: embed the question, then hybrid-search."""
    return retrieve_hybrid(question, embed_query(question), k)


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
