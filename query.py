"""Query: given a question, retrieve the most relevant chunks and have Claude
answer using only those chunks.

Run it:  python query.py "What has Bayo built with Postgres?"

Flow (default):
  question -> hybrid retrieval (vector + keyword, RRF) -> Claude answer

HYBRID RETRIEVAL fuses two searches into one candidate ranking:
  - vector search:  semantic similarity (meaning) via pgvector
  - keyword search: exact-term relevance (tsvector/ts_rank) via Postgres FTS
  Reciprocal Rank Fusion (RRF) merges the two ranked lists.

A cross-encoder RERANK stage (retrieve_reranked) is also implemented — the
standard "retrieve wide, then rerank narrow" pattern — but it is NOT the default:
on this small, clean corpus it measured worse than plain hybrid (see eval.py /
RESULTS.md). It's kept for the comparison and for larger, noisier corpora.
"""

import sys
import time

import anthropic
import voyageai

from config import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    EMBED_MODEL,
    RERANK_MODEL,
    VOYAGE_API_KEY,
)
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


def rerank(question, candidates, k):
    """Re-score candidate chunks with Voyage's cross-encoder reranker; keep top k.

    Embedding search compares the question and a chunk as two SEPARATE vectors.
    A cross-encoder instead reads the question and chunk TOGETHER and scores how
    well the chunk answers the question — more accurate, but too slow to run over
    the whole corpus, so we only apply it to a shortlist (`candidates`).

    `candidates` is an ordered list of (id, source, content). Voyage's reranker
    takes the raw texts and returns results with the original `.index` and a
    `.relevance_score`, best first — we map those indices back to our tuples.
    """
    if not candidates:
        return []
    texts = [content for _id, _source, content in candidates]
    # Voyage's free tier is rate-limited (3 req/min without a payment method).
    # Wait out the window and retry rather than crash. Adding a payment method
    # (still free — the token allowance is unchanged) removes the wait.
    for attempt in range(6):
        try:
            result = vo.rerank(question, texts, model=RERANK_MODEL, top_k=k)
            return [candidates[r.index] for r in result.results]
        except voyageai.error.RateLimitError:
            if attempt == 5:
                raise
            time.sleep(21)


def retrieve_reranked(question, qvec, k=5, pool=20):
    """Full retrieval pipeline: hybrid-retrieve a pool, then cross-encoder rerank.

    Retrieval (vector + keyword) casts a wide, cheap net to pull `pool` plausible
    chunks; the reranker then does the accurate, expensive scoring on just those.
    """
    fused = reciprocal_rank_fusion(
        [vector_search(qvec, pool), keyword_search(question, pool)]
    )
    reranked = rerank(question, fused[:pool], k)
    return [(source, content) for _id, source, content in reranked]


def retrieve(question, k=5):
    """The app's default retrieval: embed the question, then hybrid-search.

    NOTE: reranking (retrieve_reranked) is implemented and compared in eval.py, but
    it is deliberately NOT the default here. On this small, clean corpus it measured
    WORSE (recall@1 0.91 -> 0.73) — retrieval is already near-saturated, so the
    cross-encoder adds noise rather than signal. Kept for the comparison and for
    when the corpus grows large/noisy enough to benefit. See RESULTS.md.
    """
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
