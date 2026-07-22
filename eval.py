"""Evaluate retrieval quality: how often does our search return the right document?

Run it:  python eval.py

What it does:
  1. Loads the golden set (our hand-labeled answer key from golden_set.py).
  2. Embeds every question in ONE Voyage call, then runs the REAL search
     (retrieve_by_vector from query.py) once per question.
  3. Reports recall at several cutoffs (recall@1, @3, @5) and, for each question,
     the RANK at which the expected document's first chunk appears. Rank is the
     real diagnostic: recall@5 can look perfect while recall@1 reveals weak
     ranking. The gap between them is the story worth documenting.

Why recall@k? If the right document is not retrieved, Claude cannot answer from
it — so retrieval is the first thing to measure, before touching prompts or
generation.

Why several k's? With only a few documents, recall@5 is easy: retrieving 5 chunks
from 4 docs almost can't miss. recall@1 asks the harder question — is the right
doc ranked FIRST? — and that's where plain vector search slips (especially on
"spanning" questions whose answer touches more than one doc).

Prerequisites (same as the rest of the app): a filled-in .env (API keys +
DATABASE_URL) and documents already ingested (`python ingest.py`). This script
only READS — it never changes the database.
"""

# reuse the ACTUAL search (retrieve_by_vector) — we grade the real thing.
# embed_queries batches all questions into one Voyage call.
from query import retrieve_by_vector, embed_queries
from golden_set import GOLDEN

K_VALUES = [1, 3, 5]        # report recall at each of these cutoffs
RETRIEVE_N = max(K_VALUES)  # retrieve enough chunks to evaluate the largest k


def chunk_sources(qvec, n=RETRIEVE_N):
    """Ordered list of the source filenames of the top-n chunks (closest first).

    We keep DUPLICATES and ORDER on purpose: recall@k is about the top-k *chunks*
    we'd actually feed the LLM, and rank is about WHERE the expected source first
    shows up in that ranking. Both need the raw ordered chunk list, not a set.
    """
    return [source for source, _ in retrieve_by_vector(qvec, n)]


def first_rank(sources, expected):
    """1-based position of the first chunk from `expected`, or None if absent."""
    for i, source in enumerate(sources, start=1):
        if source == expected:
            return i
    return None


def evaluate(k_values=K_VALUES):
    # Split the golden set: answerable questions get scored; unanswerable ones
    # are inspected by hand (they have no single "right" file to hit).
    answerable = [g for g in GOLDEN if g["expected_source"] is not None]
    unanswerable = [g for g in GOLDEN if g["expected_source"] is None]

    # Embed EVERY question in a single Voyage request, then reuse the vectors.
    all_questions = [g["question"] for g in GOLDEN]
    vec_by_question = dict(zip(all_questions, embed_queries(all_questions)))

    # One search per answerable question; record the ordered sources + the rank
    # of the expected doc. Everything below is computed from this — no re-search.
    rows = []  # (question, expected, ordered_sources, rank_or_None)
    for item in answerable:
        q, expected = item["question"], item["expected_source"]
        sources = chunk_sources(vec_by_question[q])
        rows.append((q, expected, sources, first_rank(sources, expected)))

    total = len(rows)
    print(f"\n=== Retrieval eval - recall@k over the golden set ===")
    print(f"{total} answerable questions, retrieving top-{RETRIEVE_N} chunks each.\n")

    # recall@k = fraction whose expected doc appears within the top k chunks,
    # i.e. rank exists and rank <= k. Computed from the ranks we already have.
    for k in k_values:
        hits = sum(1 for _, _, _, rank in rows if rank is not None and rank <= k)
        print(f"recall@{k} = {hits}/{total} = {hits / total:.2f}")

    # Per-question ranks, worst first, so misses at k=1 jump out. These flagged
    # rows are the raw material for the "documented failure" write-up.
    print("\nPer-question (rank = position of the expected doc's first chunk):")
    for q, expected, sources, rank in sorted(
        rows, key=lambda r: (r[3] is None, -(r[3] or 0))
    ):
        if rank is None:
            flag = "  <- MISS: not in top " + str(RETRIEVE_N)
            where = f"got {sources}"
        else:
            flag = "  <- misses recall@1" if rank > 1 else ""
            where = f"[{expected}]"
        print(f"  rank {rank if rank else '-'}  {q:52}  {where}{flag}")

    # Unanswerable questions: no score, just eyeball what surfaced. The real test
    # (Claude refuses to answer) lives in query.py, not here.
    if unanswerable:
        print("\n=== Unanswerable (should NOT surface a confident single source) ===")
        for item in unanswerable:
            sources = chunk_sources(vec_by_question[item["question"]])
            print(f"  {item['question']}")
            print(f"    top chunks from: {sources}")
        print()

    # Return the recall@k map so callers/tests can assert on it.
    return {
        k: sum(1 for _, _, _, rank in rows if rank is not None and rank <= k) / total
        for k in k_values
    }


if __name__ == "__main__":
    evaluate()
