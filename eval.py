"""Evaluate retrieval quality: how often does our search return the right document?

Run it:  python eval.py

What it does:
  1. Loads the golden set (our hand-labeled answer key from golden_set.py).
  2. Embeds every question in ONE Voyage call, then runs EACH retrieval method
     (vector-only and hybrid) once per question, reusing the same vectors.
  3. Reports recall@1/@3/@5 for each method side-by-side, plus a per-question
     comparison of the RANK at which the expected document first appears — so a
     change's effect shows up as a before/after, not a guess.

Why recall@k? If the right document is not retrieved, Claude cannot answer from
it — so retrieval is the first thing to measure, before touching prompts or
generation. recall@1 is the real signal: with only a few documents recall@3/@5
saturate at 1.00, while recall@1 asks the harder question — is the right doc
ranked FIRST? — which is where ranking quality (and hybrid search) shows up.

Prerequisites: a filled-in .env (API keys + DATABASE_URL) and documents already
ingested (`python ingest.py`). This script only READS — it never writes.
"""

from query import embed_queries, retrieve_vector, retrieve_hybrid
from golden_set import GOLDEN

K_VALUES = [1, 3, 5]        # report recall at each of these cutoffs
RETRIEVE_N = max(K_VALUES)  # retrieve enough chunks to evaluate the largest k

# The methods we compare. Each takes (question, qvec) and returns an ordered
# list of (source, content), best first. Same signature => easy to add more.
METHODS = {
    "vector-only": lambda question, qvec: retrieve_vector(qvec, RETRIEVE_N),
    "hybrid (RRF)": lambda question, qvec: retrieve_hybrid(question, qvec, RETRIEVE_N),
}


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

    # Embed EVERY question in a single Voyage request, then reuse the vectors
    # across both methods (keyword search needs no embedding at all).
    all_questions = [g["question"] for g in GOLDEN]
    vec_by_question = dict(zip(all_questions, embed_queries(all_questions)))

    # For each method: question -> rank of the expected doc (None = missed).
    ranks = {name: {} for name in METHODS}
    for name, retriever in METHODS.items():
        for item in answerable:
            q, expected = item["question"], item["expected_source"]
            sources = [s for s, _ in retriever(q, vec_by_question[q])]
            ranks[name][q] = first_rank(sources, expected)

    total = len(answerable)
    print("\n=== Retrieval eval - recall@k over the golden set ===")
    print(f"{total} answerable questions, retrieving top-{RETRIEVE_N} chunks each.\n")

    # recall@k table, one row per method. recall@k = fraction whose expected doc
    # appears within the top k (rank exists and <= k).
    header = "method".ljust(14) + "".join(f"recall@{k}".rjust(11) for k in k_values)
    print(header)
    recall = {}
    for name in METHODS:
        recall[name] = {}
        line = name.ljust(14)
        for k in k_values:
            hits = sum(1 for r in ranks[name].values() if r is not None and r <= k)
            recall[name][k] = hits / total
            line += f"{recall[name][k]:.2f}".rjust(11)
        print(line)

    # Per-question rank comparison, so you can see EXACTLY which questions moved.
    # (Only meaningful with 2+ methods; assumes vector-only vs hybrid here.)
    names = list(METHODS)
    if len(names) == 2:
        a, b = names
        print(f"\nPer-question rank of the expected doc ({a} -> {b}; lower is better):")
        for item in answerable:
            q = item["question"]
            ra, rb = ranks[a][q], ranks[b][q]
            if ra == rb:
                tag = ""
            elif rb is None:
                tag = "  REGRESSED (dropped out)"
            elif ra is None or rb < ra:
                tag = "  <- IMPROVED"
            else:
                tag = "  <- regressed"
            fa, fb = (ra if ra else "-"), (rb if rb else "-")
            print(f"  {q:52}  {fa} -> {fb}{tag}")

    # Unanswerable questions: no score, just eyeball what surfaced under hybrid.
    # The real test (Claude refuses to answer) lives in query.py, not here.
    if unanswerable:
        print("\n=== Unanswerable (should NOT surface a confident single source) ===")
        for item in unanswerable:
            sources = [s for s, _ in retrieve_hybrid(item["question"],
                                                     vec_by_question[item["question"]],
                                                     RETRIEVE_N)]
            print(f"  {item['question']}")
            print(f"    hybrid top chunks from: {sources}")
        print()

    return recall


if __name__ == "__main__":
    evaluate()
