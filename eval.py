"""Evaluate retrieval quality: how often does our search return the right document?

Run it:  python eval.py

What it does:
  1. Loads the golden set (our hand-labeled answer key from golden_set.py).
  2. For each question, calls the REAL retriever from query.py (retrieve()).
  3. Checks whether the expected source file appears in the top-k results.
  4. Prints recall@k plus a per-question breakdown, so you can see WHICH
     questions miss and what got retrieved instead.

Why recall@k? See HIRING-SIGNAL.md. Short version: if the right document is not
retrieved, Claude cannot answer from it — so retrieval is the first thing to
measure, before touching prompts or generation.

Prerequisites (same as the rest of the app): a filled-in .env (API keys +
DATABASE_URL) and documents already ingested (`python ingest.py`). This script
only READS — it never changes the database.
"""

from query import retrieve          # reuse the ACTUAL search — we grade the real thing
from golden_set import GOLDEN

K = 5  # how many chunks retrieve() returns; matches the app's default


def sources_for(question, k=K):
    """Run the real retriever and return just the SET of source filenames it found.

    retrieve() returns (source, content) pairs; for grading recall we only care
    which files came back, so we collapse it to a set of source names.
    """
    results = retrieve(question, k)            # list of (source, content) tuples
    return {source for source, _ in results}


def evaluate(k=K):
    # Split the golden set: answerable questions get scored; unanswerable ones
    # are inspected by hand (they have no single "right" file to hit).
    answerable = [g for g in GOLDEN if g["expected_source"] is not None]
    unanswerable = [g for g in GOLDEN if g["expected_source"] is None]

    hits = 0
    print(f"\n=== Retrieval eval (recall@{k}) ===\n")

    for item in answerable:
        question = item["question"]
        expected = item["expected_source"]
        found = sources_for(question, k)

        if expected in found:
            hits += 1
            print(f"[PASS] {question}")
        else:
            # Show what came back instead — this is your failure diagnosis, and
            # the raw material for the "documented failure" write-up recruiters value.
            print(f"[MISS] {question}")
            print(f"       expected: {expected}")
            print(f"       got:      {sorted(found)}")

    total = len(answerable)
    recall = hits / total if total else 0.0
    print(f"\nrecall@{k} = {hits}/{total} = {recall:.2f}\n")

    # Unanswerable questions: no score, just eyeball what (if anything) surfaced.
    if unanswerable:
        print("=== Unanswerable (manual check — should retrieve nothing useful) ===")
        for item in unanswerable:
            found = sources_for(item["question"], k)
            print(f"[?] {item['question']}")
            print(f"    got: {sorted(found)}")
        print()

    return recall


if __name__ == "__main__":
    evaluate()
