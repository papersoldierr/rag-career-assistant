# Retrieval Eval Results

A running log of `python eval.py` runs, one row per meaningful change. The point
is the **before/after**: when a retrieval change (hybrid search, reranking, chunk
tuning) moves recall@1, record it here so the improvement is documented, not
guessed. This table is the raw material for the résumé metric and the interview
story.

Metric = recall@k (of the top-k retrieved chunks, does the expected document
appear?). recall@1 is the real signal — recall@3/@5 saturate fast on a small corpus.

| Date       | Change                              | recall@1 | recall@3 | recall@5 | Notes |
| ---------- | ----------------------------------- | -------- | -------- | -------- | ----- |
| 2026-07-21 | Baseline — vector-only retrieval    | 0.91 (10/11) | 1.00 | 1.00 | Caught a real miss: "what roles is Bayo targeting?" ranks `resume.md` above `career-context.md` (summary chunk out-competes on pure vector similarity). |
|            | _(next: hybrid BM25 + vector, RRF)_ |          |          |          | Expected to lift recall@1 by ranking the keyword-matching doc higher. |

## How to use this
1. Run `python eval.py`, note recall@1/@3/@5.
2. Make ONE change (e.g. add hybrid search). Re-run.
3. Add a row: date, what changed, the new numbers, and one line on what moved and why.
4. Keep the golden set fixed across a comparison — otherwise before/after isn't apples-to-apples.
