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
| 2026-07-23 | Hybrid search (vector + `tsvector` keyword, RRF fusion) | 0.91 (10/11) | 1.00 | 1.00 | **No change** — and the *why* is the finding. On the failing question the methods disagree symmetrically (resume #1 vec / #2 kw; career #2 vec / #1 kw), so RRF scores them identically (`1/(60+1)+1/(60+2)` both ways) and the tie holds. Equal-weight fusion *cannot* break a symmetric swap. Diagnosis: the miss is a **chunking** artifact (resume's dense summary chunk wins on both axes), not a keyword-relevance gap. Hybrid kept as the better default architecture. |
| 2026-07-23 | Paragraph-aware chunking (size 500) — **tried, reverted** | vec 0.82 / hyb 0.91 | 1.00 | 1.00 | Split docs on blank lines (11 → 19 chunks) to break the resume "kitchen-sink" chunk. Did **not** fix the target miss (still rank 2), and **regressed vector-only 0.91 → 0.82** by fragmenting content. Notably it *did* make hybrid beat vector-only (0.91 vs 0.82) — first concrete evidence hybrid earns its keep — but since it didn't improve the headline (hybrid) metric and hurt the baseline, it was **reverted** to fixed-width 800. Honest negative result. |
|            | _(next: cross-encoder reranking)_ |          |          |          | The stubborn miss is that `resume.md` genuinely contains career-direction text (a legitimate secondary answer). Reading question + chunk together (reranker) is the tool for "right doc is in the set but mis-ranked." |

## How to use this
1. Run `python eval.py`, note recall@1/@3/@5.
2. Make ONE change (e.g. add hybrid search). Re-run.
3. Add a row: date, what changed, the new numbers, and one line on what moved and why.
4. Keep the golden set fixed across a comparison — otherwise before/after isn't apples-to-apples.
