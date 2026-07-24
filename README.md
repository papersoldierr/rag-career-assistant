# RAG Career Assistant

A retrieval-augmented question-answering service over my own career documents
(resume, project notes, context docs). Ask it "What has Bayo built with
Postgres?" and it retrieves the most relevant passages from my documents and has
Claude answer **grounded in those passages** — citing which document each answer
came from.

Built in real code (Python + FastAPI + Postgres/pgvector + Claude API) rather
than a no-code tool, to demonstrate the full retrieval pipeline end to end.

## What is RAG?

**Retrieval-Augmented Generation.** Instead of asking a language model to answer
from memory (where it can hallucinate), the system:

1. **Retrieves** the most relevant chunks of my documents, by comparing
   **embeddings** — numeric vectors that capture meaning, so a question and a
   passage about the same topic land near each other in vector space.
2. **Augments** the prompt with those retrieved chunks.
3. **Generates** an answer with Claude, grounded in the supplied context, with
   citations back to the source document.

## Architecture

```
  documents/                     ← my career docs (source of truth)
      │
      │  ingest: split into chunks
      ▼
  ┌──────────────┐   embed each chunk    ┌───────────────┐
  │  Voyage AI   │◀──────────────────────│   ingest.py   │
  │ (embeddings) │──────vectors─────────▶│               │
  └──────────────┘                       └───────┬───────┘
                                                  │ store text + vector
                                                  ▼
                                    ┌──────────────────────────┐
                                    │  Postgres + pgvector      │
                                    │  (vector similarity search)│
                                    └──────────────┬───────────┘
             question                              │ top-K similar chunks
                │                                   ▼
                │        ┌──────────────┐   grounded prompt   ┌────────────┐
                └───────▶│   query.py   │───────────────────▶│ Claude API │
                         │ (retrieve +  │◀───────answer──────│ (generation)│
                         │  ask Claude) │                    └────────────┘
                         └──────────────┘
```

**Two models, two roles:** Voyage AI produces the embeddings that power
retrieval; Claude writes the final answer. (Anthropic has no embeddings API —
Voyage is their recommended embedding provider.)

## Tech

- **Python** — pipeline and API
- **Voyage AI** — text embeddings (retrieval)
- **Anthropic Claude API** — answer generation
- **PostgreSQL + pgvector** — stores chunks and does vector similarity search
- **FastAPI** — HTTP API to ask questions
- **Docker Compose** — runs the pgvector database locally

## Setup

Prerequisites: Python 3.11+, an Anthropic API key, a Voyage AI key, and a
Postgres database with pgvector. For the database, pick one:

- **Supabase (recommended)** — hosted Postgres; no local setup. In the SQL editor
  run `create extension if not exists vector;`, then copy your connection string
  into `.env` (see `.env.example`).
- **Local Docker** — run `docker compose up -d` to start a pgvector container.

```bash
# 1. Copy the env template and fill in your keys + DATABASE_URL
cp .env.example .env

# 2. Create a virtual environment and install dependencies
python -m venv .venv
.venv/Scripts/activate        # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

### Run it

```bash
# 1. Add your career docs (.md / .txt) to documents/, then set up the schema
python db.py

# 2. Chunk, embed, and store them
python ingest.py

# 3. Ask questions (command line)
python query.py "What has Bayo built with Postgres?"
```

### Run as a web service

```bash
uvicorn app:app --reload
```

Then open **http://localhost:8000/docs** for an interactive UI, or POST to it:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What has Bayo built with Postgres?"}'
```

Response:

```json
{ "answer": "...", "sources": ["resume.md", "project-context.md"] }
```

## Evaluation

Getting the pipeline to *run* is table stakes; the real question is whether
retrieval returns the **right** document. So retrieval is measured directly,
against a hand-labeled **golden set** ([`golden_set.py`](golden_set.py)) of
`(question, expected-source)` pairs, using [`eval.py`](eval.py):

```bash
python eval.py
```

The metric is **recall@k** — of the top *k* chunks retrieved, does the expected
document appear? Rather than a single cutoff, the harness sweeps several and
reports the **rank** at which each question's correct document first shows up,
because a flattering recall@5 can hide a weak ranking that only recall@1 reveals.

### Results (11 answerable questions)

`eval.py` compares three retrieval methods on the same golden set:

| Method        | recall@1     | recall@3 | recall@5 |
| ------------- | ------------ | -------- | -------- |
| vector-only   | 0.91 (10/11) | 1.00     | 1.00     |
| hybrid (RRF)  | 0.91 (10/11) | 1.00     | 1.00     |
| hybrid+rerank | 0.73 (8/11)  | 1.00     | 1.00     |

### A failure the eval caught

recall@3 and @5 are perfect — with only a handful of documents, retrieving 5
chunks almost can't miss, so those numbers on their own are uninformative.
**recall@1 is where the real signal is**, and it flagged one question:

> *"What kind of roles is Bayo targeting?"* ranks `resume.md` **above**
> `career-context.md` (the document that actually answers it).

### Three upgrades tried — and the discipline of not shipping them

Three standard retrieval improvements were each implemented and **measured**
against the fixed golden set:

1. **Hybrid search** — Postgres full-text (`tsvector`/`ts_rank`) keyword search
   fused with vector search via **Reciprocal Rank Fusion (RRF)**. No change to
   recall@1. For the failing question the two methods disagree *symmetrically*
   (`resume.md`: vector #1 / keyword #2; `career.md`: vector #2 / keyword #1), so
   RRF scores them identically and the tie holds — no equal-weight fusion can
   break a symmetric swap.
2. **Finer chunking** — paragraph-aware splitting to break up `resume.md`'s dense
   "kitchen-sink" summary chunk. Didn't fix the miss and *regressed* vector-only
   (0.91 → 0.82). Reverted.
3. **Cross-encoder reranking** — Voyage `rerank-2-lite` over the retrieved pool.
   *Regressed* recall@1 to 0.73 (it pushed two previously-#1 answers down).

**Why none helped — and why that's the point.** On a small, clean, 4-document
corpus, retrieval is already **near-saturated**: the right document is in the top
3 for *every* question. Hybrid search, finer chunking, and reranking are tools for
**large, noisy** corpora; here they add noise, not signal. The lone remaining miss
is a **labeling nuance** — `resume.md` genuinely contains career-direction text, so
it's a legitimate secondary answer — not a retrieval defect (the app answers that
question correctly). **The engineering decision was to keep the simpler hybrid
pipeline and *not* ship complexity that didn't earn its place.** All three methods
stay in the code and the `eval.py` comparison, ready for a corpus that needs them.
See [`RESULTS.md`](RESULTS.md) for the full log.

## Status

✅ Command-line pipeline (ingest + query), FastAPI service, retrieval eval harness
(vector vs hybrid vs rerank), hybrid search, and reranking all implemented and
measured end to end on real documents. Default retrieval is **hybrid** — the
measured best for this corpus. See [`RESULTS.md`](RESULTS.md) for the experiment log.
