"""FastAPI web layer: exposes the RAG pipeline over HTTP.

Run it:  uvicorn app:app --reload
Then open http://localhost:8000/docs for an interactive UI to try it.

This reuses answer() from query.py — the web layer is a thin wrapper; all the
real work (retrieve + generate) already lives in the pipeline.
"""

from fastapi import FastAPI
from pydantic import BaseModel

from query import answer

app = FastAPI(title="RAG Career Assistant")


class Question(BaseModel):
    """The shape of an incoming request body. FastAPI validates it for us."""

    question: str
    k: int = 5  # how many chunks to retrieve


@app.get("/health")
def health():
    """Simple liveness check — handy for deploys and uptime monitors."""
    return {"status": "ok"}


@app.post("/ask")
def ask(payload: Question):
    """Answer a question and return the answer plus which documents it used."""
    text, chunks = answer(payload.question, payload.k)
    return {
        "answer": text,
        "sources": sorted({source for source, _ in chunks}),
    }
