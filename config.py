"""Central configuration. Loads secrets from .env and defines the models we use.

Keeping these in one place means a model or dimension change is a one-line edit,
not a hunt-and-replace across the codebase.
"""

import os

from dotenv import load_dotenv

load_dotenv()  # reads the .env file into environment variables

# --- Secrets (from .env) ---
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
VOYAGE_API_KEY = os.environ["VOYAGE_API_KEY"]
DATABASE_URL = os.environ["DATABASE_URL"]

# --- Models ---
# Embedding model (retrieval). voyage-3.5-lite is cheap and outputs 1024-dim vectors.
EMBED_MODEL = "voyage-3.5-lite"
EMBED_DIM = 1024  # MUST match the model's output size — it's the size of our DB vector column

# Generation model (writes the answer). Haiku is plenty for grounded Q&A and the cheapest tier.
CLAUDE_MODEL = "claude-haiku-4-5"

# Reranker (cross-encoder). Re-scores a shortlist of retrieved chunks by reading
# the question and each chunk TOGETHER — more accurate than embedding similarity,
# so we only run it on the top candidates. rerank-2-lite is Voyage's small/cheap tier.
RERANK_MODEL = "rerank-2-lite"
