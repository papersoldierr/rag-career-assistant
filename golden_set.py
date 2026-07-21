"""The golden set: our hand-labeled answer key for grading retrieval.

Each entry pairs a QUESTION with the SOURCE FILE that should answer it. eval.py
runs each question through the real retriever and checks whether that source
came back. This is the "truth" the grader measures against — so it must be
accurate. You certify it by hand; that is what makes it "golden."

IMPORTANT — these are PLACEHOLDERS. documents/ is currently empty, so the
filenames below are guesses. When you add your real docs on the desktop:
  1. Make every `expected_source` match an actual filename in documents/.
  2. Fix the questions so they match what those docs really say.
An answer key that points at files which do not exist will score everything as a
miss.

`expected_source = None` marks an UNANSWERABLE question — your docs should NOT
contain the answer. eval.py reports these separately: a good system retrieves
nothing useful and Claude says "I don't know" instead of inventing an answer.
"""

GOLDEN = [
    # --- straightforward fact lookups (answer sits in one document) ---
    {"question": "Where is Bayo located?", "expected_source": "resume.md"},
    {"question": "What is Bayo's day job?", "expected_source": "resume.md"},
    {"question": "What industry does Bayo work in?", "expected_source": "resume.md"},

    # --- project / technical questions ---
    {"question": "What is the tech stack of the RAG career assistant?", "expected_source": "project-context.md"},
    {"question": "What has Bayo built with Postgres?", "expected_source": "project-context.md"},
    {"question": "What is the HVAC Parts Finder project?", "expected_source": "project-context.md"},
    {"question": "What is AirScore?", "expected_source": "project-context.md"},

    # --- career questions ---
    {"question": "What kind of roles is Bayo targeting?", "expected_source": "career-context.md"},
    {"question": "What AI projects is Bayo planning to build?", "expected_source": "career-context.md"},

    # --- web dev ---
    {"question": "Which web design clients has Bayo worked with?", "expected_source": "webdev-context.md"},

    # --- a harder, "spanning" question (answer touches more than one doc) ---
    # We label the single best source, but in review note whether the other doc
    # also should have surfaced. Spanning questions are where plain vector search
    # often slips — a good stress test.
    {"question": "How does Bayo combine web development with AI?", "expected_source": "career-context.md"},

    # --- an UNANSWERABLE question (your docs do not cover this) ---
    # A good system should NOT confidently retrieve a real answer here.
    # Reported separately by eval.py, not scored in recall.
    {"question": "What is Bayo's favorite movie?", "expected_source": None},
]
