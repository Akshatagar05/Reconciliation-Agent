"""api — the FastAPI layer (ARCHITECTURE.md §10). Stage 13 of this
16-stage relay build.

A thin API surface over the matching pipeline, evaluation harness, and
LLM call-budget governor built in earlier stages — see ``app.py`` for
the five endpoints (``POST /reconcile``, ``GET /report/{run_id}``,
``GET /audit/{run_id}``, ``GET /llm-budget/{run_id}``, ``GET /health``),
``storage.py`` for the SQLite persistence layer this stage adds (no
pipeline run was persisted anywhere before this stage — every
CLI/test invocation just ran in-memory), ``service.py`` for the
orchestration between them, and ``schemas.py`` for the request/response
Pydantic models. See README.md's Build Status for how to run it.
"""

from recon_agent.api.app import app

__all__ = ["app"]
