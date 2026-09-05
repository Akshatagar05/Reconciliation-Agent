"""Stage 14 — Streamlit judge-facing dashboard.

A presentation layer only, per this stage's own scope: everything in
this package either (a) calls the existing Stage 13 FastAPI endpoints
over real HTTP (``api_client.py``), or (b) loads/lists the existing
bundled ``data/*/records.json`` + ``ground_truth/*/ground_truth.json``
fixtures from disk (``datasets.py``) so a judge can pick one without
retyping a batch by hand. No matching, verification, LLM, or evaluation
logic is reimplemented here — nothing in this package computes a metric,
a match, or a decision; it only renders what ``GET /report``,
``GET /audit``, and ``GET /llm-budget`` already return.

Run the dashboard with:

    streamlit run src/recon_agent/dashboard/app.py

The FastAPI layer must already be running (a separate process) — the
dashboard is an HTTP *client* of it, never an in-process caller of
``matching``/``evaluation``/``llm`` modules:

    uvicorn recon_agent.api.app:app --reload

See ``app.py``'s module docstring and README.md's "Streamlit Dashboard
(Stage 14)" section for the full instructions.
"""
