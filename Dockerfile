# Reconciliation Agent — Docker image (Stage 15 of 16, ARCHITECTURE.md §9/§14;
# dependency cleanup below by Stage 16 of 16)
#
# One image, two processes (wired together by docker-compose.yml):
#   - the FastAPI service   (recon_agent.api.app)
#   - the Streamlit dashboard (recon_agent.dashboard.app)
#
# This image guarantees, per §14's acceptance gate:
#
#   1. data/calibration and data/evaluation (plus their ground_truth/
#      counterparts) are generated INTO THE IMAGE at build time, from
#      the deterministic seeded generator — not copied from whatever
#      happens to be on the host's disk. README.md's own "Synthetic
#      Data (Stage 2)" section is explicit that neither directory is
#      committed to git and both regenerate byte-identically from a
#      fixed seed, so doing this at build time is both correct and
#      reproducible for a judge doing a completely fresh `git clone`.
#
# --- Stage 16 cleanup note ---------------------------------------------------
# Earlier revisions of this Dockerfile also pre-baked a `sentence-transformers`
# embedding model (`all-MiniLM-L6-v2`) into the image at build time, per an
# explicit ARCHITECTURE.md §10 line item from an earlier design round. Stage 15
# flagged, in both this file and README.md's Build Status, that a full search
# of `src/` found no actual import of `sentence_transformers` anywhere —
# Stage 9's fuzzy candidate retrieval was built entirely on `rapidfuzz`
# identifier + counterparty similarity and hit its target (100% precision, 0%
# false-match rate) without embeddings. Stage 16 acts on that finding: the
# dependency, the pre-bake `RUN` step, and the HF/sentence-transformers-only
# env vars (`HF_HOME`, `SENTENCE_TRANSFORMERS_HOME`, `HF_HUB_OFFLINE`,
# `TRANSFORMERS_OFFLINE`, `HF_HUB_DISABLE_TELEMETRY`) are all removed below —
# nothing else in the codebase referenced them (checked directly). This drops
# `torch` (sentence-transformers' transitive dependency) from the image too,
# which is the bulk of the size/time savings. ARCHITECTURE.md §10 has been
# updated to match (see its own "Removed" note).

FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="recon-agent" \
      org.opencontainers.image.description="Exception-aware multi-source settlement reconciliation agent (Razorpay AI Buildathon, Track 04)"

# --- OS-level basics -------------------------------------------------------
# No compiler toolchain is installed on purpose: every dependency in
# requirements.txt ships manylinux wheels for this base image's glibc, so
# nothing here needs to be built from source. Keeping build-essential etc.
# out keeps the image smaller and the attack surface lower.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# --- Python dependencies (cached as its own layer) --------------------------
# Copied and installed before the rest of the source so that editing
# application code doesn't invalidate this layer.
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "pytest>=8,<9" "httpx>=0.27,<1"

# --- Application source ------------------------------------------------------
COPY . .

# Install the package itself in editable mode, without re-resolving
# dependencies (already satisfied by the layer above).
RUN pip install --no-cache-dir --no-deps -e .

# --- Pre-bake the calibration/evaluation datasets ---------------------------
# Deterministic from the fixed seeds in testdata/generator.py (see
# README.md's "Synthetic Data (Stage 2)" section) — regenerating here
# rather than relying on COPY means this works identically for a judge's
# completely fresh `git clone`, where data/ and ground_truth/ don't exist
# on disk yet (neither directory is committed — see .gitignore).
RUN python -m recon_agent.testdata.generator --seed both && \
    test -s data/calibration/records.json && \
    test -s data/evaluation/records.json && \
    test -s ground_truth/calibration/ground_truth.json && \
    test -s ground_truth/evaluation/ground_truth.json

# --- Non-root runtime user ---------------------------------------------------
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data-runtime \
    && chown -R appuser:appuser /app
USER appuser

# Where API_DB_PATH / GOVERNOR_DB_PATH point by default in
# docker-compose.yml, so a named volume can persist run state across
# container restarts (see docker-compose.yml).
VOLUME ["/app/data-runtime"]

EXPOSE 8000 8501

HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

# docker-compose.yml overrides this per-service (uvicorn for the api
# service, streamlit for the dashboard service); this default is what
# `docker run <image>` alone gives you — the API, so the image is still
# useful/testable standalone.
CMD ["uvicorn", "recon_agent.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
