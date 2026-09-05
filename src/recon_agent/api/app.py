"""FastAPI endpoints — ARCHITECTURE.md §10's API surface. Stage 13 of
this 16-stage relay build (see README.md's Build Status).

A thin API layer over what earlier stages already built:

  - ``POST /reconcile``     -> matching.pipeline.run_pipeline
  - ``GET /report/{id}``    -> evaluation.harness's metric functions
  - ``GET /audit/{id}``     -> the persisted DecisionEvent/MatchGroup/
                               MatchGroupMember/Exception trail
  - ``GET /llm-budget/{id}``-> llm.governor.CallBudgetGovernor.status
  - ``PATCH /report/{id}/groups/{group_id}/review`` -> api.service.review_group,
                               the human review loop: closes a
                               PENDING_REVIEW group the system flagged
                               but never let a human act on (see
                               review_group's own docstring)
  - ``GET /health``         -> liveness (+ a selfcheck if one exists —
                               it doesn't yet in this codebase, see
                               ``_run_selfcheck`` below)

No matching/verification/LLM/evaluation logic lives here — this module
only calls into ``api.service`` (orchestration), ``api.storage``
(persistence), and the existing modules those two wrap.
"""

from __future__ import annotations

import importlib
from typing import Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from recon_agent.api.schemas import (
    AuditResponse,
    HealthResponse,
    LLMBudgetResponse,
    ReconcileRequest,
    ReconcileResponse,
    ReportResponse,
    ReviewRequest,
    ReviewResponse,
)
from recon_agent.api.service import (
    ReviewConflictError,
    ReviewNotFoundError,
    execute_reconciliation,
    review_group,
)
from recon_agent.api.storage import RunStore
from recon_agent.api import service as service_module
from recon_agent.config import get_settings
from recon_agent.llm.governor import CallBudgetGovernor
from recon_agent.models import MatchGroupStatus

app = FastAPI(
    title="Reconciliation Agent API",
    description=(
        "Thin FastAPI layer (Stage 13) over the reconciliation matching "
        "pipeline, evaluation harness, and LLM call-budget governor built "
        "in earlier stages of this project — see ARCHITECTURE.md."
    ),
    version="1.0.0",
)

_store: Optional[RunStore] = None


def get_store() -> RunStore:
    """Process-wide RunStore, lazily constructed from the current
    settings' ``api_db_path`` on first use. A module-level singleton is
    safe to share across concurrent requests here specifically because
    ``RunStore`` itself holds no connection on ``self`` — every method
    call opens, uses, and closes its own fresh ``sqlite3.Connection``
    (see ``api/storage.py``'s class docstring), so concurrent requests
    never touch the same connection object at the same time even though
    they share this one ``RunStore`` instance."""
    global _store
    if _store is None:
        _store = RunStore(get_settings().api_db_path)
    return _store


def reset_store_for_tests(db_path: Optional[str] = None) -> RunStore:
    """Test-only hook: force a fresh RunStore (e.g. pointed at a
    temporary db_path), so test runs never share state with each other
    or with a real API_DB_PATH file."""
    global _store
    if _store is not None:
        _store.close()
    _store = RunStore(db_path if db_path is not None else get_settings().api_db_path)
    return _store


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError) -> JSONResponse:
    """FastAPI already returns 422 for a malformed request body by
    default; this handler only exists to keep the error body's shape
    stable/documented for API consumers rather than relying on the
    framework's default format."""
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.post("/reconcile", response_model=ReconcileResponse, status_code=status.HTTP_201_CREATED)
def reconcile(request: ReconcileRequest) -> ReconcileResponse:
    """Run the full existing matching/verification/LLM pipeline
    (matching/pipeline.py) over the uploaded batch and persist the
    result. Returns a ``run_id`` for use with the GET endpoints below.
    """
    store = get_store()
    run, result, elapsed, groq_calls_made = execute_reconciliation(
        records=request.records,
        store=store,
        dataset=request.dataset,
        ground_truth=request.ground_truth,
    )
    verified_count = sum(1 for g in result.match_groups if g.status == MatchGroupStatus.VERIFIED)
    pending_count = sum(1 for g in result.match_groups if g.status == MatchGroupStatus.PENDING_REVIEW)
    rejected_count = sum(1 for g in result.match_groups if g.status == MatchGroupStatus.REJECTED)

    return ReconcileResponse(
        run_id=run.run_id,
        status=run.status,
        started_at=run.started_at,
        total_records=len(request.records),
        matched_record_count=len(result.matched_record_ids),
        verified_group_count=verified_count,
        pending_review_group_count=pending_count,
        rejected_group_count=rejected_count,
        exception_count=len(result.exceptions),
        decision_event_count=len(result.decision_events),
        elapsed_seconds=elapsed,
        groq_calls_made=groq_calls_made,
    )


@app.get("/report/{run_id}", response_model=ReportResponse)
def get_report(run_id: str) -> ReportResponse:
    """The evaluation harness's metrics for this run (evaluation/harness.py) —
    the full report if ground truth was supplied at /reconcile time,
    otherwise only the ground-truth-independent metrics. Reuses
    harness.py's own metric functions; nothing here reimplements one.
    """
    store = get_store()
    if not store.run_exists(run_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No run found with run_id={run_id!r}")
    report_dict = service_module.build_report(run_id, store)
    return ReportResponse(**report_dict)


@app.get("/audit/{run_id}", response_model=AuditResponse)
def get_audit(run_id: str) -> AuditResponse:
    """The full DecisionEvent trail for this run, plus the MatchGroup/
    MatchGroupMember/Exception rows those events refer to."""
    store = get_store()
    if not store.run_exists(run_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No run found with run_id={run_id!r}")
    return AuditResponse(
        run_id=run_id,
        decision_events=store.get_decision_events(run_id),
        match_groups=store.get_match_groups(run_id),
        match_group_members=store.get_match_group_members(run_id),
        exceptions=store.get_exceptions(run_id),
    )


@app.patch("/report/{run_id}/groups/{group_id}/review", response_model=ReviewResponse)
def review_match_group(run_id: str, group_id: str, request: ReviewRequest) -> ReviewResponse:
    """The human review loop (see ``api.service.review_group``'s
    docstring for the full field-by-field behavior per action).

    ``reviewed_by``/``reviewed_at``/``review_action`` have existed on
    ``MatchGroup`` since early in the build (ARCHITECTURE.md §6/§7) but
    nothing before this endpoint ever let a human actually set them —
    PENDING_REVIEW groups existed with no way to close them. Only valid
    on a group currently PENDING_REVIEW; a 409 is returned for a group
    already VERIFIED or REJECTED (re-reviewing a closed group is never
    allowed silently).
    """
    store = get_store()
    if not store.run_exists(run_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No run found with run_id={run_id!r}")

    try:
        updated_group, event = review_group(
            run_id=run_id,
            group_id=group_id,
            reviewer=request.reviewer,
            review_action=request.review_action,
            note=request.note,
            store=store,
        )
    except ReviewNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ReviewConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    return ReviewResponse(run_id=run_id, match_group=updated_group, decision_event=event)


@app.get("/llm-budget/{run_id}", response_model=LLMBudgetResponse)
def get_llm_budget(run_id: str) -> LLMBudgetResponse:
    """The governor's call ledger (llm/governor.py) for this run —
    ``CallBudgetGovernor.status(run_id)`` verbatim, nothing computed
    independently of it."""
    store = get_store()
    if not store.run_exists(run_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No run found with run_id={run_id!r}")

    settings = get_settings()
    governor = CallBudgetGovernor(settings)
    try:
        snapshot = governor.status(run_id)
    finally:
        governor.close()

    return LLMBudgetResponse(
        run_id=run_id,
        decision=snapshot.decision.value,
        calls_used_today=snapshot.calls_used_today,
        ceiling=snapshot.ceiling,
        consecutive_failures=snapshot.consecutive_failures,
        circuit_open=snapshot.circuit_open,
        allowed=snapshot.allowed,
    )


def _run_selfcheck() -> Optional[dict]:
    """Run this codebase's existing selfcheck module, if one exists.

    No ``selfcheck`` module exists anywhere in this codebase as of
    Stage 13 (checked via grep across the whole tree before writing
    this endpoint) — so this returns ``None`` and ``/health`` reports
    basic liveness only, per this stage's own scope note. Written as a
    dynamic, best-effort import (rather than a hardcoded absence) so
    that if a later stage adds ``recon_agent.selfcheck`` with a
    ``run()`` function, this endpoint picks it up with no code change
    here.
    """
    try:
        module = importlib.import_module("recon_agent.selfcheck")
    except ModuleNotFoundError:
        return None
    run_fn = getattr(module, "run", None)
    if run_fn is None:
        return None
    return run_fn()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness check. Also runs this codebase's existing selfcheck
    module if one exists; as of Stage 13, none does yet, so this
    endpoint reports basic liveness only (see ``_run_selfcheck``)."""
    selfcheck_result = _run_selfcheck()
    return HealthResponse(
        status="ok",
        selfcheck_available=selfcheck_result is not None,
        selfcheck=selfcheck_result,
        note=None
        if selfcheck_result is not None
        else (
            "No recon_agent.selfcheck module exists in this codebase yet — "
            "reporting basic liveness only, per this stage's scope."
        ),
    )
