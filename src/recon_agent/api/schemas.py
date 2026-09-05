"""Request/response Pydantic models for the Stage 13 FastAPI layer.

These are pure typing/shape wrappers around the entities already
defined in ``recon_agent.models`` — ``NormalizedRecord``, ``MatchGroup``,
``MatchGroupMember``, ``DecisionEvent``, ``ReconciliationException`` are
reused directly wherever a response needs to carry one of them, rather
than redeclared here. Only genuinely new shapes (the request envelope,
the optional ground-truth payload, and each endpoint's response
envelope) get their own model.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from recon_agent.models import (
    DecisionEvent,
    MatchGroup,
    MatchGroupMember,
    NormalizedRecord,
    ReconciliationException,
    ReconciliationRunStatus,
    ReviewAction,
)


class GroundTruthPayload(BaseModel):
    """Optional held-out ground truth for an uploaded batch, shaped
    exactly like ``ground_truth/<dataset>/ground_truth.json`` (see
    ``evaluation.harness.load_ground_truth``). Supplying this is what
    lets ``GET /report/{run_id}`` compute the full, ground-truth-scored
    evaluation report for an arbitrary uploaded batch rather than only
    the ground-truth-independent metrics (runtime/cost, group status
    counts).

    Nested cluster/case entries are kept as loosely-typed dicts (rather
    than fully modeled) since ``load_ground_truth`` itself only reads a
    handful of keys off each one and different entries carry different
    optional bookkeeping fields (see the real
    ``ground_truth/*/ground_truth.json`` files for examples) — modeling
    them strictly here would just be re-deriving harness.py's own
    parsing rules a second time.
    """

    dataset: Optional[str] = None
    seed: Optional[int] = None
    num_logical_events: Optional[int] = None
    num_physical_records: Optional[int] = None
    match_groups: list[dict[str, Any]] = Field(default_factory=list)
    duplicates: list[dict[str, Any]] = Field(default_factory=list)
    unresolved: list[dict[str, Any]] = Field(default_factory=list)
    honest_abstention: list[dict[str, Any]] = Field(default_factory=list)
    category_counts: dict[str, int] = Field(default_factory=dict)


class ReconcileRequest(BaseModel):
    """POST /reconcile body — a batch of records in the same shape as
    ``data/*/records.json``, plus two optional fields that only affect
    what ``GET /report`` can later compute (never the matching run
    itself, which only ever sees ``records``)."""

    records: list[NormalizedRecord] = Field(
        ...,
        min_length=1,
        description="The batch to reconcile — same shape as data/*/records.json.",
    )
    dataset: Optional[str] = Field(
        default=None,
        description="Optional free-text label for this batch (e.g. 'evaluation'), stored for reference only.",
    )
    ground_truth: Optional[GroundTruthPayload] = Field(
        default=None,
        description=(
            "Optional held-out ground truth for this batch. When supplied, "
            "GET /report/{run_id} returns the full evaluation report "
            "(precision, coverage, false-match rate, exception quality, "
            "tier contribution); when omitted, that endpoint returns only "
            "the metrics that don't require ground truth."
        ),
    )


class ReconcileResponse(BaseModel):
    """POST /reconcile response — a summary of the completed run plus
    its run_id, for use with the other three GET endpoints."""

    run_id: str
    status: ReconciliationRunStatus
    started_at: datetime
    total_records: int
    matched_record_count: int
    verified_group_count: int
    pending_review_group_count: int
    rejected_group_count: int
    exception_count: int
    decision_event_count: int
    elapsed_seconds: float
    groq_calls_made: int


class ReportResponse(BaseModel):
    """GET /report/{run_id} response.

    When the run was created with ground truth (either inline or via a
    ``dataset`` name that matched an existing ``ground_truth/`` fixture),
    ``report`` holds ``evaluation.harness.EvaluationReport.to_dict()``
    verbatim — every field on it comes straight out of harness.py's own
    metric functions, nothing here recomputes any of them. Otherwise
    ``report`` is null and the ground-truth-independent fields
    (``runtime_and_cost``, the group status counts) are populated
    instead, using ``harness.compute_runtime_and_cost`` directly.
    """

    run_id: str
    ground_truth_available: bool
    note: Optional[str] = None
    report: Optional[dict[str, Any]] = None
    total_records: Optional[int] = None
    verified_group_count: Optional[int] = None
    pending_review_group_count: Optional[int] = None
    rejected_group_count: Optional[int] = None
    runtime_and_cost: Optional[dict[str, Any]] = None


class AuditResponse(BaseModel):
    """GET /audit/{run_id} response — the full DecisionEvent trail for
    the run, plus the MatchGroup/MatchGroupMember/Exception rows each
    event refers to, for a self-contained audit view."""

    run_id: str
    decision_events: list[DecisionEvent]
    match_groups: list[MatchGroup]
    match_group_members: list[MatchGroupMember]
    exceptions: list[ReconciliationException]


class LLMBudgetResponse(BaseModel):
    """GET /llm-budget/{run_id} response — ``llm.governor.CallBudgetGovernor``'s
    own call-ledger snapshot for this run, via its public ``status()``
    method. Nothing here is computed independently of the governor.

    Per the governor's own scoping (llm/governor.py's module docstring):
    ``calls_used_today`` and ``ceiling`` are the shared *daily* counters
    (scoped by calendar day across every concurrently-running run, not
    per-run), while ``consecutive_failures``/``circuit_open`` are scoped
    specifically to this ``run_id``'s own circuit breaker state.
    """

    run_id: str
    decision: str
    calls_used_today: int
    ceiling: int
    consecutive_failures: int
    circuit_open: bool
    allowed: bool


class ReviewRequest(BaseModel):
    """PATCH /report/{run_id}/groups/{group_id}/review body — a human
    reviewer closing out (or escalating) a ``PENDING_REVIEW`` group.

    ``reviewer`` is a free-text name/identifier string (no auth/user
    system exists anywhere in this codebase to validate it against —
    same trust level as every other caller-supplied field in this API).
    """

    reviewer: str = Field(
        ...,
        min_length=1,
        description="The human reviewer's name or identifier, stored verbatim as reviewed_by.",
    )
    review_action: ReviewAction = Field(
        ...,
        description="APPROVED closes the group as VERIFIED, REJECTED closes it as REJECTED, "
        "ESCALATED flags it for further attention without resolving it.",
    )
    note: Optional[str] = Field(
        default=None,
        description="Optional free-text note from the reviewer, recorded on the audit-trail "
        "DecisionEvent for this action (MatchGroup itself has no note field — see §6/§7).",
    )


class ReviewResponse(BaseModel):
    """PATCH /report/{run_id}/groups/{group_id}/review response — the
    updated ``MatchGroup`` plus the ``DecisionEvent`` this action
    emitted, so a caller sees the result without a second GET."""

    run_id: str
    match_group: MatchGroup
    decision_event: DecisionEvent


class HealthResponse(BaseModel):
    """GET /health response."""

    status: str
    selfcheck_available: bool
    selfcheck: Optional[dict[str, Any]] = None
    note: Optional[str] = None
