"""Orchestration for the Stage 13 API layer.

This module is deliberately thin: it calls ``matching.pipeline.run_pipeline``
and ``evaluation.harness``'s metric functions exactly as they're already
built, and calls ``llm.governor.CallBudgetGovernor`` exactly as it's
already built. It does not implement, alter, or duplicate any matching,
verification, LLM, or evaluation logic — its only two jobs are (1) run
the existing pipeline once and persist what it returns, and (2)
reassemble an ``evaluation.harness.EvaluationReport`` from persisted
data on demand, using harness.py's own metric functions directly.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from recon_agent.api.schemas import GroundTruthPayload
from recon_agent.api.storage import RunStore
from recon_agent.config import Settings, get_settings
from recon_agent.evaluation.harness import (
    EvaluationReport,
    compute_auto_match_precision,
    compute_bank_credit_coverage,
    compute_complete_cluster_resolution,
    compute_exception_quality,
    compute_false_match_rate,
    compute_record_coverage,
    compute_runtime_and_cost,
    compute_tier_contribution,
    compute_value_coverage,
    load_ground_truth,
)
from recon_agent.llm.governor import CallBudgetGovernor, GovernorCheckResult
from recon_agent.matching.pipeline import MatchingResult, run_pipeline
from recon_agent.models import (
    DecisionEvent,
    DecisionStage,
    MatchGroup,
    MatchGroupStatus,
    NormalizedRecord,
    ReconciliationRun,
    ReconciliationRunStatus,
    ReviewAction,
    VerificationResult,
    VerifiedBy,
)


class ReviewNotFoundError(LookupError):
    """Raised when the referenced ``(run_id, group_id)`` doesn't exist —
    ``api/app.py`` maps this to a 404."""


class ReviewConflictError(RuntimeError):
    """Raised when a review action is attempted on a group that isn't
    currently ``PENDING_REVIEW`` — ``api/app.py`` maps this to a 409.
    Carries the group's actual current status for the error detail."""

    def __init__(self, group_id: str, current_status: MatchGroupStatus) -> None:
        self.group_id = group_id
        self.current_status = current_status
        super().__init__(
            f"Group {group_id!r} is not PENDING_REVIEW (current status: "
            f"{current_status.value}) — only a PENDING_REVIEW group can be reviewed."
        )


def _members_by_group(members) -> dict[str, list]:
    out: dict[str, list] = {}
    for m in members:
        out.setdefault(m.group_id, []).append(m)
    return out


def _input_hash(records: list[NormalizedRecord]) -> str:
    payload = json.dumps(
        [r.model_dump(mode="json") for r in records], sort_keys=True, default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _determine_status(result: MatchingResult, governor_snapshot: GovernorCheckResult) -> ReconciliationRunStatus:
    """A judgment call, documented here since ARCHITECTURE.md §7 defines
    the ``ReconciliationRunStatus`` enum but not a rule for picking
    between its values: a run whose Groq budget was exhausted or whose
    circuit breaker tripped is reported DEGRADED regardless of whether
    it also produced exceptions (the degradation is the more important
    fact to surface); otherwise a run with any ``ReconciliationException``
    is COMPLETED_WITH_EXCEPTIONS, and a clean run is COMPLETE. FAILED is
    never returned here — a pipeline run that raises is never persisted
    at all (see ``execute_reconciliation``), so there is no run row to
    attach FAILED to.
    """
    if not governor_snapshot.allowed:
        return ReconciliationRunStatus.DEGRADED
    if result.exceptions:
        return ReconciliationRunStatus.COMPLETED_WITH_EXCEPTIONS
    return ReconciliationRunStatus.COMPLETE


def execute_reconciliation(
    records: list[NormalizedRecord],
    store: RunStore,
    dataset: Optional[str] = None,
    ground_truth: Optional[GroundTruthPayload] = None,
    settings: Optional[Settings] = None,
) -> tuple[ReconciliationRun, MatchingResult, float, int]:
    """Run the full existing pipeline over ``records`` once, persist the
    result, and return ``(run, result, elapsed_seconds, groq_calls_made)``.
    """
    settings = settings if settings is not None else get_settings()
    run_id = uuid.uuid4().hex
    started_at = datetime.now(timezone.utc)

    governor = CallBudgetGovernor(settings)
    try:
        start = time.perf_counter()
        result = run_pipeline(records, settings, run_id=run_id, governor=governor)
        elapsed = time.perf_counter() - start
        governor_snapshot = governor.status(run_id)
    finally:
        governor.close()

    status = _determine_status(result, governor_snapshot)

    run = ReconciliationRun(
        run_id=run_id,
        input_hash=_input_hash(records),
        rules_version=settings.rules_version,
        model_version=settings.model_version,
        threshold_version=settings.threshold_version,
        policy_version=settings.policy_version,
        started_at=started_at,
        status=status,
    )

    store.save_run(
        run=run,
        records=records,
        match_groups=result.match_groups,
        match_group_members=result.match_group_members,
        decision_events=result.decision_events,
        exceptions=result.exceptions,
        group_id_to_stage=result.group_id_to_stage,
        elapsed_seconds=elapsed,
        dataset=dataset,
        ground_truth=ground_truth.model_dump(mode="json") if ground_truth is not None else None,
    )

    return run, result, elapsed, governor_snapshot.calls_used_today


def build_report(run_id: str, store: RunStore) -> dict[str, Any]:
    """Reassemble an ``EvaluationReport`` for an already-persisted run,
    using ``evaluation.harness``'s own metric functions directly — see
    module docstring. Returns a plain dict shaped like ``ReportResponse``
    (never raises for an unknown run; callers check ``store.run_exists``
    first and raise the HTTP 404 themselves).
    """
    run = store.get_run(run_id)
    records = store.get_records(run_id)
    records_by_id = {r.record_id: r for r in records}
    match_groups = store.get_match_groups(run_id)
    members = store.get_match_group_members(run_id)
    members_by_group = _members_by_group(members)
    decision_events = store.get_decision_events(run_id)
    exceptions = store.get_exceptions(run_id)
    group_id_to_stage = store.get_group_id_to_stage(run_id)
    elapsed = store.get_elapsed_seconds(run_id) or 0.0
    dataset = store.get_dataset(run_id)

    settings = get_settings()
    governor = CallBudgetGovernor(settings)
    try:
        groq_calls_made = governor.status(run_id).calls_used_today
    finally:
        governor.close()

    from recon_agent.models import MatchGroupStatus

    verified_count = sum(1 for g in match_groups if g.status == MatchGroupStatus.VERIFIED)
    pending_count = sum(1 for g in match_groups if g.status == MatchGroupStatus.PENDING_REVIEW)
    rejected_count = sum(1 for g in match_groups if g.status == MatchGroupStatus.REJECTED)

    ground_truth_raw = store.get_ground_truth_raw(run_id)

    if ground_truth_raw is None:
        runtime_cost = compute_runtime_and_cost(
            elapsed, len(records), decision_events, records_by_id, groq_calls_made
        )
        return {
            "run_id": run_id,
            "ground_truth_available": False,
            "note": (
                "No ground truth was supplied for this run, so ground-truth"
                "-dependent metrics (auto-match precision, record/value "
                "coverage, false-match rate, exception quality, tier "
                "contribution, bank credit coverage, complete cluster "
                "resolution) cannot be computed by the evaluation harness "
                "— it requires held-out ground truth to score a run, and "
                "none exists for an uploaded batch unless one is supplied. "
                "POST a `ground_truth` object (shaped like "
                "ground_truth/<dataset>/ground_truth.json) alongside the "
                "records in /reconcile to get the full report."
            ),
            "total_records": len(records),
            "verified_group_count": verified_count,
            "pending_review_group_count": pending_count,
            "rejected_group_count": rejected_count,
            "runtime_and_cost": dataclasses.asdict(runtime_cost),
        }

    # Reuse evaluation.harness.load_ground_truth unchanged by writing the
    # supplied/looked-up ground truth to a temp file, rather than
    # re-deriving its JSON-parsing rules here a second time.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(ground_truth_raw, tmp)
        tmp_path = Path(tmp.name)
    try:
        gt = load_ground_truth(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    precision = compute_auto_match_precision(match_groups, members_by_group, gt)
    coverage = compute_record_coverage(records, match_groups, members_by_group, gt)
    value_coverage = compute_value_coverage(records, match_groups, members_by_group, gt)
    false_match = compute_false_match_rate(match_groups, members_by_group, gt)
    exception_quality = compute_exception_quality(match_groups, members_by_group, exceptions, gt)
    tier_contribution = compute_tier_contribution(
        match_groups, members_by_group, group_id_to_stage, gt, len(records)
    )
    runtime_cost = compute_runtime_and_cost(
        elapsed, len(records), decision_events, records_by_id, groq_calls_made
    )
    bank_credit_coverage = compute_bank_credit_coverage(records, match_groups, members_by_group, gt)
    complete_cluster_resolution = compute_complete_cluster_resolution(match_groups, members_by_group, gt)

    report = EvaluationReport(
        dataset=dataset or gt.dataset or "uploaded_batch",
        total_records=len(records),
        dataset_info={
            "num_logical_events": gt.num_logical_events,
            "num_physical_records": gt.num_physical_records,
            "category_counts": gt.category_counts,
            "groq_configured": bool(settings.groq_api_key),
        },
        auto_match_precision=precision,
        record_coverage=coverage,
        value_coverage=value_coverage,
        false_match_rate=false_match,
        exception_quality=exception_quality,
        tier_contribution=tier_contribution,
        runtime_and_cost=runtime_cost,
        bank_credit_coverage=bank_credit_coverage,
        complete_cluster_resolution=complete_cluster_resolution,
        verified_group_count=verified_count,
        pending_review_group_count=pending_count,
        rejected_group_count=rejected_count,
    )

    return {
        "run_id": run_id,
        "ground_truth_available": True,
        "note": None,
        "report": report.to_dict(),
        "total_records": len(records),
        "verified_group_count": verified_count,
        "pending_review_group_count": pending_count,
        "rejected_group_count": rejected_count,
        "runtime_and_cost": dataclasses.asdict(runtime_cost),
    }


def review_group(
    run_id: str,
    group_id: str,
    reviewer: str,
    review_action: ReviewAction,
    note: Optional[str],
    store: RunStore,
) -> tuple[MatchGroup, DecisionEvent]:
    """Apply a human reviewer's decision to one ``PENDING_REVIEW`` group
    — the orchestration behind ``PATCH
    /report/{run_id}/groups/{group_id}/review`` (api/app.py).

    This is the human closing a loop the system deliberately left open:
    ``reviewed_by``/``reviewed_at``/``review_action`` have existed on
    ``MatchGroup`` since early in the build (ARCHITECTURE.md §6/§7),
    but nothing before this endpoint ever set them. No matching or
    verification logic is touched here — this only records a decision
    a human already made about a group the automated verifier already
    evaluated and couldn't resolve on its own.

    Outcome per ``review_action`` (see ARCHITECTURE.md §6):

    - ``APPROVED`` — the human is *completing* verification the system
      couldn't finish alone: ``verified_by`` becomes ``HUMAN_REVIEWER``,
      ``verification_result`` becomes ``PASSED``, ``status`` becomes
      ``VERIFIED``. This is spelled out explicitly in §6 ("At that
      point verified_by becomes HUMAN_REVIEWER and verification_result
      reflects the reviewer's decision").

    - ``REJECTED`` — a human's explicit "no" is a final decision, but
      deliberately does NOT overwrite ``verified_by``/
      ``verification_result``: those two fields already recorded *why*
      the automated verifier routed this group to PENDING_REVIEW
      (e.g. ``FAILED_MARGIN``, ``RESIDUAL_EXCEEDS_TOLERANCE``), and
      that historical reasoning stays true and useful after the fact —
      overwriting it with a human-rejection-flavored value would erase
      the very evidence that explains *why* a human's judgment was
      needed here in the first place. The human's own verdict is fully
      captured by ``review_action=REJECTED`` + ``reviewed_by`` +
      ``reviewed_at`` + ``status=REJECTED``, which is everything the
      task requires and everything the audit trail needs. Unlike an
      *automatic* system rejection (``run_verification`` in
      ``matching/pipeline.py``, which releases the group's member
      records back into ``matched_record_ids`` so a later stage in the
      *same* pipeline run can retry them), a human REJECTED verdict
      releases nothing: there is no "later stage" for a human decision
      to hand off to within this run, the pipeline that produced this
      group has already finished, and a human's explicit "no" on a
      specific group is a considered, final answer that should not be
      silently retried by some other matching path. Records named by a
      human-REJECTED group simply stay claimed by that (now terminally
      REJECTED) group; nothing re-enters an unmatched pool.

    - ``ESCALATED`` — "flagged for further attention", not resolved:
      ``reviewed_by``/``reviewed_at``/``review_action`` are recorded
      (so the audit trail shows who flagged it and when) but ``status``
      stays ``PENDING_REVIEW``, ``verified_by``/``verification_result``
      are untouched, and the group remains eligible for a later,
      different review action to actually resolve it.

    Every action also emits a ``DecisionEvent`` for the review action
    itself, stage ``STAGE7_VERIFICATION`` — reusing that existing stage
    value rather than adding a new one (documented choice, see
    ARCHITECTURE.md and BUILD_LOG.md): ``STAGE7_VERIFICATION`` is
    already defined as "the verifier's own pass/fail evaluation" of a
    group, and a human review is exactly that same kind of event — an
    evaluation of the group as a whole, not a new *proposal* — just
    performed by ``HUMAN_REVIEWER`` instead of
    ``FINANCIAL_AND_EVIDENCE_VERIFIER``. This also means the dashboard's
    existing "every DecisionEvent naming this group_id" drill-down
    (dashboard/app.py's ``_events_for``) picks up review events with no
    changes needed there.

    Raises ``ReviewNotFoundError`` if no such group exists in this run,
    or ``ReviewConflictError`` if the group is not currently
    ``PENDING_REVIEW`` — callers (api/app.py) map these to 404 / 409.
    """
    group = store.get_match_group(run_id, group_id)
    if group is None:
        raise ReviewNotFoundError(
            f"No group found with group_id={group_id!r} in run_id={run_id!r}"
        )
    if group.status != MatchGroupStatus.PENDING_REVIEW:
        raise ReviewConflictError(group_id, group.status)

    now = datetime.now(timezone.utc)

    if review_action == ReviewAction.APPROVED:
        updated_group = group.model_copy(
            update={
                "reviewed_by": reviewer,
                "reviewed_at": now,
                "review_action": ReviewAction.APPROVED,
                "verified_by": VerifiedBy.HUMAN_REVIEWER,
                "verification_result": VerificationResult.PASSED,
                "status": MatchGroupStatus.VERIFIED,
            }
        )
        reason_code = "HUMAN_APPROVED"
        explanation = (
            f"Human reviewer {reviewer!r} approved this PENDING_REVIEW group, "
            "closing the loop the automated verifier left open. "
            "verified_by -> HUMAN_REVIEWER, verification_result -> PASSED, "
            "status -> VERIFIED."
        )
    elif review_action == ReviewAction.REJECTED:
        updated_group = group.model_copy(
            update={
                "reviewed_by": reviewer,
                "reviewed_at": now,
                "review_action": ReviewAction.REJECTED,
                "status": MatchGroupStatus.REJECTED,
            }
        )
        reason_code = "HUMAN_REJECTED"
        explanation = (
            f"Human reviewer {reviewer!r} rejected this PENDING_REVIEW group. "
            "status -> REJECTED. Unlike an automatic system rejection, this is "
            "a final human decision: the group's member records are NOT "
            "released back to the unmatched pool for another stage to retry "
            "(see api/service.py's review_group docstring for the full "
            "reasoning). verified_by/verification_result are left as the "
            "automated verifier originally set them, preserving the record "
            "of why this group needed human judgment in the first place."
        )
    else:  # ESCALATED
        updated_group = group.model_copy(
            update={
                "reviewed_by": reviewer,
                "reviewed_at": now,
                "review_action": ReviewAction.ESCALATED,
            }
        )
        reason_code = "HUMAN_ESCALATED"
        explanation = (
            f"Human reviewer {reviewer!r} escalated this PENDING_REVIEW group "
            "for further attention. status remains PENDING_REVIEW — this "
            "records who flagged it and when, without resolving it; a later "
            "review action can still approve or reject it."
        )

    candidate_scores: dict[str, Any] = {"reviewer": reviewer, "review_action": review_action.value}
    if note is not None:
        candidate_scores["note"] = note

    event = DecisionEvent(
        event_id=f"review_evt_{uuid.uuid4().hex[:12]}",
        group_id=group_id,
        stage=DecisionStage.STAGE7_VERIFICATION,
        candidate_scores=candidate_scores,
        reason_code=reason_code,
        explanation=explanation,
        timestamp=now,
    )

    store.apply_review(run_id, updated_group, event)
    return updated_group, event
