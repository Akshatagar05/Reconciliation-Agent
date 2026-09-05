"""Pipeline orchestration — ARCHITECTURE.md §6.

Runs Stage 1 (exact) -> Stage 2 (constrained) -> Stage 3 (aggregate) ->
Stage 4 (adjustment) -> Stage 5 (fuzzy candidate retrieval) in sequence
over one pool of ``NormalizedRecord`` instances, each stage only ever
working on records the earlier stages left unmatched, then runs every
proposed ``MatchGroup`` through the Financial and Evidence Verifier
(``verification.verifier.verify_match_group``, ARCHITECTURE.md §2) as a
verification pass (Stage 7 per §7's ``DecisionEvent.stage`` enum — see
``run_verification`` below). This is orchestration only: the matching
logic in stage1_exact.py / stage2_constrained.py / stage3_aggregate.py /
stage4_adjustment.py / stage5_fuzzy.py / stage6_llm.py, and the
verification logic in verification/verifier.py, are all untouched by
this module.

After verification, a group's ``status`` reflects the verifier's
outcome — VERIFIED, PENDING_REVIEW (ambiguous, e.g. insufficient
margin, or any STAGE6_LLM proposal per §2), or REJECTED (actively
wrong, e.g. conservation/currency/uniqueness failed) — rather than
every proposal uniformly landing on PENDING_REVIEW the way it did
before Stage 6 (verifier wiring) existed. A REJECTED group's records
are released back to unmatched (removed from ``matched_record_ids``)
so Stage 6 (LLM recommendation) gets a chance to resolve them properly
— see ``run_pipeline``'s two verification passes below. Stages 1-5 are
never re-run against released records within one ``run_pipeline``
call; only Stage 6 sees them.

After Stage 6 and its own verification pass, ``run_pipeline`` runs one
more thing: ``run_final_catchall``, a post-Stage-12 bugfix (see its own
comment below) that generates an explicit ``ReconciliationException`` +
``DecisionEvent`` (also attributed to Stage 7, since it is a final
evaluative pass rather than a new matching capability) for any record
that no stage ever attempted at all — e.g. a standalone ADJUSTMENT
record, which no stage's eligibility rule treats as a seeker or
candidate in its own right. This never changes what any individual
stage decides to attempt; it only ensures such a record is never
silently absent from both ``matched_record_ids`` and the exception
list, per §14's acceptance gate.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Optional

from recon_agent.config import Settings, get_settings
from recon_agent.llm.governor import CallBudgetGovernor
from recon_agent.matching.common import IdAllocator, make_decision_event
from recon_agent.matching.stage1_exact import run_stage1_exact
from recon_agent.matching.stage2_constrained import run_stage2_constrained
from recon_agent.matching.stage3_aggregate import run_stage3_aggregate
from recon_agent.matching.stage4_adjustment import run_stage4_adjustment
from recon_agent.matching.stage5_fuzzy import run_stage5_fuzzy
from recon_agent.matching.stage6_llm import run_stage6_llm
from recon_agent.models import (
    CommitPolicy,
    DecisionEvent,
    DecisionStage,
    ExceptionCategory,
    ExceptionReviewStatus,
    ExceptionSeverity,
    MatchGroup,
    MatchGroupMember,
    MatchGroupStatus,
    NormalizedRecord,
    ProposedBy,
    ReconciliationException,
    VerificationResult,
    VerifiedBy,
)
from recon_agent.verification.verifier import verify_match_group

# ---------------------------------------------------------------------------
# Stage 7 (verification) — REJECTED-outcome -> ExceptionCategory mapping.
#
# §8's taxonomy has no dedicated "verification failed" categories, so this
# is a judgment call, documented here rather than left implicit:
#
#   FAILED_CURRENCY     -> CURRENCY_MISMATCH
#       Direct, unambiguous fit — the category name literally matches
#       what the check re-derives (member records disagree on currency).
#   FAILED_UNIQUENESS   -> REUSED_COUNTERPART
#       The check that fails here is "two or more members occupy the same
#       (source, entity_type) slot" — i.e. a counterpart record has been
#       claimed more than once for a slot that must be unique. That is
#       exactly what REUSED_COUNTERPART names.
#   FAILED_CONSERVATION -> PARTIAL_SETTLEMENT
#       This check fires when the recomputed conservation equation
#       (gross - fees - GST - refunds +/- adjustments vs. the credit/
#       reference leg) doesn't balance within tolerance, when there's no
#       CREDIT/GROSS leg to net against at all, or when a member's
#       role/amount sign is internally contradictory. In every case the
#       group's proposed membership does not actually account for the
#       full monetary picture it claims to settle — the closest existing
#       category for "this settlement doesn't fully reconcile" is
#       PARTIAL_SETTLEMENT, rather than the fee/GST- or refund/
#       chargeback-specific categories, since this check doesn't know
#       *which* leg is the culprit, only that the sum is off.
#
# Any other verification_result reaching this map (there shouldn't be
# one, given how ``run_verification`` dispatches below) falls back to
# INSUFFICIENT_EVIDENCE rather than raising, since generating a
# less-specific exception is preferable to losing the released records'
# audit trail entirely.
# ---------------------------------------------------------------------------
REJECTION_EXCEPTION_CATEGORY: dict[VerificationResult, ExceptionCategory] = {
    VerificationResult.FAILED_CURRENCY: ExceptionCategory.CURRENCY_MISMATCH,
    VerificationResult.FAILED_UNIQUENESS: ExceptionCategory.REUSED_COUNTERPART,
    VerificationResult.FAILED_CONSERVATION: ExceptionCategory.PARTIAL_SETTLEMENT,
}


@dataclass
class MatchingResult:
    match_groups: list[MatchGroup] = field(default_factory=list)
    match_group_members: list[MatchGroupMember] = field(default_factory=list)
    decision_events: list[DecisionEvent] = field(default_factory=list)
    exceptions: list[ReconciliationException] = field(default_factory=list)
    matched_record_ids: set[str] = field(default_factory=set)
    # Informational: which stage proposed each group, for summaries/tests.
    group_id_to_stage: dict[str, str] = field(default_factory=dict)
    # Informational: records a REJECTED group's verification released
    # back to unmatched, for summaries/tests. Always empty on the
    # pre-verification MatchingResult that run_stage1_and_stage2 returns.
    released_record_ids: set[str] = field(default_factory=set)


def run_stage1_and_stage2(
    records: list[NormalizedRecord],
    settings: Optional[Settings] = None,
) -> MatchingResult:
    """Stage 1 + Stage 2 only — kept for callers that just want the
    identifier-based and constrained-composite passes without the
    aggregation stages. ``run_pipeline`` is the full Stage 1-4 (+
    verification) entry point. Deliberately does NOT run verification —
    callers that want verified output should use ``run_pipeline``.
    """
    settings = settings if settings is not None else get_settings()
    ids = IdAllocator()

    stage1_result = run_stage1_exact(records, settings, ids)
    stage2_result = run_stage2_constrained(records, settings, stage1_result, ids)

    return MatchingResult(
        match_groups=stage2_result.match_groups,
        match_group_members=stage2_result.match_group_members,
        decision_events=list(stage1_result.decision_events) + list(stage2_result.decision_events),
        matched_record_ids=stage2_result.matched_record_ids,
    )


def run_verification(
    records: list[NormalizedRecord],
    settings: Settings,
    proposed: MatchingResult,
    ids: IdAllocator,
) -> MatchingResult:
    """Stage 7 — run every proposed MatchGroup through the Financial and
    Evidence Verifier and apply its outcome (ARCHITECTURE.md §2):

      PASSED         -> VERIFIED; records stay matched.
      FAILED_MARGIN / RESIDUAL_EXCEEDS_TOLERANCE
                     -> PENDING_REVIEW, commit_policy=HUMAN_REVIEW_REQUIRED;
                        ambiguous, not disproven — records stay matched
                        (claimed pending review, not released).
                        RESIDUAL_EXCEEDS_TOLERANCE is Stage 1's own
                        analog of FAILED_MARGIN: a bounded residual
                        check on Stage 1's identifier-match groups (a
                        real match with an unexplained monetary gap
                        beyond the configured tolerance, still real
                        evidence, not disproven).
      FAILED_CONSERVATION / FAILED_CURRENCY / FAILED_UNIQUENESS
                     -> REJECTED; an actively wrong proposal, so a
                        ReconciliationException is raised and the
                        group's records are released back to unmatched.

    A single pass: this does not re-run Stages 1-5 against records a
    REJECTED outcome releases — those simply remain unmatched, for
    Stage 6's LLM recommendation to pick up.
    """
    records_by_id = {r.record_id: r for r in records}
    members_by_group: dict[str, list[MatchGroupMember]] = {}
    for member in proposed.match_group_members:
        members_by_group.setdefault(member.group_id, []).append(member)

    verified_groups: list[MatchGroup] = []
    decision_events = list(proposed.decision_events)
    exceptions = list(proposed.exceptions)
    matched_record_ids = set(proposed.matched_record_ids)
    released_record_ids: set[str] = set()

    for group in proposed.match_groups:
        members = members_by_group.get(group.group_id, [])
        outcome = verify_match_group(group, members, records_by_id, settings)

        decision_events.append(
            make_decision_event(
                ids,
                group.group_id,
                DecisionStage.STAGE7_VERIFICATION,
                candidate_scores=dict(outcome.checks),
                reason_code=outcome.reason_code,
                explanation=outcome.explanation,
            )
        )

        if outcome.verification_result == VerificationResult.PASSED:
            # commit_policy is left as whatever the proposing stage already
            # set (AUTO_COMMIT_STAGE1 / AUTO_COMMIT_STAGE2_5_THRESHOLD) —
            # that value already reflects the correct auto-commit tier for
            # a group that clears verification (§2).
            verified_groups.append(
                group.model_copy(
                    update={
                        "status": MatchGroupStatus.VERIFIED,
                        "verified_by": VerifiedBy.FINANCIAL_AND_EVIDENCE_VERIFIER,
                        "verification_result": outcome.verification_result,
                    }
                )
            )
            continue

        if outcome.verification_result in (
            VerificationResult.FAILED_MARGIN,
            VerificationResult.RESIDUAL_EXCEEDS_TOLERANCE,
        ):
            # Ambiguous, not disproven — a human should look at it. Records
            # stay claimed (matched_record_ids untouched) rather than
            # released, per this stage's scope. RESIDUAL_EXCEEDS_TOLERANCE
            # is Stage 1's own bounded-residual analog of FAILED_MARGIN —
            # see verify_match_group's own docstring/verifier.py.
            verified_groups.append(
                group.model_copy(
                    update={
                        "status": MatchGroupStatus.PENDING_REVIEW,
                        "verified_by": VerifiedBy.FINANCIAL_AND_EVIDENCE_VERIFIER,
                        "commit_policy": CommitPolicy.HUMAN_REVIEW_REQUIRED,
                        "verification_result": outcome.verification_result,
                    }
                )
            )
            continue

        if group.proposed_by == ProposedBy.STAGE6_LLM:
            # §2: Stage 6 has no auto-commit path, ever. verify_match_group's
            # _verify_stage6 branch deliberately returns
            # verification_result=NOT_YET_RUN / verified_by=NOT_YET_VERIFIED
            # (see its own docstring) rather than PASSED/FAILED_MARGIN/
            # FAILED_* — only a human reviewer's action ever changes
            # verified_by away from NOT_YET_VERIFIED for an LLM proposal.
            # Routed explicitly here rather than falling into the REJECTED
            # branch below: REJECTED is reserved for outcomes the verifier
            # positively found wrong, and a Stage 6 proposal was never run
            # through automated policy checks at all, so REJECTED would
            # misrepresent it as actively disproven. Records stay claimed
            # (matched_record_ids untouched) — a PENDING_REVIEW proposal
            # awaiting a human is not the same as a disproven one.
            verified_groups.append(
                group.model_copy(
                    update={
                        "status": MatchGroupStatus.PENDING_REVIEW,
                        "verified_by": VerifiedBy.NOT_YET_VERIFIED,
                        "commit_policy": CommitPolicy.HUMAN_REVIEW_REQUIRED,
                        "verification_result": outcome.verification_result,
                    }
                )
            )
            continue

        # Everything else the verifier can return here (FAILED_CONSERVATION,
        # FAILED_CURRENCY, FAILED_UNIQUENESS) is an actively wrong proposal,
        # not merely an unconfirmed one: REJECTED, an exception is raised,
        # and its records are released back to unmatched. commit_policy is
        # deliberately left as-is: the enum has no "not applicable" value,
        # and status=REJECTED / verification_result already carry the real
        # disposition — commit_policy only ever mattered for a group that
        # was actually going to commit.
        rejected_group = group.model_copy(
            update={
                "status": MatchGroupStatus.REJECTED,
                "verified_by": VerifiedBy.FINANCIAL_AND_EVIDENCE_VERIFIER,
                "verification_result": outcome.verification_result,
            }
        )
        verified_groups.append(rejected_group)

        member_record_ids = [m.record_id for m in members]
        for record_id in member_record_ids:
            matched_record_ids.discard(record_id)
            released_record_ids.add(record_id)

        category = REJECTION_EXCEPTION_CATEGORY.get(
            outcome.verification_result, ExceptionCategory.INSUFFICIENT_EVIDENCE
        )
        exceptions.append(
            ReconciliationException(
                group_id_or_record_id=group.group_id,
                category=category,
                severity=ExceptionSeverity.HIGH,
                evidence={
                    "verification_result": outcome.verification_result.value,
                    "reason_code": outcome.reason_code,
                    "checks": outcome.checks,
                    "released_record_ids": member_record_ids,
                },
                recommended_action=(
                    "Financial and Evidence Verifier rejected this proposed "
                    f"group ({outcome.reason_code}): {outcome.explanation} "
                    "Records have been released back to unmatched for a "
                    "later stage (LLM recommendation) to resolve, rather "
                    "than left silently matched or lost."
                ),
                review_status=ExceptionReviewStatus.OPEN,
            )
        )

    return MatchingResult(
        match_groups=verified_groups,
        match_group_members=proposed.match_group_members,
        decision_events=decision_events,
        exceptions=exceptions,
        matched_record_ids=matched_record_ids,
        group_id_to_stage=proposed.group_id_to_stage,
        released_record_ids=released_record_ids,
    )


# ---------------------------------------------------------------------------
# Final catch-all pass (post-Stage-12 bugfix) — §14's acceptance gate
# requires "every unresolved record has a reason code, evidence, and a
# recommended action." Stages 1-6 only ever attempt a record that some
# stage's own eligibility rule recognizes as a seeker or a candidate —
# per common.py's ENTITY_TO_ROLE / LEDGER_TO_GATEWAY_ENTITY /
# GATEWAY_TO_BANK_ENTITY tables and stage6_llm.py's own target-entity
# mapping, a standalone ADJUSTMENT record is never one (adjustments are
# only ever consumed as a MEMBER inside another group's conservation
# check — Stage 4's role, not a role of their own). Nothing about that
# is wrong: these are honest_abstention cases with no real counterpart,
# and the system correctly never force-matches them. But nothing ever
# explicitly declines them either — they fall through every stage
# invisibly, with zero DecisionEvents and zero Exceptions anywhere in
# the output, which is exactly the silent-drop §14 forbids.
#
# This pass runs once, after Stage 6 and its own verification pass, and
# only ever *adds* an exception + decision event for a record that is
# still not in the final matched_record_ids AND does not already have
# its own ReconciliationException — it never inspects, reclassifies, or
# overrides an existing decision.
#
# Bugfix (found against real hand-crafted data, not synthetic
# calibration/evaluation data — see BUILD_LOG.md): this pass used to key
# its "already handled" check off DecisionEvents rather than
# Exceptions — any DecisionEvent anywhere that merely *mentioned* a
# record's id (e.g. as one of several near-miss candidates offered to a
# completely different seeker, or as a rejected aggregation target with
# 0 eligible subsets) was enough to skip it, even though nothing had
# ever raised a real Exception naming that record as its own subject.
# A genuinely orphaned BANK_CREDIT record — one Stage 3/4's aggregation
# search correctly declines with NO_AGGREGATION_FOUND (an honest "found
# nothing" outcome that deliberately doesn't raise
# Exception: AMBIGUOUS_AGGREGATION, since there's nothing ambiguous
# about zero candidates) but that also gets mentioned inside some other
# record's own decision event along the way — could end up with a
# DecisionEvent trail but *zero* Exceptions and still be silently
# skipped here. That's exactly the silent-drop §14 forbids, just one
# layer deeper than the original post-Stage-12 bug. The fix: check
# coverage by Exception, not by DecisionEvent — an Exception is always
# genuinely about its subject (a real match group's records via
# ``evidence["released_record_ids"]``, or a record named directly as
# ``group_id_or_record_id``), never a side mention.
#
# "Already covered by an Exception" is checked structurally rather than
# by re-deriving each stage's own eligibility logic (which would
# duplicate, and could drift from, Stages 1-6's real behavior): every
# Exception's group_id_or_record_id, category, recommended_action, and
# evidence are searched for the record id as a literal substring. This
# is safe because every dataset's record ids share one fixed width
# (e.g. "evl_rec_00282", "cal_rec_00001") within a single pipeline run,
# so one record id can never be a substring of another — a plain
# substring match cannot produce a false positive here, and it
# naturally covers every shape Stages 1-6 already use to name a record
# in an Exception (a bare record_id, a ``f"rec:{record_id}"`` aggregation
# target_id, a match group's ``evidence["released_record_ids"]`` list,
# or a record_id embedded in free-text evidence/recommended_action).
#
# DecisionStage has no dedicated "final catch-all" value, and adding one
# would violate enums.py's own "transcribed directly from
# ARCHITECTURE.md — do not add, remove, or rename values" rule. Of the
# existing values, STAGE7_VERIFICATION fits best: like the verifier's
# own pass, this is not a new matching capability proposing candidate
# groups, it's a final evaluative pass over the whole record pool that
# runs after every matching stage — the same role STAGE7_VERIFICATION
# already plays for proposed groups, just extended to cover records no
# stage ever proposed anything for at all.
# ---------------------------------------------------------------------------

_UNATTEMPTED_REASON_CODE = "NO_STAGE_ATTEMPTED"


def _exception_blob(exc: ReconciliationException) -> str:
    return " ".join(
        (
            exc.group_id_or_record_id,
            exc.category.value,
            exc.recommended_action,
            json.dumps(exc.evidence, default=str),
        )
    )


def run_final_catchall(
    records: list[NormalizedRecord],
    matched_record_ids: set[str],
    exceptions: list[ReconciliationException],
    ids: IdAllocator,
) -> tuple[list[DecisionEvent], list[ReconciliationException]]:
    """Stage 7 (final safety-net pass) — see the module-level comment
    above. Returns the new ``DecisionEvent``/``ReconciliationException``
    rows only (empty when every record was already matched or already
    covered by its own Exception); callers append these to the
    pipeline's own lists rather than this function mutating them.

    ``exceptions`` is every ``ReconciliationException`` already raised
    by Stages 1-7 (never ``decision_events`` — see the bugfix note in
    this module's own header comment above for why a DecisionEvent, on
    its own, is not a reliable "already handled" signal).
    """
    candidates = [r for r in records if r.record_id not in matched_record_ids]
    if not candidates:
        return [], []

    blobs = [_exception_blob(e) for e in exceptions]

    new_decision_events: list[DecisionEvent] = []
    new_exceptions: list[ReconciliationException] = []

    for record in candidates:
        if any(record.record_id in blob for blob in blobs):
            # Some stage already raised a real Exception naming this
            # record as its own subject — that Exception already covers
            # it; this is purely a safety net for records that ended up
            # with zero Exceptions anywhere, however many DecisionEvents
            # merely mention them along the way.
            continue

        explanation = (
            f"{record.record_id} (source={record.source.value}, "
            f"entity_type={record.entity_type.value}) was never attempted "
            "by any matching stage: no stage's eligibility rule treats a "
            "standalone record of this shape as a seeker or a candidate "
            "counterpart in its own right. Recorded here so it is never "
            "silently absent from both matched_record_ids and the "
            "exception list."
        )
        new_decision_events.append(
            make_decision_event(
                ids,
                f"unmatched:{record.record_id}",
                DecisionStage.STAGE7_VERIFICATION,
                candidate_scores={
                    "record_id": record.record_id,
                    "source": record.source.value,
                    "entity_type": record.entity_type.value,
                },
                reason_code=_UNATTEMPTED_REASON_CODE,
                explanation=explanation,
            )
        )
        new_exceptions.append(
            ReconciliationException(
                group_id_or_record_id=record.record_id,
                category=ExceptionCategory.INSUFFICIENT_EVIDENCE,
                severity=ExceptionSeverity.MEDIUM,
                evidence={
                    "record_id": record.record_id,
                    "source": record.source.value,
                    "entity_type": record.entity_type.value,
                    "reason": _UNATTEMPTED_REASON_CODE,
                },
                recommended_action=(
                    "No matching stage found a plausible counterpart or "
                    f"role to attempt for {record.record_id}; route to "
                    "manual review rather than leaving it invisible."
                ),
                review_status=ExceptionReviewStatus.OPEN,
            )
        )

    return new_decision_events, new_exceptions


def run_pipeline(
    records: list[NormalizedRecord],
    settings: Optional[Settings] = None,
    run_id: Optional[str] = None,
    governor: Optional[CallBudgetGovernor] = None,
    groq_client: object = None,
) -> MatchingResult:
    """Full Stage 1 -> 2 -> 3 -> 4 -> 5 -> 7 (verification) -> 6 (LLM) ->
    7 (verification again, for Stage 6's own proposals) pipeline.

    Each matching stage only works on records the earlier stages left
    unmatched (Stage 3/4 also extend Stage 1/2's PENDING_REVIEW groups
    in place when adding a missing bank leg, rather than proposing a
    disjoint duplicate group for the same records; Stage 5 never
    extends an earlier group — see stage5_fuzzy.py's own docstring).

    Stage 1-5's proposals are verified first (Stage 7), which is what
    determines the *real* residual pool Stage 6 receives per §2 ("Stage
    6 only ever receives residue that Stages 1-5 already failed to
    safely commit") — a REJECTED Stage 1-5 group's records are released
    back to unmatched by that first verification pass and become
    eligible Stage 6 seekers, exactly like a record Stages 1-5 never
    proposed anything for at all. Stage 6's own proposals are then run
    through the same verifier a second time (its ``_verify_stage6``
    branch is a deliberate no-op pass-through per §2 — see
    ``run_verification``'s STAGE6_LLM branch) purely for audit-trail
    consistency ("same audit discipline as every other stage"), not
    because Stage 6 output needs the automated margin/conservation/
    currency checks Stage 1-5 output does.

    ``run_id`` scopes the circuit breaker's per-run consecutive-failure
    tracking (llm/governor.py); a fresh id is generated if omitted.
    ``governor`` defaults to a ``CallBudgetGovernor`` built from
    ``settings`` — pass one explicitly to share daily-ceiling state
    across multiple ``run_pipeline`` calls, or to inject a governor
    pointed at a test database. ``groq_client`` is forwarded to
    ``get_llm_recommendation`` unchanged (e.g. a mock in tests).
    """
    settings = settings if settings is not None else get_settings()
    run_id = run_id if run_id is not None else uuid.uuid4().hex
    governor = governor if governor is not None else CallBudgetGovernor(settings)
    ids = IdAllocator()

    stage1_result = run_stage1_exact(records, settings, ids)
    stage2_result = run_stage2_constrained(records, settings, stage1_result, ids)

    group_id_to_stage: dict[str, str] = {g.group_id: "STAGE1_EXACT" for g in stage1_result.match_groups}
    for g in stage2_result.match_groups:
        group_id_to_stage.setdefault(g.group_id, "STAGE2_CONSTRAINED")

    stage3_result = run_stage3_aggregate(
        records,
        settings,
        stage2_result.match_groups,
        stage2_result.match_group_members,
        stage2_result.matched_record_ids,
        ids,
    )
    stage3_group_ids = {g.group_id for g in stage3_result.match_groups}
    # Any group id present after Stage 3 that wasn't proposed by Stage
    # 1/2 is a brand-new Stage 3 proposal (merged groups get a fresh id).
    for g in stage3_result.match_groups:
        if g.group_id not in group_id_to_stage:
            group_id_to_stage[g.group_id] = "STAGE3_AGGREGATE"
    # Groups Stage 3 consumed (merged away) no longer exist; drop their
    # stage attribution too, since the merged group now carries its own.
    group_id_to_stage = {gid: stage for gid, stage in group_id_to_stage.items() if gid in stage3_group_ids}

    stage4_result = run_stage4_adjustment(
        records,
        settings,
        stage3_result.match_groups,
        stage3_result.match_group_members,
        stage3_result.matched_record_ids,
        ids,
    )
    stage4_group_ids = {g.group_id for g in stage4_result.match_groups}
    for g in stage4_result.match_groups:
        if g.group_id not in group_id_to_stage:
            group_id_to_stage[g.group_id] = "STAGE4_ADJUSTMENT"
    group_id_to_stage = {gid: stage for gid, stage in group_id_to_stage.items() if gid in stage4_group_ids}

    stage5_result = run_stage5_fuzzy(
        records,
        settings,
        stage4_result.match_groups,
        stage4_result.match_group_members,
        stage4_result.matched_record_ids,
        ids,
    )
    stage5_group_ids = {g.group_id for g in stage5_result.match_groups}
    # Stage 5 never extends an existing group (see stage5_fuzzy.py's own
    # docstring) — every group id present after Stage 5 that wasn't
    # already attributed is a brand-new Stage 5 proposal.
    for g in stage5_result.match_groups:
        if g.group_id not in group_id_to_stage:
            group_id_to_stage[g.group_id] = "STAGE5_FUZZY"
    group_id_to_stage = {gid: stage for gid, stage in group_id_to_stage.items() if gid in stage5_group_ids}

    proposed = MatchingResult(
        match_groups=stage5_result.match_groups,
        match_group_members=stage5_result.match_group_members,
        decision_events=(
            list(stage1_result.decision_events)
            + list(stage2_result.decision_events)
            + list(stage3_result.decision_events)
            + list(stage4_result.decision_events)
            + list(stage5_result.decision_events)
        ),
        exceptions=list(stage3_result.exceptions) + list(stage4_result.exceptions),
        matched_record_ids=stage5_result.matched_record_ids,
        group_id_to_stage=group_id_to_stage,
    )

    verified_1_to_5 = run_verification(records, settings, proposed, ids)

    # ---------------- Stage 6 — LLM recommendation ----------------
    # Only now, after Stage 1-5's own proposals have actually been
    # verified (so a REJECTED group's records are already released
    # back into ``verified_1_to_5.matched_record_ids``'s complement),
    # is the real Stage 6 residual pool known — see this function's own
    # docstring and §2.
    stage6_result = run_stage6_llm(
        records,
        settings,
        verified_1_to_5.matched_record_ids,
        ids,
        governor,
        run_id,
        groq_client=groq_client,
    )

    stage6_group_id_to_stage = {g.group_id: "STAGE6_LLM" for g in stage6_result.match_groups}

    proposed_stage6 = MatchingResult(
        match_groups=stage6_result.match_groups,
        match_group_members=stage6_result.match_group_members,
        decision_events=list(verified_1_to_5.decision_events) + list(stage6_result.decision_events),
        exceptions=list(verified_1_to_5.exceptions) + list(stage6_result.exceptions),
        matched_record_ids=set(verified_1_to_5.matched_record_ids) | stage6_result.matched_record_ids,
        group_id_to_stage=stage6_group_id_to_stage,
    )

    verified_stage6 = run_verification(records, settings, proposed_stage6, ids)

    combined_group_id_to_stage = dict(verified_1_to_5.group_id_to_stage)
    combined_group_id_to_stage.update(stage6_group_id_to_stage)

    # ---------------- Final catch-all pass (post-Stage-12 bugfix) ----------
    # Runs once, after every matching stage and both verification passes,
    # against the *real* final matched_record_ids — see run_final_catchall's
    # own module-level comment above for why this is scoped to records
    # nothing ever touched at all, not a new matching capability.
    catchall_decision_events, catchall_exceptions = run_final_catchall(
        records,
        verified_stage6.matched_record_ids,
        verified_stage6.exceptions,
        ids,
    )

    return MatchingResult(
        match_groups=list(verified_1_to_5.match_groups) + list(verified_stage6.match_groups),
        match_group_members=(
            list(verified_1_to_5.match_group_members) + list(verified_stage6.match_group_members)
        ),
        decision_events=list(verified_stage6.decision_events) + catchall_decision_events,
        exceptions=list(verified_stage6.exceptions) + catchall_exceptions,
        matched_record_ids=verified_stage6.matched_record_ids,
        group_id_to_stage=combined_group_id_to_stage,
        released_record_ids=(
            set(verified_1_to_5.released_record_ids) | set(verified_stage6.released_record_ids)
        ),
    )
