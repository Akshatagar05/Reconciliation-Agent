"""Stage 6 — LLM recommendation, wired into the pipeline — ARCHITECTURE.md
§2, §6, §8, §10. Stage 11 of this 16-stage relay build.

This module is the "whatever calls this function next" that
``llm/recommender.py``'s own docstring and README.md's Stage 10 section
explicitly deferred. It does three things, and three things only:

  1. **Near-miss candidate gathering.** For each record still unmatched
     after Stages 1-5 and the Financial and Evidence Verifier's pass over
     Stages 1-5's proposals (i.e. the real residue Stage 6 is meant to
     receive per §2 — "Stage 6 only ever receives residue that Stages 1-5
     already failed to safely commit"), gather up to
     ``Settings.stage6_max_candidates`` near-miss candidates from the same
     eligible-entity-type pool Stage 5 draws from
     (``matching.common.LEDGER_TO_GATEWAY_ENTITY`` /
     ``GATEWAY_TO_BANK_ENTITY``). This is deliberately a **looser** ranking
     than Stage 5's own hard-gated scoring (see ``_soft_score`` below) —
     "near-miss" per this stage's own brief means *even candidates that
     didn't clear Stage 5's bar*, so this never reuses Stage 5's
     all-or-nothing gates. Currency match is the one hard prefilter kept
     here: a cross-currency pair is categorically not the same
     transaction, not merely a weak candidate for one (mirrors §5's own
     "currency remains a hard pre-filter, never a weighted component").

  2. **Calling the recommender through the governor.** Every attempt
     first asks ``CallBudgetGovernor.try_consume`` (the atomic
     check-and-increment gate, llm/governor.py) whether a call is even
     allowed; only an ALLOW decision results in an actual
     ``get_llm_recommendation`` call. ``recommender.py``'s own logic —
     prompt construction, the API call, response validation — is
     untouched here.

  3. **Turning the result into pipeline objects**, always per §2's
     non-negotiable rule: a Stage 6 proposal that recommends a candidate
     becomes a ``MatchGroup`` with ``proposed_by=STAGE6_LLM``,
     ``status=PENDING_REVIEW``, ``verified_by=NOT_YET_VERIFIED``,
     ``commit_policy=HUMAN_REVIEW_REQUIRED``,
     ``verification_result=NOT_YET_RUN`` — **regardless of the LLM's
     stated confidence**. There is no branch anywhere in this module that
     ever sets a different ``commit_policy`` for a Stage 6 group. A
     ``DecisionEvent`` (stage=STAGE6_LLM) is emitted for every residual
     record this module considers, whatever the outcome: a recommended
     match, a NO_MATCH recommendation, a call failure, no candidates to
     even offer, or the governor declining the call outright — "same
     audit discipline as every other stage" per this stage's brief.

Every path that does NOT produce a ``MatchGroup`` (NO_MATCH, no
candidates, a call failure, or governor degradation) produces a
``ReconciliationException`` instead, so every unresolved record still
carries a reason code, evidence, and recommended action (§14's
acceptance gate) rather than silently vanishing back into "unmatched"
with only a DecisionEvent to explain why.

**A documented, deliberate deviation from this stage's own brief's exact
wording, made explicit rather than silently "fixed":** the brief asks for
degraded records to get exception category ``LLM_UNAVAILABLE`` *or*
``LLM_BUDGET_EXHAUSTED``. §8's Exception Taxonomy — and
``models/enums.py``'s own explicit "do not add, remove, or rename
values, this file is the single source of truth" instruction — only
defines ``LLM_UNAVAILABLE``; there is no ``LLM_BUDGET_EXHAUSTED`` value
anywhere in the schema. Rather than adding an enum member that contradicts
the schema's own stated single-source-of-truth rule (and that §8 itself
labels "unchanged"), both governor-degradation cases here use
``ExceptionCategory.LLM_UNAVAILABLE`` and distinguish *which* kind of
unavailability it was (``CEILING_EXCEEDED`` vs. ``CIRCUIT_OPEN``) in the
exception's own ``evidence``/``recommended_action`` text — the same kind
of documented judgment call ``pipeline.py``'s own
``REJECTION_EXCEPTION_CATEGORY`` table already makes for verification
failures that have no dedicated taxonomy entry either.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from rapidfuzz import fuzz

from recon_agent.config import Settings
from recon_agent.llm.governor import CallBudgetGovernor, GovernorDecision, classify_failure
from recon_agent.llm.recommender import RecommenderError, get_llm_recommendation
from recon_agent.matching.common import (
    GATEWAY_TO_BANK_ENTITY,
    LEDGER_TO_GATEWAY_ENTITY,
    IdAllocator,
    currencies_match,
    date_gap_days,
    make_decision_event,
    make_member,
)
from recon_agent.models import (
    Cardinality,
    CommitPolicy,
    DecisionEvent,
    DecisionStage,
    EntityType,
    ExceptionCategory,
    ExceptionReviewStatus,
    ExceptionSeverity,
    MatchGroup,
    MatchGroupMember,
    MatchGroupStatus,
    NormalizedRecord,
    ProposedBy,
    ReconciliationException,
    Source,
    VerificationResult,
    VerifiedBy,
)
from recon_agent.normalization import normalize_reference

# ---------------------------------------------------------------------------
# Near-miss candidate ranking. Deliberately NOT Stage 5's hard-gated
# ``_score`` (stage5_fuzzy.py) — every component here is a soft [0, 1]
# score with no pass/fail cutoff, so a candidate that fails one or more of
# Stage 5's hard gates can still surface here, ranked lower. Only currency
# is kept as a hard prefilter (see module docstring). Weights mirror Stage
# 5's own composite weights for conceptual consistency (same four signals
# matter for the same reasons), not because Stage 5's module is imported.
# ---------------------------------------------------------------------------
_WEIGHT_IDENTIFIER = 0.30
_WEIGHT_AMOUNT = 0.30
_WEIGHT_DATE = 0.15
_WEIGHT_COUNTERPARTY = 0.25

# A generous date window for *ranking* purposes only (near-miss retrieval,
# not a policy gate) — wide enough that a date-outlier candidate still
# gets a (low) score instead of being silently excluded from the pool
# entirely; the LLM, and ultimately a human reviewer, gets to see it and
# judge for itself.
_SOFT_DATE_WINDOW_DAYS = 30


@dataclass(frozen=True)
class _RankedCandidate:
    record: NormalizedRecord
    soft_score: float


def _soft_identifier_score(seeker: NormalizedRecord, candidate: NormalizedRecord) -> float:
    a = normalize_reference(seeker.reference)
    b = normalize_reference(candidate.reference)
    if not a or not b:
        return 0.0
    return fuzz.ratio(a, b) / 100.0


def _soft_counterparty_score(seeker: NormalizedRecord, candidate: NormalizedRecord) -> float:
    a = normalize_reference(seeker.counterparty)
    b = normalize_reference(candidate.counterparty)
    if not a or not b:
        return 0.0
    return fuzz.ratio(a, b) / 100.0


def _soft_amount_score(seeker: NormalizedRecord, candidate: NormalizedRecord) -> float:
    a, b = abs(seeker.amount_paise), abs(candidate.amount_paise)
    if a == 0 and b == 0:
        return 1.0
    denom = max(a, b, 1)
    relative_gap = abs(a - b) / denom
    return max(0.0, 1.0 - relative_gap)


def _soft_date_score(seeker: NormalizedRecord, candidate: NormalizedRecord) -> float:
    gap = date_gap_days(seeker, candidate)
    return max(0.0, 1.0 - gap / _SOFT_DATE_WINDOW_DAYS)


def _soft_composite(seeker: NormalizedRecord, candidate: NormalizedRecord) -> float:
    return (
        _WEIGHT_IDENTIFIER * _soft_identifier_score(seeker, candidate)
        + _WEIGHT_AMOUNT * _soft_amount_score(seeker, candidate)
        + _WEIGHT_DATE * _soft_date_score(seeker, candidate)
        + _WEIGHT_COUNTERPARTY * _soft_counterparty_score(seeker, candidate)
    )


def _target_entity_type(seeker: NormalizedRecord) -> Optional[EntityType]:
    if seeker.source == Source.LEDGER:
        return LEDGER_TO_GATEWAY_ENTITY.get(seeker.entity_type)
    if seeker.source == Source.GATEWAY:
        return GATEWAY_TO_BANK_ENTITY.get(seeker.entity_type)
    return None


def _candidate_pool(
    seeker: NormalizedRecord, unmatched: list[NormalizedRecord]
) -> list[NormalizedRecord]:
    target_entity = _target_entity_type(seeker)
    if target_entity is None:
        return []
    target_source = Source.GATEWAY if seeker.source == Source.LEDGER else Source.BANK
    return [
        r
        for r in unmatched
        if r.source == target_source
        and r.entity_type == target_entity
        and r.record_id != seeker.record_id
        and currencies_match(seeker, r)
    ]


def gather_near_miss_candidates(
    seeker: NormalizedRecord,
    unmatched: list[NormalizedRecord],
    max_candidates: int,
) -> list[NormalizedRecord]:
    """Top-``max_candidates`` near-miss candidates for ``seeker`` — see
    module docstring for why this is deliberately looser than Stage 5's
    own hard-gated scoring. Deterministic ordering (ties broken by
    ``record_id``) mirrors every other stage's own reproducibility
    policy (§4)."""
    pool = _candidate_pool(seeker, unmatched)
    ranked = sorted(
        (
            _RankedCandidate(record=c, soft_score=_soft_composite(seeker, c))
            for c in pool
        ),
        key=lambda rc: (-rc.soft_score, rc.record.record_id),
    )
    return [rc.record for rc in ranked[:max_candidates]]


def _eligible_seekers_by_source(
    unmatched: list[NormalizedRecord],
) -> tuple[list[NormalizedRecord], list[NormalizedRecord]]:
    """Only LEDGER/GATEWAY records with a defined target entity type are
    eligible Stage 6 seekers — the same seeker roles Stage 5 itself
    considers (stage5_fuzzy.py Pass A / Pass B). An unmatched BANK record
    was never a Stage 5 seeker either (it's only ever a candidate), so
    Stage 6 does not newly invent a seeker role for it here.

    Returns ``(ledger_seekers, gateway_seekers)`` — each list stably
    sorted by ``record_id`` — instead of one combined, alphabetically
    interleaved list. Ordering reflects the actual reconciliation
    hierarchy, not an arbitrary alphabetical sort: a GATEWAY record is
    legitimately eligible to seek its own BANK-side match (mirrors
    Stage 5's own convention, per this module's docstring), but it is
    *also* a legitimate match candidate — and near-miss candidate offer
    — for a LEDGER seeker. Interleaving the two roles alphabetically
    meant a GATEWAY record could be processed as its own seeker before
    the LEDGER seeker that would otherwise consider it ever got a turn,
    producing a decision event/exception for that GATEWAY record (most
    visibly "NO_CANDIDATES_AVAILABLE", but the same staleness applies to
    any outcome) that's immediately made stale, moments later in the
    same pass, once the LEDGER seeker runs and either claims the record
    as its match or simply offers it to the LLM as a candidate. Nothing
    retracts the earlier entry.

    ``run_stage6_llm`` processes every LEDGER seeker in the returned
    order first, then processes only the GATEWAY seekers that were
    never offered as a near-miss candidate to any LEDGER seeker along
    the way — see ``_offered_to_ledger`` there. This prevents the stale
    entry from ever being generated, rather than generating and later
    retracting it — no retroactive mutation of already-emitted
    DecisionEvents."""
    eligible = [r for r in unmatched if _target_entity_type(r) is not None]
    ledger_seekers = sorted(
        (r for r in eligible if r.source == Source.LEDGER),
        key=lambda r: r.record_id,
    )
    gateway_seekers = sorted(
        (r for r in eligible if r.source == Source.GATEWAY),
        key=lambda r: r.record_id,
    )
    return ledger_seekers, gateway_seekers


@dataclass
class Stage6Result:
    match_groups: list[MatchGroup] = field(default_factory=list)
    match_group_members: list[MatchGroupMember] = field(default_factory=list)
    decision_events: list[DecisionEvent] = field(default_factory=list)
    exceptions: list[ReconciliationException] = field(default_factory=list)
    matched_record_ids: set[str] = field(default_factory=set)
    # Informational, for summaries/tests: how many eligible seekers Stage
    # 6 attempted a Groq call for, and the outcome breakdown.
    attempted_count: int = 0
    recommended_count: int = 0
    no_match_count: int = 0
    failure_count: int = 0
    degraded_count: int = 0
    no_candidates_count: int = 0


def _exception(
    record_id: str,
    category: ExceptionCategory,
    severity: ExceptionSeverity,
    evidence: dict,
    recommended_action: str,
) -> ReconciliationException:
    return ReconciliationException(
        group_id_or_record_id=record_id,
        category=category,
        severity=severity,
        evidence=evidence,
        recommended_action=recommended_action,
        review_status=ExceptionReviewStatus.OPEN,
    )


def run_stage6_llm(
    records: list[NormalizedRecord],
    settings: Settings,
    matched_record_ids: set[str],
    ids: IdAllocator,
    governor: CallBudgetGovernor,
    run_id: str,
    groq_client: object = None,
) -> Stage6Result:
    """Stage 6 — for every record still unmatched after Stages 1-5 and
    verification, gather near-miss candidates and call the recommender
    through the governor.

    ``matched_record_ids`` is the cumulative matched set *after* Stage
    1-5's proposals have already been run through the Financial and
    Evidence Verifier (so records a REJECTED verification released are
    correctly included in this stage's residual pool, per §2's framing
    of Stage 6 as receiving only genuine residue). ``groq_client`` is
    forwarded to ``get_llm_recommendation`` unchanged (e.g. a mock in
    tests, or a pre-constructed ``groq.Groq`` client); omit it to let
    ``recommender.py`` construct one from ``settings.groq_api_key``.
    """
    unmatched = [r for r in records if r.record_id not in matched_record_ids]
    unmatched_by_id = {r.record_id: r for r in unmatched}

    # §14's acceptance gate: "the system completes when the Groq key is
    # missing, exhausted, or returns malformed JSON." A missing key (no
    # injected client AND no configured settings.groq_api_key) is
    # checked up front, once, rather than per-record — there is no
    # scenario where the key becomes present partway through one
    # ``run_stage6_llm`` call, and skipping straight to graceful
    # degradation avoids ever constructing a Groq client that
    # ``get_llm_recommendation`` (untouched — see module docstring)
    # would otherwise have to fail out of on every single record.
    llm_configured = groq_client is not None or bool(settings.groq_api_key)

    result = Stage6Result()
    newly_matched: set[str] = set()
    # Every GATEWAY record ID that turns up in *any* LEDGER seeker's
    # near-miss candidate list this pass, whether or not that LEDGER
    # seeker's attempt ends in a recommended match. Once a LEDGER
    # seeker has had a look at it, that record already has a Stage 6
    # audit trail entry (the LEDGER seeker's own DecisionEvent
    # references it in ``candidate_scores``) — giving it a second,
    # independent seeker turn in the same pass would only add a
    # redundant, easily-stale entry for a record that a LEDGER seeker
    # already got first crack at. See ``_eligible_seekers_by_source``.
    offered_to_ledger: set[str] = set()

    ledger_seekers, gateway_seekers = _eligible_seekers_by_source(unmatched)

    def _run_seeker(seeker: NormalizedRecord, *, track_offered: bool) -> None:
        pool = [
            r
            for r in unmatched
            if r.record_id not in newly_matched or r.record_id == seeker.record_id
        ]
        candidates = gather_near_miss_candidates(
            seeker, pool, settings.stage6_max_candidates
        )
        candidates = [c for c in candidates if c.record_id not in newly_matched]
        if track_offered:
            offered_to_ledger.update(c.record_id for c in candidates)

        if not candidates:
            result.no_candidates_count += 1
            result.decision_events.append(
                make_decision_event(
                    ids,
                    f"unmatched:{seeker.record_id}",
                    DecisionStage.STAGE6_LLM,
                    candidate_scores={"seeker_record_id": seeker.record_id, "candidate_pool_size": 0},
                    reason_code="NO_CANDIDATES_AVAILABLE",
                    explanation=(
                        f"No eligible near-miss candidates found for "
                        f"{seeker.record_id}; the LLM was not called."
                    ),
                )
            )
            result.exceptions.append(
                _exception(
                    seeker.record_id,
                    ExceptionCategory.INSUFFICIENT_EVIDENCE,
                    ExceptionSeverity.MEDIUM,
                    evidence={"stage": "STAGE6_LLM", "reason": "NO_CANDIDATES_AVAILABLE"},
                    recommended_action=(
                        "No candidate records were available to offer the LLM "
                        f"for {seeker.record_id}; route to manual investigation."
                    ),
                )
            )
            return

        if not llm_configured:
            result.degraded_count += 1
            result.decision_events.append(
                make_decision_event(
                    ids,
                    f"unmatched:{seeker.record_id}",
                    DecisionStage.STAGE6_LLM,
                    candidate_scores={
                        "seeker_record_id": seeker.record_id,
                        "candidate_pool_size": len(candidates),
                    },
                    reason_code="LLM_NOT_CONFIGURED",
                    explanation=(
                        f"No Groq client/API key configured; skipping the LLM "
                        f"call for {seeker.record_id} rather than failing out "
                        "of client construction on every record."
                    ),
                )
            )
            result.exceptions.append(
                _exception(
                    seeker.record_id,
                    ExceptionCategory.LLM_UNAVAILABLE,
                    ExceptionSeverity.MEDIUM,
                    evidence={"stage": "STAGE6_LLM", "degradation_reason": "LLM_NOT_CONFIGURED"},
                    recommended_action=(
                        f"LLM recommendation unavailable for {seeker.record_id} "
                        "(no Groq API key configured); route to manual "
                        "investigation or configure GROQ_API_KEY."
                    ),
                )
            )
            return

        result.attempted_count += 1
        governor_check = governor.try_consume(run_id)
        if not governor_check.allowed:
            result.degraded_count += 1
            reason = (
                "LLM_BUDGET_EXHAUSTED"
                if governor_check.decision == GovernorDecision.CEILING_EXCEEDED
                else "LLM_CIRCUIT_BREAKER_OPEN"
            )
            result.decision_events.append(
                make_decision_event(
                    ids,
                    f"unmatched:{seeker.record_id}",
                    DecisionStage.STAGE6_LLM,
                    candidate_scores={
                        "seeker_record_id": seeker.record_id,
                        "candidate_pool_size": len(candidates),
                        "governor_decision": governor_check.decision.value,
                        "calls_used_today": governor_check.calls_used_today,
                        "ceiling": governor_check.ceiling,
                        "consecutive_failures": governor_check.consecutive_failures,
                    },
                    reason_code=reason,
                    explanation=(
                        f"Governor declined the Groq call for {seeker.record_id} "
                        f"({reason}); degrading gracefully rather than "
                        "attempting the call or stalling the pipeline."
                    ),
                )
            )
            # See module docstring's "documented deviation" note: the
            # taxonomy (§8) only defines LLM_UNAVAILABLE; both
            # degradation reasons use it, distinguished in evidence/
            # recommended_action rather than inventing an
            # LLM_BUDGET_EXHAUSTED enum value the schema doesn't have.
            result.exceptions.append(
                _exception(
                    seeker.record_id,
                    ExceptionCategory.LLM_UNAVAILABLE,
                    ExceptionSeverity.MEDIUM,
                    evidence={
                        "stage": "STAGE6_LLM",
                        "degradation_reason": reason,
                        "governor_decision": governor_check.decision.value,
                        "calls_used_today": governor_check.calls_used_today,
                        "ceiling": governor_check.ceiling,
                        "consecutive_failures": governor_check.consecutive_failures,
                    },
                    recommended_action=(
                        f"LLM recommendation unavailable for {seeker.record_id} "
                        f"({reason}); route to manual investigation rather than "
                        "leave unresolved with no reason code."
                    ),
                )
            )
            return

        try:
            recommendation = get_llm_recommendation(
                seeker, candidates, settings, client=groq_client
            )
        except RecommenderError as exc:
            governor.record_failure(run_id, exc)
            result.failure_count += 1
            failure_kind = classify_failure(exc)
            result.decision_events.append(
                make_decision_event(
                    ids,
                    f"unmatched:{seeker.record_id}",
                    DecisionStage.STAGE6_LLM,
                    candidate_scores={
                        "seeker_record_id": seeker.record_id,
                        "candidate_pool_size": len(candidates),
                        "offered_candidate_ids": [c.record_id for c in candidates],
                    },
                    reason_code=f"LLM_CALL_FAILED_{failure_kind}",
                    explanation=f"Groq call failed for {seeker.record_id}: {exc}",
                )
            )
            result.exceptions.append(
                _exception(
                    seeker.record_id,
                    ExceptionCategory.LLM_UNAVAILABLE,
                    ExceptionSeverity.MEDIUM,
                    evidence={
                        "stage": "STAGE6_LLM",
                        "failure_kind": failure_kind,
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                    },
                    recommended_action=(
                        f"LLM recommendation call failed for {seeker.record_id} "
                        f"({failure_kind}); route to manual investigation."
                    ),
                )
            )
            return
        except Exception as exc:  # noqa: BLE001 — deliberate last-resort safety net
            # recommender.py's own contract is to "never let a groq-sdk-
            # specific exception type escape this module" for the actual
            # API call — but that guarantee covers the call itself
            # (``_call_groq``), not client *construction*
            # (``Groq(api_key=...)``), which happens one line earlier in
            # ``get_llm_recommendation`` and is outside this module's
            # control. Not fed to ``governor.record_failure`` (which
            # requires a genuine ``RecommenderError`` by design, so the
            # breaker only trips on real recommender failures) — this is
            # tracked only in this record's own audit trail, per §14's
            # "the system completes ... when the Groq key is missing"
            # and the broader "never stall or crash" requirement.
            result.failure_count += 1
            failure_kind = "UNEXPECTED_ERROR"
            result.decision_events.append(
                make_decision_event(
                    ids,
                    f"unmatched:{seeker.record_id}",
                    DecisionStage.STAGE6_LLM,
                    candidate_scores={
                        "seeker_record_id": seeker.record_id,
                        "candidate_pool_size": len(candidates),
                    },
                    reason_code=f"LLM_CALL_FAILED_{failure_kind}",
                    explanation=(
                        f"Unexpected error calling Groq for {seeker.record_id}: {exc}"
                    ),
                )
            )
            result.exceptions.append(
                _exception(
                    seeker.record_id,
                    ExceptionCategory.LLM_UNAVAILABLE,
                    ExceptionSeverity.MEDIUM,
                    evidence={
                        "stage": "STAGE6_LLM",
                        "failure_kind": failure_kind,
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                    },
                    recommended_action=(
                        f"LLM recommendation call failed unexpectedly for "
                        f"{seeker.record_id}; route to manual investigation."
                    ),
                )
            )
            return

        governor.record_success(run_id)

        if not recommendation.is_match:
            result.no_match_count += 1
            result.decision_events.append(
                make_decision_event(
                    ids,
                    f"unmatched:{seeker.record_id}",
                    DecisionStage.STAGE6_LLM,
                    candidate_scores={
                        "seeker_record_id": seeker.record_id,
                        "candidate_pool_size": len(candidates),
                        "offered_candidate_ids": list(recommendation.offered_candidate_ids),
                        "confidence": recommendation.confidence,
                    },
                    reason_code=recommendation.reason_code,
                    explanation=recommendation.explanation,
                )
            )
            result.exceptions.append(
                _exception(
                    seeker.record_id,
                    ExceptionCategory.INSUFFICIENT_EVIDENCE,
                    ExceptionSeverity.MEDIUM,
                    evidence={
                        "stage": "STAGE6_LLM",
                        "llm_reason_code": recommendation.reason_code,
                        "llm_confidence": recommendation.confidence,
                        "offered_candidate_ids": list(recommendation.offered_candidate_ids),
                    },
                    recommended_action=(
                        f"LLM recommended NO_MATCH for {seeker.record_id} "
                        f"({recommendation.reason_code}); route to manual "
                        "investigation."
                    ),
                )
            )
            return

        # A recommended match. Per §2, no exceptions, ever: commit_policy
        # is always HUMAN_REVIEW_REQUIRED regardless of confidence.
        result.recommended_count += 1
        best = unmatched_by_id[recommendation.candidate_id]
        group_id = ids.next_group_id()
        expected = abs(seeker.amount_paise)
        matched_amt = abs(best.amount_paise)
        group = MatchGroup(
            group_id=group_id,
            cardinality=Cardinality.ONE_TO_ONE,
            expected_amount_paise=expected,
            matched_amount_paise=matched_amt,
            residual_amount_paise=expected - matched_amt,
            status=MatchGroupStatus.PENDING_REVIEW,
            evidence_score=recommendation.confidence,
            runner_up_score=None,
            score_margin=None,
            # No calibrated threshold applies to an LLM recommendation
            # (§2: Stage 6 has no auto-commit path, so there is nothing
            # to threshold against) — 0.0 is a documented placeholder,
            # not a real bar this group was checked against.
            threshold_applied=0.0,
            threshold_version=settings.threshold_version,
            policy_checks={
                "stage": "STAGE6_LLM",
                "llm_confidence": recommendation.confidence,
                "llm_reason_code": recommendation.reason_code,
                "offered_candidate_ids": list(recommendation.offered_candidate_ids),
                "human_review_required": True,
                "auto_commit_eligible": False,
            },
            proposed_by=ProposedBy.STAGE6_LLM,
            verified_by=VerifiedBy.NOT_YET_VERIFIED,
            commit_policy=CommitPolicy.HUMAN_REVIEW_REQUIRED,
            verification_result=VerificationResult.NOT_YET_RUN,
        )
        result.match_groups.append(group)
        result.match_group_members.append(make_member(group_id, seeker))
        result.match_group_members.append(make_member(group_id, best))
        newly_matched.add(seeker.record_id)
        newly_matched.add(best.record_id)

        result.decision_events.append(
            make_decision_event(
                ids,
                group_id,
                DecisionStage.STAGE6_LLM,
                candidate_scores={
                    "seeker_record_id": seeker.record_id,
                    "candidate_pool_size": len(candidates),
                    "offered_candidate_ids": list(recommendation.offered_candidate_ids),
                    "recommended_candidate_id": recommendation.candidate_id,
                    "confidence": recommendation.confidence,
                },
                reason_code=recommendation.reason_code,
                explanation=(
                    f"LLM recommended {best.record_id} as {seeker.record_id}'s "
                    f"match (confidence {recommendation.confidence:.3f}): "
                    f"{recommendation.explanation} Per §2, this proposal is "
                    "HUMAN_REVIEW_REQUIRED regardless of confidence — no "
                    "auto-commit path exists for STAGE6_LLM."
                ),
            )
        )

    # Phase 1: every eligible LEDGER seeker gets first crack at claiming
    # GATEWAY candidates. Phase 2: only the GATEWAY seekers never offered
    # as a candidate to a LEDGER seeker along the way get their own turn
    # seeking a BANK-side match. See ``_eligible_seekers_by_source``.
    for seeker in ledger_seekers:
        if seeker.record_id in newly_matched:
            continue
        _run_seeker(seeker, track_offered=True)

    for seeker in gateway_seekers:
        if seeker.record_id in newly_matched or seeker.record_id in offered_to_ledger:
            continue
        _run_seeker(seeker, track_offered=False)

    result.matched_record_ids = set(newly_matched)
    return result
