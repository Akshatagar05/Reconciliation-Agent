"""Stage 2 — constrained match — ARCHITECTURE.md §6, §2, §5.

Per the v4.1 pre-coding correction to §2: "Stage 2 exists precisely
because there's no clean shared identifier, so it uses the same evidence
bar as Stages 3-5" — a composite score over amount compatibility,
date-window compatibility, currency match, and counterparty similarity,
gated by uniqueness + threshold + margin + a simple amount-tolerance
check. Full fee/GST/refund-netting conservation is explicitly Stage 4's
job (out of scope here); the amount check here is a direct tolerance
comparison only.

Two passes, run in sequence:

  Pass A (ledger <-> gateway): picks up records Stage 1 couldn't cluster
  by identifier — mainly "truncation"/"typo" dirty references, which
  normalization deliberately does not recover (see
  ``normalization.reference``). Matched on amount + date + counterparty
  evidence (see note below).

  Pass B (gateway <-> bank): bank credits never share an identifier with
  anything (they carry a UTR-style batch reference by construction, not
  the payment reference — see §2's own rationale for why Stage 2 exists
  at all), so *every* gateway<->bank link is Stage 2's job. When the
  gateway-side record already belongs to a group (from Stage 1 or Pass
  A), Pass B extends that group with the bank leg rather than creating a
  disjoint duplicate group for the same record — and because that leg's
  evidence is constrained-only, the *whole* group's proposed_by/
  commit_policy is upgraded to the stricter Stage 2-5 bar, never left
  looking like a lighter Stage 1 auto-commit.

  Pass B only ever proposes a single gateway record <-> a single bank
  record. A multi-member consolidated settlement batch (several gateway
  settlements summing to one bank credit) has no single gateway record
  whose amount alone will pass Stage 2's simple tolerance check against
  the batch total — that's the many-to-one aggregation search, and it's
  explicitly Stage 3's job, not this one. Leaving those unmatched here is
  correct, not a gap.

On counterparty evidence: NormalizedRecord (§7, corrected) now carries a
real ``counterparty`` field — the name/account label as it appears on
that source's own record. This replaces the earlier reference-text
proxy: the two records' ``counterparty`` strings are compared directly
(case/punctuation-insensitively) with rapidfuzz, rather than fuzzing
their ``reference`` strings against each other. This is still
deliberately given the smallest of the three composite weights and is
never, by itself, enough to clear the threshold — consistent with §5's
caution that fuzzy string similarity is retrieval evidence only, never a
commit basis alone. Gateway<->bank scoring still leans mostly on
amount + date in practice: bank credits carry the payment aggregator's
own name as counterparty (the aggregator, not the underlying customer,
is who credits the bank), so a customer-named gateway settlement and its
aggregator-named bank credit share little counterparty similarity — that
is exactly how real bank-side reconciliation usually has to work, and is
now a property of the data rather than an artifact of a missing field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from rapidfuzz import fuzz

from recon_agent.config import Settings
from recon_agent.matching.common import (
    GATEWAY_TO_BANK_ENTITY,
    LEDGER_TO_GATEWAY_ENTITY,
    IdAllocator,
    currencies_match,
    date_gap_days,
    make_decision_event,
    make_member,
)
from recon_agent.matching.stage1_exact import Stage1Result
from recon_agent.models import (
    Cardinality,
    CommitPolicy,
    DecisionStage,
    EntityType,
    MatchGroup,
    MatchGroupMember,
    MatchGroupStatus,
    NormalizedRecord,
    ProposedBy,
    Source,
    VerificationResult,
    VerifiedBy,
)
from recon_agent.normalization import normalize_reference

# --- Composite score weights (currency is a hard pre-filter, not a
# weighted component — a currency mismatch is never "partially fine"). ---
WEIGHT_AMOUNT = 0.55
WEIGHT_DATE = 0.30
WEIGHT_COUNTERPARTY = 0.15

# --- Calibrated thresholds, keyed by threshold_version (§7's note that
# thresholds are versioned separately from rules/model). No calibration
# pass against data/calibration/ has actually been run yet in this build
# (that's Day 8 per §13's build sequence) — "v1" is a documented starting
# default, not a tuned value. ---
MATCH_THRESHOLDS: dict[str, float] = {"v1": 0.70}
DEFAULT_THRESHOLD = 0.70
MIN_SCORE_MARGIN = 0.08

# --- Date-window policy. A short grace period scores T+0..T+2 (completely
# normal settlement rhythm) at full marks rather than tapering immediately,
# so routine timing variance doesn't starve the composite score of weight
# it shouldn't lose. ---
LEDGER_GATEWAY_DATE_WINDOW_DAYS = 7
GATEWAY_BANK_DATE_WINDOW_DAYS = 5
DATE_GRACE_DAYS = 2

# --- Amount-compatibility policy. "net_to_net" pairs (refund<->refund,
# chargeback<->chargeback, settlement<->bank credit) should be almost
# exactly equal; "gross_to_net" pairs (ledger PAYMENT <-> gateway
# GATEWAY_SETTLEMENT) differ by the MDR fee + GST, and occasionally a
# small adjustment, deducted between the two. ---
NET_TOLERANCE_PAISE = 200
NET_TOLERANCE_FRACTION = 0.005
MAX_FEE_FRACTION = 0.05  # generator's realistic ceiling is ~2.95%
MAX_POSITIVE_ADJUSTMENT_FRACTION = 0.02


@dataclass
class Stage2Result:
    match_groups: list[MatchGroup] = field(default_factory=list)
    match_group_members: list[MatchGroupMember] = field(default_factory=list)
    decision_events: list = field(default_factory=list)
    matched_record_ids: set[str] = field(default_factory=set)


@dataclass
class _ScoreBreakdown:
    amount_score: float
    amount_ok: bool
    date_score: float
    date_ok: bool
    counterparty_score: float
    composite: float


def _amount_compatibility(
    seeker_amount: int, candidate_amount: int, expects_deduction: bool
) -> tuple[float, bool]:
    a, b = abs(seeker_amount), abs(candidate_amount)
    if expects_deduction:
        # b (net) is expected to be a bit less than a (gross): a fee+GST
        # deduction, and occasionally a small positive adjustment.
        if a == 0:
            return (0.0, False)
        fraction = (a - b) / a
        low = -MAX_POSITIVE_ADJUSTMENT_FRACTION
        high = MAX_FEE_FRACTION
        ok = low <= fraction <= high
        if not ok:
            return (0.0, False)
        span = high - low
        center = low + span / 2
        relative_distance = abs(fraction - center) / (span / 2) if span > 0 else 0.0
        score = 1.0 - 0.15 * min(1.0, relative_distance)
        return (score, True)

    tolerance = max(NET_TOLERANCE_PAISE, NET_TOLERANCE_FRACTION * a)
    diff = abs(a - b)
    ok = diff <= tolerance
    if not ok:
        return (0.0, False)
    relative_distance = diff / tolerance if tolerance > 0 else 0.0
    score = 1.0 - 0.15 * min(1.0, relative_distance)
    return (score, True)


def _date_compatibility(gap_days: int, window_days: int) -> tuple[float, bool]:
    ok = gap_days <= window_days
    if not ok:
        return (0.0, False)
    if gap_days <= DATE_GRACE_DAYS:
        return (1.0, True)
    remaining = max(1, window_days - DATE_GRACE_DAYS)
    score = max(0.0, 1.0 - (gap_days - DATE_GRACE_DAYS) / remaining)
    return (score, True)


def _normalize_counterparty(name: str) -> str:
    """Case/whitespace/punctuation-insensitive form of a counterparty
    label, for comparison only — mirrors ``normalize_reference``'s
    "strip everything non-alphanumeric, fold case" approach, but is kept
    as its own function since counterparty and reference are
    semantically different fields that just happen to normalize the same
    way today; they should not be assumed to stay in lockstep.
    """
    return normalize_reference(name)


def _counterparty_similarity(seeker: NormalizedRecord, candidate: NormalizedRecord) -> float:
    a = _normalize_counterparty(seeker.counterparty)
    b = _normalize_counterparty(candidate.counterparty)
    if not a or not b:
        return 0.0
    return fuzz.ratio(a, b) / 100.0


def _score(
    seeker: NormalizedRecord,
    candidate: NormalizedRecord,
    expects_deduction: bool,
    date_window_days: int,
) -> Optional[_ScoreBreakdown]:
    if not currencies_match(seeker, candidate):
        return None
    amount_score, amount_ok = _amount_compatibility(
        seeker.amount_paise, candidate.amount_paise, expects_deduction
    )
    if not amount_ok:
        return None
    date_score, date_ok = _date_compatibility(
        date_gap_days(seeker, candidate), date_window_days
    )
    if not date_ok:
        return None
    counterparty_score = _counterparty_similarity(seeker, candidate)
    composite = (
        WEIGHT_AMOUNT * amount_score
        + WEIGHT_DATE * date_score
        + WEIGHT_COUNTERPARTY * counterparty_score
    )
    return _ScoreBreakdown(
        amount_score=amount_score,
        amount_ok=amount_ok,
        date_score=date_score,
        date_ok=date_ok,
        counterparty_score=counterparty_score,
        composite=composite,
    )


def _best_candidate(
    seeker: NormalizedRecord,
    candidates: list[NormalizedRecord],
    expects_deduction: bool,
    date_window_days: int,
) -> tuple[Optional[NormalizedRecord], Optional[_ScoreBreakdown], Optional[float]]:
    """Returns (best_candidate, best_breakdown, runner_up_score)."""
    scored: list[tuple[NormalizedRecord, _ScoreBreakdown]] = []
    for candidate in candidates:
        breakdown = _score(seeker, candidate, expects_deduction, date_window_days)
        if breakdown is not None:
            scored.append((candidate, breakdown))
    if not scored:
        return None, None, None
    scored.sort(key=lambda pair: pair[1].composite, reverse=True)
    best_record, best_breakdown = scored[0]
    runner_up_score = scored[1][1].composite if len(scored) > 1 else 0.0
    return best_record, best_breakdown, runner_up_score


def _threshold_for(settings: Settings) -> float:
    return MATCH_THRESHOLDS.get(settings.threshold_version, DEFAULT_THRESHOLD)


def run_stage2_constrained(
    records: list[NormalizedRecord],
    settings: Settings,
    stage1_result: Stage1Result,
    ids: IdAllocator,
) -> Stage2Result:
    threshold = _threshold_for(settings)

    by_id = {r.record_id: r for r in records}
    matched_ids: set[str] = set(stage1_result.matched_record_ids)

    all_groups: dict[str, MatchGroup] = {g.group_id: g for g in stage1_result.match_groups}
    all_members: dict[str, list[MatchGroupMember]] = {}
    record_to_group: dict[str, str] = {}
    for member in stage1_result.match_group_members:
        all_members.setdefault(member.group_id, []).append(member)
        record_to_group[member.record_id] = member.group_id

    decision_events: list = []

    # ---------------- Pass A: ledger <-> gateway ----------------
    unmatched_ledger = sorted(
        (r for r in records if r.source == Source.LEDGER and r.record_id not in matched_ids),
        key=lambda r: r.record_id,
    )
    for seeker in unmatched_ledger:
        target_entity = LEDGER_TO_GATEWAY_ENTITY.get(seeker.entity_type)
        candidate_group_id = ids.next_group_id()
        if target_entity is None:
            continue

        candidates = [
            r
            for r in records
            if r.source == Source.GATEWAY
            and r.entity_type == target_entity
            and r.record_id not in matched_ids
        ]
        expects_deduction = target_entity == EntityType.GATEWAY_SETTLEMENT
        best, breakdown, runner_up = _best_candidate(
            seeker, candidates, expects_deduction, LEDGER_GATEWAY_DATE_WINDOW_DAYS
        )

        candidate_scores = {
            "seeker_record_id": seeker.record_id,
            "num_eligible_candidates": len(candidates),
            "best_candidate_record_id": best.record_id if best else None,
            "amount_score": breakdown.amount_score if breakdown else None,
            "date_score": breakdown.date_score if breakdown else None,
            "counterparty_score": breakdown.counterparty_score if breakdown else None,
            "composite_score": breakdown.composite if breakdown else None,
            "runner_up_score": runner_up,
            "threshold": threshold,
        }

        margin = (breakdown.composite - runner_up) if (breakdown and runner_up is not None) else None
        accepted = (
            best is not None
            and breakdown is not None
            and breakdown.composite >= threshold
            and margin is not None
            and margin >= MIN_SCORE_MARGIN
        )

        if accepted:
            expected = abs(seeker.amount_paise)
            matched_amt = abs(best.amount_paise)
            group = MatchGroup(
                group_id=candidate_group_id,
                cardinality=Cardinality.ONE_TO_ONE,
                expected_amount_paise=expected,
                matched_amount_paise=matched_amt,
                residual_amount_paise=expected - matched_amt,
                status=MatchGroupStatus.PENDING_REVIEW,
                evidence_score=breakdown.composite,
                runner_up_score=runner_up,
                score_margin=margin,
                threshold_applied=threshold,
                threshold_version=settings.threshold_version,
                policy_checks={
                    "currency_consistent": True,
                    "amount_tolerance": True,
                    "date_policy_window": True,
                    "unique_candidate": True,
                    "margin_sufficient": True,
                },
                proposed_by=ProposedBy.STAGE2_CONSTRAINED,
                verified_by=VerifiedBy.NOT_YET_VERIFIED,
                commit_policy=CommitPolicy.AUTO_COMMIT_STAGE2_5_THRESHOLD,
                verification_result=VerificationResult.NOT_YET_RUN,
            )
            all_groups[candidate_group_id] = group
            members = [make_member(candidate_group_id, seeker), make_member(candidate_group_id, best)]
            all_members[candidate_group_id] = members
            for record in (seeker, best):
                matched_ids.add(record.record_id)
                record_to_group[record.record_id] = candidate_group_id

            decision_events.append(
                make_decision_event(
                    ids,
                    candidate_group_id,
                    DecisionStage.STAGE2_CONSTRAINED,
                    candidate_scores=candidate_scores,
                    reason_code="CONSTRAINED_LEDGER_GATEWAY_MATCH",
                    explanation=(
                        f"{seeker.record_id} matched to {best.record_id}: composite "
                        f"score {breakdown.composite:.3f} >= threshold {threshold:.2f}, "
                        f"margin {margin:.3f} over runner-up."
                    ),
                )
            )
        else:
            reason = "NO_ELIGIBLE_CANDIDATE" if best is None else (
                "BELOW_THRESHOLD" if breakdown.composite < threshold else "INSUFFICIENT_MARGIN"
            )
            decision_events.append(
                make_decision_event(
                    ids,
                    candidate_group_id,
                    DecisionStage.STAGE2_CONSTRAINED,
                    candidate_scores=candidate_scores,
                    reason_code=reason,
                    explanation=(
                        f"No confident ledger<->gateway match for {seeker.record_id} "
                        f"({len(candidates)} eligible candidate(s) considered)."
                    ),
                )
            )

    # ---------------- Pass B: gateway <-> bank ----------------
    unmatched_bank = lambda: [  # noqa: E731 - recomputed each iteration, bank pool shrinks
        r for r in records if r.source == Source.BANK and r.record_id not in matched_ids
    ]

    gateway_seekers = sorted(
        (
            r
            for r in records
            if r.source == Source.GATEWAY and r.entity_type in GATEWAY_TO_BANK_ENTITY
        ),
        key=lambda r: r.record_id,
    )
    for seeker in gateway_seekers:
        existing_group_id = record_to_group.get(seeker.record_id)
        if existing_group_id is not None:
            existing_sources = {m.source for m in all_members[existing_group_id]}
            if Source.BANK in existing_sources:
                continue  # this group already has its bank leg

        target_entity = GATEWAY_TO_BANK_ENTITY[seeker.entity_type]
        candidates = [
            r
            for r in unmatched_bank()
            if r.entity_type == target_entity and r.record_id not in matched_ids
        ]
        best, breakdown, runner_up = _best_candidate(
            seeker, candidates, expects_deduction=False, date_window_days=GATEWAY_BANK_DATE_WINDOW_DAYS
        )

        candidate_group_id = existing_group_id if existing_group_id is not None else ids.next_group_id()
        candidate_scores = {
            "seeker_record_id": seeker.record_id,
            "num_eligible_candidates": len(candidates),
            "best_candidate_record_id": best.record_id if best else None,
            "amount_score": breakdown.amount_score if breakdown else None,
            "date_score": breakdown.date_score if breakdown else None,
            "counterparty_score": breakdown.counterparty_score if breakdown else None,
            "composite_score": breakdown.composite if breakdown else None,
            "runner_up_score": runner_up,
            "threshold": threshold,
        }

        margin = (breakdown.composite - runner_up) if (breakdown and runner_up is not None) else None
        accepted = (
            best is not None
            and breakdown is not None
            and breakdown.composite >= threshold
            and margin is not None
            and margin >= MIN_SCORE_MARGIN
        )

        if accepted:
            if existing_group_id is not None:
                prior_group = all_groups[existing_group_id]
                new_members = all_members[existing_group_id] + [make_member(existing_group_id, best)]
                all_members[existing_group_id] = new_members
                matched_amt = abs(best.amount_paise)
                updated_group = prior_group.model_copy(
                    update={
                        "matched_amount_paise": matched_amt,
                        "residual_amount_paise": prior_group.expected_amount_paise - matched_amt,
                        "evidence_score": breakdown.composite,
                        "runner_up_score": runner_up,
                        "score_margin": margin,
                        "threshold_applied": threshold,
                        "threshold_version": settings.threshold_version,
                        "proposed_by": ProposedBy.STAGE2_CONSTRAINED,
                        "commit_policy": CommitPolicy.AUTO_COMMIT_STAGE2_5_THRESHOLD,
                        "policy_checks": {
                            **prior_group.policy_checks,
                            "bank_leg_amount_tolerance": True,
                            "bank_leg_date_policy_window": True,
                            "bank_leg_unique_candidate": True,
                            "bank_leg_margin_sufficient": True,
                        },
                    }
                )
                all_groups[existing_group_id] = updated_group
                matched_ids.add(best.record_id)
                record_to_group[best.record_id] = existing_group_id
                reason_code = "CONSTRAINED_BANK_LEG_ATTACHED"
                explanation = (
                    f"Bank record {best.record_id} attached to existing group "
                    f"{existing_group_id} via gateway record {seeker.record_id}: "
                    f"composite score {breakdown.composite:.3f} >= threshold "
                    f"{threshold:.2f}, margin {margin:.3f}."
                )
            else:
                expected = abs(seeker.amount_paise)
                matched_amt = abs(best.amount_paise)
                group = MatchGroup(
                    group_id=candidate_group_id,
                    cardinality=Cardinality.ONE_TO_ONE,
                    expected_amount_paise=expected,
                    matched_amount_paise=matched_amt,
                    residual_amount_paise=expected - matched_amt,
                    status=MatchGroupStatus.PENDING_REVIEW,
                    evidence_score=breakdown.composite,
                    runner_up_score=runner_up,
                    score_margin=margin,
                    threshold_applied=threshold,
                    threshold_version=settings.threshold_version,
                    policy_checks={
                        "currency_consistent": True,
                        "amount_tolerance": True,
                        "date_policy_window": True,
                        "unique_candidate": True,
                        "margin_sufficient": True,
                    },
                    proposed_by=ProposedBy.STAGE2_CONSTRAINED,
                    verified_by=VerifiedBy.NOT_YET_VERIFIED,
                    commit_policy=CommitPolicy.AUTO_COMMIT_STAGE2_5_THRESHOLD,
                    verification_result=VerificationResult.NOT_YET_RUN,
                )
                all_groups[candidate_group_id] = group
                all_members[candidate_group_id] = [
                    make_member(candidate_group_id, seeker),
                    make_member(candidate_group_id, best),
                ]
                matched_ids.update({seeker.record_id, best.record_id})
                record_to_group[seeker.record_id] = candidate_group_id
                record_to_group[best.record_id] = candidate_group_id
                reason_code = "CONSTRAINED_GATEWAY_BANK_MATCH"
                explanation = (
                    f"{seeker.record_id} matched to {best.record_id}: composite score "
                    f"{breakdown.composite:.3f} >= threshold {threshold:.2f}, margin "
                    f"{margin:.3f} over runner-up."
                )

            decision_events.append(
                make_decision_event(
                    ids,
                    candidate_group_id,
                    DecisionStage.STAGE2_CONSTRAINED,
                    candidate_scores=candidate_scores,
                    reason_code=reason_code,
                    explanation=explanation,
                )
            )
        else:
            reason = "NO_ELIGIBLE_CANDIDATE" if best is None else (
                "BELOW_THRESHOLD" if breakdown.composite < threshold else "INSUFFICIENT_MARGIN"
            )
            decision_events.append(
                make_decision_event(
                    ids,
                    candidate_group_id,
                    DecisionStage.STAGE2_CONSTRAINED,
                    candidate_scores=candidate_scores,
                    reason_code=reason,
                    explanation=(
                        f"No confident gateway<->bank match for {seeker.record_id} "
                        f"({len(candidates)} eligible candidate(s) considered)."
                    ),
                )
            )

    result = Stage2Result(
        match_groups=list(all_groups.values()),
        match_group_members=[m for members in all_members.values() for m in members],
        decision_events=decision_events,
        matched_record_ids=matched_ids,
    )
    return result
