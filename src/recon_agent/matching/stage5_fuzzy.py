"""Stage 5 — fuzzy candidate retrieval — ARCHITECTURE.md §6, §5, §2.

Per §5's corrected policy ("Must-Fix #4"), fuzzy identifier similarity is
**candidate retrieval only, never a commit basis by itself** — a
one-character difference in a payment ID can point at a completely
different transaction. Stage 5 is the last matching tier before the LLM
(Stage 6, not built in this stage) and is deliberately the most
evidence-hungry of the five matching stages: a proposal here requires
**ALL** of the following to hold at once, never any subset:

    fuzzy identifier match (rapidfuzz on normalized ``reference``)
  + same currency
  + amount compatibility (within tolerance)
  + date compatibility (within the policy window)
  + counterparty evidence (rapidfuzz on the real ``counterparty`` field,
    above a real similarity bar — not merely weighted into a composite
    where a bad score could be outvoted by the others)
  + a unique score margin over the runner-up candidate

Any one of these failing declines the pair outright; there is no partial
credit. This is what distinguishes Stage 5 from Stage 2: Stage 2's
composite score (amount + date + counterparty) never uses the
identifier at all (see stage2_constrained.py's own docstring — dirty/
truncated/typo'd references are recovered there through *other*
evidence). Stage 5 instead starts from records Stage 1-4 could not
place at all and asks whether fuzzy identifier similarity, combined
with every other required signal, is strong enough to retrieve — never
assume — a genuine candidate.

Scope, mirroring Stage 2's own two passes (§6): Pass A (ledger<->
gateway) and Pass B (gateway<->bank), using the same
LEDGER_TO_GATEWAY_ENTITY / GATEWAY_TO_BANK_ENTITY eligibility tables
from matching.common. Only records still unmatched after Stage 1-4 (and
Stage 3's conflict resolution, already folded into Stage 3/4's own
matched_record_ids bookkeeping) are considered — Stage 5 proposes brand
new 1:1 pairs among that leftover pool; it does not extend or reopen an
existing Stage 1-4 group, since anything already claimed had stronger
evidence than fuzzy retrieval could add.

Stage 5 is explicitly NOT an aggregation stage: it only ever proposes a
single seeker record <-> a single candidate record (cardinality
ONE_TO_ONE), consistent with §5's framing as retrieval, not search. A
record whose true counterpart is a many-to-one consolidated settlement
that Stage 3/4's subset-sum search already couldn't resolve (window too
large, genuine overlap) is correctly left unmatched here too — that is
a documented aggregation-search limitation (§9), not something a
pairwise identifier-retrieval stage can or should paper over.

Same auto-commit policy as Stages 2-4 (§2): every proposal is
status=PENDING_REVIEW, verified_by=NOT_YET_VERIFIED,
commit_policy=AUTO_COMMIT_STAGE2_5_THRESHOLD (§2's shared Stage 3-5
evidence bar), verification_result=NOT_YET_RUN. Nothing here marks a
group VERIFIED — every proposal still has to clear the Financial and
Evidence Verifier (verification/verifier.py) before that can happen.
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

# ---------------------------------------------------------------------------
# Composite score weights. Unlike Stage 2 (amount 0.55 / date 0.30 /
# counterparty 0.15, no identifier term at all), Stage 5's composite gives
# real weight to BOTH of the two signals that make it distinct from Stage
# 2 -- fuzzy identifier similarity and real counterparty evidence -- since
# §5 names both as required, substantive evidence, not tie-breaking
# garnish. Currency remains a hard pre-filter, never a weighted component.
# ---------------------------------------------------------------------------
WEIGHT_IDENTIFIER = 0.30
WEIGHT_AMOUNT = 0.30
WEIGHT_DATE = 0.15
WEIGHT_COUNTERPARTY = 0.25

# --- Hard gates. Each of these is a required condition per §5 -- failing
# any one declines the candidate outright, regardless of how the other
# signals score. These are deliberately independent of (and stricter
# than) "weight in a composite that could still clear a threshold on the
# strength of other signals alone" -- that is exactly the shortcut §5
# exists to close off. ---

# Fuzzy identifier match: rapidfuzz ratio on the normalized reference.
# Set below 1.0 (an exact normalized match would already have been
# caught by Stage 1) but high enough that "fuzzy" still means "close" --
# a single truncation/typo per testdata/generator.py's own dirty-
# reference model, not two unrelated identifiers that happen to share
# some characters.
STAGE5_MIN_IDENTIFIER_SIMILARITY = 0.72

# Counterparty evidence: rapidfuzz ratio on the normalized counterparty
# field. Per §5's own correction text -- "name/account match, not just
# proximity" -- this is a real similarity bar, not merely a weighted
# input a bad score could be outvoted on.
STAGE5_MIN_COUNTERPARTY_SIMILARITY = 0.60

# --- Calibrated thresholds. Mirrors stage2_constrained.py's own
# versioned-threshold pattern; no calibration pass against
# data/calibration/ has been run for Stage 5 specifically yet, so "v1"
# is a documented starting default, not a tuned value (same caveat as
# Stage 2's own MATCH_THRESHOLDS). ---
MATCH_THRESHOLDS: dict[str, float] = {"v1": 0.78}
DEFAULT_THRESHOLD = 0.78
# Stricter than Stage 2's MIN_SCORE_MARGIN (0.08): Stage 5 is the last,
# most evidence-hungry tier before the LLM, and its evidence (fuzzy
# identifier + fuzzy counterparty) is inherently softer than Stage 2's
# amount/date-led score, so a close call is even less trustworthy here.
STAGE5_MIN_MARGIN = 0.10

# --- Date-window policy. Slightly wider than Stage 2's own windows
# (7 / 5 days) since Stage 5 is deliberately the last chance before
# falling through to Stage 6 -- but still a real, bounded policy window,
# not "any date at all" (§5 explicitly names date compatibility as one
# of the required conditions, not an optional one). ---
LEDGER_GATEWAY_DATE_WINDOW_DAYS = 10
GATEWAY_BANK_DATE_WINDOW_DAYS = 7
DATE_GRACE_DAYS = 2

# --- Amount-compatibility policy. Same two shapes as Stage 2: "net_to_net"
# pairs (refund<->refund, chargeback<->chargeback, gateway settlement<->
# bank credit) should be almost exactly equal; "gross_to_net" pairs
# (ledger PAYMENT <-> gateway GATEWAY_SETTLEMENT) differ by the MDR fee +
# GST (and occasionally a small adjustment) deducted between the two.
# Values match Stage 2's own constants -- this is the same underlying fee
#/GST policy, not a different one for Stage 5. ---
NET_TOLERANCE_PAISE = 200
NET_TOLERANCE_FRACTION = 0.005
MAX_FEE_FRACTION = 0.05  # generator's realistic ceiling is ~2.95%
MAX_POSITIVE_ADJUSTMENT_FRACTION = 0.02


@dataclass
class Stage5Result:
    match_groups: list[MatchGroup] = field(default_factory=list)
    match_group_members: list[MatchGroupMember] = field(default_factory=list)
    decision_events: list = field(default_factory=list)
    matched_record_ids: set[str] = field(default_factory=set)


@dataclass
class _ScoreBreakdown:
    identifier_score: float
    identifier_ok: bool
    amount_score: float
    amount_ok: bool
    date_score: float
    date_ok: bool
    counterparty_score: float
    counterparty_ok: bool
    composite: float


def _normalize_counterparty(name: str) -> str:
    """Case/whitespace/punctuation-insensitive form of a counterparty
    label, for comparison only -- mirrors stage2_constrained.py's own
    helper of the same shape (kept as its own copy rather than a shared
    import: counterparty and reference are semantically different
    fields that just happen to normalize the same way today, and should
    not be assumed to stay in lockstep -- see that module's own note)."""
    return normalize_reference(name)


def _identifier_similarity(seeker: NormalizedRecord, candidate: NormalizedRecord) -> float:
    a = normalize_reference(seeker.reference)
    b = normalize_reference(candidate.reference)
    if not a or not b:
        return 0.0
    return fuzz.ratio(a, b) / 100.0


def _counterparty_similarity(seeker: NormalizedRecord, candidate: NormalizedRecord) -> float:
    a = _normalize_counterparty(seeker.counterparty)
    b = _normalize_counterparty(candidate.counterparty)
    if not a or not b:
        return 0.0
    return fuzz.ratio(a, b) / 100.0


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


def _score(
    seeker: NormalizedRecord,
    candidate: NormalizedRecord,
    expects_deduction: bool,
    date_window_days: int,
) -> Optional[_ScoreBreakdown]:
    """Returns None the moment any one of §5's required conditions
    fails -- currency, identifier similarity, amount, date, or
    counterparty. There is no partial credit: each is a hard gate, and
    only a candidate that clears every one of them gets a composite
    score at all (used solely for ranking/margin purposes among
    already-qualified candidates)."""
    if not currencies_match(seeker, candidate):
        return None

    identifier_score = _identifier_similarity(seeker, candidate)
    identifier_ok = identifier_score >= STAGE5_MIN_IDENTIFIER_SIMILARITY
    if not identifier_ok:
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
    counterparty_ok = counterparty_score >= STAGE5_MIN_COUNTERPARTY_SIMILARITY
    if not counterparty_ok:
        return None

    composite = (
        WEIGHT_IDENTIFIER * identifier_score
        + WEIGHT_AMOUNT * amount_score
        + WEIGHT_DATE * date_score
        + WEIGHT_COUNTERPARTY * counterparty_score
    )
    return _ScoreBreakdown(
        identifier_score=identifier_score,
        identifier_ok=identifier_ok,
        amount_score=amount_score,
        amount_ok=amount_ok,
        date_score=date_score,
        date_ok=date_ok,
        counterparty_score=counterparty_score,
        counterparty_ok=counterparty_ok,
        composite=composite,
    )


def _best_candidate(
    seeker: NormalizedRecord,
    candidates: list[NormalizedRecord],
    expects_deduction: bool,
    date_window_days: int,
) -> tuple[Optional[NormalizedRecord], Optional[_ScoreBreakdown], Optional[float]]:
    """Returns (best_candidate, best_breakdown, runner_up_score). Only
    candidates that already passed every §5 hard gate (i.e. ``_score``
    returned non-None) are scored at all -- a candidate that fails a
    hard gate is not "a weaker candidate", it is not a candidate."""
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


def run_stage5_fuzzy(
    records: list[NormalizedRecord],
    settings: Settings,
    prior_groups: list[MatchGroup],
    prior_members: list[MatchGroupMember],
    matched_record_ids: set[str],
    ids: IdAllocator,
) -> Stage5Result:
    """Stage 5 -- fuzzy candidate retrieval over records Stage 1-4 (plus
    conflict resolution, already folded into ``matched_record_ids``)
    left unmatched. ``prior_groups`` / ``prior_members`` are carried
    through unchanged (Stage 5 never extends or reopens an earlier
    stage's group -- see module docstring) so callers can treat this
    stage's output the same way as Stage 3/4's cumulative result.
    """
    threshold = _threshold_for(settings)

    matched_ids: set[str] = set(matched_record_ids)
    all_groups: dict[str, MatchGroup] = {g.group_id: g for g in prior_groups}
    all_members: dict[str, list[MatchGroupMember]] = {}
    for member in prior_members:
        all_members.setdefault(member.group_id, []).append(member)

    decision_events: list = []

    def _unmatched(source: Source) -> list[NormalizedRecord]:
        return sorted(
            (r for r in records if r.source == source and r.record_id not in matched_ids),
            key=lambda r: r.record_id,
        )

    # ---------------- Pass A: ledger <-> gateway ----------------
    for seeker in _unmatched(Source.LEDGER):
        target_entity = LEDGER_TO_GATEWAY_ENTITY.get(seeker.entity_type)
        if target_entity is None:
            continue

        candidates = [
            r
            for r in _unmatched(Source.GATEWAY)
            if r.entity_type == target_entity
        ]
        expects_deduction = target_entity == EntityType.GATEWAY_SETTLEMENT
        best, breakdown, runner_up = _best_candidate(
            seeker, candidates, expects_deduction, LEDGER_GATEWAY_DATE_WINDOW_DAYS
        )
        _propose_or_decline(
            seeker,
            best,
            breakdown,
            runner_up,
            candidates,
            threshold,
            settings,
            ids,
            all_groups,
            all_members,
            matched_ids,
            decision_events,
            reason_prefix="LEDGER_GATEWAY",
        )

    # ---------------- Pass B: gateway <-> bank ----------------
    for seeker in _unmatched(Source.GATEWAY):
        target_entity = GATEWAY_TO_BANK_ENTITY.get(seeker.entity_type)
        if target_entity is None:
            continue

        candidates = [
            r
            for r in _unmatched(Source.BANK)
            if r.entity_type == target_entity
        ]
        best, breakdown, runner_up = _best_candidate(
            seeker, candidates, expects_deduction=False, date_window_days=GATEWAY_BANK_DATE_WINDOW_DAYS
        )
        _propose_or_decline(
            seeker,
            best,
            breakdown,
            runner_up,
            candidates,
            threshold,
            settings,
            ids,
            all_groups,
            all_members,
            matched_ids,
            decision_events,
            reason_prefix="GATEWAY_BANK",
        )

    return Stage5Result(
        match_groups=list(all_groups.values()),
        match_group_members=[m for members in all_members.values() for m in members],
        decision_events=decision_events,
        matched_record_ids=matched_ids,
    )


def _propose_or_decline(
    seeker: NormalizedRecord,
    best: Optional[NormalizedRecord],
    breakdown: Optional[_ScoreBreakdown],
    runner_up: Optional[float],
    candidates: list[NormalizedRecord],
    threshold: float,
    settings: Settings,
    ids: IdAllocator,
    all_groups: dict[str, MatchGroup],
    all_members: dict[str, list[MatchGroupMember]],
    matched_ids: set[str],
    decision_events: list,
    reason_prefix: str,
) -> None:
    """Shared accept/decline bookkeeping for both Stage 5 passes.
    ``best``/``breakdown`` are None when no candidate cleared every §5
    hard gate at all -- distinguished in the decision event from a
    candidate that cleared the gates but not the threshold/margin bar,
    same three-way split Stage 2 uses."""
    candidate_scores = {
        "seeker_record_id": seeker.record_id,
        # Same-entity-type unmatched pool size considered, not the count
        # that cleared every §5 hard gate -- "best_candidate_record_id"
        # being None is what shows none of them did.
        "candidate_pool_size": len(candidates),
        "best_candidate_record_id": best.record_id if best else None,
        "identifier_score": breakdown.identifier_score if breakdown else None,
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
        and margin >= STAGE5_MIN_MARGIN
    )

    if not accepted:
        if best is None:
            reason = "NO_ELIGIBLE_CANDIDATE"
        elif breakdown.composite < threshold:
            reason = "BELOW_THRESHOLD"
        else:
            reason = "INSUFFICIENT_MARGIN"
        decision_events.append(
            make_decision_event(
                ids,
                f"unmatched:{seeker.record_id}",
                DecisionStage.STAGE5_FUZZY,
                candidate_scores=candidate_scores,
                reason_code=reason,
                explanation=(
                    f"No confident fuzzy-retrieval match for {seeker.record_id} "
                    f"({len(candidates)} eligible candidate(s) considered; "
                    f"{reason})."
                ),
            )
        )
        return

    assert best is not None and breakdown is not None and margin is not None

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
        evidence_score=breakdown.composite,
        runner_up_score=runner_up,
        score_margin=margin,
        threshold_applied=threshold,
        threshold_version=settings.threshold_version,
        policy_checks={
            "fuzzy_identifier_match": breakdown.identifier_ok,
            "identifier_score": breakdown.identifier_score,
            "currency_consistent": True,
            "amount_tolerance": True,
            "date_policy_window": True,
            "counterparty_evidence": breakdown.counterparty_ok,
            "counterparty_score": breakdown.counterparty_score,
            "unique_candidate": True,
            "margin_sufficient": True,
        },
        proposed_by=ProposedBy.STAGE5_FUZZY,
        verified_by=VerifiedBy.NOT_YET_VERIFIED,
        commit_policy=CommitPolicy.AUTO_COMMIT_STAGE2_5_THRESHOLD,
        verification_result=VerificationResult.NOT_YET_RUN,
    )
    all_groups[group_id] = group
    members = [make_member(group_id, seeker), make_member(group_id, best)]
    all_members[group_id] = members
    matched_ids.add(seeker.record_id)
    matched_ids.add(best.record_id)

    decision_events.append(
        make_decision_event(
            ids,
            group_id,
            DecisionStage.STAGE5_FUZZY,
            candidate_scores=candidate_scores,
            reason_code=f"{reason_prefix}_FUZZY_RETRIEVAL_MATCH",
            explanation=(
                f"{seeker.record_id} matched to {best.record_id}: fuzzy "
                f"identifier similarity {breakdown.identifier_score:.3f}, "
                f"counterparty similarity {breakdown.counterparty_score:.3f}, "
                f"composite score {breakdown.composite:.3f} >= threshold "
                f"{threshold:.2f}, margin {margin:.3f} over runner-up. All "
                "six §5 conditions independently satisfied."
            ),
        )
    )
