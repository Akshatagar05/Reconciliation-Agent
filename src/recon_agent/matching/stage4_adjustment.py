"""Stage 4 — adjustment-aware reconciliation — ARCHITECTURE.md §6, §4, §9.

Applies the real conservation equation from §4:

    bank credit = gross payments - fees - GST - refunds +/- adjustments

using ``MatchGroupMember.role`` (GROSS, CREDIT, FEE, TAX, REFUND,
CHARGEBACK, REVERSAL, ADJUSTMENT) to net a candidate group's members
against its bank-side counterpart, within tolerance — the full
role-based netting Stage 2 explicitly deferred ("full fee/GST/refund-
netting conservation is explicitly Stage 4's job").

Every ``NormalizedRecord.amount_paise`` is already stored signed per its
role by construction (see ``testdata.generator``: GROSS/CREDIT positive,
FEE/TAX/REFUND/CHARGEBACK/REVERSAL negative, ADJUSTMENT signed either
way), so the conservation equation reduces to plain signed addition:
summing every non-CREDIT member's signed amount should land within
tolerance of the CREDIT (bank) member's amount. ``conservation_diff``
below is the standalone, directly-unit-testable form of that arithmetic;
everything else in this module is about finding *which* records to net
together.

Like Stage 3, this reuses ``aggregation_common``'s disjoint-windowing +
bounded subset-sum search: items are still partitioned into windows by
currency and settlement-date proximity before any search runs, and a
window that can't be cleanly resolved (genuine overlap, an oversized
candidate pool, or a search timeout) is declined as
``AMBIGUOUS_AGGREGATION`` rather than guessed. Stage 4 differs from
Stage 3 only in *what* the items are: instead of one pre-netted
settlement total per event, each individual role-tagged record (plus any
Stage 1-3 leftover group's still-incomplete net total) is its own item,
so combinations that Stage 3's coarser per-settlement view couldn't
represent — a settlement's GROSS leg plus a FEE/TAX/REFUND/ADJUSTMENT
record that never got clustered with it — are within reach here.

Only records Stage 1-3 left unmatched are considered. Nothing here marks
a group VERIFIED: every proposal is status=PENDING_REVIEW,
verified_by=NOT_YET_VERIFIED, commit_policy AUTO_COMMIT_STAGE2_5_THRESHOLD
(§2's shared Stage 3-5 evidence bar), verification_result=NOT_YET_RUN.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Iterable, Optional

from recon_agent.config import Settings
from recon_agent.matching.aggregation_common import (
    AggregationItem,
    AggregationTarget,
    TargetOutcome,
    WindowResult,
    run_aggregation_search,
)
from recon_agent.matching.common import ENTITY_TO_ROLE, IdAllocator, make_decision_event, make_member
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
    MemberRole,
    NormalizedRecord,
    ProposedBy,
    ReconciliationException,
    Source,
    VerificationResult,
    VerifiedBy,
)

STAGE4_THRESHOLD = 0.85
STAGE4_MIN_MARGIN = 0.05
STAGE4_TOLERANCE_PAISE = 100
STAGE4_TOLERANCE_FRACTION = 0.001

# Same lag rationale as Stage 3: a bank credit lands 1-2 days after the
# latest contributing record's date; earlier contributors can be several
# days before that.
MAX_RECORD_BANK_LAG_DAYS = 10

# Roles that legitimately land on the bank side of the conservation
# equation as a positive "gross payments" leg or a netting deduction.
# REVERSAL is excluded even though it has a role mapping: a reversed
# payment never reaches the bank at all (§4/common.py's own note), so a
# standalone REVERSAL record is never a candidate for bank-side netting.
_NETTABLE_ENTITY_TYPES = frozenset(
    {
        EntityType.GATEWAY_SETTLEMENT,
        EntityType.FEE,
        EntityType.TAX,
        EntityType.REFUND,
        EntityType.CHARGEBACK,
        EntityType.ADJUSTMENT,
    }
)


def conservation_diff(signed_amounts_paise: Iterable[int], credit_amount_paise: int) -> int:
    """The standalone conservation-equation arithmetic from §4:
    ``bank credit = gross - fees - GST - refunds +/- adjustments``,
    rearranged as ``sum(non-credit signed amounts) - credit == 0``.

    Every non-CREDIT ``NormalizedRecord.amount_paise`` is already signed
    per its role (GROSS/ADJUSTMENT-positive, FEE/TAX/REFUND/CHARGEBACK
    negative, per the generator), so this is plain addition — the
    interesting logic is entirely in *which* records get passed in.
    Returns the signed difference (computed - actual); zero means exact
    conservation.
    """
    return sum(signed_amounts_paise) - credit_amount_paise


def _tolerance_fn(amount: int) -> int:
    return max(STAGE4_TOLERANCE_PAISE, int(STAGE4_TOLERANCE_FRACTION * amount))



@dataclass
class Stage4Result:
    match_groups: list[MatchGroup] = field(default_factory=list)
    match_group_members: list[MatchGroupMember] = field(default_factory=list)
    decision_events: list[DecisionEvent] = field(default_factory=list)
    exceptions: list[ReconciliationException] = field(default_factory=list)
    matched_record_ids: set[str] = field(default_factory=set)


@dataclass
class _NetUnit:
    """One "many" side element for the role-based netting search: either
    a single unmatched, role-nettable record, or an entire Stage 1-3
    PENDING_REVIEW group that's still missing its bank leg (contributing
    its net signed total, so records already clustered together aren't
    double-counted at the individual-record level).
    """

    unit_id: str
    record_ids: tuple[str, ...]
    signed_amount_paise: int
    currency: str
    window_date: date
    existing_group_id: Optional[str]


def _build_net_units(
    records_by_id: dict[str, NormalizedRecord],
    all_groups: dict[str, MatchGroup],
    all_members: dict[str, list[MatchGroupMember]],
    matched_ids: set[str],
) -> list[_NetUnit]:
    units: list[_NetUnit] = []

    for group_id, group in all_groups.items():
        members = all_members.get(group_id, [])
        if not members:
            continue
        if any(m.source == Source.BANK for m in members):
            continue  # already has its bank leg
        member_records = [records_by_id[m.record_id] for m in members if m.record_id in records_by_id]
        if not member_records:
            continue
        gateway_nettable = next(
            (
                r
                for r in member_records
                if r.source == Source.GATEWAY and r.entity_type in _NETTABLE_ENTITY_TYPES
            ),
            None,
        )
        if gateway_nettable is None:
            continue
        currencies = {r.currency for r in member_records}
        if len(currencies) != 1:
            continue
        # Net signed total of the group's own members that would
        # eventually stand on the non-CREDIT side of the equation: use
        # the pre-existing matched_amount_paise (the gateway net amount,
        # already fee/GST-adjusted) as this unit's contribution.
        units.append(
            _NetUnit(
                unit_id=f"group:{group_id}",
                record_ids=tuple(r.record_id for r in member_records),
                signed_amount_paise=abs(group.matched_amount_paise),
                currency=next(iter(currencies)),
                window_date=max(r.occurred_at for r in member_records),
                existing_group_id=group_id,
            )
        )

    grouped_record_ids = {rid for u in units for rid in u.record_ids}
    for record in records_by_id.values():
        if record.record_id in matched_ids or record.record_id in grouped_record_ids:
            continue
        if record.source != Source.GATEWAY or record.entity_type not in _NETTABLE_ENTITY_TYPES:
            continue
        units.append(
            _NetUnit(
                unit_id=f"rec:{record.record_id}",
                record_ids=(record.record_id,),
                signed_amount_paise=record.amount_paise,
                currency=record.currency,
                window_date=record.occurred_at,
                existing_group_id=None,
            )
        )

    return units


def _unmatched_bank_credits(
    records_by_id: dict[str, NormalizedRecord], matched_ids: set[str]
) -> list[NormalizedRecord]:
    return sorted(
        (
            r
            for r in records_by_id.values()
            if r.source == Source.BANK and r.entity_type == EntityType.BANK_CREDIT and r.record_id not in matched_ids
        ),
        key=lambda r: r.record_id,
    )


def _eligible(item_date: date, target_date: date) -> bool:
    lag = (target_date - item_date).days
    return 0 <= lag <= MAX_RECORD_BANK_LAG_DAYS


def _make_exception_for(outcome: TargetOutcome, involved_item_ids: tuple[str, ...], reason_code: str) -> ReconciliationException:
    return ReconciliationException(
        group_id_or_record_id=outcome.target.target_id,
        category=ExceptionCategory.AMBIGUOUS_AGGREGATION,
        severity=ExceptionSeverity.MEDIUM,
        evidence={
            "target_id": outcome.target.target_id,
            "target_record_ids": list(outcome.target.record_ids),
            "target_amount_paise": outcome.target.amount_paise,
            "reason": reason_code,
            "involved_item_ids": list(involved_item_ids),
            "num_eligible_subsets": outcome.num_eligible_subsets,
        },
        recommended_action=(
            "Manual review required: role-based netting search could not "
            f"uniquely resolve this bank credit ({reason_code}). Do not "
            "force a guess."
        ),
        review_status=ExceptionReviewStatus.OPEN,
    )


def run_stage4_adjustment(
    records: list[NormalizedRecord],
    settings: Settings,
    prior_groups: list[MatchGroup],
    prior_members: list[MatchGroupMember],
    matched_record_ids: set[str],
    ids: IdAllocator,
) -> Stage4Result:
    records_by_id = {r.record_id: r for r in records}
    matched_ids: set[str] = set(matched_record_ids)

    all_groups: dict[str, MatchGroup] = {g.group_id: g for g in prior_groups}
    all_members: dict[str, list[MatchGroupMember]] = {}
    for member in prior_members:
        all_members.setdefault(member.group_id, []).append(member)

    decision_events: list[DecisionEvent] = []
    exceptions: list[ReconciliationException] = []

    net_units = {
        u.unit_id: u for u in _build_net_units(records_by_id, all_groups, all_members, matched_ids)
    }
    bank_credits = {r.record_id: r for r in _unmatched_bank_credits(records_by_id, matched_ids)}

    items = [
        AggregationItem(
            item_id=f"item:{u.unit_id}",
            record_ids=u.record_ids,
            amount_paise=u.signed_amount_paise,
            currency=u.currency,
            window_date=u.window_date,
        )
        for u in net_units.values()
    ]
    unit_by_item_id = {f"item:{u.unit_id}": u for u in net_units.values()}

    targets = [
        AggregationTarget(
            target_id=f"rec:{r.record_id}",
            record_ids=(r.record_id,),
            amount_paise=abs(r.amount_paise),
            currency=r.currency,
            window_date=r.occurred_at,
        )
        for r in bank_credits.values()
    ]

    def _eligible_fn(item: AggregationItem, target: AggregationTarget) -> bool:
        return item.currency == target.currency and _eligible(item.window_date, target.window_date)

    limits = settings.aggregation_search
    window = run_aggregation_search(
        items,
        targets,
        eligible=_eligible_fn,
        max_candidates_per_window=limits.max_candidates_per_window,
        max_group_size=limits.max_group_size,
        window_timeout_seconds=limits.window_timeout_seconds,
        tolerance_fn=_tolerance_fn,
        threshold=STAGE4_THRESHOLD,
        min_margin=STAGE4_MIN_MARGIN,
    )

    for outcome in window.outcomes:
        target = outcome.target
        bank_record = bank_credits[target.record_ids[0]]

        if outcome.declined_reason == "AMBIGUOUS_OVERLAP":
            exceptions.append(_make_exception_for(outcome, tuple(window.ambiguous_item_ids), "AMBIGUOUS_OVERLAP"))
            decision_events.append(
                make_decision_event(
                    ids,
                    target.target_id,
                    DecisionStage.STAGE4_ADJUSTMENT,
                    candidate_scores={"target_id": target.target_id},
                    reason_code="AMBIGUOUS_AGGREGATION",
                    explanation=(
                        f"Bank credit {bank_record.record_id}'s best-fit netted "
                        "combination shares a candidate record with another bank "
                        "credit's best fit; declined rather than guessed."
                    ),
                )
            )
            continue
        if outcome.declined_reason in ("CANDIDATE_POOL_TOO_LARGE", "WINDOW_TIMEOUT"):
            exceptions.append(_make_exception_for(outcome, (), outcome.declined_reason))
            decision_events.append(
                make_decision_event(
                    ids,
                    target.target_id,
                    DecisionStage.STAGE4_ADJUSTMENT,
                    candidate_scores={"target_id": target.target_id},
                    reason_code="AMBIGUOUS_AGGREGATION",
                    explanation=(
                        f"Bank credit {bank_record.record_id}'s netting window "
                        f"could not be searched exactly ({outcome.declined_reason}); "
                        "declined rather than guessed."
                    ),
                )
            )
            continue
        if outcome.declined_reason in ("NO_AGGREGATION_FOUND", "BELOW_THRESHOLD", "INSUFFICIENT_MARGIN"):
            decision_events.append(
                make_decision_event(
                    ids,
                    target.target_id,
                    DecisionStage.STAGE4_ADJUSTMENT,
                    candidate_scores={"target_id": target.target_id, "num_eligible_subsets": outcome.num_eligible_subsets},
                    reason_code=outcome.declined_reason,
                    explanation=(
                        f"No confident role-based netting found for bank credit "
                        f"{bank_record.record_id} ({outcome.num_eligible_subsets} "
                        "eligible subset(s) considered)."
                    ),
                )
            )
            continue

        best = outcome.accepted
        contributing_units = [unit_by_item_id[iid] for iid in best.item_ids]

        group_id = ids.next_group_id()
        group_members: list[MatchGroupMember] = []
        for u in contributing_units:
            if u.existing_group_id is not None:
                group_members.extend(all_members.get(u.existing_group_id, []))
                all_groups.pop(u.existing_group_id, None)
                all_members.pop(u.existing_group_id, None)
            else:
                for rid in u.record_ids:
                    group_members.append(make_member(group_id, records_by_id[rid]))
        group_members.append(make_member(group_id, bank_record))
        group_members = [m.model_copy(update={"group_id": group_id}) for m in group_members]

        expected = sum(u.signed_amount_paise for u in contributing_units)
        matched = abs(bank_record.amount_paise)
        margin = best.score - (outcome.runner_up_score or 0.0)
        # Sanity-check the actual conservation arithmetic against the
        # signed member amounts, independent of the search's own score
        # bookkeeping — this is the "actually uses role-based netting"
        # contract, verified on the real member rows.
        diff = conservation_diff((m.signed_amount_paise for m in group_members if m.role != MemberRole.CREDIT), matched)

        cardinality = Cardinality.MANY_TO_ONE if len(contributing_units) > 1 else Cardinality.ONE_TO_ONE
        group = MatchGroup(
            group_id=group_id,
            cardinality=cardinality,
            expected_amount_paise=expected,
            matched_amount_paise=matched,
            residual_amount_paise=diff,
            status=MatchGroupStatus.PENDING_REVIEW,
            evidence_score=best.score,
            runner_up_score=outcome.runner_up_score,
            score_margin=margin,
            threshold_applied=STAGE4_THRESHOLD,
            threshold_version=settings.threshold_version,
            policy_checks={
                "currency_consistent": True,
                "disjoint_window": True,
                "conservation_within_tolerance": abs(diff) <= _tolerance_fn(matched),
                "unique_candidate": True,
                "margin_sufficient": True,
            },
            proposed_by=ProposedBy.STAGE4_ADJUSTMENT,
            verified_by=VerifiedBy.NOT_YET_VERIFIED,
            commit_policy=CommitPolicy.AUTO_COMMIT_STAGE2_5_THRESHOLD,
            verification_result=VerificationResult.NOT_YET_RUN,
        )
        all_groups[group_id] = group
        all_members[group_id] = group_members
        for m in group_members:
            matched_ids.add(m.record_id)

        decision_events.append(
            make_decision_event(
                ids,
                group_id,
                DecisionStage.STAGE4_ADJUSTMENT,
                candidate_scores={
                    "contributing_unit_ids": [u.unit_id for u in contributing_units],
                    "bank_record_id": bank_record.record_id,
                    "score": best.score,
                    "runner_up_score": outcome.runner_up_score,
                    "conservation_diff_paise": diff,
                    "threshold": STAGE4_THRESHOLD,
                },
                reason_code="ADJUSTMENT_AWARE_NET_MATCH",
                explanation=(
                    f"{len(contributing_units)} role-tagged record(s)/group(s) net "
                    f"to within tolerance of bank credit {bank_record.record_id}: "
                    f"score {best.score:.3f} >= threshold {STAGE4_THRESHOLD:.2f}, "
                    f"margin {margin:.3f}, conservation diff {diff} paise."
                ),
            )
        )

    return Stage4Result(
        match_groups=list(all_groups.values()),
        match_group_members=[m for members in all_members.values() for m in members],
        decision_events=decision_events,
        exceptions=exceptions,
        matched_record_ids=matched_ids,
    )
