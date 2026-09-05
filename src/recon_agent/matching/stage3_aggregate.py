"""Stage 3 — aggregate/split match — ARCHITECTURE.md §6, §4, §9.

Finds consolidated settlements Stage 1/2 couldn't place because no single
record's amount lines up with its counterpart: several gateway
settlements summing to one consolidated bank credit (many-to-one payout
batching), or the reverse — one settlement split across several bank
credits (a partial-settlement payout). Stage 2 explicitly declines both
shapes (see its own docstring); this is where they get resolved, or
declined honestly as ``AMBIGUOUS_AGGREGATION`` when they can't be.

Per the corrected §4 scope note, there is no merchant/account field to
key on, so candidates are partitioned into disjoint windows using only
currency and settlement-date proximity, via
``aggregation_common.run_aggregation_search`` (shared with Stage 4).
Partitioning happens *before* any subset-sum search runs — that's what
makes a per-window, per-target search exact rather than a heuristic
guess over a possibly-overlapping shared pool. Two runs of that shared
search machinery cover both directions:

  Direction A (many settlements -> one bank credit): items are
  "settlement units" — either a whole existing Stage 1/2 PENDING_REVIEW
  group missing its bank leg (unit amount = that group's
  matched_amount_paise, the gateway-side net amount), or a standalone
  gateway settlement/refund/chargeback record that never joined any
  group at all. Targets are unmatched BANK_CREDIT records.

  Direction B (one settlement -> many bank credits): the reverse. Items
  are unmatched BANK_CREDIT records; targets are the settlement units
  left over after Direction A (a settlement can't be claimed by both
  directions).

A window whose candidate pool is too large to search exactly, or that
exceeds its search timeout, is abandoned wholesale rather than
partially resolved (§9) — every target in it is declined with
AMBIGUOUS_AGGREGATION, same as a genuine overlap between two targets'
winning subsets. Nothing here marks a group VERIFIED: every proposal is
status=PENDING_REVIEW, verified_by=NOT_YET_VERIFIED, commit_policy
AUTO_COMMIT_STAGE2_5_THRESHOLD (§2's shared Stage 3-5 evidence bar),
verification_result=NOT_YET_RUN.

**Global Conflict Resolution wiring (relay Stage 8, ARCHITECTURE.md §4's
"1:1 conflicts" row):** before declining an AMBIGUOUS_OVERLAP target
outright, its conflict cluster is first checked against
``conflict_resolver.resolve_conflict`` using each contesting target's
own real tentative-winning-candidate score (``TargetOutcome.tentative``/
``tentative_runner_up_score``, added to ``aggregation_common`` for
exactly this purpose — see that module). A cluster reduces cleanly only
when every contesting target's tentative winner is a single atomic
settlement unit / bank credit, never a multi-item subset-sum bundle
(``conflict_resolver``'s own scope, unchanged here). When it does
reduce, the Hungarian-optimal winners still have to clear the same
STAGE3_THRESHOLD/STAGE3_MIN_MARGIN evidence bar every other Stage 3
proposal does before being promoted to a real ``MatchGroup`` (cardinality
ONE_TO_ONE) that proceeds to verification exactly like any other
successful proposal; a target that wins the assignment but doesn't clear
that bar, or that has no real item left after the optimal assignment, or
whose cluster doesn't reduce at all, still declines as
AMBIGUOUS_AGGREGATION exactly as before — with a DecisionEvent that
records which of these paths was taken, for the audit trail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from recon_agent.config import Settings
from recon_agent.matching.aggregation_common import (
    AggregationItem,
    AggregationTarget,
    SubsetCandidate,
    TargetOutcome,
    WindowResult,
    run_aggregation_search,
)
from recon_agent.matching.common import GATEWAY_TO_BANK_ENTITY, IdAllocator, make_decision_event, make_member
from recon_agent.matching.conflict_resolver import resolve_conflict
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

# --- Search policy (§9's documented starting defaults; the actual bounds
# — max candidates per window, max group size, window timeout — come from
# Settings.aggregation_search, tuned independently of these). Amounts here
# are pre-netted settlement totals, so an exact (or near-exact) sum is
# expected; the threshold is set high because a *wrong* combination summing
# to within a wide tolerance by chance is very unlikely with real amounts,
# but a genuinely-correct aggregation should score at or near 1.0. ---
STAGE3_THRESHOLD = 0.85
STAGE3_MIN_MARGIN = 0.05
STAGE3_TOLERANCE_PAISE = 100
STAGE3_TOLERANCE_FRACTION = 0.001

# Bank credits land 1-2 days after the *latest* settlement date in their
# batch, per the generator; earlier batch members can be several days
# before that. This bounds how far apart a settlement unit and a bank
# credit can be and still be considered eligible for the same window.
MAX_SETTLEMENT_BANK_LAG_DAYS = 10


def _tolerance_fn(amount: int) -> int:
    return max(STAGE3_TOLERANCE_PAISE, int(STAGE3_TOLERANCE_FRACTION * amount))


def _resolve_direction(
    items: list[AggregationItem],
    targets: list[AggregationTarget],
    eligible,
    settings: Settings,
) -> WindowResult:
    limits = settings.aggregation_search
    return run_aggregation_search(
        items,
        targets,
        eligible=eligible,
        max_candidates_per_window=limits.max_candidates_per_window,
        max_group_size=limits.max_group_size,
        window_timeout_seconds=limits.window_timeout_seconds,
        tolerance_fn=_tolerance_fn,
        threshold=STAGE3_THRESHOLD,
        min_margin=STAGE3_MIN_MARGIN,
    )


@dataclass
class Stage3Result:
    match_groups: list[MatchGroup] = field(default_factory=list)
    match_group_members: list[MatchGroupMember] = field(default_factory=list)
    decision_events: list[DecisionEvent] = field(default_factory=list)
    exceptions: list[ReconciliationException] = field(default_factory=list)
    matched_record_ids: set[str] = field(default_factory=set)


@dataclass
class _SettlementUnit:
    unit_id: str
    record_ids: tuple[str, ...]
    amount_paise: int
    currency: str
    window_date: date
    existing_group_id: Optional[str]


def _ambiguous_identifier_duplicate_ids(records_by_id: dict[str, NormalizedRecord]) -> set[str]:
    """Record ids to exclude from Stage 3's raw-record fallback because
    they are the non-canonical half of a same-(source, entity_type)-slot
    reference collision — exactly Stage 1's own "identifier is unique"
    condition (stage1_exact.py's ``_slot_unique``), re-derived
    defensively here rather than duplicating Stage 1's clustering logic.

    Two records colliding on this slot are precisely the generator's
    injected duplicate decoys (identical reference, amount, currency,
    and counterparty — see testdata/generator.py's "duplicates"
    category): nothing in the data distinguishes which twin is the real
    settlement leg and which is the decoy, which is exactly why Stage 1
    declines to cluster either of them at all rather than guess.

    Naively excluding *both* twins from Stage 3's candidate pool would
    be wrong, though: one of them is frequently a genuinely necessary
    leg of a real consolidated settlement, and dropping it entirely
    reproduces the same orphaned-bank-credit failure this fix is for.
    Naively keeping *both* is also wrong: since they share the same
    amount, including both as independent items lets the subset-sum
    search build two candidate subsets that differ only by which twin
    they used, tying at the same score — which deterministically fails
    the margin check no matter how confident the real answer otherwise
    is, silently orphaning the bank credit it should have completed.

    The resolution: keep exactly one canonical representative per
    colliding slot (the lowest record_id, matching this codebase's
    existing "lowest record_id wins ties" convention — see
    aggregation_common.py's ``_candidates_for_target`` sort key) as a
    candidate item, and exclude the rest. The non-canonical twin(s)
    stay unmatched, which is the correct outcome for a genuine decoy —
    Stage 1/2/4 and the verifier are untouched by this.
    """
    slots: dict[tuple[str, Source, EntityType], list[str]] = {}
    for record in records_by_id.values():
        key = (normalize_reference(record.reference), record.source, record.entity_type)
        slots.setdefault(key, []).append(record.record_id)
    excluded: set[str] = set()
    for record_ids in slots.values():
        if len(record_ids) > 1:
            excluded.update(sorted(record_ids)[1:])
    return excluded


def _build_settlement_units(
    records_by_id: dict[str, NormalizedRecord],
    all_groups: dict[str, MatchGroup],
    all_members: dict[str, list[MatchGroupMember]],
    matched_ids: set[str],
) -> list[_SettlementUnit]:
    units: list[_SettlementUnit] = []
    ambiguous_ids = _ambiguous_identifier_duplicate_ids(records_by_id)

    for group_id, group in all_groups.items():
        members = all_members.get(group_id, [])
        if not members:
            continue
        if any(m.source == Source.BANK for m in members):
            continue  # already has its bank leg
        member_records = [records_by_id[m.record_id] for m in members if m.record_id in records_by_id]
        if not member_records:
            continue
        # REVERSAL groups never get a bank leg by design (a reversed
        # payment never reaches the bank) — only groups containing a
        # GATEWAY-side record type that actually maps to a bank
        # counterpart (GATEWAY_TO_BANK_ENTITY) are real settlement units.
        gateway_bank_eligible = next(
            (
                r
                for r in member_records
                if r.source == Source.GATEWAY and r.entity_type in GATEWAY_TO_BANK_ENTITY
            ),
            None,
        )
        if gateway_bank_eligible is None:
            continue
        currencies = {r.currency for r in member_records}
        if len(currencies) != 1:
            continue  # Stage 1/2 wouldn't have produced this, but be defensive
        units.append(
            _SettlementUnit(
                unit_id=f"group:{group_id}",
                record_ids=tuple(r.record_id for r in member_records),
                amount_paise=abs(group.matched_amount_paise),
                currency=next(iter(currencies)),
                window_date=max(r.occurred_at for r in member_records),
                existing_group_id=group_id,
            )
        )

    grouped_record_ids = {rid for u in units for rid in u.record_ids}
    for record in records_by_id.values():
        if record.record_id in matched_ids or record.record_id in grouped_record_ids:
            continue
        if record.source != Source.GATEWAY or record.entity_type not in GATEWAY_TO_BANK_ENTITY:
            continue
        if record.record_id in ambiguous_ids:
            continue  # known-ambiguous identifier (duplicate decoy) — never a candidate
        units.append(
            _SettlementUnit(
                unit_id=f"rec:{record.record_id}",
                record_ids=(record.record_id,),
                amount_paise=abs(record.amount_paise),
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


def _settlement_eligible_for_bank(item_date: date, target_date: date) -> bool:
    lag = (target_date - item_date).days
    return 0 <= lag <= MAX_SETTLEMENT_BANK_LAG_DAYS


def _bank_eligible_for_settlement(item_date: date, target_date: date) -> bool:
    lag = (item_date - target_date).days
    return 0 <= lag <= MAX_SETTLEMENT_BANK_LAG_DAYS


def _make_exception_for(
    outcome: TargetOutcome,
    involved_item_ids: tuple[str, ...],
    reason_code: str,
    extra_evidence: Optional[dict] = None,
) -> ReconciliationException:
    evidence = {
        "target_id": outcome.target.target_id,
        "target_record_ids": list(outcome.target.record_ids),
        "target_amount_paise": outcome.target.amount_paise,
        "reason": reason_code,
        "involved_item_ids": list(involved_item_ids),
        "num_eligible_subsets": outcome.num_eligible_subsets,
    }
    if extra_evidence:
        evidence.update(extra_evidence)
    return ReconciliationException(
        group_id_or_record_id=outcome.target.target_id,
        category=ExceptionCategory.AMBIGUOUS_AGGREGATION,
        severity=ExceptionSeverity.MEDIUM,
        evidence=evidence,
        recommended_action=(
            "Manual review required: aggregation search could not uniquely "
            f"resolve this target ({reason_code}). Do not force a guess."
        ),
        review_status=ExceptionReviewStatus.OPEN,
    )


# ---------------------------------------------------------------------------
# Global Conflict Resolution wiring (relay Stage 8) — see module docstring.
#
# Every AMBIGUOUS_OVERLAP target in a window is first grouped, with every
# other AMBIGUOUS_OVERLAP target it transitively shares a contested item
# with, into one conflict cluster (a plain union-find over "target" and
# "item" nodes — the same connectivity ``run_aggregation_search`` already
# used to decide these targets conflict at all, just partitioned here into
# independent clusters instead of one flat set). Each cluster is then
# handed to ``conflict_resolver.resolve_conflict`` as real (target_id,
# tentative_winning_item_ids, tentative_winning_score) edges — one edge per
# target, using exactly the score this module already computed to decide
# there was a conflict in the first place, never a placeholder.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ConflictWiring:
    """What happened to one AMBIGUOUS_OVERLAP target after its conflict
    cluster was run through ``conflict_resolver``.

    - PROMOTED: this target won the globally-optimal 1:1 assignment AND
      its winning score/margin clears the same STAGE3_THRESHOLD/
      STAGE3_MIN_MARGIN bar every other Stage 3 proposal must clear —
      caller should emit a real MatchGroup.
    - BELOW_EVIDENCE_BAR: this target won the assignment but its own
      score or margin doesn't clear that bar — declined, not forced.
    - UNASSIGNED_IN_RESOLVED_CLUSTER: the cluster resolved, but this
      particular target structurally had no item left after the optimal
      assignment (e.g. more targets than distinct contested items) — a
      legitimate partial outcome per conflict_resolver's own contract,
      not a forced guess.
    - DECLINED_BUNDLE: the cluster did not reduce to a clean bipartite
      1:1 case (at least one contesting target's tentative winner is
      itself a multi-item subset-sum bundle) — conflict_resolver declined
      it outright, exactly matching its documented scope.
    """

    kind: str
    winning_item_id: Optional[str] = None
    best: Optional[SubsetCandidate] = None
    runner_up_score: Optional[float] = None
    cluster_target_ids: frozenset = frozenset()
    cluster_candidate_ids: frozenset = frozenset()
    decline_reason: Optional[str] = None


def _wire_conflicts(window: WindowResult) -> dict[str, _ConflictWiring]:
    """Returns a ``target_id -> _ConflictWiring`` map covering every
    AMBIGUOUS_OVERLAP target in ``window``. Targets not declined
    AMBIGUOUS_OVERLAP are absent from the returned map.
    """
    ambiguous_outcomes = [o for o in window.outcomes if o.declined_reason == "AMBIGUOUS_OVERLAP"]
    if not ambiguous_outcomes:
        return {}

    # Union-find over ("T", target_id) / ("I", item_id) nodes, so targets
    # that don't actually share an item end up in independent clusters
    # instead of one conflict spanning the whole window.
    parent: dict[tuple[str, str], tuple[str, str]] = {}

    def _find(x: tuple[str, str]) -> tuple[str, str]:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: tuple[str, str], b: tuple[str, str]) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[ra] = rb

    def _ensure(node: tuple[str, str]) -> None:
        parent.setdefault(node, node)

    outcome_by_target: dict[str, TargetOutcome] = {}
    for outcome in ambiguous_outcomes:
        target_id = outcome.target.target_id
        outcome_by_target[target_id] = outcome
        tnode = ("T", target_id)
        _ensure(tnode)
        # outcome.tentative is guaranteed set here: AMBIGUOUS_OVERLAP is
        # only ever assigned to a target that had a tentative candidate
        # (see aggregation_common.run_aggregation_search).
        for item_id in outcome.tentative.item_ids:
            inode = ("I", item_id)
            _ensure(inode)
            _union(tnode, inode)

    clusters: dict[tuple[str, str], list[str]] = {}
    for target_id in outcome_by_target:
        root = _find(("T", target_id))
        clusters.setdefault(root, []).append(target_id)

    result: dict[str, _ConflictWiring] = {}
    for target_ids_in_cluster in clusters.values():
        edges = [
            (
                target_id,
                outcome_by_target[target_id].tentative.item_ids,
                outcome_by_target[target_id].tentative.score,
            )
            for target_id in target_ids_in_cluster
        ]
        resolution = resolve_conflict(edges)

        if not resolution.resolved:
            for target_id in target_ids_in_cluster:
                result[target_id] = _ConflictWiring(
                    kind="DECLINED_BUNDLE",
                    cluster_target_ids=resolution.target_ids,
                    cluster_candidate_ids=resolution.candidate_ids,
                    decline_reason=resolution.decline_reason,
                )
            continue

        for target_id in target_ids_in_cluster:
            outcome = outcome_by_target[target_id]
            cand = outcome.tentative
            runner_up = outcome.tentative_runner_up_score or 0.0

            if target_id not in resolution.assignments:
                # Structurally had no real item left after the optimal
                # assignment (e.g. more contesting targets than distinct
                # contested items) — legitimate partial outcome, not a
                # forced guess (conflict_resolver's own contract).
                result[target_id] = _ConflictWiring(
                    kind="UNASSIGNED_IN_RESOLVED_CLUSTER",
                    cluster_target_ids=resolution.target_ids,
                    cluster_candidate_ids=resolution.candidate_ids,
                )
                continue

            margin = cand.score - runner_up
            if cand.score >= STAGE3_THRESHOLD and margin >= STAGE3_MIN_MARGIN:
                result[target_id] = _ConflictWiring(
                    kind="PROMOTED",
                    winning_item_id=resolution.assignments[target_id],
                    best=cand,
                    runner_up_score=runner_up,
                    cluster_target_ids=resolution.target_ids,
                    cluster_candidate_ids=resolution.candidate_ids,
                )
            else:
                result[target_id] = _ConflictWiring(
                    kind="BELOW_EVIDENCE_BAR",
                    best=cand,
                    runner_up_score=runner_up,
                    cluster_target_ids=resolution.target_ids,
                    cluster_candidate_ids=resolution.candidate_ids,
                    decline_reason=(
                        "BELOW_THRESHOLD" if cand.score < STAGE3_THRESHOLD else "INSUFFICIENT_MARGIN"
                    ),
                )

    return result


def _conflict_wiring_evidence(wiring: _ConflictWiring) -> dict:
    """Common evidence/candidate_scores fields for a declined
    AMBIGUOUS_OVERLAP target whose conflict cluster went through
    conflict_resolver (i.e. every kind except PROMOTED, which gets a real
    MatchGroup + its own decision event instead)."""
    evidence: dict = {
        "resolution_path": wiring.kind,
        "conflict_target_ids": sorted(wiring.cluster_target_ids),
        "conflict_candidate_ids": sorted(wiring.cluster_candidate_ids),
    }
    if wiring.decline_reason:
        evidence["conflict_resolver_decline_reason"] = wiring.decline_reason
    if wiring.best is not None:
        evidence["own_tentative_score"] = wiring.best.score
        evidence["own_tentative_runner_up_score"] = wiring.runner_up_score
    return evidence


def _conflict_wiring_explanation(wiring: _ConflictWiring) -> str:
    """Human-readable sentence distinguishing why this AMBIGUOUS_OVERLAP
    target still declined after being run through conflict_resolver."""
    if wiring.kind == "DECLINED_BUNDLE":
        return (
            f"conflict_resolver was consulted ({wiring.decline_reason}): at "
            "least one contesting target's own winning candidate is a "
            "multi-item combination, not the single-atomic-item case it "
            "supports."
        )
    if wiring.kind == "BELOW_EVIDENCE_BAR":
        return (
            "conflict_resolver found this target the globally optimal "
            f"winner of its conflict, but its own score/margin ({wiring.decline_reason}) "
            "does not clear the Stage 3-5 evidence bar; declined rather "
            "than force a weak match."
        )
    if wiring.kind == "UNASSIGNED_IN_RESOLVED_CLUSTER":
        return (
            "conflict_resolver resolved this conflict, but this target had "
            "no real candidate left after the globally optimal assignment."
        )
    return ""


def run_stage3_aggregate(
    records: list[NormalizedRecord],
    settings: Settings,
    prior_groups: list[MatchGroup],
    prior_members: list[MatchGroupMember],
    matched_record_ids: set[str],
    ids: IdAllocator,
) -> Stage3Result:
    records_by_id = {r.record_id: r for r in records}
    matched_ids: set[str] = set(matched_record_ids)

    all_groups: dict[str, MatchGroup] = {g.group_id: g for g in prior_groups}
    all_members: dict[str, list[MatchGroupMember]] = {}
    for member in prior_members:
        all_members.setdefault(member.group_id, []).append(member)

    decision_events: list[DecisionEvent] = []
    exceptions: list[ReconciliationException] = []

    settlement_units = {
        u.unit_id: u
        for u in _build_settlement_units(records_by_id, all_groups, all_members, matched_ids)
    }
    bank_credits = {r.record_id: r for r in _unmatched_bank_credits(records_by_id, matched_ids)}

    consumed_unit_ids: set[str] = set()
    consumed_bank_ids: set[str] = set()

    def _settlement_to_item(u: "_SettlementUnit") -> AggregationItem:
        return AggregationItem(
            item_id=u.unit_id,
            record_ids=u.record_ids,
            amount_paise=u.amount_paise,
            currency=u.currency,
            window_date=u.window_date,
        )

    def _bank_to_item(r: NormalizedRecord) -> AggregationItem:
        return AggregationItem(
            item_id=f"rec:{r.record_id}",
            record_ids=(r.record_id,),
            amount_paise=abs(r.amount_paise),
            currency=r.currency,
            window_date=r.occurred_at,
        )

    def _bank_to_target(r: NormalizedRecord) -> AggregationTarget:
        return AggregationTarget(
            target_id=f"rec:{r.record_id}",
            record_ids=(r.record_id,),
            amount_paise=abs(r.amount_paise),
            currency=r.currency,
            window_date=r.occurred_at,
        )

    def _settlement_to_target(u: "_SettlementUnit") -> AggregationTarget:
        return AggregationTarget(
            target_id=u.unit_id,
            record_ids=u.record_ids,
            amount_paise=u.amount_paise,
            currency=u.currency,
            window_date=u.window_date,
        )

    def _emit_group_from_units(
        contributing_units: list["_SettlementUnit"],
        bank_records: list[NormalizedRecord],
        best: SubsetCandidate,
        runner_up_score: Optional[float],
        reason_code: str,
        cardinality: Cardinality,
        extra_explanation: str = "",
        extra_candidate_scores: Optional[dict] = None,
    ) -> None:
        group_id = ids.next_group_id()
        members: list[MatchGroupMember] = []
        for u in contributing_units:
            if u.existing_group_id is not None:
                members.extend(all_members.get(u.existing_group_id, []))
                all_groups.pop(u.existing_group_id, None)
                all_members.pop(u.existing_group_id, None)
            else:
                for rid in u.record_ids:
                    members.append(make_member(group_id, records_by_id[rid]))
        for bank_record in bank_records:
            members.append(make_member(group_id, bank_record))
        # Re-point every member row at the new merged group id.
        members = [m.model_copy(update={"group_id": group_id}) for m in members]

        expected = sum(u.amount_paise for u in contributing_units)
        matched = sum(abs(b.amount_paise) for b in bank_records)
        margin = best.score - (runner_up_score or 0.0)

        group = MatchGroup(
            group_id=group_id,
            cardinality=cardinality,
            expected_amount_paise=expected,
            matched_amount_paise=matched,
            residual_amount_paise=expected - matched,
            status=MatchGroupStatus.PENDING_REVIEW,
            evidence_score=best.score,
            runner_up_score=runner_up_score,
            score_margin=margin,
            threshold_applied=STAGE3_THRESHOLD,
            threshold_version=settings.threshold_version,
            policy_checks={
                "currency_consistent": True,
                "disjoint_window": True,
                "subset_sum_within_tolerance": True,
                "unique_candidate": True,
                "margin_sufficient": True,
            },
            proposed_by=ProposedBy.STAGE3_AGGREGATE,
            verified_by=VerifiedBy.NOT_YET_VERIFIED,
            commit_policy=CommitPolicy.AUTO_COMMIT_STAGE2_5_THRESHOLD,
            verification_result=VerificationResult.NOT_YET_RUN,
        )
        all_groups[group_id] = group
        all_members[group_id] = members

        for m in members:
            matched_ids.add(m.record_id)

        candidate_scores = {
            "contributing_unit_ids": [u.unit_id for u in contributing_units],
            "bank_record_ids": [b.record_id for b in bank_records],
            "score": best.score,
            "runner_up_score": runner_up_score,
            "threshold": STAGE3_THRESHOLD,
        }
        if extra_candidate_scores:
            candidate_scores.update(extra_candidate_scores)

        explanation = (
            f"{len(contributing_units)} settlement unit(s) matched against "
            f"{len(bank_records)} bank credit(s): score "
            f"{best.score:.3f} >= threshold {STAGE3_THRESHOLD:.2f}, margin "
            f"{margin:.3f}."
        )
        if extra_explanation:
            explanation = f"{explanation} {extra_explanation}"

        decision_events.append(
            make_decision_event(
                ids,
                group_id,
                DecisionStage.STAGE3_AGGREGATE,
                candidate_scores=candidate_scores,
                reason_code=reason_code,
                explanation=explanation,
            )
        )

    # ---------------- Direction A: many settlements -> one bank credit ----------------
    unit_by_item_id = {f"item:{u.unit_id}": u for u in settlement_units.values()}
    items_a = [
        AggregationItem(
            item_id=f"item:{u.unit_id}",
            record_ids=u.record_ids,
            amount_paise=u.amount_paise,
            currency=u.currency,
            window_date=u.window_date,
        )
        for u in settlement_units.values()
    ]
    targets_a = [_bank_to_target(r) for r in bank_credits.values()]

    def _eligible_a(item: AggregationItem, target: AggregationTarget) -> bool:
        return item.currency == target.currency and _settlement_eligible_for_bank(
            item.window_date, target.window_date
        )

    window_a = _resolve_direction(items_a, targets_a, _eligible_a, settings)
    conflict_wiring_a = _wire_conflicts(window_a)
    for outcome in window_a.outcomes:
        target = outcome.target
        bank_record = bank_credits[target.record_ids[0]]

        if outcome.declined_reason == "AMBIGUOUS_OVERLAP":
            wiring = conflict_wiring_a[target.target_id]

            if wiring.kind == "PROMOTED":
                unit = unit_by_item_id[wiring.winning_item_id]
                _emit_group_from_units(
                    [unit],
                    [bank_record],
                    wiring.best,
                    wiring.runner_up_score,
                    "AGGREGATE_CONFLICT_RESOLVED_HUNGARIAN",
                    Cardinality.ONE_TO_ONE,
                    extra_explanation=(
                        f"Globally optimal 1:1 assignment from conflict_resolver "
                        f"among {len(wiring.cluster_target_ids)} bank credit(s) "
                        f"contesting {len(wiring.cluster_candidate_ids)} "
                        "overlapping settlement candidate(s)."
                    ),
                    extra_candidate_scores={
                        "resolution_path": "HUNGARIAN_CONFLICT_RESOLVED",
                        "conflict_target_ids": sorted(wiring.cluster_target_ids),
                        "conflict_candidate_ids": sorted(wiring.cluster_candidate_ids),
                    },
                )
                consumed_bank_ids.add(bank_record.record_id)
                consumed_unit_ids.add(unit.unit_id)
                continue

            exceptions.append(
                _make_exception_for(
                    outcome,
                    tuple(window_a.ambiguous_item_ids),
                    "AMBIGUOUS_OVERLAP",
                    extra_evidence=_conflict_wiring_evidence(wiring),
                )
            )
            decision_events.append(
                make_decision_event(
                    ids,
                    target.target_id,
                    DecisionStage.STAGE3_AGGREGATE,
                    candidate_scores={
                        "target_id": target.target_id,
                        **_conflict_wiring_evidence(wiring),
                    },
                    reason_code="AMBIGUOUS_AGGREGATION",
                    explanation=(
                        f"Bank credit {bank_record.record_id}'s best-fit settlement "
                        "combination shares a candidate with another bank credit's "
                        f"best fit; declined rather than guessed. {_conflict_wiring_explanation(wiring)}"
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
                    DecisionStage.STAGE3_AGGREGATE,
                    candidate_scores={"target_id": target.target_id},
                    reason_code="AMBIGUOUS_AGGREGATION",
                    explanation=(
                        f"Bank credit {bank_record.record_id}'s aggregation window "
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
                    DecisionStage.STAGE3_AGGREGATE,
                    candidate_scores={"target_id": target.target_id, "num_eligible_subsets": outcome.num_eligible_subsets},
                    reason_code=outcome.declined_reason,
                    explanation=(
                        f"No confident settlement aggregation found for bank credit "
                        f"{bank_record.record_id} ({outcome.num_eligible_subsets} "
                        "eligible subset(s) considered)."
                    ),
                )
            )
            continue

        # accepted
        best = outcome.accepted
        contributing_units = [unit_by_item_id[iid] for iid in best.item_ids]
        cardinality = Cardinality.MANY_TO_ONE if len(contributing_units) > 1 else Cardinality.ONE_TO_ONE
        _emit_group_from_units(
            contributing_units,
            [bank_record],
            best,
            outcome.runner_up_score,
            "AGGREGATE_MANY_TO_ONE_MATCH",
            cardinality,
        )
        consumed_bank_ids.add(bank_record.record_id)
        for u in contributing_units:
            consumed_unit_ids.add(u.unit_id)

    # ---------------- Direction B: one settlement -> many bank credits ----------------
    remaining_units = [u for u in settlement_units.values() if u.unit_id not in consumed_unit_ids]
    remaining_bank = [r for rid, r in bank_credits.items() if rid not in consumed_bank_ids]

    bank_item_by_id = {f"bankitem:{r.record_id}": r for r in remaining_bank}
    items_b = [
        AggregationItem(
            item_id=f"bankitem:{r.record_id}",
            record_ids=(r.record_id,),
            amount_paise=abs(r.amount_paise),
            currency=r.currency,
            window_date=r.occurred_at,
        )
        for r in remaining_bank
    ]
    targets_b = [_settlement_to_target(u) for u in remaining_units]

    def _eligible_b(item: AggregationItem, target: AggregationTarget) -> bool:
        return item.currency == target.currency and _bank_eligible_for_settlement(
            item.window_date, target.window_date
        )

    unit_by_target_id = {u.unit_id: u for u in remaining_units}

    window_b = _resolve_direction(items_b, targets_b, _eligible_b, settings)
    conflict_wiring_b = _wire_conflicts(window_b)
    for outcome in window_b.outcomes:
        target = outcome.target
        unit = unit_by_target_id[target.target_id]

        if outcome.declined_reason == "AMBIGUOUS_OVERLAP":
            wiring = conflict_wiring_b[target.target_id]

            if wiring.kind == "PROMOTED":
                bank_record = bank_item_by_id[wiring.winning_item_id]
                _emit_group_from_units(
                    [unit],
                    [bank_record],
                    wiring.best,
                    wiring.runner_up_score,
                    "AGGREGATE_CONFLICT_RESOLVED_HUNGARIAN",
                    Cardinality.ONE_TO_ONE,
                    extra_explanation=(
                        f"Globally optimal 1:1 assignment from conflict_resolver "
                        f"among {len(wiring.cluster_target_ids)} settlement(s) "
                        f"contesting {len(wiring.cluster_candidate_ids)} "
                        "overlapping bank-credit candidate(s)."
                    ),
                    extra_candidate_scores={
                        "resolution_path": "HUNGARIAN_CONFLICT_RESOLVED",
                        "conflict_target_ids": sorted(wiring.cluster_target_ids),
                        "conflict_candidate_ids": sorted(wiring.cluster_candidate_ids),
                    },
                )
                consumed_bank_ids.add(bank_record.record_id)
                consumed_unit_ids.add(unit.unit_id)
                continue

            exceptions.append(
                _make_exception_for(
                    outcome,
                    tuple(window_b.ambiguous_item_ids),
                    "AMBIGUOUS_OVERLAP",
                    extra_evidence=_conflict_wiring_evidence(wiring),
                )
            )
            decision_events.append(
                make_decision_event(
                    ids,
                    target.target_id,
                    DecisionStage.STAGE3_AGGREGATE,
                    candidate_scores={
                        "target_id": target.target_id,
                        **_conflict_wiring_evidence(wiring),
                    },
                    reason_code="AMBIGUOUS_AGGREGATION",
                    explanation=(
                        f"Settlement {target.target_id}'s best-fit bank-credit "
                        "combination shares a candidate with another settlement's "
                        f"best fit; declined rather than guessed. {_conflict_wiring_explanation(wiring)}"
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
                    DecisionStage.STAGE3_AGGREGATE,
                    candidate_scores={"target_id": target.target_id},
                    reason_code="AMBIGUOUS_AGGREGATION",
                    explanation=(
                        f"Settlement {target.target_id}'s aggregation window could "
                        f"not be searched exactly ({outcome.declined_reason}); "
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
                    DecisionStage.STAGE3_AGGREGATE,
                    candidate_scores={"target_id": target.target_id, "num_eligible_subsets": outcome.num_eligible_subsets},
                    reason_code=outcome.declined_reason,
                    explanation=(
                        f"No confident bank-credit aggregation found for settlement "
                        f"{target.target_id} ({outcome.num_eligible_subsets} eligible "
                        "subset(s) considered)."
                    ),
                )
            )
            continue

        best = outcome.accepted
        bank_records = [bank_item_by_id[iid] for iid in best.item_ids]
        _emit_group_from_units(
            [unit],
            bank_records,
            best,
            outcome.runner_up_score,
            "AGGREGATE_ONE_TO_MANY_MATCH",
            Cardinality.ONE_TO_MANY,
        )
        for b in bank_records:
            consumed_bank_ids.add(b.record_id)
        consumed_unit_ids.add(unit.unit_id)

    return Stage3Result(
        match_groups=list(all_groups.values()),
        match_group_members=[m for members in all_members.values() for m in members],
        decision_events=decision_events,
        exceptions=exceptions,
        matched_record_ids=matched_ids,
    )
