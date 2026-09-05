"""Unit tests for src/recon_agent/matching/stage3_aggregate.py."""

from __future__ import annotations

from datetime import date

from recon_agent.config import Settings
from recon_agent.matching.common import IdAllocator, make_member
from recon_agent.matching.stage3_aggregate import run_stage3_aggregate
from recon_agent.models import (
    Cardinality,
    CommitPolicy,
    EntityType,
    MatchGroup,
    MatchGroupStatus,
    NormalizedRecord,
    ProposedBy,
    Source,
    VerificationResult,
    VerifiedBy,
)


def _record(
    record_id: str,
    source: Source,
    entity_type: EntityType,
    amount_paise: int,
    reference: str,
    occurred_at: date,
    currency: str = "INR",
    counterparty: str = "Acme Retail Pvt Ltd",
) -> NormalizedRecord:
    return NormalizedRecord(
        record_id=record_id,
        source=source,
        entity_type=entity_type,
        amount_paise=amount_paise,
        currency=currency,
        reference=reference,
        counterparty=counterparty,
        occurred_at=occurred_at,
        raw_hash=f"hash_{record_id}",
    )


def _settings() -> Settings:
    return Settings()


def _stub_group(group_id: str = "match_grp_00001", matched_amount_paise: int = 9_700) -> MatchGroup:
    return MatchGroup(
        group_id=group_id,
        cardinality=Cardinality.ONE_TO_ONE,
        expected_amount_paise=matched_amount_paise,
        matched_amount_paise=matched_amount_paise,
        residual_amount_paise=0,
        status=MatchGroupStatus.PENDING_REVIEW,
        threshold_applied=1.0,
        threshold_version="v1",
        policy_checks={},
        proposed_by=ProposedBy.STAGE1_EXACT,
        verified_by=VerifiedBy.NOT_YET_VERIFIED,
        commit_policy=CommitPolicy.AUTO_COMMIT_STAGE1,
        verification_result=VerificationResult.NOT_YET_RUN,
    )


def test_many_to_one_settlements_matched_to_one_bank_credit() -> None:
    records = [
        _record("s1", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 10_000, "ref1", date(2026, 1, 1)),
        _record("s2", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 15_000, "ref2", date(2026, 1, 1)),
        _record("s3", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 5_000, "ref3", date(2026, 1, 1)),
        _record("bx", Source.BANK, EntityType.BANK_CREDIT, 30_000, "UTR1", date(2026, 1, 2)),
    ]
    result = run_stage3_aggregate(records, _settings(), [], [], set(), IdAllocator())

    assert len(result.match_groups) == 1
    group = result.match_groups[0]
    assert group.cardinality == Cardinality.MANY_TO_ONE
    assert group.matched_amount_paise == 30_000
    assert group.status == MatchGroupStatus.PENDING_REVIEW
    assert group.verified_by == VerifiedBy.NOT_YET_VERIFIED
    assert group.proposed_by == ProposedBy.STAGE3_AGGREGATE
    assert result.matched_record_ids == {"s1", "s2", "s3", "bx"}
    assert not result.exceptions


def test_one_to_many_settlement_split_across_bank_credits() -> None:
    records = [
        _record("s1", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 30_000, "ref1", date(2026, 1, 1)),
        _record("b1", Source.BANK, EntityType.BANK_CREDIT, 18_000, "UTR1", date(2026, 1, 2)),
        _record("b2", Source.BANK, EntityType.BANK_CREDIT, 12_000, "UTR2", date(2026, 1, 3)),
    ]
    result = run_stage3_aggregate(records, _settings(), [], [], set(), IdAllocator())

    assert len(result.match_groups) == 1
    group = result.match_groups[0]
    assert group.cardinality == Cardinality.ONE_TO_MANY
    assert group.matched_amount_paise == 30_000
    assert result.matched_record_ids == {"s1", "b1", "b2"}
    assert not result.exceptions


def test_disjoint_windowing_declines_genuine_overlap_as_ambiguous_aggregation() -> None:
    """Construct a case where two bank credits could each plausibly claim
    the same settlement candidate: BankX (250) is uniquely completed by
    {item_a=100, item_b=150}; BankY (200) is uniquely completed by
    {item_b=150, item_c=50}. item_b is shared between both targets'
    otherwise-unique best subsets. The search must decline BOTH targets
    with AMBIGUOUS_AGGREGATION rather than silently picking a winner.
    """
    records = [
        _record("item_a", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 10_000, "refa", date(2026, 1, 1)),
        _record("item_b", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 15_000, "refb", date(2026, 1, 1)),
        _record("item_c", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 5_000, "refc", date(2026, 1, 1)),
        _record("bank_x", Source.BANK, EntityType.BANK_CREDIT, 25_000, "UTR_X", date(2026, 1, 2)),
        _record("bank_y", Source.BANK, EntityType.BANK_CREDIT, 20_000, "UTR_Y", date(2026, 1, 2)),
    ]
    result = run_stage3_aggregate(records, _settings(), [], [], set(), IdAllocator())

    # Nothing should be proposed — the ambiguity must be declined, not
    # resolved by an arbitrary tie-break between two independently
    # plausible targets.
    assert result.match_groups == []
    assert result.matched_record_ids == set()

    assert len(result.exceptions) == 2
    declined_ids = {e.group_id_or_record_id for e in result.exceptions}
    assert declined_ids == {"rec:bank_x", "rec:bank_y"}
    for exc in result.exceptions:
        assert exc.category.value == "AMBIGUOUS_AGGREGATION"
        assert exc.evidence["reason"] == "AMBIGUOUS_OVERLAP"

    reason_codes = {e.reason_code for e in result.decision_events if e.stage.value == "STAGE3_AGGREGATE"}
    assert "AMBIGUOUS_AGGREGATION" in reason_codes


def test_reversal_groups_are_never_treated_as_settlement_units() -> None:
    """A REVERSAL group (ledger+gateway only, no bank leg by design) must
    never be pulled into the aggregation pool — it isn't waiting on a
    bank leg at all.
    """
    ledger_rev = _record("lr1", Source.LEDGER, EntityType.REVERSAL, -10_000, "revref", date(2026, 1, 1))
    gateway_rev = _record("gr1", Source.GATEWAY, EntityType.REVERSAL, -9_700, "revref", date(2026, 1, 1))
    bank = _record("bx", Source.BANK, EntityType.BANK_CREDIT, 9_700, "UTR1", date(2026, 1, 2))

    group = _stub_group()
    members = [make_member("match_grp_00001", ledger_rev), make_member("match_grp_00001", gateway_rev)]

    result = run_stage3_aggregate(
        [ledger_rev, gateway_rev, bank], _settings(), [group], members, {"lr1", "gr1"}, IdAllocator()
    )

    # The reversal group must be left untouched, and the lone bank
    # credit must not be force-matched to it.
    assert result.match_groups == [group]
    assert "bx" not in result.matched_record_ids


def test_extends_existing_stage1_group_with_missing_bank_leg() -> None:
    ledger = _record("l1", Source.LEDGER, EntityType.PAYMENT, 10_000, "ref1", date(2026, 1, 1))
    gateway = _record("g1", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 9_700, "ref1", date(2026, 1, 1))
    bank = _record("bx", Source.BANK, EntityType.BANK_CREDIT, 9_700, "UTR1", date(2026, 1, 2))

    group = _stub_group()
    members = [make_member("match_grp_00001", ledger), make_member("match_grp_00001", gateway)]
    matched = {"l1", "g1"}

    ids = IdAllocator()
    ids.next_group_id()  # simulate one id already consumed upstream

    result = run_stage3_aggregate([ledger, gateway, bank], _settings(), [group], members, matched, ids)

    assert len(result.match_groups) == 1
    merged = result.match_groups[0]
    assert merged.group_id != "match_grp_00001"
    assert merged.proposed_by == ProposedBy.STAGE3_AGGREGATE
    member_ids = {m.record_id for m in result.match_group_members if m.group_id == merged.group_id}
    assert member_ids == {"l1", "g1", "bx"}
    assert result.matched_record_ids == {"l1", "g1", "bx"}


def test_extends_multiple_existing_bank_less_groups_with_one_bank_credit() -> None:
    """Generalization of test_extends_existing_stage1_group_with_missing_bank_leg
    to the real N-group case this bugfix targets: a consolidated
    settlement batch where Stage 1 has already proposed SEVERAL separate
    bank-less ledger<->gateway pairs (each internally correct, none
    including the bank leg), and exactly one later bank credit
    consolidates all of them. Mirrors the calibration dataset's
    cal_grp_00026 shape (multiple pre-existing pairs + one bank credit)
    but without any surrounding dataset noise.

    Stage 3 must: (1) see all three pre-existing bank-less groups as
    aggregation candidates (not just raw unmatched records), (2) find
    that their net totals plus the raw bank credit sum within
    tolerance, (3) emit exactly ONE unified MatchGroup containing every
    member of all three sub-groups plus the bank credit, and (4) retract
    the three smaller sub-group proposals entirely — they must not
    appear anywhere in the final proposed-groups list or member list,
    only the new aggregate should.
    """
    pairs = [
        (_record("l1", Source.LEDGER, EntityType.PAYMENT, 10_000, "ref1", date(2026, 1, 1)),
         _record("g1", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 10_000, "ref1", date(2026, 1, 1))),
        (_record("l2", Source.LEDGER, EntityType.PAYMENT, 15_000, "ref2", date(2026, 1, 1)),
         _record("g2", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 15_000, "ref2", date(2026, 1, 1))),
        (_record("l3", Source.LEDGER, EntityType.PAYMENT, 5_000, "ref3", date(2026, 1, 1)),
         _record("g3", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 5_000, "ref3", date(2026, 1, 1))),
    ]
    bank = _record("bx", Source.BANK, EntityType.BANK_CREDIT, 30_000, "UTR1", date(2026, 1, 2))

    records = [r for pair in pairs for r in pair] + [bank]

    groups = [
        _stub_group(f"match_grp_0000{i + 1}", matched_amount_paise=amount)
        for i, amount in enumerate([10_000, 15_000, 5_000])
    ]
    members = []
    matched: set[str] = set()
    for group, (ledger, gateway) in zip(groups, pairs):
        members.append(make_member(group.group_id, ledger))
        members.append(make_member(group.group_id, gateway))
        matched.update({ledger.record_id, gateway.record_id})

    ids = IdAllocator()
    ids.next_group_id()
    ids.next_group_id()
    ids.next_group_id()  # simulate three ids already consumed upstream (Stage 1)

    result = run_stage3_aggregate(records, _settings(), groups, members, matched, ids)

    assert len(result.match_groups) == 1
    merged = result.match_groups[0]
    assert merged.cardinality == Cardinality.MANY_TO_ONE
    assert merged.matched_amount_paise == 30_000
    assert merged.proposed_by == ProposedBy.STAGE3_AGGREGATE
    assert merged.group_id not in {g.group_id for g in groups}

    member_ids = {m.record_id for m in result.match_group_members if m.group_id == merged.group_id}
    assert member_ids == {"l1", "g1", "l2", "g2", "l3", "g3", "bx"}

    # The smaller sub-group proposals must be retracted entirely: none
    # of the original group_ids may appear anywhere in the final
    # member list, and matched_record_ids must be exactly the unified
    # group's members.
    surviving_group_ids = {m.group_id for m in result.match_group_members}
    assert surviving_group_ids == {merged.group_id}
    assert result.matched_record_ids == {"l1", "g1", "l2", "g2", "l3", "g3", "bx"}
    assert not result.exceptions

    reason_codes = {e.reason_code for e in result.decision_events if e.group_id == merged.group_id}
    assert "AGGREGATE_MANY_TO_ONE_MATCH" in reason_codes


def test_duplicate_decoy_record_does_not_block_consolidation() -> None:
    """Regression test for the actual bug found in the calibration
    dataset (cal_grp_00026): a consolidated settlement batch where one
    of the raw, still-ungrouped settlement legs has an injected
    "duplicate decoy" twin — a second GATEWAY_SETTLEMENT record sharing
    the exact same reference, amount, currency, and date, which Stage 1
    correctly declines to cluster at all (non-unique identifier) so
    BOTH twins remain permanently unmatched going into Stage 3.

    Before the fix, Stage 3's raw-record fallback treated both twins as
    independent standalone candidate items. Since they share the same
    amount, that let the subset-sum search build two equally-scoring
    (score=1.0) candidate subsets differing only by which twin they
    used — a guaranteed exact tie that deterministically fails the
    margin check no matter how confident the real answer is, silently
    orphaning the bank credit. The fix keeps exactly one canonical
    representative (lowest record_id) per colliding identifier slot.
    """
    pairs = [
        (_record("l1", Source.LEDGER, EntityType.PAYMENT, 10_000, "ref1", date(2026, 1, 1)),
         _record("g1", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 10_000, "ref1", date(2026, 1, 1))),
        (_record("l2", Source.LEDGER, EntityType.PAYMENT, 15_000, "ref2", date(2026, 1, 1)),
         _record("g2", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 15_000, "ref2", date(2026, 1, 1))),
    ]
    # The third leg was never clustered by Stage 1 at all: its gateway
    # settlement record has a duplicate decoy twin sharing the exact
    # same reference/amount/date, so the whole ref3 cluster (l3, g3,
    # g3_dup) was declined for a non-unique identifier and all three
    # remain raw and unmatched going into Stage 3 — exactly like
    # cal_rec_00161/00162/00163 in the calibration dataset.
    l3 = _record("l3", Source.LEDGER, EntityType.PAYMENT, 5_000, "ref3", date(2026, 1, 1))
    g3 = _record("g3", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 5_000, "ref3", date(2026, 1, 1))
    g3_dup = _record("g3_dup", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 5_000, "ref3", date(2026, 1, 1))
    bank = _record("bx", Source.BANK, EntityType.BANK_CREDIT, 30_000, "UTR1", date(2026, 1, 2))

    records = [r for pair in pairs for r in pair] + [l3, g3, g3_dup, bank]

    groups = [
        _stub_group(f"match_grp_0000{i + 1}", matched_amount_paise=amount)
        for i, amount in enumerate([10_000, 15_000])
    ]
    members = []
    matched: set[str] = set()
    for group, (ledger, gateway) in zip(groups, pairs):
        members.append(make_member(group.group_id, ledger))
        members.append(make_member(group.group_id, gateway))
        matched.update({ledger.record_id, gateway.record_id})
    # l3/g3/g3_dup are intentionally NOT added to matched/members: Stage 1
    # declined that whole cluster, so they reach Stage 3 as raw records.

    ids = IdAllocator()
    ids.next_group_id()
    ids.next_group_id()

    result = run_stage3_aggregate(records, _settings(), groups, members, matched, ids)

    assert len(result.match_groups) == 1
    merged = result.match_groups[0]
    assert merged.matched_amount_paise == 30_000

    member_ids = {m.record_id for m in result.match_group_members if m.group_id == merged.group_id}
    # The genuine gateway leg (g3, the lower record_id) is absorbed
    # along with the bank credit; the decoy (g3_dup) is correctly left
    # out. (l3, the ledger side of that never-clustered pair, is out of
    # scope for Stage 3's settlement-unit fallback — which only ever
    # considers the GATEWAY-side leg — so it stays unmatched either
    # way; that's an existing, unrelated scope boundary, not part of
    # this fix.)
    assert member_ids == {"l1", "g1", "l2", "g2", "g3", "bx"}
    assert "g3_dup" not in result.matched_record_ids
    assert "l3" not in result.matched_record_ids
    assert not result.exceptions


# ---------------------------------------------------------------------------
# Relay Stage 8: conflict_resolver.py wired into the AMBIGUOUS_OVERLAP path.
# ---------------------------------------------------------------------------


def test_single_item_overlap_conflict_resolved_via_hungarian() -> None:
    """Two bank credits both see the SAME lone settlement item as their
    only eligible candidate (a clean 1:1 conflict, not a multi-item
    bundle): bank_p (10_000) matches item_x (10_000) exactly; bank_q
    (10_095) also only fits item_x, but with a much weaker score and,
    on its own, below STAGE3_THRESHOLD. Old behavior declined BOTH as
    AMBIGUOUS_AGGREGATION. New behavior: conflict_resolver finds bank_p
    the unambiguous globally-optimal winner (real score/margin clear the
    evidence bar) and promotes it to a real ONE_TO_ONE match group that
    proceeds like any other Stage 3 proposal; bank_q has no candidate
    left after the optimal assignment and is still declined -- honestly,
    not by force -- with a DecisionEvent that says why.
    """
    records = [
        _record("item_x", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 10_000, "refx", date(2026, 1, 1)),
        _record("bank_p", Source.BANK, EntityType.BANK_CREDIT, 10_000, "UTR_P", date(2026, 1, 2)),
        _record("bank_q", Source.BANK, EntityType.BANK_CREDIT, 10_095, "UTR_Q", date(2026, 1, 2)),
    ]
    result = run_stage3_aggregate(records, _settings(), [], [], set(), IdAllocator())

    assert len(result.match_groups) == 1
    promoted = result.match_groups[0]
    assert promoted.cardinality == Cardinality.ONE_TO_ONE
    assert promoted.evidence_score == 1.0
    assert promoted.status == MatchGroupStatus.PENDING_REVIEW
    assert promoted.verified_by == VerifiedBy.NOT_YET_VERIFIED
    assert promoted.proposed_by == ProposedBy.STAGE3_AGGREGATE
    member_ids = {m.record_id for m in result.match_group_members if m.group_id == promoted.group_id}
    assert member_ids == {"item_x", "bank_p"}
    assert result.matched_record_ids == {"item_x", "bank_p"}

    promoted_events = [e for e in result.decision_events if e.reason_code == "AGGREGATE_CONFLICT_RESOLVED_HUNGARIAN"]
    assert len(promoted_events) == 1
    assert promoted_events[0].group_id == promoted.group_id
    assert promoted_events[0].candidate_scores["resolution_path"] == "HUNGARIAN_CONFLICT_RESOLVED"

    # bank_q must NOT be silently dropped: still declined, honestly.
    assert len(result.exceptions) == 1
    exc = result.exceptions[0]
    assert exc.group_id_or_record_id == "rec:bank_q"
    assert exc.category.value == "AMBIGUOUS_AGGREGATION"
    assert exc.evidence["reason"] == "AMBIGUOUS_OVERLAP"
    assert exc.evidence["resolution_path"] == "UNASSIGNED_IN_RESOLVED_CLUSTER"
    assert "bank_q" not in result.matched_record_ids


def test_multi_item_bundle_overlap_still_declines_via_conflict_resolver() -> None:
    """The existing disjoint-windowing scenario (both targets' winning
    subsets are 2-item bundles sharing item_b) must still decline BOTH
    targets exactly as before -- conflict_resolver's own documented
    scope excludes multi-item bundles, so it declines the cluster
    outright rather than forcing a resolution. This is the same fixture
    as test_disjoint_windowing_declines_genuine_overlap_as_ambiguous_aggregation,
    now also asserting the wiring evidence distinguishes this path.
    """
    records = [
        _record("item_a", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 10_000, "refa", date(2026, 1, 1)),
        _record("item_b", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 15_000, "refb", date(2026, 1, 1)),
        _record("item_c", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 5_000, "refc", date(2026, 1, 1)),
        _record("bank_x", Source.BANK, EntityType.BANK_CREDIT, 25_000, "UTR_X", date(2026, 1, 2)),
        _record("bank_y", Source.BANK, EntityType.BANK_CREDIT, 20_000, "UTR_Y", date(2026, 1, 2)),
    ]
    result = run_stage3_aggregate(records, _settings(), [], [], set(), IdAllocator())

    assert result.match_groups == []
    assert result.matched_record_ids == set()
    assert len(result.exceptions) == 2
    for exc in result.exceptions:
        assert exc.category.value == "AMBIGUOUS_AGGREGATION"
        assert exc.evidence["reason"] == "AMBIGUOUS_OVERLAP"
        assert exc.evidence["resolution_path"] == "DECLINED_BUNDLE"
        assert exc.evidence["conflict_resolver_decline_reason"] == "MULTI_ITEM_BUNDLE_NOT_1_TO_1"

    reason_codes = {e.reason_code for e in result.decision_events if e.stage.value == "STAGE3_AGGREGATE"}
    assert "AMBIGUOUS_AGGREGATION" in reason_codes
    assert "AGGREGATE_CONFLICT_RESOLVED_HUNGARIAN" not in reason_codes


def test_resolved_winner_below_evidence_bar_still_declines() -> None:
    """A conflict can reduce cleanly to 1:1 (single atomic item on both
    sides) and still correctly decline: if the Hungarian-optimal winner's
    OWN score/margin doesn't clear STAGE3_THRESHOLD/STAGE3_MIN_MARGIN,
    promoting it would let a conflict resolution bypass the same evidence
    bar every other Stage 3 proposal must clear. Two bank credits, both
    only fitting the same lone settlement item, both far enough off that
    neither one's own score clears STAGE3_THRESHOLD (0.85) alone.
    """
    records = [
        _record("item_x", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 10_000, "refx", date(2026, 1, 1)),
        _record("bank_p", Source.BANK, EntityType.BANK_CREDIT, 10_090, "UTR_P", date(2026, 1, 2)),
        _record("bank_q", Source.BANK, EntityType.BANK_CREDIT, 10_095, "UTR_Q", date(2026, 1, 2)),
    ]
    result = run_stage3_aggregate(records, _settings(), [], [], set(), IdAllocator())

    assert result.match_groups == []
    assert result.matched_record_ids == set()
    assert len(result.exceptions) == 2
    resolution_paths = {e.evidence["target_id"]: e.evidence["resolution_path"] for e in result.exceptions}
    # The higher-scoring of the two (bank_p, diff=90) wins the Hungarian
    # assignment but still doesn't clear the evidence bar; the loser
    # (bank_q) has no candidate left at all.
    assert resolution_paths["rec:bank_p"] == "BELOW_EVIDENCE_BAR"
    assert resolution_paths["rec:bank_q"] == "UNASSIGNED_IN_RESOLVED_CLUSTER"

    reason_codes = {e.reason_code for e in result.decision_events if e.stage.value == "STAGE3_AGGREGATE"}
    assert "AGGREGATE_CONFLICT_RESOLVED_HUNGARIAN" not in reason_codes
