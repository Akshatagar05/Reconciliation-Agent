"""Unit tests for src/recon_agent/matching/stage2_constrained.py."""

from __future__ import annotations

from datetime import date

from recon_agent.config import Settings
from recon_agent.matching.common import IdAllocator
from recon_agent.matching.stage1_exact import Stage1Result, run_stage1_exact
from recon_agent.matching.stage2_constrained import (
    DEFAULT_THRESHOLD,
    MIN_SCORE_MARGIN,
    run_stage2_constrained,
)
from recon_agent.models import EntityType, NormalizedRecord, ProposedBy, Source


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


def _empty_stage1() -> Stage1Result:
    return Stage1Result()


def test_typo_reference_ledger_gateway_pair_is_matched_by_pass_a() -> None:
    # Reference differs by one character (a typo) — normalization
    # deliberately does not recover this, so it's Stage 2's job.
    ledger = _record("r1", Source.LEDGER, EntityType.PAYMENT, 100000, "pay_abc123xyz9", date(2026, 7, 1))
    gateway = _record(
        "r2",
        Source.GATEWAY,
        EntityType.GATEWAY_SETTLEMENT,
        98000,
        "pay_abc123xyz8",  # last char differs
        date(2026, 7, 1),
    )
    result = run_stage2_constrained([ledger, gateway], _settings(), _empty_stage1(), IdAllocator())

    assert len(result.match_groups) == 1
    group = result.match_groups[0]
    assert group.proposed_by == ProposedBy.STAGE2_CONSTRAINED
    assert group.commit_policy.value == "AUTO_COMMIT_STAGE2_5_THRESHOLD"
    assert group.evidence_score is not None and group.evidence_score >= DEFAULT_THRESHOLD
    assert result.matched_record_ids == {"r1", "r2"}


def test_gateway_bank_single_member_batch_is_matched_by_pass_b() -> None:
    gateway = _record(
        "r1", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 98000, "pay_abc123", date(2026, 7, 1)
    )
    bank = _record(
        "r2", Source.BANK, EntityType.BANK_CREDIT, 98000, "UTR20260702001", date(2026, 7, 2)
    )
    result = run_stage2_constrained([gateway, bank], _settings(), _empty_stage1(), IdAllocator())

    assert len(result.match_groups) == 1
    assert result.matched_record_ids == {"r1", "r2"}


def test_pass_b_extends_an_existing_stage1_group_and_upgrades_its_policy() -> None:
    ledger = _record("r1", Source.LEDGER, EntityType.PAYMENT, 100000, "pay_abc123", date(2026, 7, 1))
    gateway = _record(
        "r2", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 98000, "pay_abc123", date(2026, 7, 1)
    )
    bank = _record(
        "r3", Source.BANK, EntityType.BANK_CREDIT, 98000, "UTR20260702001", date(2026, 7, 2)
    )
    stage1 = run_stage1_exact([ledger, gateway], _settings(), IdAllocator())
    assert len(stage1.match_groups) == 1
    original_group_id = stage1.match_groups[0].group_id

    result = run_stage2_constrained(
        [ledger, gateway, bank], _settings(), stage1, IdAllocator()
    )

    # Still one group overall — the bank leg was attached, not duplicated.
    assert len(result.match_groups) == 1
    group = result.match_groups[0]
    assert group.group_id == original_group_id
    # Because part of its evidence is now constrained-only, the whole
    # group's policy must reflect the stricter Stage 2-5 bar, never the
    # lighter Stage 1 label.
    assert group.proposed_by == ProposedBy.STAGE2_CONSTRAINED
    assert group.commit_policy.value == "AUTO_COMMIT_STAGE2_5_THRESHOLD"
    assert group.matched_amount_paise == 98000
    assert result.matched_record_ids == {"r1", "r2", "r3"}
    member_record_ids = {m.record_id for m in result.match_group_members if m.group_id == original_group_id}
    assert member_record_ids == {"r1", "r2", "r3"}


def test_close_call_ambiguous_candidates_fail_the_margin_check() -> None:
    # Two gateway settlements with (nearly) identical amount, date, and
    # reference — a genuine duplicate-style ambiguity. Neither should be
    # proposed: the margin between best and runner-up is far below
    # MIN_SCORE_MARGIN, so this must NOT pass the margin check even
    # though both individually clear the threshold.
    ledger = _record("r1", Source.LEDGER, EntityType.PAYMENT, 100000, "pay_abc123xyz9", date(2026, 7, 1))
    candidate_a = _record(
        "r2", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 98000, "pay_abc123xyz8", date(2026, 7, 1)
    )
    candidate_b = _record(
        "r3", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 98000, "pay_abc123xyz8", date(2026, 7, 1)
    )
    result = run_stage2_constrained(
        [ledger, candidate_a, candidate_b], _settings(), _empty_stage1(), IdAllocator()
    )

    assert result.match_groups == []
    assert result.matched_record_ids == set()
    events = [e for e in result.decision_events if e.reason_code == "INSUFFICIENT_MARGIN"]
    assert len(events) == 1
    assert events[0].candidate_scores["runner_up_score"] is not None
    margin = (
        events[0].candidate_scores["composite_score"] - events[0].candidate_scores["runner_up_score"]
    )
    assert margin < MIN_SCORE_MARGIN


def test_amount_far_outside_tolerance_is_never_proposed() -> None:
    gateway = _record(
        "r1", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 98000, "pay_abc123", date(2026, 7, 1)
    )
    bank = _record(
        # Amount is wildly different — nowhere near tolerance.
        "r2",
        Source.BANK,
        EntityType.BANK_CREDIT,
        5_000_000,
        "UTR20260702001",
        date(2026, 7, 2),
    )
    result = run_stage2_constrained([gateway, bank], _settings(), _empty_stage1(), IdAllocator())

    assert result.match_groups == []
    assert result.matched_record_ids == set()


def test_date_far_outside_window_is_never_proposed() -> None:
    gateway = _record(
        "r1", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 98000, "pay_abc123", date(2026, 1, 1)
    )
    bank = _record(
        "r2", Source.BANK, EntityType.BANK_CREDIT, 98000, "UTR20260702001", date(2026, 7, 2)
    )
    result = run_stage2_constrained([gateway, bank], _settings(), _empty_stage1(), IdAllocator())

    assert result.match_groups == []


def test_currency_mismatch_excludes_candidate_entirely() -> None:
    gateway = _record(
        "r1", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 98000, "pay_abc123", date(2026, 7, 1)
    )
    bank = _record(
        "r2",
        Source.BANK,
        EntityType.BANK_CREDIT,
        98000,
        "UTR20260702001",
        date(2026, 7, 2),
        currency="USD",
    )
    result = run_stage2_constrained([gateway, bank], _settings(), _empty_stage1(), IdAllocator())

    assert result.match_groups == []


def test_partial_settlement_split_across_two_bank_credits_is_not_forced() -> None:
    # A single settlement's payout split across two bank credits, neither
    # of which alone matches the settlement's full amount within
    # tolerance. Stage 2 does a simple 1:1 tolerance check only (the full
    # subset-sum aggregation search is explicitly out of scope) so it
    # must correctly decline rather than force a wrong single-leg match.
    gateway = _record(
        "r1", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 100000, "pay_abc123", date(2026, 7, 1)
    )
    bank_1 = _record(
        "r2", Source.BANK, EntityType.BANK_CREDIT, 60000, "UTR20260702001", date(2026, 7, 2)
    )
    bank_2 = _record(
        "r3", Source.BANK, EntityType.BANK_CREDIT, 40000, "UTR20260703001", date(2026, 7, 3)
    )
    result = run_stage2_constrained(
        [gateway, bank_1, bank_2], _settings(), _empty_stage1(), IdAllocator()
    )

    assert result.match_groups == []
    assert result.matched_record_ids == set()


def test_no_candidates_logs_decision_event_with_no_group() -> None:
    lone_gateway = _record(
        "r1", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 98000, "pay_abc123", date(2026, 7, 1)
    )
    result = run_stage2_constrained([lone_gateway], _settings(), _empty_stage1(), IdAllocator())

    assert result.match_groups == []
    assert any(e.reason_code == "NO_ELIGIBLE_CANDIDATE" for e in result.decision_events)
