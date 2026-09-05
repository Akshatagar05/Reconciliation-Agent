"""Unit tests for src/recon_agent/matching/stage1_exact.py."""

from __future__ import annotations

from datetime import date

from recon_agent.config import Settings
from recon_agent.matching.common import IdAllocator
from recon_agent.matching.stage1_exact import run_stage1_exact
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


def test_clean_pair_is_proposed() -> None:
    ledger = _record("r1", Source.LEDGER, EntityType.PAYMENT, 100000, "pay_abc123", date(2026, 7, 1))
    gateway = _record(
        "r2", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 98000, "pay_abc123", date(2026, 7, 2)
    )
    result = run_stage1_exact([ledger, gateway], _settings(), IdAllocator())

    assert len(result.match_groups) == 1
    group = result.match_groups[0]
    assert group.proposed_by == ProposedBy.STAGE1_EXACT
    assert group.expected_amount_paise == 100000
    assert group.matched_amount_paise == 98000
    assert group.residual_amount_paise == 2000
    assert group.status.value == "PENDING_REVIEW"
    assert group.verified_by.value == "NOT_YET_VERIFIED"
    assert group.verification_result.value == "NOT_YET_RUN"
    assert {"r1", "r2"} == result.matched_record_ids
    assert len(result.match_group_members) == 2
    assert len(result.decision_events) == 1


def test_dirty_reference_recovered_by_normalization_still_matches() -> None:
    ledger = _record("r1", Source.LEDGER, EntityType.PAYMENT, 100000, "pay_abc123", date(2026, 7, 1))
    gateway = _record(
        "r2",
        Source.GATEWAY,
        EntityType.GATEWAY_SETTLEMENT,
        98000,
        "  PAY-ABC_123  ",  # whitespace + casing + separator dirtying
        date(2026, 7, 2),
    )
    result = run_stage1_exact([ledger, gateway], _settings(), IdAllocator())
    assert len(result.match_groups) == 1
    assert {"r1", "r2"} == result.matched_record_ids


def test_fee_and_tax_rows_join_the_same_cluster() -> None:
    ledger = _record("r1", Source.LEDGER, EntityType.PAYMENT, 100000, "pay_abc123", date(2026, 7, 1))
    gateway = _record(
        "r2", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 97500, "pay_abc123", date(2026, 7, 1)
    )
    fee = _record("r3", Source.GATEWAY, EntityType.FEE, -2000, "pay_abc123", date(2026, 7, 1))
    tax = _record("r4", Source.GATEWAY, EntityType.TAX, -500, "pay_abc123", date(2026, 7, 1))
    result = run_stage1_exact([ledger, gateway, fee, tax], _settings(), IdAllocator())

    assert len(result.match_groups) == 1
    assert result.matched_record_ids == {"r1", "r2", "r3", "r4"}


def test_duplicate_settlement_makes_identifier_non_unique_and_declines_whole_cluster() -> None:
    ledger = _record("r1", Source.LEDGER, EntityType.PAYMENT, 100000, "pay_abc123", date(2026, 7, 1))
    gateway = _record(
        "r2", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 98000, "pay_abc123", date(2026, 7, 1)
    )
    duplicate = _record(
        "r3", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 98000, "pay_abc123", date(2026, 7, 1)
    )
    result = run_stage1_exact([ledger, gateway, duplicate], _settings(), IdAllocator())

    assert result.match_groups == []
    assert result.matched_record_ids == set()
    assert len(result.decision_events) == 1
    assert result.decision_events[0].reason_code == "NON_UNIQUE_IDENTIFIER"


def test_currency_mismatch_declines_cluster() -> None:
    ledger = _record("r1", Source.LEDGER, EntityType.PAYMENT, 100000, "pay_abc123", date(2026, 7, 1))
    gateway = _record(
        "r2",
        Source.GATEWAY,
        EntityType.GATEWAY_SETTLEMENT,
        98000,
        "pay_abc123",
        date(2026, 7, 1),
        currency="USD",
    )
    result = run_stage1_exact([ledger, gateway], _settings(), IdAllocator())

    assert result.match_groups == []
    assert result.decision_events[0].reason_code == "CURRENCY_MISMATCH"


def test_date_outside_policy_window_declines_cluster() -> None:
    ledger = _record("r1", Source.LEDGER, EntityType.PAYMENT, 100000, "pay_abc123", date(2026, 1, 1))
    gateway = _record(
        "r2", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 98000, "pay_abc123", date(2026, 7, 1)
    )
    result = run_stage1_exact([ledger, gateway], _settings(), IdAllocator())

    assert result.match_groups == []
    assert result.decision_events[0].reason_code == "DATE_OUTSIDE_POLICY_WINDOW"


def test_refund_sharing_original_payment_reference_is_kept_as_separate_group() -> None:
    ledger_payment = _record(
        "r1", Source.LEDGER, EntityType.PAYMENT, 100000, "pay_abc123", date(2026, 7, 1)
    )
    gateway_settlement = _record(
        "r2", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 98000, "pay_abc123", date(2026, 7, 1)
    )
    ledger_refund = _record(
        "r3", Source.LEDGER, EntityType.REFUND, -50000, "pay_abc123", date(2026, 7, 10)
    )
    gateway_refund = _record(
        "r4", Source.GATEWAY, EntityType.REFUND, -50000, "pay_abc123", date(2026, 7, 10)
    )
    result = run_stage1_exact(
        [ledger_payment, gateway_settlement, ledger_refund, gateway_refund], _settings(), IdAllocator()
    )

    assert len(result.match_groups) == 2
    member_sets = []
    members_by_group: dict[str, list[str]] = {}
    for m in result.match_group_members:
        members_by_group.setdefault(m.group_id, []).append(m.record_id)
    for group in result.match_groups:
        member_sets.append(set(members_by_group[group.group_id]))
    assert {"r1", "r2"} in member_sets
    assert {"r3", "r4"} in member_sets


def test_singleton_reference_produces_no_group_and_no_decision_event() -> None:
    lone = _record("r1", Source.LEDGER, EntityType.PAYMENT, 100000, "pay_onlyone", date(2026, 7, 1))
    result = run_stage1_exact([lone], _settings(), IdAllocator())

    assert result.match_groups == []
    assert result.decision_events == []
    assert result.matched_record_ids == set()


def test_bank_credit_never_joins_a_stage1_cluster() -> None:
    # Bank credits carry a UTR-style reference by construction, never the
    # payment reference, so they should never end up in the same
    # normalized-reference cluster as the ledger/gateway rows.
    ledger = _record("r1", Source.LEDGER, EntityType.PAYMENT, 100000, "pay_abc123", date(2026, 7, 1))
    gateway = _record(
        "r2", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 98000, "pay_abc123", date(2026, 7, 1)
    )
    bank = _record(
        "r3", Source.BANK, EntityType.BANK_CREDIT, 98000, "UTR20260702001", date(2026, 7, 2)
    )
    result = run_stage1_exact([ledger, gateway, bank], _settings(), IdAllocator())

    assert len(result.match_groups) == 1
    assert result.matched_record_ids == {"r1", "r2"}
