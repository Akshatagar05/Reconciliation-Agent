"""Unit tests for src/recon_agent/matching/stage4_adjustment.py."""

from __future__ import annotations

from datetime import date

from recon_agent.config import Settings
from recon_agent.matching.common import IdAllocator, make_member
from recon_agent.matching.stage4_adjustment import conservation_diff, run_stage4_adjustment
from recon_agent.models import (
    Cardinality,
    CommitPolicy,
    EntityType,
    MatchGroup,
    MatchGroupStatus,
    MemberRole,
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


# --------------------------------------------------------------------
# conservation_diff — the actual §4 arithmetic, tested standalone
# --------------------------------------------------------------------


def test_conservation_diff_is_zero_when_the_equation_balances_exactly() -> None:
    # bank credit = gross - fee - tax - refund + adjustment
    # 100000 - 2000 - 500 - 1000 + 300 = 96800
    gross = 100_000
    fee = -2_000
    tax = -500
    refund = -1_000
    adjustment = 300
    credit = 96_800

    diff = conservation_diff([gross, fee, tax, refund, adjustment], credit)
    assert diff == 0


def test_conservation_diff_is_nonzero_when_the_equation_does_not_balance() -> None:
    gross = 100_000
    fee = -2_000
    credit = 90_000  # doesn't account for the fee correctly

    diff = conservation_diff([gross, fee], credit)
    assert diff == 100_000 - 2_000 - 90_000
    assert diff != 0


def test_conservation_diff_handles_chargeback_and_negative_adjustment() -> None:
    gross = 50_000
    tax = -450
    chargeback = -20_000
    adjustment = -250  # a negative (penalty) adjustment
    credit = 29_300

    diff = conservation_diff([gross, tax, chargeback, adjustment], credit)
    assert diff == 0


# --------------------------------------------------------------------
# run_stage4_adjustment — end-to-end role-based netting search
# --------------------------------------------------------------------


def test_role_tagged_records_net_to_bank_credit_within_tolerance() -> None:
    gross = _record("g1", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 100_000, "ref1", date(2026, 1, 1))
    fee = _record("f1", Source.GATEWAY, EntityType.FEE, -2_000, "ref1", date(2026, 1, 1))
    tax = _record("t1", Source.GATEWAY, EntityType.TAX, -500, "ref1", date(2026, 1, 1))
    bank = _record("b1", Source.BANK, EntityType.BANK_CREDIT, 97_500, "UTR1", date(2026, 1, 2))

    result = run_stage4_adjustment([gross, fee, tax, bank], _settings(), [], [], set(), IdAllocator())

    assert len(result.match_groups) == 1
    group = result.match_groups[0]
    assert group.matched_amount_paise == 97_500
    assert group.residual_amount_paise == 0
    assert group.status == MatchGroupStatus.PENDING_REVIEW
    assert group.verified_by == VerifiedBy.NOT_YET_VERIFIED
    assert group.proposed_by == ProposedBy.STAGE4_ADJUSTMENT
    assert result.matched_record_ids == {"g1", "f1", "t1", "b1"}

    member_roles = {m.record_id: m.role for m in result.match_group_members if m.group_id == group.group_id}
    assert member_roles["g1"] == MemberRole.GROSS
    assert member_roles["f1"] == MemberRole.FEE
    assert member_roles["t1"] == MemberRole.TAX
    assert member_roles["b1"] == MemberRole.CREDIT


def test_adjustment_role_is_correctly_netted_either_sign() -> None:
    gross = _record("g1", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 50_000, "ref1", date(2026, 1, 1))
    positive_adjustment = _record("a1", Source.GATEWAY, EntityType.ADJUSTMENT, 500, "ref1", date(2026, 1, 1))
    bank = _record("b1", Source.BANK, EntityType.BANK_CREDIT, 50_500, "UTR1", date(2026, 1, 2))

    result = run_stage4_adjustment([gross, positive_adjustment, bank], _settings(), [], [], set(), IdAllocator())

    assert len(result.match_groups) == 1
    assert result.match_groups[0].matched_amount_paise == 50_500
    assert result.match_groups[0].residual_amount_paise == 0


def test_no_confident_netting_found_produces_no_false_proposal() -> None:
    gross = _record("g1", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 50_000, "ref1", date(2026, 1, 1))
    bank = _record("b1", Source.BANK, EntityType.BANK_CREDIT, 999_999, "UTR1", date(2026, 1, 2))

    result = run_stage4_adjustment([gross, bank], _settings(), [], [], set(), IdAllocator())

    assert result.match_groups == []
    assert result.matched_record_ids == set()


def test_reversal_records_never_participate_in_bank_side_netting() -> None:
    # A REVERSAL never reaches the bank at all — it must never be pulled
    # into a role-based netting search even if the arithmetic happens to
    # work out.
    gross = _record("g1", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 50_000, "ref1", date(2026, 1, 1))
    reversal = _record("r1", Source.GATEWAY, EntityType.REVERSAL, -50_000, "ref2", date(2026, 1, 1))
    bank = _record("b1", Source.BANK, EntityType.BANK_CREDIT, 1, "UTR1", date(2026, 1, 2))

    result = run_stage4_adjustment([gross, reversal, bank], _settings(), [], [], set(), IdAllocator())

    for group in result.match_groups:
        member_ids = {m.record_id for m in result.match_group_members if m.group_id == group.group_id}
        assert "r1" not in member_ids


def test_every_proposal_has_pending_review_status_and_decision_event() -> None:
    gross = _record("g1", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 100_000, "ref1", date(2026, 1, 1))
    fee = _record("f1", Source.GATEWAY, EntityType.FEE, -2_000, "ref1", date(2026, 1, 1))
    bank = _record("b1", Source.BANK, EntityType.BANK_CREDIT, 98_000, "UTR1", date(2026, 1, 2))

    result = run_stage4_adjustment([gross, fee, bank], _settings(), [], [], set(), IdAllocator())

    assert len(result.match_groups) == 1
    group = result.match_groups[0]
    assert group.status == MatchGroupStatus.PENDING_REVIEW
    assert group.verified_by == VerifiedBy.NOT_YET_VERIFIED
    assert group.verification_result == VerificationResult.NOT_YET_RUN
    assert group.commit_policy == CommitPolicy.AUTO_COMMIT_STAGE2_5_THRESHOLD

    logged_group_ids = {e.group_id for e in result.decision_events}
    assert group.group_id in logged_group_ids
