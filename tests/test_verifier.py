"""Unit tests for src/recon_agent/verification/verifier.py (Stage 5).

Every MatchGroup/MatchGroupMember/NormalizedRecord fixture here is built
directly, without running the actual matching pipeline — this stage's
scope is the verifier module in isolation, not an end-to-end pipeline
test (see test_stage1to4_integration.py for that).
"""

from __future__ import annotations

from datetime import date

import pytest

from recon_agent.config import Settings
from recon_agent.models import (
    Cardinality,
    CommitPolicy,
    EntityType,
    MatchGroup,
    MatchGroupMember,
    MatchGroupStatus,
    MemberRole,
    NormalizedRecord,
    ProposedBy,
    Source,
    VerificationResult,
    VerifiedBy,
)
from recon_agent.verification.verifier import verify_match_group

# ---------------------------------------------------------------------------
# Shared fixture builders
# ---------------------------------------------------------------------------


def _record(
    record_id: str,
    source: Source,
    entity_type: EntityType,
    amount_paise: int,
    reference: str = "REF001",
    occurred_at: date = date(2026, 1, 1),
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


def _member(
    group_id: str,
    record: NormalizedRecord,
    role: MemberRole,
    signed_amount_paise: int,
) -> MatchGroupMember:
    return MatchGroupMember(
        group_id=group_id,
        record_id=record.record_id,
        source=record.source,
        role=role,
        signed_amount_paise=signed_amount_paise,
        allocated_amount_paise=abs(signed_amount_paise),
    )


def _settings() -> Settings:
    return Settings()


def _group(
    group_id: str,
    proposed_by: ProposedBy,
    *,
    evidence_score: float | None = None,
    runner_up_score: float | None = None,
    score_margin: float | None = None,
    threshold_applied: float = 0.70,
    threshold_version: str = "v1",
    policy_checks: dict | None = None,
    expected_amount_paise: int = 0,
    matched_amount_paise: int = 0,
    commit_policy: CommitPolicy = CommitPolicy.AUTO_COMMIT_STAGE2_5_THRESHOLD,
) -> MatchGroup:
    return MatchGroup(
        group_id=group_id,
        cardinality=Cardinality.ONE_TO_ONE,
        expected_amount_paise=expected_amount_paise,
        matched_amount_paise=matched_amount_paise,
        residual_amount_paise=expected_amount_paise - matched_amount_paise,
        status=MatchGroupStatus.PENDING_REVIEW,
        evidence_score=evidence_score,
        runner_up_score=runner_up_score,
        score_margin=score_margin,
        threshold_applied=threshold_applied,
        threshold_version=threshold_version,
        policy_checks=policy_checks if policy_checks is not None else {},
        proposed_by=proposed_by,
        verified_by=VerifiedBy.NOT_YET_VERIFIED,
        commit_policy=commit_policy,
        verification_result=VerificationResult.NOT_YET_RUN,
    )


# ===========================================================================
# Stage 1 (exact)
# ===========================================================================


def test_stage1_passes_when_identifier_unique_and_currency_date_policy_ok() -> None:
    ledger = _record("led_1", Source.LEDGER, EntityType.LEDGER_ENTRY, 100_000, occurred_at=date(2026, 1, 1))
    gateway = _record("gw_1", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 100_000, occurred_at=date(2026, 1, 2))
    records_by_id = {ledger.record_id: ledger, gateway.record_id: gateway}

    group = _group(
        "grp_1",
        ProposedBy.STAGE1_EXACT,
        threshold_applied=1.0,
        policy_checks={"identifier_unique": True, "currency_consistent": True, "date_policy_window": True},
        commit_policy=CommitPolicy.AUTO_COMMIT_STAGE1,
    )
    members = [
        _member("grp_1", ledger, MemberRole.GROSS, 100_000),
        _member("grp_1", gateway, MemberRole.GROSS, 100_000),
    ]

    outcome = verify_match_group(group, members, records_by_id, settings=_settings())

    assert outcome.passed
    assert outcome.verification_result == VerificationResult.PASSED
    assert outcome.verified_by == VerifiedBy.FINANCIAL_AND_EVIDENCE_VERIFIER
    assert outcome.checks["identifier_unique"] is True
    assert outcome.checks["currency_consistent"] is True
    assert outcome.checks["date_policy_window"] is True


def test_stage1_fails_uniqueness_even_though_stored_policy_checks_claims_it_was_unique() -> None:
    # Two GATEWAY_SETTLEMENT rows (same source, same entity_type) is
    # exactly the injected-duplicate slot collision from stage1_exact.py.
    # The group's own stored policy_checks lies and claims it was unique
    # — the verifier must catch this independently, not just read the
    # dict back.
    dup_a = _record("gw_1", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 100_000, occurred_at=date(2026, 1, 1))
    dup_b = _record("gw_2", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 100_000, occurred_at=date(2026, 1, 1))
    records_by_id = {dup_a.record_id: dup_a, dup_b.record_id: dup_b}

    group = _group(
        "grp_2",
        ProposedBy.STAGE1_EXACT,
        threshold_applied=1.0,
        policy_checks={"identifier_unique": True, "currency_consistent": True, "date_policy_window": True},
        commit_policy=CommitPolicy.AUTO_COMMIT_STAGE1,
    )
    members = [
        _member("grp_2", dup_a, MemberRole.GROSS, 100_000),
        _member("grp_2", dup_b, MemberRole.GROSS, 100_000),
    ]

    outcome = verify_match_group(group, members, records_by_id, settings=_settings())

    assert not outcome.passed
    assert outcome.verification_result == VerificationResult.FAILED_UNIQUENESS
    assert outcome.checks["identifier_unique"] is False


def test_stage1_fails_currency_even_though_stored_policy_checks_claims_it_was_consistent() -> None:
    ledger = _record("led_1", Source.LEDGER, EntityType.LEDGER_ENTRY, 100_000, currency="INR")
    gateway = _record("gw_1", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 100_000, currency="USD")
    records_by_id = {ledger.record_id: ledger, gateway.record_id: gateway}

    group = _group(
        "grp_3",
        ProposedBy.STAGE1_EXACT,
        threshold_applied=1.0,
        policy_checks={"identifier_unique": True, "currency_consistent": True, "date_policy_window": True},
        commit_policy=CommitPolicy.AUTO_COMMIT_STAGE1,
    )
    members = [
        _member("grp_3", ledger, MemberRole.GROSS, 100_000),
        _member("grp_3", gateway, MemberRole.GROSS, 100_000),
    ]

    outcome = verify_match_group(group, members, records_by_id, settings=_settings())

    assert not outcome.passed
    assert outcome.verification_result == VerificationResult.FAILED_CURRENCY
    assert outcome.checks["currency_consistent"] is False


# ===========================================================================
# Stage 2 (constrained)
# ===========================================================================


def test_stage2_passes_when_margin_conservation_and_currency_all_hold() -> None:
    gateway = _record("gw_1", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 100_000)
    bank = _record("bank_1", Source.BANK, EntityType.BANK_CREDIT, 100_000)
    records_by_id = {gateway.record_id: gateway, bank.record_id: bank}

    group = _group("grp_10", ProposedBy.STAGE2_CONSTRAINED, evidence_score=0.90, runner_up_score=0.70, score_margin=0.20)
    members = [
        _member("grp_10", gateway, MemberRole.GROSS, 100_000),
        _member("grp_10", bank, MemberRole.CREDIT, 100_000),
    ]

    outcome = verify_match_group(group, members, records_by_id, settings=_settings())

    assert outcome.passed
    assert outcome.checks["margin_sufficient"] is True
    assert outcome.checks["conservation_balances"] is True
    assert outcome.checks["currency_consistent"] is True


def test_stage2_fails_margin_when_score_margin_too_thin() -> None:
    gateway = _record("gw_1", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 100_000)
    bank = _record("bank_1", Source.BANK, EntityType.BANK_CREDIT, 100_000)
    records_by_id = {gateway.record_id: gateway, bank.record_id: bank}

    group = _group("grp_11", ProposedBy.STAGE2_CONSTRAINED, evidence_score=0.72, runner_up_score=0.71, score_margin=0.01)
    members = [
        _member("grp_11", gateway, MemberRole.GROSS, 100_000),
        _member("grp_11", bank, MemberRole.CREDIT, 100_000),
    ]

    outcome = verify_match_group(group, members, records_by_id, settings=_settings())

    assert not outcome.passed
    assert outcome.verification_result == VerificationResult.FAILED_MARGIN
    assert outcome.checks["margin_sufficient"] is False


def test_stage2_fails_conservation_even_though_stored_amount_tolerance_check_claims_it_passed() -> None:
    gateway = _record("gw_1", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 100_000)
    bank = _record("bank_1", Source.BANK, EntityType.BANK_CREDIT, 90_000)  # doesn't actually net out
    records_by_id = {gateway.record_id: gateway, bank.record_id: bank}

    group = _group(
        "grp_12",
        ProposedBy.STAGE2_CONSTRAINED,
        evidence_score=0.90,
        runner_up_score=0.70,
        score_margin=0.20,
        policy_checks={"amount_tolerance": True},  # the proposing stage's own (wrong) claim
    )
    members = [
        _member("grp_12", gateway, MemberRole.GROSS, 100_000),
        _member("grp_12", bank, MemberRole.CREDIT, 90_000),
    ]

    outcome = verify_match_group(group, members, records_by_id, settings=_settings())

    assert not outcome.passed
    assert outcome.verification_result == VerificationResult.FAILED_CONSERVATION
    assert outcome.checks["conservation_balances"] is False
    assert outcome.checks["conservation_diff_paise"] == 100_000 - 90_000


def test_stage2_fails_currency_even_though_stored_policy_checks_claims_it_was_consistent() -> None:
    gateway = _record("gw_1", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 100_000, currency="INR")
    bank = _record("bank_1", Source.BANK, EntityType.BANK_CREDIT, 100_000, currency="USD")
    records_by_id = {gateway.record_id: gateway, bank.record_id: bank}

    group = _group(
        "grp_13",
        ProposedBy.STAGE2_CONSTRAINED,
        evidence_score=0.90,
        runner_up_score=0.70,
        score_margin=0.20,
        policy_checks={"currency_consistent": True},
    )
    members = [
        _member("grp_13", gateway, MemberRole.GROSS, 100_000),
        _member("grp_13", bank, MemberRole.CREDIT, 100_000),
    ]

    outcome = verify_match_group(group, members, records_by_id, settings=_settings())

    assert not outcome.passed
    assert outcome.verification_result == VerificationResult.FAILED_CURRENCY
    assert outcome.checks["currency_consistent"] is False


# ===========================================================================
# Stage 3 (aggregate)
# ===========================================================================


def test_stage3_passes_when_several_gross_legs_conserve_against_one_credit() -> None:
    leg1 = _record("gw_1", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 40_000)
    leg2 = _record("gw_2", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 35_000)
    leg3 = _record("gw_3", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 25_000)
    bank = _record("bank_1", Source.BANK, EntityType.BANK_CREDIT, 100_000)
    records_by_id = {r.record_id: r for r in (leg1, leg2, leg3, bank)}

    group = _group(
        "grp_20",
        ProposedBy.STAGE3_AGGREGATE,
        evidence_score=0.95,
        runner_up_score=0.80,
        score_margin=0.15,
        threshold_applied=0.85,
    )
    members = [
        _member("grp_20", leg1, MemberRole.GROSS, 40_000),
        _member("grp_20", leg2, MemberRole.GROSS, 35_000),
        _member("grp_20", leg3, MemberRole.GROSS, 25_000),
        _member("grp_20", bank, MemberRole.CREDIT, 100_000),
    ]

    outcome = verify_match_group(group, members, records_by_id, settings=_settings())

    assert outcome.passed


def test_stage3_fails_margin_when_score_margin_too_thin() -> None:
    leg1 = _record("gw_1", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 60_000)
    leg2 = _record("gw_2", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 40_000)
    bank = _record("bank_1", Source.BANK, EntityType.BANK_CREDIT, 100_000)
    records_by_id = {r.record_id: r for r in (leg1, leg2, bank)}

    group = _group(
        "grp_21",
        ProposedBy.STAGE3_AGGREGATE,
        evidence_score=0.86,
        runner_up_score=0.84,
        score_margin=0.02,
        threshold_applied=0.85,
    )
    members = [
        _member("grp_21", leg1, MemberRole.GROSS, 60_000),
        _member("grp_21", leg2, MemberRole.GROSS, 40_000),
        _member("grp_21", bank, MemberRole.CREDIT, 100_000),
    ]

    outcome = verify_match_group(group, members, records_by_id, settings=_settings())

    assert not outcome.passed
    assert outcome.verification_result == VerificationResult.FAILED_MARGIN


def test_stage3_fails_conservation_when_aggregate_legs_do_not_sum_to_the_credit() -> None:
    leg1 = _record("gw_1", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 60_000)
    leg2 = _record("gw_2", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 40_000)
    bank = _record("bank_1", Source.BANK, EntityType.BANK_CREDIT, 105_000)  # 100_000 claimed, 5_000 off
    records_by_id = {r.record_id: r for r in (leg1, leg2, bank)}

    group = _group(
        "grp_22",
        ProposedBy.STAGE3_AGGREGATE,
        evidence_score=0.95,
        runner_up_score=0.80,
        score_margin=0.15,
        threshold_applied=0.85,
    )
    members = [
        _member("grp_22", leg1, MemberRole.GROSS, 60_000),
        _member("grp_22", leg2, MemberRole.GROSS, 40_000),
        _member("grp_22", bank, MemberRole.CREDIT, 105_000),
    ]

    outcome = verify_match_group(group, members, records_by_id, settings=_settings())

    assert not outcome.passed
    assert outcome.verification_result == VerificationResult.FAILED_CONSERVATION


def test_stage3_fails_currency_when_one_leg_uses_a_different_currency() -> None:
    leg1 = _record("gw_1", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 60_000, currency="INR")
    leg2 = _record("gw_2", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 40_000, currency="AED")
    bank = _record("bank_1", Source.BANK, EntityType.BANK_CREDIT, 100_000, currency="INR")
    records_by_id = {r.record_id: r for r in (leg1, leg2, bank)}

    group = _group(
        "grp_23",
        ProposedBy.STAGE3_AGGREGATE,
        evidence_score=0.95,
        runner_up_score=0.80,
        score_margin=0.15,
        threshold_applied=0.85,
    )
    members = [
        _member("grp_23", leg1, MemberRole.GROSS, 60_000),
        _member("grp_23", leg2, MemberRole.GROSS, 40_000),
        _member("grp_23", bank, MemberRole.CREDIT, 100_000),
    ]

    outcome = verify_match_group(group, members, records_by_id, settings=_settings())

    assert not outcome.passed
    assert outcome.verification_result == VerificationResult.FAILED_CURRENCY


# ---------------------------------------------------------------------------
# Stage 6 bugfix regression coverage: the verifier's independent
# conservation recomputation (matching.common.recompute_group_conservation)
# used to get two real shapes wrong. See that function's docstring and
# README.md's Stage 6 note for the full story; these pin down both.
# ---------------------------------------------------------------------------


def test_stage3_passes_when_a_pure_refund_reflected_across_all_three_legs_conserves() -> None:
    # Root cause 1: a pure REFUND/CHARGEBACK/REVERSAL event has no
    # GROSS-role and no CREDIT-role leg at all — it's a standalone
    # outflow, not a payment, so it surfaces as the SAME
    # signed_amount_paise reflected on its ledger/gateway/bank legs.
    # Three-way agreement on that one amount trivially IS conservation;
    # it must not be treated as "nothing to reconcile against".
    ledger = _record("ledger_1", Source.LEDGER, EntityType.REFUND, -3_422_830)
    gateway = _record("gw_1b", Source.GATEWAY, EntityType.REFUND, -3_422_830)
    bank = _record("bank_1b", Source.BANK, EntityType.REFUND, -3_422_830)
    records_by_id = {r.record_id: r for r in (ledger, gateway, bank)}

    group = _group(
        "grp_28",
        ProposedBy.STAGE2_CONSTRAINED,
        evidence_score=0.97,
        runner_up_score=0.80,
        score_margin=0.17,
        threshold_applied=0.85,
    )
    members = [
        _member("grp_28", ledger, MemberRole.REFUND, -3_422_830),
        _member("grp_28", gateway, MemberRole.REFUND, -3_422_830),
        _member("grp_28", bank, MemberRole.REFUND, -3_422_830),
    ]

    outcome = verify_match_group(group, members, records_by_id, settings=_settings())

    assert outcome.passed
    assert outcome.checks["conservation_diff_paise"] == 0


def test_stage3_fails_when_the_three_refund_legs_do_not_actually_agree() -> None:
    # Same shape as above, but one leg's amount doesn't match the other
    # two — the trivial same-role/same-amount case must not become a
    # rubber stamp for any all-one-role group.
    ledger = _record("ledger_2", Source.LEDGER, EntityType.REFUND, -3_422_830)
    gateway = _record("gw_2b", Source.GATEWAY, EntityType.REFUND, -3_422_830)
    bank = _record("bank_2b", Source.BANK, EntityType.REFUND, -3_000_000)
    records_by_id = {r.record_id: r for r in (ledger, gateway, bank)}

    group = _group(
        "grp_29",
        ProposedBy.STAGE2_CONSTRAINED,
        evidence_score=0.97,
        runner_up_score=0.80,
        score_margin=0.17,
        threshold_applied=0.85,
    )
    members = [
        _member("grp_29", ledger, MemberRole.REFUND, -3_422_830),
        _member("grp_29", gateway, MemberRole.REFUND, -3_422_830),
        _member("grp_29", bank, MemberRole.REFUND, -3_000_000),
    ]

    outcome = verify_match_group(group, members, records_by_id, settings=_settings())

    assert not outcome.passed
    assert outcome.verification_result == VerificationResult.FAILED_CONSERVATION


def test_stage3_passes_a_multi_credit_split_without_double_counting_the_ledger_side_gross() -> None:
    # Root cause 2: a partial settlement split across two bank credits,
    # where the settlement's LEDGER-side (pre-fee gross) and
    # GATEWAY-side (already fee/GST-netted) rows for the SAME event
    # both appear in the group. Naively summing every non-CREDIT member
    # double-counts that one settlement's money — the correct total is
    # the GATEWAY-side (net) figure alone, matched against the sum of
    # the two bank credits.
    ledger = _record("ledger_3", Source.LEDGER, EntityType.PAYMENT, 1_151_369)
    gateway = _record("gw_3b", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 1_120_802)
    bank1 = _record("bank_3a", Source.BANK, EntityType.BANK_CREDIT, 672_481)
    bank2 = _record("bank_3b", Source.BANK, EntityType.BANK_CREDIT, 448_321)
    records_by_id = {r.record_id: r for r in (ledger, gateway, bank1, bank2)}

    group = _group(
        "grp_30b",
        ProposedBy.STAGE3_AGGREGATE,
        evidence_score=0.95,
        runner_up_score=0.78,
        score_margin=0.17,
        threshold_applied=0.85,
    )
    members = [
        _member("grp_30b", ledger, MemberRole.GROSS, 1_151_369),
        _member("grp_30b", gateway, MemberRole.GROSS, 1_120_802),
        _member("grp_30b", bank1, MemberRole.CREDIT, 672_481),
        _member("grp_30b", bank2, MemberRole.CREDIT, 448_321),
    ]

    outcome = verify_match_group(group, members, records_by_id, settings=_settings())

    assert outcome.passed
    assert outcome.checks["conservation_diff_paise"] == 0


def test_stage3_still_catches_a_genuine_multi_credit_mismatch() -> None:
    # Same ledger+gateway shape as above, but the two bank credits
    # genuinely don't sum to the settlement's net amount — the
    # double-counting fix must not turn into a blanket pass for every
    # multi-CREDIT group.
    ledger = _record("ledger_4", Source.LEDGER, EntityType.PAYMENT, 1_151_369)
    gateway = _record("gw_4b", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 1_120_802)
    bank1 = _record("bank_4a", Source.BANK, EntityType.BANK_CREDIT, 672_481)
    bank2 = _record("bank_4b", Source.BANK, EntityType.BANK_CREDIT, 400_000)
    records_by_id = {r.record_id: r for r in (ledger, gateway, bank1, bank2)}

    group = _group(
        "grp_31b",
        ProposedBy.STAGE3_AGGREGATE,
        evidence_score=0.95,
        runner_up_score=0.78,
        score_margin=0.17,
        threshold_applied=0.85,
    )
    members = [
        _member("grp_31b", ledger, MemberRole.GROSS, 1_151_369),
        _member("grp_31b", gateway, MemberRole.GROSS, 1_120_802),
        _member("grp_31b", bank1, MemberRole.CREDIT, 672_481),
        _member("grp_31b", bank2, MemberRole.CREDIT, 400_000),
    ]

    outcome = verify_match_group(group, members, records_by_id, settings=_settings())

    assert not outcome.passed
    assert outcome.verification_result == VerificationResult.FAILED_CONSERVATION


# ===========================================================================
# Stage 4 (adjustment)
# ===========================================================================


def _stage4_records() -> dict[str, NormalizedRecord]:
    gross = _record("gw_1", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 100_000)
    fee = _record("fee_1", Source.GATEWAY, EntityType.FEE, -2_000)
    tax = _record("tax_1", Source.GATEWAY, EntityType.TAX, -500)
    refund = _record("refund_1", Source.GATEWAY, EntityType.REFUND, -1_000)
    adjustment = _record("adj_1", Source.GATEWAY, EntityType.ADJUSTMENT, 300)
    bank = _record("bank_1", Source.BANK, EntityType.BANK_CREDIT, 96_800)
    return {r.record_id: r for r in (gross, fee, tax, refund, adjustment, bank)}


def _stage4_members(group_id: str, records: dict[str, NormalizedRecord], bank_amount: int | None = None) -> list[MatchGroupMember]:
    bank_amount = records["bank_1"].amount_paise if bank_amount is None else bank_amount
    return [
        _member(group_id, records["gw_1"], MemberRole.GROSS, 100_000),
        _member(group_id, records["fee_1"], MemberRole.FEE, -2_000),
        _member(group_id, records["tax_1"], MemberRole.TAX, -500),
        _member(group_id, records["refund_1"], MemberRole.REFUND, -1_000),
        _member(group_id, records["adj_1"], MemberRole.ADJUSTMENT, 300),
        _member(group_id, records["bank_1"], MemberRole.CREDIT, bank_amount),
    ]


def test_stage4_passes_when_full_role_based_netting_conserves() -> None:
    records = _stage4_records()
    group = _group(
        "grp_30",
        ProposedBy.STAGE4_ADJUSTMENT,
        evidence_score=0.93,
        runner_up_score=0.75,
        score_margin=0.18,
        threshold_applied=0.85,
    )
    members = _stage4_members("grp_30", records)

    outcome = verify_match_group(group, members, records, settings=_settings())

    assert outcome.passed
    assert outcome.checks["conservation_diff_paise"] == 0


def test_stage4_fails_margin_when_score_margin_too_thin() -> None:
    records = _stage4_records()
    group = _group(
        "grp_31",
        ProposedBy.STAGE4_ADJUSTMENT,
        evidence_score=0.86,
        runner_up_score=0.85,
        score_margin=0.01,
        threshold_applied=0.85,
    )
    members = _stage4_members("grp_31", records)

    outcome = verify_match_group(group, members, records, settings=_settings())

    assert not outcome.passed
    assert outcome.verification_result == VerificationResult.FAILED_MARGIN


def test_stage4_fails_conservation_when_netting_does_not_balance() -> None:
    records = _stage4_records()
    group = _group(
        "grp_32",
        ProposedBy.STAGE4_ADJUSTMENT,
        evidence_score=0.93,
        runner_up_score=0.75,
        score_margin=0.18,
        threshold_applied=0.85,
    )
    # Bank credit doesn't actually match gross - fee - tax - refund + adjustment.
    members = _stage4_members("grp_32", records, bank_amount=90_000)

    outcome = verify_match_group(group, members, records, settings=_settings())

    assert not outcome.passed
    assert outcome.verification_result == VerificationResult.FAILED_CONSERVATION


def test_stage4_fails_currency_when_a_netted_leg_uses_a_different_currency() -> None:
    records = _stage4_records()
    records["fee_1"] = _record("fee_1", Source.GATEWAY, EntityType.FEE, -2_000, currency="USD")
    group = _group(
        "grp_33",
        ProposedBy.STAGE4_ADJUSTMENT,
        evidence_score=0.93,
        runner_up_score=0.75,
        score_margin=0.18,
        threshold_applied=0.85,
    )
    members = _stage4_members("grp_33", records)

    outcome = verify_match_group(group, members, records, settings=_settings())

    assert not outcome.passed
    assert outcome.verification_result == VerificationResult.FAILED_CURRENCY


def test_stage4_fails_conservation_when_a_member_role_amount_sign_is_contradictory() -> None:
    records = _stage4_records()
    group = _group(
        "grp_34",
        ProposedBy.STAGE4_ADJUSTMENT,
        evidence_score=0.93,
        runner_up_score=0.75,
        score_margin=0.18,
        threshold_applied=0.85,
    )
    members = _stage4_members("grp_34", records)
    # A FEE-role member is not allowed to carry a positive amount.
    members[1] = _member("grp_34", records["fee_1"], MemberRole.FEE, 2_000)

    outcome = verify_match_group(group, members, records, settings=_settings())

    assert not outcome.passed
    assert outcome.verification_result == VerificationResult.FAILED_CONSERVATION
    assert outcome.checks["role_amount_consistent"] is False


def test_verifier_recomputes_conservation_independently_of_a_deceptively_strong_evidence_score() -> None:
    """The whole point of §2's 'necessary but not sufficient': a group can
    look extremely confident on paper (high evidence_score, wide margin,
    policy_checks that all claim success) and still not actually conserve
    money once you recompute from the raw MatchGroupMembers. The verifier
    must still fail it.
    """
    records = _stage4_records()
    group = _group(
        "grp_35",
        ProposedBy.STAGE4_ADJUSTMENT,
        evidence_score=0.99,  # looks great
        runner_up_score=0.40,  # huge margin
        score_margin=0.59,
        threshold_applied=0.85,
        policy_checks={
            "conservation_balances": True,  # the proposing stage's own (wrong) claim
            "currency_consistent": True,
            "amount_tolerance": True,
        },
    )
    # The actual members don't conserve: bank credit is far off from
    # gross - fee - tax - refund + adjustment.
    members = _stage4_members("grp_35", records, bank_amount=50_000)

    outcome = verify_match_group(group, members, records, settings=_settings())

    assert not outcome.passed
    assert outcome.verification_result == VerificationResult.FAILED_CONSERVATION
    assert outcome.checks["conservation_balances"] is False
    assert outcome.checks["conservation_diff_paise"] != 0


# ===========================================================================
# Stage 6 (LLM) — always human review, no exceptions
# ===========================================================================


def test_stage6_always_resolves_to_human_review_regardless_of_evidence() -> None:
    gross = _record("gw_1", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 100_000)
    bank = _record("bank_1", Source.BANK, EntityType.BANK_CREDIT, 100_000)
    records_by_id = {gross.record_id: gross, bank.record_id: bank}

    # Even with evidence that would otherwise sail through Stage 2-5's bar.
    group = _group(
        "grp_40",
        ProposedBy.STAGE6_LLM,
        evidence_score=0.99,
        runner_up_score=0.10,
        score_margin=0.89,
        commit_policy=CommitPolicy.HUMAN_REVIEW_REQUIRED,
    )
    members = [
        _member("grp_40", gross, MemberRole.GROSS, 100_000),
        _member("grp_40", bank, MemberRole.CREDIT, 100_000),
    ]

    outcome = verify_match_group(group, members, records_by_id, settings=_settings())

    assert not outcome.passed
    assert outcome.verification_result == VerificationResult.NOT_YET_RUN
    assert outcome.verified_by == VerifiedBy.NOT_YET_VERIFIED
    assert "human" in outcome.explanation.lower()


def test_stage5_fuzzy_now_implemented_shares_stage2_to_4_policy_bar() -> None:
    """Regression test, post-Stage-9 (relay): this test used to assert
    that ``STAGE5_FUZZY`` raised ``NotImplementedError`` here, back when
    Stage 5 fuzzy matching itself didn't exist yet (see this module's
    original docstring note, and verifier.py's own "Stage 5 wiring
    note"). Now that stage5_fuzzy.py exists and proposes real
    STAGE5_FUZZY groups, the verifier must route them through the same
    shared Stage 2-5 evidence bar as every other aggregate/adjustment/
    fuzzy proposal (§2) — not raise. A gateway<->bank pair with clean
    margin/conservation/currency evidence should PASS exactly like an
    equivalent STAGE2_CONSTRAINED group would.
    """
    gross = _record("gw_1", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 100_000)
    bank = _record("bank_1", Source.BANK, EntityType.BANK_CREDIT, 100_000)
    records_by_id = {gross.record_id: gross, bank.record_id: bank}
    group = _group(
        "grp_50",
        ProposedBy.STAGE5_FUZZY,
        evidence_score=0.9,
        runner_up_score=0.5,
        score_margin=0.4,
        expected_amount_paise=100_000,
        matched_amount_paise=100_000,
    )
    members = [
        _member("grp_50", gross, MemberRole.GROSS, 100_000),
        _member("grp_50", bank, MemberRole.CREDIT, 100_000),
    ]

    outcome = verify_match_group(group, members, records_by_id, settings=_settings())

    assert outcome.passed
    assert outcome.verification_result == VerificationResult.PASSED
    assert outcome.verified_by == VerifiedBy.FINANCIAL_AND_EVIDENCE_VERIFIER


def test_unimplemented_proposed_by_still_raises_rather_than_silently_passing() -> None:
    """Every real ``ProposedBy`` value is now covered (Stage 1 through
    Stage 6) — this test exercises the fail-closed default branch
    itself, for a hypothetical future enum value, by directly mutating
    a group's ``proposed_by`` past construction-time validation (no
    ``validate_assignment`` configured on ``MatchGroup``, so plain
    attribute assignment bypasses enum validation the same way a
    not-yet-added enum member would appear to this function) rather
    than asserting a specific stage is unimplemented (there isn't one
    anymore)."""
    gross = _record("gw_1", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 100_000)
    records_by_id = {gross.record_id: gross}
    group = _group("grp_51", ProposedBy.STAGE1_EXACT, evidence_score=0.9, runner_up_score=0.5, score_margin=0.4)
    members = [_member("grp_51", gross, MemberRole.GROSS, 100_000)]

    group.proposed_by = "SOME_FUTURE_STAGE_NOT_YET_IMPLEMENTED"  # type: ignore[assignment]

    with pytest.raises(NotImplementedError):
        verify_match_group(group, members, records_by_id, settings=_settings())
