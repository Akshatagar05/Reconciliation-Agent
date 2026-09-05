"""Schema smoke tests for Stage 1.

These only confirm that every model in ARCHITECTURE.md §7 constructs
successfully with valid data and round-trips through JSON. They contain
no matching, normalization, or verification logic — that is out of
scope for this stage.
"""

from __future__ import annotations

from datetime import date, datetime

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
    ReconciliationRun,
    ReconciliationRunStatus,
    Source,
    VerificationResult,
    VerifiedBy,
)


def test_reconciliation_run_round_trip() -> None:
    run = ReconciliationRun(
        run_id="run_001",
        input_hash="abc123",
        rules_version="v1",
        model_version="v1",
        threshold_version="v1",
        policy_version="v1",
        started_at=datetime(2026, 8, 24, 9, 0, 0),
        status=ReconciliationRunStatus.RUNNING,
    )
    assert ReconciliationRun.model_validate_json(run.model_dump_json()) == run


def test_normalized_record_round_trip() -> None:
    record = NormalizedRecord(
        record_id="rec_001",
        source=Source.BANK,
        entity_type=EntityType.BANK_CREDIT,
        amount_paise=100000,
        currency="INR",
        reference="UTR123456",
        counterparty="Razorpay Settlements Pvt Ltd",
        occurred_at=date(2026, 8, 20),
        raw_hash="deadbeef",
    )
    assert NormalizedRecord.model_validate_json(record.model_dump_json()) == record


def test_match_group_round_trip() -> None:
    group = MatchGroup(
        group_id="grp_001",
        cardinality=Cardinality.ONE_TO_ONE,
        expected_amount_paise=100000,
        matched_amount_paise=100000,
        residual_amount_paise=0,
        status=MatchGroupStatus.VERIFIED,
        evidence_score=0.98,
        runner_up_score=0.41,
        score_margin=0.57,
        threshold_applied=0.9,
        threshold_version="v1",
        policy_checks={"conservation": True, "currency": True, "uniqueness": True},
        proposed_by=ProposedBy.STAGE1_EXACT,
        verified_by=VerifiedBy.FINANCIAL_AND_EVIDENCE_VERIFIER,
        commit_policy=CommitPolicy.AUTO_COMMIT_STAGE1,
        verification_result=VerificationResult.PASSED,
    )
    assert MatchGroup.model_validate_json(group.model_dump_json()) == group


def test_match_group_member_round_trip() -> None:
    member = MatchGroupMember(
        group_id="grp_001",
        record_id="rec_001",
        source=Source.BANK,
        role=MemberRole.CREDIT,
        signed_amount_paise=100000,
        allocated_amount_paise=100000,
    )
    assert MatchGroupMember.model_validate_json(member.model_dump_json()) == member


def test_decision_event_round_trip() -> None:
    event = DecisionEvent(
        event_id="evt_001",
        group_id="grp_001",
        stage=DecisionStage.STAGE1_EXACT,
        candidate_scores={"rec_001": 0.98},
        reason_code="EXACT_UTR_MATCH",
        explanation="Unique normalized identifier match.",
        timestamp=datetime(2026, 8, 24, 9, 5, 0),
    )
    assert DecisionEvent.model_validate_json(event.model_dump_json()) == event


def test_exception_round_trip() -> None:
    exc = ReconciliationException(
        group_id_or_record_id="rec_002",
        category=ExceptionCategory.ORPHAN_BANK_CREDIT,
        severity=ExceptionSeverity.MEDIUM,
        evidence={"amount_paise": 50000},
        recommended_action="Await matching gateway settlement.",
        review_status=ExceptionReviewStatus.OPEN,
    )
    assert ReconciliationException.model_validate_json(exc.model_dump_json()) == exc
