"""Unit tests for Stage 12: the evaluation harness (evaluation/harness.py).

Two kinds of coverage, per this stage's brief:

  - Unit-level (below): small, hand-constructed fixtures with known
    correct answers for each metric formula, proving each computation
    in isolation rather than only running the full pipeline and
    eyeballing the result.

  - Integration-level (bottom): runs the real harness against the real
    calibration and evaluation datasets end-to-end (records -> real
    pipeline -> real ground truth -> a full EvaluationReport), the same
    "skip if data/ground-truth missing" pattern used by
    test_stage6_pipeline_integration.py, printing the real numbers this
    system currently achieves.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from recon_agent.evaluation.harness import (
    AbstentionCase,
    GroundTruth,
    UnresolvedCase,
    compute_auto_match_precision,
    compute_bank_credit_coverage,
    compute_complete_cluster_resolution,
    compute_exception_quality,
    compute_false_match_rate,
    compute_record_coverage,
    compute_runtime_and_cost,
    compute_tier_contribution,
    compute_value_coverage,
    is_verified_group_correct,
    load_ground_truth,
    load_records,
    run_evaluation,
)
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

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Fixture builders — mirror test_pipeline_verification.py's own shapes.
# ---------------------------------------------------------------------------


def _record(
    record_id: str,
    amount_paise: int = 1000,
    source: Source = Source.LEDGER,
    entity_type: EntityType = EntityType.PAYMENT,
) -> NormalizedRecord:
    return NormalizedRecord(
        record_id=record_id,
        source=source,
        entity_type=entity_type,
        amount_paise=amount_paise,
        currency="INR",
        reference="REF001",
        counterparty="Acme Retail Pvt Ltd",
        occurred_at=date(2026, 1, 1),
        raw_hash=f"hash_{record_id}",
    )


def _member(group_id: str, record_id: str, source: Source = Source.LEDGER) -> MatchGroupMember:
    return MatchGroupMember(
        group_id=group_id,
        record_id=record_id,
        source=source,
        role=MemberRole.GROSS,
        signed_amount_paise=1000,
        allocated_amount_paise=1000,
    )


def _group(
    group_id: str,
    status: MatchGroupStatus,
    proposed_by: ProposedBy = ProposedBy.STAGE1_EXACT,
) -> MatchGroup:
    return MatchGroup(
        group_id=group_id,
        cardinality=Cardinality.ONE_TO_ONE,
        expected_amount_paise=1000,
        matched_amount_paise=1000,
        residual_amount_paise=0,
        status=status,
        threshold_applied=1.0,
        threshold_version="v1",
        policy_checks={},
        proposed_by=proposed_by,
        verified_by=VerifiedBy.FINANCIAL_AND_EVIDENCE_VERIFIER,
        commit_policy=CommitPolicy.AUTO_COMMIT_STAGE1,
        verification_result=VerificationResult.PASSED,
    )


def _exception(
    group_id_or_record_id: str,
    category: ExceptionCategory,
    evidence: dict | None = None,
) -> ReconciliationException:
    return ReconciliationException(
        group_id_or_record_id=group_id_or_record_id,
        category=category,
        severity=ExceptionSeverity.MEDIUM,
        evidence=evidence or {},
        recommended_action="test",
        review_status=ExceptionReviewStatus.OPEN,
    )


def _decision_event(
    group_id: str,
    candidate_scores: dict,
    reason_code: str,
    stage: DecisionStage = DecisionStage.STAGE6_LLM,
) -> DecisionEvent:
    return DecisionEvent(
        event_id=f"evt_{group_id}_{reason_code}",
        group_id=group_id,
        stage=stage,
        candidate_scores=candidate_scores,
        reason_code=reason_code,
        explanation="test",
        timestamp="2026-01-01T00:00:00Z",
    )


def _gt(
    match_group_clusters: tuple[frozenset[str], ...] = (),
    unresolved_cases: tuple[UnresolvedCase, ...] = (),
    abstention_cases: tuple[AbstentionCase, ...] = (),
    duplicate_ids: frozenset[str] = frozenset(),
) -> GroundTruth:
    true_clusters = match_group_clusters + tuple(u.record_ids for u in unresolved_cases)
    return GroundTruth(
        dataset="test",
        match_group_clusters=match_group_clusters,
        unresolved_cases=unresolved_cases,
        abstention_cases=abstention_cases,
        duplicate_ids=duplicate_ids,
        true_clusters=true_clusters,
        abstention_ids=frozenset(a.record_id for a in abstention_cases),
    )


# ---------------------------------------------------------------------------
# is_verified_group_correct — the shared correctness check.
# ---------------------------------------------------------------------------


def test_correct_when_subset_of_match_group_cluster():
    gt = _gt(match_group_clusters=(frozenset({"r1", "r2", "r3"}),))
    ok, reason = is_verified_group_correct(frozenset({"r1", "r2"}), gt)
    assert ok is True
    assert reason == "correct"


def test_correct_when_subset_of_unresolved_cluster():
    gt = _gt(
        unresolved_cases=(UnresolvedCase(record_ids=frozenset({"r1", "r2"}), reason="MISSING_SETTLEMENT", categories=()),)
    )
    ok, _ = is_verified_group_correct(frozenset({"r1", "r2"}), gt)
    assert ok is True


def test_incorrect_when_touches_duplicate():
    gt = _gt(match_group_clusters=(frozenset({"r1", "r2"}),), duplicate_ids=frozenset({"r2"}))
    ok, reason = is_verified_group_correct(frozenset({"r1", "r2"}), gt)
    assert ok is False
    assert reason == "includes_known_duplicate"


def test_incorrect_when_touches_abstention_decoy():
    gt = _gt(
        match_group_clusters=(frozenset({"r1", "r2"}),),
        abstention_cases=(AbstentionCase(record_id="r2", reason="NO_COUNTERPART_EXISTS", categories=()),),
    )
    # Even though {r1, r2} IS a subset of a real cluster, touching an
    # abstention id must still fail — abstention exclusion is checked
    # before the subset check.
    ok, reason = is_verified_group_correct(frozenset({"r1", "r2"}), gt)
    assert ok is False
    assert reason == "includes_honest_abstention_decoy"


def test_incorrect_when_not_subset_of_any_true_cluster():
    gt = _gt(match_group_clusters=(frozenset({"r1", "r2"}),))
    ok, reason = is_verified_group_correct(frozenset({"r1", "r99"}), gt)
    assert ok is False
    assert reason == "not_subset_of_true_cluster"


# ---------------------------------------------------------------------------
# Metric 1 — auto-match precision.
# ---------------------------------------------------------------------------


def test_auto_match_precision_basic():
    # 2 VERIFIED groups: one correct, one wrong (touches a duplicate).
    # 1 PENDING_REVIEW group, which must NOT count (auto-match = VERIFIED only).
    g1 = _group("g1", MatchGroupStatus.VERIFIED)
    g2 = _group("g2", MatchGroupStatus.VERIFIED)
    g3 = _group("g3", MatchGroupStatus.PENDING_REVIEW)
    members_by_group = {
        "g1": [_member("g1", "r1")],
        "g2": [_member("g2", "r2")],
        "g3": [_member("g3", "r3")],
    }
    gt = _gt(match_group_clusters=(frozenset({"r1"}),), duplicate_ids=frozenset({"r2"}))
    result = compute_auto_match_precision([g1, g2, g3], members_by_group, gt)
    assert result.total == 2  # only VERIFIED groups counted
    assert result.correct == 1
    assert result.precision == pytest.approx(0.5)
    assert result.incorrect_group_ids == ("g2",)


def test_auto_match_precision_none_when_no_verified_groups():
    result = compute_auto_match_precision([], {}, _gt())
    assert result.total == 0
    assert result.precision is None


def test_stage6_group_never_counts_toward_auto_match_precision():
    # A STAGE6_LLM proposal should never be VERIFIED in real pipeline
    # output (§2), but even if one somehow were, this metric's scope
    # (VERIFIED status) already covers it correctly without a special
    # case — proven here directly.
    g1 = _group("g1", MatchGroupStatus.VERIFIED, proposed_by=ProposedBy.STAGE6_LLM)
    members_by_group = {"g1": [_member("g1", "r1")]}
    gt = _gt(match_group_clusters=(frozenset({"r1"}),))
    result = compute_auto_match_precision([g1], members_by_group, gt)
    assert result.total == 1
    assert result.correct == 1


# ---------------------------------------------------------------------------
# Metric 2 — record coverage.
# ---------------------------------------------------------------------------


def test_record_coverage_raw_and_excluding_abstention():
    records = [_record(f"r{i}") for i in range(1, 6)]  # r1..r5
    g1 = _group("g1", MatchGroupStatus.VERIFIED)
    members_by_group = {"g1": [_member("g1", "r1"), _member("g1", "r2")]}
    gt = _gt(
        match_group_clusters=(frozenset({"r1", "r2"}),),
        abstention_cases=(
            AbstentionCase(record_id="r5", reason="NO_COUNTERPART_EXISTS", categories=()),
        ),
    )
    result = compute_record_coverage(records, [g1], members_by_group, gt)
    assert result.matched_record_count == 2
    assert result.total_records == 5
    assert result.abstention_count == 1
    assert result.raw_coverage == pytest.approx(2 / 5)
    # ceiling excludes the one honest-abstention record from the denominator
    assert result.coverage_excluding_abstention == pytest.approx(2 / 4)


def test_record_coverage_zero_records_is_zero_not_a_crash():
    result = compute_record_coverage([], [], {}, _gt())
    assert result.raw_coverage == 0.0
    assert result.coverage_excluding_abstention is None


# ---------------------------------------------------------------------------
# Metric 3 — value coverage (abs() for negative refund/chargeback amounts).
# ---------------------------------------------------------------------------


def test_value_coverage_uses_absolute_value_for_negative_amounts():
    # r1: +1000 (a gross payment), r2: -400 (a refund, stored negative
    # per testdata/generator.py) — matched. r3: +200, unmatched.
    records = [
        _record("r1", amount_paise=1000),
        _record("r2", amount_paise=-400, entity_type=EntityType.REFUND),
        _record("r3", amount_paise=200),
    ]
    g1 = _group("g1", MatchGroupStatus.VERIFIED)
    members_by_group = {"g1": [_member("g1", "r1"), _member("g1", "r2")]}
    gt = _gt(match_group_clusters=(frozenset({"r1", "r2"}),))
    result = compute_value_coverage(records, [g1], members_by_group, gt)
    # If netting had happened instead of abs(), matched value would be
    # 1000 + (-400) = 600, not 1400 — this proves abs() is actually applied.
    assert result.matched_value_paise == 1000 + 400
    assert result.eligible_value_paise == 1000 + 400 + 200
    assert result.value_coverage == pytest.approx(1400 / 1600)


def test_value_coverage_excluding_abstention():
    records = [
        _record("r1", amount_paise=1000),
        _record("r2", amount_paise=500),  # honest abstention, never matched
    ]
    gt = _gt(abstention_cases=(AbstentionCase(record_id="r2", reason="NO_COUNTERPART_EXISTS", categories=()),))
    result = compute_value_coverage(records, [], {}, gt)
    assert result.eligible_value_paise == 1500
    assert result.eligible_value_paise_excluding_abstention == 1000
    assert result.value_coverage == 0.0
    assert result.value_coverage_excluding_abstention == 0.0


# ---------------------------------------------------------------------------
# Metric 4 — false-match rate.
# ---------------------------------------------------------------------------


def test_false_match_rate_basic():
    g1 = _group("g1", MatchGroupStatus.VERIFIED)
    g2 = _group("g2", MatchGroupStatus.VERIFIED)
    g3 = _group("g3", MatchGroupStatus.VERIFIED)
    members_by_group = {
        "g1": [_member("g1", "r1")],
        "g2": [_member("g2", "r2")],  # duplicate -> wrong
        "g3": [_member("g3", "r3")],  # not a subset of anything -> wrong
    }
    gt = _gt(match_group_clusters=(frozenset({"r1"}),), duplicate_ids=frozenset({"r2"}))
    result = compute_false_match_rate([g1, g2, g3], members_by_group, gt)
    assert result.total == 3
    assert result.incorrect == 2
    assert result.false_match_rate == pytest.approx(2 / 3)
    assert set(result.incorrect_group_ids) == {"g2", "g3"}


def test_false_match_rate_is_precision_complement():
    g1 = _group("g1", MatchGroupStatus.VERIFIED)
    g2 = _group("g2", MatchGroupStatus.VERIFIED)
    members_by_group = {"g1": [_member("g1", "r1")], "g2": [_member("g2", "r2")]}
    gt = _gt(match_group_clusters=(frozenset({"r1"}),), duplicate_ids=frozenset({"r2"}))
    precision = compute_auto_match_precision([g1, g2], members_by_group, gt)
    false_match = compute_false_match_rate([g1, g2], members_by_group, gt)
    assert precision.precision + false_match.false_match_rate == pytest.approx(1.0)


def test_false_match_rate_none_when_no_committed_matches():
    result = compute_false_match_rate([], {}, _gt())
    assert result.total == 0
    assert result.false_match_rate is None


# ---------------------------------------------------------------------------
# Metric 5 — exception quality.
# ---------------------------------------------------------------------------


def test_exception_quality_plausible_when_matching_category_present():
    gt = _gt(
        unresolved_cases=(UnresolvedCase(record_ids=frozenset({"r1"}), reason="MISSING_SETTLEMENT", categories=()),),
        abstention_cases=(AbstentionCase(record_id="r2", reason="NO_COUNTERPART_EXISTS", categories=()),),
    )
    exceptions = [
        _exception("r1", ExceptionCategory.PARTIAL_SETTLEMENT),
        _exception("r2", ExceptionCategory.INSUFFICIENT_EVIDENCE),
    ]
    result = compute_exception_quality([], {}, exceptions, gt)
    assert result.total == 2
    assert result.plausible == 2
    assert result.exception_quality == pytest.approx(1.0)
    assert result.implausible_cases == ()


def test_exception_quality_implausible_when_silent():
    gt = _gt(
        abstention_cases=(AbstentionCase(record_id="r2", reason="NO_COUNTERPART_EXISTS", categories=()),),
    )
    result = compute_exception_quality([], {}, [], gt)  # no exceptions at all
    assert result.total == 1
    assert result.plausible == 0
    assert result.exception_quality == pytest.approx(0.0)
    assert len(result.implausible_cases) == 1
    assert result.implausible_cases[0]["record_ids"] == ["r2"]


def test_exception_quality_excludes_unresolved_case_legitimately_verified():
    # r1/r2 ground truth calls "unresolved", but they ended up correctly
    # VERIFIED (a real, legitimate partial match) — this case must be
    # excluded from the denominator entirely, not scored as implausible
    # silence, since it's already credited by auto-match precision.
    g1 = _group("g1", MatchGroupStatus.VERIFIED)
    members_by_group = {"g1": [_member("g1", "r1"), _member("g1", "r2")]}
    gt = _gt(
        unresolved_cases=(
            UnresolvedCase(record_ids=frozenset({"r1", "r2"}), reason="MISSING_SETTLEMENT", categories=()),
        ),
    )
    result = compute_exception_quality([g1], members_by_group, [], gt)
    assert result.total == 0
    assert result.resolved_as_verified_match == 1
    assert result.exception_quality is None


def test_exception_quality_does_not_exclude_abstention_case_even_if_verified():
    # An abstention record that ends up VERIFIED is always a genuine
    # false positive (is_verified_group_correct never scores it as
    # correct) — unlike the unresolved-case exclusion above, this must
    # NOT be excluded from the denominator; it should show implausible
    # (no exception, because it was wrongly absorbed instead).
    g1 = _group("g1", MatchGroupStatus.VERIFIED)
    members_by_group = {"g1": [_member("g1", "r1")]}
    gt = _gt(abstention_cases=(AbstentionCase(record_id="r1", reason="NO_COUNTERPART_EXISTS", categories=()),))
    result = compute_exception_quality([g1], members_by_group, [], gt)
    assert result.total == 1
    assert result.plausible == 0
    assert result.resolved_as_verified_match == 0


def test_exception_quality_none_when_no_cases():
    result = compute_exception_quality([], {}, [], _gt())
    assert result.total == 0
    assert result.exception_quality is None


# ---------------------------------------------------------------------------
# Metric 6 — tier contribution.
# ---------------------------------------------------------------------------


def test_tier_contribution_breaks_down_by_stage():
    g1 = _group("g1", MatchGroupStatus.VERIFIED, proposed_by=ProposedBy.STAGE1_EXACT)
    g2 = _group("g2", MatchGroupStatus.VERIFIED, proposed_by=ProposedBy.STAGE3_AGGREGATE)
    g3 = _group("g3", MatchGroupStatus.PENDING_REVIEW, proposed_by=ProposedBy.STAGE6_LLM)
    members_by_group = {
        "g1": [_member("g1", "r1")],
        "g2": [_member("g2", "r2"), _member("g2", "r3")],
        "g3": [_member("g3", "r4")],
    }
    group_id_to_stage = {"g1": "STAGE1_EXACT", "g2": "STAGE3_AGGREGATE", "g3": "STAGE6_LLM"}
    gt = _gt(match_group_clusters=(frozenset({"r1"}), frozenset({"r2", "r3"})))

    stats = compute_tier_contribution([g1, g2, g3], members_by_group, group_id_to_stage, gt, total_records=10)
    by_stage = {s.stage: s for s in stats}

    assert by_stage["STAGE1_EXACT"].proposed == 1
    assert by_stage["STAGE1_EXACT"].verified == 1
    assert by_stage["STAGE1_EXACT"].precision == pytest.approx(1.0)
    assert by_stage["STAGE1_EXACT"].matched_record_count == 1
    assert by_stage["STAGE1_EXACT"].coverage_contribution == pytest.approx(0.1)

    assert by_stage["STAGE3_AGGREGATE"].matched_record_count == 2
    assert by_stage["STAGE3_AGGREGATE"].coverage_contribution == pytest.approx(0.2)

    # STAGE6_LLM: proposed but never VERIFIED -> precision is None, not 0/0.
    assert by_stage["STAGE6_LLM"].proposed == 1
    assert by_stage["STAGE6_LLM"].verified == 0
    assert by_stage["STAGE6_LLM"].precision is None
    assert by_stage["STAGE6_LLM"].coverage_contribution == 0.0

    # Every ProposedBy value gets an entry, even ones with zero proposals.
    assert by_stage["STAGE2_CONSTRAINED"].proposed == 0
    assert by_stage["STAGE2_CONSTRAINED"].precision is None


# ---------------------------------------------------------------------------
# Metric 7 — runtime and cost.
# ---------------------------------------------------------------------------


def test_runtime_and_cost_zero_calls_zero_cost():
    result = compute_runtime_and_cost(
        wall_clock_seconds=2.0,
        total_records=200,
        decision_events=[],
        records_by_id={},
        groq_calls_made=0,
    )
    assert result.wall_clock_seconds == 2.0
    assert result.records_per_second == pytest.approx(100.0)
    assert result.groq_calls_made == 0
    assert result.estimated_cost_usd == 0.0
    assert result.estimated_total_input_tokens == 0.0


def test_runtime_and_cost_reconstructs_prompt_size_for_real_calls():
    seeker = _record("seeker1", source=Source.LEDGER, entity_type=EntityType.PAYMENT)
    candidate = _record("cand1", source=Source.GATEWAY, entity_type=EntityType.GATEWAY_SETTLEMENT)
    records_by_id = {"seeker1": seeker, "cand1": candidate}
    decision_events = [
        _decision_event(
            "unmatched:seeker1",
            {"seeker_record_id": "seeker1", "offered_candidate_ids": ["cand1"]},
            reason_code="SOME_LLM_REASON_CODE",  # a real call happened
        ),
        _decision_event(
            "unmatched:seeker2",
            {"seeker_record_id": "seeker2", "candidate_pool_size": 0},
            reason_code="NO_CANDIDATES_AVAILABLE",  # no call made — must be excluded
        ),
    ]
    result = compute_runtime_and_cost(
        wall_clock_seconds=1.0,
        total_records=2,
        decision_events=decision_events,
        records_by_id=records_by_id,
        groq_calls_made=1,
    )
    assert result.reconstructed_call_events == 1  # the NO_CANDIDATES_AVAILABLE one is excluded
    assert result.estimated_avg_input_tokens_per_call > 0
    assert result.estimated_total_output_tokens == 120.0  # ASSUMED_OUTPUT_TOKENS_PER_CALL * 1 call
    assert result.estimated_cost_usd > 0


def test_runtime_and_cost_uses_governor_ledger_as_authoritative_call_count():
    # Even if only 1 decision event is reconstructable, the estimate
    # scales to the governor's real call count (5), per this metric's
    # documented "governor ledger is authoritative" approach.
    seeker = _record("seeker1")
    candidate = _record("cand1")
    records_by_id = {"seeker1": seeker, "cand1": candidate}
    decision_events = [
        _decision_event(
            "g1",
            {"seeker_record_id": "seeker1", "offered_candidate_ids": ["cand1"]},
            reason_code="HEURISTIC_MOCK_STRONG_MATCH",
        ),
    ]
    result = compute_runtime_and_cost(
        wall_clock_seconds=1.0,
        total_records=10,
        decision_events=decision_events,
        records_by_id=records_by_id,
        groq_calls_made=5,
    )
    assert result.groq_calls_made == 5
    assert result.estimated_total_output_tokens == 120.0 * 5


# ---------------------------------------------------------------------------
# Metric 8 — bank credit coverage.
# ---------------------------------------------------------------------------


def test_bank_credit_coverage_basic():
    # r1 (BANK, BANK_CREDIT) + r2 (LEDGER) form a real match_group
    # cluster and r1 ends up in a VERIFIED group -> counts as covered.
    # r3 (BANK, BANK_CREDIT) + r4 (LEDGER) form a second real cluster
    # but r3 is NOT in any VERIFIED group -> counts as uncovered.
    # r5 (BANK, BANK_CREDIT) is only part of an `unresolved` cluster,
    # not a real match_group -> excluded from the denominator entirely.
    records = [
        _record("r1", source=Source.BANK, entity_type=EntityType.BANK_CREDIT),
        _record("r2", source=Source.LEDGER),
        _record("r3", source=Source.BANK, entity_type=EntityType.BANK_CREDIT),
        _record("r4", source=Source.LEDGER),
        _record("r5", source=Source.BANK, entity_type=EntityType.BANK_CREDIT),
    ]
    g1 = _group("g1", MatchGroupStatus.VERIFIED)
    members_by_group = {"g1": [_member("g1", "r1", source=Source.BANK), _member("g1", "r2")]}
    gt = _gt(
        match_group_clusters=(frozenset({"r1", "r2"}), frozenset({"r3", "r4"})),
        unresolved_cases=(UnresolvedCase(record_ids=frozenset({"r5"}), reason="MISSING_SETTLEMENT", categories=()),),
    )
    result = compute_bank_credit_coverage(records, [g1], members_by_group, gt)
    assert result.total == 2  # r1, r3 — r5 excluded (unresolved, not a match_group)
    assert result.matched == 1  # only r1
    assert result.bank_credit_coverage == pytest.approx(0.5)
    assert result.unmatched_record_ids == ("r3",)


def test_bank_credit_coverage_excludes_bank_side_refund_chargeback_reversal():
    # Bugfix regression: a BANK-source REFUND leg (the bank-side
    # reflection of a refund event, per testdata/generator.py's "one
    # real money movement stored identically on every source leg"
    # design note) is a real BANK-source record, but it is not a
    # credit landing in the account, so it must never inflate this
    # metric's denominator even though it sits in a real ground-truth
    # match_group cluster. r1 (BANK, BANK_CREDIT) is the only genuinely
    # eligible record here.
    records = [
        _record("r1", source=Source.BANK, entity_type=EntityType.BANK_CREDIT),
        _record("r2", source=Source.LEDGER),
        _record(
            "r3",
            source=Source.BANK,
            entity_type=EntityType.REFUND,
            amount_paise=-500,
        ),
        _record("r4", source=Source.LEDGER, entity_type=EntityType.REFUND, amount_paise=-500),
    ]
    g1 = _group("g1", MatchGroupStatus.VERIFIED)
    members_by_group = {"g1": [_member("g1", "r1", source=Source.BANK), _member("g1", "r2")]}
    gt = _gt(
        match_group_clusters=(
            frozenset({"r1", "r2"}),
            frozenset({"r3", "r4"}),
        ),
    )
    result = compute_bank_credit_coverage(records, [g1], members_by_group, gt)
    assert result.total == 1  # r3 excluded — BANK-source but not BANK_CREDIT
    assert result.matched == 1
    assert result.bank_credit_coverage == pytest.approx(1.0)
    assert "r3" not in result.unmatched_record_ids


def test_bank_credit_coverage_none_when_no_eligible_bank_records():
    records = [_record("r1", source=Source.LEDGER), _record("r2", source=Source.GATEWAY)]
    gt = _gt(match_group_clusters=(frozenset({"r1", "r2"}),))
    result = compute_bank_credit_coverage(records, [], {}, gt)
    assert result.total == 0
    assert result.bank_credit_coverage is None


def test_bank_credit_coverage_counts_group_regardless_of_correctness():
    # r1 (BANK, BANK_CREDIT) lands in a VERIFIED group alongside a
    # duplicate id — is_verified_group_correct would call this group
    # WRONG, but bank credit coverage only asks "did the credit end up
    # in SOME VERIFIED group", so it still counts r1 as covered.
    records = [
        _record("r1", source=Source.BANK, entity_type=EntityType.BANK_CREDIT),
        _record("r2", source=Source.LEDGER),
    ]
    g1 = _group("g1", MatchGroupStatus.VERIFIED)
    members_by_group = {"g1": [_member("g1", "r1", source=Source.BANK), _member("g1", "rDUP")]}
    gt = _gt(match_group_clusters=(frozenset({"r1", "r2"}),), duplicate_ids=frozenset({"rDUP"}))
    result = compute_bank_credit_coverage(records, [g1], members_by_group, gt)
    assert result.total == 1
    assert result.matched == 1
    assert result.bank_credit_coverage == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Metric 9 — complete cluster resolution.
# ---------------------------------------------------------------------------


def test_complete_cluster_resolution_basic():
    # Cluster {r1, r2} is closed EXACTLY by a VERIFIED group with the
    # same membership. Cluster {r3, r4, r5} is only PARTIALLY closed —
    # the VERIFIED group covers {r3, r4} but not r5 — so it must NOT
    # count, even though {r3, r4} is a correct subset (would count
    # toward auto-match precision, but not toward this metric).
    g1 = _group("g1", MatchGroupStatus.VERIFIED)
    g2 = _group("g2", MatchGroupStatus.VERIFIED)
    members_by_group = {
        "g1": [_member("g1", "r1"), _member("g1", "r2")],
        "g2": [_member("g2", "r3"), _member("g2", "r4")],
    }
    gt = _gt(match_group_clusters=(frozenset({"r1", "r2"}), frozenset({"r3", "r4", "r5"})))
    result = compute_complete_cluster_resolution([g1, g2], members_by_group, gt)
    assert result.total == 2
    assert result.exact == 1
    assert result.complete_cluster_resolution == pytest.approx(0.5)
    assert result.unresolved_cluster_record_ids == (("r3", "r4", "r5"),)


def test_complete_cluster_resolution_none_when_no_match_group_clusters():
    # Only an `unresolved` cluster exists — no real match_groups at
    # all — so this metric has nothing to score.
    gt = _gt(
        unresolved_cases=(UnresolvedCase(record_ids=frozenset({"r1"}), reason="MISSING_SETTLEMENT", categories=()),)
    )
    result = compute_complete_cluster_resolution([], {}, gt)
    assert result.total == 0
    assert result.complete_cluster_resolution is None


def test_complete_cluster_resolution_rejects_group_that_overshoots_cluster():
    # A VERIFIED group with EXTRA members beyond the real cluster is
    # neither a correct nor an exact match — must not count.
    g1 = _group("g1", MatchGroupStatus.VERIFIED)
    members_by_group = {"g1": [_member("g1", "r1"), _member("g1", "r2"), _member("g1", "rEXTRA")]}
    gt = _gt(match_group_clusters=(frozenset({"r1", "r2"}),))
    result = compute_complete_cluster_resolution([g1], members_by_group, gt)
    assert result.total == 1
    assert result.exact == 0
    assert result.unresolved_cluster_record_ids == (("r1", "r2"),)


# ---------------------------------------------------------------------------
# Ground truth loading — smoke test against the real fixture shape.
# ---------------------------------------------------------------------------


def test_load_ground_truth_parses_real_shape(tmp_path: Path):
    payload = {
        "dataset": "test",
        "seed": 1,
        "num_logical_events": 2,
        "num_physical_records": 4,
        "match_groups": [{"group_id": "g1", "record_ids": ["r1", "r2"], "categories": ["clean"]}],
        "duplicates": [{"record_id": "r4", "duplicate_of": "r3", "categories": ["structural_duplicate"]}],
        "unresolved": [{"record_ids": ["r3"], "reason": "MISSING_SETTLEMENT", "categories": ["clean"]}],
        "honest_abstention": [{"record_id": "r5", "reason": "NO_COUNTERPART_EXISTS", "categories": ["honest_abstention"]}],
        "category_counts": {"clean": 2},
    }
    path = tmp_path / "ground_truth.json"
    path.write_text(json.dumps(payload))
    gt = load_ground_truth(path)
    assert gt.match_group_clusters == (frozenset({"r1", "r2"}),)
    assert gt.duplicate_ids == frozenset({"r4"})
    assert gt.abstention_ids == frozenset({"r5"})
    assert frozenset({"r3"}) in gt.true_clusters
    assert gt.num_logical_events == 2


# ---------------------------------------------------------------------------
# Integration — run the real harness against the real datasets.
# ---------------------------------------------------------------------------

CALIBRATION_RECORDS = REPO_ROOT / "data" / "calibration" / "records.json"
CALIBRATION_GROUND_TRUTH = REPO_ROOT / "ground_truth" / "calibration" / "ground_truth.json"
EVALUATION_RECORDS = REPO_ROOT / "data" / "evaluation" / "records.json"
EVALUATION_GROUND_TRUTH = REPO_ROOT / "ground_truth" / "evaluation" / "ground_truth.json"

pytestmark_integration = pytest.mark.skipif(
    not (
        CALIBRATION_RECORDS.exists()
        and CALIBRATION_GROUND_TRUTH.exists()
        and EVALUATION_RECORDS.exists()
        and EVALUATION_GROUND_TRUTH.exists()
    ),
    reason="calibration/evaluation data or ground truth not found",
)


@pytestmark_integration
@pytest.mark.parametrize("dataset", ["calibration", "evaluation"])
def test_harness_runs_end_to_end_against_real_dataset(dataset: str, tmp_path: Path):
    report = run_evaluation(
        dataset,
        data_root=REPO_ROOT,
        governor_db_path=str(tmp_path / f"{dataset}-governor.db"),
        run_id=f"unit-test-{dataset}",
    )

    assert report.dataset == dataset
    assert report.total_records > 0
    # Auto-match precision must be a valid probability (or None if no
    # VERIFIED groups at all, which would itself be a red flag worth
    # seeing fail loudly rather than silently).
    assert report.auto_match_precision.total > 0
    assert 0.0 <= report.auto_match_precision.precision <= 1.0
    assert 0.0 <= report.record_coverage.raw_coverage <= 1.0
    assert 0.0 <= report.value_coverage.value_coverage <= 1.0
    assert 0.0 <= report.false_match_rate.false_match_rate <= 1.0
    assert len(report.tier_contribution) == len(ProposedBy)
    assert report.runtime_and_cost.wall_clock_seconds > 0
    assert report.bank_credit_coverage.total > 0
    assert 0.0 <= report.bank_credit_coverage.bank_credit_coverage <= 1.0
    assert report.complete_cluster_resolution.total > 0
    assert 0.0 <= report.complete_cluster_resolution.complete_cluster_resolution <= 1.0

    # Round-trips through JSON export cleanly.
    exported = json.dumps(report.to_dict())
    assert json.loads(exported)["dataset"] == dataset

    print(f"\n[Stage 12] {dataset}:\n{report.render_human_readable()}")
