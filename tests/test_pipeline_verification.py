"""Tests for Stage 6 of the build: wiring the Financial and Evidence
Verifier (verification/verifier.py, built and tested in Stage 5) into
the matching pipeline (matching/pipeline.py's ``run_verification`` /
``run_pipeline``).

Two kinds of coverage, per this stage's task:

  - Unit-level (below): builds MatchGroup/MatchGroupMember/
    NormalizedRecord fixtures directly, the same way test_verifier.py
    does, and calls ``run_verification`` directly to prove the
    orchestration wiring applies the verifier's outcome correctly —
    in particular, that a REJECTED group's records are actually
    released from ``matched_record_ids`` and a PENDING_REVIEW
    (FAILED_MARGIN) group's records are NOT.

  - Integration-level (bottom): extends test_stage1to4_integration.py's
    pattern to run the *full* pipeline (Stage 1-4 + verification)
    against the real calibration dataset, printing a summary of how
    many groups ended VERIFIED / PENDING_REVIEW / REJECTED and how
    many records got released, with the zero-false-positive ground
    truth check now scoped to the VERIFIED subset specifically — a
    PENDING_REVIEW or REJECTED outcome is the system correctly
    declining to commit, not a false positive.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import date
from pathlib import Path

import pytest

from recon_agent.config import Settings
from recon_agent.matching.common import IdAllocator
from recon_agent.matching.pipeline import MatchingResult, run_pipeline, run_verification
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

# ---------------------------------------------------------------------------
# Unit-level: run_verification's application of the verifier's outcome.
# Fixture builders mirror test_verifier.py's own (same shapes, same
# defaults) so the two test files read consistently.
# ---------------------------------------------------------------------------


def _record(
    record_id: str,
    source: Source,
    entity_type: EntityType,
    amount_paise: int,
    currency: str = "INR",
) -> NormalizedRecord:
    return NormalizedRecord(
        record_id=record_id,
        source=source,
        entity_type=entity_type,
        amount_paise=amount_paise,
        currency=currency,
        reference="REF001",
        counterparty="Acme Retail Pvt Ltd",
        occurred_at=date(2026, 1, 1),
        raw_hash=f"hash_{record_id}",
    )


def _member(group_id: str, record: NormalizedRecord, role: MemberRole, signed_amount_paise: int) -> MatchGroupMember:
    return MatchGroupMember(
        group_id=group_id,
        record_id=record.record_id,
        source=record.source,
        role=role,
        signed_amount_paise=signed_amount_paise,
        allocated_amount_paise=abs(signed_amount_paise),
    )


def _group(
    group_id: str,
    proposed_by: ProposedBy,
    *,
    evidence_score: float | None = None,
    runner_up_score: float | None = None,
    score_margin: float | None = None,
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
        threshold_applied=0.70,
        threshold_version="v1",
        policy_checks={},
        proposed_by=proposed_by,
        verified_by=VerifiedBy.NOT_YET_VERIFIED,
        commit_policy=commit_policy,
        verification_result=VerificationResult.NOT_YET_RUN,
    )


def _settings() -> Settings:
    return Settings()


def test_rejected_group_releases_its_records_from_matched_record_ids() -> None:
    # Conservation deliberately doesn't balance (gateway 100_000 vs. bank
    # 90_000, nothing else to make up the difference) -> FAILED_CONSERVATION
    # -> REJECTED per run_verification's policy table.
    gateway = _record("gw_1", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 100_000)
    bank = _record("bank_1", Source.BANK, EntityType.BANK_CREDIT, 90_000)
    records = [gateway, bank]

    group = _group(
        "grp_rej",
        ProposedBy.STAGE2_CONSTRAINED,
        evidence_score=0.90,
        runner_up_score=0.70,
        score_margin=0.20,
    )
    members = [
        _member("grp_rej", gateway, MemberRole.GROSS, 100_000),
        _member("grp_rej", bank, MemberRole.CREDIT, 90_000),
    ]
    proposed = MatchingResult(
        match_groups=[group],
        match_group_members=members,
        matched_record_ids={"gw_1", "bank_1"},
    )

    result = run_verification(records, _settings(), proposed, IdAllocator())

    assert len(result.match_groups) == 1
    verified_group = result.match_groups[0]
    assert verified_group.status == MatchGroupStatus.REJECTED
    assert verified_group.verification_result == VerificationResult.FAILED_CONSERVATION
    assert verified_group.verified_by == VerifiedBy.FINANCIAL_AND_EVIDENCE_VERIFIER

    # The group's records are no longer claimed.
    assert "gw_1" not in result.matched_record_ids
    assert "bank_1" not in result.matched_record_ids
    assert result.released_record_ids == {"gw_1", "bank_1"}

    # An exception was raised for the rejection, and a verification
    # DecisionEvent was emitted.
    assert len(result.exceptions) == 1
    assert result.exceptions[0].group_id_or_record_id == "grp_rej"
    assert result.exceptions[0].review_status.value == "OPEN"
    verification_events = [e for e in result.decision_events if e.stage.value == "STAGE7_VERIFICATION"]
    assert len(verification_events) == 1
    assert verification_events[0].group_id == "grp_rej"


def test_pending_review_margin_failure_does_not_release_its_records() -> None:
    # Same amounts, so conservation/currency both hold, but the margin
    # between the top and runner-up score is razor-thin -> FAILED_MARGIN
    # -> PENDING_REVIEW, not REJECTED, per run_verification's policy table.
    gateway = _record("gw_2", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 100_000)
    bank = _record("bank_2", Source.BANK, EntityType.BANK_CREDIT, 100_000)
    records = [gateway, bank]

    group = _group(
        "grp_pending",
        ProposedBy.STAGE2_CONSTRAINED,
        evidence_score=0.72,
        runner_up_score=0.71,
        score_margin=0.01,
    )
    members = [
        _member("grp_pending", gateway, MemberRole.GROSS, 100_000),
        _member("grp_pending", bank, MemberRole.CREDIT, 100_000),
    ]
    proposed = MatchingResult(
        match_groups=[group],
        match_group_members=members,
        matched_record_ids={"gw_2", "bank_2"},
    )

    result = run_verification(records, _settings(), proposed, IdAllocator())

    assert len(result.match_groups) == 1
    verified_group = result.match_groups[0]
    assert verified_group.status == MatchGroupStatus.PENDING_REVIEW
    assert verified_group.verification_result == VerificationResult.FAILED_MARGIN
    assert verified_group.verified_by == VerifiedBy.FINANCIAL_AND_EVIDENCE_VERIFIER
    assert verified_group.commit_policy == CommitPolicy.HUMAN_REVIEW_REQUIRED

    # Records stay claimed — this is ambiguous, not disproven.
    assert "gw_2" in result.matched_record_ids
    assert "bank_2" in result.matched_record_ids
    assert result.released_record_ids == set()

    # No exception for an ambiguous (not actively wrong) outcome.
    assert result.exceptions == []


def test_passed_group_is_marked_verified_and_keeps_its_commit_policy() -> None:
    gateway = _record("gw_3", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 100_000)
    bank = _record("bank_3", Source.BANK, EntityType.BANK_CREDIT, 100_000)
    records = [gateway, bank]

    group = _group(
        "grp_ok",
        ProposedBy.STAGE2_CONSTRAINED,
        evidence_score=0.90,
        runner_up_score=0.70,
        score_margin=0.20,
        commit_policy=CommitPolicy.AUTO_COMMIT_STAGE2_5_THRESHOLD,
    )
    members = [
        _member("grp_ok", gateway, MemberRole.GROSS, 100_000),
        _member("grp_ok", bank, MemberRole.CREDIT, 100_000),
    ]
    proposed = MatchingResult(
        match_groups=[group],
        match_group_members=members,
        matched_record_ids={"gw_3", "bank_3"},
    )

    result = run_verification(records, _settings(), proposed, IdAllocator())

    verified_group = result.match_groups[0]
    assert verified_group.status == MatchGroupStatus.VERIFIED
    assert verified_group.verification_result == VerificationResult.PASSED
    assert verified_group.verified_by == VerifiedBy.FINANCIAL_AND_EVIDENCE_VERIFIER
    # commit_policy is untouched — still the tier the proposing stage set.
    assert verified_group.commit_policy == CommitPolicy.AUTO_COMMIT_STAGE2_5_THRESHOLD
    assert {"gw_3", "bank_3"} <= result.matched_record_ids
    assert result.exceptions == []


# ---------------------------------------------------------------------------
# Integration-level: full pipeline (Stage 1-4 + verification) against the
# real calibration dataset.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_RECORDS = REPO_ROOT / "data" / "calibration" / "records.json"
CALIBRATION_GROUND_TRUTH = REPO_ROOT / "ground_truth" / "calibration" / "ground_truth.json"
EVALUATION_RECORDS = REPO_ROOT / "data" / "evaluation" / "records.json"
EVALUATION_GROUND_TRUTH = REPO_ROOT / "ground_truth" / "evaluation" / "ground_truth.json"

pytestmark = pytest.mark.skipif(
    not (
        CALIBRATION_RECORDS.exists()
        and CALIBRATION_GROUND_TRUTH.exists()
        and EVALUATION_RECORDS.exists()
        and EVALUATION_GROUND_TRUTH.exists()
    ),
    reason=(
        "calibration/evaluation data or ground truth not found — run "
        "`python -m recon_agent.testdata.generator --seed calibration` and "
        "`--seed evaluation` first"
    ),
)


def _load_records(path: Path) -> list[NormalizedRecord]:
    payload = json.loads(path.read_text())
    return [NormalizedRecord.model_validate(r) for r in payload]


def _load_ground_truth(path: Path) -> dict:
    return json.loads(path.read_text())


def test_verified_groups_are_zero_false_positives_against_calibration() -> None:
    records = _load_records(CALIBRATION_RECORDS)
    settings = Settings()
    result = run_pipeline(records, settings)

    ground_truth = _load_ground_truth(CALIBRATION_GROUND_TRUTH)
    true_clusters = [set(g["record_ids"]) for g in ground_truth["match_groups"]]
    true_clusters += [set(u["record_ids"]) for u in ground_truth["unresolved"]]
    duplicate_ids = {d["record_id"] for d in ground_truth["duplicates"]}
    abstention_ids = {a["record_id"] for a in ground_truth["honest_abstention"]}

    members_by_group: dict[str, list[str]] = {}
    for member in result.match_group_members:
        members_by_group.setdefault(member.group_id, []).append(member.record_id)

    by_status = Counter(g.status.value for g in result.match_groups)
    verified_groups = [g for g in result.match_groups if g.status == MatchGroupStatus.VERIFIED]

    false_positives = []
    for group in verified_groups:
        member_ids = set(members_by_group[group.group_id])

        if member_ids & duplicate_ids:
            false_positives.append((group.group_id, "includes a known duplicate", member_ids))
            continue
        if member_ids & abstention_ids:
            false_positives.append((group.group_id, "includes an honest-abstention decoy", member_ids))
            continue
        if not any(member_ids <= true_cluster for true_cluster in true_clusters):
            false_positives.append((group.group_id, "not a subset of any true cluster", member_ids))

    # The one hard requirement: a VERIFIED group is a claim the system is
    # willing to auto-commit on, so it's the subset that actually has to
    # be zero-false-positive. PENDING_REVIEW / REJECTED outcomes are the
    # system correctly declining to commit, not false positives.
    assert not false_positives, (
        f"Verification marked {len(false_positives)} false-positive group(s) "
        f"VERIFIED: {false_positives[:5]}"
    )

    print(
        f"\n[Stage 6 verification] calibration: {len(result.match_groups)} group(s) "
        f"verified — VERIFIED={by_status.get('VERIFIED', 0)}, "
        f"PENDING_REVIEW={by_status.get('PENDING_REVIEW', 0)}, "
        f"REJECTED={by_status.get('REJECTED', 0)}; "
        f"{len(result.released_record_ids)} record(s) released back to "
        f"unmatched by a rejection; 0 false-positive VERIFIED group(s)."
    )


def test_rejected_groups_records_are_absent_from_matched_record_ids_on_calibration() -> None:
    records = _load_records(CALIBRATION_RECORDS)
    settings = Settings()
    result = run_pipeline(records, settings)

    members_by_group: dict[str, list[str]] = {}
    for member in result.match_group_members:
        members_by_group.setdefault(member.group_id, []).append(member.record_id)

    for group in result.match_groups:
        member_ids = members_by_group.get(group.group_id, [])
        if group.status == MatchGroupStatus.REJECTED:
            for record_id in member_ids:
                assert record_id not in result.matched_record_ids
                assert record_id in result.released_record_ids
        elif group.status in (MatchGroupStatus.VERIFIED, MatchGroupStatus.PENDING_REVIEW):
            for record_id in member_ids:
                assert record_id in result.matched_record_ids


# ---------------------------------------------------------------------------
# Stage 6 bugfix regression coverage (REQUIRED direction, previously
# missing): the suite above only ever checked "zero false VERIFIED".
# A REJECTED group whose membership is EXACTLY a real ground-truth group
# is just as much a bug — a correct match the system wrongly threw away
# — and nothing here used to catch that. Both directions must hold
# simultaneously; see matching.common.recompute_group_conservation's
# docstring and README.md's Stage 6 note for what was wrong and why.
# ---------------------------------------------------------------------------


def _assert_zero_false_rejections(records_path: Path, ground_truth_path: Path, label: str) -> None:
    records = _load_records(records_path)
    settings = Settings()
    result = run_pipeline(records, settings)

    ground_truth = _load_ground_truth(ground_truth_path)
    true_clusters = [set(g["record_ids"]) for g in ground_truth["match_groups"]]

    members_by_group: dict[str, list[str]] = {}
    for member in result.match_group_members:
        members_by_group.setdefault(member.group_id, []).append(member.record_id)

    by_status = Counter(g.status.value for g in result.match_groups)
    rejected_groups = [g for g in result.match_groups if g.status == MatchGroupStatus.REJECTED]

    false_rejections = []
    for group in rejected_groups:
        member_ids = set(members_by_group[group.group_id])
        if member_ids in true_clusters:
            false_rejections.append((group.group_id, group.verification_result.value, member_ids))

    assert not false_rejections, (
        f"Verification wrongly REJECTED {len(false_rejections)} group(s) whose "
        f"membership exactly matches a real ground-truth group on {label}: "
        f"{false_rejections[:5]}"
    )

    print(
        f"\n[Stage 6 verification] {label}: {len(result.match_groups)} group(s) — "
        f"VERIFIED={by_status.get('VERIFIED', 0)}, "
        f"PENDING_REVIEW={by_status.get('PENDING_REVIEW', 0)}, "
        f"REJECTED={by_status.get('REJECTED', 0)}; "
        f"{len(rejected_groups)} rejected, 0 of them false rejections "
        f"against ground truth."
    )


def test_no_rejected_group_exactly_matches_a_ground_truth_group_on_calibration() -> None:
    _assert_zero_false_rejections(CALIBRATION_RECORDS, CALIBRATION_GROUND_TRUTH, "calibration")


def test_no_rejected_group_exactly_matches_a_ground_truth_group_on_evaluation() -> None:
    _assert_zero_false_rejections(EVALUATION_RECORDS, EVALUATION_GROUND_TRUTH, "evaluation")
