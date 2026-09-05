"""Integration test — Stage 1 + Stage 2 against the real calibration
dataset, checked against its hidden ground truth.

Ground truth (ground_truth/calibration/ground_truth.json) is read ONLY
inside this test file, never inside the matching code itself — the
matching pipeline must never have access to it (see
testdata/generator.py's own docstring and README.md's Stage 2 section).

Per the task: this is a sanity check, not the evaluation harness. The
one hard requirement is zero false proposals — every group Stage 1/2
proposed must correspond to a real ground-truth structure. Full
precision/coverage/false-match-rate metrics are a later stage's job;
this test just prints a short summary.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from recon_agent.config import Settings
from recon_agent.matching import run_stage1_and_stage2
from recon_agent.models import NormalizedRecord

REPO_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_RECORDS = REPO_ROOT / "data" / "calibration" / "records.json"
CALIBRATION_GROUND_TRUTH = REPO_ROOT / "ground_truth" / "calibration" / "ground_truth.json"

pytestmark = pytest.mark.skipif(
    not (CALIBRATION_RECORDS.exists() and CALIBRATION_GROUND_TRUTH.exists()),
    reason=(
        "calibration data/ground truth not found — run "
        "`python -m recon_agent.testdata.generator --seed calibration` first"
    ),
)


def _load_records() -> list[NormalizedRecord]:
    payload = json.loads(CALIBRATION_RECORDS.read_text())
    return [NormalizedRecord.model_validate(r) for r in payload]


def _load_ground_truth() -> dict:
    return json.loads(CALIBRATION_GROUND_TRUTH.read_text())


def test_stage1_and_stage2_propose_zero_false_matches_against_calibration() -> None:
    records = _load_records()
    settings = Settings()
    result = run_stage1_and_stage2(records, settings)

    ground_truth = _load_ground_truth()
    # A proposal is "true" if its member set is fully contained in some
    # real ground-truth structure — either a genuine match_group, or an
    # "unresolved" pending-counterpart pair (Stage 1/2 correctly finding
    # a partial-but-real subset, e.g. a ledger+gateway pair still missing
    # its bank leg, is not a false proposal — it's just incomplete).
    true_clusters = [set(g["record_ids"]) for g in ground_truth["match_groups"]]
    true_clusters += [set(u["record_ids"]) for u in ground_truth["unresolved"]]
    duplicate_ids = {d["record_id"] for d in ground_truth["duplicates"]}
    abstention_ids = {a["record_id"] for a in ground_truth["honest_abstention"]}

    members_by_group: dict[str, list[str]] = {}
    for member in result.match_group_members:
        members_by_group.setdefault(member.group_id, []).append(member.record_id)

    false_proposals = []
    for group in result.match_groups:
        member_ids = set(members_by_group[group.group_id])

        if member_ids & duplicate_ids:
            false_proposals.append((group.group_id, "includes a known duplicate", member_ids))
            continue
        if member_ids & abstention_ids:
            false_proposals.append((group.group_id, "includes an honest-abstention decoy", member_ids))
            continue
        if not any(member_ids <= true_cluster for true_cluster in true_clusters):
            false_proposals.append((group.group_id, "not a subset of any true cluster", member_ids))

    assert not false_proposals, f"Stage 1/2 proposed {len(false_proposals)} false match(es): {false_proposals[:5]}"

    total_records = len(records)
    matched = len(result.matched_record_ids)
    print(
        f"\n[Stage 1+2 sanity check] calibration: {matched}/{total_records} "
        f"records proposed into {len(result.match_groups)} group(s), "
        f"{len(result.decision_events)} decision event(s), 0 false proposals."
    )


def test_every_proposed_group_has_a_matching_member_role_and_amounts() -> None:
    records = _load_records()
    settings = Settings()
    result = run_stage1_and_stage2(records, settings)

    members_by_group: dict[str, list] = {}
    for member in result.match_group_members:
        members_by_group.setdefault(member.group_id, []).append(member)

    for group in result.match_groups:
        members = members_by_group[group.group_id]
        assert len(members) >= 2
        assert all(m.group_id == group.group_id for m in members)
        assert group.expected_amount_paise >= 0
        assert group.matched_amount_paise >= 0


def test_every_group_is_pending_review_not_verified() -> None:
    # Neither stage should ever mark anything VERIFIED — that's the
    # not-yet-built Financial and Evidence Verifier's job.
    records = _load_records()
    settings = Settings()
    result = run_stage1_and_stage2(records, settings)

    assert result.match_groups, "expected at least some proposals to check"
    for group in result.match_groups:
        assert group.status.value == "PENDING_REVIEW"
        assert group.verified_by.value == "NOT_YET_VERIFIED"
        assert group.verification_result.value == "NOT_YET_RUN"


def test_every_proposal_has_a_decision_event() -> None:
    records = _load_records()
    settings = Settings()
    result = run_stage1_and_stage2(records, settings)

    logged_group_ids = {e.group_id for e in result.decision_events}
    for group in result.match_groups:
        assert group.group_id in logged_group_ids
