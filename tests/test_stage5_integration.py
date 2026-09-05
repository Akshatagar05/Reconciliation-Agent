"""Integration test — Stage 5 specifically, against both the calibration
and evaluation datasets, per this stage's task.

Three things this file reports/asserts, isolated to Stage 5's own
contribution rather than the whole pipeline (test_stage1to4_integration.py
and test_pipeline_verification.py already cover the whole-pipeline
zero-false-positive checks; this file adds the Stage-5-specific view the
task explicitly asked for):

  1. How many additional records Stage 5 specifically proposed a match
     for (i.e. records unmatched after Stage 1-4 that Stage 5 retrieved
     a fuzzy candidate for), broken down by final VERIFIED /
     PENDING_REVIEW / REJECTED outcome.
  2. Every newly-VERIFIED group (system-wide, not just Stage 5's own —
     a VERIFIED group from any earlier stage is just as much a
     regression risk if Stage 5 somehow interfered with it) is
     independently cross-checked against ALL THREE ground-truth
     categories: real match_groups, unresolved (a correct partial
     subset isn't a false positive), and duplicates — not just
     match_groups, which a prior review flagged as an incomplete check.
  3. A documented near-miss found while building this test, worth
     keeping as a permanent regression check: on the evaluation
     dataset, Stage 5 retrieves and proposes a fuzzy match between a
     genuine stray record (evl_rec_00098, a real member of a
     MANY_TO_ONE group Stage 3/4's aggregation search couldn't fully
     resolve) and evl_rec_00102 — which ground truth marks as a
     structural DUPLICATE of evl_rec_00099, not a real transaction of
     its own. Stage 5's own hard gates (identifier + counterparty +
     amount + date + margin) all pass, because a duplicate decoy is by
     construction a near-perfect look-alike of the real record it
     copies. The Financial and Evidence Verifier's independent
     conservation re-check (§2's "necessary but not sufficient" second
     layer) is what actually catches this one, since a lone
     ledger<->gateway pair with no bank leg can never independently
     prove conservation — see matching/common.py's
     ``recompute_group_conservation`` docstring, and note this is
     exactly the same fate an equivalent unextended STAGE2_CONSTRAINED
     ledger<->gateway pair gets (see
     test_stage1to4_integration.py's calibration run, where five such
     Stage 2 pairs are REJECTED the same way — this is not a Stage
     5-specific weakness, it's how the whole relay's layered defense is
     supposed to behave when retrieval evidence outruns provable
     conservation). This is exactly why §5 frames Stage 5 as retrieval,
     never a sufficient commit basis by itself: the retrieval step
     found a plausible-looking candidate; a different, independent
     layer is what stopped it from being wrongly committed.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from recon_agent.config import Settings
from recon_agent.matching import run_pipeline
from recon_agent.models import MatchGroupStatus, NormalizedRecord

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


def _cross_check_verified_groups_against_all_three_categories(
    result, ground_truth: dict
) -> list[tuple[str, str, set[str]]]:
    """The zero-false-positive check, scoped to VERIFIED groups, cross-checked
    against all three ground-truth categories (match_groups, unresolved,
    duplicates) — not just match_groups."""
    true_clusters = [set(g["record_ids"]) for g in ground_truth["match_groups"]]
    true_clusters += [set(u["record_ids"]) for u in ground_truth["unresolved"]]
    duplicate_ids = {d["record_id"] for d in ground_truth["duplicates"]}
    abstention_ids = {a["record_id"] for a in ground_truth["honest_abstention"]}

    members_by_group: dict[str, list[str]] = {}
    for member in result.match_group_members:
        members_by_group.setdefault(member.group_id, []).append(member.record_id)

    false_positives = []
    for group in result.match_groups:
        if group.status != MatchGroupStatus.VERIFIED:
            continue
        member_ids = set(members_by_group[group.group_id])
        if member_ids & duplicate_ids:
            false_positives.append((group.group_id, "includes a known duplicate", member_ids))
            continue
        if member_ids & abstention_ids:
            false_positives.append((group.group_id, "includes an honest-abstention decoy", member_ids))
            continue
        if not any(member_ids <= true_cluster for true_cluster in true_clusters):
            false_positives.append((group.group_id, "not a subset of any true cluster", member_ids))

    return false_positives


def _run_and_report(records_path: Path, ground_truth_path: Path, label: str):
    records = _load_records(records_path)
    settings = Settings()
    result = run_pipeline(records, settings)
    ground_truth = _load_ground_truth(ground_truth_path)

    members_by_group: dict[str, list[str]] = {}
    for member in result.match_group_members:
        members_by_group.setdefault(member.group_id, []).append(member.record_id)

    stage5_groups = [
        g for g in result.match_groups if result.group_id_to_stage.get(g.group_id) == "STAGE5_FUZZY"
    ]
    stage5_records: set[str] = set()
    for g in stage5_groups:
        stage5_records.update(members_by_group[g.group_id])
    stage5_by_status = Counter(g.status.value for g in stage5_groups)

    # (1) Requirement #1 — every group any stage proposed must correspond
    # to a real ground-truth structure, scoped to Stage 5's own proposals
    # specifically (this is the same "no false proposals" bar
    # test_stage1to4_integration.py applies to Stage 1-4 — a PENDING/
    # REJECTED Stage 5 proposal being a real ground-truth subset is fine;
    # a proposal with no ground-truth basis at all is not).
    true_clusters = [set(g["record_ids"]) for g in ground_truth["match_groups"]]
    true_clusters += [set(u["record_ids"]) for u in ground_truth["unresolved"]]
    duplicate_ids = {d["record_id"] for d in ground_truth["duplicates"]}
    abstention_ids = {a["record_id"] for a in ground_truth["honest_abstention"]}

    stage5_no_ground_truth_basis = []
    stage5_touches_duplicate = []
    for g in stage5_groups:
        member_ids = set(members_by_group[g.group_id])
        if member_ids & abstention_ids:
            stage5_no_ground_truth_basis.append((g.group_id, member_ids))
        elif not (member_ids & duplicate_ids) and not any(
            member_ids <= tc for tc in true_clusters
        ):
            stage5_no_ground_truth_basis.append((g.group_id, member_ids))
        if member_ids & duplicate_ids:
            stage5_touches_duplicate.append((g.group_id, g.status.value, member_ids))

    # (2) Zero false positives among VERIFIED groups, system-wide,
    # cross-checked against all three ground-truth categories.
    false_positives = _cross_check_verified_groups_against_all_three_categories(result, ground_truth)

    print(f"\n[Stage 5 integration] {label}: {len(records)} total record(s).")
    print(
        f"  Stage 5 proposed {len(stage5_groups)} group(s) covering "
        f"{len(stage5_records)} additional record(s): "
        f"VERIFIED={stage5_by_status.get('VERIFIED', 0)}, "
        f"PENDING_REVIEW={stage5_by_status.get('PENDING_REVIEW', 0)}, "
        f"REJECTED={stage5_by_status.get('REJECTED', 0)}."
    )
    if stage5_touches_duplicate:
        print(
            f"  {len(stage5_touches_duplicate)} Stage 5 proposal(s) touched a "
            f"known duplicate (see module docstring's documented near-miss): "
            f"{stage5_touches_duplicate}"
        )
    print(
        f"  Zero-false-positive cross-check (match_groups + unresolved + "
        f"duplicates) across ALL VERIFIED groups (any stage): "
        f"{len(false_positives)} false positive(s)."
    )

    return result, stage5_groups, stage5_no_ground_truth_basis, false_positives


def test_stage5_calibration_zero_false_positives_all_three_categories() -> None:
    result, stage5_groups, stage5_no_ground_truth_basis, false_positives = _run_and_report(
        CALIBRATION_RECORDS, CALIBRATION_GROUND_TRUTH, "calibration"
    )
    assert not stage5_no_ground_truth_basis, (
        f"Stage 5 proposed {len(stage5_no_ground_truth_basis)} group(s) on "
        f"calibration with no basis in any ground-truth category: "
        f"{stage5_no_ground_truth_basis}"
    )
    assert not false_positives, (
        f"{len(false_positives)} VERIFIED group(s) on calibration are false "
        f"positives against match_groups/unresolved/duplicates: {false_positives[:5]}"
    )


def test_stage5_evaluation_zero_false_positives_all_three_categories() -> None:
    result, stage5_groups, stage5_no_ground_truth_basis, false_positives = _run_and_report(
        EVALUATION_RECORDS, EVALUATION_GROUND_TRUTH, "evaluation"
    )
    assert not stage5_no_ground_truth_basis, (
        f"Stage 5 proposed {len(stage5_no_ground_truth_basis)} group(s) on "
        f"evaluation with no basis in any ground-truth category: "
        f"{stage5_no_ground_truth_basis}"
    )
    assert not false_positives, (
        f"{len(false_positives)} VERIFIED group(s) on evaluation are false "
        f"positives against match_groups/unresolved/duplicates: {false_positives[:5]}"
    )


def test_stage5_never_verifies_a_group_touching_a_known_duplicate() -> None:
    """The specific near-miss documented in this module's docstring,
    pinned as a permanent regression check: whatever Stage 5 proposes
    that happens to touch a ground-truth duplicate record must never
    reach VERIFIED status — the layered defense (Stage 5's own gates
    plus the verifier's independent conservation re-check) must keep
    catching it even as the dataset or thresholds change."""
    for records_path, gt_path, label in (
        (CALIBRATION_RECORDS, CALIBRATION_GROUND_TRUTH, "calibration"),
        (EVALUATION_RECORDS, EVALUATION_GROUND_TRUTH, "evaluation"),
    ):
        records = _load_records(records_path)
        result = run_pipeline(records, Settings())
        ground_truth = _load_ground_truth(gt_path)
        duplicate_ids = {d["record_id"] for d in ground_truth["duplicates"]}

        members_by_group: dict[str, list[str]] = {}
        for member in result.match_group_members:
            members_by_group.setdefault(member.group_id, []).append(member.record_id)

        for g in result.match_groups:
            if result.group_id_to_stage.get(g.group_id) != "STAGE5_FUZZY":
                continue
            member_ids = set(members_by_group[g.group_id])
            if member_ids & duplicate_ids:
                assert g.status != MatchGroupStatus.VERIFIED, (
                    f"Stage 5 group {g.group_id} on {label} touches a known "
                    f"duplicate {member_ids & duplicate_ids} and must never be "
                    f"VERIFIED, but is."
                )
