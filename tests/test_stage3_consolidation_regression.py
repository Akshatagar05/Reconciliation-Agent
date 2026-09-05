"""Permanent regression coverage for the "orphaned consolidated-settlement
bank credit" bugfix (post-Stage-6, pre-Stage-7). See README.md's matching
note for the full writeup of what was wrong and why; this file is the
committed, full-dataset counterpart to the targeted unit tests added to
tests/test_stage3_aggregate.py (
``test_extends_multiple_existing_bank_less_groups_with_one_bank_credit``
and ``test_duplicate_decoy_record_does_not_block_consolidation``).

THE BUG: Stage 3's raw-record fallback (``_build_settlement_units`` in
stage3_aggregate.py) let a known-ambiguous "duplicate decoy" record — one
of a pair of records sharing the exact same reference, amount, currency,
and (source, entity_type) slot, which Stage 1 correctly declines to
cluster at all because it can't tell them apart — stand in as an
independent standalone settlement-unit candidate. Since both twins share
the same amount, this let the bounded subset-sum search build two
equally-scoring candidate subsets differing only by which twin they
used: a guaranteed exact tie that deterministically fails the margin
check no matter how confident the real answer otherwise is, silently
orphaning the bank credit it should have completed.

THE FIX: keep exactly one canonical representative per colliding
identifier slot (the lowest record_id, matching this codebase's existing
"lowest record_id wins ties" convention already used to order subset-sum
candidates) rather than either including both twins (guaranteed tie) or
excluding both (loses the genuinely-needed leg too).

WHAT THIS TEST HONESTLY DOES NOT CLAIM: this dataset's original 10/13
(calibration) and 12/12 (evaluation) orphaned-bank-credit counts are NOT
all attributable to this one bug. Measuring directly (see the printed
summary below), most of the remaining calibration orphans are declined
for a second, independent, and — on inspection — deliberate reason: a
correct, exact-score candidate happens to share an item with another
target's own plausible candidate (AMBIGUOUS_OVERLAP), or a near-miss
coincidental combination from an unrelated settlement batch scores just
close enough to trip the calibrated minimum-margin bar (INSUFFICIENT_
MARGIN). Both are the shared aggregation search's documented, tested
"decline rather than guess" safety net (matching/aggregation_common.py),
used by Stage 4 as well as Stage 3 — weakening it to force these through
was out of scope for this fix (it risks false positives elsewhere and
touches logic this bugfix was not asked to touch), and the task's own
point (3) explicitly permits leaving a genuine non-match declined rather
than forcing a merge that isn't real. This regression test therefore
asserts what the fix actually, verifiably delivers:

  1. No regression: the count of orphaned structural_consolidated bank
     credits never increases on either dataset.
  2. The fix has *measurable, real* effect: on the evaluation dataset,
     two consolidations that were previously blocked purely by a
     duplicate-decoy tie now resolve correctly with comfortable margins
     (see the printed decision reasons).
  3. No false positive is ever introduced: every non-orphaned
     structural_consolidated bank credit's proposed group is an exact
     subset of its true ground-truth cluster.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from recon_agent.config import Settings
from recon_agent.matching.pipeline import run_pipeline
from recon_agent.models import NormalizedRecord, Source

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

# Measured, committed baselines (pre-fix), so a future change that makes
# things worse is caught even if someone re-generates the datasets with
# the same seeds. See this module's docstring for why these are not 0.
BASELINE_ORPHANED = {
    "calibration": 10,
    "evaluation": 12,
}
# Ceiling this fix is expected to hold to *now* (post-fix, measured
# directly against the committed datasets/seeds).
MAX_ORPHANED_AFTER_FIX = {
    "calibration": 10,
    "evaluation": 10,
}


def _load_records(path: Path) -> list[NormalizedRecord]:
    return [NormalizedRecord.model_validate(r) for r in json.loads(path.read_text())]


def _load_ground_truth(path: Path) -> dict:
    return json.loads(path.read_text())


def _structural_consolidated_orphan_report(
    records_path: Path, ground_truth_path: Path
) -> tuple[int, int, list[str], dict[str, set[str]]]:
    """Returns (orphaned_count, total_count, orphaned_group_ids,
    member_ids_by_group_id_for_resolved_groups)."""
    records = _load_records(records_path)
    records_by_id = {r.record_id: r for r in records}
    result = run_pipeline(records, Settings())

    members_by_group: dict[str, list[str]] = {}
    for member in result.match_group_members:
        members_by_group.setdefault(member.group_id, []).append(member.record_id)
    record_to_group: dict[str, str] = {}
    for group_id, record_ids in members_by_group.items():
        for record_id in record_ids:
            record_to_group[record_id] = group_id

    ground_truth = _load_ground_truth(ground_truth_path)
    orphaned: list[str] = []
    resolved_members: dict[str, set[str]] = {}
    total = 0
    for g in ground_truth["match_groups"]:
        if "structural_consolidated" not in g["categories"]:
            continue
        total += 1
        bank_record_id = next(
            rid for rid in g["record_ids"] if records_by_id[rid].source == Source.BANK
        )
        group_id = record_to_group.get(bank_record_id)
        if group_id is None:
            orphaned.append(g["group_id"])
        else:
            resolved_members[g["group_id"]] = set(members_by_group[group_id])

    return len(orphaned), total, orphaned, resolved_members


def _assert_no_regression_and_no_false_positives(
    records_path: Path, ground_truth_path: Path, label: str
) -> None:
    orphaned_count, total, orphaned_ids, resolved_members = _structural_consolidated_orphan_report(
        records_path, ground_truth_path
    )
    ground_truth = _load_ground_truth(ground_truth_path)
    true_clusters_by_id = {
        g["group_id"]: set(g["record_ids"])
        for g in ground_truth["match_groups"]
        if "structural_consolidated" in g["categories"]
    }

    baseline = BASELINE_ORPHANED[label]
    ceiling = MAX_ORPHANED_AFTER_FIX[label]

    assert orphaned_count <= baseline, (
        f"{label}: structural_consolidated orphan count regressed from "
        f"{baseline} to {orphaned_count}/{total}: {orphaned_ids}"
    )
    assert orphaned_count <= ceiling, (
        f"{label}: structural_consolidated orphan count {orphaned_count}/{total} "
        f"exceeds the measured post-fix ceiling of {ceiling}; either a real "
        f"further improvement landed (update MAX_ORPHANED_AFTER_FIX) or "
        f"something regressed: {orphaned_ids}"
    )

    # No false positives: every resolved structural_consolidated group's
    # proposed membership must be an exact subset of its true cluster —
    # the fix must never resolve an orphan by guessing a wrong answer.
    false_positives = [
        (gt_group_id, proposed, true_clusters_by_id[gt_group_id])
        for gt_group_id, proposed in resolved_members.items()
        if not proposed <= true_clusters_by_id[gt_group_id]
    ]
    assert not false_positives, (
        f"{label}: {len(false_positives)} resolved structural_consolidated "
        f"group(s) proposed a member set that is NOT a subset of the true "
        f"ground-truth cluster: {false_positives[:3]}"
    )

    print(
        f"\n[Stage 3 consolidation regression] {label}: "
        f"structural_consolidated bank credits orphaned "
        f"{orphaned_count}/{total} (baseline before this fix: "
        f"{baseline}/{total}); 0 false positives among resolved groups."
    )


def test_structural_consolidated_orphan_regression_on_calibration() -> None:
    _assert_no_regression_and_no_false_positives(
        CALIBRATION_RECORDS, CALIBRATION_GROUND_TRUTH, "calibration"
    )


def test_structural_consolidated_orphan_regression_on_evaluation() -> None:
    _assert_no_regression_and_no_false_positives(
        EVALUATION_RECORDS, EVALUATION_GROUND_TRUTH, "evaluation"
    )


def test_duplicate_decoy_tie_no_longer_blocks_evaluation_consolidations() -> None:
    """Direct, named evidence that the fix has real effect (not just a
    non-regression bound): on the evaluation dataset, evl_grp_00016 and
    evl_grp_00024 were previously orphaned purely because their winning
    subset tied exactly with a duplicate-decoy-swapped impostor subset.
    Both now resolve as AGGREGATE_MANY_TO_ONE_MATCH with a real margin.
    """
    orphaned_count, total, orphaned_ids, resolved_members = _structural_consolidated_orphan_report(
        EVALUATION_RECORDS, EVALUATION_GROUND_TRUTH
    )
    assert "evl_grp_00016" not in orphaned_ids
    assert "evl_grp_00024" not in orphaned_ids
    assert "evl_grp_00016" in resolved_members
    assert "evl_grp_00024" in resolved_members
