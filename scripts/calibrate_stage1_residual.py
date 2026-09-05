"""Stage 1 residual-tolerance calibration analysis (bugfix follow-up).

CONTEXT: verifier.py's Stage 1 residual check (see
config.Settings.stage1_residual_tolerance_fraction) was added correctly
and safely — but its shipped default (2.5%, picked from the
LEDGER/GATEWAY fee-tier arithmetic hardcoded into
testdata/generator.py's MDR_BPS_CHOICES, not from measuring what the
data actually looks like) is measurably too tight: it flags a
meaningful share of genuinely correct Stage 1 matches as
RESIDUAL_EXCEEDS_TOLERANCE, dropping record coverage from ~87% to
~62% and value coverage from ~70% to ~45%.

This script re-derives the threshold the honest way, per
ARCHITECTURE.md §11's calibration/evaluation discipline: run the full
Stage 1-5 proposal pipeline against data/calibration/records.json
ONLY, take every group that is *still* tagged STAGE1_EXACT once
Stage 3/4's in-place group extension has happened (a raw Stage 1
LEDGER+GATEWAY pair that later gets a bank leg attached is re-proposed
as a new STAGE3_AGGREGATE group per pipeline.py's own accounting, and
is verified by Stage 3's conservation check instead — this residual
check never runs against it, so including it here would analyze a
population the config value doesn't actually gate). This is exactly
the population evaluation/harness.py's own tier_contribution metric
counts as "STAGE1_EXACT proposed", so the two numbers agree.

For every one of those groups: compute the residual as a percentage of
the larger of its two amounts (mirroring
verification/verifier.py's own ``_stage1_residual``/``_verify_stage1``
math exactly), then independently cross-reference correctness against
calibration ground truth (match_groups + unresolved as the "true"
clusters, duplicates + honest_abstention as automatic disqualifiers —
the same four-way check evaluation/harness.is_verified_group_correct
already uses, reused here rather than re-implemented), and report the
real distribution instead of guessing at one.

data/evaluation/ is never read by this script — it stays held out.

Usage:
    python -m scripts.calibrate_stage1_residual
    (or) python scripts/calibrate_stage1_residual.py
"""

from __future__ import annotations

import statistics
import uuid
from pathlib import Path

from recon_agent.config import Settings
from recon_agent.evaluation.harness import (
    GroundTruth,
    is_verified_group_correct,
    load_ground_truth,
    load_records,
)
from recon_agent.llm.governor import CallBudgetGovernor
from recon_agent.matching.pipeline import run_pipeline
from recon_agent.models import NormalizedRecord

REPO_ROOT = Path(__file__).resolve().parent.parent


def analyze(dataset: str = "calibration") -> None:
    records_path = REPO_ROOT / "data" / dataset / "records.json"
    gt_path = REPO_ROOT / "ground_truth" / dataset / "ground_truth.json"

    records: list[NormalizedRecord] = load_records(records_path)
    gt: GroundTruth = load_ground_truth(gt_path)

    settings = Settings.from_env()
    governor = CallBudgetGovernor(settings, db_path=":memory:")
    # groq_client=object() forces graceful degradation (§14) rather than
    # a live Groq call — Stage 6 is irrelevant to this analysis, which
    # only looks at groups still tagged STAGE1_EXACT after Stage 1-5.
    result = run_pipeline(
        records, settings, run_id=uuid.uuid4().hex, governor=governor, groq_client=object()
    )
    governor.close()

    members_by_group: dict[str, list[str]] = {}
    for m in result.match_group_members:
        members_by_group.setdefault(m.group_id, []).append(m.record_id)

    stage1_groups = [
        g for g in result.match_groups if result.group_id_to_stage.get(g.group_id) == "STAGE1_EXACT"
    ]

    rows = []
    for group in stage1_groups:
        member_ids = frozenset(members_by_group[group.group_id])
        is_correct, reason = is_verified_group_correct(member_ids, gt)
        expected = group.expected_amount_paise
        matched = group.matched_amount_paise
        denom = max(expected, matched)
        residual_pct = (abs(expected - matched) / denom * 100.0) if denom else 0.0
        rows.append(
            {
                "group_id": group.group_id,
                "record_ids": sorted(member_ids),
                "expected_paise": expected,
                "matched_paise": matched,
                "residual_pct": residual_pct,
                "is_correct": is_correct,
                "reason": reason,
            }
        )

    rows.sort(key=lambda r: r["residual_pct"])

    correct = [r for r in rows if r["is_correct"]]
    incorrect = [r for r in rows if not r["is_correct"]]
    correct_pcts = [r["residual_pct"] for r in correct]
    incorrect_pcts = [r["residual_pct"] for r in incorrect]

    print(f"=== Stage 1 residual analysis — dataset: {dataset} ===")
    print(f"Total STAGE1_EXACT proposals: {len(rows)}")
    print(f"  correct (per ground truth):   {len(correct)}")
    print(f"  incorrect (per ground truth): {len(incorrect)}")
    print()

    print("--- Full per-proposal table (sorted by residual %) ---")
    print(f"{'group_id':<14}{'residual_%':>11}  {'expected':>12}  {'matched':>12}  {'correct':>8}  reason")
    for r in rows:
        print(
            f"{r['group_id']:<14}{r['residual_pct']:>10.4f}%  "
            f"{r['expected_paise']:>12}  {r['matched_paise']:>12}  "
            f"{str(r['is_correct']):>8}  {r['reason']}"
        )
    print()

    def describe(label: str, pcts: list[float]) -> None:
        if not pcts:
            print(f"{label}: (none)")
            return
        pcts_sorted = sorted(pcts)
        n = len(pcts_sorted)
        median = statistics.median(pcts_sorted)

        def pct_at(p: float) -> float:
            # nearest-rank percentile, no interpolation needed at this n
            idx = min(n - 1, max(0, round(p / 100 * (n - 1))))
            return pcts_sorted[idx]

        print(f"{label} (n={n}):")
        print(f"  min:    {pcts_sorted[0]:.4f}%")
        print(f"  p25:    {pct_at(25):.4f}%")
        print(f"  median: {median:.4f}%")
        print(f"  p75:    {pct_at(75):.4f}%")
        print(f"  p90:    {pct_at(90):.4f}%")
        print(f"  p95:    {pct_at(95):.4f}%")
        print(f"  max:    {pcts_sorted[-1]:.4f}%")
        # unique values, since the generator produces a small discrete set
        # of fee tiers rather than a continuous spread
        uniq = sorted(set(round(p, 4) for p in pcts_sorted))
        print(f"  distinct values observed: {uniq}")

    describe("Residuals on GENUINELY CORRECT matches", correct_pcts)
    print()
    describe("Residuals on INCORRECT proposals (if any)", incorrect_pcts)
    print()

    if correct_pcts:
        obs_max = max(correct_pcts)
        print(
            f"Highest residual % seen among genuinely correct calibration "
            f"matches: {obs_max:.4f}%"
        )
        if incorrect_pcts:
            print(
                f"Lowest residual % seen among incorrect proposals: "
                f"{min(incorrect_pcts):.4f}% "
                f"({'ABOVE' if min(incorrect_pcts) > obs_max else 'AT-OR-BELOW'} "
                "the correct-match maximum — "
                + (
                    "clean separation exists in this data."
                    if min(incorrect_pcts) > obs_max
                    else "no clean magnitude-only separation; the incorrect "
                    "case(s) must be caught by something other than residual "
                    "size alone (see verifier's other checks)."
                )
            )
        else:
            print("No incorrect STAGE1_EXACT proposals exist in this dataset at all.")


if __name__ == "__main__":
    import sys

    dataset = sys.argv[1] if len(sys.argv) > 1 else "calibration"
    analyze(dataset)
