"""Integration test — Stage 1 -> 2 -> 3 -> 4 against the real calibration
dataset, checked against its hidden ground truth.

Ground truth (ground_truth/calibration/ground_truth.json) is read ONLY
inside this test file, never inside the matching code itself — see
test_matching_integration.py's identical note and testdata/generator.py's
own docstring.

Per the task: this extends test_matching_integration.py's Stage 1+2
pattern to the full Stage 1-4 pipeline. The one hard requirement is zero
false proposals — every group any stage proposed must correspond to a
real ground-truth structure, regardless of what the verifier later
decides about it. This test also prints the requested summary
(records proposed vs. total, broken down by which stage proposed each
group).

As of Stage 6, ``run_pipeline`` also wires every proposal through the
Financial and Evidence Verifier before returning it (see
matching/pipeline.py's ``run_verification``), so every group returned
here has already been marked VERIFIED / PENDING_REVIEW / REJECTED —
see test_pipeline_verification.py for verification-specific coverage,
including the VERIFIED-subset zero-false-positive check and the
release-on-rejection behavior.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from recon_agent.config import Settings
from recon_agent.matching import run_pipeline
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


def test_stage1to4_propose_zero_false_matches_against_calibration() -> None:
    records = _load_records()
    settings = Settings()
    result = run_pipeline(records, settings)

    ground_truth = _load_ground_truth()
    # A proposal is "true" if its member set is fully contained in some
    # real ground-truth structure — either a genuine match_group, or an
    # "unresolved" pending-counterpart pair (a stage correctly finding a
    # partial-but-real subset is not a false proposal — it's just
    # incomplete).
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

    assert not false_proposals, f"Stage 1-4 proposed {len(false_proposals)} false match(es): {false_proposals[:5]}"

    total_records = len(records)
    matched = len(result.matched_record_ids)
    by_stage = Counter(result.group_id_to_stage.values())

    print(
        f"\n[Stage 1-4 integration] calibration: {matched}/{total_records} "
        f"records proposed into {len(result.match_groups)} group(s), "
        f"{len(result.decision_events)} decision event(s), "
        f"{len(result.exceptions)} exception(s), 0 false proposals."
    )
    for stage_name in ("STAGE1_EXACT", "STAGE2_CONSTRAINED", "STAGE3_AGGREGATE", "STAGE4_ADJUSTMENT"):
        print(f"  {stage_name}: {by_stage.get(stage_name, 0)} group(s)")

    assert sum(by_stage.values()) == len(result.match_groups)


def test_stage3_and_stage4_only_operate_on_records_earlier_stages_left_unmatched() -> None:
    records = _load_records()
    settings = Settings()
    result = run_pipeline(records, settings)

    # Every record_id referenced by any proposed group must be a real
    # record in the input pool, and no record should appear as a member
    # of two different groups (each stage claims disjoint records from
    # what came before it).
    record_ids = {r.record_id for r in records}
    seen_in_group: dict[str, str] = {}
    for member in result.match_group_members:
        assert member.record_id in record_ids
        assert member.record_id not in seen_in_group, (
            f"record {member.record_id} appears in both group "
            f"{seen_in_group.get(member.record_id)} and {member.group_id}"
        )
        seen_in_group[member.record_id] = member.group_id


def test_every_group_has_actually_been_verified() -> None:
    # As of Stage 6, run_pipeline wires every Stage 1-4 proposal through
    # the Financial and Evidence Verifier (matching/pipeline.py's
    # run_verification) before returning it — so, unlike before that
    # stage existed, no group should still be sitting at the
    # NOT_YET_VERIFIED/NOT_YET_RUN placeholder values a bare proposal
    # carries. See test_pipeline_verification.py for the full
    # VERIFIED/PENDING_REVIEW/REJECTED behavior this replaced.
    records = _load_records()
    settings = Settings()
    result = run_pipeline(records, settings)

    for group in result.match_groups:
        assert group.verified_by.value == "FINANCIAL_AND_EVIDENCE_VERIFIER"
        assert group.verification_result.value != "NOT_YET_RUN"
        assert group.status.value in ("VERIFIED", "PENDING_REVIEW", "REJECTED")


def test_every_exception_has_an_expected_category_and_is_open() -> None:
    # Exceptions can now come from three sources: Stage 3/4's own
    # AMBIGUOUS_AGGREGATION (aggregation search couldn't cleanly resolve
    # a window), Stage 7 verification REJECTED outcomes (mapped per
    # matching/pipeline.py's REJECTION_EXCEPTION_CATEGORY), and Stage 6's
    # LLM recommendation pass (matching/stage6_llm.py) — a NO_MATCH
    # recommendation or no near-miss candidates produce
    # INSUFFICIENT_EVIDENCE, and a missing/exhausted/circuit-broken Groq
    # call produces LLM_UNAVAILABLE (see stage6_llm.py's module docstring
    # for why LLM_BUDGET_EXHAUSTED and LLM_CIRCUIT_BREAKER_OPEN are both
    # folded into LLM_UNAVAILABLE rather than the taxonomy growing new
    # enum values). This allow-list predates Stage 6; with no Groq key
    # configured in this test environment, the pipeline gracefully
    # degrades to LLM_UNAVAILABLE for its residual records — correct
    # behavior, not a bug — so the allow-list is updated to include
    # Stage 6's legitimate categories. Every exception, regardless of
    # source, should still be freshly OPEN.
    records = _load_records()
    settings = Settings()
    result = run_pipeline(records, settings)

    expected_categories = {
        "AMBIGUOUS_AGGREGATION",
        "CURRENCY_MISMATCH",
        "REUSED_COUNTERPART",
        "PARTIAL_SETTLEMENT",
        "LLM_UNAVAILABLE",
        "LLM_BUDGET_EXHAUSTED",
        "INSUFFICIENT_EVIDENCE",
    }
    for exc in result.exceptions:
        assert exc.category.value in expected_categories
        assert exc.review_status.value == "OPEN"
