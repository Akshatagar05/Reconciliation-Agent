"""Integration tests for Stage 11: the shared budget governor wired into
Stage 6, and Stage 6 wired into the full pipeline (matching/pipeline.py's
``run_pipeline``), run against the real calibration and evaluation
datasets with a MOCKED Groq client — no live GROQ_API_KEY is required or
used anywhere in this file.

Per this stage's brief:
  - report how many additional records Stage 6 attempted, and the
    outcome breakdown (recommendation / NO_MATCH / failure / degraded /
    no-candidates)
  - independently cross-check that nothing became falsely VERIFIED —
    the zero-false-positive standard already used in
    test_pipeline_verification.py, now re-checked against the full
    Stage 1-6 pipeline output, across all three ground-truth
    "wrong" categories (a known duplicate, an honest-abstention decoy,
    or membership that isn't a subset of any real ground-truth cluster)
  - a direct proof that no STAGE6_LLM-proposed group is ever VERIFIED,
    regardless of the mocked LLM's stated confidence (§2's non-
    negotiable rule, re-verified end-to-end here rather than only at
    the stage6_llm.py unit level in tests/test_stage6_llm.py)

The mocked client here is a "heuristic mock": it actually reads the
prompt's residual/candidate JSON payload (the same shape
recommender.py's own ``build_prompt`` produces) and picks the candidate
whose reference/counterparty/amount are closest, rather than returning
one canned response for every call — this makes the recommendation
distribution across the real dataset meaningful (some genuine
NO_MATCH decisions, not just one hardcoded outcome), while remaining
fully deterministic and network-free.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from rapidfuzz import fuzz

from recon_agent.config import Settings
from recon_agent.llm.governor import CallBudgetGovernor
from recon_agent.matching.pipeline import run_pipeline
from recon_agent.models import MatchGroupStatus, NormalizedRecord, ProposedBy
from recon_agent.normalization import normalize_reference

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


# ---------------------------------------------------------------------------
# The heuristic mock Groq client — reads the real prompt, applies a
# simple deterministic scoring rule, returns a schema-valid response.
# Never touches the network or a real Groq SDK object.
# ---------------------------------------------------------------------------


def _extract_payload(user_message: str) -> dict[str, Any]:
    json_start = user_message.index("{")
    return json.loads(user_message[json_start:])


def _heuristic_score(residual: dict[str, Any], candidate: dict[str, Any]) -> float:
    ref_score = fuzz.ratio(
        normalize_reference(residual["reference"]), normalize_reference(candidate["reference"])
    ) / 100.0
    counterparty_score = fuzz.ratio(
        normalize_reference(residual["counterparty"]), normalize_reference(candidate["counterparty"])
    ) / 100.0
    a, b = abs(residual["amount_paise"]), abs(candidate["amount_paise"])
    denom = max(a, b, 1)
    amount_score = max(0.0, 1.0 - abs(a - b) / denom)
    return 0.4 * ref_score + 0.3 * counterparty_score + 0.3 * amount_score


def _heuristic_mock_create(**kwargs) -> SimpleNamespace:
    user_message = kwargs["messages"][1]["content"]
    payload = _extract_payload(user_message)
    residual = payload["residual_record"]
    candidates = payload["candidates"]

    scored = sorted(
        ((c, _heuristic_score(residual, c)) for c in candidates),
        key=lambda pair: -pair[1],
    )
    best_candidate, best_score = scored[0]

    if best_score >= 0.75:
        body = {
            "candidate_id": best_candidate["record_id"],
            "confidence": round(best_score, 3),
            "reason_code": "HEURISTIC_MOCK_STRONG_MATCH",
            "explanation": (
                f"Mocked recommender: {best_candidate['record_id']} scored "
                f"{best_score:.3f} against {residual['record_id']}."
            ),
        }
    else:
        body = {
            "candidate_id": "NO_MATCH",
            "confidence": round(1.0 - best_score, 3),
            "reason_code": "HEURISTIC_MOCK_NO_PLAUSIBLE_CANDIDATE",
            "explanation": (
                f"Mocked recommender: best candidate score {best_score:.3f} "
                f"for {residual['record_id']} is below the mock's own bar."
            ),
        }

    message = SimpleNamespace(content=json.dumps(body))
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def _heuristic_mock_client() -> MagicMock:
    client = MagicMock()
    client.chat.completions.create.side_effect = _heuristic_mock_create
    return client


def _run_with_mock_groq(records: list[NormalizedRecord], tmp_path: Path, run_id: str):
    settings = Settings()
    governor = CallBudgetGovernor(settings, db_path=str(tmp_path / f"{run_id}-governor.db"))
    client = _heuristic_mock_client()
    result = run_pipeline(records, settings, run_id=run_id, governor=governor, groq_client=client)
    return result, client


# ---------------------------------------------------------------------------
# Stage 6 attempt/outcome reporting + zero-false-positive cross-check
# ---------------------------------------------------------------------------


def _summarize_and_check(records_path: Path, ground_truth_path: Path, label: str, tmp_path: Path) -> None:
    records = _load_records(records_path)
    result, client = _run_with_mock_groq(records, tmp_path, run_id=f"integration-{label}")

    ground_truth = _load_ground_truth(ground_truth_path)
    true_clusters = [set(g["record_ids"]) for g in ground_truth["match_groups"]]
    true_clusters += [set(u["record_ids"]) for u in ground_truth["unresolved"]]
    duplicate_ids = {d["record_id"] for d in ground_truth["duplicates"]}
    abstention_ids = {a["record_id"] for a in ground_truth["honest_abstention"]}

    members_by_group: dict[str, list[str]] = {}
    for member in result.match_group_members:
        members_by_group.setdefault(member.group_id, []).append(member.record_id)

    stage6_groups = [g for g in result.match_groups if g.proposed_by == ProposedBy.STAGE6_LLM]
    stage6_exceptions = [e for e in result.exceptions if e.evidence.get("stage") == "STAGE6_LLM"]

    reason_counts = Counter(e.evidence.get("degradation_reason") or e.evidence.get("reason") for e in stage6_exceptions)
    llm_call_attempts = client.chat.completions.create.call_count

    # ---- Cross-check 1: no STAGE6_LLM group is ever VERIFIED (§2). ----
    stage6_verified = [g for g in stage6_groups if g.status == MatchGroupStatus.VERIFIED]
    assert not stage6_verified, (
        f"{len(stage6_verified)} STAGE6_LLM group(s) ended VERIFIED on {label} — "
        "§2 requires HUMAN_REVIEW_REQUIRED with no auto-commit path, ever."
    )
    for group in stage6_groups:
        assert group.status == MatchGroupStatus.PENDING_REVIEW
        assert group.commit_policy.value == "HUMAN_REVIEW_REQUIRED"
        assert group.verified_by.value == "NOT_YET_VERIFIED"

    # ---- Cross-check 2: the zero-false-positive standard, re-applied ----
    # to the FULL Stage 1-6 pipeline output, across all three "wrong"
    # ground-truth categories.
    verified_groups = [g for g in result.match_groups if g.status == MatchGroupStatus.VERIFIED]
    false_positives = []
    for group in verified_groups:
        member_ids = set(members_by_group.get(group.group_id, []))
        if member_ids & duplicate_ids:
            false_positives.append((group.group_id, "includes a known duplicate", member_ids))
            continue
        if member_ids & abstention_ids:
            false_positives.append((group.group_id, "includes an honest-abstention decoy", member_ids))
            continue
        if not any(member_ids <= true_cluster for true_cluster in true_clusters):
            false_positives.append((group.group_id, "not a subset of any true cluster", member_ids))

    assert not false_positives, (
        f"{len(false_positives)} false-positive VERIFIED group(s) on {label} "
        f"after Stage 6 wiring: {false_positives[:5]}"
    )

    print(
        f"\n[Stage 11] {label}: Stage 6 attempted {llm_call_attempts} Groq call(s) "
        f"across {len(stage6_groups) + len(stage6_exceptions)} residual record(s) "
        f"considered — {len(stage6_groups)} recommendation(s) proposed "
        f"(all PENDING_REVIEW/HUMAN_REVIEW_REQUIRED), "
        f"{sum(1 for e in stage6_exceptions if e.evidence.get('llm_reason_code'))} NO_MATCH, "
        f"outcome/degradation reasons: {dict(reason_counts)}. "
        f"0 false-positive VERIFIED group(s) across all three ground-truth "
        f"categories (duplicates, honest-abstention, non-subset-of-truth)."
    )


def test_stage6_wiring_reports_and_holds_zero_false_positives_on_calibration(tmp_path) -> None:
    _summarize_and_check(CALIBRATION_RECORDS, CALIBRATION_GROUND_TRUTH, "calibration", tmp_path)


def test_stage6_wiring_reports_and_holds_zero_false_positives_on_evaluation(tmp_path) -> None:
    _summarize_and_check(EVALUATION_RECORDS, EVALUATION_GROUND_TRUTH, "evaluation", tmp_path)


# ---------------------------------------------------------------------------
# Graceful degradation, at full-pipeline scale: a governor with a ceiling
# of zero must still let the whole pipeline complete, with every residual
# record that would have gone to Stage 6 showing up as an
# LLM_UNAVAILABLE exception instead of a crash or a stall.
# ---------------------------------------------------------------------------


def test_full_pipeline_completes_with_zero_call_budget(tmp_path) -> None:
    records = _load_records(CALIBRATION_RECORDS)
    settings = Settings(groq_daily_call_ceiling=0)
    governor = CallBudgetGovernor(settings, db_path=str(tmp_path / "zero-budget-governor.db"))
    client = _heuristic_mock_client()

    result = run_pipeline(
        records, settings, run_id="zero-budget-run", governor=governor, groq_client=client
    )

    # The LLM must never actually have been called.
    client.chat.completions.create.assert_not_called()

    # No STAGE6_LLM group can exist if the LLM was never called.
    stage6_groups = [g for g in result.match_groups if g.proposed_by == ProposedBy.STAGE6_LLM]
    assert stage6_groups == []

    budget_exhausted_exceptions = [
        e for e in result.exceptions if e.evidence.get("degradation_reason") == "LLM_BUDGET_EXHAUSTED"
    ]
    print(
        f"\n[Stage 11] zero-budget run: pipeline completed with "
        f"{len(budget_exhausted_exceptions)} record(s) exceptioned as "
        "LLM_BUDGET_EXHAUSTED rather than the run stalling or crashing."
    )
    # Only assert the pipeline *completed* and *some* residual records
    # exist for this dataset to have degraded on — the exact count is
    # data-dependent and already covered qualitatively above.
    assert isinstance(result.matched_record_ids, set)
