"""The judge-facing evaluation harness — ARCHITECTURE.md §11. Stage 12
of this 16-stage relay build.

This module is the ONE place in the codebase allowed to read ground
truth (``ground_truth/<dataset>/ground_truth.json``) — every earlier
stage's own docstrings state that invariant for the matching pipeline
itself (``matching/``, ``verification/``, ``llm/``), and it is
unchanged here: this module only ever *reads* ground truth, after a
completed ``matching.pipeline.run_pipeline`` call, to score that run's
output. It never feeds ground truth into the pipeline, and it does not
modify any matching/verification/LLM code.

Nine §11 judge-facing metrics, each computed by its own pure,
independently-unit-testable function below (see the "Metric functions"
section):

  1. Auto-match precision   — ``compute_auto_match_precision``
  2. Record coverage        — ``compute_record_coverage``
  3. Value coverage         — ``compute_value_coverage``
  4. False-match rate       — ``compute_false_match_rate``
  5. Exception quality      — ``compute_exception_quality``
  6. Tier contribution      — ``compute_tier_contribution``
  7. Runtime and cost       — ``compute_runtime_and_cost``
  8. Bank credit coverage        — ``compute_bank_credit_coverage``
  9. Complete cluster resolution — ``compute_complete_cluster_resolution``

Metrics 1-7 measure whether *claimed* relationships are correct — a
VERIFIED group's own membership is a genuine subset of some real
ground-truth cluster. They do not measure whether the full
bank-gateway-ledger settlement loop for a given event actually closed.
Metrics 8-9 measure closure directly, and are reported alongside 1-7
rather than as a replacement for them, since the two questions are
different and both matter: a system can have 100% auto-match precision
(everything it commits is correct) while still leaving most settlement
loops only partially closed. See each function's own docstring for its
exact definition.

All four "was this actually correct" metrics (auto-match precision,
false-match rate, tier contribution, and indirectly exception quality)
share one ground-truth-correctness check, ``is_verified_group_correct``
— the exact check already proven out in
``tests/test_stage6_pipeline_integration.py``'s
``_summarize_and_check``: a VERIFIED group's membership is correct only
if it doesn't touch a known duplicate or an honest-abstention decoy,
AND is a subset of some real ground-truth cluster (a ``match_groups``
cluster or an ``unresolved`` cluster — checking all three "wrong"
categories, not just ``match_groups``, per that test's own documented
history of false positives from a narrower check).

CLI entry point: ``python -m recon_agent.evaluation.harness --dataset
evaluation`` (the default — the evaluation/demo set is the held-out,
judge-facing set per §11's calibration/evaluation separation; the
calibration set was already spent tuning Stage 2-5 thresholds and is
never the default here).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

from recon_agent.config import Settings
from recon_agent.llm.governor import CallBudgetGovernor
from recon_agent.llm.recommender import build_prompt
from recon_agent.matching.pipeline import run_pipeline
from recon_agent.models import (
    DecisionEvent,
    DecisionStage,
    EntityType,
    ExceptionCategory,
    MatchGroup,
    MatchGroupMember,
    MatchGroupStatus,
    NormalizedRecord,
    ProposedBy,
    ReconciliationException,
    Source,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UnresolvedCase:
    """One ``ground_truth["unresolved"]`` entry — a cluster of record ids
    that the system is *supposed* to leave unresolved (e.g. a bank leg
    that never arrives in this batch), plus why."""

    record_ids: frozenset[str]
    reason: str
    categories: tuple[str, ...]


@dataclass(frozen=True)
class AbstentionCase:
    """One ``ground_truth["honest_abstention"]`` entry — a single record
    with no real counterpart at all."""

    record_id: str
    reason: str
    categories: tuple[str, ...]


@dataclass(frozen=True)
class GroundTruth:
    """The held-out truth for one dataset, loaded once per harness run.

    ``true_clusters`` is ``match_groups`` record-id sets plus
    ``unresolved`` record-id sets — exactly the population
    ``is_verified_group_correct`` checks subset-membership against,
    mirroring ``tests/test_stage6_pipeline_integration.py``.
    """

    dataset: str
    match_group_clusters: tuple[frozenset[str], ...]
    unresolved_cases: tuple[UnresolvedCase, ...]
    abstention_cases: tuple[AbstentionCase, ...]
    duplicate_ids: frozenset[str]
    true_clusters: tuple[frozenset[str], ...]
    abstention_ids: frozenset[str]
    num_logical_events: Optional[int] = None
    num_physical_records: Optional[int] = None
    category_counts: dict[str, int] = field(default_factory=dict)


def load_ground_truth(path: Path) -> GroundTruth:
    payload = json.loads(path.read_text())

    match_group_clusters = tuple(
        frozenset(g["record_ids"]) for g in payload.get("match_groups", [])
    )
    unresolved_cases = tuple(
        UnresolvedCase(
            record_ids=frozenset(u["record_ids"]),
            reason=u["reason"],
            categories=tuple(u.get("categories", ())),
        )
        for u in payload.get("unresolved", [])
    )
    abstention_cases = tuple(
        AbstentionCase(
            record_id=a["record_id"],
            reason=a["reason"],
            categories=tuple(a.get("categories", ())),
        )
        for a in payload.get("honest_abstention", [])
    )
    duplicate_ids = frozenset(d["record_id"] for d in payload.get("duplicates", []))
    abstention_ids = frozenset(a.record_id for a in abstention_cases)
    true_clusters = match_group_clusters + tuple(u.record_ids for u in unresolved_cases)

    return GroundTruth(
        dataset=payload.get("dataset", path.parent.name),
        match_group_clusters=match_group_clusters,
        unresolved_cases=unresolved_cases,
        abstention_cases=abstention_cases,
        duplicate_ids=duplicate_ids,
        true_clusters=true_clusters,
        abstention_ids=abstention_ids,
        num_logical_events=payload.get("num_logical_events"),
        num_physical_records=payload.get("num_physical_records"),
        category_counts=dict(payload.get("category_counts", {})),
    )


def load_records(path: Path) -> list[NormalizedRecord]:
    payload = json.loads(path.read_text())
    return [NormalizedRecord.model_validate(r) for r in payload]


# ---------------------------------------------------------------------------
# The shared ground-truth-correctness check (item 4's "SAME check").
# ---------------------------------------------------------------------------


def is_verified_group_correct(member_record_ids: frozenset[str], gt: GroundTruth) -> tuple[bool, str]:
    """Is a VERIFIED group's membership actually correct?

    Checks ALL THREE "wrong" ground-truth categories — not just
    ``match_groups`` — per ``test_stage6_pipeline_integration.py``'s own
    documented history ("an earlier review round found checking only
    match_groups produces false positives"):

      1. touches a known duplicate            -> wrong
      2. touches an honest-abstention decoy    -> wrong
      3. not a subset of any real true cluster (a ``match_groups``
         cluster OR an ``unresolved`` cluster) -> wrong

    Returns ``(is_correct, reason)`` — ``reason`` is ``"correct"`` or
    one of the three failure labels above, for evidence/debugging.
    """
    if member_record_ids & gt.duplicate_ids:
        return False, "includes_known_duplicate"
    if member_record_ids & gt.abstention_ids:
        return False, "includes_honest_abstention_decoy"
    if any(member_record_ids <= cluster for cluster in gt.true_clusters):
        return True, "correct"
    return False, "not_subset_of_true_cluster"


def _members_by_group(members: list[MatchGroupMember]) -> dict[str, list[MatchGroupMember]]:
    out: dict[str, list[MatchGroupMember]] = {}
    for m in members:
        out.setdefault(m.group_id, []).append(m)
    return out


def _verified_groups(match_groups: list[MatchGroup]) -> list[MatchGroup]:
    return [g for g in match_groups if g.status == MatchGroupStatus.VERIFIED]


# ---------------------------------------------------------------------------
# Metric 1 — Auto-match precision (correct auto-matches / all auto-matches).
#
# "Auto-match" means VERIFIED status specifically (§11 table, and this
# stage's own brief): Stage 6 proposals never reach VERIFIED by design
# (§2's no-auto-commit-for-LLM rule, re-proven end-to-end in
# test_stage6_pipeline_integration.py), so restricting to VERIFIED
# naturally covers only Stages 1-5's committed output, with no separate
# Stage 6 exclusion needed.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrecisionResult:
    correct: int
    total: int
    precision: Optional[float]  # None when total == 0 (nothing to divide)
    incorrect_group_ids: tuple[str, ...]


def compute_auto_match_precision(
    match_groups: list[MatchGroup],
    members_by_group: dict[str, list[MatchGroupMember]],
    gt: GroundTruth,
) -> PrecisionResult:
    verified = _verified_groups(match_groups)
    correct = 0
    incorrect_ids: list[str] = []
    for g in verified:
        member_ids = frozenset(m.record_id for m in members_by_group.get(g.group_id, []))
        ok, _reason = is_verified_group_correct(member_ids, gt)
        if ok:
            correct += 1
        else:
            incorrect_ids.append(g.group_id)
    total = len(verified)
    precision = (correct / total) if total > 0 else None
    return PrecisionResult(correct=correct, total=total, precision=precision, incorrect_group_ids=tuple(incorrect_ids))


# ---------------------------------------------------------------------------
# Metric 2 — Record coverage (matched records / eligible records).
#
# "Matched" is scoped to VERIFIED-group membership, the same population
# auto-match precision scores — this measures what fraction of the
# eligible universe the system actually auto-resolved (committed)
# without a human, not merely proposed/held for review.
#
# honest_abstention records genuinely cannot be matched (there is no
# real counterpart for them anywhere in the dataset), so counting them
# in the denominator makes 100% coverage structurally impossible and
# unfairly penalizes a system that behaves exactly as designed. Rather
# than silently excluding them (which would hide how many genuinely
# unmatchable records exist), both numbers are reported: raw coverage
# (honest reflection of "fraction of ALL records auto-resolved") and
# coverage-excluding-honest-abstention (the realistic ceiling this
# system could ever reach).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoverageResult:
    matched_record_count: int
    total_records: int
    abstention_count: int
    raw_coverage: float
    coverage_excluding_abstention: Optional[float]  # None if every record is abstention


def _verified_member_record_ids(
    match_groups: list[MatchGroup], members_by_group: dict[str, list[MatchGroupMember]]
) -> frozenset[str]:
    ids: set[str] = set()
    for g in _verified_groups(match_groups):
        ids.update(m.record_id for m in members_by_group.get(g.group_id, []))
    return frozenset(ids)


def compute_record_coverage(
    records: list[NormalizedRecord],
    match_groups: list[MatchGroup],
    members_by_group: dict[str, list[MatchGroupMember]],
    gt: GroundTruth,
) -> CoverageResult:
    matched_ids = _verified_member_record_ids(match_groups, members_by_group)
    total = len(records)
    abstention_count = len(gt.abstention_ids & {r.record_id for r in records})
    raw = (len(matched_ids) / total) if total > 0 else 0.0
    denom_excl = total - abstention_count
    excl = (len(matched_ids) / denom_excl) if denom_excl > 0 else None
    return CoverageResult(
        matched_record_count=len(matched_ids),
        total_records=total,
        abstention_count=abstention_count,
        raw_coverage=raw,
        coverage_excluding_abstention=excl,
    )


# ---------------------------------------------------------------------------
# Metric 3 — Value coverage (matched monetary value / eligible monetary
# value). Uses abs(amount_paise) uniformly for every record — the
# generalization of "absolute value for refunds and chargebacks" (§11):
# refund/reversal/chargeback rows are stored with a negative
# amount_paise (see testdata/generator.py), so summing raw signed
# amounts would let a refund silently net against its original payment
# and hide genuine reconciliation activity from the metric. Taking
# abs() of every record's amount before summing avoids that regardless
# of which entity_type produced the negative sign.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValueCoverageResult:
    matched_value_paise: int
    eligible_value_paise: int
    value_coverage: float
    matched_value_paise_excluding_abstention: int
    eligible_value_paise_excluding_abstention: int
    value_coverage_excluding_abstention: Optional[float]


def compute_value_coverage(
    records: list[NormalizedRecord],
    match_groups: list[MatchGroup],
    members_by_group: dict[str, list[MatchGroupMember]],
    gt: GroundTruth,
) -> ValueCoverageResult:
    matched_ids = _verified_member_record_ids(match_groups, members_by_group)
    eligible_value = sum(abs(r.amount_paise) for r in records)
    matched_value = sum(abs(r.amount_paise) for r in records if r.record_id in matched_ids)
    coverage = (matched_value / eligible_value) if eligible_value > 0 else 0.0

    non_abstention = [r for r in records if r.record_id not in gt.abstention_ids]
    eligible_excl = sum(abs(r.amount_paise) for r in non_abstention)
    matched_excl = sum(abs(r.amount_paise) for r in non_abstention if r.record_id in matched_ids)
    coverage_excl = (matched_excl / eligible_excl) if eligible_excl > 0 else None

    return ValueCoverageResult(
        matched_value_paise=matched_value,
        eligible_value_paise=eligible_value,
        value_coverage=coverage,
        matched_value_paise_excluding_abstention=matched_excl,
        eligible_value_paise_excluding_abstention=eligible_excl,
        value_coverage_excluding_abstention=coverage_excl,
    )


# ---------------------------------------------------------------------------
# Metric 4 — False-match rate (incorrect committed matches / committed
# matches). "Committed matches" is the same VERIFIED population as
# auto-match precision, scored with the same is_verified_group_correct
# check — so this is precision's complement by construction. Computed
# independently (not as ``1 - precision``) so each metric's own fixture
# test proves its own formula, per this stage's brief.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FalseMatchResult:
    incorrect: int
    total: int
    false_match_rate: Optional[float]
    incorrect_group_ids: tuple[str, ...]


def compute_false_match_rate(
    match_groups: list[MatchGroup],
    members_by_group: dict[str, list[MatchGroupMember]],
    gt: GroundTruth,
) -> FalseMatchResult:
    verified = _verified_groups(match_groups)
    incorrect_ids: list[str] = []
    for g in verified:
        member_ids = frozenset(m.record_id for m in members_by_group.get(g.group_id, []))
        ok, _reason = is_verified_group_correct(member_ids, gt)
        if not ok:
            incorrect_ids.append(g.group_id)
    total = len(verified)
    rate = (len(incorrect_ids) / total) if total > 0 else None
    return FalseMatchResult(
        incorrect=len(incorrect_ids), total=total, false_match_rate=rate, incorrect_group_ids=tuple(incorrect_ids)
    )


# ---------------------------------------------------------------------------
# Metric 5 — Exception quality: for records ground truth says should
# have ended up unresolved (an ``unresolved`` case) or honestly
# abstained on (a ``honest_abstention`` case), is the system's own
# exception categorization *plausible*, rather than the record being
# silently dropped (no exception at all) or silently absorbed into a
# VERIFIED group (a false match — already penalized above, but silence
# on the exception side is a distinct failure worth its own count)?
#
# §8's Exception Taxonomy has no category that maps 1:1 onto "ground
# truth reason = MISSING_SETTLEMENT" or "= NO_COUNTERPART_EXISTS" — a
# grep across matching/ confirms only six categories are ever actually
# *assigned* anywhere in this codebase (AMBIGUOUS_AGGREGATION,
# CURRENCY_MISMATCH, REUSED_COUNTERPART, PARTIAL_SETTLEMENT,
# INSUFFICIENT_EVIDENCE, LLM_UNAVAILABLE); MISSING_SETTLEMENT,
# ORPHAN_BANK_CREDIT, DUPLICATE_RECORD, FEE_GST_MISMATCH,
# REFUND_CHARGEBACK_MISMATCH, CURRENCY_MISMATCH-as-abstention-reason,
# DATE_OUTSIDE_POLICY_WINDOW, and LLM_OUTPUT_REJECTED_BY_VERIFIER are
# never assigned by any stage. So rather than overfitting to an exact
# category string the system literally cannot produce, "plausible"
# here means: the record has AT LEAST ONE system-emitted exception
# whose category is one of the six the pipeline is actually capable of
# raising (documented in _PLAUSIBLE_CATEGORIES_BY_REASON below) —
# i.e. "some exception, not silence", per this stage's own brief,
# with the category set narrowed only enough to exclude a category
# that would be a nonsensical explanation for that reason (there are
# none among the six in practice, but the table is kept explicit
# rather than an unconditional "any category" so a future taxonomy
# change that adds a genuinely unrelated category doesn't silently
# start counting as plausible here).
# ---------------------------------------------------------------------------

_ALL_EMITTED_CATEGORIES = frozenset(
    {
        ExceptionCategory.AMBIGUOUS_AGGREGATION,
        ExceptionCategory.CURRENCY_MISMATCH,
        ExceptionCategory.REUSED_COUNTERPART,
        ExceptionCategory.PARTIAL_SETTLEMENT,
        ExceptionCategory.INSUFFICIENT_EVIDENCE,
        ExceptionCategory.LLM_UNAVAILABLE,
    }
)

_PLAUSIBLE_CATEGORIES_BY_REASON: dict[str, frozenset[ExceptionCategory]] = {
    "MISSING_SETTLEMENT": _ALL_EMITTED_CATEGORIES,
    "NO_COUNTERPART_EXISTS": _ALL_EMITTED_CATEGORIES,
}
# Fallback for a ground-truth reason string not in the table above
# (e.g. a future dataset generator adds a new reason): any emitted
# category still counts as "not silence", rather than raising.
_DEFAULT_PLAUSIBLE_CATEGORIES = _ALL_EMITTED_CATEGORIES


def _referenced_record_ids(exc: ReconciliationException) -> frozenset[str]:
    """Which real record id(s) an exception is actually about.

    ``ReconciliationException.group_id_or_record_id`` is sometimes a
    bare record id (Stage 6's per-seeker exceptions), sometimes a
    MatchGroup id (verification-rejection exceptions in pipeline.py),
    and sometimes a synthetic aggregation target id (Stage 3/4's
    AMBIGUOUS_AGGREGATION exceptions) — never a record id in the
    latter two cases. The real record ids live in specific evidence
    keys instead: ``released_record_ids`` (verification rejections) or
    ``target_record_ids`` / ``involved_item_ids`` (aggregation
    ambiguity). Falls back to treating ``group_id_or_record_id`` itself
    as a record id only when none of those evidence keys are present —
    exactly the Stage 6 shape.
    """
    ev = exc.evidence
    ids: set[str] = set()
    for key in ("released_record_ids", "target_record_ids", "involved_item_ids"):
        value = ev.get(key)
        if value:
            ids.update(value)
    if not ids:
        ids.add(exc.group_id_or_record_id)
    return frozenset(ids)


def _exception_categories_by_record(
    exceptions: list[ReconciliationException],
) -> dict[str, set[ExceptionCategory]]:
    out: dict[str, set[ExceptionCategory]] = {}
    for exc in exceptions:
        for record_id in _referenced_record_ids(exc):
            out.setdefault(record_id, set()).add(exc.category)
    return out


@dataclass(frozen=True)
class ExceptionQualityResult:
    plausible: int
    total: int
    exception_quality: Optional[float]
    implausible_cases: tuple[dict[str, Any], ...]
    resolved_as_verified_match: int


def compute_exception_quality(
    match_groups: list[MatchGroup],
    members_by_group: dict[str, list[MatchGroupMember]],
    exceptions: list[ReconciliationException],
    gt: GroundTruth,
) -> ExceptionQualityResult:
    """Real-data finding that shaped this function's scope (documented
    here rather than left implicit): an ``unresolved`` ground-truth case
    can legitimately end up VERIFIED — e.g. a ledger+gateway pair ground
    truth calls "unresolved" because a THIRD (bank) leg never arrives in
    this batch, but the ledger+gateway pair's own relationship is real,
    and Stage 1-5 correctly matches just those two records into a
    VERIFIED group whose membership is an exact subset of the
    ``unresolved`` cluster — ``is_verified_group_correct`` (shared with
    false-match-rate/precision) correctly scores that as CORRECT, not a
    false positive. Re-penalizing the same outcome here as "silence" (no
    exception) would contradict that scoring: the record wasn't dropped,
    it was legitimately committed. So an ``unresolved`` case whose full
    record set ends up inside a VERIFIED group is excluded from this
    metric's denominator entirely (counted separately in
    ``resolved_as_verified_match``) — it's already credited by auto-match
    precision, and isn't "an unresolved record" in the final output
    anymore. ``honest_abstention`` cases are NOT given this exclusion:
    ``is_verified_group_correct`` never scores a group touching an
    abstention id as correct (there is no real counterpart to find), so
    an abstention record ending up VERIFIED is always a genuine false
    positive — for exception-quality purposes, still correctly showing
    no exception (it should have gotten one instead of being absorbed).
    """
    verified_ids = _verified_member_record_ids(match_groups, members_by_group)
    categories_by_record = _exception_categories_by_record(exceptions)

    plausible = 0
    total = 0
    resolved_as_verified_match = 0
    implausible: list[dict[str, Any]] = []

    for case in gt.unresolved_cases:
        if case.record_ids <= verified_ids:
            resolved_as_verified_match += 1
            continue
        total += 1
        allowed = _PLAUSIBLE_CATEGORIES_BY_REASON.get(case.reason, _DEFAULT_PLAUSIBLE_CATEGORIES)
        found_categories: set[ExceptionCategory] = set()
        for record_id in case.record_ids:
            found_categories |= categories_by_record.get(record_id, set())
        if found_categories & allowed:
            plausible += 1
        else:
            implausible.append(
                {
                    "record_ids": sorted(case.record_ids),
                    "ground_truth_reason": case.reason,
                    "found_categories": sorted(c.value for c in found_categories),
                }
            )

    for abstention in gt.abstention_cases:
        total += 1
        allowed = _PLAUSIBLE_CATEGORIES_BY_REASON.get(abstention.reason, _DEFAULT_PLAUSIBLE_CATEGORIES)
        found_categories = categories_by_record.get(abstention.record_id, set())
        if found_categories & allowed:
            plausible += 1
        else:
            implausible.append(
                {
                    "record_ids": [abstention.record_id],
                    "ground_truth_reason": abstention.reason,
                    "found_categories": sorted(c.value for c in found_categories),
                }
            )

    quality = (plausible / total) if total > 0 else None
    return ExceptionQualityResult(
        plausible=plausible,
        total=total,
        exception_quality=quality,
        implausible_cases=tuple(implausible),
        resolved_as_verified_match=resolved_as_verified_match,
    )


# ---------------------------------------------------------------------------
# Metric 6 — Tier contribution: precision + coverage broken down by
# proposing stage (STAGE1_EXACT..STAGE6_LLM) — proves the later, more
# expensive tiers add value rather than merely being present. STAGE6_LLM
# will always show 0 VERIFIED (§2's no-auto-commit rule) and therefore
# an undefined (None) precision — that's the correct, expected shape,
# not a bug in this computation.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TierStats:
    stage: str
    proposed: int
    verified: int
    correct: int
    precision: Optional[float]
    matched_record_count: int
    coverage_contribution: float  # matched_record_count / total_records


def compute_tier_contribution(
    match_groups: list[MatchGroup],
    members_by_group: dict[str, list[MatchGroupMember]],
    group_id_to_stage: dict[str, str],
    gt: GroundTruth,
    total_records: int,
) -> tuple[TierStats, ...]:
    stats: list[TierStats] = []
    for stage in ProposedBy:
        stage_groups = [g for g in match_groups if group_id_to_stage.get(g.group_id) == stage.value]
        proposed = len(stage_groups)
        verified_groups = [g for g in stage_groups if g.status == MatchGroupStatus.VERIFIED]
        correct = 0
        matched_record_count = 0
        for g in verified_groups:
            member_ids = frozenset(m.record_id for m in members_by_group.get(g.group_id, []))
            ok, _reason = is_verified_group_correct(member_ids, gt)
            if ok:
                correct += 1
            matched_record_count += len(member_ids)
        verified = len(verified_groups)
        precision = (correct / verified) if verified > 0 else None
        coverage_contribution = (matched_record_count / total_records) if total_records > 0 else 0.0
        stats.append(
            TierStats(
                stage=stage.value,
                proposed=proposed,
                verified=verified,
                correct=correct,
                precision=precision,
                matched_record_count=matched_record_count,
                coverage_contribution=coverage_contribution,
            )
        )
    return tuple(stats)


# ---------------------------------------------------------------------------
# Metric 7 — Runtime and cost: wall-clock time for the run, total Groq
# calls made (read from the governor's ledger), and a rough
# token-count-based estimated cost.
#
# Documented cost assumptions (all approximations, not measured
# actuals — Groq's chat-completions API used here does not return this
# harness a usage/token-count field to read, so this is the "token-
# count-based approximation" the brief explicitly allows):
#   - Input tokens: reconstruct the exact (system, user) prompt text
#     Stage 6 would have sent for each real Groq call attempt (via
#     llm.recommender.build_prompt, read-only reuse — not modified),
#     then approximate tokens as len(text) / CHARS_PER_TOKEN (a
#     standard rough heuristic for English text; real tokenizers vary).
#   - Output tokens: a fixed ASSUMED_OUTPUT_TOKENS_PER_CALL, since the
#     response schema (candidate_id/confidence/reason_code/explanation)
#     is small and bounded (recommender.py's own prompt caps
#     explanation to "one or two sentences").
#   - Pricing: Settings.groq_model's default
#     ("meta-llama/llama-4-scout-17b-16e-instruct") priced at Groq's
#     published $0.11 / MTok input, $0.34 / MTok output — current as of
#     this stage's build; update PRICE_PER_MTOK_INPUT/OUTPUT below if
#     Groq's list price or the configured model changes.
#   - Call count: read from CallBudgetGovernor's own ledger
#     (``governor.status(run_id).calls_used_today``), not re-derived
#     from decision events — per this stage's brief ("total Groq calls
#     made, read from the governor's ledger, already built"). The
#     per-call size average is applied to that authoritative count.
# ---------------------------------------------------------------------------

CHARS_PER_TOKEN = 4.0
ASSUMED_OUTPUT_TOKENS_PER_CALL = 120.0
PRICE_PER_MTOK_INPUT = 0.11
PRICE_PER_MTOK_OUTPUT = 0.34

_DEGRADED_OR_SKIPPED_REASON_CODES = frozenset(
    {"NO_CANDIDATES_AVAILABLE", "LLM_NOT_CONFIGURED", "LLM_BUDGET_EXHAUSTED", "LLM_CIRCUIT_BREAKER_OPEN"}
)


@dataclass(frozen=True)
class RuntimeCostResult:
    wall_clock_seconds: float
    records_per_second: float
    groq_calls_made: int
    reconstructed_call_events: int
    estimated_avg_input_tokens_per_call: float
    estimated_total_input_tokens: float
    estimated_total_output_tokens: float
    estimated_cost_usd: float
    cost_assumptions: dict[str, Any]


def _real_call_decision_events(decision_events: list[DecisionEvent]) -> list[DecisionEvent]:
    """Stage 6 decision events that correspond to an actual attempted
    Groq API call (i.e. the governor allowed it) — everything except
    the four degradation/no-candidate reason codes, which never reach
    ``get_llm_recommendation`` at all (see stage6_llm.py)."""
    return [
        de
        for de in decision_events
        if de.stage == DecisionStage.STAGE6_LLM and de.reason_code not in _DEGRADED_OR_SKIPPED_REASON_CODES
    ]


def compute_runtime_and_cost(
    wall_clock_seconds: float,
    total_records: int,
    decision_events: list[DecisionEvent],
    records_by_id: dict[str, NormalizedRecord],
    groq_calls_made: int,
) -> RuntimeCostResult:
    call_events = _real_call_decision_events(decision_events)

    per_call_chars: list[int] = []
    for de in call_events:
        seeker_id = de.candidate_scores.get("seeker_record_id")
        candidate_ids = de.candidate_scores.get("offered_candidate_ids", [])
        seeker = records_by_id.get(seeker_id) if seeker_id else None
        if seeker is None:
            continue
        candidates = [records_by_id[c] for c in candidate_ids if c in records_by_id]
        system_prompt, user_message = build_prompt(seeker, candidates)
        per_call_chars.append(len(system_prompt) + len(user_message))

    avg_input_tokens = (
        (sum(per_call_chars) / len(per_call_chars)) / CHARS_PER_TOKEN if per_call_chars else 0.0
    )

    total_input_tokens = avg_input_tokens * groq_calls_made
    total_output_tokens = ASSUMED_OUTPUT_TOKENS_PER_CALL * groq_calls_made
    estimated_cost = (
        (total_input_tokens / 1_000_000) * PRICE_PER_MTOK_INPUT
        + (total_output_tokens / 1_000_000) * PRICE_PER_MTOK_OUTPUT
    )

    return RuntimeCostResult(
        wall_clock_seconds=wall_clock_seconds,
        records_per_second=(total_records / wall_clock_seconds) if wall_clock_seconds > 0 else 0.0,
        groq_calls_made=groq_calls_made,
        reconstructed_call_events=len(call_events),
        estimated_avg_input_tokens_per_call=avg_input_tokens,
        estimated_total_input_tokens=total_input_tokens,
        estimated_total_output_tokens=total_output_tokens,
        estimated_cost_usd=estimated_cost,
        cost_assumptions={
            "chars_per_token": CHARS_PER_TOKEN,
            "assumed_output_tokens_per_call": ASSUMED_OUTPUT_TOKENS_PER_CALL,
            "price_per_mtok_input_usd": PRICE_PER_MTOK_INPUT,
            "price_per_mtok_output_usd": PRICE_PER_MTOK_OUTPUT,
            "note": (
                "Rough approximation: input tokens reconstructed from the "
                "real Stage 6 prompt text via len(text)/chars_per_token; "
                "output tokens are a fixed per-call assumption, not "
                "measured. groq_calls_made is the authoritative count "
                "(from CallBudgetGovernor's ledger); the average per-call "
                "size from reconstructed prompts is applied to it."
            ),
        },
    )


# ---------------------------------------------------------------------------
# Metric 8 — Bank credit coverage: of all BANK-source, entity_type=
# BANK_CREDIT records that are part of a real ground-truth match_group
# (i.e. genuinely have a counterpart somewhere in the batch — this
# deliberately excludes ``unresolved`` clusters, whose whole point is
# that a leg is *not* expected to arrive), what fraction end up in SOME
# VERIFIED group (not necessarily a correct or complete one — this
# metric is about whether the bank leg was picked up at all, not
# whether the group it landed in is right; auto-match precision/
# false-match rate already score correctness separately).
#
# Bugfix (found against real hand-crafted data, not synthetic
# calibration/evaluation data — see BUILD_LOG.md): this metric is named
# "bank credit coverage", but the eligibility filter used to be "any
# BANK-source record", not "BANK-source records that are genuinely
# BANK_CREDIT". A BANK-source REFUND/CHARGEBACK/REVERSAL leg (the
# bank-side reflection of a refund/chargeback/reversal event — see
# testdata/generator.py's own "one real money movement stored
# identically on every source leg" design note) is a real BANK-source
# record, but it is not a credit landing in the account; counting it
# here inflated the denominator with rows the metric's own name doesn't
# describe, distorting what the reported percentage actually means.
# Narrowed to ``entity_type == BANK_CREDIT`` specifically so the metric
# measures exactly what its name says.
#
# This is a settlement-closure metric, not a correctness metric: a
# system can score 100% here while still being wrong about which other
# records a bank credit was grouped with. It exists because "was every
# real bank credit even touched" is a distinct, honesty-relevant
# question from "was every touch correct".
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BankCreditCoverageResult:
    matched: int
    total: int
    bank_credit_coverage: Optional[float]  # None when total == 0
    unmatched_record_ids: tuple[str, ...]


def compute_bank_credit_coverage(
    records: list[NormalizedRecord],
    match_groups: list[MatchGroup],
    members_by_group: dict[str, list[MatchGroupMember]],
    gt: GroundTruth,
) -> BankCreditCoverageResult:
    verified_ids = _verified_member_record_ids(match_groups, members_by_group)
    gt_match_group_ids: set[str] = set()
    for cluster in gt.match_group_clusters:
        gt_match_group_ids.update(cluster)

    eligible_bank_ids = frozenset(
        r.record_id
        for r in records
        if r.source == Source.BANK
        and r.entity_type == EntityType.BANK_CREDIT
        and r.record_id in gt_match_group_ids
    )
    matched_ids = frozenset(rid for rid in eligible_bank_ids if rid in verified_ids)
    total = len(eligible_bank_ids)
    matched = len(matched_ids)
    coverage = (matched / total) if total > 0 else None
    return BankCreditCoverageResult(
        matched=matched,
        total=total,
        bank_credit_coverage=coverage,
        unmatched_record_ids=tuple(sorted(eligible_bank_ids - matched_ids)),
    )


# ---------------------------------------------------------------------------
# Metric 9 — Complete cluster resolution: of all ground-truth
# match_groups, what fraction are covered EXACTLY — not merely as a
# subset — by a single VERIFIED group's membership.
#
# A VERIFIED group whose members are a strict subset of a real cluster
# already scores as "correct" under ``is_verified_group_correct`` (and
# so counts toward auto-match precision), but that is a *partial*
# settlement: some real counterpart record for that event is still
# sitting outside the group, unresolved. This metric is the honesty
# check on top of precision — it only credits a cluster when one
# VERIFIED group's membership equals the full ground-truth cluster,
# record for record, i.e. the settlement loop for that event is fully
# closed, not just partially and correctly matched.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompleteClusterResolutionResult:
    exact: int
    total: int
    complete_cluster_resolution: Optional[float]  # None when total == 0
    unresolved_cluster_record_ids: tuple[tuple[str, ...], ...]


def compute_complete_cluster_resolution(
    match_groups: list[MatchGroup],
    members_by_group: dict[str, list[MatchGroupMember]],
    gt: GroundTruth,
) -> CompleteClusterResolutionResult:
    verified_member_sets = frozenset(
        frozenset(m.record_id for m in members_by_group.get(g.group_id, []))
        for g in _verified_groups(match_groups)
    )
    exact = 0
    unresolved: list[tuple[str, ...]] = []
    for cluster in gt.match_group_clusters:
        if cluster in verified_member_sets:
            exact += 1
        else:
            unresolved.append(tuple(sorted(cluster)))
    total = len(gt.match_group_clusters)
    rate = (exact / total) if total > 0 else None
    return CompleteClusterResolutionResult(
        exact=exact,
        total=total,
        complete_cluster_resolution=rate,
        unresolved_cluster_record_ids=tuple(unresolved),
    )


# ---------------------------------------------------------------------------
# The full report
# ---------------------------------------------------------------------------


def _asdict(obj: Any) -> Any:
    """A tiny, enum/tuple/frozenset-safe dataclass -> dict converter
    (``dataclasses.asdict`` chokes on nothing here, but this keeps enum
    values as their plain string form rather than ``Enum.X`` repr, and
    tuples/frozensets as sorted lists, for clean JSON export)."""
    if hasattr(obj, "__dataclass_fields__"):
        return {f.name: _asdict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [_asdict(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(_asdict(v) for v in obj)
    if isinstance(obj, dict):
        return {k: _asdict(v) for k, v in obj.items()}
    if hasattr(obj, "value") and hasattr(obj, "name") and not isinstance(obj, (str, int, float, bool)):
        return obj.value  # Enum
    return obj


@dataclass(frozen=True)
class EvaluationReport:
    dataset: str
    total_records: int
    dataset_info: dict[str, Any]
    auto_match_precision: PrecisionResult
    record_coverage: CoverageResult
    value_coverage: ValueCoverageResult
    false_match_rate: FalseMatchResult
    exception_quality: ExceptionQualityResult
    tier_contribution: tuple[TierStats, ...]
    runtime_and_cost: RuntimeCostResult
    bank_credit_coverage: BankCreditCoverageResult
    complete_cluster_resolution: CompleteClusterResolutionResult
    verified_group_count: int
    pending_review_group_count: int
    rejected_group_count: int

    def to_dict(self) -> dict[str, Any]:
        return _asdict(self)

    def render_human_readable(self) -> str:
        lines: list[str] = []
        a = lines.append
        a(f"Evaluation report — dataset: {self.dataset}")
        a("=" * 60)
        a(
            f"Records: {self.total_records}  |  "
            f"logical events: {self.dataset_info.get('num_logical_events')}"
        )
        a(
            f"Groups: {self.verified_group_count} VERIFIED, "
            f"{self.pending_review_group_count} PENDING_REVIEW, "
            f"{self.rejected_group_count} REJECTED"
        )
        a("")
        p = self.auto_match_precision
        a(
            f"1. Auto-match precision: "
            f"{_pct(p.precision)}  ({p.correct}/{p.total} VERIFIED groups correct)"
        )
        c = self.record_coverage
        a(
            f"2. Record coverage: raw {_pct(c.raw_coverage)} "
            f"({c.matched_record_count}/{c.total_records}), "
            f"excl. honest-abstention {_pct(c.coverage_excluding_abstention)} "
            f"(abstention records: {c.abstention_count})"
        )
        v = self.value_coverage
        a(
            f"3. Value coverage: {_pct(v.value_coverage)} "
            f"({v.matched_value_paise} / {v.eligible_value_paise} paise), "
            f"excl. honest-abstention {_pct(v.value_coverage_excluding_abstention)}"
        )
        f_ = self.false_match_rate
        a(f"4. False-match rate: {_pct(f_.false_match_rate)}  ({f_.incorrect}/{f_.total} VERIFIED groups wrong)")
        e = self.exception_quality
        a(
            f"5. Exception quality: {_pct(e.exception_quality)}  ({e.plausible}/{e.total} plausible; "
            f"{e.resolved_as_verified_match} unresolved-case(s) were instead legitimately resolved "
            "into a correct VERIFIED match and excluded from this metric — see docstring)"
        )
        a("6. Tier contribution:")
        for t in self.tier_contribution:
            a(
                f"   {t.stage:<20} proposed={t.proposed:<4} verified={t.verified:<4} "
                f"precision={_pct(t.precision):<7} "
                f"records_matched={t.matched_record_count:<4} "
                f"coverage_contribution={_pct(t.coverage_contribution)}"
            )
        r = self.runtime_and_cost
        a(
            f"7. Runtime and cost: {r.wall_clock_seconds:.3f}s "
            f"({r.records_per_second:.1f} records/sec), "
            f"{r.groq_calls_made} Groq call(s), "
            f"est. cost ${r.estimated_cost_usd:.6f} "
            f"(~{r.estimated_total_input_tokens:.0f} input + "
            f"{r.estimated_total_output_tokens:.0f} output tokens, rough approximation)"
        )
        a("")
        a(
            "Settlement-closure honesty note: metrics 1-4 above measure "
            "whether CLAIMED relationships are correct — not whether the "
            "full bank-gateway-ledger settlement loop actually closed. "
            "The two metrics below measure closure directly and should "
            "always be read alongside the precision/false-match numbers "
            "above, never in place of them."
        )
        b = self.bank_credit_coverage
        a(
            f"8. Bank credit coverage: {_pct(b.bank_credit_coverage)}  "
            f"({b.matched}/{b.total} real BANK-side credits ended up in "
            "some VERIFIED group)"
        )
        cc = self.complete_cluster_resolution
        a(
            f"9. Complete cluster resolution: {_pct(cc.complete_cluster_resolution)}  "
            f"({cc.exact}/{cc.total} ground-truth settlement clusters closed "
            "EXACTLY by a single VERIFIED group, not merely as a correct "
            "partial subset)"
        )
        return "\n".join(lines)


def _pct(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


# ---------------------------------------------------------------------------
# Orchestration — run the real pipeline, then score it.
# ---------------------------------------------------------------------------


def run_evaluation(
    dataset: str,
    data_root: Path = REPO_ROOT,
    settings: Optional[Settings] = None,
    governor_db_path: Optional[str] = None,
    run_id: Optional[str] = None,
    groq_client: object = None,
) -> EvaluationReport:
    """Run the full real pipeline against ``dataset``'s records, then
    score the result against that dataset's held-out ground truth.

    ``groq_client`` defaults to ``None`` — i.e. a real, unmocked run:
    Stage 6 calls the real Groq API if ``settings.groq_api_key`` (or
    the ``GROQ_API_KEY`` environment variable) is configured, and
    degrades gracefully (§14) if it isn't. Pass a mock client (e.g. the
    same heuristic mock ``test_stage6_pipeline_integration.py`` uses)
    to evaluate Stage 6 without live Groq credentials.
    """
    records_path = data_root / "data" / dataset / "records.json"
    ground_truth_path = data_root / "ground_truth" / dataset / "ground_truth.json"
    if not records_path.exists():
        raise FileNotFoundError(
            f"No records found at {records_path} — run "
            f"`python -m recon_agent.testdata.generator --seed {dataset}` first."
        )
    if not ground_truth_path.exists():
        raise FileNotFoundError(f"No ground truth found at {ground_truth_path}.")

    records = load_records(records_path)
    records_by_id = {r.record_id: r for r in records}
    gt = load_ground_truth(ground_truth_path)

    settings = settings if settings is not None else Settings.from_env()
    run_id = run_id if run_id is not None else uuid.uuid4().hex
    governor = CallBudgetGovernor(settings, db_path=governor_db_path)

    start = time.perf_counter()
    result = run_pipeline(records, settings, run_id=run_id, governor=governor, groq_client=groq_client)
    elapsed = time.perf_counter() - start

    groq_calls_made = governor.status(run_id).calls_used_today
    governor.close()

    members_by_group = _members_by_group(result.match_group_members)

    precision = compute_auto_match_precision(result.match_groups, members_by_group, gt)
    coverage = compute_record_coverage(records, result.match_groups, members_by_group, gt)
    value_coverage = compute_value_coverage(records, result.match_groups, members_by_group, gt)
    false_match = compute_false_match_rate(result.match_groups, members_by_group, gt)
    exception_quality = compute_exception_quality(result.match_groups, members_by_group, result.exceptions, gt)
    tier_contribution = compute_tier_contribution(
        result.match_groups, members_by_group, result.group_id_to_stage, gt, len(records)
    )
    runtime_cost = compute_runtime_and_cost(elapsed, len(records), result.decision_events, records_by_id, groq_calls_made)
    bank_credit_coverage = compute_bank_credit_coverage(records, result.match_groups, members_by_group, gt)
    complete_cluster_resolution = compute_complete_cluster_resolution(result.match_groups, members_by_group, gt)

    verified_count = sum(1 for g in result.match_groups if g.status == MatchGroupStatus.VERIFIED)
    pending_count = sum(1 for g in result.match_groups if g.status == MatchGroupStatus.PENDING_REVIEW)
    rejected_count = sum(1 for g in result.match_groups if g.status == MatchGroupStatus.REJECTED)

    return EvaluationReport(
        dataset=dataset,
        total_records=len(records),
        dataset_info={
            "num_logical_events": gt.num_logical_events,
            "num_physical_records": gt.num_physical_records,
            "category_counts": gt.category_counts,
            "groq_configured": bool(groq_client is not None or settings.groq_api_key),
        },
        auto_match_precision=precision,
        record_coverage=coverage,
        value_coverage=value_coverage,
        false_match_rate=false_match,
        exception_quality=exception_quality,
        tier_contribution=tier_contribution,
        runtime_and_cost=runtime_cost,
        bank_credit_coverage=bank_credit_coverage,
        complete_cluster_resolution=complete_cluster_resolution,
        verified_group_count=verified_count,
        pending_review_group_count=pending_count,
        rejected_group_count=rejected_count,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full reconciliation pipeline against a named dataset and "
            "report the §11 judge-facing evaluation metrics against its held-out "
            "ground truth."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=["calibration", "evaluation"],
        default="evaluation",
        help=(
            "Which dataset to evaluate against. Default: evaluation — the "
            "held-out, judge-facing set (calibration was already spent tuning "
            "thresholds; see ARCHITECTURE.md §11)."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write the JSON export. Default: reports/<dataset>_evaluation_report.json",
    )
    parser.add_argument(
        "--data-root",
        default=str(REPO_ROOT),
        help="Repo root containing data/ and ground_truth/. Default: this repo.",
    )
    args = parser.parse_args(argv)

    data_root = Path(args.data_root)
    report = run_evaluation(args.dataset, data_root=data_root)

    print(report.render_human_readable())

    output_path = (
        Path(args.output) if args.output else data_root / "reports" / f"{args.dataset}_evaluation_report.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.to_dict(), indent=2))
    print(f"\nJSON report written to {output_path}")


if __name__ == "__main__":
    main()
