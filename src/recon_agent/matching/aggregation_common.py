"""Shared bounded subset-sum search for Stage 3 (aggregate) and Stage 4
(adjustment) — ARCHITECTURE.md §4, §9.

Both stages need the same primitive: given a pool of candidate items and
a pool of targets, find, per target, a bounded subset of items whose
amounts sum within tolerance of the target — then decline (never
silently resolve) any target whose winning subset shares an item with
another target's winning subset. That is what "partition into disjoint
windows before searching" (§4) is actually protecting against: a
per-target subset-sum is only trustworthy when candidate pools can't
secretly overlap.

Each target's own *window* is simply its eligible candidate pool — the
items the caller-supplied ``eligible`` predicate says could plausibly
belong to it (by currency + a bounded settlement-date lag, per §4's
scope note that there's no merchant/account field to key on instead).
Computing that window is what happens "before searching": the pool is
fixed first, then the bounded subset-sum search runs only inside it.
Overlap between two different targets' windows is not a windowing
failure to avoid by cleverer partitioning — it is data, and it is
exactly the "genuine overlap" case this module exists to catch by
comparing every target's winning subset against every other's.

(A single global connected-components partition was tried and rejected:
in this dataset, consolidated-settlement batches are carved out of one
globally date-sorted list with no forced calendar gap between them, so
transitive date-proximity chains the entire dataset into one
computationally intractable window. Scoping each target to its own
local eligible pool avoids that without weakening the overlap check,
which is still computed globally across every target.)
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AggregationItem:
    """One element of the "many" side of a many-to-one (or one-to-many)
    aggregation — e.g. a settlement's net amount (Stage 3) or a single
    role-tagged record's signed amount (Stage 4)."""

    item_id: str
    record_ids: tuple[str, ...]
    amount_paise: int
    currency: str
    window_date: date


@dataclass(frozen=True)
class AggregationTarget:
    """The "one" side — e.g. a consolidated bank credit (Stage 3) or the
    counterpart amount a role-netted group must land within tolerance of
    (Stage 4)."""

    target_id: str
    record_ids: tuple[str, ...]
    amount_paise: int
    currency: str
    window_date: date


@dataclass
class SubsetCandidate:
    target_id: str
    item_ids: tuple[str, ...]
    subset_sum: int
    diff: int
    score: float


@dataclass
class TargetOutcome:
    target: AggregationTarget
    accepted: Optional[SubsetCandidate] = None
    runner_up_score: Optional[float] = None
    num_eligible_subsets: int = 0
    num_eligible_items: int = 0
    declined_reason: Optional[str] = None  # set when accepted is None
    # This target's own top-ranked candidate subset, if it had any eligible
    # candidate at all -- set regardless of accepted/declined_reason (in
    # particular, also set for AMBIGUOUS_OVERLAP declines, where `accepted`
    # is never set). This is exactly the same value this module already
    # computes internally (as `tentative[target.target_id]`) to decide
    # whether an overlap conflict exists in the first place; it's exposed
    # here, unchanged, so a caller wiring in a downstream conflict resolver
    # (see stage3_aggregate.py) can use the real computed score instead of
    # a placeholder. Exposing it does not change any accept/decline
    # decision this module makes.
    tentative: Optional[SubsetCandidate] = None
    # This target's own second-best candidate score (0.0 if it had only
    # one candidate), computed the same way the "accepted" branch below
    # computes `runner_up_score` -- exposed for every target with at least
    # one eligible candidate, not just accepted ones, for the same reason
    # as `tentative` above.
    tentative_runner_up_score: Optional[float] = None


@dataclass
class WindowResult:
    """Kept as the overall result container name for continuity with
    Stage 3/4's callers; "window" here means the full set of targets
    processed together for global overlap detection, not a single merged
    candidate pool — each target searches only its own eligible items
    (see module docstring).
    """

    outcomes: list[TargetOutcome] = field(default_factory=list)
    ambiguous_target_ids: set[str] = field(default_factory=set)
    ambiguous_item_ids: set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Bounded, exact subset-sum search for one target's eligible pool
# ---------------------------------------------------------------------------


def _score_subset(target_amount: int, subset_sum: int, tolerance: int) -> Optional[float]:
    diff = abs(target_amount - subset_sum)
    if diff > tolerance:
        return None
    if tolerance <= 0:
        return 1.0
    return 1.0 - 0.5 * min(1.0, diff / tolerance)


def _candidates_for_target(
    target: AggregationTarget,
    items: list[AggregationItem],
    max_group_size: int,
    tolerance_fn: Callable[[int], int],
    deadline: Optional[float],
) -> tuple[list[SubsetCandidate], bool]:
    """Returns (candidates sorted best-first, timed_out)."""
    tolerance = tolerance_fn(target.amount_paise)
    found: list[SubsetCandidate] = []
    n = len(items)
    for size in range(1, min(max_group_size, n) + 1):
        for combo in itertools.combinations(items, size):
            if deadline is not None and time.monotonic() > deadline:
                found.sort(key=lambda c: (-c.score, c.item_ids))
                return found, True
            subset_sum = sum(it.amount_paise for it in combo)
            score = _score_subset(target.amount_paise, subset_sum, tolerance)
            if score is None:
                continue
            item_ids = tuple(sorted(it.item_id for it in combo))
            found.append(
                SubsetCandidate(
                    target_id=target.target_id,
                    item_ids=item_ids,
                    subset_sum=subset_sum,
                    diff=abs(target.amount_paise - subset_sum),
                    score=score,
                )
            )
    # Deterministic ordering: best score first; ties broken by lowest
    # item_ids tuple (per §9's "lowest record_id wins ties").
    found.sort(key=lambda c: (-c.score, c.item_ids))
    return found, False


def run_aggregation_search(
    items: list[AggregationItem],
    targets: list[AggregationTarget],
    eligible: Callable[[AggregationItem, AggregationTarget], bool],
    max_candidates_per_window: int,
    max_group_size: int,
    window_timeout_seconds: float,
    tolerance_fn: Callable[[int], int],
    threshold: float,
    min_margin: float,
) -> WindowResult:
    """For every target, compute its own eligible candidate pool (its
    "window", fixed before any search runs), run the bounded subset-sum
    search inside it, then decline — globally, across every target, not
    just within one target's pool — any target whose winning subset
    shares an item with another target's winning subset.
    """
    result = WindowResult()
    if not targets:
        return result

    deadline = time.monotonic() + window_timeout_seconds if window_timeout_seconds > 0 else None

    per_target_candidates: dict[str, list[SubsetCandidate]] = {}
    per_target_pool_size: dict[str, int] = {}
    declined_upfront: dict[str, str] = {}

    for target in targets:
        eligible_items = [it for it in items if eligible(it, target)]
        per_target_pool_size[target.target_id] = len(eligible_items)

        if len(eligible_items) > max_candidates_per_window:
            declined_upfront[target.target_id] = "CANDIDATE_POOL_TOO_LARGE"
            per_target_candidates[target.target_id] = []
            continue

        candidates, timed_out = _candidates_for_target(
            target, eligible_items, max_group_size, tolerance_fn, deadline
        )
        if timed_out:
            declined_upfront[target.target_id] = "WINDOW_TIMEOUT"
            per_target_candidates[target.target_id] = []
            continue
        per_target_candidates[target.target_id] = candidates

    # Tentative best pick per target (only for targets not already
    # declined upfront).
    tentative: dict[str, SubsetCandidate] = {}
    for target in targets:
        if target.target_id in declined_upfront:
            continue
        candidates = per_target_candidates[target.target_id]
        if candidates:
            tentative[target.target_id] = candidates[0]

    # Detect genuine overlaps GLOBALLY: an item claimed by more than one
    # target's tentative best subset. Every target sharing that item is
    # declined, not just one of them — there's no principled way to pick
    # a winner between two independently-justified targets.
    item_to_targets: dict[str, set[str]] = {}
    for target_id, cand in tentative.items():
        for item_id in cand.item_ids:
            item_to_targets.setdefault(item_id, set()).add(target_id)

    conflicted_targets: set[str] = set()
    conflicted_items: set[str] = set()
    for item_id, target_ids in item_to_targets.items():
        if len(target_ids) > 1:
            conflicted_targets |= target_ids
            conflicted_items.add(item_id)

    result.ambiguous_item_ids = conflicted_items

    for target in targets:
        candidates = per_target_candidates.get(target.target_id, [])
        outcome = TargetOutcome(
            target=target,
            num_eligible_subsets=len(candidates),
            num_eligible_items=per_target_pool_size.get(target.target_id, 0),
        )
        if candidates:
            outcome.tentative = candidates[0]
            outcome.tentative_runner_up_score = candidates[1].score if len(candidates) > 1 else 0.0

        if target.target_id in declined_upfront:
            outcome.declined_reason = declined_upfront[target.target_id]
            result.ambiguous_target_ids.add(target.target_id)
            result.outcomes.append(outcome)
            continue

        if target.target_id in conflicted_targets:
            outcome.declined_reason = "AMBIGUOUS_OVERLAP"
            result.ambiguous_target_ids.add(target.target_id)
            result.outcomes.append(outcome)
            continue

        if not candidates:
            outcome.declined_reason = "NO_AGGREGATION_FOUND"
            result.outcomes.append(outcome)
            continue

        best = candidates[0]
        runner_up = candidates[1].score if len(candidates) > 1 else 0.0
        margin = best.score - runner_up
        outcome.runner_up_score = runner_up

        if best.score < threshold:
            outcome.declined_reason = "BELOW_THRESHOLD"
        elif margin < min_margin:
            outcome.declined_reason = "INSUFFICIENT_MARGIN"
        else:
            outcome.accepted = best

        result.outcomes.append(outcome)

    return result
