"""Global Conflict Resolver — ARCHITECTURE.md §4 ("1:1 conflicts" row) —
wired into stage3_aggregate.py's aggregation search (Stage 8).

Stage 3's aggregation search (``aggregation_common.run_aggregation_search``)
already detects the case this module targets: two different targets (e.g.
two unmatched bank credits) both independently pick the same item (e.g. the
same settlement-unit) as their best-fit candidate. Today it declines both
targets honestly as ``AMBIGUOUS_AGGREGATION`` / ``AMBIGUOUS_OVERLAP`` rather
than guessing (see stage3_aggregate.py's module docstring) — a
priority-ordered greedy pick would be order-dependent and could miss the
globally best assignment. This module is the real fix for that: a genuine
assignment problem (multiple targets, overlapping candidate pools, no way to
know from a single target's perspective who should win) solved globally and
optimally via the Hungarian algorithm (``scipy.optimize.linear_sum_
assignment``), exactly as ARCHITECTURE.md §4's table prescribes for "1:1
conflicts".

Scope, per §4's corrected table (Hungarian is one row of that table, "Many-
to-one aggregation" is a *different* row handled by disjoint-windowed
subset-sum, not by this module):

  This module ONLY resolves the case that reduces cleanly to a bipartite
  1:1 assignment — every competing option is a single atomic item, so
  "target A gets item X" and "target B gets item Y" (X != Y) are always
  mutually compatible, and picking the globally optimal such assignment is
  exactly what the Hungarian algorithm computes.

  It deliberately does NOT attempt arbitrary N:M resolution. A conflict
  fails to reduce to 1:1 when at least one competing option is itself a
  multi-item *combination* (e.g. a subset-sum aggregation match — several
  settlement units summed to satisfy one target's amount). Two such
  bundles can look like "different options" (different bundle ids) while
  still sharing an underlying atomic item, so bundle-distinctness does not
  imply compatibility the way single-item-distinctness does — modeling
  that correctly is a set-packing problem, and §1/§4 explicitly declare
  arbitrary many-to-many resolution **unsupported** rather than
  approximated with an unreliable heuristic. Any conflict containing a
  multi-item option is therefore declined outright as unresolved here,
  matching how the rest of the codebase already documents this same
  N:M-unsupported scope (see stage3_aggregate.py, aggregation_common.py).

This module is pure: it takes plain conflict data in and returns plain
resolution data out. No dependency on the pipeline, the database, or
MatchGroup/NormalizedRecord objects, so the next stage (wiring this into
stage3_aggregate.py's AMBIGUOUS_OVERLAP path) can adapt the input/output
shape to real pipeline objects without this module changing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable, Optional, Union

import numpy as np
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------------------
# Input/output shape
# ---------------------------------------------------------------------------

# One competing edge in a conflict: (target_id, candidate_option, score).
# `candidate_option` is normally a plain id string for a single atomic item
# (following the existing codebase convention, e.g. "item:group:<gid>" /
# "item:rec:<record_id>" from aggregation_common.py). It may instead be a
# frozenset/tuple/list of such ids to represent a multi-item combination —
# that is the signal this module uses to detect the "genuinely tangled N:M,
# decline rather than force" case (see module docstring).
ConflictOption = tuple[str, Union[str, frozenset, tuple, list], float]

# A conflict is any collection of competing edges that share at least one
# contested item — callers (e.g. the next stage's pipeline-wiring code) are
# responsible for grouping raw AMBIGUOUS_OVERLAP evidence into these
# clusters; this module does not do that grouping itself.
Conflict = Iterable[ConflictOption]

# Sentinel cost for a (target, candidate) pair that was never actually
# scored (no edge in the input). Large enough that the solver only ever
# uses it when structurally forced to (see _build_cost_matrix); any
# assignment that ends up using it is recognized after solving and treated
# as "no real option available" rather than a genuine win (see
# _solve_bipartite).
_SENTINEL_COST = 1e9

# Deterministic tie-break epsilon (§4: "lowest record_id wins ties"). Added
# to real-edge costs only, scaled by each row/column's position in the
# lexicographically-sorted target/candidate id ordering, so that among
# assignments with otherwise-identical total score, the solver prefers the
# one that favors lower target/candidate ids. Small enough to never change
# the outcome between two options with genuinely different scores.
_TIEBREAK_EPSILON = 1e-9


@dataclass(frozen=True)
class ConflictResolution:
    """Result of resolving one conflict.

    - ``resolved=False``: the conflict did not reduce to a clean bipartite
      1:1 assignment (see ``decline_reason``); ``assignments`` is empty and
      nothing here should be treated as a recommendation.
    - ``resolved=True``: ``assignments`` maps each winning target_id to the
      candidate_option (single-item id) it was awarded, globally optimal
      with respect to the input scores. ``unresolved_targets`` holds any
      target(s) in this same conflict that structurally had no real option
      left to win (e.g. more targets than distinct contested items) — this
      is a legitimate partial outcome, not a forced guess, since every
      target in it either won its optimal item or genuinely had none
      available.
    """

    resolved: bool
    assignments: dict[str, str]
    unresolved_targets: frozenset[str]
    decline_reason: Optional[str]
    target_ids: frozenset[str]
    candidate_ids: frozenset[str]


# ---------------------------------------------------------------------------
# Reducibility check
# ---------------------------------------------------------------------------


def _atomic_items(candidate_option: Union[str, frozenset, tuple, list]) -> frozenset:
    """Normalize a candidate_option to the set of atomic item ids it would
    consume if assigned. A plain id is a single-item set (itself); a
    collection is treated as a multi-item combination.
    """
    if isinstance(candidate_option, (frozenset, set, tuple, list)):
        return frozenset(candidate_option)
    return frozenset({candidate_option})


def _check_reducible(options: list[ConflictOption]) -> Optional[str]:
    """Returns None if the conflict is a clean bipartite 1:1 case, else a
    short decline reason code.
    """
    if not options:
        return "EMPTY_CONFLICT"

    score_by_pair: dict[tuple[str, str], float] = {}
    for target_id, candidate_option, score in options:
        items = _atomic_items(candidate_option)
        if len(items) != 1:
            # A multi-item bundle -- this is aggregation (subset-sum)
            # territory, not a direct 1:1 preference conflict. Two bundles
            # with different candidate_option ids could still share an
            # atomic item, which the bipartite "different column = safe to
            # assign both" assumption cannot see -- decline rather than
            # risk a double-claim. See module docstring.
            return "MULTI_ITEM_BUNDLE_NOT_1_TO_1"

        candidate_id = next(iter(items))
        key = (target_id, candidate_id)
        if key in score_by_pair and score_by_pair[key] != score:
            # Same target/candidate pair scored two different ways --
            # inconsistent input, can't build a well-defined cost matrix.
            return "INCONSISTENT_SCORES"
        score_by_pair[key] = score

    return None


# ---------------------------------------------------------------------------
# Cost matrix construction + solve
# ---------------------------------------------------------------------------


def _build_cost_matrix(
    target_ids: list[str],
    candidate_ids: list[str],
    score_by_pair: dict[tuple[str, str], float],
) -> np.ndarray:
    """Builds a (targets x candidates) cost matrix. Rows/cols are expected
    to already be sorted ascending (lowest id first) by the caller, so that
    row/col index doubles as the "id rank" used for deterministic
    tie-breaking. Cost = -score for a real edge (Hungarian minimizes cost,
    we want to maximize score); unscored pairs get `_SENTINEL_COST` so the
    solver avoids them unless structurally forced to use one.
    """
    n, m = len(target_ids), len(candidate_ids)
    cost = np.full((n, m), _SENTINEL_COST, dtype=float)
    for i, target_id in enumerate(target_ids):
        for j, candidate_id in enumerate(candidate_ids):
            score = score_by_pair.get((target_id, candidate_id))
            if score is not None:
                cost[i, j] = -score + _TIEBREAK_EPSILON * (i * m + j)
    return cost


def _solve_bipartite(
    target_ids: list[str],
    candidate_ids: list[str],
    score_by_pair: dict[tuple[str, str], float],
) -> tuple[dict[str, str], frozenset[str]]:
    """Runs the Hungarian algorithm and returns (assignments,
    unresolved_targets). A row assigned to a sentinel-cost column means
    the solver was structurally forced to use a non-edge to complete the
    assignment (e.g. more targets than distinct real candidates) -- that
    target is reported as unresolved, not as a fabricated win.
    """
    cost = _build_cost_matrix(target_ids, candidate_ids, score_by_pair)
    row_ind, col_ind = linear_sum_assignment(cost)

    assignments: dict[str, str] = {}
    assigned_targets: set[str] = set()
    for i, j in zip(row_ind, col_ind):
        target_id = target_ids[i]
        candidate_id = candidate_ids[j]
        assigned_targets.add(target_id)
        if (target_id, candidate_id) in score_by_pair:
            assignments[target_id] = candidate_id
        # else: forced onto a sentinel cell -- leave unassigned, falls
        # through to unresolved_targets below.

    unresolved = frozenset(target_ids) - set(assignments.keys())
    return assignments, unresolved


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_conflict(conflict: Conflict) -> ConflictResolution:
    """Resolve a single conflict.

    ``conflict`` is any iterable of ``(target_id, candidate_option,
    score)`` tuples sharing at least one contested item. Returns a
    ``ConflictResolution`` -- see its docstring for the resolved/declined
    shape. Deterministic: the same input always produces the same output
    (see module-level tie-break notes).
    """
    options = list(conflict)
    target_ids_all = frozenset(t for t, _c, _s in options)
    candidate_ids_all = frozenset(
        item for _t, c, _s in options for item in _atomic_items(c)
    )

    decline_reason = _check_reducible(options)
    if decline_reason is not None:
        return ConflictResolution(
            resolved=False,
            assignments={},
            unresolved_targets=frozenset(),
            decline_reason=decline_reason,
            target_ids=target_ids_all,
            candidate_ids=candidate_ids_all,
        )

    score_by_pair: dict[tuple[str, str], float] = {}
    for target_id, candidate_option, score in options:
        candidate_id = next(iter(_atomic_items(candidate_option)))
        score_by_pair[(target_id, candidate_id)] = score

    # Sorted ascending so index position == id rank, for both the
    # deterministic tie-break and the "lowest id wins ties" convention.
    target_ids = sorted(target_ids_all)
    candidate_ids = sorted(candidate_ids_all)

    assignments, unresolved_targets = _solve_bipartite(target_ids, candidate_ids, score_by_pair)

    return ConflictResolution(
        resolved=True,
        assignments=assignments,
        unresolved_targets=unresolved_targets,
        decline_reason=None,
        target_ids=target_ids_all,
        candidate_ids=candidate_ids_all,
    )


def resolve_conflicts(conflicts: Iterable[Conflict]) -> list[ConflictResolution]:
    """Resolve a set of independent conflicts, one Hungarian solve per
    conflict. Conflicts are resolved independently of each other -- if two
    separate conflicts (as grouped by the caller) actually share a
    contested item, that sharing should have made them one conflict in the
    first place; this function does not detect cross-conflict overlap.
    """
    return [resolve_conflict(conflict) for conflict in conflicts]
