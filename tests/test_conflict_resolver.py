"""Unit tests for src/recon_agent/matching/conflict_resolver.py.

Stage 7 (relay numbering) -- standalone module, not wired into the
pipeline. The last test is a real-world-relevance sanity check only: it
loads the actual calibration dataset, runs the existing pipeline, pulls
the real AMBIGUOUS_AGGREGATION exceptions Stage 3/4 already produce, and
confirms their shape (atomic item ids, not multi-item bundles) is
something ``conflict_resolver``'s cost-matrix construction can represent
as a clean 1:1 assignment problem. It does not resolve them for real
(the exception evidence doesn't retain per-target-per-item scores) and it
does not wire anything into the pipeline -- that is a later, separate
stage.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from recon_agent.matching.conflict_resolver import (
    ConflictResolution,
    resolve_conflict,
    resolve_conflicts,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_RECORDS = REPO_ROOT / "data" / "calibration" / "records.json"


def test_two_targets_sharing_top_choice_resolve_to_globally_optimal_assignment() -> None:
    """Matches what Stage 3 actually produces: two bank-credit targets
    (A, B) both score their #1 candidate against the same settlement-unit
    ("unit1"), each with a different, non-conflicting second choice.

    A: unit1=0.90 (top), unitA2=0.85 (second)
    B: unit1=0.88 (top), unitB2=0.20 (second)

    A naive "process targets in order, each grabs its own best available
    item" greedy pass would give A unit1 (it iterates first) and leave B
    with unitB2 -- total score 0.90 + 0.20 = 1.10. A naive "give the item
    to whoever scores it highest" rule would reach the same wrong answer
    (0.90 > 0.88, so A wins unit1) for the same total. The actual globally
    optimal assignment is the other way around -- B keeps unit1 (0.88)
    and A falls back to its very-close second choice (0.85) -- total
    0.88 + 0.85 = 1.73, strictly better than either naive approach and
    not order-dependent.
    """
    conflict = [
        ("A", "unit1", 0.90),
        ("A", "unitA2", 0.85),
        ("B", "unit1", 0.88),
        ("B", "unitB2", 0.20),
    ]

    result = resolve_conflict(conflict)

    assert result.resolved
    assert result.decline_reason is None
    assert result.unresolved_targets == frozenset()
    assert result.assignments == {"A": "unitA2", "B": "unit1"}

    # Confirm it really is the *globally* optimal total, not merely "a"
    # feasible assignment.
    scores = {(t, c): s for t, c, s in conflict}
    achieved_total = sum(scores[(t, c)] for t, c in result.assignments.items())
    naive_total = scores[("A", "unit1")] + scores[("B", "unitB2")]
    assert achieved_total > naive_total
    assert achieved_total == pytest.approx(0.88 + 0.85)


def test_reversed_input_order_still_finds_same_optimum() -> None:
    """Same conflict as above, tuples supplied in a different order --
    the resolution must not depend on iteration order."""
    conflict = [
        ("B", "unitB2", 0.20),
        ("B", "unit1", 0.88),
        ("A", "unitA2", 0.85),
        ("A", "unit1", 0.90),
    ]
    result = resolve_conflict(conflict)
    assert result.resolved
    assert result.assignments == {"A": "unitA2", "B": "unit1"}


def test_multi_item_bundle_conflict_is_declined_not_forced() -> None:
    """A genuinely tangled N:M case: two targets' best options are each a
    *combination* of items (Stage 3/4 subset-sum aggregation territory),
    and the combinations share an underlying item even though they have
    different candidate ids. This does not reduce to a clean bipartite
    1:1 assignment -- per ARCHITECTURE.md §4, arbitrary N:M is
    unsupported, so this must be declined outright rather than
    approximated.
    """
    conflict = [
        ("A", frozenset({"itemX", "itemY"}), 0.90),
        ("B", frozenset({"itemY", "itemZ"}), 0.80),
    ]

    result = resolve_conflict(conflict)

    assert result.resolved is False
    assert result.decline_reason == "MULTI_ITEM_BUNDLE_NOT_1_TO_1"
    assert result.assignments == {}
    # Even when declined, the resolution still records who/what was
    # involved, for audit purposes.
    assert result.target_ids == frozenset({"A", "B"})
    assert result.candidate_ids == frozenset({"itemX", "itemY", "itemZ"})


def test_single_item_option_mixed_with_bundle_option_still_declines() -> None:
    """A conflict need not be *entirely* bundles to be non-1:1 -- a single
    multi-item option anywhere in the conflict is enough to make the
    bipartite "different option id => safe to assign both" assumption
    unsafe, so the whole conflict declines.
    """
    conflict = [
        ("A", "unit1", 0.90),
        ("B", frozenset({"unit1", "unit2"}), 0.80),
    ]
    result = resolve_conflict(conflict)
    assert result.resolved is False
    assert result.decline_reason == "MULTI_ITEM_BUNDLE_NOT_1_TO_1"


def test_inconsistent_scores_for_same_pair_is_declined() -> None:
    conflict = [
        ("A", "unit1", 0.90),
        ("A", "unit1", 0.50),  # same (target, candidate), different score
        ("B", "unit2", 0.70),
    ]
    result = resolve_conflict(conflict)
    assert result.resolved is False
    assert result.decline_reason == "INCONSISTENT_SCORES"


def test_more_targets_than_candidates_resolves_the_winnable_subset() -> None:
    """Three targets, only one contested item and no alternatives for two
    of them -- the highest scorer wins it; the other two structurally had
    nothing else to fall back on, so they're reported unresolved rather
    than a decline of the whole conflict (this is a legitimate optimal
    partial outcome, not a forced guess).
    """
    conflict = [
        ("A", "unit1", 0.95),
        ("B", "unit1", 0.80),
        ("C", "unit1", 0.60),
    ]
    result = resolve_conflict(conflict)
    assert result.resolved
    assert result.assignments == {"A": "unit1"}
    assert result.unresolved_targets == frozenset({"B", "C"})


def test_determinism_same_input_always_same_output() -> None:
    conflict = [
        ("T1", "c1", 0.75),
        ("T1", "c2", 0.75),  # exact tie for T1
        ("T2", "c1", 0.75),  # exact tie with T1's c1 score too
        ("T2", "c2", 0.75),
    ]
    results = [resolve_conflict(conflict) for _ in range(25)]
    first = results[0]
    for r in results[1:]:
        assert r.assignments == first.assignments
        assert r.unresolved_targets == first.unresolved_targets
        assert r.resolved == first.resolved

    # Deterministic tie-break convention: lowest id wins ties. T1 < T2 and
    # c1 < c2 lexicographically, so with every score exactly tied, T1
    # should land on c1.
    assert first.assignments == {"T1": "c1", "T2": "c2"}


def test_resolve_conflicts_processes_each_conflict_independently() -> None:
    conflict_1 = [("A", "x", 0.9), ("B", "x", 0.5), ("A", "y", 0.1), ("B", "z", 0.4)]
    conflict_2 = [("C", frozenset({"p", "q"}), 0.9), ("D", frozenset({"q", "r"}), 0.8)]

    results = resolve_conflicts([conflict_1, conflict_2])

    assert len(results) == 2
    assert results[0].resolved is True
    assert results[1].resolved is False
    assert results[1].decline_reason == "MULTI_ITEM_BUNDLE_NOT_1_TO_1"


def test_empty_conflict_declines_cleanly() -> None:
    result = resolve_conflict([])
    assert result.resolved is False
    assert result.decline_reason == "EMPTY_CONFLICT"


# ---------------------------------------------------------------------------
# Real-world-relevance sanity check (not wiring, not resolution -- shape only)
# ---------------------------------------------------------------------------

pytestmark_real_data = pytest.mark.skipif(
    not CALIBRATION_RECORDS.exists(),
    reason="data/calibration/records.json not found",
)


@pytestmark_real_data
def test_real_ambiguous_aggregation_conflicts_fit_the_resolver_input_shape() -> None:
    """Loads data/calibration/records.json, runs the existing pipeline,
    and pulls the real AMBIGUOUS_AGGREGATION exceptions Stage 3/4 already
    produce (evidence contains `involved_item_ids`, a mix of
    "item:group:<group_id>" and "item:rec:<record_id>" strings).

    This is a sanity check on *shape*, not a resolution: the exception
    evidence records which items ended up contested and which targets
    were declined, but (correctly, per its own scope) does not retain the
    real per-target-per-item score that led to that scoring rank -- Stage
    3/4 only kept the best subset and the fact that it collided, per
    aggregation_common.WindowResult. So each real conflict is rebuilt here
    as a synthetic ConflictOption set using the real target ids and real
    contested item ids, with a placeholder equal score per pair (since the
    real scores aren't preserved in exception evidence) -- enough to
    confirm the *dimensions* (atomic item ids, not multi-item bundles;
    N targets x M items) are exactly what this module's cost-matrix
    construction expects, as a sanity check for the next stage (wiring
    real per-pair scores through instead of a placeholder).
    """
    from recon_agent.config import Settings
    from recon_agent.matching import run_pipeline
    from recon_agent.models import NormalizedRecord

    records = [NormalizedRecord.model_validate(r) for r in json.loads(CALIBRATION_RECORDS.read_text())]
    result = run_pipeline(records, Settings())

    # Real AMBIGUOUS_AGGREGATION exceptions that actually name contested
    # items (excludes CANDIDATE_POOL_TOO_LARGE / WINDOW_TIMEOUT declines,
    # which never got far enough to have a tentative winner to conflict
    # over -- there's no item-sharing data to build a conflict from).
    overlap_exceptions = [
        e
        for e in result.exceptions
        if e.category.name == "AMBIGUOUS_AGGREGATION" and e.evidence.get("involved_item_ids")
    ]

    assert overlap_exceptions, "expected at least one real AMBIGUOUS_OVERLAP exception in calibration data"

    # Group into distinct real conflicts by their shared contested-item
    # pool (targets that were declined over the same involved_item_ids
    # are the same conflict cluster).
    clusters: dict[frozenset, set[str]] = {}
    for e in overlap_exceptions:
        items = frozenset(e.evidence["involved_item_ids"])
        target_id = e.evidence["target_id"]
        clusters.setdefault(items, set()).add(target_id)

    fits_shape = 0
    for items, target_ids in clusters.items():
        synthetic_conflict = [
            (target_id, item_id, 1.0)  # placeholder score -- see docstring
            for target_id in sorted(target_ids)
            for item_id in sorted(items)
        ]
        resolution = resolve_conflict(synthetic_conflict)
        assert isinstance(resolution, ConflictResolution)
        # Every real item_id here is a plain string (atomic), never a
        # bundle, so this must never decline as MULTI_ITEM_BUNDLE_NOT_1_TO_1
        # -- that would mean a real exception's involved_item_ids somehow
        # contained a non-atomic entry.
        assert resolution.decline_reason != "MULTI_ITEM_BUNDLE_NOT_1_TO_1"
        if resolution.resolved:
            fits_shape += 1

    print(
        f"\n[conflict_resolver real-data check] {len(overlap_exceptions)} real "
        f"AMBIGUOUS_AGGREGATION exception(s) with involved_item_ids, grouped "
        f"into {len(clusters)} distinct real conflict(s); "
        f"{fits_shape}/{len(clusters)} fit this resolver's 1:1 cost-matrix "
        "shape cleanly (placeholder-scored, shape check only -- not a real "
        "resolution)."
    )
    for items, target_ids in clusters.items():
        print(f"  conflict: {len(target_ids)} target(s) x {len(items)} item(s)")

    assert fits_shape == len(clusters)
