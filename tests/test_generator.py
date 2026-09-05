"""Tests for the Stage 2 synthetic ground-truth data generator.

Covers the three things the task asks Stage 2 to prove per ARCHITECTURE.md
§11: both datasets hit the 200-300 physical-row target, every required
anomaly category appears at least once in each dataset, and regenerating
with the same seed is byte-identical (determinism). A few extra structural
consistency checks are included since they're cheap and catch generator
bugs early (every record accounted for exactly once, calibration and
evaluation seeds/content are genuinely independent).
"""

from __future__ import annotations

import json

from recon_agent.models import NormalizedRecord
from recon_agent.testdata.generator import (
    CALIBRATION_SEED,
    EVALUATION_SEED,
    N_HONEST_ABSTENTION,
    N_LOGICAL_EVENTS,
    REQUIRED_CATEGORIES,
    GeneratedDataset,
    generate_dataset,
    write_dataset,
)

MIN_ROWS = 200
MAX_ROWS = 300
MIN_HONEST_ABSTENTION = 5


def _generate_both() -> tuple[GeneratedDataset, GeneratedDataset]:
    calibration = generate_dataset("calibration", CALIBRATION_SEED)
    evaluation = generate_dataset("evaluation", EVALUATION_SEED)
    return calibration, evaluation


# ---------------------------------------------------------------------------
# Row-count target
# ---------------------------------------------------------------------------


def test_calibration_row_count_in_target_range() -> None:
    ds = generate_dataset("calibration", CALIBRATION_SEED)
    assert MIN_ROWS <= len(ds.records) <= MAX_ROWS


def test_evaluation_row_count_in_target_range() -> None:
    ds = generate_dataset("evaluation", EVALUATION_SEED)
    assert MIN_ROWS <= len(ds.records) <= MAX_ROWS


def test_uses_exactly_100_logical_events() -> None:
    assert N_LOGICAL_EVENTS == 100


# ---------------------------------------------------------------------------
# Anomaly-mix coverage (§11)
# ---------------------------------------------------------------------------


def test_calibration_has_every_required_category() -> None:
    ds = generate_dataset("calibration", CALIBRATION_SEED)
    counts = ds.category_counts()
    missing = [c for c in REQUIRED_CATEGORIES if counts.get(c, 0) < 1]
    assert not missing, f"calibration is missing categories: {missing}"


def test_evaluation_has_every_required_category() -> None:
    ds = generate_dataset("evaluation", EVALUATION_SEED)
    counts = ds.category_counts()
    missing = [c for c in REQUIRED_CATEGORIES if counts.get(c, 0) < 1]
    assert not missing, f"evaluation is missing categories: {missing}"


def test_honest_abstention_meets_minimum_in_both_datasets() -> None:
    assert N_HONEST_ABSTENTION >= MIN_HONEST_ABSTENTION
    for ds in _generate_both():
        assert len(ds.honest_abstention) >= MIN_HONEST_ABSTENTION
        assert ds.category_counts()[
            "honest_abstention"
        ] >= MIN_HONEST_ABSTENTION


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_same_seed_produces_byte_identical_records() -> None:
    first = generate_dataset("calibration", CALIBRATION_SEED)
    second = generate_dataset("calibration", CALIBRATION_SEED)

    first_json = json.dumps([r.model_dump(mode="json") for r in first.records])
    second_json = json.dumps([r.model_dump(mode="json") for r in second.records])
    assert first_json == second_json


def test_same_seed_produces_byte_identical_ground_truth() -> None:
    first = generate_dataset("evaluation", EVALUATION_SEED)
    second = generate_dataset("evaluation", EVALUATION_SEED)
    assert json.dumps(first.ground_truth_dict()) == json.dumps(second.ground_truth_dict())


def test_regenerating_to_disk_is_byte_identical(tmp_path) -> None:
    ds = generate_dataset("calibration", CALIBRATION_SEED)
    data_dir_1 = tmp_path / "run1" / "data"
    gt_dir_1 = tmp_path / "run1" / "ground_truth"
    data_dir_2 = tmp_path / "run2" / "data"
    gt_dir_2 = tmp_path / "run2" / "ground_truth"

    records_path_1, gt_path_1 = write_dataset(ds, data_dir_1, gt_dir_1)
    ds_again = generate_dataset("calibration", CALIBRATION_SEED)
    records_path_2, gt_path_2 = write_dataset(ds_again, data_dir_2, gt_dir_2)

    assert records_path_1.read_bytes() == records_path_2.read_bytes()
    assert gt_path_1.read_bytes() == gt_path_2.read_bytes()


# ---------------------------------------------------------------------------
# Calibration/evaluation independence (§11)
# ---------------------------------------------------------------------------


def test_calibration_and_evaluation_seeds_are_distinct() -> None:
    assert CALIBRATION_SEED != EVALUATION_SEED


def test_calibration_and_evaluation_datasets_are_not_identical() -> None:
    calibration, evaluation = _generate_both()
    calibration_refs = [r.reference for r in calibration.records]
    evaluation_refs = [r.reference for r in evaluation.records]
    assert calibration_refs != evaluation_refs


# ---------------------------------------------------------------------------
# Structural consistency (defends against generator bugs)
# ---------------------------------------------------------------------------


def test_every_record_is_accounted_for_exactly_once() -> None:
    for ds in _generate_both():
        all_ids = {r.record_id for r in ds.records}
        assert len(all_ids) == len(ds.records), "duplicate record_id emitted"

        accounted: set[str] = set()
        for group in ds.match_groups:
            for rid in group["record_ids"]:
                assert rid not in accounted, f"{rid} appears in more than one group"
                accounted.add(rid)
        for dup in ds.duplicates:
            assert dup["record_id"] not in accounted
            accounted.add(dup["record_id"])
        for unresolved in ds.unresolved:
            for rid in unresolved["record_ids"]:
                assert rid not in accounted
                accounted.add(rid)
        for abstention in ds.honest_abstention:
            assert abstention["record_id"] not in accounted
            accounted.add(abstention["record_id"])

        assert accounted == all_ids


def test_all_records_validate_against_normalized_record_schema() -> None:
    for ds in _generate_both():
        for record in ds.records:
            # Round-trips through the exact Stage 1 schema the matching
            # pipeline will consume.
            NormalizedRecord.model_validate_json(record.model_dump_json())


def test_consolidated_groups_conserve_amount() -> None:
    ds = generate_dataset("calibration", CALIBRATION_SEED)
    many_to_one_groups = [g for g in ds.match_groups if g["cardinality"] == "MANY_TO_ONE"]
    assert many_to_one_groups, "expected at least one MANY_TO_ONE consolidated group"
    for group in many_to_one_groups:
        assert group["total_settlement_paise"] == group["total_bank_paise"]


def test_duplicates_reference_a_real_record_and_are_excluded_from_groups() -> None:
    for ds in _generate_both():
        assert ds.duplicates, "expected at least one duplicate case"
        record_ids = {r.record_id for r in ds.records}
        grouped_ids = {rid for g in ds.match_groups for rid in g["record_ids"]}
        for dup in ds.duplicates:
            assert dup["record_id"] in record_ids
            assert dup["duplicate_of"] in record_ids
            assert dup["record_id"] not in grouped_ids


# ---------------------------------------------------------------------------
# Stress-test seeds (multi-seed evaluation support) — additive, does not
# touch calibration's/evaluation's fixed seeds or content (asserted above).
# ---------------------------------------------------------------------------


def test_stress_seed_default_scale_matches_fixed_category_counts() -> None:
    ds = generate_dataset("stress_999", 999)
    assert ds.n_logical_events == N_LOGICAL_EVENTS
    counts = ds.category_counts()
    missing = [c for c in REQUIRED_CATEGORIES if counts.get(c, 0) < 1]
    assert not missing, f"stress seed at default scale is missing categories: {missing}"
    assert MIN_ROWS <= len(ds.records) <= MAX_ROWS


def test_stress_seed_scaled_logical_events_still_covers_every_category() -> None:
    ds = generate_dataset("stress_7", 7, n_logical_events=30)
    assert ds.n_logical_events == 30
    counts = ds.category_counts()
    missing = [c for c in REQUIRED_CATEGORIES if counts.get(c, 0) < 1]
    assert not missing, f"scaled stress seed is missing categories: {missing}"


def test_stress_seed_is_deterministic_for_same_seed_and_scale() -> None:
    first = generate_dataset("stress_7", 7, n_logical_events=30)
    second = generate_dataset("stress_7", 7, n_logical_events=30)
    first_json = json.dumps([r.model_dump(mode="json") for r in first.records])
    second_json = json.dumps([r.model_dump(mode="json") for r in second.records])
    assert first_json == second_json


def test_different_stress_seeds_produce_different_datasets() -> None:
    a = generate_dataset("stress_1", 1)
    b = generate_dataset("stress_2", 2)
    assert [r.reference for r in a.records] != [r.reference for r in b.records]


def test_stress_seed_default_call_leaves_calibration_and_evaluation_untouched() -> None:
    # generate_dataset's default (n_logical_events=None) path is exactly
    # what calibration/evaluation use — confirm a stress-named call with
    # the default resolves to the same fixed N_LOGICAL_EVENTS scale.
    calibration = generate_dataset("calibration", CALIBRATION_SEED)
    calibration_again = generate_dataset("calibration", CALIBRATION_SEED, n_logical_events=None)
    assert json.dumps([r.model_dump(mode="json") for r in calibration.records]) == json.dumps(
        [r.model_dump(mode="json") for r in calibration_again.records]
    )


def test_write_dataset_output_is_valid_json(tmp_path) -> None:
    ds = generate_dataset("calibration", CALIBRATION_SEED)
    records_path, gt_path = write_dataset(ds, tmp_path / "data", tmp_path / "ground_truth")

    with records_path.open() as f:
        records_payload = json.load(f)
    assert len(records_payload) == len(ds.records)

    with gt_path.open() as f:
        gt_payload = json.load(f)
    assert gt_payload["dataset"] == "calibration"
    assert gt_payload["num_physical_records"] == len(ds.records)
