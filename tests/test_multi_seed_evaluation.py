"""Tests for scripts/multi_seed_evaluation.py's aggregation math.

Deliberately scoped to the aggregation functions themselves — 2-3 tiny
constructed per-seed metric fixtures with known correct mean/min/max —
not a full 20-seed generate+pipeline+evaluate run. That real run is
executed manually (see BUILD_LOG.md / README.md for the actual
numbers); it has no place in a test suite that needs to stay fast.

Also covers seed derivation (distinct from calibration/evaluation,
distinct from each other, and reproducible for a fixed meta-seed) since
that's cheap, pure, and the other piece of "correctness" this script
depends on beyond the pipeline it's driving.
"""

from __future__ import annotations

from dataclasses import dataclass

from recon_agent.testdata.generator import CALIBRATION_SEED, EVALUATION_SEED
from scripts.multi_seed_evaluation import (
    METRIC_SPECS,
    MetricAggregate,
    aggregate_metric_values,
    aggregate_reports,
    derive_stress_seeds,
)


# ---------------------------------------------------------------------------
# aggregate_metric_values — the core mean/min/max math
# ---------------------------------------------------------------------------


def test_aggregate_metric_values_known_mean_min_max() -> None:
    # Three tiny, hand-picked values with an exactly-known mean.
    agg = aggregate_metric_values("precision", [1.0, 0.5, 0.75])
    assert agg.mean == (1.0 + 0.5 + 0.75) / 3
    assert agg.min == 0.5
    assert agg.max == 1.0
    assert agg.n == 3
    assert agg.missing == 0


def test_aggregate_metric_values_all_identical() -> None:
    agg = aggregate_metric_values("false_match_rate", [0.0, 0.0, 0.0])
    assert agg.mean == 0.0
    assert agg.min == 0.0
    assert agg.max == 0.0
    assert agg.n == 3
    assert agg.missing == 0


def test_aggregate_metric_values_two_point_fixture() -> None:
    agg = aggregate_metric_values("bank_credit_coverage", [0.4, 0.6])
    assert agg.mean == 0.5
    assert agg.min == 0.4
    assert agg.max == 0.6
    assert agg.n == 2
    assert agg.missing == 0


def test_aggregate_metric_values_skips_none_and_counts_missing() -> None:
    # A metric that was structurally undefined (None) on one of three
    # seeds — e.g. zero VERIFIED groups makes precision undefined, not
    # zero. mean/min/max must be computed only from the two real values,
    # while ``missing`` still reflects that one seed didn't contribute.
    agg = aggregate_metric_values("auto_match_precision", [1.0, None, 0.8])
    assert agg.mean == (1.0 + 0.8) / 2
    assert agg.min == 0.8
    assert agg.max == 1.0
    assert agg.n == 2
    assert agg.missing == 1


def test_aggregate_metric_values_all_none_returns_none_not_a_crash() -> None:
    agg = aggregate_metric_values("exception_quality", [None, None])
    assert agg.mean is None
    assert agg.min is None
    assert agg.max is None
    assert agg.n == 0
    assert agg.missing == 2


# ---------------------------------------------------------------------------
# aggregate_reports — wiring METRIC_SPECS' extractors over a list of
# report-like objects. Uses tiny fake objects shaped like the real
# EvaluationReport's nested result dataclasses (same attribute names
# METRIC_SPECS' lambdas read) rather than running the real pipeline.
# ---------------------------------------------------------------------------


@dataclass
class _FakePrecision:
    precision: float


@dataclass
class _FakeCoverage:
    raw_coverage: float
    coverage_excluding_abstention: float


@dataclass
class _FakeValueCoverage:
    value_coverage: float
    value_coverage_excluding_abstention: float


@dataclass
class _FakeFalseMatch:
    false_match_rate: float


@dataclass
class _FakeExceptionQuality:
    exception_quality: float


@dataclass
class _FakeBankCreditCoverage:
    bank_credit_coverage: float


@dataclass
class _FakeCompleteClusterResolution:
    complete_cluster_resolution: float


@dataclass
class _FakeReport:
    auto_match_precision: _FakePrecision
    record_coverage: _FakeCoverage
    value_coverage: _FakeValueCoverage
    false_match_rate: _FakeFalseMatch
    exception_quality: _FakeExceptionQuality
    bank_credit_coverage: _FakeBankCreditCoverage
    complete_cluster_resolution: _FakeCompleteClusterResolution


def _fake_report(precision: float, coverage: float, bank_credit: float) -> _FakeReport:
    return _FakeReport(
        auto_match_precision=_FakePrecision(precision=precision),
        record_coverage=_FakeCoverage(raw_coverage=coverage, coverage_excluding_abstention=coverage + 0.02),
        value_coverage=_FakeValueCoverage(value_coverage=coverage - 0.1, value_coverage_excluding_abstention=coverage - 0.08),
        false_match_rate=_FakeFalseMatch(false_match_rate=0.0),
        exception_quality=_FakeExceptionQuality(exception_quality=1.0),
        bank_credit_coverage=_FakeBankCreditCoverage(bank_credit_coverage=bank_credit),
        complete_cluster_resolution=_FakeCompleteClusterResolution(complete_cluster_resolution=bank_credit - 0.05),
    )


def test_aggregate_reports_uses_every_metric_spec_key() -> None:
    reports = [
        _fake_report(precision=1.0, coverage=0.8, bank_credit=0.5),
        _fake_report(precision=0.9, coverage=0.85, bank_credit=0.4),
        _fake_report(precision=1.0, coverage=0.75, bank_credit=0.6),
    ]
    aggregates = aggregate_reports(reports)  # type: ignore[arg-type]
    assert set(aggregates) == {key for key, _label, _fn in METRIC_SPECS}
    assert isinstance(aggregates["auto_match_precision"], MetricAggregate)


def test_aggregate_reports_precision_mean_min_max_matches_known_fixture() -> None:
    reports = [
        _fake_report(precision=1.0, coverage=0.8, bank_credit=0.5),
        _fake_report(precision=0.9, coverage=0.85, bank_credit=0.4),
        _fake_report(precision=1.0, coverage=0.75, bank_credit=0.6),
    ]
    aggregates = aggregate_reports(reports)  # type: ignore[arg-type]
    precision_agg = aggregates["auto_match_precision"]
    assert precision_agg.mean == (1.0 + 0.9 + 1.0) / 3
    assert precision_agg.min == 0.9
    assert precision_agg.max == 1.0

    bank_credit_agg = aggregates["bank_credit_coverage"]
    assert bank_credit_agg.mean == (0.5 + 0.4 + 0.6) / 3
    assert bank_credit_agg.min == 0.4
    assert bank_credit_agg.max == 0.6


# ---------------------------------------------------------------------------
# derive_stress_seeds — deterministic, distinct from calibration/
# evaluation, distinct from each other, reproducible for a fixed
# meta-seed. All cheap/pure, no dataset generation involved.
# ---------------------------------------------------------------------------


def test_derive_stress_seeds_returns_requested_count() -> None:
    seeds = derive_stress_seeds(5, meta_seed=12345)
    assert len(seeds) == 5
    assert len(set(seeds)) == 5


def test_derive_stress_seeds_excludes_calibration_and_evaluation_seeds() -> None:
    seeds = derive_stress_seeds(20, meta_seed=20261231)
    assert CALIBRATION_SEED not in seeds
    assert EVALUATION_SEED not in seeds


def test_derive_stress_seeds_is_deterministic_for_same_meta_seed() -> None:
    first = derive_stress_seeds(20, meta_seed=20261231)
    second = derive_stress_seeds(20, meta_seed=20261231)
    assert first == second


def test_derive_stress_seeds_differs_across_meta_seeds() -> None:
    a = derive_stress_seeds(20, meta_seed=1)
    b = derive_stress_seeds(20, meta_seed=2)
    assert a != b
