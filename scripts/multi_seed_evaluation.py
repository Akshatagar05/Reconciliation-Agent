"""On-demand multi-seed stress evaluation (ARCHITECTURE.md §11) — additive
tooling only. Does not touch matching/verification/LLM logic.

CONTEXT: an external review independently generated 20 additional
fresh-seeded datasets and reported strong, consistent precision across
all of them — but that analysis lives outside this repo and cannot be
reproduced by anyone reading it. This script makes that kind of
multi-seed validation something the repo can generate and prove for
itself, on demand, instead of citing an external number.

What it does, end to end:

1. Derives N (default 20, matching the external review's own
   methodology so the numbers are directly comparable) fresh, distinct
   integer seeds from a single ``--meta-seed`` (default fixed, so this
   script's own seed selection is itself reproducible — pass a
   different ``--meta-seed`` to get a genuinely different batch).
2. For each seed, generates a fresh stress-test dataset via
   ``testdata/generator.py``'s ``--seed stress`` path — the exact same
   §11 anomaly-mix generation logic as calibration/evaluation, just a
   different RNG seed — written to a throwaway temp directory (never
   the repo's own ``data/``/``ground_truth/``, per "keep it lean: the
   point is reproducibility of the *process*, not storing 20 more data
   files in git").
3. Runs the full real pipeline against each, via
   ``evaluation/harness.py``'s own ``run_evaluation`` — using whatever
   thresholds are ALREADY configured/calibrated (``Settings.from_env()``
   loaded once, up front, and reused for every seed). Nothing here
   retunes anything against these datasets; they are validation-only,
   same calibration/evaluation-set discipline as the rest of this repo.
4. Computes the harness's full §11 metric set for each seed, and
   aggregates mean/min/max across all seeds for each metric.
5. Prints a human-readable summary table and writes a JSON export,
   mirroring ``evaluation/harness.py``'s own CLI output style.

Groq: every seed shares one ``Settings`` (loaded once from the
environment) and one temp governor DB, so the real daily-call-ceiling
and circuit-breaker protections in ``llm/governor.py`` apply across the
WHOLE stress batch, not per-seed — this is what keeps a misconfigured
key from burning 20x the intended budget. If ``GROQ_API_KEY`` is unset,
Stage 6 degrades gracefully on every seed exactly as it does for
calibration/evaluation (§14) — this script makes zero live calls in
that case. Either way, the actual outcome (configured vs. degraded,
and how many live calls were really made) is measured from the run's
own ``runtime_and_cost.groq_calls_made`` and printed, never assumed.

Usage:
    python -m scripts.multi_seed_evaluation
    python -m scripts.multi_seed_evaluation --n-seeds 20 --meta-seed 20261231
    (or) python scripts/multi_seed_evaluation.py
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from recon_agent.config import Settings
from recon_agent.evaluation.harness import EvaluationReport, run_evaluation
from recon_agent.testdata.generator import (
    CALIBRATION_SEED,
    EVALUATION_SEED,
    N_LOGICAL_EVENTS,
    generate_dataset,
    write_dataset,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_N_SEEDS = 20  # matches the external review's own methodology
DEFAULT_META_SEED = 20261231  # fixed and distinct from CALIBRATION_SEED/
# EVALUATION_SEED — controls only which N seeds this script itself picks,
# not the pipeline/thresholds. Kept fixed so re-running this script with
# no arguments reproduces the same batch of stress seeds every time;
# pass --meta-seed to get a genuinely different batch.


def derive_stress_seeds(n: int, meta_seed: int) -> list[int]:
    """Deterministically derive ``n`` distinct integer seeds from
    ``meta_seed``, guaranteed distinct from CALIBRATION_SEED/
    EVALUATION_SEED and from each other. Uses its own seeded
    ``random.Random`` — never the global ``random`` module — so this
    selection never leaks state into (or out of) dataset generation
    itself, which reseeds independently per stress seed.
    """
    rng = random.Random(meta_seed)
    excluded = {CALIBRATION_SEED, EVALUATION_SEED}
    seeds: set[int] = set()
    while len(seeds) < n:
        candidate = rng.randint(10_000_000, 99_999_999)
        if candidate not in excluded:
            seeds.add(candidate)
    return sorted(seeds)


# ---------------------------------------------------------------------------
# Aggregation math — the part covered by fast, tiny-fixture unit tests
# (tests/test_multi_seed_evaluation.py). Kept as pure functions, entirely
# independent of dataset generation or the pipeline, so it's testable
# without a full 20-seed run.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricAggregate:
    metric: str
    mean: Optional[float]
    min: Optional[float]
    max: Optional[float]
    n: int
    missing: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "mean": self.mean,
            "min": self.min,
            "max": self.max,
            "n": self.n,
            "missing": self.missing,
        }


def aggregate_metric_values(metric: str, values: list[Optional[float]]) -> MetricAggregate:
    """Mean/min/max across ``values``, skipping any ``None`` (a metric
    that was structurally undefined for that seed — e.g. zero VERIFIED
    groups makes precision undefined, not zero; see
    ``evaluation/harness.py``'s own ``Optional[float]`` result fields).
    ``missing`` counts how many of the ``n_total`` inputs were ``None``,
    so a metric that's undefined on every seed is visibly distinct from
    one that's cleanly 0.0 on every seed.
    """
    present = [v for v in values if v is not None]
    if not present:
        return MetricAggregate(metric=metric, mean=None, min=None, max=None, n=0, missing=len(values))
    return MetricAggregate(
        metric=metric,
        mean=statistics.mean(present),
        min=min(present),
        max=max(present),
        n=len(present),
        missing=len(values) - len(present),
    )


# The §11 judge-facing metrics this script aggregates, and how to pull
# each one out of a single-seed EvaluationReport. Kept as a plain list of
# (key, label, extractor) so both the aggregation and the human-readable
# table below iterate the exact same source of truth.
METRIC_SPECS: tuple[tuple[str, str, Callable[[EvaluationReport], Optional[float]]], ...] = (
    ("auto_match_precision", "1. Auto-match precision", lambda r: r.auto_match_precision.precision),
    ("record_coverage_raw", "2. Record coverage (raw)", lambda r: r.record_coverage.raw_coverage),
    (
        "record_coverage_excl_abstention",
        "2. Record coverage (excl. abstention)",
        lambda r: r.record_coverage.coverage_excluding_abstention,
    ),
    ("value_coverage", "3. Value coverage", lambda r: r.value_coverage.value_coverage),
    (
        "value_coverage_excl_abstention",
        "3. Value coverage (excl. abstention)",
        lambda r: r.value_coverage.value_coverage_excluding_abstention,
    ),
    ("false_match_rate", "4. False-match rate", lambda r: r.false_match_rate.false_match_rate),
    ("exception_quality", "5. Exception quality", lambda r: r.exception_quality.exception_quality),
    ("bank_credit_coverage", "8. Bank credit coverage", lambda r: r.bank_credit_coverage.bank_credit_coverage),
    (
        "complete_cluster_resolution",
        "9. Complete cluster resolution",
        lambda r: r.complete_cluster_resolution.complete_cluster_resolution,
    ),
)


def aggregate_reports(reports: list[EvaluationReport]) -> dict[str, MetricAggregate]:
    return {
        key: aggregate_metric_values(key, [extractor(r) for r in reports])
        for key, _label, extractor in METRIC_SPECS
    }


# ---------------------------------------------------------------------------
# Orchestration — generate N fresh datasets, run the real pipeline
# against each, aggregate. Never touches matching/verification/LLM code;
# never retunes anything against these datasets.
# ---------------------------------------------------------------------------


def run_multi_seed_evaluation(
    n_seeds: int = DEFAULT_N_SEEDS,
    meta_seed: int = DEFAULT_META_SEED,
    logical_events: int = N_LOGICAL_EVENTS,
    settings: Optional[Settings] = None,
    keep_temp_dir: bool = False,
) -> tuple[list[int], list[EvaluationReport], dict[str, MetricAggregate], Path, float, bool]:
    """Generate ``n_seeds`` fresh stress datasets and evaluate each with
    the already-calibrated thresholds. Returns (seeds, reports,
    aggregates, temp_dir_used, wall_clock_seconds, temp_dir_kept).
    """
    seeds = derive_stress_seeds(n_seeds, meta_seed)
    settings = settings if settings is not None else Settings.from_env()

    tmp_root = Path(tempfile.mkdtemp(prefix="recon_agent_stress_"))
    governor_db_path = str(tmp_root / "stress_governor.db")

    reports: list[EvaluationReport] = []
    start = time.perf_counter()
    try:
        for seed in seeds:
            name = f"stress_{seed}"
            dataset = generate_dataset(name, seed, n_logical_events=logical_events)
            write_dataset(dataset, tmp_root / "data", tmp_root / "ground_truth")
            report = run_evaluation(
                name,
                data_root=tmp_root,
                settings=settings,
                governor_db_path=governor_db_path,
            )
            reports.append(report)
            print(
                f"[seed {seed}] records={report.total_records} "
                f"precision={_pct(report.auto_match_precision.precision)} "
                f"false_match={_pct(report.false_match_rate.false_match_rate)} "
                f"groq_calls={report.runtime_and_cost.groq_calls_made}"
            )
    finally:
        elapsed = time.perf_counter() - start
        if not keep_temp_dir:
            shutil.rmtree(tmp_root, ignore_errors=True)

    aggregates = aggregate_reports(reports)
    return seeds, reports, aggregates, tmp_root, elapsed, keep_temp_dir


def _pct(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


# ---------------------------------------------------------------------------
# Rendering — human-readable summary + JSON export
# ---------------------------------------------------------------------------


def render_human_readable(
    seeds: list[int],
    reports: list[EvaluationReport],
    aggregates: dict[str, MetricAggregate],
    meta_seed: int,
    logical_events: int,
    settings: Settings,
    wall_clock_seconds: float,
) -> str:
    lines: list[str] = []
    a = lines.append
    a(
        f"Multi-seed stress evaluation — N={len(seeds)} fresh-seeded datasets "
        f"(meta_seed={meta_seed}, {logical_events} logical events each)"
    )
    a("=" * 72)

    groq_configured = bool(settings.groq_api_key)
    total_groq_calls = sum(r.runtime_and_cost.groq_calls_made for r in reports)
    if not groq_configured:
        groq_line = "Groq: not configured — every seed degraded gracefully (0 live calls, §14)."
    elif total_groq_calls == 0:
        groq_line = (
            "Groq: configured, but 0 live calls were made across all seeds "
            "(no Stage 6 candidates arose, or the circuit breaker/budget "
            "ceiling held) — degraded path exercised regardless."
        )
    else:
        groq_line = f"Groq: configured — {total_groq_calls} live call(s) made across all {len(seeds)} seed(s)."
    a(groq_line)
    a(
        f"Wall clock: {wall_clock_seconds:.2f}s total "
        f"({wall_clock_seconds / len(seeds):.2f}s/seed avg)"
    )
    a("")
    a(f"{'Metric':<42}{'mean':>10}{'min':>10}{'max':>10}")
    for key, label, _extractor in METRIC_SPECS:
        agg = aggregates[key]
        missing_note = f"  ({agg.missing} undefined)" if agg.missing else ""
        a(f"{label:<42}{_pct(agg.mean):>10}{_pct(agg.min):>10}{_pct(agg.max):>10}{missing_note}")
    a("")
    a(f"Seeds evaluated ({len(seeds)}): {', '.join(str(s) for s in seeds)}")
    return "\n".join(lines)


def build_json_export(
    seeds: list[int],
    reports: list[EvaluationReport],
    aggregates: dict[str, MetricAggregate],
    meta_seed: int,
    logical_events: int,
    settings: Settings,
    wall_clock_seconds: float,
) -> dict[str, Any]:
    total_groq_calls = sum(r.runtime_and_cost.groq_calls_made for r in reports)
    return {
        "n_seeds": len(seeds),
        "meta_seed": meta_seed,
        "logical_events_per_seed": logical_events,
        "seeds": seeds,
        "groq_configured": bool(settings.groq_api_key),
        "total_groq_calls": total_groq_calls,
        "wall_clock_seconds": wall_clock_seconds,
        "aggregate": {key: agg.to_dict() for key, agg in aggregates.items()},
        "per_seed": [
            {
                "seed": seed,
                "dataset": report.dataset,
                "total_records": report.total_records,
                "groq_calls_made": report.runtime_and_cost.groq_calls_made,
                **{key: extractor(report) for key, _label, extractor in METRIC_SPECS},
            }
            for seed, report in zip(seeds, reports)
        ],
        "per_seed_reports": [report.to_dict() for report in reports],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate N fresh-seeded stress-test datasets on demand, run the "
            "full pipeline against each with the already-calibrated "
            "thresholds, and aggregate the §11 evaluation metrics across all "
            "of them (mean/min/max) — a reproducible, in-repo alternative to "
            "citing an external multi-seed analysis."
        )
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=DEFAULT_N_SEEDS,
        help=f"Number of fresh seeds to generate and evaluate. Default: {DEFAULT_N_SEEDS}.",
    )
    parser.add_argument(
        "--meta-seed",
        type=int,
        default=DEFAULT_META_SEED,
        help=(
            "Seed controlling which N stress seeds are picked (not the "
            f"pipeline itself). Default: {DEFAULT_META_SEED} — fixed, so the "
            "default invocation is itself reproducible."
        ),
    )
    parser.add_argument(
        "--logical-events",
        type=int,
        default=N_LOGICAL_EVENTS,
        help=(
            "Logical events per stress dataset, reusing "
            "testdata/generator.py's scaled anomaly-mix logic. Default: "
            f"{N_LOGICAL_EVENTS}, matching calibration/evaluation's scale."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write the JSON export. Default: reports/multi_seed_evaluation_report.json",
    )
    parser.add_argument(
        "--keep-temp-dir",
        action="store_true",
        help=(
            "Don't delete the generated stress datasets' temp directory "
            "after the run (debugging only — they're never written into "
            "this repo's own data/ or ground_truth/ either way)."
        ),
    )
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    seeds, reports, aggregates, tmp_root, wall_clock_seconds, kept = run_multi_seed_evaluation(
        n_seeds=args.n_seeds,
        meta_seed=args.meta_seed,
        logical_events=args.logical_events,
        settings=settings,
        keep_temp_dir=args.keep_temp_dir,
    )

    print("")
    print(
        render_human_readable(
            seeds, reports, aggregates, args.meta_seed, args.logical_events, settings, wall_clock_seconds
        )
    )
    if kept:
        print(f"\n(Stress datasets kept at: {tmp_root})")

    output_path = Path(args.output) if args.output else REPO_ROOT / "reports" / "multi_seed_evaluation_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export = build_json_export(
        seeds, reports, aggregates, args.meta_seed, args.logical_events, settings, wall_clock_seconds
    )
    output_path.write_text(json.dumps(export, indent=2))
    print(f"\nJSON report written to {output_path}")


if __name__ == "__main__":
    main()
