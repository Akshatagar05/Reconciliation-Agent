"""Configuration loader — ARCHITECTURE.md §7, §9, §10.

Holds the four independently-versioned axes (rules_version,
model_version, threshold_version, policy_version — see §7's note that
"threshold_version" and "policy_version" are versioned separately from
rules/model, and §2's stage-differentiated auto-commit policy that
depends on this split), the §9 aggregation search limits (placeholders
only — no aggregation logic exists yet in this stage), and the Groq
credential (§10; §14 requires the system to degrade gracefully when the
key is missing, exhausted, or returns malformed JSON).

This uses a plain Pydantic v2 BaseModel loaded explicitly from the
environment, not pydantic-settings' BaseSettings — pydantic-settings is
not in the pinned requirements.txt dependency list (§10), so we avoid
depending on it.
"""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field


class AggregationSearchLimits(BaseModel):
    """§9 — Aggregation Search, Documented Limits.

    These are documented starting defaults ("tune against real
    synthetic-data density during Days 5-6" per §9's closing note), not
    hardcoded magic numbers. No many-to-one aggregation *search* logic
    exists yet (Stage 1/2 matching is implemented — see matching/ — but
    it deliberately does not do subset-sum aggregation across several
    settlements into one consolidated bank credit; that's Stage 3
    (aggregate)'s job, a later stage of this build). These are
    placeholders for that not-yet-built resolver.
    """

    model_config = ConfigDict(use_enum_values=False)

    max_candidates_per_window: int = Field(
        default=20,
        description=(
            "Keeps the many-to-one subset-sum search bounded and exact "
            "rather than heuristic (§9)."
        ),
    )
    max_group_size: int = Field(
        default=8,
        description=(
            "Reflects realistic consolidated-settlement sizes; larger "
            "claims route to Exception: AMBIGUOUS_AGGREGATION (§9)."
        ),
    )
    window_timeout_seconds: float = Field(
        default=2.0,
        description=(
            "If exceeded, the window is abandoned and its records fall "
            "through to Exception, never left half-resolved (§9)."
        ),
    )


class Settings(BaseModel):
    """Top-level application settings.

    rules_version / model_version / threshold_version / policy_version
    are four separate configurable values, not one combined version
    string, because each axis is versioned and tuned independently
    (§7). These map directly onto the matching fields on
    ReconciliationRun and onto MatchGroup.threshold_version.
    """

    model_config = ConfigDict(use_enum_values=False)

    rules_version: str = "v1"
    model_version: str = "v1"
    threshold_version: str = "v1"
    policy_version: str = "v1"

    aggregation_search: AggregationSearchLimits = Field(
        default_factory=AggregationSearchLimits
    )

    groq_api_key: Optional[str] = Field(
        default=None,
        description=(
            "Read from the GROQ_API_KEY environment variable. May be "
            "unset — §14 requires the system to complete gracefully when "
            "the Groq key is missing, exhausted, or returns malformed "
            "JSON, not fail outright."
        ),
    )
    groq_model: str = Field(
        default="openai/gpt-oss-120b",
        description=(
            "Read from the GROQ_MODEL environment variable. §10: 'Groq — "
            "recommend-only'. Kept configurable rather than hardcoded into "
            "llm/recommender.py so the model in use can be changed (or "
            "pinned to a different snapshot) without a code change. "
            "Was 'meta-llama/llama-4-scout-17b-16e-instruct' until Groq "
            "deprecated it (announced June 2026); switched to "
            "'openai/gpt-oss-120b', Groq's own 1:1 replacement "
            "recommendation, on 2026-09-04 — see BUILD_LOG.md's 'Groq "
            "model deprecation' entry. gpt-oss-120b is a reasoning model, "
            "which llm/recommender.py's response parsing accounts for."
        ),
    )
    groq_timeout_seconds: float = Field(
        default=15.0,
        description=(
            "Read from the GROQ_TIMEOUT_SECONDS environment variable. "
            "Per-call timeout for llm/recommender.py's Groq API call — "
            "§14 requires graceful degradation (LLM_UNAVAILABLE) rather "
            "than the run hanging indefinitely on a slow/unresponsive call."
        ),
    )

    # -- §10/§14 budget governor + circuit breaker (llm/governor.py). --
    groq_daily_call_ceiling: int = Field(
        default=500,
        description=(
            "Read from the GROQ_DAILY_CALL_CEILING environment variable. "
            "Hard daily cap on Groq calls across all concurrently-running "
            "reconciliation runs (§8's governance table; §10's 'budget "
            "counter'). Enforced atomically by llm/governor.py so "
            "concurrent runs can't double-count or under-count usage "
            "(§10's revision-history correction)."
        ),
    )
    groq_circuit_breaker_max_consecutive_failures: int = Field(
        default=3,
        description=(
            "Read from the GROQ_CIRCUIT_BREAKER_MAX_CONSECUTIVE_FAILURES "
            "environment variable. After this many consecutive Groq call "
            "failures (timeout/API error/malformed response — detected via "
            "recommender.py's distinct exception types, never string "
            "matching) within one ReconciliationRun, llm/governor.py trips "
            "the breaker and stops calling Groq for the rest of that run. "
            "A rate-limit failure (HTTP 429, LLMRateLimitError) is the one "
            "exception to this threshold — it trips the breaker "
            "immediately, on its first occurrence, regardless of this "
            "setting."
        ),
    )
    governor_db_path: str = Field(
        default="recon_agent_governor.db",
        description=(
            "Read from the GOVERNOR_DB_PATH environment variable. SQLite "
            "file backing the governor's daily-call-budget and circuit- "
            "breaker state. Concurrent access is made safe via a single "
            "atomic transaction per check-and-increment (BEGIN IMMEDIATE) "
            "— see llm/governor.py. Tests typically override this with a "
            "temporary file path so suite runs never share state."
        ),
    )
    stage6_max_candidates: int = Field(
        default=5,
        description=(
            "Read from the STAGE6_MAX_CANDIDATES environment variable. "
            "Cap on how many Stage 5 near-miss candidates are offered to "
            "the LLM per residual record (matching/stage6_llm.py) — keeps "
            "prompts bounded and keeps the per-record Groq call cost "
            "predictable, mirroring §9's other documented search limits."
        ),
    )
    api_db_path: str = Field(
        default="recon_agent_api.db",
        description=(
            "Read from the API_DB_PATH environment variable. SQLite file "
            "backing the Stage 13 FastAPI layer's run persistence "
            "(api/storage.py) — a separate file from governor_db_path so "
            "the API's ReconciliationRun/MatchGroup/DecisionEvent/"
            "Exception rows never share a table namespace with the "
            "budget governor's own counters. Tests typically override "
            "this with a temporary file path."
        ),
    )

    stage1_residual_tolerance_fraction: float = Field(
        default=0.0325,
        description=(
            "Read from the STAGE1_RESIDUAL_TOLERANCE_FRACTION environment "
            "variable. Bounded tolerance for Stage 1 (exact identifier "
            "match) groups' residual between their LEDGER-side gross "
            "amount and their GATEWAY-side (already fee/GST-netted) "
            "amount, expressed as a fraction of the larger of the two "
            "amounts. A shared unique reference is strong evidence on its "
            "own (§2's Stage 1 auto-commit condition doesn't require a "
            "composite score), but without a ceiling that same logic would "
            "silently verify two unrelated transactions that happen to "
            "share a reference by data-quality accident. "
            "\n\n"
            "Default 0.0325 (3.25%) was CALIBRATED, not guessed — see "
            "scripts/calibrate_stage1_residual.py and BUILD_LOG.md's "
            "'Threshold recalibration' entry for the full methodology and "
            "the underlying numbers. Every genuinely correct STAGE1_EXACT "
            "proposal in data/calibration/records.json (n=80, cross-"
            "checked against calibration ground truth's match_groups, "
            "unresolved, and duplicates/honest_abstention categories) has "
            "a residual between 0% and 2.9562% of the larger amount, "
            "clustering at exactly the five MDR-tier + 18%-GST fee "
            "percentages testdata/generator.py's MDR_BPS_CHOICES produces "
            "(1.77%, 2.065%, 2.36%, 2.655%, 2.95%). 3.25% sits a "
            "deliberate ~0.3-percentage-point margin above that observed "
            "maximum — comfortably covering every real fee tier in the "
            "data plus rounding noise, while remaining a real, meaningful "
            "ceiling rather than one raised so high the check stops "
            "firing (it is still well below, e.g., a second unrelated "
            "transaction's amount landing in the same identifier-cluster "
            "by data-quality accident). "
            "\n\n"
            "Known, accepted consequence: this calibrated value is loose "
            "enough that the repo's own hand-crafted reproduction case "
            "(demo_ledger_payment_002/demo_gateway_settlement_002, "
            "residual ~2.9502%) now auto-verifies instead of routing to "
            "PENDING_REVIEW, because its residual is numerically "
            "indistinguishable from — and in fact slightly smaller than — "
            "genuine top-tier MDR+GST variance seen throughout real "
            "calibration data. No magnitude-only threshold can separate "
            "that case from real data without also cutting into "
            "genuinely correct matches; catching it would need a "
            "different signal entirely (e.g. requiring a corroborating "
            "FEE record), which is a logic change, not a threshold change "
            "— see tests/test_repro_bugfix_dataset.py's updated "
            "expectations and BUILD_LOG.md for the full discussion. See "
            "verification/verifier.py's residual check, which "
            "independently re-derives this residual from each member's "
            "underlying record rather than trusting the group's own "
            "stored expected/matched/residual fields."
        ),
    )

    min_score_margin_by_threshold_version: dict[str, float] = Field(
        default_factory=lambda: {"v1": 0.05},
        description=(
            "The Financial and Evidence Verifier's own minimum "
            "score_margin bar for Stage 2-5 groups (§2's shared evidence "
            "bar), keyed by threshold_version exactly like "
            "MatchGroup.threshold_version and mirroring the "
            "MATCH_THRESHOLDS-by-version pattern in "
            "matching/stage2_constrained.py. Deliberately separate from "
            "each matching stage's own MIN_SCORE_MARGIN /"
            " STAGE{3,4}_MIN_MARGIN constants — this is the verifier's "
            "independent re-check, not a shared constant with the "
            "proposing stages (§2's 'necessary but not sufficient')."
        ),
    )
    default_min_score_margin: float = Field(
        default=0.05,
        description=(
            "Fallback margin bar when a MatchGroup's threshold_version "
            "has no entry in min_score_margin_by_threshold_version."
        ),
    )

    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> "Settings":
        """Load settings from the process environment, optionally
        reading a .env file first (see .env.example for the expected
        keys)."""
        load_dotenv(dotenv_path=env_file)

        return cls(
            rules_version=os.environ.get("RULES_VERSION", "v1"),
            model_version=os.environ.get("MODEL_VERSION", "v1"),
            threshold_version=os.environ.get("THRESHOLD_VERSION", "v1"),
            policy_version=os.environ.get("POLICY_VERSION", "v1"),
            aggregation_search=AggregationSearchLimits(
                max_candidates_per_window=int(
                    os.environ.get("MAX_CANDIDATES_PER_WINDOW", 20)
                ),
                max_group_size=int(os.environ.get("MAX_GROUP_SIZE", 8)),
                window_timeout_seconds=float(
                    os.environ.get("WINDOW_TIMEOUT_SECONDS", 2.0)
                ),
            ),
            groq_api_key=os.environ.get("GROQ_API_KEY"),
            groq_model=os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"),
            groq_timeout_seconds=float(os.environ.get("GROQ_TIMEOUT_SECONDS", 15.0)),
            stage1_residual_tolerance_fraction=float(
                os.environ.get("STAGE1_RESIDUAL_TOLERANCE_FRACTION", 0.0325)
            ),
            groq_daily_call_ceiling=int(
                os.environ.get("GROQ_DAILY_CALL_CEILING", 500)
            ),
            groq_circuit_breaker_max_consecutive_failures=int(
                os.environ.get(
                    "GROQ_CIRCUIT_BREAKER_MAX_CONSECUTIVE_FAILURES", 3
                )
            ),
            governor_db_path=os.environ.get(
                "GOVERNOR_DB_PATH", "recon_agent_governor.db"
            ),
            stage6_max_candidates=int(
                os.environ.get("STAGE6_MAX_CANDIDATES", 5)
            ),
            api_db_path=os.environ.get("API_DB_PATH", "recon_agent_api.db"),
        )


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Return process-wide settings, loading from the environment on
    first call and caching the result."""
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings
