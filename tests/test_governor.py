"""Unit tests for src/recon_agent/llm/governor.py (Stage 11).

Coverage, per this stage's required scenarios:
  - daily ceiling enforcement (and that it's shared across concurrently
    running reconciliation runs, i.e. scoped by day, not by run_id)
  - atomic counting under simulated concurrent access (separate
    governor instances / separate SQLite connections pointed at the
    same file, hammered from multiple threads) — the calls_used counter
    must land exactly on min(attempts, ceiling), never over or under
  - circuit breaker tripping after N consecutive failures, using
    recommender.py's real exception types (never string matching), and
    correctly refusing further calls afterward
  - the breaker is scoped per run_id, the ceiling is scoped globally
  - record_failure rejects anything that isn't a RecommenderError
  - classify_failure labels each of the five failure types correctly
"""

from __future__ import annotations

import os
import tempfile
import threading
from datetime import datetime, timezone

import pytest

from recon_agent.config import Settings
from recon_agent.llm.governor import (
    CallBudgetGovernor,
    GovernorDecision,
    classify_failure,
)
from recon_agent.llm.recommender import (
    LLMAPIError,
    LLMHallucinatedCandidateError,
    LLMMalformedJSONError,
    LLMRateLimitError,
    LLMSchemaValidationError,
    LLMTimeoutError,
    RecommenderError,
)


@pytest.fixture()
def db_path(tmp_path) -> str:
    return str(tmp_path / "governor_test.db")


def _settings(ceiling: int = 500, breaker_threshold: int = 3) -> Settings:
    return Settings(
        groq_daily_call_ceiling=ceiling,
        groq_circuit_breaker_max_consecutive_failures=breaker_threshold,
    )


# ---------------------------------------------------------------------------
# Ceiling enforcement
# ---------------------------------------------------------------------------


def test_ceiling_allows_up_to_the_configured_limit(db_path: str) -> None:
    governor = CallBudgetGovernor(_settings(ceiling=3), db_path=db_path)
    results = [governor.try_consume("run-a") for _ in range(5)]

    decisions = [r.decision for r in results]
    assert decisions == [
        GovernorDecision.ALLOW,
        GovernorDecision.ALLOW,
        GovernorDecision.ALLOW,
        GovernorDecision.CEILING_EXCEEDED,
        GovernorDecision.CEILING_EXCEEDED,
    ]
    assert results[2].calls_used_today == 3
    # A refused call never increments the counter.
    assert results[4].calls_used_today == 3


def test_ceiling_is_shared_across_different_run_ids(db_path: str) -> None:
    """The daily ceiling is a budget concern (Groq spend/day), not a
    per-run concern — two different runs draw from the same pool."""
    governor = CallBudgetGovernor(_settings(ceiling=2), db_path=db_path)

    assert governor.try_consume("run-a").decision == GovernorDecision.ALLOW
    assert governor.try_consume("run-b").decision == GovernorDecision.ALLOW
    # Third call, regardless of which run_id, exceeds the shared ceiling.
    assert governor.try_consume("run-a").decision == GovernorDecision.CEILING_EXCEEDED
    assert governor.try_consume("run-c").decision == GovernorDecision.CEILING_EXCEEDED


def test_status_is_read_only_and_does_not_consume_budget(db_path: str) -> None:
    governor = CallBudgetGovernor(_settings(ceiling=1), db_path=db_path)

    before = governor.status("run-a")
    assert before.decision == GovernorDecision.ALLOW
    assert before.calls_used_today == 0

    # Querying status repeatedly must never itself use up the ceiling.
    for _ in range(5):
        assert governor.status("run-a").calls_used_today == 0

    consumed = governor.try_consume("run-a")
    assert consumed.decision == GovernorDecision.ALLOW
    assert governor.status("run-a").calls_used_today == 1


def test_ceiling_resets_on_a_new_day(db_path: str) -> None:
    day_one = datetime(2026, 1, 1, tzinfo=timezone.utc)
    day_two = datetime(2026, 1, 2, tzinfo=timezone.utc)
    clock_state = {"now": day_one}
    governor = CallBudgetGovernor(
        _settings(ceiling=1), db_path=db_path, clock=lambda: clock_state["now"]
    )

    assert governor.try_consume("run-a").decision == GovernorDecision.ALLOW
    assert governor.try_consume("run-a").decision == GovernorDecision.CEILING_EXCEEDED

    clock_state["now"] = day_two
    assert governor.try_consume("run-a").decision == GovernorDecision.ALLOW


# ---------------------------------------------------------------------------
# Atomic counting under simulated concurrent access
# ---------------------------------------------------------------------------


def test_atomic_counting_under_concurrent_access_never_double_or_under_counts(
    db_path: str,
) -> None:
    """Separate CallBudgetGovernor instances (separate SQLite
    connections) pointed at the same file, hammered from many threads
    at once — models concurrent reconciliation runs sharing one budget
    database. The final calls_used must land exactly on
    min(total_attempts, ceiling); a check-then-increment race would
    show up here as either over-counting (lost updates letting more
    than `ceiling` calls through) or under-counting (a torn read
    letting the counter fall behind actual ALLOW decisions).
    """
    ceiling = 50
    num_threads = 20
    attempts_per_thread = 10  # 200 total attempts against a ceiling of 50
    settings = _settings(ceiling=ceiling)

    allow_count_lock = threading.Lock()
    allow_count = {"n": 0}

    def worker() -> None:
        # Each thread gets its OWN governor / SQLite connection pointed
        # at the same file, mirroring separate concurrent processes.
        governor = CallBudgetGovernor(settings, db_path=db_path)
        local_allows = 0
        for _ in range(attempts_per_thread):
            result = governor.try_consume("shared-run")
            if result.decision == GovernorDecision.ALLOW:
                local_allows += 1
        governor.close()
        with allow_count_lock:
            allow_count["n"] += local_allows

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    final_governor = CallBudgetGovernor(settings, db_path=db_path)
    final_status = final_governor.status("shared-run")

    assert allow_count["n"] == ceiling, (
        f"Expected exactly {ceiling} ALLOW decisions across "
        f"{num_threads * attempts_per_thread} concurrent attempts, got "
        f"{allow_count['n']} — a race would show up as a different number."
    )
    assert final_status.calls_used_today == ceiling, (
        f"Expected the durable counter to read back exactly {ceiling}, got "
        f"{final_status.calls_used_today} — double-counting or "
        "under-counting under concurrent access."
    )


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


def test_circuit_breaker_trips_after_n_consecutive_failures(db_path: str) -> None:
    governor = CallBudgetGovernor(_settings(breaker_threshold=3), db_path=db_path)

    r1 = governor.record_failure("run-a", LLMTimeoutError("timed out"))
    assert r1.decision == GovernorDecision.ALLOW
    assert r1.consecutive_failures == 1
    assert r1.circuit_open is False

    r2 = governor.record_failure("run-a", LLMAPIError("boom"))
    assert r2.decision == GovernorDecision.ALLOW
    assert r2.consecutive_failures == 2
    assert r2.circuit_open is False

    r3 = governor.record_failure("run-a", LLMMalformedJSONError("not json"))
    assert r3.decision == GovernorDecision.CIRCUIT_OPEN
    assert r3.consecutive_failures == 3
    assert r3.circuit_open is True


def test_circuit_breaker_refuses_further_calls_once_tripped(db_path: str) -> None:
    governor = CallBudgetGovernor(_settings(breaker_threshold=2), db_path=db_path)

    governor.record_failure("run-a", LLMTimeoutError("t1"))
    governor.record_failure("run-a", LLMTimeoutError("t2"))

    # The breaker being open must actually block try_consume, not just
    # be a recorded fact nobody checks.
    check = governor.try_consume("run-a")
    assert check.decision == GovernorDecision.CIRCUIT_OPEN
    # And it must not have also spent a budget slot on the refused call.
    assert check.calls_used_today == 0

    # Still refuses on subsequent attempts, not just the first one after
    # tripping.
    for _ in range(5):
        assert governor.try_consume("run-a").decision == GovernorDecision.CIRCUIT_OPEN


def test_circuit_breaker_is_scoped_per_run_id(db_path: str) -> None:
    governor = CallBudgetGovernor(_settings(breaker_threshold=2), db_path=db_path)

    governor.record_failure("run-a", LLMTimeoutError("t1"))
    governor.record_failure("run-a", LLMTimeoutError("t2"))
    assert governor.status("run-a").circuit_open is True

    # A different run_id's breaker is independent — one run's Groq
    # trouble doesn't veto a different run.
    assert governor.status("run-b").circuit_open is False
    assert governor.try_consume("run-b").decision == GovernorDecision.ALLOW


# ---------------------------------------------------------------------------
# Rate limit (HTTP 429) — trips the breaker immediately, unlike every
# other failure type, which needs the configured consecutive-failure
# threshold.
# ---------------------------------------------------------------------------


def test_rate_limit_trips_circuit_breaker_on_first_occurrence(db_path: str) -> None:
    """A high threshold (5) would normally mean 5 consecutive failures
    are needed — but a single LLMRateLimitError must trip the breaker
    immediately regardless, since retrying into an active Groq rate
    limit is expected to keep failing the same way."""
    governor = CallBudgetGovernor(_settings(breaker_threshold=5), db_path=db_path)

    r1 = governor.record_failure("run-a", LLMRateLimitError("429 rate limited"))

    assert r1.decision == GovernorDecision.CIRCUIT_OPEN
    assert r1.circuit_open is True
    assert r1.consecutive_failures == 1


def test_rate_limit_immediately_blocks_further_calls_this_run(db_path: str) -> None:
    governor = CallBudgetGovernor(_settings(breaker_threshold=10), db_path=db_path)

    governor.record_failure("run-a", LLMRateLimitError("429 rate limited"))

    check = governor.try_consume("run-a")
    assert check.decision == GovernorDecision.CIRCUIT_OPEN
    assert check.calls_used_today == 0


def test_non_rate_limit_failures_still_need_the_configured_threshold(
    db_path: str,
) -> None:
    """Confirms the immediate-trip behavior is specific to
    LLMRateLimitError — a generic LLMAPIError (e.g. a 5xx) still needs
    the full configured number of consecutive failures, unchanged."""
    governor = CallBudgetGovernor(_settings(breaker_threshold=3), db_path=db_path)

    r1 = governor.record_failure("run-a", LLMAPIError("500 internal error"))
    assert r1.decision == GovernorDecision.ALLOW
    assert r1.circuit_open is False

    r2 = governor.record_failure("run-a", LLMAPIError("500 internal error"))
    assert r2.decision == GovernorDecision.ALLOW
    assert r2.circuit_open is False


def test_rate_limit_is_labeled_distinctly_by_classify_failure() -> None:
    assert classify_failure(LLMRateLimitError("429")) == "RATE_LIMITED"
    # And still isn't confused with a generic API error.
    assert classify_failure(LLMAPIError("500")) == "API_ERROR"


def test_record_success_resets_consecutive_failures_but_not_an_open_circuit(
    db_path: str,
) -> None:
    governor = CallBudgetGovernor(_settings(breaker_threshold=5), db_path=db_path)

    governor.record_failure("run-a", LLMTimeoutError("t1"))
    governor.record_failure("run-a", LLMTimeoutError("t2"))
    assert governor.status("run-a").consecutive_failures == 2

    governor.record_success("run-a")
    status = governor.status("run-a")
    assert status.consecutive_failures == 0
    assert status.circuit_open is False


def test_record_failure_rejects_non_recommender_errors(db_path: str) -> None:
    governor = CallBudgetGovernor(_settings(), db_path=db_path)

    with pytest.raises(TypeError):
        governor.record_failure("run-a", ValueError("not a RecommenderError"))


def test_all_five_recommender_failure_types_count_toward_the_breaker(
    db_path: str,
) -> None:
    """The breaker must detect failures via real exception types (
    isinstance), not string matching — exercise each of
    recommender.py's five distinct failure types and confirm every one
    of them counts as a consecutive failure."""
    failure_instances: list[RecommenderError] = [
        LLMTimeoutError("timeout"),
        LLMAPIError("api error"),
        LLMMalformedJSONError("bad json"),
        LLMSchemaValidationError("bad schema"),
        LLMHallucinatedCandidateError("hallucinated"),
    ]
    governor = CallBudgetGovernor(
        _settings(breaker_threshold=len(failure_instances)), db_path=db_path
    )

    result = None
    for exc in failure_instances:
        result = governor.record_failure("run-a", exc)

    assert result is not None
    assert result.consecutive_failures == len(failure_instances)
    assert result.circuit_open is True


@pytest.mark.parametrize(
    "exc, expected_kind",
    [
        (LLMTimeoutError("x"), "TIMEOUT"),
        (LLMRateLimitError("x"), "RATE_LIMITED"),
        (LLMAPIError("x"), "API_ERROR"),
        (LLMMalformedJSONError("x"), "MALFORMED_RESPONSE"),
        (LLMSchemaValidationError("x"), "MALFORMED_RESPONSE"),
        (LLMHallucinatedCandidateError("x"), "MALFORMED_RESPONSE"),
    ],
)
def test_classify_failure_labels_each_type(exc: RecommenderError, expected_kind: str) -> None:
    assert classify_failure(exc) == expected_kind
