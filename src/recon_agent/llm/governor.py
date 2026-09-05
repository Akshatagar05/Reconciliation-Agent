"""Shared budget governor / circuit breaker — ARCHITECTURE.md §8, §10, §14.

Stage 11 of this 16-stage relay build (see README.md's Build Status). This
module is the "next stage" that llm/recommender.py's own docstring and
README.md's Stage 10 section explicitly deferred: budget/quota accounting
and circuit-breaking around Groq calls. It does not touch
``recommender.py``'s prompt construction, API call, or response-validation
logic at all — it only decides, before and after each call, whether the
call should happen and what to do when it shouldn't.

Two independent gates, per this stage's brief:

  1. **Daily call ceiling** (§8's governance table) — a hard cap on Groq
     calls per day, shared across every concurrently-running
     ``ReconciliationRun``. Scoped by calendar day only (not by run), since
     the ceiling is a budget concern (Groq spend/day), not a per-run
     concern.

  2. **Circuit breaker** — after
     ``Settings.groq_circuit_breaker_max_consecutive_failures`` consecutive
     call failures *within one run*, stop calling Groq for the rest of
     that run. Scoped by ``run_id`` (the ``ReconciliationRun``-level
     tracking §10 points at), since one run's bad luck (a transient Groq
     outage during its window) shouldn't necessarily veto a different,
     later run's ability to try. One exception to the "consecutive
     failures" counting: an ``LLMRateLimitError`` (HTTP 429 from Groq)
     trips the breaker immediately, on its first occurrence — see
     ``record_failure`` below for why a rate limit doesn't wait for the
     same threshold a transient/generic failure does.

Both gates are backed by SQLite with a single atomic transaction
(``BEGIN IMMEDIATE``) per check-and-mutate operation — this was an
explicit correction in ARCHITECTURE.md's revision history (§10: "budget
counter now transactional — atomic increment ... so concurrent requests
can't double-count or under-count Groq call usage") and is the reason
``try_consume`` below folds the ceiling/breaker *check* and the usage
*increment* into one transaction rather than two separate calls — a
check-then-increment pair done as two round trips is exactly the
time-of-check/time-of-use race the correction exists to close.

Detecting a "failure" for the circuit breaker is done exclusively via
``isinstance`` checks against ``recommender.py``'s own distinct exception
types (``RecommenderError`` and its subclasses) — never by inspecting an
exception's string message. Any ``RecommenderError`` — timeout, API
error, malformed JSON, schema violation, or a hallucinated candidate —
counts as one failed attempt at producing a usable recommendation,
regardless of which specific subclass it is; ``classify_failure`` below
exists only to label *which* subclass fired, for the audit trail
(DecisionEvent/Exception evidence), not to change whether it counts.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Optional

from recon_agent.config import Settings
from recon_agent.llm.recommender import (
    LLMAPIError,
    LLMHallucinatedCandidateError,
    LLMMalformedJSONError,
    LLMRateLimitError,
    LLMSchemaValidationError,
    LLMTimeoutError,
    RecommenderError,
)

# ---------------------------------------------------------------------------
# Failure classification (audit-trail labeling only — see module docstring).
# ---------------------------------------------------------------------------

# LLMRateLimitError is checked ahead of LLMAPIError below — it's a
# subclass of LLMAPIError (see recommender.py), and isinstance checks in
# this tuple are matched in order, so the more specific type must come
# first or every rate limit would be mislabeled as a generic API_ERROR.
_FAILURE_KIND_BY_EXCEPTION_TYPE: tuple[tuple[type[RecommenderError], str], ...] = (
    (LLMTimeoutError, "TIMEOUT"),
    (LLMRateLimitError, "RATE_LIMITED"),
    (LLMAPIError, "API_ERROR"),
    (LLMMalformedJSONError, "MALFORMED_RESPONSE"),
    (LLMSchemaValidationError, "MALFORMED_RESPONSE"),
    (LLMHallucinatedCandidateError, "MALFORMED_RESPONSE"),
)


def classify_failure(exc: BaseException) -> str:
    """Label which distinct recommender.py failure mode ``exc`` is, via
    ``isinstance`` against its real exception types — never string
    matching. Returns ``"UNKNOWN_FAILURE"`` for anything that isn't a
    recognized ``RecommenderError`` subclass (should not happen in
    practice, since ``record_failure`` below only accepts
    ``RecommenderError`` instances at all)."""
    for exc_type, kind in _FAILURE_KIND_BY_EXCEPTION_TYPE:
        if isinstance(exc, exc_type):
            return kind
    return "UNKNOWN_FAILURE"


# ---------------------------------------------------------------------------
# Public result shape
# ---------------------------------------------------------------------------


class GovernorDecision(str, Enum):
    """The governor's cleanly-queryable decision — so a caller (the
    pipeline) can branch on graceful degradation without inspecting
    internal counters itself."""

    ALLOW = "ALLOW"
    CEILING_EXCEEDED = "CEILING_EXCEEDED"
    CIRCUIT_OPEN = "CIRCUIT_OPEN"


@dataclass(frozen=True)
class GovernorCheckResult:
    """A snapshot of the governor's state at the moment of the call,
    returned by every public method below so callers never need a
    separate follow-up query to explain *why* a decision came back the
    way it did."""

    decision: GovernorDecision
    calls_used_today: int
    ceiling: int
    consecutive_failures: int
    circuit_open: bool

    @property
    def allowed(self) -> bool:
        return self.decision == GovernorDecision.ALLOW


# ---------------------------------------------------------------------------
# The governor itself
# ---------------------------------------------------------------------------


class CallBudgetGovernor:
    """SQLite-backed budget governor / circuit breaker shared across
    concurrently-running reconciliation runs.

    ``db_path`` defaults to ``settings.governor_db_path`` — pass an
    explicit path (or ``":memory:"`` for a single-process, single-
    connection test) to isolate one governor's state from another's,
    since every governor instance pointed at the same path shares the
    same durable counters by design (that sharing is exactly what makes
    the daily ceiling meaningful across concurrent runs).

    Every public method that reads-then-writes state does so inside one
    ``BEGIN IMMEDIATE`` transaction — see module docstring on why
    check-then-increment must be one atomic step, not two.

    Concurrency: no connection is held on ``self`` across calls. Each
    public method opens its own fresh ``sqlite3.Connection`` (via
    ``_connect()``), does its one atomic operation, and closes it. A
    single ``CallBudgetGovernor`` instance (and the connection it used
    to hold) is reached from every concurrent request's thread (see
    ``api/app.py`` — FastAPI's sync endpoints run on a threadpool), so a
    shared, long-lived connection object here has the exact same
    cross-thread transaction-collision risk that ``api/storage.py``'s
    ``RunStore`` had (see that module's docstring and BUILD_LOG.md for
    the full diagnosis) — this class was built with the same shared-
    connection pattern and needed the same fix, not a different one.
    SQLite still serializes the underlying file access across those
    separate connections via ``BEGIN IMMEDIATE`` + ``PRAGMA
    busy_timeout``, so the atomic check-and-increment guarantee the
    module docstring describes is unchanged.
    """

    def __init__(
        self,
        settings: Settings,
        db_path: Optional[str] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self._ceiling = settings.groq_daily_call_ceiling
        self._breaker_threshold = settings.groq_circuit_breaker_max_consecutive_failures
        self._db_path = db_path if db_path is not None else settings.governor_db_path
        self._clock = clock if clock is not None else (lambda: datetime.now(timezone.utc))
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        """Open a fresh connection, scoped to the caller's own use (one
        method call) — never stored on ``self`` and never shared across
        threads/requests.

        ``isolation_level=None`` -> autocommit mode: schema DDL and
        plain reads run without an implicit transaction, and every
        mutating operation issues its own explicit BEGIN IMMEDIATE /
        COMMIT / ROLLBACK rather than fighting the driver's own implicit
        transaction management.
        """
        conn = sqlite3.connect(self._db_path, timeout=30.0, isolation_level=None)
        # Concurrent writers block (up to 30s) rather than immediately
        # raising "database is locked" — necessary for BEGIN IMMEDIATE to
        # actually serialize concurrent callers instead of erroring one
        # of them out.
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn

    def _init_schema(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_call_budget (
                    day TEXT PRIMARY KEY,
                    calls_used INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS circuit_breaker_state (
                    run_id TEXT PRIMARY KEY,
                    consecutive_failures INTEGER NOT NULL DEFAULT 0,
                    circuit_open INTEGER NOT NULL DEFAULT 0
                )
                """
            )
        finally:
            conn.close()

    def _today(self) -> str:
        return self._clock().strftime("%Y-%m-%d")

    def _read_breaker(self, conn: sqlite3.Connection, run_id: str) -> tuple[int, bool]:
        row = conn.execute(
            "SELECT consecutive_failures, circuit_open FROM circuit_breaker_state "
            "WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            return 0, False
        return int(row[0]), bool(row[1])

    def _read_calls_used(self, conn: sqlite3.Connection, day: str) -> int:
        row = conn.execute(
            "SELECT calls_used FROM daily_call_budget WHERE day = ?", (day,)
        ).fetchone()
        return int(row[0]) if row is not None else 0

    def _result(
        self, decision: GovernorDecision, calls_used: int, consecutive_failures: int, circuit_open: bool
    ) -> GovernorCheckResult:
        return GovernorCheckResult(
            decision=decision,
            calls_used_today=calls_used,
            ceiling=self._ceiling,
            consecutive_failures=consecutive_failures,
            circuit_open=circuit_open,
        )

    # -----------------------------------------------------------------
    # Read-only query — never mutates. Lets the pipeline check the
    # governor's state (e.g. to decide whether it's even worth gathering
    # candidates for a residual record) without spending a call slot.
    # -----------------------------------------------------------------
    def status(self, run_id: str) -> GovernorCheckResult:
        conn = self._connect()
        try:
            consecutive_failures, circuit_open = self._read_breaker(conn, run_id)
            calls_used = self._read_calls_used(conn, self._today())
        finally:
            conn.close()
        if circuit_open:
            decision = GovernorDecision.CIRCUIT_OPEN
        elif calls_used >= self._ceiling:
            decision = GovernorDecision.CEILING_EXCEEDED
        else:
            decision = GovernorDecision.ALLOW
        return self._result(decision, calls_used, consecutive_failures, circuit_open)

    # -----------------------------------------------------------------
    # The atomic check-and-increment gate. Call this immediately before
    # actually calling Groq; only proceed with the call if the returned
    # result is ALLOW.
    # -----------------------------------------------------------------
    def try_consume(self, run_id: str) -> GovernorCheckResult:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            consecutive_failures, circuit_open = self._read_breaker(conn, run_id)
            if circuit_open:
                conn.commit()
                return self._result(
                    GovernorDecision.CIRCUIT_OPEN,
                    self._read_calls_used(conn, self._today()),
                    consecutive_failures,
                    True,
                )

            day = self._today()
            calls_used = self._read_calls_used(conn, day)
            if calls_used >= self._ceiling:
                conn.commit()
                return self._result(
                    GovernorDecision.CEILING_EXCEEDED, calls_used, consecutive_failures, False
                )

            new_calls_used = calls_used + 1
            conn.execute(
                "INSERT INTO daily_call_budget (day, calls_used) VALUES (?, ?) "
                "ON CONFLICT(day) DO UPDATE SET calls_used = excluded.calls_used",
                (day, new_calls_used),
            )
            conn.commit()
            return self._result(
                GovernorDecision.ALLOW, new_calls_used, consecutive_failures, False
            )
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    # -----------------------------------------------------------------
    # Post-call bookkeeping — always call exactly one of these after a
    # call that ``try_consume`` allowed.
    # -----------------------------------------------------------------
    def record_success(self, run_id: str) -> GovernorCheckResult:
        """Reset the consecutive-failure counter after a successful
        call. Deliberately never clears ``circuit_open`` — once tripped,
        the breaker stays open for the rest of the run (§8); a success
        can only happen before that point anyway, since ``try_consume``
        refuses calls once the breaker is open."""
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT INTO circuit_breaker_state (run_id, consecutive_failures, circuit_open) "
                "VALUES (?, 0, 0) "
                "ON CONFLICT(run_id) DO UPDATE SET consecutive_failures = 0",
                (run_id,),
            )
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()
        return self.status(run_id)

    def record_failure(self, run_id: str, exc: BaseException) -> GovernorCheckResult:
        """Record one failed call attempt and trip the breaker if this
        pushes ``consecutive_failures`` to the configured threshold —
        or immediately, regardless of threshold, if ``exc`` is an
        ``LLMRateLimitError`` (see below).

        Raises ``TypeError`` if ``exc`` is not a ``recommender.RecommenderError``
        — the whole point of this method is detecting failures via real
        exception types, never by pattern-matching a message string, so
        anything else is a caller bug, not a recommender-shaped failure.
        """
        if not isinstance(exc, RecommenderError):
            raise TypeError(
                "CallBudgetGovernor.record_failure requires a "
                "recon_agent.llm.recommender.RecommenderError instance (detected "
                "via isinstance, never string matching) so the circuit breaker "
                f"only trips on genuine recommender failures; got {type(exc)!r}."
            )

        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            prior_failures, circuit_open = self._read_breaker(conn, run_id)
            consecutive_failures = prior_failures + 1
            # A Groq rate limit (HTTP 429) is a distinct, unambiguous
            # signal — unlike a transient timeout/5xx/malformed-response,
            # it means every further call this run is very likely to
            # fail the same way (the account/key is over its limit right
            # now). Waiting for the same N-consecutive-failures threshold
            # generic failures need would mean burning several more real
            # (failing) Groq calls, and the latency of each, before
            # degrading — exactly the stall this exists to prevent. So a
            # rate limit trips the breaker on its very first occurrence,
            # regardless of ``consecutive_failures`` or the configured
            # threshold; every other RecommenderError subclass keeps the
            # existing N-consecutive-failures behavior unchanged.
            if consecutive_failures >= self._breaker_threshold or isinstance(
                exc, LLMRateLimitError
            ):
                circuit_open = True
            conn.execute(
                "INSERT INTO circuit_breaker_state (run_id, consecutive_failures, circuit_open) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(run_id) DO UPDATE SET "
                "consecutive_failures = excluded.consecutive_failures, "
                "circuit_open = excluded.circuit_open",
                (run_id, consecutive_failures, int(circuit_open)),
            )
            conn.commit()
            decision = GovernorDecision.CIRCUIT_OPEN if circuit_open else GovernorDecision.ALLOW
            return self._result(
                decision, self._read_calls_used(conn, self._today()), consecutive_failures, circuit_open
            )
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def close(self) -> None:
        """No-op: kept for backward compatibility with callers (e.g.
        ``api/app.py``'s ``get_llm_budget``) that call ``close()`` after
        using a governor instance. There is no long-lived connection on
        ``self`` to close anymore — every operation opens and closes its
        own connection (see class docstring)."""
        return None
