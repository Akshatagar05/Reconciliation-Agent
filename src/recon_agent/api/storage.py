"""SQLite persistence for API-triggered pipeline runs — Stage 13.

Pipeline runs aren't persisted anywhere else in this codebase: every
CLI/test invocation of ``matching.pipeline.run_pipeline`` or
``evaluation.harness.run_evaluation`` just runs in-memory and the
caller does whatever it likes with the returned
``MatchingResult``/``EvaluationReport``. Since a POST to this stage's
``/reconcile`` endpoint needs to be looked up later by ``run_id`` via
the GET endpoints, this module adds the minimal SQLite-backed store
needed for that — matching the rest of the stack's storage choice
(``llm/governor.py`` is also plain ``sqlite3``, no ORM).

Deliberately NOT a new schema: every row stored here is just the JSON
form of an existing Pydantic model (``ReconciliationRun``, ``MatchGroup``,
``MatchGroupMember``, ``DecisionEvent``, ``ReconciliationException``,
``NormalizedRecord``) produced by that model's own ``model_dump_json``/
``model_dump``, round-tripped back through ``model_validate`` on read.
This module only adds the run_id-keyed lookup tables around those
models; it does not invent any new fields or relationships.

Composite (not single-column) primary keys on group_id/event_id: note
that ``matching.pipeline.run_pipeline`` allocates ids via a fresh
``IdAllocator()`` (default prefix ``"match"``) on every call, so
``group_id``/``event_id`` values are only unique *within* one run, not
across runs (a second POST /reconcile produces "match_grp_00001" again).
Every table here is therefore keyed by ``(run_id, group_id)`` /
``(run_id, event_id)``, never by the bare id column alone.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Optional

from recon_agent.models import (
    DecisionEvent,
    MatchGroup,
    MatchGroupMember,
    NormalizedRecord,
    ReconciliationException,
    ReconciliationRun,
)


class RunStore:
    """SQLite-backed store for reconciliation runs and everything a run
    produced, keyed by ``run_id``.

    No connection is held on ``self`` across calls. Every public method
    below opens its own fresh ``sqlite3.Connection`` (via ``_connect()``),
    does its work, and closes it before returning — see the "Concurrency"
    note below for why.

    Each write is wrapped in its own ``BEGIN IMMEDIATE`` transaction (on
    that call's own connection) so a run's rows are all written atomically
    or not at all; that part of the design is unchanged.

    Concurrency: this class is called from FastAPI's sync path, which
    Starlette runs on a threadpool — so concurrent requests really do
    call into this class from different threads at the same time. A
    single shared ``sqlite3.Connection`` reused across those threads
    (even with ``check_same_thread=False``) is not safe for concurrent
    use: the Python sqlite3 module tracks a connection's "am I mid-
    transaction" state on the ``Connection`` object itself, not per
    thread, so two threads issuing ``BEGIN IMMEDIATE``/statements on the
    *same* connection object at overlapping times can corrupt each
    other's in-flight transaction (this is exactly what a production
    ``POST /reconcile`` colliding with concurrent ``GET`` traffic hit —
    see BUILD_LOG.md). Opening a fresh, short-lived connection per call
    sidesteps this entirely: each request gets its own connection object
    for the lifetime of that one call, so there is never a connection
    object visible to more than one thread at a time. SQLite itself
    still serializes the underlying file access across those separate
    connections via ``BEGIN IMMEDIATE`` + ``PRAGMA busy_timeout``, same
    as before.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        """Open a fresh connection, scoped to the caller's own use (one
        request's worth of work) — never stored on ``self`` and never
        shared across threads/requests."""
        conn = sqlite3.connect(self._db_path, timeout=30.0, isolation_level=None)
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn

    def _init_schema(self) -> None:
        conn = self._connect()
        try:
            self._create_tables(conn)
        finally:
            conn.close()

    def _create_tables(self, conn: sqlite3.Connection) -> None:
        c = conn
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS reconciliation_runs (
                run_id TEXT PRIMARY KEY,
                run_data TEXT NOT NULL,
                dataset TEXT,
                ground_truth_data TEXT,
                elapsed_seconds REAL NOT NULL,
                records_data TEXT NOT NULL
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS match_groups (
                run_id TEXT NOT NULL,
                group_id TEXT NOT NULL,
                stage TEXT,
                data TEXT NOT NULL,
                PRIMARY KEY (run_id, group_id)
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS match_group_members (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                group_id TEXT NOT NULL,
                data TEXT NOT NULL
            )
            """
        )
        c.execute("CREATE INDEX IF NOT EXISTS idx_members_run ON match_group_members(run_id)")
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS decision_events (
                run_id TEXT NOT NULL,
                event_id TEXT NOT NULL,
                data TEXT NOT NULL,
                PRIMARY KEY (run_id, event_id)
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS exceptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                data TEXT NOT NULL
            )
            """
        )
        c.execute("CREATE INDEX IF NOT EXISTS idx_exceptions_run ON exceptions(run_id)")

    # -----------------------------------------------------------------
    # Write
    # -----------------------------------------------------------------
    def save_run(
        self,
        run: ReconciliationRun,
        records: list[NormalizedRecord],
        match_groups: list[MatchGroup],
        match_group_members: list[MatchGroupMember],
        decision_events: list[DecisionEvent],
        exceptions: list[ReconciliationException],
        group_id_to_stage: dict[str, str],
        elapsed_seconds: float,
        dataset: Optional[str] = None,
        ground_truth: Optional[dict[str, Any]] = None,
    ) -> None:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT INTO reconciliation_runs "
                "(run_id, run_data, dataset, ground_truth_data, elapsed_seconds, records_data) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    run.run_id,
                    run.model_dump_json(),
                    dataset,
                    json.dumps(ground_truth) if ground_truth is not None else None,
                    elapsed_seconds,
                    json.dumps([r.model_dump(mode="json") for r in records]),
                ),
            )
            for g in match_groups:
                conn.execute(
                    "INSERT INTO match_groups (run_id, group_id, stage, data) VALUES (?, ?, ?, ?)",
                    (run.run_id, g.group_id, group_id_to_stage.get(g.group_id), g.model_dump_json()),
                )
            for m in match_group_members:
                conn.execute(
                    "INSERT INTO match_group_members (run_id, group_id, data) VALUES (?, ?, ?)",
                    (run.run_id, m.group_id, m.model_dump_json()),
                )
            for e in decision_events:
                conn.execute(
                    "INSERT INTO decision_events (run_id, event_id, data) VALUES (?, ?, ?)",
                    (run.run_id, e.event_id, e.model_dump_json()),
                )
            for exc in exceptions:
                conn.execute(
                    "INSERT INTO exceptions (run_id, data) VALUES (?, ?)",
                    (run.run_id, exc.model_dump_json()),
                )
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    # -----------------------------------------------------------------
    # Write — single-group human review (see api/app.py's PATCH
    # /report/{run_id}/groups/{group_id}/review)
    # -----------------------------------------------------------------
    def apply_review(self, run_id: str, group: MatchGroup, event: DecisionEvent) -> None:
        """Persist a human reviewer's decision on one already-existing
        ``MatchGroup`` row, plus the ``DecisionEvent`` documenting the
        review itself, atomically — same ``BEGIN IMMEDIATE`` discipline
        as ``save_run`` above, so a partially-applied review (group
        updated but no audit event, or vice versa) is never possible.

        Unlike ``save_run`` (INSERT of a brand-new run's rows), this is
        an UPDATE of one existing ``match_groups`` row keyed by
        ``(run_id, group_id)`` plus an INSERT of one new
        ``decision_events`` row — the caller (``api/service.py``'s
        ``review_group``) is responsible for having already validated
        that the group exists and is in a reviewable state; this method
        only persists the already-decided outcome.
        """
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "UPDATE match_groups SET data = ? WHERE run_id = ? AND group_id = ?",
                (group.model_dump_json(), run_id, group.group_id),
            )
            conn.execute(
                "INSERT INTO decision_events (run_id, event_id, data) VALUES (?, ?, ?)",
                (run_id, event.event_id, event.model_dump_json()),
            )
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    # -----------------------------------------------------------------
    # Read — each helper below opens its own fresh connection (via
    # ``_connect()``/``_fetchone``/``_fetchall``), same rationale as the
    # write paths above: no connection object is ever shared across
    # requests/threads.
    # -----------------------------------------------------------------
    def _fetchone(self, query: str, params: tuple) -> Optional[tuple]:
        conn = self._connect()
        try:
            return conn.execute(query, params).fetchone()
        finally:
            conn.close()

    def _fetchall(self, query: str, params: tuple) -> list[tuple]:
        conn = self._connect()
        try:
            return conn.execute(query, params).fetchall()
        finally:
            conn.close()

    def run_exists(self, run_id: str) -> bool:
        row = self._fetchone("SELECT 1 FROM reconciliation_runs WHERE run_id = ?", (run_id,))
        return row is not None

    def get_run(self, run_id: str) -> Optional[ReconciliationRun]:
        row = self._fetchone("SELECT run_data FROM reconciliation_runs WHERE run_id = ?", (run_id,))
        if row is None:
            return None
        return ReconciliationRun.model_validate_json(row[0])

    def get_dataset(self, run_id: str) -> Optional[str]:
        row = self._fetchone("SELECT dataset FROM reconciliation_runs WHERE run_id = ?", (run_id,))
        return row[0] if row else None

    def get_elapsed_seconds(self, run_id: str) -> Optional[float]:
        row = self._fetchone(
            "SELECT elapsed_seconds FROM reconciliation_runs WHERE run_id = ?", (run_id,)
        )
        return float(row[0]) if row else None

    def get_ground_truth_raw(self, run_id: str) -> Optional[dict[str, Any]]:
        row = self._fetchone(
            "SELECT ground_truth_data FROM reconciliation_runs WHERE run_id = ?", (run_id,)
        )
        if row is None or row[0] is None:
            return None
        return json.loads(row[0])

    def get_records(self, run_id: str) -> list[NormalizedRecord]:
        row = self._fetchone("SELECT records_data FROM reconciliation_runs WHERE run_id = ?", (run_id,))
        if row is None:
            return []
        return [NormalizedRecord.model_validate(r) for r in json.loads(row[0])]

    def get_match_groups(self, run_id: str) -> list[MatchGroup]:
        rows = self._fetchall("SELECT data FROM match_groups WHERE run_id = ?", (run_id,))
        return [MatchGroup.model_validate_json(r[0]) for r in rows]

    def get_match_group(self, run_id: str, group_id: str) -> Optional[MatchGroup]:
        """Single-group lookup, added for the human-review endpoint
        (``PATCH /report/{run_id}/groups/{group_id}/review`` — see
        ``api/app.py``), which needs to check one group's current
        ``status`` before deciding whether a review action is valid."""
        row = self._fetchone(
            "SELECT data FROM match_groups WHERE run_id = ? AND group_id = ?",
            (run_id, group_id),
        )
        if row is None:
            return None
        return MatchGroup.model_validate_json(row[0])

    def get_group_id_to_stage(self, run_id: str) -> dict[str, str]:
        rows = self._fetchall("SELECT group_id, stage FROM match_groups WHERE run_id = ?", (run_id,))
        return {group_id: stage for group_id, stage in rows if stage is not None}

    def get_match_group_members(self, run_id: str) -> list[MatchGroupMember]:
        rows = self._fetchall("SELECT data FROM match_group_members WHERE run_id = ?", (run_id,))
        return [MatchGroupMember.model_validate_json(r[0]) for r in rows]

    def get_decision_events(self, run_id: str) -> list[DecisionEvent]:
        rows = self._fetchall("SELECT data FROM decision_events WHERE run_id = ?", (run_id,))
        return [DecisionEvent.model_validate_json(r[0]) for r in rows]

    def get_exceptions(self, run_id: str) -> list[ReconciliationException]:
        rows = self._fetchall("SELECT data FROM exceptions WHERE run_id = ?", (run_id,))
        return [ReconciliationException.model_validate_json(r[0]) for r in rows]

    def close(self) -> None:
        """No-op: kept for backward compatibility with callers (e.g.
        ``api/app.py``'s ``reset_store_for_tests``) that close a
        previous store before replacing it. There is no long-lived
        connection on ``self`` to close anymore — every operation opens
        and closes its own connection (see class docstring)."""
        return None
