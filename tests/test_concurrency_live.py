"""Regression test for the live-traffic concurrency bug fixed in
``api/storage.py`` (``RunStore``) and ``llm/governor.py``
(``CallBudgetGovernor``) — see BUILD_LOG.md.

THE ORIGINAL BUG (found via live testing on a real machine, not a
synthetic/sequential test — see BUILD_LOG.md for the full account): a
``POST /reconcile`` request colliding with a burst of concurrent
``GET /health`` requests caused a hard crash inside ``save_run()``,
which manifested to the client as a 30-second ``ReadTimeout`` (the
crash happened in a background thread and the connection just hung).
Root cause: ``RunStore``/``CallBudgetGovernor`` each held a single
``sqlite3.Connection`` on ``self`` and reused it across every request;
FastAPI's sync endpoints run on Starlette's threadpool, so concurrent
requests really did call into that *same* connection object from
different threads at the same time, and two threads issuing
``BEGIN IMMEDIATE`` on one connection object can corrupt each other's
in-flight transaction.

Every other test in this project (``test_api.py`` included) drives the
API through ``TestClient`` with sequential calls — none of them ever
exercised genuinely concurrent traffic, which is exactly why this bug
went uncaught until it was hit live. This file is deliberately
different: it starts a *real* uvicorn server on a real TCP socket and
fires *real* concurrent HTTP requests at it from multiple threads,
mixing ``POST /reconcile`` with a burst of ``GET /health`` calls — the
same traffic shape that triggered the original crash — and repeats
that mixed burst across several rounds, since this was a race
condition and a single passing round proves very little.
"""

from __future__ import annotations

import importlib
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
import pytest
import uvicorn

from recon_agent import config as config_module
from recon_agent.llm.governor import CallBudgetGovernor
from recon_agent.llm.recommender import LLMTimeoutError

app_module = importlib.import_module("recon_agent.api.app")

# Reuse the same real, hand-picked refund triad test_api.py uses, so the
# pipeline runs its real end-to-end matching logic (no mocking) on every
# concurrent POST /reconcile.
VALID_RECORDS = [
    {
        "record_id": "conc_rec_001",
        "source": "LEDGER",
        "entity_type": "REFUND",
        "amount_paise": -1626135,
        "currency": "INR",
        "reference": "pay_lzwtpfigf6duyr",
        "counterparty": "Solstice Electronics LLP",
        "occurred_at": "2026-08-23",
        "raw_hash": "cdd0ed905b863c5a",
    },
    {
        "record_id": "conc_rec_002",
        "source": "GATEWAY",
        "entity_type": "REFUND",
        "amount_paise": -1626135,
        "currency": "INR",
        "reference": "pay_lzwtpfigf6duyr",
        "counterparty": "SOLSTICE ELECTRONICS LLP",
        "occurred_at": "2026-08-23",
        "raw_hash": "267ec6abf1d08b6c",
    },
    {
        "record_id": "conc_rec_003",
        "source": "BANK",
        "entity_type": "REFUND",
        "amount_paise": -1626135,
        "currency": "INR",
        "reference": "UTR20260824RF0032",
        "counterparty": "Razorpay Settlements Pvt Ltd",
        "occurred_at": "2026-08-24",
        "raw_hash": "86c2398a9380052d",
    },
]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _LiveServer:
    """Runs the real FastAPI app under a real uvicorn server, in a
    background thread of this same test process, bound to a real TCP
    port on localhost — genuine HTTP traffic, not TestClient's in-process
    ASGI shortcut."""

    def __init__(self, port: int) -> None:
        config = uvicorn.Config(app_module.app, host="127.0.0.1", port=port, log_level="warning")
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self) -> None:
        self.thread.start()
        deadline = time.time() + 10
        while not self.server.started and time.time() < deadline:
            time.sleep(0.05)
        if not self.server.started:
            raise RuntimeError("uvicorn server did not start in time")

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)


@pytest.fixture()
def live_server(tmp_path, monkeypatch):
    """A real, live uvicorn server wired to fresh temporary SQLite files
    — same isolation strategy test_api.py's ``client`` fixture uses, just
    pointed at a real socket instead of TestClient."""
    api_db = str(tmp_path / "api_live.db")
    governor_db = str(tmp_path / "governor_live.db")

    monkeypatch.setenv("API_DB_PATH", api_db)
    monkeypatch.setenv("GOVERNOR_DB_PATH", governor_db)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(config_module, "_settings", None)

    app_module.reset_store_for_tests(api_db)

    port = _free_port()
    server = _LiveServer(port)
    server.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.stop()


def _post_reconcile(base_url: str, timeout: float) -> httpx.Response:
    return httpx.post(
        f"{base_url}/reconcile",
        json={"records": VALID_RECORDS, "dataset": "concurrency_test"},
        timeout=timeout,
    )


def _get_health(base_url: str, timeout: float) -> httpx.Response:
    return httpx.get(f"{base_url}/health", timeout=timeout)


def test_concurrent_reconcile_and_health_bursts_never_crash_or_hang(live_server):
    """Fires several real concurrent HTTP requests at the live server —
    a mix of POST /reconcile and a burst of GET /health, matching the
    production traffic shape that caused the original crash — and
    repeats that mixed burst across multiple rounds (this was a race
    condition; one clean round doesn't prove the fix).

    A per-request timeout well under the original 30s ReadTimeout is
    used deliberately: if the shared-connection bug were still present,
    a colliding request would hang, and this test would fail with a
    ReadTimeout (or a 500) well before 30s — instead of, e.g., quietly
    passing because the process happened to be given enough wall-clock
    time to recover.
    """
    base_url = live_server
    per_request_timeout = 8.0
    rounds = 8
    health_bursts_per_round = 6

    for round_num in range(rounds):
        with ThreadPoolExecutor(max_workers=health_bursts_per_round + 1) as pool:
            futures = {
                pool.submit(_post_reconcile, base_url, per_request_timeout): "reconcile",
            }
            for _ in range(health_bursts_per_round):
                futures[pool.submit(_get_health, base_url, per_request_timeout)] = "health"

            results = {}
            errors = []
            for future in as_completed(futures, timeout=per_request_timeout + 5):
                kind = futures[future]
                try:
                    response = future.result()
                    results.setdefault(kind, []).append(response)
                except Exception as exc:  # noqa: BLE001 - we want to see ANY failure, including timeouts
                    errors.append((kind, round_num, repr(exc)))

        assert not errors, f"round {round_num}: request(s) failed or hung: {errors}"

        assert len(results.get("health", [])) == health_bursts_per_round
        for resp in results["health"]:
            assert resp.status_code == 200, f"round {round_num}: /health returned {resp.status_code}: {resp.text}"

        assert len(results.get("reconcile", [])) == 1
        reconcile_resp = results["reconcile"][0]
        assert reconcile_resp.status_code == 201, (
            f"round {round_num}: /reconcile returned {reconcile_resp.status_code}: {reconcile_resp.text}"
        )
        run_id = reconcile_resp.json()["run_id"]

        # Confirm the run this request just wrote is actually readable
        # back out — i.e. save_run()'s transaction genuinely committed
        # and wasn't silently corrupted/lost by a colliding request.
        audit_resp = httpx.get(f"{base_url}/audit/{run_id}", timeout=per_request_timeout)
        assert audit_resp.status_code == 200, f"round {round_num}: /audit/{run_id} returned {audit_resp.status_code}"


def test_concurrent_reconcile_requests_never_crash_or_hang(live_server):
    """A second, complementary shape: several POST /reconcile requests
    fired at once (each its own dataset/run_id), which is what actually
    drives concurrent *writers* through ``save_run()`` at the same
    moment — the most direct way to hit the shared-transaction race this
    bug was about. Repeated across multiple rounds for the same reason
    as above."""
    base_url = live_server
    per_request_timeout = 8.0
    rounds = 6
    concurrent_writers = 5

    for round_num in range(rounds):
        with ThreadPoolExecutor(max_workers=concurrent_writers) as pool:
            futures = [pool.submit(_post_reconcile, base_url, per_request_timeout) for _ in range(concurrent_writers)]
            responses = []
            errors = []
            for future in as_completed(futures, timeout=per_request_timeout + 5):
                try:
                    responses.append(future.result())
                except Exception as exc:  # noqa: BLE001
                    errors.append((round_num, repr(exc)))

        assert not errors, f"round {round_num}: request(s) failed or hung: {errors}"
        assert len(responses) == concurrent_writers
        for resp in responses:
            assert resp.status_code == 201, f"round {round_num}: /reconcile returned {resp.status_code}: {resp.text}"

        run_ids = [r.json()["run_id"] for r in responses]
        assert len(set(run_ids)) == concurrent_writers, "expected a distinct run_id per concurrent reconcile call"


# ---------------------------------------------------------------------------
# CallBudgetGovernor — direct concurrency stress test.
#
# The live-server tests above exercise RunStore under real concurrent
# traffic, but this test dataset's records are a clean, fully-matched
# triad that never reaches stage6_llm.py's residual-record path, so
# try_consume()/record_success()/record_failure() are never actually
# called by those HTTP requests. Test CallBudgetGovernor directly with
# many threads instead, to exercise the exact same
# shared-connection-across-threads risk this class had.
# ---------------------------------------------------------------------------


class _FakeSettings:
    groq_daily_call_ceiling = 100_000
    groq_circuit_breaker_max_consecutive_failures = 100_000


def test_governor_try_consume_under_real_concurrent_threads(tmp_path):
    """Hammers ``CallBudgetGovernor.try_consume`` from many real threads
    at once, repeated across multiple rounds. If the connection sharing
    bug were still present, this raises (e.g. sqlite3.OperationalError
    from two threads' BEGIN IMMEDIATE colliding on one connection
    object) or produces a corrupted/undercounted total. Instead, this
    asserts every call succeeds and the final count is exactly right —
    proving the per-call fresh connection serializes correctly via
    SQLite's own file-level locking rather than merely "not crashing by
    luck"."""
    db_path = str(tmp_path / "governor_concurrency_test.db")
    governor = CallBudgetGovernor(_FakeSettings(), db_path=db_path)  # type: ignore[arg-type]

    rounds = 5
    calls_per_round = 40

    for round_num in range(rounds):
        errors = []

        def _consume(i: int) -> None:
            try:
                result = governor.try_consume(f"run_{round_num}")
                assert result.allowed, f"round {round_num} call {i}: unexpectedly not allowed: {result}"
            except Exception as exc:  # noqa: BLE001
                errors.append((round_num, i, repr(exc)))

        with ThreadPoolExecutor(max_workers=calls_per_round) as pool:
            futures = [pool.submit(_consume, i) for i in range(calls_per_round)]
            for future in as_completed(futures, timeout=30):
                future.result()  # re-raise anything _consume didn't catch itself

        assert not errors, f"round {round_num}: concurrent try_consume failures: {errors}"

        status = governor.status(f"run_{round_num}")
        expected_total = calls_per_round * (round_num + 1)
        assert status.calls_used_today == expected_total, (
            f"round {round_num}: expected {expected_total} total calls used today, "
            f"got {status.calls_used_today} (a mismatch here means calls were lost "
            f"or double-counted under concurrency)"
        )


def test_governor_record_success_and_failure_under_real_concurrent_threads(tmp_path):
    """Same shape, for ``record_success``/``record_failure`` (the other
    two BEGIN IMMEDIATE-protected mutating paths in governor.py),
    hammered concurrently across several different run_ids at once."""
    db_path = str(tmp_path / "governor_concurrency_test_2.db")
    governor = CallBudgetGovernor(_FakeSettings(), db_path=db_path)  # type: ignore[arg-type]

    run_ids = [f"run_{i}" for i in range(6)]
    errors = []

    def _worker(run_id: str, idx: int) -> None:
        try:
            if idx % 2 == 0:
                governor.record_success(run_id)
            else:
                governor.record_failure(run_id, LLMTimeoutError("simulated timeout"))
        except Exception as exc:  # noqa: BLE001
            errors.append((run_id, idx, repr(exc)))

    with ThreadPoolExecutor(max_workers=len(run_ids) * 10) as pool:
        futures = [
            pool.submit(_worker, run_id, idx)
            for run_id in run_ids
            for idx in range(10)
        ]
        for future in as_completed(futures, timeout=30):
            future.result()

    assert not errors, f"concurrent record_success/record_failure failures: {errors}"
