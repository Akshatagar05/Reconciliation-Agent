"""Tests for the Stage 13 FastAPI layer (api/app.py).

Scope, per this stage's brief: this file tests that the API layer
correctly exposes the pipeline/harness/governor — it does NOT re-test
the underlying matching, verification, or evaluation logic itself
(that's already covered by tests/test_pipeline_verification.py,
tests/test_evaluation_harness.py, tests/test_governor.py, etc.).

Uses a real (small, hand-picked) three-record refund triad lifted
verbatim from data/evaluation/records.json + its corresponding
ground_truth/evaluation/ground_truth.json entry, so the round-trip test
exercises the real pipeline end-to-end (no mocking of matching logic)
while staying fast and deterministic. No GROQ_API_KEY is set anywhere
in this file, so Stage 6 degrades gracefully rather than making a real
network call, exactly as it's designed to (§14).
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

import importlib

from recon_agent import config as config_module

# NOTE: recon_agent.api.__init__ does `from recon_agent.api.app import app`,
# which rebinds the *attribute* `recon_agent.api.app` to the FastAPI
# instance itself (package __init__ assignments become package
# attributes) — so `import recon_agent.api.app as x` would silently
# bind `x` to the FastAPI app, not the submodule. Fetch the actual
# submodule (which has `app` and `reset_store_for_tests` as attributes)
# via sys.modules/importlib instead.
app_module = importlib.import_module("recon_agent.api.app")

# ---------------------------------------------------------------------------
# Fixture data — a real, isolated refund triad (LEDGER + GATEWAY + BANK)
# taken from data/evaluation/records.json / ground_truth/evaluation/ground_truth.json
# (group evl_grp_00005), renamed to standalone ids so this test never
# depends on the rest of that dataset being present.
# ---------------------------------------------------------------------------

VALID_RECORDS = [
    {
        "record_id": "test_rec_001",
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
        "record_id": "test_rec_002",
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
        "record_id": "test_rec_003",
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

VALID_GROUND_TRUTH = {
    "dataset": "api_test",
    "num_logical_events": 1,
    "num_physical_records": 3,
    "match_groups": [
        {
            "group_id": "test_grp_001",
            "cardinality": "ONE_TO_ONE",
            "record_ids": ["test_rec_001", "test_rec_002", "test_rec_003"],
            "categories": ["lifecycle_refund"],
        }
    ],
    "duplicates": [],
    "unresolved": [],
    "honest_abstention": [],
    "category_counts": {"lifecycle_refund": 1},
}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """A TestClient wired to fresh, temporary SQLite files for both the
    API's own RunStore and the governor's db — never sharing state with
    a real API_DB_PATH/GOVERNOR_DB_PATH or with another test."""
    api_db = str(tmp_path / "api_test.db")
    governor_db = str(tmp_path / "governor_test.db")

    monkeypatch.setenv("API_DB_PATH", api_db)
    monkeypatch.setenv("GOVERNOR_DB_PATH", governor_db)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    # Force get_settings() to reload from the patched environment rather
    # than reuse whatever an earlier test/module import already cached.
    monkeypatch.setattr(config_module, "_settings", None)

    app_module.reset_store_for_tests(api_db)

    with TestClient(app_module.app) as c:
        yield c


# ---------------------------------------------------------------------------
# Full round trip: POST /reconcile, then GET each of the other endpoints.
# ---------------------------------------------------------------------------


def test_full_round_trip(client: TestClient) -> None:
    response = client.post(
        "/reconcile",
        json={"records": VALID_RECORDS, "dataset": "api_test", "ground_truth": VALID_GROUND_TRUTH},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    run_id = body["run_id"]
    assert run_id
    assert body["total_records"] == 3
    assert body["status"] in {"COMPLETE", "COMPLETED_WITH_EXCEPTIONS", "DEGRADED"}
    assert body["decision_event_count"] >= 1
    # The refund triad shares one reference across LEDGER/GATEWAY, and a
    # constrained/exact match should account for all three records —
    # this is the one place this file leans on real matching-pipeline
    # behavior, so assert loosely (>= 1 verified group) rather than
    # asserting exact internals the pipeline itself already covers.
    assert body["verified_group_count"] + body["pending_review_group_count"] >= 1

    # GET /report/{run_id} — ground truth was supplied, so the full
    # harness-computed report should come back.
    report_resp = client.get(f"/report/{run_id}")
    assert report_resp.status_code == 200, report_resp.text
    report_body = report_resp.json()
    assert report_body["run_id"] == run_id
    assert report_body["ground_truth_available"] is True
    assert report_body["report"] is not None
    assert report_body["report"]["total_records"] == 3
    assert "auto_match_precision" in report_body["report"]
    assert "runtime_and_cost" in report_body["report"]

    # GET /audit/{run_id} — the full DecisionEvent trail.
    audit_resp = client.get(f"/audit/{run_id}")
    assert audit_resp.status_code == 200, audit_resp.text
    audit_body = audit_resp.json()
    assert audit_body["run_id"] == run_id
    assert len(audit_body["decision_events"]) >= 1
    assert all(de["stage"] for de in audit_body["decision_events"])

    # GET /llm-budget/{run_id} — the governor's ledger for this run.
    budget_resp = client.get(f"/llm-budget/{run_id}")
    assert budget_resp.status_code == 200, budget_resp.text
    budget_body = budget_resp.json()
    assert budget_body["run_id"] == run_id
    assert budget_body["decision"] in {"ALLOW", "CEILING_EXCEEDED", "CIRCUIT_OPEN"}
    assert isinstance(budget_body["calls_used_today"], int)
    assert isinstance(budget_body["ceiling"], int)


def test_reconcile_without_ground_truth_gives_partial_report(client: TestClient) -> None:
    response = client.post("/reconcile", json={"records": VALID_RECORDS})
    assert response.status_code == 201, response.text
    run_id = response.json()["run_id"]

    report_resp = client.get(f"/report/{run_id}")
    assert report_resp.status_code == 200
    report_body = report_resp.json()
    assert report_body["ground_truth_available"] is False
    assert report_body["report"] is None
    assert report_body["note"] is not None
    assert report_body["runtime_and_cost"] is not None
    assert report_body["total_records"] == 3


# ---------------------------------------------------------------------------
# 404 for an unknown run_id.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("endpoint", ["/report", "/audit", "/llm-budget"])
def test_unknown_run_id_returns_404(client: TestClient, endpoint: str) -> None:
    response = client.get(f"{endpoint}/does-not-exist")
    assert response.status_code == 404
    assert "does-not-exist" in response.json()["detail"]


# ---------------------------------------------------------------------------
# 422 for malformed input.
# ---------------------------------------------------------------------------


def test_malformed_record_returns_422(client: TestClient) -> None:
    malformed_records = [
        {
            "record_id": "bad_rec_001",
            # "source" is missing entirely, and entity_type is not a
            # valid EntityType value — both should fail Pydantic
            # validation before the pipeline ever runs.
            "entity_type": "NOT_A_REAL_ENTITY_TYPE",
            "amount_paise": "not-an-integer",
            "currency": "INR",
            "reference": "pay_bad",
            "counterparty": "Nobody",
            "occurred_at": "2026-08-23",
            "raw_hash": "deadbeef",
        }
    ]
    response = client.post("/reconcile", json={"records": malformed_records})
    assert response.status_code == 422
    assert "detail" in response.json()


def test_empty_records_list_returns_422(client: TestClient) -> None:
    response = client.post("/reconcile", json={"records": []})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert isinstance(body["selfcheck_available"], bool)


# ---------------------------------------------------------------------------
# OpenAPI docs render (typed endpoint signatures produce a usable schema).
# ---------------------------------------------------------------------------


def test_openapi_schema_includes_all_endpoints(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    paths = schema["paths"]
    assert "/reconcile" in paths
    assert "/report/{run_id}" in paths
    assert "/audit/{run_id}" in paths
    assert "/llm-budget/{run_id}" in paths
    assert "/health" in paths

    docs_response = client.get("/docs")
    assert docs_response.status_code == 200


# ---------------------------------------------------------------------------
# Server-survival regression test.
#
# Written after a reviewer, testing a live `uvicorn` process in a sandbox
# that had already run 15+ rounds of pip installs/pytest runs/server
# launches in one very long session, saw the server die silently (no
# traceback, no log line, the process just vanished) shortly after
# startup around POST /reconcile calls, twice, on two different ports.
#
# A fresh-environment investigation (many sequential live POST /reconcile
# calls with real data, plus a battery of malformed bodies — a bare array,
# missing required fields, wrong types, invalid JSON, a `null` body, an
# empty `records` list — all interleaved with GET /health checks) could
# NOT reproduce any crash: every malformed body cleanly produced a 422 and
# every /health check in between succeeded, including after 10 total
# successful /reconcile calls. That live investigation isn't repeated
# here since a live uvicorn process is inherently less reliable to
# automate/reproduce in CI than FastAPI's TestClient (no real socket, no
# real process, no port reuse across runs) — see ARCHITECTURE.md/this
# stage's own scope note on preferring TestClient for exactly that
# reason. The conclusion this test exists to guard is therefore: the API
# layer has no code path that can bring the whole process down, which is
# equally checkable, deterministically, in-process.
#
# This test is regression coverage for that conclusion, not a
# reproduction of a confirmed bug: it interleaves real POST /reconcile
# calls with the same categories of malformed bodies used in the live
# investigation, asserting a /health check succeeds after every single
# one — i.e. the TestClient's app instance (and therefore the process it
# lives in) never becomes unresponsive, regardless of which request came
# before it.
# ---------------------------------------------------------------------------


MALFORMED_BODIES = [
    # A bare array instead of the {"records": [...]} envelope.
    [{"record_id": "x"}],
    # Missing required fields (source, entity_type, amount_paise, ...).
    {"records": [{"record_id": "x"}]},
    # Wrong types throughout (amount_paise as str, occurred_at not a date).
    {
        "records": [
            {
                "record_id": 123,
                "source": "BANK",
                "entity_type": "BANK_CREDIT",
                "amount_paise": "not-a-number",
                "currency": "INR",
                "reference": "X",
                "counterparty": "Y",
                "occurred_at": "not-a-date",
                "raw_hash": "z",
            }
        ]
    },
    # Empty records list (violates min_length=1).
    {"records": []},
    # A JSON `null` body.
    None,
    # Records key present but not a list at all.
    {"records": "not-a-list"},
]


def test_server_survives_interleaved_valid_and_malformed_reconcile_calls(
    client: TestClient,
) -> None:
    """Regression test for the silently-dying-server report described
    above. Sends several real POST /reconcile calls interleaved with
    every malformed-body case above, asserting after *each* one — valid
    or malformed — that the app is still responsive (GET /health
    succeeds). If any request ever crashed the app/process instead of
    returning an HTTP error, the very next /health call in this loop
    would fail or the TestClient would raise, and this test would catch
    it immediately rather than requiring a live, long-running server to
    notice.
    """
    assert client.get("/health").status_code == 200

    request_count = 0
    for round_num in range(3):
        # A real, valid POST /reconcile.
        response = client.post("/reconcile", json={"records": VALID_RECORDS})
        assert response.status_code == 201, response.text
        request_count += 1
        health = client.get("/health")
        assert health.status_code == 200, (
            f"server unresponsive after valid reconcile #{request_count} "
            f"(round {round_num})"
        )

        # Every malformed-body case, each followed by its own health check.
        for case_index, bad_body in enumerate(MALFORMED_BODIES):
            response = client.post("/reconcile", json=bad_body)
            assert response.status_code in (400, 422), (
                f"expected a clean 4xx for malformed body #{case_index} "
                f"(round {round_num}), got {response.status_code}: "
                f"{response.text}"
            )
            health = client.get("/health")
            assert health.status_code == 200, (
                f"server unresponsive after malformed body #{case_index} "
                f"(round {round_num}): {bad_body!r}"
            )

    # One final valid call at the very end confirms the app is still
    # fully functional, not merely "still answering /health".
    final_response = client.post("/reconcile", json={"records": VALID_RECORDS})
    assert final_response.status_code == 201, final_response.text
    assert client.get("/health").status_code == 200
