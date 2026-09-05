"""Tests for the Stage 14 Streamlit dashboard's data-fetching layer.

Streamlit apps themselves are awkward to unit test conventionally (per
this stage's own brief), so this file does NOT test ``dashboard/app.py``'s
UI rendering. Instead it proves the thing that actually matters for
correctness: every function in ``dashboard/api_client.py`` correctly
calls the real Stage 13 FastAPI endpoints and returns their JSON
bodies unchanged. This mirrors ``tests/test_api.py``'s own pattern
exactly — a ``fastapi.testclient.TestClient`` wired to fresh, temporary
SQLite files, no live server, no real sockets, no shared state between
tests.

Also covers ``dashboard/datasets.py``'s pure file-I/O helpers (bundled
dataset discovery + records/ground-truth loading), using ``tmp_path``
fixtures rather than depending on this repo's actual (not committed,
regenerated-from-seed) ``data/`` directory.
"""

from __future__ import annotations

import importlib
import json

import pytest
from fastapi.testclient import TestClient

from recon_agent import config as config_module
from recon_agent.dashboard import api_client, datasets

app_module = importlib.import_module("recon_agent.api.app")

# ---------------------------------------------------------------------------
# Fixture data — same real refund triad tests/test_api.py uses, so this
# suite exercises the real pipeline end-to-end (no mocking of matching
# logic) while staying fast and deterministic.
# ---------------------------------------------------------------------------

VALID_RECORDS = [
    {
        "record_id": "dash_rec_001",
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
        "record_id": "dash_rec_002",
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
        "record_id": "dash_rec_003",
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
    "dataset": "dashboard_test",
    "num_logical_events": 1,
    "num_physical_records": 3,
    "match_groups": [
        {
            "group_id": "dash_grp_001",
            "cardinality": "ONE_TO_ONE",
            "record_ids": ["dash_rec_001", "dash_rec_002", "dash_rec_003"],
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
    """A TestClient wired to fresh, temporary SQLite files — identical
    setup to tests/test_api.py's own fixture, reused here rather than
    redefined differently, since it must behave the same way."""
    api_db = str(tmp_path / "dash_api_test.db")
    governor_db = str(tmp_path / "dash_governor_test.db")

    monkeypatch.setenv("API_DB_PATH", api_db)
    monkeypatch.setenv("GOVERNOR_DB_PATH", governor_db)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(config_module, "_settings", None)

    app_module.reset_store_for_tests(api_db)

    with TestClient(app_module.app) as c:
        yield c


# ---------------------------------------------------------------------------
# api_client functions against a TestClient-backed API
# ---------------------------------------------------------------------------


def test_check_health(client: TestClient) -> None:
    result = api_client.check_health(client)
    assert result["status"] == "ok"


def test_reconcile_then_report_audit_budget_round_trip(client: TestClient) -> None:
    reconcile_result = api_client.reconcile(
        client,
        records=VALID_RECORDS,
        dataset="dashboard_test",
        ground_truth=VALID_GROUND_TRUTH,
    )
    run_id = reconcile_result["run_id"]
    assert run_id
    assert reconcile_result["total_records"] == 3
    assert reconcile_result["status"] in {"COMPLETE", "COMPLETED_WITH_EXCEPTIONS", "DEGRADED"}

    report = api_client.get_report(client, run_id)
    assert report["run_id"] == run_id
    assert report["ground_truth_available"] is True
    assert report["report"] is not None
    assert "auto_match_precision" in report["report"]
    assert "tier_contribution" in report["report"]

    audit = api_client.get_audit(client, run_id)
    assert audit["run_id"] == run_id
    assert isinstance(audit["match_groups"], list)
    assert isinstance(audit["match_group_members"], list)
    assert isinstance(audit["exceptions"], list)
    # The refund triad shares one reference, so at least one group should
    # exist with the provenance/verification fields the drill-down needs.
    assert len(audit["match_groups"]) >= 1
    group = audit["match_groups"][0]
    for field in ("proposed_by", "verified_by", "commit_policy", "verification_result"):
        assert field in group

    budget = api_client.get_llm_budget(client, run_id)
    assert budget["run_id"] == run_id
    assert budget["decision"] in {"ALLOW", "CEILING_EXCEEDED", "CIRCUIT_OPEN"}
    assert isinstance(budget["calls_used_today"], int)


def test_reconcile_without_ground_truth_gives_partial_report(client: TestClient) -> None:
    reconcile_result = api_client.reconcile(client, records=VALID_RECORDS)
    run_id = reconcile_result["run_id"]

    report = api_client.get_report(client, run_id)
    assert report["ground_truth_available"] is False
    assert report["report"] is None
    assert report["runtime_and_cost"] is not None


def test_unknown_run_id_raises_dashboard_api_error(client: TestClient) -> None:
    with pytest.raises(api_client.DashboardAPIError) as exc_info:
        api_client.get_report(client, "does-not-exist")
    assert exc_info.value.status_code == 404
    assert "does-not-exist" in str(exc_info.value.detail)


def test_review_group_round_trip(client: TestClient) -> None:
    """api_client.review_group correctly calls PATCH
    /report/{run_id}/groups/{group_id}/review against a real
    PENDING_REVIEW group produced by actually running the pipeline
    (same RESIDUAL_EXCEEDS_TOLERANCE construction as
    tests/test_human_review.py — a LEDGER/GATEWAY pair sharing a
    reference with a ~7% residual, past the calibrated 3.25%
    tolerance)."""
    records = [
        {
            "record_id": "dash_review_ledger_1",
            "source": "LEDGER",
            "entity_type": "PAYMENT",
            "amount_paise": 100_000,
            "currency": "INR",
            "reference": "pay_dash_review_001",
            "counterparty": "Acme Retail Pvt Ltd",
            "occurred_at": "2026-01-01",
            "raw_hash": "dash_review_hash_l1",
        },
        {
            "record_id": "dash_review_gateway_1",
            "source": "GATEWAY",
            "entity_type": "GATEWAY_SETTLEMENT",
            "amount_paise": 93_000,
            "currency": "INR",
            "reference": "pay_dash_review_001",
            "counterparty": "ACME RETAIL PVT LTD",
            "occurred_at": "2026-01-02",
            "raw_hash": "dash_review_hash_g1",
        },
    ]
    reconcile_result = api_client.reconcile(client, records=records)
    run_id = reconcile_result["run_id"]
    assert reconcile_result["pending_review_group_count"] == 1

    audit = api_client.get_audit(client, run_id)
    pending = [g for g in audit["match_groups"] if g["status"] == "PENDING_REVIEW"]
    assert len(pending) == 1
    group_id = pending[0]["group_id"]

    result = api_client.review_group(
        client, run_id, group_id, reviewer="dash_reviewer", review_action="APPROVED", note="looks right"
    )
    assert result["match_group"]["status"] == "VERIFIED"
    assert result["match_group"]["reviewed_by"] == "dash_reviewer"
    assert result["decision_event"]["reason_code"] == "HUMAN_APPROVED"

    # Re-reviewing an already-VERIFIED group surfaces the API's 409.
    with pytest.raises(api_client.DashboardAPIError) as exc_info:
        api_client.review_group(client, run_id, group_id, reviewer="mallory", review_action="REJECTED")
    assert exc_info.value.status_code == 409


def test_malformed_records_raises_dashboard_api_error(client: TestClient) -> None:
    bad_records = [
        {
            "record_id": "bad_rec_001",
            "entity_type": "NOT_A_REAL_ENTITY_TYPE",
            "amount_paise": "not-an-integer",
            "currency": "INR",
            "reference": "pay_bad",
            "counterparty": "Nobody",
            "occurred_at": "2026-08-23",
            "raw_hash": "deadbeef",
        }
    ]
    with pytest.raises(api_client.DashboardAPIError) as exc_info:
        api_client.reconcile(client, records=bad_records)
    assert exc_info.value.status_code == 422


# ---------------------------------------------------------------------------
# dashboard/datasets.py — pure file I/O, no HTTP involved.
# ---------------------------------------------------------------------------


def test_list_bundled_datasets_finds_records_and_ground_truth(tmp_path) -> None:
    (tmp_path / "data" / "calibration").mkdir(parents=True)
    (tmp_path / "data" / "evaluation").mkdir(parents=True)
    (tmp_path / "ground_truth" / "evaluation").mkdir(parents=True)

    (tmp_path / "data" / "calibration" / "records.json").write_text("[]")
    (tmp_path / "data" / "evaluation" / "records.json").write_text("[]")
    (tmp_path / "ground_truth" / "evaluation" / "ground_truth.json").write_text("{}")

    found = datasets.list_bundled_datasets(root=tmp_path)
    by_name = {d.name: d for d in found}

    assert set(by_name) == {"calibration", "evaluation"}
    assert by_name["calibration"].ground_truth_path is None
    assert by_name["evaluation"].ground_truth_path is not None


def test_list_bundled_datasets_empty_when_no_data_dir(tmp_path) -> None:
    assert datasets.list_bundled_datasets(root=tmp_path) == []


def test_load_records_round_trip(tmp_path) -> None:
    path = tmp_path / "records.json"
    path.write_text(json.dumps(VALID_RECORDS))
    loaded = datasets.load_records(path)
    assert loaded == VALID_RECORDS


def test_load_records_rejects_non_list(tmp_path) -> None:
    path = tmp_path / "records.json"
    path.write_text(json.dumps({"not": "a list"}))
    with pytest.raises(ValueError):
        datasets.load_records(path)


def test_load_records_from_bytes(tmp_path) -> None:
    raw = json.dumps(VALID_RECORDS).encode("utf-8")
    assert datasets.load_records_from_bytes(raw) == VALID_RECORDS


def test_load_ground_truth_round_trip(tmp_path) -> None:
    path = tmp_path / "ground_truth.json"
    path.write_text(json.dumps(VALID_GROUND_TRUTH))
    assert datasets.load_ground_truth(path) == VALID_GROUND_TRUTH
