"""End-to-end tests for the human review loop —
``PATCH /report/{run_id}/groups/{group_id}/review`` (api/app.py,
api/service.py's ``review_group``).

Scope, mirroring test_api.py's own scoping note: this file tests that
the review endpoint correctly closes (or escalates) a real
PENDING_REVIEW group and writes a correct audit trail — it does not
re-test Stage 1-6 matching or the Financial and Evidence Verifier
themselves (already covered by test_pipeline_verification.py,
test_stage1_exact.py, etc).

Per this stage's own requirement, the PENDING_REVIEW group(s) reviewed
below are produced by actually running the real pipeline via
``POST /reconcile`` (real records -> real Stage 1 matching -> real
verifier) — never a hand-built ``MatchGroup`` fixture with
``status=PENDING_REVIEW`` set directly. Three independent
LEDGER/GATEWAY pairs are used, one per review action, since APPROVED
and REJECTED both permanently leave PENDING_REVIEW (so each action
needs its own group; only ESCALATED could reuse one, but three
independent groups keeps every assertion below unambiguous about which
action produced it).

Each pair shares one reference (a real Stage 1 STAGE1_EXACT candidate:
same normalized reference, a LEDGER PAYMENT + a GATEWAY_SETTLEMENT) but
with the GATEWAY-side amount reduced ~7% below the LEDGER-side amount —
comfortably past config.py's calibrated
``stage1_residual_tolerance_fraction`` (3.25%), so the verifier's real,
independent residual re-check routes each one to
PENDING_REVIEW / RESIDUAL_EXCEEDS_TOLERANCE, exactly the kind of gap
this whole feature exists to let a human close.
"""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

from recon_agent import config as config_module

app_module = importlib.import_module("recon_agent.api.app")


def _pair(n: int, ledger_amount: int, gateway_amount: int, reference: str) -> list[dict]:
    return [
        {
            "record_id": f"review_ledger_{n}",
            "source": "LEDGER",
            "entity_type": "PAYMENT",
            "amount_paise": ledger_amount,
            "currency": "INR",
            "reference": reference,
            "counterparty": "Acme Retail Pvt Ltd",
            "occurred_at": "2026-01-01",
            "raw_hash": f"review_hash_ledger_{n}",
        },
        {
            "record_id": f"review_gateway_{n}",
            "source": "GATEWAY",
            "entity_type": "GATEWAY_SETTLEMENT",
            "amount_paise": gateway_amount,
            "currency": "INR",
            "reference": reference,
            "counterparty": "ACME RETAIL PVT LTD",
            "occurred_at": "2026-01-02",
            "raw_hash": f"review_hash_gateway_{n}",
        },
    ]


# Three independent pairs, each with a ~7% LEDGER-vs-GATEWAY residual —
# well past the 3.25% calibrated tolerance, so each one is a genuine,
# real RESIDUAL_EXCEEDS_TOLERANCE / PENDING_REVIEW outcome.
REVIEW_TEST_RECORDS = (
    _pair(1, 100_000, 93_000, "pay_review_apr001")
    + _pair(2, 200_000, 186_000, "pay_review_apr002")
    + _pair(3, 300_000, 279_000, "pay_review_apr003")
)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Same fresh-temporary-SQLite pattern as test_api.py's own
    ``client`` fixture — never shares state with a real DB or another
    test."""
    api_db = str(tmp_path / "api_review_test.db")
    governor_db = str(tmp_path / "governor_review_test.db")

    monkeypatch.setenv("API_DB_PATH", api_db)
    monkeypatch.setenv("GOVERNOR_DB_PATH", governor_db)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(config_module, "_settings", None)

    app_module.reset_store_for_tests(api_db)

    with TestClient(app_module.app) as c:
        yield c


@pytest.fixture()
def pending_groups(client: TestClient) -> tuple[str, list[str]]:
    """Run the real pipeline (POST /reconcile) once over
    REVIEW_TEST_RECORDS and return ``(run_id, [group_id, group_id,
    group_id])`` for the three real PENDING_REVIEW groups it produces —
    the "actually running the pipeline, not a hand-built fixture"
    starting point every test below reviews."""
    response = client.post("/reconcile", json={"records": REVIEW_TEST_RECORDS})
    assert response.status_code == 201, response.text
    run_id = response.json()["run_id"]

    audit = client.get(f"/audit/{run_id}").json()
    pending = [g for g in audit["match_groups"] if g["status"] == "PENDING_REVIEW"]
    assert len(pending) == 3, (
        f"expected 3 real PENDING_REVIEW groups from the pipeline run, got "
        f"{len(pending)}: {[(g['group_id'], g['status'], g['verification_result']) for g in audit['match_groups']]}"
    )
    for g in pending:
        assert g["verification_result"] == "RESIDUAL_EXCEEDS_TOLERANCE"
        assert g["proposed_by"] == "STAGE1_EXACT"
        assert g["reviewed_by"] is None
        assert g["reviewed_at"] is None
        assert g["review_action"] is None

    return run_id, [g["group_id"] for g in pending]


# ---------------------------------------------------------------------------
# APPROVED
# ---------------------------------------------------------------------------


def test_approve_closes_group_as_verified(client: TestClient, pending_groups) -> None:
    run_id, group_ids = pending_groups
    group_id = group_ids[0]

    response = client.patch(
        f"/report/{run_id}/groups/{group_id}/review",
        json={"reviewer": "alice", "review_action": "APPROVED", "note": "checked against bank statement"},
    )
    assert response.status_code == 200, response.text
    body = response.json()

    group = body["match_group"]
    assert group["group_id"] == group_id
    assert group["status"] == "VERIFIED"
    assert group["reviewed_by"] == "alice"
    assert group["reviewed_at"] is not None
    assert group["review_action"] == "APPROVED"
    assert group["verified_by"] == "HUMAN_REVIEWER"
    assert group["verification_result"] == "PASSED"

    event = body["decision_event"]
    assert event["group_id"] == group_id
    assert event["stage"] == "STAGE7_VERIFICATION"
    assert event["reason_code"] == "HUMAN_APPROVED"

    # The audit trail (GET /audit) shows this as a real DecisionEvent,
    # not just fields on the MatchGroup.
    audit = client.get(f"/audit/{run_id}").json()
    matching_events = [e for e in audit["decision_events"] if e["group_id"] == group_id and e["reason_code"] == "HUMAN_APPROVED"]
    assert len(matching_events) == 1
    persisted_group = next(g for g in audit["match_groups"] if g["group_id"] == group_id)
    assert persisted_group["status"] == "VERIFIED"
    assert persisted_group["reviewed_by"] == "alice"


# ---------------------------------------------------------------------------
# REJECTED
# ---------------------------------------------------------------------------


def test_reject_closes_group_as_rejected_without_releasing_records(client: TestClient, pending_groups) -> None:
    run_id, group_ids = pending_groups
    group_id = group_ids[1]

    response = client.patch(
        f"/report/{run_id}/groups/{group_id}/review",
        json={"reviewer": "bob", "review_action": "REJECTED", "note": "reference reused by accident"},
    )
    assert response.status_code == 200, response.text
    body = response.json()

    group = body["match_group"]
    assert group["status"] == "REJECTED"
    assert group["reviewed_by"] == "bob"
    assert group["reviewed_at"] is not None
    assert group["review_action"] == "REJECTED"
    # A human's explicit "no" is final — unlike an automatic system
    # rejection, verified_by/verification_result are left exactly as
    # the automated verifier set them (they still document *why* this
    # group needed human judgment), rather than being overwritten.
    assert group["verified_by"] == "FINANCIAL_AND_EVIDENCE_VERIFIER"
    assert group["verification_result"] == "RESIDUAL_EXCEEDS_TOLERANCE"

    event = body["decision_event"]
    assert event["stage"] == "STAGE7_VERIFICATION"
    assert event["reason_code"] == "HUMAN_REJECTED"

    audit = client.get(f"/audit/{run_id}").json()
    persisted_group = next(g for g in audit["match_groups"] if g["group_id"] == group_id)
    assert persisted_group["status"] == "REJECTED"
    matching_events = [e for e in audit["decision_events"] if e["group_id"] == group_id and e["reason_code"] == "HUMAN_REJECTED"]
    assert len(matching_events) == 1


# ---------------------------------------------------------------------------
# ESCALATED
# ---------------------------------------------------------------------------


def test_escalate_records_reviewer_but_leaves_status_pending(client: TestClient, pending_groups) -> None:
    run_id, group_ids = pending_groups
    group_id = group_ids[2]

    response = client.patch(
        f"/report/{run_id}/groups/{group_id}/review",
        json={"reviewer": "carol", "review_action": "ESCALATED", "note": "needs finance sign-off"},
    )
    assert response.status_code == 200, response.text
    body = response.json()

    group = body["match_group"]
    # "Flagged for further attention", not resolved.
    assert group["status"] == "PENDING_REVIEW"
    assert group["reviewed_by"] == "carol"
    assert group["reviewed_at"] is not None
    assert group["review_action"] == "ESCALATED"
    # Untouched — an escalation isn't a verification outcome.
    assert group["verified_by"] == "FINANCIAL_AND_EVIDENCE_VERIFIER"
    assert group["verification_result"] == "RESIDUAL_EXCEEDS_TOLERANCE"

    event = body["decision_event"]
    assert event["stage"] == "STAGE7_VERIFICATION"
    assert event["reason_code"] == "HUMAN_ESCALATED"
    assert event["candidate_scores"].get("note") == "needs finance sign-off"

    # Still PENDING_REVIEW, so it can still be approved/rejected later.
    follow_up = client.patch(
        f"/report/{run_id}/groups/{group_id}/review",
        json={"reviewer": "dave", "review_action": "APPROVED"},
    )
    assert follow_up.status_code == 200, follow_up.text
    assert follow_up.json()["match_group"]["status"] == "VERIFIED"
    assert follow_up.json()["match_group"]["reviewed_by"] == "dave"


# ---------------------------------------------------------------------------
# Refusing to re-review an already-closed group.
# ---------------------------------------------------------------------------


def test_reviewing_already_verified_group_is_cleanly_refused(client: TestClient, pending_groups) -> None:
    run_id, group_ids = pending_groups
    group_id = group_ids[0]

    first = client.patch(
        f"/report/{run_id}/groups/{group_id}/review",
        json={"reviewer": "alice", "review_action": "APPROVED"},
    )
    assert first.status_code == 200, first.text
    assert first.json()["match_group"]["status"] == "VERIFIED"

    second = client.patch(
        f"/report/{run_id}/groups/{group_id}/review",
        json={"reviewer": "mallory", "review_action": "REJECTED"},
    )
    assert second.status_code == 409, second.text
    assert "not PENDING_REVIEW" in second.json()["detail"]
    assert "VERIFIED" in second.json()["detail"]

    # Confirm the earlier APPROVED outcome was not disturbed by the
    # refused second attempt.
    audit = client.get(f"/audit/{run_id}").json()
    persisted_group = next(g for g in audit["match_groups"] if g["group_id"] == group_id)
    assert persisted_group["status"] == "VERIFIED"
    assert persisted_group["reviewed_by"] == "alice"


def test_reviewing_already_rejected_group_is_cleanly_refused(client: TestClient, pending_groups) -> None:
    run_id, group_ids = pending_groups
    group_id = group_ids[1]

    first = client.patch(
        f"/report/{run_id}/groups/{group_id}/review",
        json={"reviewer": "bob", "review_action": "REJECTED"},
    )
    assert first.status_code == 200, first.text

    second = client.patch(
        f"/report/{run_id}/groups/{group_id}/review",
        json={"reviewer": "mallory", "review_action": "APPROVED"},
    )
    assert second.status_code == 409, second.text
    assert "REJECTED" in second.json()["detail"]


# ---------------------------------------------------------------------------
# Not-found / malformed input.
# ---------------------------------------------------------------------------


def test_reviewing_unknown_group_id_returns_404(client: TestClient, pending_groups) -> None:
    run_id, _ = pending_groups
    response = client.patch(
        f"/report/{run_id}/groups/does-not-exist/review",
        json={"reviewer": "alice", "review_action": "APPROVED"},
    )
    assert response.status_code == 404
    assert "does-not-exist" in response.json()["detail"]


def test_reviewing_group_in_unknown_run_returns_404(client: TestClient, pending_groups) -> None:
    _, group_ids = pending_groups
    response = client.patch(
        f"/report/does-not-exist/groups/{group_ids[0]}/review",
        json={"reviewer": "alice", "review_action": "APPROVED"},
    )
    assert response.status_code == 404
    assert "does-not-exist" in response.json()["detail"]


def test_invalid_review_action_returns_422(client: TestClient, pending_groups) -> None:
    run_id, group_ids = pending_groups
    response = client.patch(
        f"/report/{run_id}/groups/{group_ids[0]}/review",
        json={"reviewer": "alice", "review_action": "NOT_A_REAL_ACTION"},
    )
    assert response.status_code == 422


def test_missing_reviewer_returns_422(client: TestClient, pending_groups) -> None:
    run_id, group_ids = pending_groups
    response = client.patch(
        f"/report/{run_id}/groups/{group_ids[0]}/review",
        json={"review_action": "APPROVED"},
    )
    assert response.status_code == 422


def test_review_endpoint_appears_in_openapi_schema(client: TestClient) -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "/report/{run_id}/groups/{group_id}/review" in response.json()["paths"]
