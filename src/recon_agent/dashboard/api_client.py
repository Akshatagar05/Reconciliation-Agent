"""HTTP client functions for the Stage 14 Streamlit dashboard.

Every function here does exactly one thing: call one of the existing
Stage 13 FastAPI endpoints (``api/app.py``) and return its parsed JSON
body, unchanged. Nothing here recomputes a metric, re-derives a status,
or reaches into ``matching``/``evaluation``/``llm`` internals directly —
the dashboard exercises the same HTTP surface a judge could hit
themselves with ``curl``.

Every function takes a ``client`` as its first argument rather than
hardcoding a transport, so the exact same functions run against two
different transports without any branching:

  - In production, ``app.py`` passes a ``RequestsClient`` (below),
    which sends real HTTP requests to a running ``uvicorn`` process.
  - In tests (``tests/test_dashboard_api_client.py``), a
    ``fastapi.testclient.TestClient`` is passed directly — the same
    pattern ``tests/test_api.py`` already uses for the API layer
    itself — so the data-fetching logic is verified end-to-end
    (real Pydantic validation, real pipeline run) without a live
    server or real sockets.

Both transports expose the same minimal interface used below:
``.get(path, params=...)`` and ``.post(path, json=...)``, each
returning an object with ``.status_code`` and ``.json()`` — this is
exactly ``requests``' and ``httpx``'s (TestClient's) shared response
shape, so no adapter layer is needed beyond ``RequestsClient`` turning
relative paths into absolute ones.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol


class SupportsGetPost(Protocol):
    """The minimal transport interface every function below relies on.
    Both ``requests.Session`` (via ``RequestsClient``) and
    ``fastapi.testclient.TestClient`` satisfy this already."""

    def get(self, url: str, **kwargs: Any) -> Any: ...

    def post(self, url: str, **kwargs: Any) -> Any: ...

    def patch(self, url: str, **kwargs: Any) -> Any: ...


class RequestsClient:
    """Thin wrapper around ``requests.Session`` that accepts the same
    relative paths (``"/report/abc123"``) that
    ``fastapi.testclient.TestClient`` does, by prepending ``base_url``.
    This is what lets every function in this module stay identical
    whether it's talking to a real ``uvicorn`` process or a TestClient
    in a test — only the transport passed in differs.
    """

    def __init__(self, base_url: str, timeout: float = 180.0) -> None:
        import requests  # local import: keeps `requests` optional for

        # anything (e.g. tests) that only ever uses TestClient.
        self._session = requests.Session()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def get(self, url: str, **kwargs: Any) -> Any:
        kwargs.setdefault("timeout", self.timeout)
        return self._session.get(f"{self.base_url}{url}", **kwargs)

    def post(self, url: str, **kwargs: Any) -> Any:
        kwargs.setdefault("timeout", self.timeout)
        return self._session.post(f"{self.base_url}{url}", **kwargs)

    def patch(self, url: str, **kwargs: Any) -> Any:
        kwargs.setdefault("timeout", self.timeout)
        return self._session.patch(f"{self.base_url}{url}", **kwargs)


class DashboardAPIError(RuntimeError):
    """Raised when the FastAPI layer returns a 4xx/5xx response. Carries
    the API's own reported status/detail so the Streamlit UI can show
    *what the API actually said* (e.g. a 404's "No run found with
    run_id=...") instead of a bare stack trace or a silently wrong
    empty result."""

    def __init__(self, message: str, status_code: Optional[int] = None, detail: Any = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


def _parse_or_raise(resp: Any) -> dict:
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail")
        except Exception:
            detail = getattr(resp, "text", None)
        raise DashboardAPIError(
            f"API returned {resp.status_code}: {detail}",
            status_code=resp.status_code,
            detail=detail,
        )
    return resp.json()


def check_health(client: SupportsGetPost) -> dict:
    """GET /health — liveness, used by the dashboard to confirm the API
    is reachable before anything else in the UI is enabled."""
    return _parse_or_raise(client.get("/health"))


def reconcile(
    client: SupportsGetPost,
    records: list[dict],
    dataset: Optional[str] = None,
    ground_truth: Optional[dict] = None,
) -> dict:
    """POST /reconcile — runs the existing pipeline over ``records`` and
    returns the ``ReconcileResponse`` body (run_id + summary) verbatim.
    ``records``/``ground_truth`` are plain dicts (already-loaded JSON),
    not model instances — the API layer's own Pydantic models validate
    them on arrival, exactly as they would for any other HTTP caller.
    """
    payload: dict[str, Any] = {"records": records}
    if dataset is not None:
        payload["dataset"] = dataset
    if ground_truth is not None:
        payload["ground_truth"] = ground_truth
    return _parse_or_raise(client.post("/reconcile", json=payload))


def get_report(client: SupportsGetPost, run_id: str) -> dict:
    """GET /report/{run_id} — the evaluation harness's metrics for this
    run, verbatim (full report if ground truth was supplied, otherwise
    the ground-truth-independent subset — see ``ReportResponse``)."""
    return _parse_or_raise(client.get(f"/report/{run_id}"))


def get_audit(client: SupportsGetPost, run_id: str) -> dict:
    """GET /audit/{run_id} — the full DecisionEvent trail plus the
    MatchGroup/MatchGroupMember/ReconciliationException rows those
    events refer to, verbatim."""
    return _parse_or_raise(client.get(f"/audit/{run_id}"))


def get_llm_budget(client: SupportsGetPost, run_id: str) -> dict:
    """GET /llm-budget/{run_id} — CallBudgetGovernor's own call-ledger
    snapshot for this run, verbatim."""
    return _parse_or_raise(client.get(f"/llm-budget/{run_id}"))


def review_group(
    client: SupportsGetPost,
    run_id: str,
    group_id: str,
    reviewer: str,
    review_action: str,
    note: Optional[str] = None,
) -> dict:
    """PATCH /report/{run_id}/groups/{group_id}/review — a human
    reviewer closing (APPROVED/REJECTED) or escalating (ESCALATED) a
    PENDING_REVIEW group. Returns the ``ReviewResponse`` body (updated
    match_group + the DecisionEvent this action emitted) verbatim.
    Raises ``DashboardAPIError`` (with the API's own 409 detail) if the
    group isn't currently PENDING_REVIEW — the dashboard surfaces that
    message directly rather than allowing a silent no-op."""
    payload: dict[str, Any] = {"reviewer": reviewer, "review_action": review_action}
    if note:
        payload["note"] = note
    return _parse_or_raise(
        client.patch(f"/report/{run_id}/groups/{group_id}/review", json=payload)
    )
