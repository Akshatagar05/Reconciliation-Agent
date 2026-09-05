"""Stage 14 — Streamlit judge-facing dashboard (ARCHITECTURE.md §10, §15).

A **presentation layer only** over the Stage 13 FastAPI endpoints
(``api/app.py``). Every number and status shown here comes verbatim
from one of those endpoints' JSON responses — this file contains no
matching, verification, LLM, or evaluation logic of its own, and never
imports ``matching``/``evaluation``/``llm`` modules directly. See
``dashboard/api_client.py`` for the HTTP calls and
``dashboard/__init__.py`` for the package-level scope note.

--------------------------------------------------------------------
RUNNING THIS DASHBOARD — two processes, in this order:
--------------------------------------------------------------------

1. Start the FastAPI layer first (a separate terminal), from the repo
   root, with the package installed (``pip install -e .``):

       uvicorn recon_agent.api.app:app --reload

   This dashboard is an HTTP *client* of that process — it will not
   start it for you, and every action below will fail with a clear
   connection-error message until it's running.

2. Then start the dashboard itself (a second terminal):

       streamlit run src/recon_agent/dashboard/app.py

   By default it talks to ``http://127.0.0.1:8000`` — change this in
   the sidebar if your API is running elsewhere (a different port, a
   container, etc).
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd
import requests
import streamlit as st

from recon_agent.dashboard import api_client, datasets

st.set_page_config(page_title="Reconciliation Agent — Judge Dashboard", layout="wide")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, Any] = {
    "base_url": "http://127.0.0.1:8000",
    "run_id": None,
    "last_reconcile_response": None,
    "last_batch": None,  # {"records": [...], "dataset": ..., "ground_truth": ...}
    "degradation_runs": [],  # list of dicts, appended to by the demo tab
}
for key, value in _DEFAULTS.items():
    st.session_state.setdefault(key, value)


def get_client() -> api_client.RequestsClient:
    return api_client.RequestsClient(st.session_state["base_url"])


def _friendly_error(exc: Exception) -> str:
    if isinstance(exc, requests.exceptions.ConnectionError):
        return (
            f"Could not reach the API at {st.session_state['base_url']}. "
            "Is it running? Start it with:\n\n"
            "    uvicorn recon_agent.api.app:app --reload"
        )
    if isinstance(exc, api_client.DashboardAPIError):
        return str(exc)
    return f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Sidebar — API connection
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("API connection")
    st.session_state["base_url"] = st.text_input(
        "FastAPI base URL", value=st.session_state["base_url"]
    )
    if st.button("Check /health"):
        try:
            health = api_client.check_health(get_client())
            st.success(f"API reachable — status: {health['status']}")
            if health.get("note"):
                st.caption(health["note"])
        except Exception as exc:  # noqa: BLE001 — surfaced to the user, not swallowed
            st.error(_friendly_error(exc))

    st.divider()
    st.caption(
        "This dashboard only calls the FastAPI endpoints below over "
        "HTTP — the same surface a judge could hit with `curl`. No "
        "matching/verification/LLM/evaluation logic runs in this "
        "process."
    )
    st.code(
        "POST /reconcile\nGET /report/{run_id}\nGET /audit/{run_id}\n"
        "GET /llm-budget/{run_id}\nGET /health",
        language="text",
    )

st.title("Reconciliation Agent — Judge Dashboard")

tab_run, tab_results, tab_drilldown, tab_budget, tab_degrade = st.tabs(
    [
        "1. Run reconciliation",
        "2. Headline metrics",
        "3. Match / exception drill-down",
        "4. LLM budget",
        "5. Graceful degradation",
    ]
)

# ---------------------------------------------------------------------------
# Tab 1 — batch upload / dataset picker -> POST /reconcile
# ---------------------------------------------------------------------------

with tab_run:
    st.subheader("Choose a batch")
    source = st.radio(
        "Batch source",
        ["Bundled dataset", "Upload a records JSON file"],
        horizontal=True,
    )

    records: Optional[list[dict]] = None
    ground_truth: Optional[dict] = None
    dataset_label: Optional[str] = None

    if source == "Bundled dataset":
        bundled = datasets.list_bundled_datasets()
        if not bundled:
            st.warning(
                "No bundled datasets found under `data/`. Regenerate them "
                "with `python -m recon_agent.testdata.generator --seed both`, "
                "or switch to 'Upload a records JSON file'."
            )
        else:
            names = [d.name for d in bundled]
            chosen_name = st.selectbox("Dataset", names)
            chosen = next(d for d in bundled if d.name == chosen_name)
            records = datasets.load_records(chosen.records_path)
            dataset_label = chosen_name
            st.write(f"**{len(records)} records** loaded from `{chosen.records_path}`.")

            if chosen.ground_truth_path is not None:
                attach_gt = st.checkbox(
                    "Attach ground truth (enables the full evaluation report "
                    "in tab 2 — precision, coverage, false-match rate, "
                    "exception quality, tier contribution)",
                    value=True,
                )
                if attach_gt:
                    ground_truth = datasets.load_ground_truth(chosen.ground_truth_path)
            else:
                st.caption(
                    f"No `ground_truth/{chosen_name}/ground_truth.json` found — "
                    "only the ground-truth-independent metrics will be "
                    "available for this run."
                )
    else:
        uploaded_records = st.file_uploader(
            "Records JSON file (same shape as data/*/records.json)", type=["json"]
        )
        uploaded_gt = st.file_uploader(
            "Optional: matching ground truth JSON "
            "(same shape as ground_truth/*/ground_truth.json)",
            type=["json"],
        )
        dataset_label = st.text_input("Optional dataset label", value="")
        if uploaded_records is not None:
            try:
                records = datasets.load_records_from_bytes(uploaded_records.getvalue())
                st.write(f"**{len(records)} records** parsed from `{uploaded_records.name}`.")
            except ValueError as exc:
                st.error(str(exc))
        if uploaded_gt is not None:
            try:
                ground_truth = datasets.load_ground_truth_from_bytes(uploaded_gt.getvalue())
            except ValueError as exc:
                st.error(str(exc))

    st.divider()
    run_clicked = st.button(
        "Run reconciliation (POST /reconcile)",
        type="primary",
        disabled=records is None,
    )

    if run_clicked and records is not None:
        with st.spinner("Running the pipeline over this batch..."):
            try:
                response = api_client.reconcile(
                    get_client(),
                    records=records,
                    dataset=dataset_label or None,
                    ground_truth=ground_truth,
                )
                st.session_state["run_id"] = response["run_id"]
                st.session_state["last_reconcile_response"] = response
                st.session_state["last_batch"] = {
                    "records": records,
                    "dataset": dataset_label or None,
                    "ground_truth": ground_truth,
                }
                st.success(f"Run complete — run_id = `{response['run_id']}`")
            except Exception as exc:  # noqa: BLE001
                st.error(_friendly_error(exc))

    if st.session_state["last_reconcile_response"] is not None:
        st.subheader("Latest run summary")
        r = st.session_state["last_reconcile_response"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Status", r["status"])
        c2.metric("Total records", r["total_records"])
        c3.metric("Matched records", r["matched_record_count"])
        c4.metric("Elapsed (s)", f"{r['elapsed_seconds']:.2f}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Verified groups", r["verified_group_count"])
        c2.metric("Pending review", r["pending_review_group_count"])
        c3.metric("Rejected groups", r["rejected_group_count"])
        c4.metric("Exceptions", r["exception_count"])
        st.caption(
            f"run_id: `{r['run_id']}` · decision events: "
            f"{r['decision_event_count']} · Groq calls made: {r['groq_calls_made']}"
        )

# ---------------------------------------------------------------------------
# Tab 2 — headline metrics -> GET /report/{run_id}
# ---------------------------------------------------------------------------

with tab_results:
    run_id = st.session_state["run_id"]
    if run_id is None:
        st.info("Run a reconciliation in tab 1 first.")
    else:
        if st.button("Refresh report", key="refresh_report"):
            pass  # button press alone triggers a rerun/refetch below
        try:
            report_resp = api_client.get_report(get_client(), run_id)
        except Exception as exc:  # noqa: BLE001
            st.error(_friendly_error(exc))
            report_resp = None

        if report_resp is not None:
            if not report_resp["ground_truth_available"]:
                st.info(report_resp.get("note") or "Ground truth not available for this run.")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total records", report_resp.get("total_records"))
                c2.metric("Verified groups", report_resp.get("verified_group_count"))
                c3.metric("Pending review", report_resp.get("pending_review_group_count"))
                c4.metric("Rejected groups", report_resp.get("rejected_group_count"))
                rc = report_resp.get("runtime_and_cost") or {}
                if rc:
                    st.subheader("Runtime and cost")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Records/sec", f"{rc.get('records_per_second', 0):.2f}")
                    c2.metric("Groq calls made", rc.get("groq_calls_made"))
                    c3.metric("Est. cost (USD)", f"${rc.get('estimated_cost_usd', 0):.4f}")
            else:
                report = report_resp["report"]
                st.caption(
                    f"Dataset: **{report['dataset']}** · "
                    f"{report['total_records']} records"
                )

                p = report["auto_match_precision"]
                cov = report["record_coverage"]
                vcov = report["value_coverage"]
                fmr = report["false_match_rate"]
                eq = report["exception_quality"]

                st.subheader("Headline metrics (from the evaluation harness — nothing recomputed here)")
                c1, c2, c3 = st.columns(3)
                c1.metric(
                    "Auto-match precision",
                    f"{p['precision']:.1%}" if p["precision"] is not None else "n/a",
                    help=f"{p['correct']}/{p['total']} VERIFIED groups correct",
                )
                c2.metric(
                    "Record coverage (raw)",
                    f"{cov['raw_coverage']:.1%}",
                    help=f"{cov['matched_record_count']}/{cov['total_records']} records auto-resolved",
                )
                c3.metric(
                    "Value coverage",
                    f"{vcov['value_coverage']:.1%}",
                    help=f"{vcov['matched_value_paise']:,} / {vcov['eligible_value_paise']:,} paise",
                )
                c1, c2, c3 = st.columns(3)
                c1.metric(
                    "False-match rate",
                    f"{fmr['false_match_rate']:.1%}" if fmr["false_match_rate"] is not None else "n/a",
                    help=f"{fmr['incorrect']}/{fmr['total']} committed matches incorrect",
                )
                c2.metric(
                    "Exception quality",
                    f"{eq['exception_quality']:.1%}" if eq["exception_quality"] is not None else "n/a",
                    help=f"{eq['plausible']}/{eq['total']} unresolved cases plausibly categorized",
                )
                c3.metric(
                    "Coverage excl. abstention",
                    f"{cov['coverage_excluding_abstention']:.1%}"
                    if cov["coverage_excluding_abstention"] is not None
                    else "n/a",
                )

                st.subheader("Tier contribution — precision + coverage by proposing stage")
                tier_df = pd.DataFrame(report["tier_contribution"])
                st.dataframe(tier_df, use_container_width=True, hide_index=True)

                st.subheader("Runtime and cost")
                rc = report["runtime_and_cost"]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Wall clock (s)", f"{rc['wall_clock_seconds']:.2f}")
                c2.metric("Records/sec", f"{rc['records_per_second']:.2f}")
                c3.metric("Groq calls made", rc["groq_calls_made"])
                c4.metric("Est. cost (USD)", f"${rc['estimated_cost_usd']:.4f}")
                st.caption(rc["cost_assumptions"].get("note", ""))

                if eq["implausible_cases"]:
                    with st.expander(f"{len(eq['implausible_cases'])} implausibly-categorized unresolved case(s)"):
                        st.json(eq["implausible_cases"])

# ---------------------------------------------------------------------------
# Tab 3 — match / exception drill-down -> GET /audit/{run_id}
# ---------------------------------------------------------------------------

with tab_drilldown:
    run_id = st.session_state["run_id"]
    if run_id is None:
        st.info("Run a reconciliation in tab 1 first.")
    else:
        try:
            audit = api_client.get_audit(get_client(), run_id)
        except Exception as exc:  # noqa: BLE001
            st.error(_friendly_error(exc))
            audit = None

        if audit is not None:
            match_groups = audit["match_groups"]
            members = audit["match_group_members"]
            exceptions = audit["exceptions"]
            decision_events = audit["decision_events"]

            def _events_for(key: str) -> list[dict]:
                """Every DecisionEvent naming ``key`` (a group_id or a
                record_id) — matched the same way pipeline.py's own
                catch-all safety net matches them (a plain substring
                check against each event's group_id), so this covers a
                real group_id, a Stage 6 ``unmatched:{record_id}``
                synthetic id, and a Stage 3/4 ``rec:{record_id}``
                aggregation target id alike. Read-only rendering of
                DecisionEvents that already exist — no new computation."""
                return [e for e in decision_events if key in e["group_id"]]

            st.subheader("Match groups")
            statuses = ["VERIFIED", "PENDING_REVIEW", "REJECTED"]
            chosen_statuses = st.multiselect("Filter by status", statuses, default=statuses)
            filtered_groups = [g for g in match_groups if g["status"] in chosen_statuses]

            if not filtered_groups:
                st.write("No match groups for the selected status filter.")
            else:
                groups_df = pd.DataFrame(
                    [
                        {
                            "group_id": g["group_id"],
                            "status": g["status"],
                            "cardinality": g["cardinality"],
                            "expected_paise": g["expected_amount_paise"],
                            "matched_paise": g["matched_amount_paise"],
                            "residual_paise": g["residual_amount_paise"],
                            "proposed_by": g["proposed_by"],
                            "verified_by": g["verified_by"],
                            "commit_policy": g["commit_policy"],
                            "verification_result": g["verification_result"],
                        }
                        for g in filtered_groups
                    ]
                )
                st.dataframe(groups_df, use_container_width=True, hide_index=True)

                st.markdown("**Drill into one group — provenance vs. verification:**")
                group_ids = [g["group_id"] for g in filtered_groups]
                selected_group_id = st.selectbox("group_id", group_ids)
                selected_group = next(g for g in filtered_groups if g["group_id"] == selected_group_id)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("proposed_by", selected_group["proposed_by"])
                c2.metric("verified_by", selected_group["verified_by"])
                c3.metric("commit_policy", selected_group["commit_policy"])
                c4.metric("verification_result", selected_group["verification_result"])
                st.caption(
                    "These four fields are independent — this is the "
                    "'proposal provenance is separated from verification' "
                    "design (ARCHITECTURE.md §6): a STAGE6_LLM proposal, "
                    "for example, always carries commit_policy = "
                    "HUMAN_REVIEW_REQUIRED regardless of what verified_by "
                    "or verification_result end up saying."
                )

                c1, c2, c3, c4 = st.columns(4)
                c1.metric(
                    "Evidence score",
                    f"{selected_group['evidence_score']:.3f}" if selected_group["evidence_score"] is not None else "n/a",
                )
                c2.metric(
                    "Runner-up score",
                    f"{selected_group['runner_up_score']:.3f}" if selected_group["runner_up_score"] is not None else "n/a",
                )
                c3.metric(
                    "Score margin",
                    f"{selected_group['score_margin']:.3f}" if selected_group["score_margin"] is not None else "n/a",
                )
                c4.metric("Threshold applied", selected_group["threshold_applied"])

                c1, c2, c3 = st.columns(3)
                c1.metric("Expected (paise)", selected_group["expected_amount_paise"])
                c2.metric("Matched (paise)", selected_group["matched_amount_paise"])
                c3.metric("Residual (paise)", selected_group["residual_amount_paise"])

                with st.expander("Policy checks (itemized pass/fail, as proposed)"):
                    st.json(selected_group["policy_checks"])

                # -----------------------------------------------------
                # Human review — PATCH /report/{run_id}/groups/{group_id}/review
                # -----------------------------------------------------
                st.markdown("**Human review**")
                if selected_group["reviewed_by"] is not None:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("reviewed_by", selected_group["reviewed_by"])
                    c2.metric("reviewed_at", selected_group["reviewed_at"])
                    c3.metric("review_action", selected_group["review_action"])

                if selected_group["status"] != "PENDING_REVIEW":
                    if selected_group["reviewed_by"] is None:
                        st.caption(
                            f"This group is {selected_group['status']} via automated "
                            "verification — only a PENDING_REVIEW group can be reviewed."
                        )
                else:
                    st.caption(
                        "This group needs a human decision — the automated verifier "
                        "flagged it (see verification_result above) but couldn't "
                        "resolve it on its own. Approving closes it as VERIFIED, "
                        "rejecting closes it as REJECTED (final — its records are not "
                        "released back to the unmatched pool), escalating just "
                        "records who flagged it for further attention without "
                        "resolving it."
                    )
                    reviewer_name = st.text_input(
                        "Reviewer name", key=f"reviewer_name_{selected_group_id}"
                    )
                    review_note = st.text_input(
                        "Note (optional)", key=f"review_note_{selected_group_id}"
                    )
                    b1, b2, b3 = st.columns(3)
                    action_clicked = None
                    if b1.button("✅ Approve", key=f"approve_{selected_group_id}", use_container_width=True):
                        action_clicked = "APPROVED"
                    if b2.button("❌ Reject", key=f"reject_{selected_group_id}", use_container_width=True):
                        action_clicked = "REJECTED"
                    if b3.button("🚩 Escalate", key=f"escalate_{selected_group_id}", use_container_width=True):
                        action_clicked = "ESCALATED"

                    if action_clicked is not None:
                        if not reviewer_name.strip():
                            st.error("Enter a reviewer name before taking a review action.")
                        else:
                            try:
                                api_client.review_group(
                                    get_client(),
                                    run_id,
                                    selected_group_id,
                                    reviewer=reviewer_name.strip(),
                                    review_action=action_clicked,
                                    note=review_note.strip() or None,
                                )
                                st.success(f"{action_clicked.title()} recorded for {selected_group_id}.")
                                st.rerun()
                            except Exception as exc:  # noqa: BLE001
                                st.error(_friendly_error(exc))

                st.markdown(
                    "**Decision trail for this group** (every DecisionEvent "
                    "naming it — the proposing stage's own matching "
                    "reference/score breakdown, and the verifier's amount/"
                    "date/currency/residual re-checks and final reason "
                    "code):"
                )
                group_events = _events_for(selected_group_id)
                if not group_events:
                    st.write("No decision events found for this group.")
                else:
                    events_df = pd.DataFrame(
                        [
                            {
                                "stage": e["stage"],
                                "reason_code": e["reason_code"],
                                "explanation": e["explanation"],
                            }
                            for e in group_events
                        ]
                    )
                    st.dataframe(events_df, use_container_width=True, hide_index=True)
                    for e in group_events:
                        with st.expander(
                            f"{e['stage']} — {e['reason_code']} — full checks / matching "
                            "reference / candidate scores"
                        ):
                            st.json(e["candidate_scores"])

                member_rows = [m for m in members if m["group_id"] == selected_group_id]
                st.markdown("**Members of this group:**")
                st.dataframe(pd.DataFrame(member_rows), use_container_width=True, hide_index=True)

            st.divider()
            st.subheader("Exceptions")
            if not exceptions:
                st.write("No exceptions for this run.")
            else:
                severities = sorted({e["severity"] for e in exceptions})
                chosen_severities = st.multiselect(
                    "Filter by severity", severities, default=severities, key="exc_severity_filter"
                )
                filtered_exceptions = [e for e in exceptions if e["severity"] in chosen_severities]

                exc_df = pd.DataFrame(
                    [
                        {
                            "group_id_or_record_id": e["group_id_or_record_id"],
                            "category": e["category"],
                            "severity": e["severity"],
                            "review_status": e["review_status"],
                            "recommended_action": e["recommended_action"],
                        }
                        for e in filtered_exceptions
                    ]
                )
                st.dataframe(exc_df, use_container_width=True, hide_index=True)

                if filtered_exceptions:
                    st.markdown("**Drill into one exception:**")
                    exc_keys = [e["group_id_or_record_id"] for e in filtered_exceptions]
                    selected_exc_key = st.selectbox("group_id_or_record_id", exc_keys)
                    selected_exc = next(
                        e for e in filtered_exceptions if e["group_id_or_record_id"] == selected_exc_key
                    )
                    c1, c2 = st.columns(2)
                    c1.metric("Category", selected_exc["category"])
                    c2.metric("Severity", selected_exc["severity"])
                    st.write(f"**Recommended action:** {selected_exc['recommended_action']}")
                    with st.expander("Evidence"):
                        st.json(selected_exc["evidence"])

                    st.markdown(
                        "**Decision trail for this exception** (the "
                        "DecisionEvent(s) whose reason code and "
                        "explanation this exception was raised from):"
                    )
                    exc_events = _events_for(selected_exc_key)
                    if not exc_events:
                        st.write("No decision events found for this exception.")
                    else:
                        exc_events_df = pd.DataFrame(
                            [
                                {
                                    "stage": e["stage"],
                                    "reason_code": e["reason_code"],
                                    "explanation": e["explanation"],
                                }
                                for e in exc_events
                            ]
                        )
                        st.dataframe(exc_events_df, use_container_width=True, hide_index=True)
                        for e in exc_events:
                            with st.expander(
                                f"{e['stage']} — {e['reason_code']} — full checks / "
                                "candidate scores"
                            ):
                                st.json(e["candidate_scores"])

# ---------------------------------------------------------------------------
# Tab 4 — LLM budget -> GET /llm-budget/{run_id}
# ---------------------------------------------------------------------------

with tab_budget:
    run_id = st.session_state["run_id"]
    if run_id is None:
        st.info("Run a reconciliation in tab 1 first.")
    else:
        try:
            budget = api_client.get_llm_budget(get_client(), run_id)
        except Exception as exc:  # noqa: BLE001
            st.error(_friendly_error(exc))
            budget = None

        if budget is not None:
            c1, c2, c3 = st.columns(3)
            c1.metric("Decision", budget["decision"])
            c2.metric("Calls used today", f"{budget['calls_used_today']} / {budget['ceiling']}")
            c3.metric("Allowed", "yes" if budget["allowed"] else "no")
            c1, c2 = st.columns(2)
            c1.metric("Consecutive failures (this run)", budget["consecutive_failures"])
            c2.metric("Circuit breaker open", "OPEN" if budget["circuit_open"] else "closed")
            st.caption(
                "`calls_used_today` / `ceiling` are the shared *daily* "
                "counters across every concurrently-running run; "
                "`consecutive_failures` / `circuit_open` are scoped to "
                "this run_id's own circuit breaker (llm/governor.py)."
            )

# ---------------------------------------------------------------------------
# Tab 5 — graceful degradation, demo script §15 point 7
# ---------------------------------------------------------------------------

with tab_degrade:
    st.subheader("Prove graceful degradation live")
    st.markdown(
        "`GROQ_API_KEY` is read once, into a process-wide cached "
        "`Settings` object, when the FastAPI process starts "
        "(`config.py`'s `get_settings()`). That means **this dashboard "
        "cannot flip the key on or off over HTTP** — there is no "
        "endpoint for that, and there shouldn't be one invented here "
        "just to make a toggle switch work. The button below does not "
        "simulate anything: it only re-runs `POST /reconcile` against "
        "whatever environment the API process is *actually* running "
        "under right now. To see real degradation, restart the API "
        "process yourself between clicks, per the steps below."
    )

    st.markdown(
        "**1.** Stop the running `uvicorn` process (Ctrl+C in its terminal).\n\n"
        "**2.** Restart it *without* a Groq key:\n"
        "```\nunset GROQ_API_KEY\nuvicorn recon_agent.api.app:app --reload\n```\n"
        "**3.** Come back here and click **Re-run last batch now**.\n\n"
        "**4.** Restart it again *with* a real key "
        "(`export GROQ_API_KEY=...`) and click the button again to "
        "compare the two runs side by side below."
    )

    last_batch = st.session_state["last_batch"]
    rerun_disabled = last_batch is None
    if rerun_disabled:
        st.info("Run a reconciliation in tab 1 first, so there's a batch to re-run here.")

    if st.button("Re-run last batch now (POST /reconcile)", disabled=rerun_disabled):
        with st.spinner("Re-running the same batch..."):
            try:
                response = api_client.reconcile(
                    get_client(),
                    records=last_batch["records"],
                    dataset=last_batch["dataset"],
                    ground_truth=last_batch["ground_truth"],
                )
                budget_snapshot = api_client.get_llm_budget(get_client(), response["run_id"])
                audit_snapshot = api_client.get_audit(get_client(), response["run_id"])
                llm_unavailable_count = sum(
                    1
                    for e in audit_snapshot["exceptions"]
                    if e["category"] in ("LLM_UNAVAILABLE", "LLM_OUTPUT_REJECTED_BY_VERIFIER")
                )
                st.session_state["degradation_runs"].append(
                    {
                        "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                        "run_id": response["run_id"],
                        "status": response["status"],
                        "groq_calls_made": response["groq_calls_made"],
                        "exception_count": response["exception_count"],
                        "llm_related_exceptions": llm_unavailable_count,
                        "circuit_open": budget_snapshot["circuit_open"],
                        "llm_decision": budget_snapshot["decision"],
                    }
                )
                st.success(
                    f"Run complete (run_id = `{response['run_id']}`, "
                    f"status = {response['status']}) — the pipeline "
                    "completed regardless of whether Groq was available."
                )
            except Exception as exc:  # noqa: BLE001
                st.error(_friendly_error(exc))

    if st.session_state["degradation_runs"]:
        st.subheader("Runs so far (compare before/after restarting the API)")
        st.dataframe(
            pd.DataFrame(st.session_state["degradation_runs"]),
            use_container_width=True,
            hide_index=True,
        )
        if st.button("Clear comparison table"):
            st.session_state["degradation_runs"] = []
            st.rerun()
