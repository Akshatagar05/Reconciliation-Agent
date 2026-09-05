#!/usr/bin/env python3
"""Stage 16 failure-injection demonstration — ARCHITECTURE.md §15, points 5-7.

    5. Open a genuinely unresolved exception — explain why the system
       refused to guess.
    6. Show a STAGE6_LLM proposal sitting in PENDING_REVIEW — explain why
       the LLM doesn't get to decide alone. (Not scripted here: §2/§14
       already guarantee no STAGE6_LLM group is ever auto-committed, and
       this build's real evaluation run makes zero STAGE6_LLM proposals
       PASS to VERIFIED by construction — see tests/test_stage6_*.py.)
    7. Disable the Groq key, rerun — prove graceful degradation live.

This script does not build any new degradation, exception, or matching
logic. Every behavior it demonstrates already exists and is already
covered by the test suite (test_pipeline_final_catchall.py,
test_stage6_pipeline_integration.py, test_api.py). Its only job is to
run that existing behavior against real data, in one place, in a form
a person can point a judge at and read top-to-bottom in under a minute
— i.e. making it *demonstrable*, not re-proving it's correct.

Usage:
    python scripts/failure_injection_demo.py

Exit code 0 if all three scenarios behave as documented; non-zero and a
clear message identifying which scenario failed otherwise.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _fail(scenario: str, detail: str) -> None:
    print(f"\nFAIL — {scenario}: {detail}", file=sys.stderr)
    sys.exit(1)


def _header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def scenario_a_honest_exception() -> None:
    """A genuinely unresolved record becomes an honest exception, not a
    forced guess.

    Uses ``ground_truth/evaluation/ground_truth.json``'s own
    ``honest_abstention`` entries — records the synthetic-data generator
    deliberately created with NO real counterpart anywhere in the batch
    (e.g. a manual ledger correction with nothing on the bank/gateway
    side to match against). Runs the real pipeline over the real
    evaluation dataset and confirms: (1) the record produces a real
    ``ReconciliationException``, not a fabricated match; (2) it is never
    a member of any ``VERIFIED`` group; (3) the exception's own evidence
    says why, in a form fit to show a judge directly.
    """
    _header("Scenario (a): genuinely unresolved record -> honest exception")

    from recon_agent.matching.pipeline import run_pipeline
    from recon_agent.models.normalized_record import NormalizedRecord
    from recon_agent.models.enums import MatchGroupStatus

    records_path = REPO_ROOT / "data" / "evaluation" / "records.json"
    gt_path = REPO_ROOT / "ground_truth" / "evaluation" / "ground_truth.json"
    if not records_path.exists() or not gt_path.exists():
        _fail(
            "(a)",
            f"{records_path} / {gt_path} not found — run "
            "`python -m recon_agent.testdata.generator --seed both` first.",
        )

    records = [NormalizedRecord(**r) for r in json.loads(records_path.read_text())]
    ground_truth = json.loads(gt_path.read_text())
    abstentions = ground_truth.get("honest_abstention", [])
    if not abstentions:
        _fail("(a)", "no honest_abstention entries in ground truth to demonstrate against.")

    target_id = abstentions[0]["record_id"]
    target_reason = abstentions[0]["reason"]
    print(
        f"Ground truth says {target_id!r} has NO real counterpart anywhere "
        f"in this batch ({target_reason}) — a correct system must not "
        "invent a match for it."
    )

    result = run_pipeline(records, run_id="failure-injection-demo-a")

    verified_group_ids = {
        g.group_id for g in result.match_groups if g.status == MatchGroupStatus.VERIFIED
    }
    verified_member_ids = {
        m.record_id for m in result.match_group_members if m.group_id in verified_group_ids
    }

    if target_id in verified_member_ids:
        _fail(
            "(a)",
            f"{target_id} was force-matched into a VERIFIED group — this "
            "is exactly the forced-guess failure mode this scenario exists "
            "to catch.",
        )

    matching_exceptions = [
        e for e in result.exceptions if e.group_id_or_record_id == target_id
    ]
    if not matching_exceptions:
        _fail(
            "(a)",
            f"{target_id} is not VERIFIED, but also produced no "
            "ReconciliationException — it would be silently invisible "
            "instead of an honest, reviewable exception.",
        )

    exc = matching_exceptions[0]
    print(f"\n{target_id} correctly became an honest exception instead:")
    print(f"  category:           {exc.category.value}")
    print(f"  severity:           {exc.severity.value}")
    print(f"  evidence:           {json.dumps(exc.evidence)}")
    print(f"  recommended_action: {exc.recommended_action}")
    print(f"  review_status:      {exc.review_status.value}")
    print(
        "\nPASS — the system declined to guess and routed this record to "
        "an auditable exception with a reason code and a recommended "
        "action, exactly as §14 requires (\"every unresolved record has a "
        "reason code, evidence, and recommended action\")."
    )


def scenario_b_groq_degradation() -> None:
    """The Groq key is unset and the pipeline still completes.

    No mocking, no monkeypatching of the degradation logic itself —
    this runs the real ``run_pipeline`` with a real (empty) environment,
    exactly as a judge disabling the key live would. The degradation
    logic itself is Stage 11's (``llm/governor.py``,
    ``matching/stage6_llm.py``); this scenario only proves it is
    reachable and actually fires against real, non-trivial data.
    """
    _header("Scenario (b): Groq key unset -> graceful degradation")

    saved_key = os.environ.pop("GROQ_API_KEY", None)
    try:
        from recon_agent import config as config_module

        # Force get_settings() to reload without a cached key from an
        # earlier import/process, mirroring tests/test_api.py's own
        # fixture pattern for the same reason.
        config_module._settings = None

        from recon_agent.matching.pipeline import run_pipeline
        from recon_agent.models.normalized_record import NormalizedRecord
        from recon_agent.models.enums import ExceptionCategory

        records_path = REPO_ROOT / "data" / "evaluation" / "records.json"
        if not records_path.exists():
            _fail(
                "(b)",
                f"{records_path} not found — run "
                "`python -m recon_agent.testdata.generator --seed both` first.",
            )
        records = [NormalizedRecord(**r) for r in json.loads(records_path.read_text())]

        print("GROQ_API_KEY: unset for this run (popped from the environment).")
        result = run_pipeline(records, run_id="failure-injection-demo-b")

        degraded = [
            e for e in result.exceptions if e.category == ExceptionCategory.LLM_UNAVAILABLE
        ]
        print(
            f"\nrun_pipeline completed normally — no exception propagated, "
            f"no hang, no crash — with {len(result.match_groups)} groups, "
            f"{len(result.exceptions)} total exceptions "
            f"({len(degraded)} of them LLM_UNAVAILABLE)."
        )
        if not degraded:
            _fail(
                "(b)",
                "pipeline completed, but produced zero LLM_UNAVAILABLE "
                "exceptions — either nothing reached Stage 6 on this "
                "dataset, or degradation didn't fire as expected.",
            )

        sample = degraded[0]
        print("\nSample degraded record:")
        print(f"  record:             {sample.group_id_or_record_id}")
        print(f"  category:           {sample.category.value}")
        print(f"  evidence:           {json.dumps(sample.evidence)}")
        print(f"  recommended_action: {sample.recommended_action}")
        print(
            "\nPASS — Stage 6 residue degrades to an honest LLM_UNAVAILABLE "
            "exception per record, and the run as a whole still completes "
            "(§14: \"the system completes when the Groq key is missing\")."
        )
    finally:
        if saved_key is not None:
            os.environ["GROQ_API_KEY"] = saved_key


def scenario_c_malformed_input() -> None:
    """A malformed/adversarial input is handled without crashing.

    Builds directly on Stage 13's fresh-environment investigation (see
    README.md's "Investigated: reported silent server death" section):
    the same categories of malformed request bodies are replayed here
    against an in-process TestClient (chosen for the same reliability
    reason Stage 13 gave — no real socket/process/port to flake), each
    one interleaved with a GET /health check, to demonstrate live rather
    than just cite the existing regression test.
    """
    _header("Scenario (c): malformed/adversarial input -> no crash")

    from fastapi.testclient import TestClient
    from recon_agent import config as config_module
    import importlib

    app_module = importlib.import_module("recon_agent.api.app")

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        os.environ["API_DB_PATH"] = str(Path(tmp) / "api_demo.db")
        os.environ["GOVERNOR_DB_PATH"] = str(Path(tmp) / "governor_demo.db")
        os.environ.pop("GROQ_API_KEY", None)
        config_module._settings = None
        app_module.reset_store_for_tests(os.environ["API_DB_PATH"])

        adversarial_bodies = [
            ("bare array instead of {records: [...]} envelope", [{"record_id": "x"}]),
            ("missing required fields", {"records": [{"record_id": "x"}]}),
            (
                "wrong field types (amount_paise as str, bad date)",
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
            ),
            ("empty records list", {"records": []}),
            ("JSON null body", None),
            ("records key present but not a list", {"records": "not-a-list"}),
        ]

        # A separate, non-4xx-required case: a schema-VALID but adversarially
        # oversized field. This is deliberately not asserted as a 4xx —
        # a 50,000-character reference string doesn't violate any declared
        # schema constraint, so accepting and processing it (200/201) is the
        # *correct* behavior, not a bug. What this demonstrates is narrower
        # but still real: the API doesn't choke, hang, or 500 on an
        # adversarially-sized-but-valid payload.
        oversized_body = {
            "records": [
                {
                    "record_id": "adv_001",
                    "source": "BANK",
                    "entity_type": "BANK_CREDIT",
                    "amount_paise": 100,
                    "currency": "INR",
                    "reference": "R" * 50_000,
                    "counterparty": "Y",
                    "occurred_at": "2026-01-01",
                    "raw_hash": "z",
                }
            ]
        }

        with TestClient(app_module.app) as client:
            health = client.get("/health")
            if health.status_code != 200:
                _fail("(c)", f"/health failed before any adversarial input: {health.status_code}")

            for label, body in adversarial_bodies:
                response = client.post("/reconcile", json=body)
                print(f"  {label:55s} -> HTTP {response.status_code}")
                if response.status_code not in (400, 422):
                    _fail(
                        "(c)",
                        f"{label!r} produced HTTP {response.status_code} "
                        f"instead of a clean 4xx: {response.text[:300]}",
                    )
                health = client.get("/health")
                if health.status_code != 200:
                    _fail("(c)", f"server unresponsive after {label!r}")

            response = client.post("/reconcile", json=oversized_body)
            print(
                f"  {'schema-valid oversized field (50k chars)':55s} -> "
                f"HTTP {response.status_code}"
            )
            if response.status_code >= 500:
                _fail(
                    "(c)",
                    f"an oversized-but-valid field caused a server error: "
                    f"{response.status_code} {response.text[:300]}",
                )
            health = client.get("/health")
            if health.status_code != 200:
                _fail("(c)", "server unresponsive after the oversized-field call")

            # One valid call afterward, to prove the app is still fully
            # functional, not merely still answering /health.
            valid_body = {
                "records": [
                    {
                        "record_id": "demo_rec_001",
                        "source": "LEDGER",
                        "entity_type": "REFUND",
                        "amount_paise": -1000,
                        "currency": "INR",
                        "reference": "pay_demo001",
                        "counterparty": "Demo Corp",
                        "occurred_at": "2026-08-23",
                        "raw_hash": "deadbeef01",
                    },
                    {
                        "record_id": "demo_rec_002",
                        "source": "GATEWAY",
                        "entity_type": "REFUND",
                        "amount_paise": -1000,
                        "currency": "INR",
                        "reference": "pay_demo001",
                        "counterparty": "Demo Corp",
                        "occurred_at": "2026-08-23",
                        "raw_hash": "deadbeef02",
                    },
                ]
            }
            final = client.post("/reconcile", json=valid_body)
            if final.status_code != 201:
                _fail(
                    "(c)",
                    "a valid /reconcile call after the adversarial battery "
                    f"failed unexpectedly: {final.status_code} {final.text[:300]}",
                )
            health = client.get("/health")
            if health.status_code != 200:
                _fail("(c)", "server unresponsive after the final valid call")

    print(
        "\nPASS — every genuinely malformed body above produced a clean "
        "4xx, the schema-valid-but-oversized body was accepted and "
        "processed without a server error, the server stayed responsive "
        "after each call (per-call /health check), and a valid call "
        "afterward still succeeds — building on Stage 13's "
        "fresh-environment finding that no code path here can bring the "
        "process down."
    )


def main() -> None:
    scenario_a_honest_exception()
    scenario_b_groq_degradation()
    scenario_c_malformed_input()
    _header("ALL THREE FAILURE-INJECTION SCENARIOS PASSED")


if __name__ == "__main__":
    main()
