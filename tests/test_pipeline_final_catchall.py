"""Regression tests for the post-Stage-12 bugfix: matching/pipeline.py's
``run_final_catchall``, wired into ``run_pipeline`` after Stage 6 and
both verification passes.

The bug (found by the Stage 12 evaluation harness against real data):
a standalone record of a shape no stage's own eligibility rule ever
treats as a seeker or a candidate in its own right — the real example
being ``entity_type=ADJUSTMENT`` (adjustments are only ever consumed as
a MEMBER inside another group's conservation check, never a role of
their own; see ``common.py``'s ``LEDGER_TO_GATEWAY_ENTITY`` /
``GATEWAY_TO_BANK_ENTITY`` tables, which have no ADJUSTMENT entry, and
``stage6_llm.py``'s identical ``_target_entity_type`` mapping) — fell
through every stage silently: zero ``DecisionEvent``s, zero
``ReconciliationException``s, and absent from ``matched_record_ids``.
This violates ARCHITECTURE.md §14's acceptance gate ("every unresolved
record has a reason code, evidence, and a recommended action").

These tests build a record with exactly that shape directly (rather
than depending on the real calibration/evaluation datasets, which the
other integration tests already exercise for this) and assert the
pipeline now always produces an explicit decision + exception for it,
never silence.
"""

from __future__ import annotations

from datetime import date

from recon_agent.config import Settings
from recon_agent.matching.pipeline import run_pipeline
from recon_agent.models import (
    DecisionStage,
    EntityType,
    ExceptionCategory,
    ExceptionReviewStatus,
    NormalizedRecord,
    Source,
)


def _record(
    record_id: str,
    source: Source,
    entity_type: EntityType,
    amount_paise: int = 5_000,
) -> NormalizedRecord:
    return NormalizedRecord(
        record_id=record_id,
        source=source,
        entity_type=entity_type,
        amount_paise=amount_paise,
        currency="INR",
        reference=f"REF-{record_id}",
        counterparty="Acme Retail Pvt Ltd",
        occurred_at=date(2026, 1, 1),
        raw_hash=f"hash_{record_id}",
    )


def test_unattemptable_adjustment_record_is_never_silently_absent() -> None:
    """The real-world shape: a standalone LEDGER-sourced ADJUSTMENT
    record, mixed in with an ordinary, cleanly-matchable pair so the
    catch-all's presence doesn't accidentally depend on being the only
    record in the run. No stage should ever propose anything touching
    ``adj_orphan_1`` — it should end up unmatched AND carry an explicit
    exception, never neither.
    """
    ledger = _record("led_1", Source.LEDGER, EntityType.PAYMENT, 100_000)
    gateway = _record("gw_1", Source.GATEWAY, EntityType.GATEWAY_SETTLEMENT, 100_000)
    orphan_adjustment = _record("adj_orphan_1", Source.LEDGER, EntityType.ADJUSTMENT, 2_500)
    # Give the pair matching references/dates so Stage 1 (exact) claims
    # it cleanly — this is incidental scaffolding, not what's under test.
    ledger = ledger.model_copy(update={"reference": "SHARED-REF-001"})
    gateway = gateway.model_copy(update={"reference": "SHARED-REF-001"})

    records = [ledger, gateway, orphan_adjustment]
    result = run_pipeline(records, Settings())

    # The orphan is never force-matched.
    assert "adj_orphan_1" not in result.matched_record_ids

    # But it is never silently invisible either: at least one
    # DecisionEvent and at least one ReconciliationException reference
    # it, both attributed to the final catch-all pass.
    matching_events = [e for e in result.decision_events if "adj_orphan_1" in e.group_id]
    assert len(matching_events) >= 1
    for event in matching_events:
        assert event.stage == DecisionStage.STAGE7_VERIFICATION
        assert event.reason_code == "NO_STAGE_ATTEMPTED"

    matching_exceptions = [
        exc for exc in result.exceptions if exc.group_id_or_record_id == "adj_orphan_1"
    ]
    assert len(matching_exceptions) >= 1
    exc = matching_exceptions[0]
    assert exc.category == ExceptionCategory.INSUFFICIENT_EVIDENCE
    assert exc.review_status == ExceptionReviewStatus.OPEN
    assert exc.evidence["record_id"] == "adj_orphan_1"
    assert "manual review" in exc.recommended_action.lower()


def test_unattemptable_record_alone_still_produces_an_exception() -> None:
    """Same shape, but as the only record in the run at all — no other
    record for Stage 1-6 to occupy themselves with, so this isolates
    the catch-all pass from every other stage's behavior.
    """
    orphan_adjustment = _record("adj_orphan_2", Source.BANK, EntityType.ADJUSTMENT, 7_500)

    result = run_pipeline([orphan_adjustment], Settings())

    assert result.matched_record_ids == set()
    assert len(result.decision_events) == 1
    assert len(result.exceptions) == 1
    assert result.decision_events[0].stage == DecisionStage.STAGE7_VERIFICATION
    assert result.exceptions[0].group_id_or_record_id == "adj_orphan_2"
    assert result.exceptions[0].category == ExceptionCategory.INSUFFICIENT_EVIDENCE


def test_catchall_never_touches_a_record_some_stage_already_attempted() -> None:
    """Safety-net scope check: a record Stage 6 already attempted (and
    already produced its own DecisionEvent/Exception for, e.g.
    NO_CANDIDATES_AVAILABLE) must not get a *second*, redundant
    catch-all entry — the catch-all is only for records nothing ever
    touched at all. A lone, unmatched LEDGER PAYMENT record with no
    GATEWAY counterpart anywhere is a real Stage 6 seeker (PAYMENT has a
    LEDGER_TO_GATEWAY_ENTITY mapping), so it is exactly this case.
    """
    lonely_payment = _record("led_lonely", Source.LEDGER, EntityType.PAYMENT, 50_000)

    result = run_pipeline([lonely_payment], Settings())

    # Stage 1-6 themselves already produced real decision events for
    # it (a fuzzy-retrieval NO_ELIGIBLE_CANDIDATE, then a Stage 6
    # NO_CANDIDATES_AVAILABLE) — the point under test is that the
    # catch-all pass adds nothing further on top of that.
    events_for_record = [e for e in result.decision_events if "led_lonely" in e.group_id]
    exceptions_for_record = [
        exc for exc in result.exceptions if exc.group_id_or_record_id == "led_lonely"
    ]
    assert len(events_for_record) >= 1
    assert len(exceptions_for_record) == 1
    assert all(e.reason_code != "NO_STAGE_ATTEMPTED" for e in events_for_record)
    assert all(e.stage != DecisionStage.STAGE7_VERIFICATION for e in events_for_record)
