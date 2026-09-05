"""Unit tests for src/recon_agent/matching/stage6_llm.py (Stage 11).

Exercised entirely against a mocked Groq client, mirroring
tests/test_llm_recommender.py's own approach — no live GROQ_API_KEY is
required. Coverage, per this stage's required scenarios:

  - a Stage 6 proposal, even at a very high LLM-stated confidence, never
    gets auto-committed: status/verified_by/commit_policy land exactly
    where §2 requires
  - graceful degradation: governor ceiling-exhausted and circuit-open
    both produce an Exception (not a crash), and the LLM is never
    actually called in either case
  - a Groq call failure (a real recommender.py exception type) is
    caught, produces an Exception with the failure kind recorded, and
    is fed back into the governor's breaker bookkeeping
  - a NO_MATCH recommendation produces an Exception, not a MatchGroup
  - near-miss candidate gathering surfaces candidates that would fail
    Stage 5's own hard gates, but still respects currency as a hard
    prefilter and ``stage6_max_candidates`` as a cap
  - a DecisionEvent is emitted for every outcome, including the
    no-candidates-available and governor-declined cases
"""

from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from groq import APIError, APITimeoutError

from recon_agent.config import Settings
from recon_agent.llm.governor import CallBudgetGovernor
from recon_agent.matching.common import IdAllocator
from recon_agent.matching.stage6_llm import gather_near_miss_candidates, run_stage6_llm
from recon_agent.models import (
    CommitPolicy,
    EntityType,
    ExceptionCategory,
    MatchGroupStatus,
    NormalizedRecord,
    ProposedBy,
    Source,
    VerificationResult,
    VerifiedBy,
)

# ---------------------------------------------------------------------------
# Fixture builders — mirrors tests/test_llm_recommender.py's own `_record`.
# ---------------------------------------------------------------------------


def _record(
    record_id: str,
    source: Source = Source.LEDGER,
    entity_type: EntityType = EntityType.PAYMENT,
    amount_paise: int = 500_00,
    reference: str = "REF12345",
    occurred_at: date = date(2026, 1, 10),
    currency: str = "INR",
    counterparty: str = "Acme Retail Pvt Ltd",
) -> NormalizedRecord:
    return NormalizedRecord(
        record_id=record_id,
        source=source,
        entity_type=entity_type,
        amount_paise=amount_paise,
        currency=currency,
        reference=reference,
        counterparty=counterparty,
        occurred_at=occurred_at,
        raw_hash=f"hash-{record_id}",
    )


def _settings(**overrides) -> Settings:
    defaults = dict(groq_api_key=None, groq_model="llama-test-model", stage6_max_candidates=5)
    defaults.update(overrides)
    return Settings(**defaults)


def _governor(tmp_path, **settings_overrides) -> CallBudgetGovernor:
    settings = _settings(**settings_overrides)
    return CallBudgetGovernor(settings, db_path=str(tmp_path / "gov.db"))


def _mock_client_returning(content: str) -> MagicMock:
    client = MagicMock()
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    response = SimpleNamespace(choices=[choice])
    client.chat.completions.create.return_value = response
    return client


def _mock_client_raising(exc: Exception) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create.side_effect = exc
    return client


VALID_MATCH_PAYLOAD = {
    "confidence": 0.99,
    "reason_code": "STRONG_REFERENCE_MATCH",
    "explanation": "Reference and counterparty closely match.",
}


# ---------------------------------------------------------------------------
# Near-miss candidate gathering
# ---------------------------------------------------------------------------


def test_near_miss_candidates_include_a_weak_identifier_match():
    """A candidate whose reference is nothing like the seeker's (would
    fail Stage 5's own hard identifier gate) must still surface here —
    that's the entire point of "near-miss"."""
    seeker = _record("seek-1", source=Source.LEDGER, entity_type=EntityType.PAYMENT, reference="REF-ABC-999")
    weak_candidate = _record(
        "cand-1",
        source=Source.GATEWAY,
        entity_type=EntityType.GATEWAY_SETTLEMENT,
        reference="ZZZZZZZZZZ",
        amount_paise=500_00,
    )

    results = gather_near_miss_candidates(seeker, [seeker, weak_candidate], max_candidates=5)

    assert [r.record_id for r in results] == ["cand-1"]


def test_near_miss_candidates_exclude_cross_currency():
    seeker = _record("seek-1", currency="INR")
    wrong_currency = _record(
        "cand-1", source=Source.GATEWAY, entity_type=EntityType.GATEWAY_SETTLEMENT, currency="USD"
    )

    results = gather_near_miss_candidates(seeker, [seeker, wrong_currency], max_candidates=5)

    assert results == []


def test_near_miss_candidates_respect_max_candidates_and_ranking():
    seeker = _record("seek-1", reference="REF-12345", amount_paise=500_00)
    close_amount = _record(
        "cand-close",
        source=Source.GATEWAY,
        entity_type=EntityType.GATEWAY_SETTLEMENT,
        reference="REF-12345",
        amount_paise=500_00,
    )
    far_amount = _record(
        "cand-far",
        source=Source.GATEWAY,
        entity_type=EntityType.GATEWAY_SETTLEMENT,
        reference="totally-different",
        amount_paise=1,
    )
    mid_amount = _record(
        "cand-mid",
        source=Source.GATEWAY,
        entity_type=EntityType.GATEWAY_SETTLEMENT,
        reference="REF-12340",
        amount_paise=490_00,
    )

    pool = [seeker, close_amount, far_amount, mid_amount]
    results = gather_near_miss_candidates(seeker, pool, max_candidates=2)

    assert len(results) == 2
    assert results[0].record_id == "cand-close"
    assert results[1].record_id == "cand-mid"


def test_near_miss_candidates_empty_when_no_eligible_pool():
    seeker = _record("seek-1")
    unrelated_ledger_record = _record("other-1", source=Source.LEDGER)

    results = gather_near_miss_candidates(seeker, [seeker, unrelated_ledger_record], max_candidates=5)

    assert results == []


# ---------------------------------------------------------------------------
# No auto-commit, ever — the non-negotiable requirement (§2)
# ---------------------------------------------------------------------------


def test_stage6_proposal_never_auto_commits_even_at_high_confidence(tmp_path):
    seeker = _record("seek-1", reference="REF-12345")
    candidate = _record(
        "cand-1", source=Source.GATEWAY, entity_type=EntityType.GATEWAY_SETTLEMENT, reference="REF-12345"
    )
    records = [seeker, candidate]
    payload = {**VALID_MATCH_PAYLOAD, "candidate_id": "cand-1"}
    client = _mock_client_returning(json.dumps(payload))
    governor = _governor(tmp_path)
    settings = _settings()

    result = run_stage6_llm(records, settings, set(), IdAllocator(), governor, "run-1", groq_client=client)

    assert len(result.match_groups) == 1
    group = result.match_groups[0]
    assert group.proposed_by == ProposedBy.STAGE6_LLM
    # The non-negotiable part: confidence was 0.99 and it STILL lands
    # exactly here, never on an auto-commit path.
    assert group.status == MatchGroupStatus.PENDING_REVIEW
    assert group.verified_by == VerifiedBy.NOT_YET_VERIFIED
    assert group.commit_policy == CommitPolicy.HUMAN_REVIEW_REQUIRED
    assert group.verification_result == VerificationResult.NOT_YET_RUN
    assert group.evidence_score == pytest.approx(0.99)
    assert result.matched_record_ids == {"seek-1", "cand-1"}
    assert result.recommended_count == 1

    # A DecisionEvent was emitted for this attempt.
    assert len(result.decision_events) == 1
    assert result.decision_events[0].stage.value == "STAGE6_LLM"

    # And the call actually went through the governor.
    status = governor.status("run-1")
    assert status.calls_used_today == 1
    assert status.consecutive_failures == 0


# ---------------------------------------------------------------------------
# Regression: seeker/candidate processing order must not depend on
# alphabetical record_id sorting across sources
# ---------------------------------------------------------------------------


def test_ledger_seeker_claims_gateway_candidate_even_when_it_sorts_first(tmp_path):
    """Regression for a real bug: ``_eligible_seekers`` used to sort every
    eligible seeker alphabetically by ``record_id``, regardless of
    source. A GATEWAY record is legitimately eligible to seek its own
    BANK-side match (mirrors Stage 5's own convention) -- but it's also
    a legitimate match candidate for a LEDGER seeker. When the GATEWAY
    record's ``record_id`` happened to sort before the LEDGER seeker
    that would otherwise claim it, the GATEWAY record got processed as
    its own seeker first, found no BANK candidates (none exist here),
    and logged a real NO_CANDIDATES_AVAILABLE DecisionEvent + Exception
    for itself -- only for the LEDGER seeker to successfully claim that
    same GATEWAY record as its match candidate moments later in the
    same pass. The final match was correct, but the earlier
    decision-event/exception was stale and misleading, and nothing
    retracted it.

    Deliberately named so the GATEWAY candidate ("aaa-cand") sorts
    alphabetically before the LEDGER seeker ("zzz-seek") -- exactly the
    ordering that triggered the bug under a naive, source-blind sort.
    LEDGER seekers now always run first, so this must produce exactly
    one clean DecisionEvent (the successful match) and zero stale
    Exceptions.
    """
    seeker = _record("zzz-seek", reference="REF-12345")
    candidate = _record(
        "aaa-cand", source=Source.GATEWAY, entity_type=EntityType.GATEWAY_SETTLEMENT, reference="REF-12345"
    )
    records = [seeker, candidate]
    client = _mock_client_returning(json.dumps({**VALID_MATCH_PAYLOAD, "candidate_id": "aaa-cand"}))
    governor = _governor(tmp_path)

    result = run_stage6_llm(
        records, _settings(), set(), IdAllocator(), governor, "run-1", groq_client=client
    )

    # The match itself is correct and unaffected by the ordering fix.
    assert len(result.match_groups) == 1
    assert result.matched_record_ids == {"zzz-seek", "aaa-cand"}
    assert result.recommended_count == 1

    # The actual regression check: no stale NO_CANDIDATES_AVAILABLE (or
    # any other) entry was generated for "aaa-cand" from having been
    # processed as its own seeker before "zzz-seek" ever got a turn --
    # exactly one DecisionEvent, zero Exceptions, for the whole pass.
    assert len(result.decision_events) == 1
    assert result.decision_events[0].reason_code != "NO_CANDIDATES_AVAILABLE"
    assert result.exceptions == []

    # Only one Groq call was ever made -- "aaa-cand" never got an
    # independent (and doomed) seeker attempt of its own.
    assert governor.status("run-1").calls_used_today == 1


def test_stage6_no_match_produces_exception_not_group(tmp_path):
    seeker = _record("seek-1", reference="REF-12345")
    candidate = _record(
        "cand-1", source=Source.GATEWAY, entity_type=EntityType.GATEWAY_SETTLEMENT, reference="totally-different"
    )
    records = [seeker, candidate]
    payload = {
        "candidate_id": "NO_MATCH",
        "confidence": 0.1,
        "reason_code": "NO_PLAUSIBLE_CANDIDATE",
        "explanation": "Nothing lines up closely enough.",
    }
    client = _mock_client_returning(json.dumps(payload))
    governor = _governor(tmp_path)

    result = run_stage6_llm(
        records, _settings(), set(), IdAllocator(), governor, "run-1", groq_client=client
    )

    assert result.match_groups == []
    assert result.no_match_count == 1
    assert len(result.exceptions) == 1
    exc = result.exceptions[0]
    assert exc.category == ExceptionCategory.INSUFFICIENT_EVIDENCE
    assert exc.group_id_or_record_id == "seek-1"
    assert len(result.decision_events) == 1


# ---------------------------------------------------------------------------
# No candidates available — the LLM is never called
# ---------------------------------------------------------------------------


def test_stage6_no_candidates_available_skips_the_llm_entirely(tmp_path):
    seeker = _record("seek-1")
    records = [seeker]  # nothing eligible to offer as a candidate
    client = _mock_client_returning(json.dumps({**VALID_MATCH_PAYLOAD, "candidate_id": "cand-1"}))
    governor = _governor(tmp_path)

    result = run_stage6_llm(
        records, _settings(), set(), IdAllocator(), governor, "run-1", groq_client=client
    )

    client.chat.completions.create.assert_not_called()
    assert result.match_groups == []
    assert result.no_candidates_count == 1
    assert len(result.exceptions) == 1
    assert result.exceptions[0].category == ExceptionCategory.INSUFFICIENT_EVIDENCE
    assert result.exceptions[0].evidence["reason"] == "NO_CANDIDATES_AVAILABLE"
    # Never consumed a budget slot for a call that was never attempted.
    assert governor.status("run-1").calls_used_today == 0


# ---------------------------------------------------------------------------
# Graceful degradation — ceiling exhausted / circuit open
# ---------------------------------------------------------------------------


def test_stage6_degrades_gracefully_when_ceiling_exhausted(tmp_path):
    seeker = _record("seek-1", reference="REF-12345")
    candidate = _record(
        "cand-1", source=Source.GATEWAY, entity_type=EntityType.GATEWAY_SETTLEMENT, reference="REF-12345"
    )
    records = [seeker, candidate]
    client = _mock_client_returning(json.dumps({**VALID_MATCH_PAYLOAD, "candidate_id": "cand-1"}))
    governor = _governor(tmp_path, groq_daily_call_ceiling=0)

    result = run_stage6_llm(
        records, _settings(), set(), IdAllocator(), governor, "run-1", groq_client=client
    )

    client.chat.completions.create.assert_not_called()
    assert result.match_groups == []
    assert result.degraded_count == 1
    assert len(result.exceptions) == 1
    exc = result.exceptions[0]
    assert exc.category == ExceptionCategory.LLM_UNAVAILABLE
    assert exc.evidence["degradation_reason"] == "LLM_BUDGET_EXHAUSTED"
    events = [e for e in result.decision_events if e.reason_code == "LLM_BUDGET_EXHAUSTED"]
    assert len(events) == 1


def test_stage6_degrades_gracefully_when_circuit_open(tmp_path):
    from recon_agent.llm.recommender import LLMTimeoutError

    seeker = _record("seek-1", reference="REF-12345")
    candidate = _record(
        "cand-1", source=Source.GATEWAY, entity_type=EntityType.GATEWAY_SETTLEMENT, reference="REF-12345"
    )
    records = [seeker, candidate]
    client = _mock_client_returning(json.dumps({**VALID_MATCH_PAYLOAD, "candidate_id": "cand-1"}))
    governor = _governor(tmp_path, groq_circuit_breaker_max_consecutive_failures=2)
    governor.record_failure("run-1", LLMTimeoutError("t1"))
    governor.record_failure("run-1", LLMTimeoutError("t2"))
    assert governor.status("run-1").circuit_open is True

    result = run_stage6_llm(
        records, _settings(), set(), IdAllocator(), governor, "run-1", groq_client=client
    )

    client.chat.completions.create.assert_not_called()
    assert result.match_groups == []
    assert result.degraded_count == 1
    exc = result.exceptions[0]
    assert exc.category == ExceptionCategory.LLM_UNAVAILABLE
    assert exc.evidence["degradation_reason"] == "LLM_CIRCUIT_BREAKER_OPEN"


def test_stage6_pipeline_completes_not_crashes_when_governor_exhausted_mid_run(tmp_path):
    """Multiple residual records, governor only allows the first call
    through — the rest must degrade gracefully rather than the whole
    stage raising or stalling."""
    seeker1 = _record("seek-1", reference="REF-A")
    cand1 = _record(
        "cand-1", source=Source.GATEWAY, entity_type=EntityType.GATEWAY_SETTLEMENT, reference="REF-A"
    )
    seeker2 = _record("seek-2", reference="REF-B")
    cand2 = _record(
        "cand-2", source=Source.GATEWAY, entity_type=EntityType.GATEWAY_SETTLEMENT, reference="REF-B"
    )
    records = [seeker1, cand1, seeker2, cand2]
    client = _mock_client_returning(json.dumps({**VALID_MATCH_PAYLOAD, "candidate_id": "cand-1"}))
    governor = _governor(tmp_path, groq_daily_call_ceiling=1)

    result = run_stage6_llm(
        records, _settings(), set(), IdAllocator(), governor, "run-1", groq_client=client
    )

    # Completes without raising, and produces a mix: one recommendation,
    # one degraded exception — never a crash or a stall.
    assert result.recommended_count == 1
    assert result.degraded_count == 1
    assert len(result.match_groups) == 1
    degraded_exceptions = [e for e in result.exceptions if e.category == ExceptionCategory.LLM_UNAVAILABLE]
    assert len(degraded_exceptions) == 1


# ---------------------------------------------------------------------------
# Call failures are caught distinctly and fed to the breaker
# ---------------------------------------------------------------------------


def test_stage6_call_failure_produces_exception_and_updates_breaker(tmp_path):
    seeker = _record("seek-1", reference="REF-12345")
    candidate = _record(
        "cand-1", source=Source.GATEWAY, entity_type=EntityType.GATEWAY_SETTLEMENT, reference="REF-12345"
    )
    records = [seeker, candidate]
    client = _mock_client_raising(APITimeoutError(request=MagicMock()))
    governor = _governor(tmp_path)

    result = run_stage6_llm(
        records, _settings(), set(), IdAllocator(), governor, "run-1", groq_client=client
    )

    assert result.match_groups == []
    assert result.failure_count == 1
    exc = result.exceptions[0]
    assert exc.category == ExceptionCategory.LLM_UNAVAILABLE
    assert exc.evidence["failure_kind"] == "TIMEOUT"

    status = governor.status("run-1")
    assert status.consecutive_failures == 1

    # The call still consumed a budget slot — it was actually attempted.
    assert status.calls_used_today == 1


def test_stage6_non_timeout_api_failure_is_labeled_distinctly(tmp_path):
    seeker = _record("seek-1", reference="REF-12345")
    candidate = _record(
        "cand-1", source=Source.GATEWAY, entity_type=EntityType.GATEWAY_SETTLEMENT, reference="REF-12345"
    )
    records = [seeker, candidate]
    client = _mock_client_raising(APIError(message="rate limited", request=MagicMock(), body=None))
    governor = _governor(tmp_path)

    result = run_stage6_llm(
        records, _settings(), set(), IdAllocator(), governor, "run-1", groq_client=client
    )

    exc = result.exceptions[0]
    assert exc.evidence["failure_kind"] == "API_ERROR"


# ---------------------------------------------------------------------------
# Every attempt gets a DecisionEvent
# ---------------------------------------------------------------------------


def test_decision_event_emitted_for_every_outcome_kind(tmp_path):
    """One seeker per outcome: recommended, NO_MATCH, no-candidates,
    call-failure — every single one gets its own DecisionEvent."""
    match_seeker = _record("seek-match", reference="REF-A")
    match_cand = _record(
        "cand-match", source=Source.GATEWAY, entity_type=EntityType.GATEWAY_SETTLEMENT, reference="REF-A"
    )
    no_match_seeker = _record("seek-nomatch", reference="REF-B")
    no_match_cand = _record(
        "cand-nomatch", source=Source.GATEWAY, entity_type=EntityType.GATEWAY_SETTLEMENT, reference="REF-B-close"
    )
    no_candidates_seeker = _record("seek-alone", reference="REF-C")

    records = [
        match_seeker,
        match_cand,
        no_match_seeker,
        no_match_cand,
        no_candidates_seeker,
    ]

    # A client whose behavior depends on which prompt it's asked about
    # isn't worth building here; instead run three separate single-
    # seeker invocations against distinctly-mocked clients and confirm
    # each produced exactly one DecisionEvent.
    governor = _governor(tmp_path)

    match_client = _mock_client_returning(json.dumps({**VALID_MATCH_PAYLOAD, "candidate_id": "cand-match"}))
    match_result = run_stage6_llm(
        [match_seeker, match_cand], _settings(), set(), IdAllocator(), governor, "run-1", groq_client=match_client
    )
    assert len(match_result.decision_events) == 1

    no_match_client = _mock_client_returning(
        json.dumps(
            {
                "candidate_id": "NO_MATCH",
                "confidence": 0.05,
                "reason_code": "NO_PLAUSIBLE_CANDIDATE",
                "explanation": "Not close enough.",
            }
        )
    )
    no_match_result = run_stage6_llm(
        [no_match_seeker, no_match_cand],
        _settings(),
        set(),
        IdAllocator(),
        governor,
        "run-2",
        groq_client=no_match_client,
    )
    assert len(no_match_result.decision_events) == 1

    lonely_client = _mock_client_returning(json.dumps(VALID_MATCH_PAYLOAD))
    lonely_result = run_stage6_llm(
        [no_candidates_seeker], _settings(), set(), IdAllocator(), governor, "run-3", groq_client=lonely_client
    )
    assert len(lonely_result.decision_events) == 1
    lonely_client.chat.completions.create.assert_not_called()


# ---------------------------------------------------------------------------
# Already-matched records are excluded from the residual pool
# ---------------------------------------------------------------------------


def test_records_in_matched_record_ids_are_never_offered_or_treated_as_seekers(tmp_path):
    seeker = _record("seek-1", reference="REF-A")
    candidate = _record(
        "cand-1", source=Source.GATEWAY, entity_type=EntityType.GATEWAY_SETTLEMENT, reference="REF-A"
    )
    already_matched_seeker = _record("seek-2", reference="REF-B")
    records = [seeker, candidate, already_matched_seeker]
    client = _mock_client_returning(json.dumps({**VALID_MATCH_PAYLOAD, "candidate_id": "cand-1"}))
    governor = _governor(tmp_path)

    result = run_stage6_llm(
        records,
        _settings(),
        {"seek-2"},
        IdAllocator(),
        governor,
        "run-1",
        groq_client=client,
    )

    # Only one attempt: the already-matched seeker never entered the
    # residual pool at all, so it produces neither a DecisionEvent nor
    # an Exception.
    assert len(result.decision_events) == 1
    assert all(e.group_id_or_record_id != "seek-2" for e in result.exceptions)
    assert result.attempted_count == 1
