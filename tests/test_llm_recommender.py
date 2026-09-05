"""Unit tests for src/recon_agent/llm/recommender.py (Stage 10).

This module is exercised entirely against a mocked Groq client — no live
GROQ_API_KEY is required to run this suite (see this stage's brief: "don't
require a live API key to run the test suite"). An optional, clearly-marked
live integration test at the bottom is skipped unless
``RUN_LIVE_GROQ_TESTS`` is set in the environment.

Coverage, matching this stage's required scenarios:
  - happy path: a valid candidate pick
  - a NO_MATCH response
  - a malformed-JSON response            -> LLMMalformedJSONError
  - schema-violating-but-valid-JSON       -> LLMSchemaValidationError
  - a hallucinated candidate_id           -> LLMHallucinatedCandidateError
  - a timeout                             -> LLMTimeoutError
  - a non-timeout API failure             -> LLMAPIError
  - prompt construction never leaks raw_hash and carries the untrusted-data
    framing (security note, module docstring)
  - reasoning-wrapped responses (a <think>/<thinking> block, or plain
    prose, ahead of the real JSON answer -- as gpt-oss-120b may return
    even with reasoning_format='hidden' requested) are still parsed
    correctly, both for a candidate pick and a NO_MATCH
  - a reasoning-wrapped response with no recoverable JSON at all still
    raises LLMMalformedJSONError rather than silently guessing
  - recognized reasoning models get reasoning_format='hidden' and
    reasoning_effort='low' on the Groq call; non-reasoning models don't
"""

from __future__ import annotations

import json
import os
from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from groq import APIError, APITimeoutError, RateLimitError

from recon_agent.config import Settings
from recon_agent.llm.recommender import (
    NO_MATCH_SENTINEL,
    LLMAPIError,
    LLMHallucinatedCandidateError,
    LLMMalformedJSONError,
    LLMRateLimitError,
    LLMRecommendation,
    LLMSchemaValidationError,
    LLMTimeoutError,
    build_prompt,
    get_llm_recommendation,
)
from recon_agent.models import EntityType, NormalizedRecord, Source

# ---------------------------------------------------------------------------
# Fixture builders — mirrors tests/test_stage5_fuzzy.py's own `_record`.
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


def _settings() -> Settings:
    # No GROQ_API_KEY needed: a client is always injected in these tests.
    return Settings(groq_api_key=None, groq_model="llama-test-model")


def _mock_client_returning(content: str) -> MagicMock:
    """Build a mock groq.Groq client whose
    chat.completions.create(...) call returns `content` as the sole
    choice's message content."""
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


VALID_JSON = {
    "candidate_id": "cand-1",
    "confidence": 0.82,
    "reason_code": "STRONG_REFERENCE_MATCH",
    "explanation": "Reference and counterparty closely match candidate cand-1.",
}


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_happy_path_valid_candidate_pick():
    residual = _record("res-1")
    candidates = [_record("cand-1"), _record("cand-2")]
    client = _mock_client_returning(json.dumps(VALID_JSON))

    result = get_llm_recommendation(residual, candidates, _settings(), client=client)

    assert isinstance(result, LLMRecommendation)
    assert result.is_match is True
    assert result.candidate_id == "cand-1"
    assert result.confidence == pytest.approx(0.82)
    assert result.reason_code == "STRONG_REFERENCE_MATCH"
    assert result.residual_record_id == "res-1"
    assert result.model_used == "llama-test-model"
    assert result.offered_candidate_ids == ("cand-1", "cand-2")

    # The call actually used the configured model, not a hardcoded one.
    _, kwargs = client.chat.completions.create.call_args
    assert kwargs["model"] == "llama-test-model"


def test_happy_path_uses_configured_model_not_hardcoded():
    residual = _record("res-1")
    candidates = [_record("cand-1")]
    settings = Settings(groq_api_key=None, groq_model="some-other-llama-snapshot")
    client = _mock_client_returning(
        json.dumps({**VALID_JSON, "candidate_id": "cand-1"})
    )

    get_llm_recommendation(residual, candidates, settings, client=client)

    _, kwargs = client.chat.completions.create.call_args
    assert kwargs["model"] == "some-other-llama-snapshot"


# ---------------------------------------------------------------------------
# NO_MATCH
# ---------------------------------------------------------------------------


def test_happy_path_reasoning_wrapped_think_block_is_stripped():
    """gpt-oss-120b (and other Groq reasoning models) may prepend a
    <think>...</think> block ahead of the actual JSON answer even when
    reasoning_format='hidden' was requested — the response parsing must
    handle this without treating it as malformed JSON."""
    residual = _record("res-1")
    candidates = [_record("cand-1"), _record("cand-2")]
    reasoning_wrapped = (
        "<think>\nLet me compare the residual record against each "
        "candidate. The reference and counterparty for cand-1 line up "
        "closely with the residual record, cand-2 does not.\n</think>\n"
        + json.dumps(VALID_JSON)
    )
    client = _mock_client_returning(reasoning_wrapped)

    result = get_llm_recommendation(residual, candidates, _settings(), client=client)

    assert result.is_match is True
    assert result.candidate_id == "cand-1"
    assert result.confidence == pytest.approx(0.82)
    assert result.reason_code == "STRONG_REFERENCE_MATCH"


def test_reasoning_wrapped_response_with_no_match_is_stripped():
    residual = _record("res-1")
    candidates = [_record("cand-1"), _record("cand-2")]
    payload = {
        "candidate_id": NO_MATCH_SENTINEL,
        "confidence": 0.15,
        "reason_code": "NO_PLAUSIBLE_CANDIDATE",
        "explanation": "Neither candidate is convincingly the same transaction.",
    }
    reasoning_wrapped = (
        "<thinking>Weighing both candidates against the residual "
        "record...\nneither is a strong enough match.</thinking>"
        + json.dumps(payload)
    )
    client = _mock_client_returning(reasoning_wrapped)

    result = get_llm_recommendation(residual, candidates, _settings(), client=client)

    assert result.is_match is False
    assert result.reason_code == "NO_PLAUSIBLE_CANDIDATE"


def test_reasoning_wrapped_response_without_think_tag_falls_back_to_json_scan():
    """General safety net: a reasoning model that prefixes plain prose
    (not wrapped in a recognized <think> tag) ahead of its JSON answer
    must still be parsed, by scanning for the first valid embedded JSON
    object."""
    residual = _record("res-1")
    candidates = [_record("cand-1")]
    prose_wrapped = (
        "Sure, let's work through this step by step before answering.\n"
        "The residual record's reference matches candidate cand-1 "
        "closely, so I'll recommend it.\n\n" + json.dumps(VALID_JSON)
    )
    client = _mock_client_returning(prose_wrapped)

    result = get_llm_recommendation(residual, candidates, _settings(), client=client)

    assert result.is_match is True
    assert result.candidate_id == "cand-1"


def test_reasoning_model_requests_hidden_reasoning_format_and_low_effort():
    """For a recognized reasoning model (gpt-oss-120b), the Groq call
    itself should ask for reasoning_format='hidden' and
    reasoning_effort='low' — belt-and-suspenders alongside the response
    parsing above, and keeps latency/token usage down for a live demo."""
    residual = _record("res-1")
    candidates = [_record("cand-1")]
    settings = Settings(groq_api_key=None, groq_model="openai/gpt-oss-120b")
    client = _mock_client_returning(json.dumps(VALID_JSON))

    get_llm_recommendation(residual, candidates, settings, client=client)

    _, kwargs = client.chat.completions.create.call_args
    assert kwargs["reasoning_format"] == "hidden"
    assert kwargs["reasoning_effort"] == "low"


def test_non_reasoning_model_does_not_send_reasoning_kwargs():
    residual = _record("res-1")
    candidates = [_record("cand-1")]
    client = _mock_client_returning(json.dumps(VALID_JSON))

    get_llm_recommendation(residual, candidates, _settings(), client=client)

    _, kwargs = client.chat.completions.create.call_args
    assert "reasoning_format" not in kwargs
    assert "reasoning_effort" not in kwargs


def test_no_match_response():
    residual = _record("res-1")
    candidates = [_record("cand-1"), _record("cand-2")]
    payload = {
        "candidate_id": NO_MATCH_SENTINEL,
        "confidence": 0.2,
        "reason_code": "NO_PLAUSIBLE_CANDIDATE",
        "explanation": "Neither candidate's reference or amount is close enough.",
    }
    client = _mock_client_returning(json.dumps(payload))

    result = get_llm_recommendation(residual, candidates, _settings(), client=client)

    assert result.is_match is False
    assert result.candidate_id is None
    assert result.reason_code == "NO_PLAUSIBLE_CANDIDATE"
    assert result.offered_candidate_ids == ("cand-1", "cand-2")


# ---------------------------------------------------------------------------
# Failure modes — each distinct, not lumped together.
# ---------------------------------------------------------------------------


def test_malformed_json_response_raises_distinct_error():
    residual = _record("res-1")
    candidates = [_record("cand-1")]
    client = _mock_client_returning("this is not json at all {")

    with pytest.raises(LLMMalformedJSONError):
        get_llm_recommendation(residual, candidates, _settings(), client=client)


def test_reasoning_wrapped_response_with_no_recoverable_json_still_raises():
    """A think-block-wrapped response that never actually contains a
    valid JSON object must still fail loudly, not be silently guessed
    at."""
    residual = _record("res-1")
    candidates = [_record("cand-1")]
    client = _mock_client_returning(
        "<think>I'm not going to answer in JSON this time.</think>"
        "Sorry, I can't help with that."
    )

    with pytest.raises(LLMMalformedJSONError):
        get_llm_recommendation(residual, candidates, _settings(), client=client)


def test_schema_violating_json_raises_distinct_error():
    residual = _record("res-1")
    candidates = [_record("cand-1")]
    # Valid JSON, but confidence is out of [0, 1] and reason_code is missing.
    bad_payload = {
        "candidate_id": "cand-1",
        "confidence": 1.7,
        "explanation": "Missing reason_code and confidence is out of range.",
    }
    client = _mock_client_returning(json.dumps(bad_payload))

    with pytest.raises(LLMSchemaValidationError):
        get_llm_recommendation(residual, candidates, _settings(), client=client)


def test_schema_violating_extra_field_raises_distinct_error():
    residual = _record("res-1")
    candidates = [_record("cand-1")]
    payload = {**VALID_JSON, "candidate_id": "cand-1", "unexpected_field": "sneaky"}
    client = _mock_client_returning(json.dumps(payload))

    with pytest.raises(LLMSchemaValidationError):
        get_llm_recommendation(residual, candidates, _settings(), client=client)


def test_hallucinated_candidate_id_raises_distinct_error():
    residual = _record("res-1")
    candidates = [_record("cand-1"), _record("cand-2")]
    payload = {**VALID_JSON, "candidate_id": "cand-999-does-not-exist"}
    client = _mock_client_returning(json.dumps(payload))

    with pytest.raises(LLMHallucinatedCandidateError):
        get_llm_recommendation(residual, candidates, _settings(), client=client)


def test_timeout_raises_distinct_error():
    residual = _record("res-1")
    candidates = [_record("cand-1")]
    client = _mock_client_raising(APITimeoutError(request=MagicMock()))

    with pytest.raises(LLMTimeoutError):
        get_llm_recommendation(residual, candidates, _settings(), client=client)


def test_generic_api_error_raises_distinct_error_not_timeout():
    residual = _record("res-1")
    candidates = [_record("cand-1")]
    client = _mock_client_raising(
        APIError(message="internal error", request=MagicMock(), body=None)
    )

    with pytest.raises(LLMAPIError) as exc_info:
        get_llm_recommendation(residual, candidates, _settings(), client=client)

    # Must not be misreported as a timeout or a rate limit.
    assert not isinstance(exc_info.value, LLMTimeoutError)
    assert not isinstance(exc_info.value, LLMRateLimitError)


def _rate_limit_error() -> RateLimitError:
    """A real ``groq.RateLimitError`` (HTTP 429), constructed the way the
    groq SDK itself would build one — a genuine ``httpx.Response`` with
    ``status_code=429``, not just a generically-worded ``APIError``."""
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(status_code=429, request=request)
    return RateLimitError(message="rate limit reached", response=response, body=None)


def test_rate_limit_response_raises_distinct_error_not_generic_api_error():
    """A 429 must be its own distinct, immediately-recognized failure
    mode — never lumped into the generic LLMAPIError bucket (even
    though it *is* one, by subclassing — see recommender.py)."""
    residual = _record("res-1")
    candidates = [_record("cand-1")]
    client = _mock_client_raising(_rate_limit_error())

    with pytest.raises(LLMRateLimitError) as exc_info:
        get_llm_recommendation(residual, candidates, _settings(), client=client)

    assert not isinstance(exc_info.value, LLMTimeoutError)
    # Still catchable by anything handling the generic API-failure
    # bucket, since it's a genuine LLMAPIError subclass.
    assert isinstance(exc_info.value, LLMAPIError)


def test_rate_limit_error_is_a_distinct_type_from_generic_api_error():
    assert LLMRateLimitError is not LLMAPIError
    assert issubclass(LLMRateLimitError, LLMAPIError)


def test_client_constructed_with_max_retries_zero_when_none_injected(monkeypatch):
    """The Groq SDK retries a 429 internally (with its own backoff) by
    default — recommender.py must disable that so a rate limit is
    reported back immediately rather than silently waited out inside
    the SDK before this module ever sees it."""
    captured_kwargs: dict = {}

    class _FakeGroq:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)
            self.chat = MagicMock()
            message = SimpleNamespace(content=json.dumps(VALID_JSON))
            choice = SimpleNamespace(message=message)
            response = SimpleNamespace(choices=[choice])
            self.chat.completions.create.return_value = response

    monkeypatch.setattr("recon_agent.llm.recommender.Groq", _FakeGroq)

    residual = _record("res-1")
    candidates = [_record("cand-1")]
    get_llm_recommendation(residual, candidates, _settings())

    assert captured_kwargs.get("max_retries") == 0


def test_empty_candidate_pool_is_rejected_without_calling_api():
    residual = _record("res-1")
    client = _mock_client_returning(json.dumps(VALID_JSON))

    with pytest.raises(LLMSchemaValidationError):
        get_llm_recommendation(residual, [], _settings(), client=client)

    client.chat.completions.create.assert_not_called()


# ---------------------------------------------------------------------------
# Failure modes are genuinely distinct exception types from each other.
# ---------------------------------------------------------------------------


def test_failure_modes_are_pairwise_distinct_types():
    assert LLMTimeoutError is not LLMAPIError
    assert LLMMalformedJSONError is not LLMSchemaValidationError
    assert LLMSchemaValidationError is not LLMHallucinatedCandidateError
    assert not issubclass(LLMTimeoutError, LLMMalformedJSONError)
    assert not issubclass(LLMHallucinatedCandidateError, LLMSchemaValidationError)


# ---------------------------------------------------------------------------
# Prompt construction — §8 control table (normalized fields only, no
# raw_hash, untrusted-data framing).
# ---------------------------------------------------------------------------


def test_prompt_contains_only_normalized_fields_no_raw_hash():
    residual = _record("res-1", reference="RESIDUAL-REF")
    candidates = [_record("cand-1", reference="CAND-REF")]

    system_prompt, user_prompt = build_prompt(residual, candidates)

    assert "hash-res-1" not in user_prompt
    assert "hash-cand-1" not in user_prompt
    assert "raw_hash" not in user_prompt
    assert "RESIDUAL-REF" in user_prompt
    assert "CAND-REF" in user_prompt
    assert "res-1" in user_prompt
    assert "cand-1" in user_prompt


def test_prompt_frames_record_content_as_untrusted_data():
    residual = _record("res-1")
    candidates = [_record("cand-1")]

    system_prompt, _ = build_prompt(residual, candidates)

    lowered = system_prompt.lower()
    assert "not instructions" in lowered or "not as instructions" in lowered
    assert "data" in lowered


def test_prompt_payload_is_valid_json_with_expected_shape():
    residual = _record("res-1")
    candidates = [_record("cand-1"), _record("cand-2")]

    _, user_prompt = build_prompt(residual, candidates)

    # The user message wraps a JSON blob; extract and parse just that part.
    json_start = user_prompt.index("{")
    payload = json.loads(user_prompt[json_start:])
    assert payload["residual_record"]["record_id"] == "res-1"
    assert [c["record_id"] for c in payload["candidates"]] == ["cand-1", "cand-2"]
    assert "raw_hash" not in payload["residual_record"]


# ---------------------------------------------------------------------------
# Optional, skippable live integration test — never required to pass the
# core suite. Only runs if RUN_LIVE_GROQ_TESTS is set AND a real
# GROQ_API_KEY is present.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (os.environ.get("RUN_LIVE_GROQ_TESTS") and os.environ.get("GROQ_API_KEY")),
    reason=(
        "Live Groq integration test skipped by default; set "
        "RUN_LIVE_GROQ_TESTS=1 and a real GROQ_API_KEY to run it."
    ),
)
def test_live_groq_call_returns_a_recommendation():
    settings = Settings.from_env()
    residual = _record(
        "live-res-1", reference="INV-2026-004821", counterparty="Bluepeak Traders"
    )
    candidates = [
        _record(
            "live-cand-1",
            reference="INV-2026-004821",
            counterparty="Bluepeak Traders Pvt Ltd",
        ),
        _record("live-cand-2", reference="INV-2026-009911", counterparty="Zeta Corp"),
    ]

    result = get_llm_recommendation(residual, candidates, settings)

    assert isinstance(result, LLMRecommendation)
    assert 0.0 <= result.confidence <= 1.0
