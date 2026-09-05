"""llm — Groq LLM recommendation module + governor (§10; Stages 10-11 of 16).

``recommender.py`` implements Stage 6's recommendation step
(ARCHITECTURE.md §2, §6, §8) as a **standalone** module: prompt
construction, the Groq API call, and strict validation of the response.

``governor.py`` (Stage 11) adds the shared budget governor / circuit
breaker around it — a daily call ceiling and a per-run consecutive-
failure circuit breaker, both backed by an atomic SQLite transaction so
concurrent runs can't double-count or under-count usage (§10's revision-
history correction). Stage 11 also wires both of these into
``matching/pipeline.py`` as the Stage 6 tier — see
``matching/stage6_llm.py``.
"""

from __future__ import annotations

from recon_agent.llm.governor import (
    CallBudgetGovernor,
    GovernorCheckResult,
    GovernorDecision,
    classify_failure,
)
from recon_agent.llm.recommender import (
    LLMAPIError,
    LLMHallucinatedCandidateError,
    LLMMalformedJSONError,
    LLMRateLimitError,
    LLMRecommendation,
    LLMRecommendationPayload,
    LLMSchemaValidationError,
    LLMTimeoutError,
    NO_MATCH_SENTINEL,
    RecommenderError,
    build_prompt,
    get_llm_recommendation,
)

__all__ = [
    "LLMAPIError",
    "LLMHallucinatedCandidateError",
    "LLMMalformedJSONError",
    "LLMRateLimitError",
    "LLMRecommendation",
    "LLMRecommendationPayload",
    "LLMSchemaValidationError",
    "LLMTimeoutError",
    "NO_MATCH_SENTINEL",
    "RecommenderError",
    "build_prompt",
    "get_llm_recommendation",
    "CallBudgetGovernor",
    "GovernorCheckResult",
    "GovernorDecision",
    "classify_failure",
]
