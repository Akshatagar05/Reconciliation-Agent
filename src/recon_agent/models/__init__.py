"""Pydantic v2 data models for the reconciliation agent.

Every model here corresponds 1:1 to an entity in ARCHITECTURE.md §7
(Full Data Model): ReconciliationRun, NormalizedRecord, MatchGroup,
MatchGroupMember, DecisionEvent, ReconciliationException. Enums live in
``enums.py`` and are transcribed from §7/§8.

Note: §7 names this last entity "Exception" as the conceptual term, but
it is implemented in code as ``ReconciliationException`` (see
``exception.py``) so it never shadows Python's builtin ``Exception``
class and never needs an aliased import.
"""

from __future__ import annotations

from recon_agent.models.decision_event import DecisionEvent
from recon_agent.models.enums import (
    Cardinality,
    CommitPolicy,
    DecisionStage,
    EntityType,
    ExceptionCategory,
    ExceptionReviewStatus,
    ExceptionSeverity,
    MatchGroupStatus,
    MemberRole,
    ProposedBy,
    ReconciliationRunStatus,
    ReviewAction,
    Source,
    VerificationResult,
    VerifiedBy,
)
from recon_agent.models.exception import ReconciliationException
from recon_agent.models.match_group import MatchGroup
from recon_agent.models.match_group_member import MatchGroupMember
from recon_agent.models.normalized_record import NormalizedRecord
from recon_agent.models.reconciliation_run import ReconciliationRun

__all__ = [
    # Entities (§7)
    "ReconciliationRun",
    "NormalizedRecord",
    "MatchGroup",
    "MatchGroupMember",
    "DecisionEvent",
    "ReconciliationException",
    # Enums (§7 / §8)
    "Source",
    "EntityType",
    "ReconciliationRunStatus",
    "Cardinality",
    "MatchGroupStatus",
    "ProposedBy",
    "VerifiedBy",
    "CommitPolicy",
    "VerificationResult",
    "ReviewAction",
    "MemberRole",
    "DecisionStage",
    "ExceptionCategory",
    "ExceptionSeverity",
    "ExceptionReviewStatus",
]
