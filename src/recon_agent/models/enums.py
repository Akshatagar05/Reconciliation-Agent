"""Enumerations for the reconciliation agent's data model.

Every enum here is transcribed directly from ARCHITECTURE.md §7 (Full Data
Model) and §8 (Exception Taxonomy). Do not add, remove, or rename values —
this file is the single source of truth referenced by every model in
``recon_agent.models``.
"""

from __future__ import annotations

from enum import Enum


class Source(str, Enum):
    """Origin system of a record. Used by NormalizedRecord and
    MatchGroupMember (§7)."""

    BANK = "BANK"
    GATEWAY = "GATEWAY"
    LEDGER = "LEDGER"


class EntityType(str, Enum):
    """NormalizedRecord.entity_type (§7)."""

    PAYMENT = "PAYMENT"
    BANK_CREDIT = "BANK_CREDIT"
    GATEWAY_SETTLEMENT = "GATEWAY_SETTLEMENT"
    LEDGER_ENTRY = "LEDGER_ENTRY"
    FEE = "FEE"
    TAX = "TAX"
    REFUND = "REFUND"
    CHARGEBACK = "CHARGEBACK"
    REVERSAL = "REVERSAL"
    ADJUSTMENT = "ADJUSTMENT"


class ReconciliationRunStatus(str, Enum):
    """ReconciliationRun.status (§7)."""

    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    COMPLETED_WITH_EXCEPTIONS = "COMPLETED_WITH_EXCEPTIONS"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"


class Cardinality(str, Enum):
    """MatchGroup.cardinality (§7). Matching cardinality is scoped to 1:1
    and many-to-one per §1 — MANY_TO_MANY remains in the enum for schema
    completeness and future work only; the resolver does not produce it
    in this build (§3)."""

    ONE_TO_ONE = "ONE_TO_ONE"
    ONE_TO_MANY = "ONE_TO_MANY"
    MANY_TO_ONE = "MANY_TO_ONE"
    MANY_TO_MANY = "MANY_TO_MANY"


class MatchGroupStatus(str, Enum):
    """MatchGroup.status (§7)."""

    VERIFIED = "VERIFIED"
    PENDING_REVIEW = "PENDING_REVIEW"
    REJECTED = "REJECTED"


class ProposedBy(str, Enum):
    """MatchGroup.proposed_by (§6 / §7) — who proposed the candidate
    group. Note: LLM_CORROBORATED_AUTO_COMMIT does not exist anywhere in
    this schema; it described a dead path removed in v4.1 (§2, §6)."""

    STAGE1_EXACT = "STAGE1_EXACT"
    STAGE2_CONSTRAINED = "STAGE2_CONSTRAINED"
    STAGE3_AGGREGATE = "STAGE3_AGGREGATE"
    STAGE4_ADJUSTMENT = "STAGE4_ADJUSTMENT"
    STAGE5_FUZZY = "STAGE5_FUZZY"
    STAGE6_LLM = "STAGE6_LLM"


class VerifiedBy(str, Enum):
    """MatchGroup.verified_by (§6 / §7). Only changes away from
    NOT_YET_VERIFIED once a verification step has actually run."""

    NOT_YET_VERIFIED = "NOT_YET_VERIFIED"
    FINANCIAL_AND_EVIDENCE_VERIFIER = "FINANCIAL_AND_EVIDENCE_VERIFIER"
    HUMAN_REVIEWER = "HUMAN_REVIEWER"


class CommitPolicy(str, Enum):
    """MatchGroup.commit_policy (§6 / §7). Stage 6 (STAGE6_LLM) must
    always carry HUMAN_REVIEW_REQUIRED — there is no auto-commit path for
    LLM output, ever (§2)."""

    AUTO_COMMIT_STAGE1 = "AUTO_COMMIT_STAGE1"
    AUTO_COMMIT_STAGE2_5_THRESHOLD = "AUTO_COMMIT_STAGE2_5_THRESHOLD"
    HUMAN_REVIEW_REQUIRED = "HUMAN_REVIEW_REQUIRED"


class VerificationResult(str, Enum):
    """MatchGroup.verification_result (§6 / §7).

    RESIDUAL_EXCEEDS_TOLERANCE is a post-launch addition (bugfix,
    documented in BUILD_LOG.md): Stage 1 (exact identifier match)
    groups pass currency/date/uniqueness re-checks but were never
    checked against a bounded residual between their LEDGER and
    GATEWAY legs, so two records sharing a reference by data-quality
    accident could auto-verify with an unbounded, unexplained gap.
    Like FAILED_MARGIN, this routes to PENDING_REVIEW (ambiguous, not
    disproven), never REJECTED — the identifier match is still real
    evidence.
    """

    PASSED = "PASSED"
    FAILED_CONSERVATION = "FAILED_CONSERVATION"
    FAILED_UNIQUENESS = "FAILED_UNIQUENESS"
    FAILED_CURRENCY = "FAILED_CURRENCY"
    FAILED_MARGIN = "FAILED_MARGIN"
    RESIDUAL_EXCEEDS_TOLERANCE = "RESIDUAL_EXCEEDS_TOLERANCE"
    NOT_YET_RUN = "NOT_YET_RUN"


class ReviewAction(str, Enum):
    """MatchGroup.review_action (§6 / §7) — set only once a human
    reviewer acts on a PENDING_REVIEW group."""

    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    ESCALATED = "ESCALATED"


class MemberRole(str, Enum):
    """MatchGroupMember.role (§3 / §7)."""

    GROSS = "GROSS"
    CREDIT = "CREDIT"
    FEE = "FEE"
    TAX = "TAX"
    REFUND = "REFUND"
    CHARGEBACK = "CHARGEBACK"
    REVERSAL = "REVERSAL"
    ADJUSTMENT = "ADJUSTMENT"


class DecisionStage(str, Enum):
    """DecisionEvent.stage (§7) — includes STAGE7_VERIFICATION, which
    does not appear in MatchGroup.proposed_by since the verifier does not
    propose groups, it evaluates them (§2)."""

    STAGE1_EXACT = "STAGE1_EXACT"
    STAGE2_CONSTRAINED = "STAGE2_CONSTRAINED"
    STAGE3_AGGREGATE = "STAGE3_AGGREGATE"
    STAGE4_ADJUSTMENT = "STAGE4_ADJUSTMENT"
    STAGE5_FUZZY = "STAGE5_FUZZY"
    STAGE6_LLM = "STAGE6_LLM"
    STAGE7_VERIFICATION = "STAGE7_VERIFICATION"


class ExceptionCategory(str, Enum):
    """Exception.category (§8 Exception Taxonomy, unchanged)."""

    DUPLICATE_RECORD = "DUPLICATE_RECORD"
    MISSING_SETTLEMENT = "MISSING_SETTLEMENT"
    ORPHAN_BANK_CREDIT = "ORPHAN_BANK_CREDIT"
    PARTIAL_SETTLEMENT = "PARTIAL_SETTLEMENT"
    FEE_GST_MISMATCH = "FEE_GST_MISMATCH"
    REFUND_CHARGEBACK_MISMATCH = "REFUND_CHARGEBACK_MISMATCH"
    CURRENCY_MISMATCH = "CURRENCY_MISMATCH"
    REUSED_COUNTERPART = "REUSED_COUNTERPART"
    AMBIGUOUS_AGGREGATION = "AMBIGUOUS_AGGREGATION"
    DATE_OUTSIDE_POLICY_WINDOW = "DATE_OUTSIDE_POLICY_WINDOW"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    LLM_UNAVAILABLE = "LLM_UNAVAILABLE"
    LLM_OUTPUT_REJECTED_BY_VERIFIER = "LLM_OUTPUT_REJECTED_BY_VERIFIER"


class ExceptionSeverity(str, Enum):
    """Exception.severity (§7)."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ExceptionReviewStatus(str, Enum):
    """Exception.review_status (§7)."""

    OPEN = "OPEN"
    REVIEWED = "REVIEWED"
