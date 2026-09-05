"""MatchGroup — ARCHITECTURE.md §6 and §7.

Replaces the old pairwise MatchAllocation (§3). Carries four distinct
provenance/verification fields (proposed_by / verified_by /
commit_policy / verification_result) rather than one collapsed field,
plus the evidence fields the verifier needs to prove *why* a group
passed (evidence_score, runner_up_score, score_margin, threshold_applied,
policy_checks) — see §2's note on replacing the vague `confidence` field.

Field-by-field provenance:
  - expected_amount_paise / matched_amount_paise / residual_amount_paise:
    int, per §7 (renamed from expected_amount / matched_amount /
    residual_amount in the v4.1 pre-coding correction, for consistency
    with every other *_paise amount field in the schema).
  - evidence_score / runner_up_score / score_margin: float|null, NEW
    in v4.1 for margin proof (§7).
  - threshold_applied / threshold_version: NEW — links back to
    ReconciliationRun.threshold_version (§7).
  - policy_checks: NEW — itemized pass/fail per check (conservation,
    currency, uniqueness, margin) (§7).
  - reviewed_by / reviewed_at / review_action: NEW, set only once a
    human reviewer acts on a PENDING_REVIEW group (§6).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict

from recon_agent.models.enums import (
    Cardinality,
    CommitPolicy,
    MatchGroupStatus,
    ProposedBy,
    ReviewAction,
    VerificationResult,
    VerifiedBy,
)


class MatchGroup(BaseModel):
    model_config = ConfigDict(use_enum_values=False)

    group_id: str
    cardinality: Cardinality
    expected_amount_paise: int
    matched_amount_paise: int
    residual_amount_paise: int
    status: MatchGroupStatus
    evidence_score: Optional[float] = None
    runner_up_score: Optional[float] = None
    score_margin: Optional[float] = None
    threshold_applied: float
    threshold_version: str
    policy_checks: dict[str, Any]
    proposed_by: ProposedBy
    verified_by: VerifiedBy
    commit_policy: CommitPolicy
    verification_result: VerificationResult
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    review_action: Optional[ReviewAction] = None
