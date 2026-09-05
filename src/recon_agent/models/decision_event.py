"""DecisionEvent — ARCHITECTURE.md §7.

One audit-trail row per stage decision on a group, including the
STAGE7_VERIFICATION stage (the verifier's own pass/fail evaluation),
which is why DecisionStage carries one more value than
MatchGroup.proposed_by.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from recon_agent.models.enums import DecisionStage


class DecisionEvent(BaseModel):
    model_config = ConfigDict(use_enum_values=False)

    event_id: str
    group_id: str
    stage: DecisionStage
    candidate_scores: dict[str, Any]
    reason_code: str
    explanation: str
    timestamp: datetime
