"""ReconciliationException — ARCHITECTURE.md §7 and §8.

ARCHITECTURE.md §7 names this entity "Exception" as the conceptual term,
but §7 now explicitly notes it must be *implemented in code* as
`ReconciliationException`, not `Exception` — naming a Pydantic model
`Exception` shadows Python's builtin `Exception` class, forcing an
aliased import (`from ... import Exception as ReconException`) everywhere
the model is used. Renamed once, here, rather than carrying that
friction through every later stage. group_id_or_record_id is a single
field per §7: a ReconciliationException can be attached to either a
group or a bare record (e.g. an orphan bank credit that never entered a
group at all).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from recon_agent.models.enums import (
    ExceptionCategory,
    ExceptionReviewStatus,
    ExceptionSeverity,
)


class ReconciliationException(BaseModel):
    model_config = ConfigDict(use_enum_values=False)

    group_id_or_record_id: str
    category: ExceptionCategory
    severity: ExceptionSeverity
    evidence: dict[str, Any]
    recommended_action: str
    review_status: ExceptionReviewStatus
