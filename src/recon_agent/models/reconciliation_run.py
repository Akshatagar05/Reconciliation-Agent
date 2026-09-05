"""ReconciliationRun — ARCHITECTURE.md §7.

One row per end-to-end reconciliation run. rules_version, model_version,
threshold_version, and policy_version are deliberately four separate
fields (not one combined "version") because each axis is versioned and
tuned independently — see §2 and §7's inline note on threshold_version /
policy_version.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from recon_agent.models.enums import ReconciliationRunStatus


class ReconciliationRun(BaseModel):
    model_config = ConfigDict(use_enum_values=False)

    run_id: str
    input_hash: str
    rules_version: str
    model_version: str
    threshold_version: str
    policy_version: str
    started_at: datetime
    status: ReconciliationRunStatus
