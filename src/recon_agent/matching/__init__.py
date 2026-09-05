"""matching — Stage 1 (exact) through Stage 4 (adjustment) candidate
proposal, per ARCHITECTURE.md §6. All four stages only ever PROPOSE
candidate MatchGroups (status=PENDING_REVIEW at proposal time) —
nothing in stage1_exact.py/stage2_constrained.py/stage3_aggregate.py/
stage4_adjustment.py itself marks a group VERIFIED; that's the
Financial and Evidence Verifier's job (verification/verifier.py).
``pipeline.run_pipeline`` is the full entry point: it runs Stage 1-4
and then wires every proposed group through that verifier as a single
Stage 7 verification pass, so callers of ``run_pipeline`` see final
VERIFIED/PENDING_REVIEW/REJECTED status, not the raw proposal. Use
``pipeline.run_stage1_and_stage2`` for just the first two stages
un-verified, or ``stage1_exact``/``stage2_constrained``/
``stage3_aggregate``/``stage4_adjustment`` to run one stage at a time.
"""

from __future__ import annotations

from recon_agent.matching.pipeline import (
    MatchingResult,
    run_pipeline,
    run_stage1_and_stage2,
    run_verification,
)

__all__ = ["MatchingResult", "run_pipeline", "run_stage1_and_stage2", "run_verification"]

