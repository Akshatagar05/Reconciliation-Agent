"""recon_agent — exception-aware multi-source settlement reconciliation
agent (Razorpay AI Buildathon, Track 04).

Stages 1-3 of 10 (this build): project scaffolding + data models (Stage
1), the synthetic ground-truth data generator (Stage 2, `testdata/`),
and normalization + Stage 1 (exact) / Stage 2 (constrained) candidate
matching (Stage 3, `normalization/` + `matching/`). No aggregation,
adjustment, fuzzy matching, LLM escalation, or verification logic is
implemented yet — see README.md's Build Status section.
"""

from __future__ import annotations

__version__ = "0.1.0"
