"""normalization — value-level cleanup for comparison, not raw-format
mapping. See ``reference.py`` for the full explanation; this stage's
data already arrives as schema-correct ``NormalizedRecord`` instances,
so the only "normalization" left to do is reference-string cleanup for
matching purposes.
"""

from __future__ import annotations

from recon_agent.normalization.reference import normalize_reference

__all__ = ["normalize_reference"]

