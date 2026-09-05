"""verification — the Financial and Evidence Verifier (ARCHITECTURE.md §2).

Stage 5 of this build added ``verifier.py``: the core policy checks
(margin, conservation, currency, identifier uniqueness, and the Stage 6
always-human-review rule) as a standalone, pure function. Stage 6 of
the build wired it into ``matching.pipeline.run_pipeline`` as a Stage 7
verification pass over every Stage 1-4 proposal — see
``matching.pipeline.run_verification``. ``verifier.py``
itself is still untouched by that wiring: it remains a pure function
that does not mutate the ``MatchGroup``/``MatchGroupMember`` it's
given, and still raises ``NotImplementedError`` for ``proposed_by``
values this build doesn't implement a policy branch for yet
(STAGE5_FUZZY).
"""

from __future__ import annotations

from recon_agent.verification.verifier import VerificationOutcome, verify_match_group

__all__ = [
    "VerificationOutcome",
    "verify_match_group",
]
