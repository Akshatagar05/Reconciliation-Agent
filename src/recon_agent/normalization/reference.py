"""Reference-field normalization — ARCHITECTURE.md §6 (Stage 1's "exact
normalized identifier match" condition, §2).

The Stage 2 generator (``testdata/generator.py``) already emits
schema-correct ``NormalizedRecord`` instances — there is no raw
per-source column layout left to map here. "Normalization" in this
system means one thing only: cleaning up the ``reference`` string for
*comparison purposes* so that superficial, non-semantic differences
(case, incidental whitespace, separator punctuation) don't defeat an
otherwise-exact identifier match. This is exactly what the generator's
"dirty_reference" anomaly category exercises — whitespace/casing/
separator dirtying should still resolve to an exact match after
normalization; truncation and single-character typos deliberately do
not (those need Stage 2's constrained scoring, or Stage 5's fuzzy
retrieval later — never Stage 1's identifier-only path).

This module never touches a stored ``NormalizedRecord`` — callers pass
in a bare string and get a bare string back. Comparing two records'
references is always done by normalizing both on the fly at comparison
time, never by mutating or re-storing the record.
"""

from __future__ import annotations

import re

# Everything that isn't alphanumeric is treated as incidental formatting
# (whitespace, hyphens, underscores, slashes, dots, ...) and stripped for
# comparison purposes only.
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def normalize_reference(reference: str) -> str:
    """Case-fold and strip whitespace/separator punctuation from a
    reference string, for comparison only.

    Recovers exact-match equality across the generator's "whitespace",
    "casing", and "separator" dirty-reference variants (e.g. ``"
    PAY_6QH-09U7670LNDB "`` normalizes the same as ``"pay_6qh09u7670lndb"``
    once separators are stripped too). Deliberately does **not** recover
    "truncation" or "typo" variants — those still differ after
    normalization, by design, so they fall through to Stage 2/5 rather
    than being silently treated as identifier matches.
    """
    folded = reference.strip().lower()
    return _NON_ALNUM_RE.sub("", folded)
