"""Synthetic ground-truth data generation for the reconciliation agent.

ARCHITECTURE.md §11 ("Evaluation Plan"). This package produces two
independently seeded synthetic datasets — CALIBRATION and EVALUATION —
of bank / gateway / ledger ``NormalizedRecord`` rows, plus a hidden
ground-truth mapping of which records truly belong together. The
matching pipeline (later stages) must only ever read the
``NormalizedRecord`` batches under ``data/``; the ground truth under
``ground_truth/`` is for the evaluation harness only (§11, Day 10).

See ``generator.py`` for the generation logic and CLI entry point.
"""

from __future__ import annotations
