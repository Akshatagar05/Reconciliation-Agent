"""evaluation — the judge-facing metrics harness (ARCHITECTURE.md §11).

Stage 12 of this 16-stage relay build. This is the ONE place in the
codebase allowed to read ground truth (``ground_truth/<dataset>/ground_truth.json``)
— that invariant, stated in every earlier stage's own docstrings, is
unchanged here: the matching/verification/LLM pipeline itself
(``matching/``, ``verification/``, ``llm/``) is never touched by this
package and never imports ground truth. ``harness.py`` runs the real
pipeline (``matching.pipeline.run_pipeline``) against a named dataset's
records and then, and only then, cross-checks the result against that
dataset's held-out ground truth to compute the seven §11 judge-facing
metrics.
"""

from __future__ import annotations

# Deliberately no eager `from recon_agent.evaluation.harness import ...`
# here: this package's CLI is invoked as `python -m
# recon_agent.evaluation.harness`, and importing harness.py both via
# this package's own __init__ and via -m's module execution triggers a
# (harmless but noisy) "found in sys.modules ... prior to execution"
# RuntimeWarning. Import directly from `recon_agent.evaluation.harness`
# instead — e.g. `from recon_agent.evaluation.harness import
# run_evaluation, EvaluationReport`.
