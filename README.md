# Reconciliation Agent

An exception-aware multi-source settlement reconciliation agent that matches bank credits, gateway settlements and internal ledger entries; verifies money conservation; and produces a measurable, auditable close report.

Built for Track 04: AI Finance Controller — Razorpay AI Buildathon 2026. See `ARCHITECTURE.md` for the full system design.


## Quickstart

The fastest path from a fresh clone to a working demo — Docker is the
only prerequisite:

```
git clone <REPLACE_WITH_YOUR_GITHUB_URL — see the callout at the top of this README>
cd recon-agent
docker compose up
```

That one command builds the image (both calibration/evaluation
datasets already baked in — nothing generated on first request) and
starts both processes, networked together. Once it's up:

| What | Where |
|---|---|
| API + interactive docs | http://localhost:8000/docs |
| Judge dashboard | http://localhost:8501 |
| API health check | http://localhost:8000/health |

**Optional — enable the LLM tier.** Copy `.env.example` to `.env` and
fill in a real `GROQ_API_KEY` before running `docker compose up` to
exercise Stage 6's recommendation tier. Leaving it unset is also a
legitimate demo path — the system is designed to degrade gracefully
and this is exactly what §15's demo script exercises in its last step.

**Run the evaluation harness** (against the same datasets baked into
the image, in a second terminal once `docker compose up` is running):

```
docker compose exec api python -m recon_agent.evaluation.harness --dataset evaluation
```

**Run the multi-seed stress evaluation** (20 fresh-seeded datasets,
generated on demand — see "Multi-Seed Stress Evaluation" below):

```
docker compose exec api python -m scripts.multi_seed_evaluation
```

**Run the test suite** (277 tests — 276 passing, 1 skipped):

```
docker compose exec api pytest
```

**Without Docker** — see "Installation" directly below for the local
`pip install -e .` path; `BUILD_LOG.md`'s "Docker Packaging (Stage 15)"
section covers the image/compose design in full, plus how to verify
the dataset pre-bake yourself with `./scripts/verify_docker_image.sh`.

## Installation

```
pip install -e .
```

`pyproject.toml` uses a setuptools src-layout build (`recon_agent` lives
under `src/`), with dependencies matching `requirements.txt` and
`requires-python = ">=3.11"`.

## Key Results

Both runs below are genuine Stage 1-5 pipeline runs against this
repo's two bundled, fixed-seed synthetic datasets. No `GROQ_API_KEY`
is configured in this environment, so Stage 6 gracefully degrades to
`LLM_UNAVAILABLE` in both (§14's designed degradation path, not a
harness shortcut) — `groq_configured: false` in both JSON exports.

**These numbers moved twice after the initial build, both documented in
BUILD_LOG.md.** First, Bug 1's fix added a bounded residual check to
Stage 1 (previously it never checked amount at all, so an
exact-identifier match auto-verified regardless of how large or
unexplained its LEDGER-vs-GATEWAY gap was) — correct and necessary,
but its first shipped default (2.5%) was picked from generator-
internal fee-tier arithmetic, not measured from data, and turned out
too tight: it flagged a real slice of genuinely correct matches,
dropping VERIFIED counts and coverage well below what they should be.
Second, a **calibration pass** (`scripts/calibrate_stage1_residual.py`,
tuned only against `data/calibration/` — see BUILD_LOG.md's "Threshold
recalibration" entry) re-derived the tolerance from the real residual
distribution of every genuinely correct Stage 1 match in the
calibration set and landed on **3.25%** — comfortably above the
observed maximum (2.9562%) with a real margin, not a guess. The table
below is the fully-recalibrated state: coverage recovers to
essentially the pre-Bug-1-fix numbers, while precision and false-match
rate stay unchanged (100% / 0%) throughout all of this, because the
matches being gated were never wrong — only, briefly, under- and then
over-scrutinized.

| Metric | calibration | evaluation |
|---|---|---|
| Records / logical events | 288 / 100 | 287 / 100 |
| Groups: VERIFIED / PENDING_REVIEW / REJECTED | 87 / 0 / 5 | 96 / 0 / 5 |
| 1. Auto-match precision | 100.0% (87/87) | 100.0% (96/96) |
| 2. Record coverage (raw / excl. abstention) | 85.1% / 86.9% | 86.8% / 88.6% |
| 3. Value coverage (raw / excl. abstention) | 72.2% / 72.5% | 66.5% / 66.8% |
| 4. False-match rate | 0.0% (0/87) | 0.0% (0/96) |
| 5. Exception quality | 100.0% (7/7) | 100.0% (7/7) |
| 7. Runtime | 0.67s (431 rec/s), 0 Groq calls, $0.00 | 0.87s (330 rec/s), 0 Groq calls, $0.00 |
| **8. Bank credit coverage** | **52.4% (11/21)** | **50.0% (10/20)** |
| **9. Complete cluster resolution** | **50.0% (14/28)** | **44.4% (12/27)** |

For reference, the interim (2.5%, guessed-default) state that sat
between these two points: 60/27/5 VERIFIED/PENDING/REJECTED and
63.2%/56.7% record/value coverage on calibration; 64/32/5 and
62.4%/44.8% on evaluation — precision and false-match rate were
100%/0% there too. Metrics 8-9 are untouched by any of this, at any of
the three points — they depend on Stage 3/4 aggregation, not Stage 1's
residual check.

**What "100% precision, 0% false-match rate" does and doesn't mean.**
Metrics 1-5 measure whether *claimed* relationships are correct —
every VERIFIED group Stage 1-5 has committed so far is a genuine
subset of some real ground-truth settlement event. **They do not
measure whether the full bank-gateway-ledger settlement loop for that
event actually closed.** Metrics 8-9 measure closure directly: on the
evaluation set, only 50.0% of real bank credits ended up in some
VERIFIED group, and only 44.4% of ground-truth settlement events were
closed *exactly* (rather than as a correct but partial subset) — this
is entirely multi-leg consolidated settlements where an intentional
"decline rather than guess" policy leaves genuinely ambiguous legs
unmatched instead of risking a wrong guess (see "Notable Engineering
Findings" below); it is untouched by Stage 1's residual threshold at
any of its three historical values, since these two metrics are driven
by Stage 3/4 aggregation. Metrics 8-9 are always reported alongside 1-5, never in place
of them — see `ARCHITECTURE.md` §11 for full definitions. Bank credit
coverage's denominator also changed independently of Bug 1: it now
counts only genuine `entity_type=BANK_CREDIT` records rather than any
BANK-source record (Bug 4 in BUILD_LOG.md), which is a metric-honesty
fix, not a behavior change in the pipeline itself.

A few more honest reads on the numbers above:

- Precision and false-match rate being perfect (100% / 0%) is a real,
  earned result at this dataset size and threshold setting, but it
  isn't yet a stress test of the harder tiers — Stage 2/4/5's own
  proposals never reached VERIFIED in either run, so their precision
  is undefined, not zero.
- Record coverage (~85-87%) and value coverage (~67-72%) recovered
  back to essentially where they were before Bug 1's residual check
  existed at all — see the callout above. The dip to ~62-63% / ~45-57%
  in between was real, but was an artifact of the check's first
  (guessed, uncalibrated) default being too tight, not of the check
  itself being wrong; the check itself is still there, still real, and
  still fires on a genuinely oversized residual — it's just bounded by
  a number backed by the actual data now.
- Value coverage trails record coverage: larger transactions are
  proportionally less likely to auto-resolve, which is useful for
  prioritizing where remaining human review effort has the most
  monetary impact.
- 0 Groq calls / $0.00 cost is a property of this environment having
  no `GROQ_API_KEY` configured, not a harness limitation — the
  cost-estimation code path is still exercised and unit-tested.

Full metric-by-metric rationale, tier-contribution breakdowns, and how
these numbers evolved stage to stage (including two real bugs found
along the way): `ARCHITECTURE.md` §11 and `BUILD_LOG.md`.

## Multi-Seed Stress Evaluation — Reproducible From This Repo

The two numbers above come from one calibration dataset and one
evaluation dataset, each from a single fixed seed. An external review
independently generated 20 additional fresh-seeded datasets and
reported strong, consistent precision across all of them — but that
analysis lived outside this repo and couldn't be reproduced by anyone
reading it. This repo can now generate and prove that kind of
multi-seed validation for itself, on demand, with one command:

```
python -m scripts.multi_seed_evaluation
```

This generates 20 fresh, independently-seeded stress-test datasets
(same §11 anomaly-mix generation logic as calibration/evaluation, just
different RNG seeds — `testdata/generator.py --seed stress`), runs the
full real pipeline against each with the already-calibrated thresholds
(nothing is retuned against these — validation-only, same
calibration/evaluation discipline as everywhere else in this repo),
and aggregates mean/min/max for every §11 metric. The generated
datasets live only in a throwaway temp directory, never committed to
this repo.

**Real numbers from this repo, this run** (20 seeds, 100 logical
events each, no `GROQ_API_KEY` configured — every seed degraded
gracefully, 0 live Groq calls):

| Metric | mean | min | max |
|---|---|---|---|
| Auto-match precision | 99.8% | 98.8% | 100.0% |
| Record coverage (raw) | 86.4% | 83.3% | 91.0% |
| Value coverage | 71.8% | 64.4% | 84.4% |
| False-match rate | 0.2% | 0.0% | 1.2% |
| Bank credit coverage | 49.3% | 30.0% | 76.2% |
| Complete cluster resolution | 45.7% | 37.0% | 57.1% |

**Honest read:** mostly, but not perfectly, consistent with "strong,
consistent precision across all of them." 17 of 20 seeds hit exactly
100.0% precision / 0.0% false-match rate, matching calibration's and
evaluation's own clean numbers — but 3 of 20 seeds each produced
exactly one incorrect `VERIFIED` group (98.8-98.9% precision), a real
deviation that a single held-out evaluation set can't surface on its
own. Coverage and cluster-resolution metrics vary more across seeds
than precision does, which tracks expectation — Stage 3/4 aggregation
success depends on how a given seed's events happen to batch into
consolidated settlements. This finding is reported as-is, not chased:
per this addition's own scope, no matching/verification/LLM code was
touched to close that gap. Full methodology, the exact 20 seeds used,
and the reasoning behind each design choice: `BUILD_LOG.md`'s
"Multi-Seed Stress Evaluation" entry.

## Architecture

A three-source (bank / gateway / ledger) reconciliation pipeline runs
six matching stages of increasing sophistication — exact match,
constrained match, aggregate/adjustment matching, a financial-and-
evidence verifier, fuzzy candidate retrieval, and an LLM
recommendation tier that never auto-commits — each proposing
`MatchGroup`s that only the verifier can promote to `VERIFIED`.
Everything downstream (the evaluation harness, the API, the
dashboard) reads from that same verified state.

See `ARCHITECTURE.md` for the full system design: the data model,
verification policy, conflict resolution, exception taxonomy, and the
evaluation plan.

## Human Review Loop — Real Gap Closed 2 Days Before Submission

`MatchGroup.reviewed_by` / `reviewed_at` / `review_action` have
existed in the schema since early in the build (ARCHITECTURE.md §6/
§7), and the whole point of a `PENDING_REVIEW` status is that a human
is supposed to look at it and decide. But an external review found
that nothing in this codebase ever let that happen: the system could
flag a group for human attention (Stage 1's residual check, Stage
2-5's margin check, Stage 6's always-human-review-required LLM tier)
but there was no endpoint, no UI, no code path anywhere that a human
could actually use to act on one. Every `PENDING_REVIEW` group just
sat there permanently, its reviewer fields forever `null`.

Fixed with `PATCH /report/{run_id}/groups/{group_id}/review`
(`reviewer`, `review_action` — `APPROVED`/`REJECTED`/`ESCALATED` —
and an optional `note`), a matching Approve/Reject/Escalate section in
the dashboard's drill-down tab, and an end-to-end test
(`tests/test_human_review.py`) that reviews real `PENDING_REVIEW`
groups produced by actually running the pipeline (never a hand-built
`MatchGroup` fixture). No matching, verification, or LLM logic was
touched — this is purely the missing "let a human act on what the
system already flagged" step. Two judgment calls worth stating
explicitly (full reasoning in `api/service.py`'s `review_group`
docstring and `BUILD_LOG.md`):

- **A human `REJECTED` verdict does not release the group's records
  back to the unmatched pool**, unlike an *automatic* system
  rejection. An automatic rejection releases records so a *later
  stage in the same pipeline run* can retry them; a human's explicit
  "no" on a specific group, given after the pipeline has already
  finished, is a considered, final answer — there's no later stage
  left to hand it to, and silently re-offering a record a human just
  rejected to some other matching path would undermine the review
  itself.
- **The review action reuses the existing `STAGE7_VERIFICATION`
  `DecisionStage` value** rather than adding a new one. That stage is
  already defined as "the verifier's own pass/fail evaluation" of a
  group; a human review is the same kind of event — an evaluation of
  the group as a whole, not a new proposal — just performed by
  `HUMAN_REVIEWER` instead of `FINANCIAL_AND_EVIDENCE_VERIFIER`. This
  also means the review shows up in the dashboard's existing
  "every DecisionEvent naming this group" drill-down with no changes
  needed there, exactly like any other decision in the audit trail.

## Notable Engineering Findings

Nine real bugs, across three rounds, were found and fixed during the
build, plus one further calibration pass (Round 3, below — not a bug:
Bug 1's fix was correct from day one, only its guessed default value
needed correcting against real data) — kept here as evidence of how
the system was actually tested, not just described. Full narratives,
root-cause analysis, and regression tests are in `BUILD_LOG.md`.

### Round 2 — found by the repo owner's first real run, 4 days before submission

The four bugs below were found by the repo owner actually running the
system for the first time against their own hand-crafted dataset
(`data/repro_bugfix/records.json`) — not synthetic calibration/
evaluation data. That's exactly why they surfaced now and not earlier:
synthetic data is generated from the same assumptions the pipeline
itself encodes, so it's structurally blind to the gaps a real,
independently-authored dataset finds immediately. All four are fixed,
with regression tests against the exact reproduction dataset in
`tests/test_repro_bugfix_dataset.py`.

**Unbounded Stage 1 residual.** Stage 1's exact-identifier match
re-checked uniqueness, currency, and date policy on verification, but
never amount — so two records sharing a reference could auto-verify
with an arbitrarily large, completely unexplained gap between their
LEDGER and GATEWAY amounts. Fixed with a configurable bounded
tolerance (`stage1_residual_tolerance_fraction`) and a new
`RESIDUAL_EXCEEDS_TOLERANCE` outcome that routes to PENDING_REVIEW
(real evidence, not disproven) rather than silently verifying or
bluntly rejecting. The tolerance's *default value* went through two
more iterations after this — see "Threshold recalibration" below —
this fix itself (the ceiling existing at all) was correct from the
start and never changed.

### Round 3 — threshold recalibration, 3 days before submission

**Guessed default too tight, recalibrated against real data.** Round
2's Bug 1 fix shipped with `stage1_residual_tolerance_fraction`
defaulting to 2.5% — derived by hand from `testdata/generator.py`'s
MDR-tier arithmetic, not measured. It turned out too tight: record
coverage on both bundled datasets dropped from ~85-87% to ~62-63%
because a real slice of genuinely correct Stage 1 matches have
fee-driven residuals the guessed default flagged as suspicious.
`scripts/calibrate_stage1_residual.py` re-derives the tolerance
properly — run the full Stage 1-5 proposal pipeline against
`data/calibration/records.json` only, isolate every group still
tagged `STAGE1_EXACT` once Stage 3/4's in-place group extension has
happened (the same population `evaluation/harness.py`'s tier-
contribution metric counts), cross-check each one against calibration
ground truth, and report the real residual distribution. Every one of
the 80 genuinely correct calibration-set Stage 1 matches has a
residual between 0% and 2.9562% of the larger amount, clustering
tightly at the five real MDR+GST fee tiers (1.77% / 2.065% / 2.36% /
2.655% / 2.95%) — confirming BUILD_LOG's original hand-derivation was
directionally right but numerically un-anchored. New default: **3.25%**
— a real ~0.3-percentage-point margin above the observed maximum, not
a guess. See BUILD_LOG.md for the full numbers, the before/after
metrics on both datasets, and an important honest caveat: the repo's
own hand-crafted reproduction pair (`demo_ledger_payment_002`/
`demo_gateway_settlement_002`) now auto-verifies instead of routing to
PENDING_REVIEW, because its residual (~2.9502%) is numerically
indistinguishable from ordinary top-tier fee variance in real data —
no magnitude-only threshold can separate that specific case from real
data without also cutting into genuinely correct matches.

**Zero-exception unmatched records.** The catch-all safety net that's
supposed to guarantee every unmatched record gets a real exception
(§14's acceptance gate) had a gap: it treated any DecisionEvent that
merely *mentioned* a record's id — even as a passing candidate in some
other record's own event — as proof the record had already been
handled, when nothing had ever raised a real exception naming it. A
genuinely orphaned bank credit could accumulate a DecisionEvent trail
and still end up with zero exceptions and a falsely-COMPLETE run
status. Fixed by keying the catch-all's "already handled" check off
Exceptions (always genuinely about their subject) instead of
DecisionEvents (often just a mention).

**Dashboard drill-down missing evidence.** `MatchGroup` and
`DecisionEvent` have carried `runner_up_score`, `residual_amount_paise`,
and per-event `reason_code`/`explanation` since early in the build, but
the dashboard's match/exception drill-down only ever rendered status
and members. Fixed by rendering the full decision trail (every
DecisionEvent naming a given group or exception, with its stage,
reason code, explanation, and full checks) alongside the existing
score/threshold metrics — read-only rendering of data that already
existed, no new computation.

**Misleadingly-named bank credit coverage.** `bank_credit_coverage`'s
eligibility filter was "any BANK-source record", which silently pulled
in BANK-side REFUND/CHARGEBACK/REVERSAL legs — real BANK-source
records, but not credits landing in the account — alongside genuine
BANK_CREDIT records, inflating the denominator beyond what the metric
name describes. Narrowed to `entity_type=BANK_CREDIT` specifically;
this is why the Key Results table above shows a lower bank credit
coverage number than an earlier build of this README did — the metric
got more honest, the pipeline's behavior didn't change.

### Round 4 — found via live testing on a real machine, submission imminent

Unlike every other bug on this page, this one was **not** found by any
test in this repo, synthetic or hand-crafted. It surfaced from running
the actual API on a real machine and sending it real concurrent HTTP
traffic — a `POST /reconcile` colliding with a burst of the dashboard's
background `GET /health` polling. That's a genuinely different kind of
validation from everything else here, worth naming explicitly: every
test this project had wrote (sequential `TestClient` calls, one
request awaited before the next) is structurally incapable of catching
a race condition, no matter how many of them exist.

**Shared SQLite connection across concurrent requests.** `RunStore`
(`api/storage.py`) and `CallBudgetGovernor` (`llm/governor.py`) each
opened one `sqlite3.Connection` in `__init__` and held it for the life
of the process, reused across every request via
`check_same_thread=False`. That flag disables Python's same-thread
*check*; it does not make a connection safe for genuinely concurrent
use. FastAPI's sync endpoints run on Starlette's threadpool, so two
requests really could call into the same connection object from two
threads at once — and when one request's `save_run()` was mid-
transaction while another thread's own statement landed on that same
connection, the connection's transaction bookkeeping got corrupted.
Client-side this looked like a flat 30-second `ReadTimeout` (the actual
crash happened in a background thread and just hung); server-side, the
real exception was `sqlite3.OperationalError: cannot start a
transaction within a transaction`, thrown from inside `save_run()`'s
`BEGIN IMMEDIATE`. Fixed by removing the shared connection from both
classes entirely — every method now opens its own fresh, short-lived
connection, does its one atomic `BEGIN IMMEDIATE`-protected operation,
and closes it; SQLite's own file-level locking still serializes
concurrent writers correctly, so the atomicity guarantee is unchanged,
only the cross-thread connection sharing is gone. Verified with a new
`tests/test_concurrency_live.py` that starts a real `uvicorn` server on
a real socket and fires genuinely concurrent `httpx` requests at it
from a thread pool (mixed `POST /reconcile` + `GET /health` bursts, and
separately several concurrent `POST /reconcile` writers), repeated
across multiple rounds since this was a race condition; confirmed the
new tests reproduce the exact diagnosed error against the original
code before the fix, and pass cleanly, repeatedly, after it. Full
BUILD_LOG.md entry: "Concurrency Bug — Shared SQLite Connection Under
Live Traffic".

### Round 5 — Groq model deprecation, submission day (2026-09-04)

Not a bug in this repo's own logic — an external dependency went away.
Groq deprecated `meta-llama/llama-4-scout-17b-16e-instruct` (this
project's Stage 6 model since early in the build), announced June
2026. Swapped the default in `config.py`/`.env.example` to
`openai/gpt-oss-120b`, Groq's own 1:1 replacement recommendation.
gpt-oss-120b is a reasoning model and can prepend a
`<think>...</think>` block ahead of its actual JSON answer even when
asked not to (`reasoning_format="hidden"` is requested on the call,
but Groq's own community forum has documented cases of it still
leaking); `llm/recommender.py`'s response parsing now strips a leading
think-block defensively and, as a general safety net, falls back to
scanning for the first valid embedded JSON object if that's not
enough — so this isn't fragile to the exact reasoning-wrapping shape
of any one model. `reasoning_effort="low"` is also requested to keep
latency and token usage down. Full BUILD_LOG.md entry: "Groq model
deprecation — Llama 4 Scout to gpt-oss-120b".

### Round 1 — found during the initial build

**Verifier conservation-arithmetic bug.** Shortly after the verifier
was wired into the pipeline, 19-21% of proposed groups were being
wrongly REJECTED — most of them exact matches to real ground-truth
groups. The cause was a separate reimplementation of the conservation
equation that treated legitimate same-amount multi-source agreement as
"nothing to reconcile" and double-counted gross/net views of the same
settlement. Fixed by extracting the arithmetic into one shared
function both the verifier and the aggregation stages now call, with
regression tests confirming zero false rejections on both datasets.

**Consolidated-settlement fragmentation.** Large consolidated bank
credits (many ledger/gateway rows settling into one bank credit) were
often ending up unmatched even though the aggregation search had
already found the correct combination. The root cause was a
generator-injected duplicate "decoy" record creating a perfect tie in
the subset-sum search, which the margin check deterministically
rejected. Deduplicating candidate slots before the search runs
resolved some of the affected cases outright; the rest are legitimately
blocked by an intentional "decline rather than guess" policy on
ambiguous overlaps — which is the main reason metrics 8-9 above aren't
100%.

**Stage 6 seeker-ordering bug.** An alphabetical (not source-aware)
ordering of Stage 6's LLM seekers could let a GATEWAY record's own
turn run before the LEDGER seeker that was about to correctly claim
it, producing a stale, misleading "no candidates found" exception
moments before the record was actually matched. Fixed by running all
LEDGER seekers first and excluding any record they've already offered
as a candidate from getting an independent GATEWAY turn — an ordering
fix, not a retract-after-the-fact patch.

**Silent unattempted-record drop.** The first real run of the
evaluation harness found exception quality at only 14.3% (1/7),
because `ADJUSTMENT`-typed honest-abstention records had no entry in
the pipeline's source→target entity-type mapping — they were silently
skipped by every stage, producing zero decision events and zero
exceptions instead of a real category. Fixed by adding the missing
mapping entries; exception quality has been 100% (7/7) on both
datasets since (77.8% as of Round 2's Bug 1 fix — see BUILD_LOG.md;
the drop is one legitimately-resolved unresolved-case now excluded
from the denominator, not a new failure).

---

For the full stage-by-stage build history — all 16 stages, in order,
with the reasoning behind every design decision — see `BUILD_LOG.md`.
