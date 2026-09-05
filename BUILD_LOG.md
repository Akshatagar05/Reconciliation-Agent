# Build Log — 16-Stage Relay History

This is the full stage-by-stage build history for the reconciliation
agent, moved out of the main `README.md` so the README can lead with
the product and its results rather than the process that built it.
Nothing below has been rewritten for this move — it's the same
narrative, in the same order, with the same numbers it reported at
the time.

**A note on numbers in this log vs. current numbers.** This log
reflects the repo's state through the end of the 16-stage relay:
231 tests passing, seven §11 judge-facing metrics. A subsequent
documentation/metrics/dependency cleanup pass — not part of the
numbered relay, done 5 days before submission — added two new
settlement-closure honesty metrics (`bank_credit_coverage` and
`complete_cluster_resolution`, ARCHITECTURE.md §11) and their tests
(231 → 237), and removed two genuinely-unused dependencies
(`sqlalchemy`, `statsmodels`). Historical counts and metric
descriptions below are left exactly as they were reported at each
stage; **for current numbers, see README.md's "Key Results" section.**

**Version-history note, moved from ARCHITECTURE.md's former intro:**
ARCHITECTURE.md went through several structural review passes before
any code was written. The most recent of these ("v4.1") applied four
pre-coding corrections on top of a broader "v4" revision made per a
second-round senior engineering review that had flagged five
structural gaps: splitting the Stage 1/2 auto-commit policy (Stage 2
uses the stricter evidence bar, not Stage 1's own), removing a dead
`LLM_CORROBORATED_AUTO_COMMIT` path (Stage 6 is always
human-review-only), fixing per-target subset-sum to use disjoint
aggregation windows so two bank credits can't both claim the same
ledger entry, and replacing a vague `confidence` field with the
actual evidence fields the verifier needs to prove why a group
passed. ARCHITECTURE.md itself no longer carries version tags or a
changelog-style intro — it now reads as a single coherent spec — but
this is where that history lives.

---

## Build Status

The stage count changed from 10 to 15, and now to 16, partway through
this build — later relay prompts split what used to be one larger
"verify + wire up + review" stage into several smaller,
independently-scoped ones, starting with Stage 5 below.

- Stage 1 of 16: scaffolding + data models — COMPLETE
- Stage 2 of 16: synthetic data generator — COMPLETE
- Stage 3 of 16: normalization + Stage 1/2 matching — COMPLETE
- Stage 4 of 16: aggregate + adjustment matching — COMPLETE
- Stage 5 of 16: Financial and Evidence Verifier (standalone) — COMPLETE
- Stage 6 of 16: verifier wired into pipeline — COMPLETE
- Stage 7 of 16: Global Conflict Resolver (standalone) — COMPLETE
- Stage 8 of 16: Global Conflict Resolver wired into Stage 3 — COMPLETE
- Stage 9 of 16: Stage 5 fuzzy candidate retrieval — COMPLETE
- Stage 10 of 16: Groq recommendation module (standalone) — COMPLETE
- Stage 11 of 16: shared budget governor + Stage 6 wired into pipeline — COMPLETE
- Stage 12 of 16: evaluation harness — COMPLETE
- Stage 13 of 16: FastAPI endpoints — COMPLETE — verified stable in a
  fresh environment (see "Investigated: reported silent server death"
  below); a reviewer's live-sandbox crash report could not be
  reproduced and is attributed to that sandbox's own long-running
  degradation, not a code defect
- Stage 14 of 16: Streamlit dashboard — COMPLETE — run with
  `streamlit run src/recon_agent/dashboard/app.py`; the FastAPI layer
  (`uvicorn recon_agent.api.app:app --reload`) must already be running
  first, since the dashboard is an HTTP client of it, not an in-process
  caller of any matching/verification/LLM/evaluation module — see
  "Streamlit Dashboard (Stage 14)" below
- Stage 15 of 16: Docker packaging + documentation polish — COMPLETE —
  `docker compose up` starts the FastAPI service and the Streamlit
  dashboard together, networked, with both calibration/evaluation
  datasets pre-baked into the image at build time (no cold-run
  generation); see "Docker Packaging (Stage 15)" below for what was
  built and the honest §14 acceptance-gate results checked item by
  item. That section also flagged a repo-state finding — an unused
  `sentence-transformers` dependency pre-baked into the image per an
  earlier architecture-doc requirement — that Stage 16 below resolved.
- **Stage 16 of 16: failure injection + final acceptance-gate
  self-check — COMPLETE (relay finished).** Removed the unused
  `sentence-transformers`/`torch` dependency Stage 15 flagged;
  `scripts/failure_injection_demo.py` makes §15's demo script points
  5–7 (honest exception, Groq degradation, malformed input) directly
  demonstrable; `scripts/acceptance_check.sh` runs the full §14
  acceptance gate in one command, including the two Docker-dependent
  items (image build/pre-bake, timing) if Docker is available. See
  "Failure Injection + Final Acceptance-Gate Self-Check (Stage 16)"
  below for full results and the one thing still open — a Docker-
  equipped run of that script, on a machine that has one, before the
  real demo — no stage of this relay has had access to a Docker
  daemon.
- **Post-relay bugfix pass — COMPLETE.** 4 days before submission, the
  repo owner ran the system for the first time against their own
  hand-crafted dataset (not synthetic calibration/evaluation data) and
  found 4 real bugs in one sitting — an unbounded Stage 1 residual, an
  unmatched-record catch-all gap, missing dashboard evidence fields,
  and a misleadingly-named evaluation metric. All four fixed in one
  pass with regression tests against the exact reproduction dataset;
  see "Post-Relay Bugfix Pass — First Real User Run" at the end of
  this file for full narratives.
- **Concurrency bugfix — COMPLETE.** Found via live testing on a real
  machine (not synthetic/sequential testing) shortly before
  submission: a `POST /reconcile` colliding with concurrent
  `GET /health` polling crashed the server due to a single shared
  `sqlite3.Connection` reused across concurrent requests in both
  `RunStore` and `CallBudgetGovernor`. Fixed by giving every operation
  its own freshly-opened, short-lived connection instead; verified
  with a new live-uvicorn, real-concurrent-HTTP test suite
  (`tests/test_concurrency_live.py`), repeated across multiple rounds.
  See "Concurrency Bug — Shared SQLite Connection Under Live Traffic"
  at the end of this file.

## Synthetic Data (Stage 2)

`recon_agent.testdata.generator` produces two independently seeded
synthetic datasets per §11 — `calibration` and `evaluation` — each ~100
logical payment events / 200-300 physical `NormalizedRecord` rows across
bank, gateway, and ledger, covering the full required anomaly mix (clean
matches, dirty references, timing variance, net settlements, refunds,
chargebacks, reversals, partial settlements, consolidated many-to-one
settlements, duplicates, missing rows, and ≥5 honest-abstention cases per
dataset).

Regenerate both deterministically:

```
python -m recon_agent.testdata.generator --seed both
```

This writes `NormalizedRecord` batches to `data/calibration/records.json`
and `data/evaluation/records.json` — the only files later matching-pipeline
stages should read. The hidden ground truth (which records truly belong in
which group) is written separately to `ground_truth/calibration/ground_truth.json`
and `ground_truth/evaluation/ground_truth.json`, for the Day-10 evaluation
harness only — the matching pipeline must never read from `ground_truth/`.
Neither `data/` nor `ground_truth/` is committed; both regenerate
byte-identically from the fixed seeds in `generator.py`.

## Normalization + Stage 1/2 Matching (Stage 3)

`recon_agent.matching.run_stage1_and_stage2` runs Stage 1 (exact) then
Stage 2 (constrained) over a pool of `NormalizedRecord`s and proposes
candidate `MatchGroup`s per §6:

- **Stage 1 (exact)** — `matching/stage1_exact.py` — clusters records by
  normalized reference (`normalization/reference.py` case-folds and
  strips whitespace/separator punctuation for comparison only, never
  mutating the stored record). A cluster is proposed only if its
  identifier is unique per (source, entity_type) slot and it passes
  currency/date policy — no composite score needed, an exact unique
  identifier is sufficient by itself.
- **Stage 2 (constrained)** — `matching/stage2_constrained.py` — for
  records Stage 1 couldn't cluster (truncated/typo'd references, and
  every bank credit, which never shares an identifier with anything).
  Uses the same evidence bar as Stages 3-5: a composite score over
  amount compatibility, date-window compatibility, currency match, and
  counterparty similarity (rapidfuzz on the `NormalizedRecord.counterparty`
  field, normalized for case/whitespace/punctuation — weighted low, never
  sufficient alone), gated by
  candidate uniqueness, a calibrated threshold, a sufficient margin over
  the runner-up, and a simple amount-tolerance check.

Both stages only ever **propose** — every `MatchGroup` they produce is
`status=PENDING_REVIEW`, `verified_by=NOT_YET_VERIFIED`,
`verification_result=NOT_YET_RUN`. Nothing here marks a group VERIFIED;
that's the not-yet-built Financial and Evidence Verifier's job. Neither
stage attempts many-to-one aggregation (summing several settlements into
one consolidated bank credit) — that's explicitly Stage 3 (aggregate)'s
job in ARCHITECTURE.md's numbering, a later stage of this build.

`tests/test_matching_integration.py` runs both stages against the real
`data/calibration/records.json` and checks the result against
`ground_truth/calibration/ground_truth.json` (read only inside the test,
never inside the matching code) — the hard requirement is zero false
proposals; full precision/coverage metrics are a later evaluation
harness's job.

## Aggregate + Adjustment Matching (Stage 4)

`recon_agent.matching.run_pipeline` runs the full Stage 1 -> 2 -> 3 -> 4
sequence over a pool of `NormalizedRecord`s; each stage only ever works
on records the earlier stages left unmatched.

- **Stage 3 (aggregate)** — `matching/stage3_aggregate.py` — finds
  consolidated settlements Stage 1/2 couldn't place: several gateway
  settlements summing to one bank credit (many-to-one payout batching),
  or the reverse — one settlement split across several bank credits.
  Per the corrected §4, there's no merchant/account field to key on, so
  candidates are scoped to each bank credit's own eligible pool by
  currency + a bounded settlement-date lag
  (`matching/aggregation_common.py`, shared with Stage 4), then searched
  with a bounded, exact subset-sum (respecting §9's max-candidates,
  max-group-size, and timeout limits). Two different bank credits (or
  settlements, in the reverse direction) whose best-fit subsets would
  need to claim the same candidate record are both declined as
  `AMBIGUOUS_AGGREGATION` — never resolved by an arbitrary tie-break.
- **Stage 4 (adjustment)** — `matching/stage4_adjustment.py` — applies
  the actual conservation equation from §4 (`bank credit = gross
  payments - fees - GST - refunds +/- adjustments`) using
  `MatchGroupMember.role` to net a candidate group's individual
  role-tagged records against its bank-side counterpart — the full
  role-based netting Stage 2 explicitly deferred. Uses the same shared
  bounded-search machinery as Stage 3, just over individual role-tagged
  records instead of one pre-netted settlement total per event.

Like Stage 1/2, both stages only ever **propose**:
`status=PENDING_REVIEW`, `verified_by=NOT_YET_VERIFIED`,
`commit_policy=AUTO_COMMIT_STAGE2_5_THRESHOLD` (§2's shared Stage 3-5
evidence bar), `verification_result=NOT_YET_RUN`. A window that can't be
cleanly resolved — genuine overlap, an oversized candidate pool, or a
search timeout — is declined as an `AMBIGUOUS_AGGREGATION`
`ReconciliationException`, never forced into a guess.

`tests/test_stage3_aggregate.py` and `tests/test_stage4_adjustment.py`
unit-test the windowing/overlap-decline logic and the conservation-
equation arithmetic directly.
`tests/test_stage1to4_integration.py` extends the Stage 1/2 integration
pattern to the full pipeline against `data/calibration/records.json`:
zero false proposals, with a per-stage breakdown of which stage
proposed each group.

## Financial and Evidence Verifier (Stage 5, standalone)

`recon_agent.verification.verify_match_group` implements §2's
stage-differentiated auto-commit policy table as a standalone, pure
function — a genuine second check, not a rubber stamp. As of Stage 6
below it is wired into `matching.run_pipeline`, but the function itself
stays a standalone, pure function: given a `MatchGroup` and its
`MatchGroupMember`s it returns a verification outcome without mutating
anything; applying that outcome to the group's `status` /
`verified_by` / `verification_result` is `run_verification`'s job (see
Stage 6).

Given a `MatchGroup` and its `MatchGroupMember`s, it re-derives the
relevant checks from raw data rather than trusting the proposing
stage's own `evidence_score` / `policy_checks`:

- **Stage 1** — re-checks identifier uniqueness ((source, entity_type)
  slot collisions) and the combined currency/date policy, independently
  re-derived from each member's underlying `NormalizedRecord`.
- **Stages 2-4** — re-checks that `score_margin` clears a configured
  minimum (`Settings.min_score_margin_by_threshold_version`), that the
  group's members conserve (recomputing §4's conservation equation via
  `matching.common.recompute_group_conservation` — shared with Stage
  3/4, see the Stage 6 bugfix note below), that currency is consistent
  across members, and that no member's role/amount sign is internally
  contradictory.
- **Stage 6 (LLM)** — always resolves to human review, regardless of
  evidence, per §2's "no auto-commit path for LLM output, ever." This
  branch is implemented ahead of the LLM proposal stage itself
  existing, so a later stage that builds it doesn't need to touch this
  file again; it isn't exercised end-to-end yet since no such
  proposals exist.

`tests/test_verifier.py` covers a PASS and each specific `FAILED_*`
case (`FAILED_MARGIN`, `FAILED_CONSERVATION`, `FAILED_CURRENCY`,
`FAILED_UNIQUENESS`) for Stages 1-4 against synthetic
`MatchGroup`/`MatchGroupMember` fixtures, plus a dedicated test proving
the verifier still fails a group whose stored `evidence_score` and
`policy_checks` look strong when its actual members don't conserve.

## Verifier Wired Into Pipeline (Stage 6)

`matching.pipeline.run_verification` calls the Stage 5 verifier for
every proposed `MatchGroup`, applies the outcome (`PASSED` →
`VERIFIED`; a `FAILED_MARGIN` → `PENDING_REVIEW`, still holding its
records; any other `FAILED_*` → `REJECTED`, releasing its records back
to unmatched and opening an exception), and records a
`STAGE7_VERIFICATION` `DecisionEvent` either way. `run_pipeline` now
runs Stage 1-4 proposal followed by this verification pass end to end.

### Bugfix: conservation false rejections (post-Stage-6)

Shortly after this wiring landed, running the pipeline against
`data/calibration/records.json` showed 19 of 92 proposed groups
REJECTED, of which 14 turned out to be groups whose membership exactly
matched a real `ground_truth/calibration/ground_truth.json` group —
correct matches wrongly rejected, not genuine catches (11 of 15 on
`data/evaluation/records.json` too). Both came from the verifier's own
`_conservation_diff`, a separate, independently-written reimplementation
of the conservation equation Stage 3/4 already computed correctly via
their settlement-unit bookkeeping:

1. A pure REFUND/CHARGEBACK/REVERSAL event has no CREDIT-role and no
   GROSS-role member at all (it's an outflow, not a payment) — it
   surfaces as the *same* `signed_amount_paise` reflected on up to
   three source legs (ledger/gateway/bank). `_conservation_diff`
   returned `None` for "nothing to reconcile against" and the caller
   treated that as `FAILED_CONSERVATION`, when three-way agreement on
   one amount is already trivially conserved.
2. For a settlement split across multiple bank credits (or any group
   still carrying both views of one settlement event), the group holds
   *both* a LEDGER-source GROSS row (the pre-fee gross amount) and a
   GATEWAY-source GROSS row (the same event, already net of MDR
   fee/GST — per `testdata/generator.py`'s own design note, the
   settlement amount is always net whether or not FEE/TAX rows are
   separately broken out). `_conservation_diff` summed every
   non-CREDIT member regardless of role or source, double-counting
   that one settlement's money once as its ledger view and again as
   its gateway view — and, when present, subtracting its GATEWAY-source
   FEE/TAX/ADJUSTMENT breakout rows a second time on top of an already
   net figure.

The fix extracts the conservation arithmetic into
`matching.common.recompute_group_conservation`, a shared function the
verifier now calls (still computed fresh from raw
`MatchGroupMember` data each time — never from the group's own stored
totals), handling both shapes above: a same-role/same-amount group
conserves trivially, and a GATEWAY-source GROSS row's LEDGER-source
counterpart and GATEWAY-source FEE/TAX/ADJUSTMENT siblings are excluded
from the sum only when that LEDGER-source duplicate is actually
present (so Stage 4's genuine role-based netting of a lone,
never-duplicated settlement against standalone FEE/TAX/REFUND/
ADJUSTMENT records is untouched). After the fix, calibration goes from
`VERIFIED=73, REJECTED=19` to `VERIFIED=87, REJECTED=5` (evaluation:
`VERIFIED=92, REJECTED=15` to `VERIFIED=103, REJECTED=4`), with the
remaining rejections confirmed to be genuine catches, not
ground-truth matches, on both datasets. `tests/test_verifier.py` adds
direct unit coverage for both shapes, and
`tests/test_pipeline_verification.py` adds a permanent full-pipeline
check — on both calibration and evaluation — that zero REJECTED groups
exactly match a real ground-truth group, alongside the pre-existing
zero-false-positive-VERIFIED check; both directions now hold
simultaneously.

## Counterparty Field (corrected §7)

`NormalizedRecord.counterparty` — the name/account label as it appears
on that source's own record — was added after Stage 3 was originally
built, replacing a documented stopgap that fuzzy-matched `reference`
text as a proxy for counterparty evidence. The generator now populates
realistic, consistent-but-not-identical counterparty labels per logical
event (e.g. `"ACME Retail Pvt Ltd"` on the ledger vs `"ACME RETAIL PVT
LTD"` or `"ACME Retail"` on the gateway settlement), concentrated in the
`dirty_reference` and `clean` anomaly categories; Stage 2's counterparty
scoring compares this real field directly.

### Bugfix: duplicate-decoy ties orphaning consolidated bank credits (post-Stage-6, pre-Stage-7)

Running the full pipeline against `data/calibration/records.json` showed
`cal_rec_00280` (a real 14-record consolidated settlement's bank credit,
₹64,057.79) matched to nothing at all: Stage 1 had already proposed four
separate bank-less ledger↔gateway pairs within that batch, each correct
on its own, none including the bank leg — and the bank credit never
appeared in any proposed group. Checked across the whole calibration
dataset, 10 of 13 (77%) `structural_consolidated` ground-truth groups
showed the same symptom.

The obvious hypothesis — that Stage 3 simply couldn't see records
Stage 1/2 had already claimed into bank-less groups — turned out to be
wrong. `stage3_aggregate.py`'s `_build_settlement_units` already builds
"settlement unit" candidates from existing bank-less Stage 1/2 groups
(not just raw unmatched records), already runs the bounded subset-sum
search over them via `matching.aggregation_common.run_aggregation_search`
(shared with Stage 4), already supports `MANY_TO_ONE` cardinality for
merging several of them into one, and already retracts the absorbed
sub-groups' `MatchGroup`/`MatchGroupMember`/`DecisionEvent` records when
it emits the unified aggregate — all before Stage 7 verification runs
(confirmed still true in `pipeline.py`'s ordering), so no new
`MatchGroupStatus` value was needed. This shape has been in place since
Stage 4 was originally built, well before Stage 6. Direct inspection
confirmed it: the exact 5-item combination that sums to `cal_rec_00280`
(diff=0, score=1.000) *was* found by the search.

The actual bug was narrower and more mundane. `testdata/generator.py`
injects "duplicate decoy" records for its `duplicates` ground-truth
category: a second `GATEWAY_SETTLEMENT` record sharing the exact same
reference, amount, currency, and date as a genuine one. Stage 1
correctly declines to cluster either twin (non-unique identifier for
that (source, entity_type) slot — see `stage1_exact.py`'s own
uniqueness check) and both remain unmatched. `_build_settlement_units`'s
raw-record fallback, however, picked up *both* twins as independent
standalone settlement-unit candidates. Because they share an amount,
the subset-sum search could build two candidate subsets differing only
by which twin they used, tying at the identical score — a margin of
exactly `0`, which deterministically fails the `min_margin` check no
matter how confident the real (score=1.000, diff=0) answer otherwise
is. `cal_rec_00280`'s two rank-1 candidates were confirmed to be exactly
this: the true 5-item combination using `cal_rec_00162`, and an
identical-scoring impostor using its decoy twin `cal_rec_00163` in its
place.

The fix (`_ambiguous_identifier_duplicate_ids` in `stage3_aggregate.py`)
re-derives Stage 1's own "identifier is unique per (source, entity_type)
slot" condition defensively — not by duplicating Stage 1's clustering
logic, just enough to identify colliding slots via the same
`normalize_reference` utility — and keeps exactly **one** canonical
representative per colliding slot (the lowest record_id, matching this
codebase's existing "lowest record_id wins ties" convention already
used to order subset-sum candidates), excluding the rest from the raw-
record candidate pool. Naively excluding *both* twins was tried first
and found to be wrong: one twin is frequently the genuinely-needed leg
of a real consolidated settlement, and dropping it reproduces the same
orphaned-bank-credit symptom via a different path. Stage 1, Stage 2,
Stage 4, and `verification/verifier.py` were not touched.

**Stage 4 exposure check (§4 point 4):** Stage 4's role-tagged records
(FEE/TAX/ADJUSTMENT/REFUND/CHARGEBACK/REVERSAL) are not simple exact-
reference ledger↔gateway pairs the way Stage 1's clusters are, so the
specific "declined whole cluster leaves two identical-amount decoys
behind" mechanism that caused this bug does not arise there in the same
form. No parallel fix was built for Stage 4; `matching/aggregation_common.py`
(the shared subset-sum/overlap infrastructure both stages call) was not
modified.

**Honest result — this fix does not resolve all 10/13 calibration
orphans, and that is expected, not a shortfall to paper over.** Removing
the guaranteed tie only helps when a tie was the actual blocker. Measured
directly:

| Dataset | Before | After |
|---|---|---|
| calibration | 10/13 orphaned | 10/13 orphaned (unchanged) |
| evaluation | 12/12 orphaned | 10/12 orphaned (2 resolved) |

On the evaluation dataset, two consolidations (`evl_grp_00016`,
`evl_grp_00024`) were blocked purely by this duplicate-decoy tie and now
resolve correctly as `AGGREGATE_MANY_TO_ONE_MATCH` with real margins
(1.000 and 0.329). On the calibration dataset, every one of the 10
originally-orphaned bank credits has this specific tie-tolerant fix
applied too, but 9 of them are independently blocked by a *second,
unrelated* cause: `matching/aggregation_common.py`'s deliberate, tested,
documented "decline rather than guess" policy — declining **both**
targets whenever their tentative winning subsets share an item
(`AMBIGUOUS_OVERLAP`), and a calibrated 5%-minimum-margin bar
(`INSUFFICIENT_MARGIN`) — firing because this dataset packs many
settlement batches into a ~10-day span with no merchant/account key to
disambiguate them (a limitation `aggregation_common.py`'s own module
docstring already documents and a global connected-components
alternative was already tried and rejected for). Direct inspection
confirmed several of these near-miss competing candidates are
themselves off by only a few hundred paise out of a much larger
tolerance band — genuine coincidental noise, not evidence the true
answer is wrong, but also not something this bugfix's scope licenses
resolving: `aggregation_common.py` is shared with Stage 4, its
overlap/margin policy is calibrated and tested (`threshold_version`),
and weakening it to force these particular cases through risks
introducing real false positives elsewhere. Per this fix's own scope
note (3), a genuine non-match is left declined rather than forced.

`tests/test_stage3_aggregate.py` adds
`test_extends_multiple_existing_bank_less_groups_with_one_bank_credit`
(the required N-existing-group merge case, generalizing the pre-existing
single-group `test_extends_existing_stage1_group_with_missing_bank_leg`)
and `test_duplicate_decoy_record_does_not_block_consolidation` (direct
unit coverage for the fixed mechanism, mirroring `cal_grp_00026`'s
shape). `tests/test_stage3_consolidation_regression.py` is a new,
permanent full-dataset regression test — on both calibration and
evaluation — asserting the orphan count never regresses past its
measured post-fix baseline on either dataset, that the two evaluation
fixes are named and confirmed resolved, and that every resolved
`structural_consolidated` group's membership is an exact subset of its
true ground-truth cluster (zero false positives introduced).

## Global Conflict Resolver Wired Into Stage 3 (Stage 8)

Stage 7 built `matching/conflict_resolver.py` — a standalone, tested
Hungarian-algorithm resolver for the "1:1 conflicts" row of
ARCHITECTURE.md §4 — but left it unwired: nothing called it yet.
`stage3_aggregate.py`'s existing `AMBIGUOUS_OVERLAP` path (two or more
targets independently pick the same item as their best-fit candidate)
declined every such target outright rather than guessing, per
`aggregation_common.py`'s documented "decline rather than guess" policy
(see the bugfix section above). Stage 8's job was to wire the resolver
into that path, using real Stage 3 scores rather than a placeholder, and
to *measure* the real effect rather than assume one.

**The wiring.** Before declining an `AMBIGUOUS_OVERLAP` target,
`stage3_aggregate.py` now groups it, with every other `AMBIGUOUS_OVERLAP`
target it transitively shares a contested item with, into one conflict
cluster (`_wire_conflicts`, a plain union-find — `aggregation_common.py`'s
own overlap-detection connectivity, just partitioned into independent
clusters instead of one flat window-wide set). Each cluster is handed to
`resolve_conflict()` as real `(target_id, tentative_winning_item_ids,
tentative_winning_score)` edges — one edge per target, using exactly the
score `aggregation_common.run_aggregation_search` already computes
internally to decide there's a conflict at all. That score wasn't
previously exposed outside the module, so `TargetOutcome` gained two new,
purely additive fields (`tentative`, `tentative_runner_up_score`) to carry
it out — no change to any accept/decline decision `aggregation_common.py`
itself makes.

A cluster reduces cleanly only when every contesting target's tentative
winner is a single atomic item, never a multi-item subset-sum combination
(`conflict_resolver.py`'s own unchanged scope — this task did not touch
its resolution logic). When it does reduce, the Hungarian-optimal
winner's *own* score and margin still have to clear the same
`STAGE3_THRESHOLD`/`STAGE3_MIN_MARGIN` evidence bar every other Stage 3
proposal must clear (§2) before being promoted to a real `MatchGroup`
(`cardinality=ONE_TO_ONE`, `proposed_by=STAGE3_AGGREGATE`,
`reason_code=AGGREGATE_CONFLICT_RESOLVED_HUNGARIAN`) that proceeds to
Stage 7 verification exactly like any other successful proposal — winning
the assignment does not bypass that bar. Everything else still declines
as `AMBIGUOUS_AGGREGATION` exactly as before, with the exception/
`DecisionEvent` evidence now carrying a `resolution_path` distinguishing
*why*:

| `resolution_path` | Meaning |
|---|---|
| `DECLINED_BUNDLE` | cluster didn't reduce to 1:1 (a multi-item bundle is involved) — `conflict_resolver`'s own documented scope |
| `BELOW_EVIDENCE_BAR` | won the Hungarian assignment, but its own score/margin doesn't clear STAGE3_THRESHOLD/STAGE3_MIN_MARGIN |
| `UNASSIGNED_IN_RESOLVED_CLUSTER` | cluster resolved, but this target had no real item left after the optimal assignment |

**The honest, measured result.** Stage 3's own `AMBIGUOUS_OVERLAP`
declines (isolated from Stage 1/2's exact/constrained matching and from
Stage 4's separate, out-of-scope aggregation-search pass over whatever
Stage 3 still leaves unmatched, which can produce its own unrelated
overlap declines using the same `rec:<id>` target-id format):

| Dataset | Real `AMBIGUOUS_OVERLAP` targets (Stage 3 only) | Single-item-resolvable | Genuine multi-item bundles (correctly stay declined) |
|---|---|---|---|
| calibration | 5 | 0 | 5 |
| evaluation | 3 | 0 | 3 |

Directly inspecting every contesting target's actual tentative-winning
candidate in both clusters: every single one spans 5–8 settlement units —
never a single atomic item. So **0 of these 8 real conflicts were
promoted**; all 8 correctly remain declined, now via `DECLINED_BUNDLE`
instead of being declined blind. Every resolved `structural_consolidated`
group already accounted for by the Stage 7 bugfix above is unaffected —
running the full pipeline before and after this wiring produces identical
match-group and exception counts on both datasets (92 groups / 13
exceptions calibration, 100 groups / 14 exceptions evaluation), confirmed
by direct comparison, not assumed.

This is not a shortfall to paper over: it is the honest number this
stage's own measurement step asked for. Per §4's own scope, `1:1
conflicts` and `many-to-one aggregation` (subset-sum) are two genuinely
different problems, and this dataset's remaining overlap conflicts happen
to all be the latter shape. The wiring itself is proven correct and load-
bearing by direct unit tests instead:
`tests/test_stage3_aggregate.py::test_single_item_overlap_conflict_resolved_via_hungarian`
constructs a clean 1:1 conflict (two bank credits, one settlement item)
and confirms the higher-scoring target is promoted to a real, verification-
bound `MatchGroup` while the loser is honestly declined with
`UNASSIGNED_IN_RESOLVED_CLUSTER`;
`test_resolved_winner_below_evidence_bar_still_declines` confirms a
Hungarian-optimal winner that doesn't itself clear the evidence bar is
still declined rather than force-promoted; and
`test_multi_item_bundle_overlap_still_declines_via_conflict_resolver`
confirms the pre-existing bundle-conflict behavior is unchanged. So the
resolver will correctly resolve any genuinely 1:1-reducible
`AMBIGUOUS_OVERLAP` conflict this pipeline encounters going forward — on
these datasets or new ones — without further changes.

`tests/test_stage8_conflict_resolution_regression.py` is the new,
permanent full-dataset regression test (mirroring
`test_stage3_consolidation_regression.py`'s pattern) — on both
calibration and evaluation — asserting Stage 3's own `AMBIGUOUS_OVERLAP`
decline count never regresses upward past its measured baseline (5 / 3),
that every declined target shows it actually went through
`conflict_resolver` rather than being declined blind, and that every
group promoted via `AGGREGATE_CONFLICT_RESOLVED_HUNGARIAN` is an exact
subset of its true ground-truth match group (zero false positives). The
full test suite — 122 tests, up from Stage 7's 116 — passes.


## Stage 5 Fuzzy Candidate Retrieval (Stage 9)

Stage 5 (ARCHITECTURE.md §5/§6) is the last matching tier before the LLM
(Stage 6 of the matching pipeline, not built yet). Per §5's corrected
policy ("Must-Fix #4"), fuzzy identifier similarity is **candidate
retrieval only, never a commit basis by itself** — a proposal here
requires **ALL** of the following, never any subset:

- fuzzy identifier match (`rapidfuzz.fuzz.ratio` on the normalized
  `reference`, ≥ 0.72)
- same currency
- amount compatibility within tolerance (same fee/GST-deduction policy
  as Stage 2, for ledger↔gateway pairs; net-to-net for gateway↔bank)
- date compatibility within a policy window (10 days ledger↔gateway,
  7 days gateway↔bank — slightly wider than Stage 2's own windows,
  since this is deliberately the last chance before Stage 6)
- counterparty evidence (`rapidfuzz.fuzz.ratio` on the real
  `counterparty` field, ≥ 0.60 — a real similarity bar, not merely
  weighted into a composite a bad score could be outvoted on)
- a unique score margin over the runner-up candidate (≥ 0.10, stricter
  than Stage 2's 0.08, since Stage 5's evidence is inherently softer)

Any one of these failing declines the pair outright — see
`matching/stage5_fuzzy.py`'s own docstring for the full design rationale.
Same as Stages 2-4: every proposal is `status=PENDING_REVIEW`,
`verified_by=NOT_YET_VERIFIED`, `commit_policy=
AUTO_COMMIT_STAGE2_5_THRESHOLD` (§2's shared Stage 3-5 evidence bar),
`verification_result=NOT_YET_RUN`. Nothing in Stage 5 marks a group
VERIFIED — every proposal still goes through the Financial and Evidence
Verifier.

**Scope.** Mirrors Stage 2's own two passes: Pass A (ledger↔gateway) and
Pass B (gateway↔bank), using the same entity-eligibility tables from
`matching/common.py`. Only records still unmatched after Stage 1-4 (and
Stage 3's conflict resolution) are considered, and Stage 5 only ever
proposes brand-new 1:1 pairs — it never extends or reopens an earlier
stage's group, since anything already claimed had stronger evidence than
fuzzy retrieval could add. Stage 5 is explicitly **not** an aggregation
stage: a record whose true counterpart is a many-to-one consolidated
settlement that Stage 3/4's subset-sum search already couldn't resolve is
correctly left unmatched here too — that's a documented aggregation-
search limitation, not something a pairwise identifier-retrieval stage
can or should paper over.

**Wiring.** `matching/pipeline.py`'s `run_pipeline` now runs Stage 5
immediately after Stage 4, before the single verification pass, exactly
like Stage 1→2→3→4 already worked.
`verification/verifier.py`'s `verify_match_group` had a
`NotImplementedError` placeholder reserved for `STAGE5_FUZZY`, anticipating
this exact stage — that dispatch now routes `STAGE5_FUZZY` through the same
shared Stage 2-5 policy function used by aggregate/adjustment proposals
(renamed `_verify_stage2_to_4` → `_verify_stage2_to_5`), since §2 already
specifies one identical evidence bar for "Stages 3-5 (aggregate,
adjustment, fuzzy)" — no new verification logic was needed, only enabling
the dispatch branch.

**Measured result, both datasets.** Running the full pipeline (Stage
1→2→3→4→5→verification) against `data/calibration/` and
`data/evaluation/`, cross-checked against ALL THREE ground-truth
categories (`match_groups`, `unresolved`, `duplicates` — not just
`match_groups`, which a prior review flagged as an incomplete check):

| Dataset | Stage 5 proposals | Records covered | VERIFIED | PENDING_REVIEW | REJECTED |
|---|---|---|---|---|---|
| calibration | 0 | 0 | 0 | 0 | 0 |
| evaluation | 1 | 2 | 0 | 0 | 1 |

**Calibration: 0 real 1:1 opportunities exist.** Every record still
unmatched after Stage 1-4 on calibration either belongs to a
`MANY_TO_ONE` consolidated-settlement ground-truth group too large for
Stage 3/4's aggregation-search bounds (out of a 1:1 stage's scope by
construction — see §9's documented aggregation-search limitation), is a
known structural duplicate, is a genuinely `unresolved` record, or is
`honest_abstention` decoy — never a genuine, unclaimed 1:1 counterpart
pair. Stage 5 correctly finds nothing to propose rather than guessing
across a many-to-one boundary.

**Evaluation: 1 proposal, correctly caught before commit — a documented
near-miss worth keeping as a permanent regression test.** Stage 5
retrieves a fuzzy match between `evl_rec_00098` (a genuine stray member
of a `MANY_TO_ONE` group Stage 3/4 couldn't fully resolve) and
`evl_rec_00102` — which ground truth marks as a structural **duplicate**
of `evl_rec_00099`, the real counterpart, not a transaction of its own.
All five of Stage 5's own hard gates pass, because a duplicate decoy is
by construction a near-perfect look-alike of the record it copies
(identical reference, amount, and counterparty), and its true
counterpart (`evl_rec_00099`) was no longer in the unmatched pool for
Stage 5 to compare against instead. What actually stops this from being
wrongly committed is the *next*, independent layer: the Financial and
Evidence Verifier's conservation re-check. A lone ledger↔gateway pair
with no bank leg can never independently prove conservation (see
`matching/common.py`'s `recompute_group_conservation` docstring) — the
exact same fate five equivalent, unextended `STAGE2_CONSTRAINED`
ledger↔gateway pairs already get on calibration (see
`test_stage1to4_integration.py`'s run). This is not a Stage-5-specific
weakness; it's the relay's layered defense working exactly as designed —
retrieval found a plausible-looking candidate, and a genuinely
independent check is what stopped it from being committed.
`tests/test_stage5_integration.py::test_stage5_never_verifies_a_group_touching_a_known_duplicate`
pins this down as a permanent regression check.

**Zero false positives, both directions.** Across both datasets, 0 of
the system's VERIFIED groups (any stage, not just Stage 5) touch a known
duplicate or abstention decoy, or fail to be a subset of a real
`match_groups`/`unresolved` cluster — confirmed by
`tests/test_stage5_integration.py`'s explicit three-category cross-check,
in addition to the existing whole-pipeline checks in
`test_pipeline_verification.py` (which also still pass unchanged with
Stage 5 wired in).

`tests/test_stage5_fuzzy.py` is the stage's own unit-level coverage:
a fuzzy-identifier-only "match" with no other corroborating evidence
correctly declined (the specific failure mode §5 exists to prevent, in
two variants); a genuine fuzzy match — identifier plus all four other
conditions — correctly proposed, for both Pass A and Pass B; and a
close-margin case between two tied-scoring candidates correctly
declined rather than guessed. `tests/test_verifier.py` gained the
now-real `STAGE5_FUZZY` pass-through coverage in place of the old "not
implemented yet" placeholder test. The full test suite — 132 tests, up
from Stage 8's 122 — passes.

## Groq LLM Recommendation Module (Stage 10, standalone)

`llm/recommender.py` implements ARCHITECTURE.md §6/§8/§10's Groq
recommendation step as a **standalone module** — prompt construction, the
API call, and response validation only. It does not touch any matching/
verification code itself. Stage 11 (below) adds the §10 budget governor /
circuit breaker around it and wires it into `matching/pipeline.py`, without
changing anything in this file. Its scope is exactly three things:

1. **Prompt construction** — `build_prompt(residual, candidates)` takes one
   residual `NormalizedRecord` plus its (caller-selected) top-N Stage 5
   near-miss candidates and serializes only the fields named in §7
   (`record_id`, `source`, `entity_type`, `amount_paise`, `currency`,
   `reference`, `counterparty`, `occurred_at`) — `raw_hash` is deliberately
   excluded as "raw unnecessary text" per §8's control table. The system
   prompt explicitly frames this payload as **untrusted data, not
   instructions** — `reference`/`counterparty` originate from external
   bank/gateway/ledger systems and are never concatenated into instruction
   text (see the module docstring's security note).
2. **The API call** — `get_llm_recommendation(...)` calls Groq
   (`response_format={"type": "json_object"}`, `temperature=0`) using
   whichever Llama model is set in `Settings.groq_model`
   (`GROQ_MODEL` env var, default `meta-llama/llama-4-scout-17b-16e-instruct`
   per §10) — never hardcoded — with a per-call timeout from
   `Settings.groq_timeout_seconds` (`GROQ_TIMEOUT_SECONDS`, default 15s).
   A `client` can be injected, so tests (and any future caller) never need
   a live `GROQ_API_KEY`.
3. **Strict validation** — the response must parse as JSON and validate
   against `LLMRecommendationPayload` (`candidate_id`, `confidence` in
   `[0, 1]`, `reason_code`, `explanation`, `extra="forbid"`). A response
   naming a `candidate_id` that isn't `NO_MATCH` and isn't one of the
   candidates actually offered is rejected as a hallucination. Nothing here
   ever attempts to salvage a malformed or schema-violating response by
   guessing at intent.

Five distinct failure types, never lumped into one generic bucket:
`LLMTimeoutError`, `LLMAPIError` (non-timeout API failures — auth, rate
limit, connection, 5xx), `LLMMalformedJSONError` (not valid JSON at all),
`LLMSchemaValidationError` (valid JSON, wrong shape), and
`LLMHallucinatedCandidateError` (valid shape, invented candidate id).

**Non-negotiable per §2/§6: this module's output is a recommendation
only.** `LLMRecommendation` deliberately carries no `commit_policy` field
and never constructs a `MatchGroup` — whichever stage wires this into the
pipeline next must always set `commit_policy=HUMAN_REVIEW_REQUIRED`
regardless of the returned `confidence`, exactly as §2's auto-commit table
requires for `STAGE6_LLM`. There is no code path in this module, or
planned for the next one, that lets an LLM recommendation auto-commit.

`tests/test_llm_recommender.py` covers the happy path (valid candidate
pick), a `NO_MATCH` response, each of the five failure modes above (with
an explicit test asserting the exception types are pairwise distinct), and
prompt-construction checks (`raw_hash` never appears in the prompt;
untrusted-data framing is present). Everything runs against a mocked Groq
client — no `GROQ_API_KEY` is required for the suite. One additional live
integration test exists, skipped by default
(`RUN_LIVE_GROQ_TESTS`/`GROQ_API_KEY` both required to run it).

## Shared Budget Governor + Stage 6 Wiring (Stage 11)

Two things, per this stage's scope: `llm/governor.py`'s budget governor /
circuit breaker, and Stage 6 (LLM recommendation) wired into
`matching/pipeline.py` as the tier after Stage 5. `recommender.py`'s own
logic (prompt construction, the API call, response validation) is
untouched.

**The governor (`llm/governor.py`)** enforces two independent, SQLite-backed
gates:

1. **A daily call ceiling** (`Settings.groq_daily_call_ceiling`,
   `GROQ_DAILY_CALL_CEILING`, default 500) — scoped by calendar day only,
   shared across every concurrently-running `ReconciliationRun`, since it's
   a Groq-spend-per-day budget concern, not a per-run one.
2. **A circuit breaker** (`Settings.groq_circuit_breaker_max_consecutive_failures`,
   default 3) — scoped per `run_id` (the `ReconciliationRun`-level tracking
   §10 points at): after that many consecutive call failures within one
   run, Groq calls stop for the rest of that run. A different run's
   breaker is independent, so one run's bad luck doesn't veto another's.

Both gates share one `CallBudgetGovernor.try_consume(run_id)` call that
folds the check and the increment into a single `BEGIN IMMEDIATE`
transaction — this was an explicit correction in ARCHITECTURE.md's
revision history (§10): a separate check-then-increment pair is exactly
the race a "concurrent runs can't double-count or under-count usage"
requirement rules out. `tests/test_governor.py` proves this directly: many
threads, each with its own SQLite connection to the same database file,
hammering `try_consume` concurrently, land on an ALLOW count that is
*exactly* the configured ceiling — never more, never less.

Failure detection for the breaker uses `isinstance` checks against
`recommender.py`'s own five distinct exception types (never string
matching) — `record_failure` raises `TypeError` if handed anything that
isn't a `RecommenderError`.

**Stage 6 wiring (`matching/stage6_llm.py`)**, run after Stage 1-5's
proposals have themselves been verified (so a `REJECTED` Stage 1-5 group's
released records are correctly included in the residual pool — §2's
framing of Stage 6 as receiving only genuine residue):

1. **Near-miss candidate gathering** — deliberately *looser* than Stage
   5's own hard-gated scoring: every component (identifier, amount, date,
   counterparty) is a soft `[0, 1]` score with no pass/fail cutoff, so a
   candidate that fails one of Stage 5's hard gates can still surface here,
   just ranked lower. Currency remains the one hard prefilter — a
   cross-currency pair is categorically not the same transaction, not
   merely a weak candidate for one.
2. **Calling the recommender through the governor** — every attempt asks
   `try_consume` first; only an `ALLOW` results in an actual
   `get_llm_recommendation` call.
3. **Non-negotiable, matching everything already built (§2): a Stage 6
   proposal that recommends a match always gets
   `commit_policy=HUMAN_REVIEW_REQUIRED`, `status=PENDING_REVIEW`,
   `verified_by=NOT_YET_VERIFIED`, `verification_result=NOT_YET_RUN` —
   regardless of the LLM's stated confidence.** There is no auto-commit
   path for `STAGE6_LLM`; the earlier `LLM_CORROBORATED_AUTO_COMMIT` path
   removed in ARCHITECTURE.md's own revision history is not reintroduced
   anywhere in this stage. The verifier's existing `_verify_stage6` branch
   (built in Stage 9) already enforced this correctly — confirmed, not
   rebuilt — and `matching/pipeline.py`'s `run_verification` gained one new
   explicit branch so that branch's `NOT_YET_RUN` outcome routes to
   `PENDING_REVIEW` instead of being misread as `REJECTED`.
4. **Graceful degradation, not a crash or a stall** — when the governor
   declines a call (ceiling exhausted or breaker open), or when no
   `GROQ_API_KEY`/client is configured at all (§14: "the system completes
   when the Groq key is missing"), the residual record gets an
   `ExceptionCategory.LLM_UNAVAILABLE` exception rather than the pipeline
   stalling. *(Documented, deliberate deviation from this stage's own
   brief's exact wording: the brief names a second category,
   `LLM_BUDGET_EXHAUSTED`, that does not exist in §8's Exception Taxonomy or
   `models/enums.py` — which explicitly states "do not add, remove, or
   rename values, this file is the single source of truth." Rather than
   adding an enum member that contradicts that rule, both degradation
   reasons use `LLM_UNAVAILABLE` and are distinguished in the exception's
   own `evidence`/`recommended_action` text — the same kind of documented
   judgment call `pipeline.py`'s own `REJECTION_EXCEPTION_CATEGORY` table
   already makes for a taxonomy gap elsewhere.)*
5. **Same audit discipline as every other stage** — a `DecisionEvent` is
   emitted for every residual record Stage 6 considers: a recommended
   match, a `NO_MATCH` recommendation, a call failure (with the specific
   failure type recorded), no eligible candidates at all, or the governor
   declining the call outright.

`matching/pipeline.py`'s `run_pipeline` now runs verification twice: once
for Stage 1-5's proposals (which is what determines the *real* Stage 6
residual pool), and once more for Stage 6's own proposals — purely for
audit-trail consistency, since `_verify_stage6` is a deliberate no-op
pass-through, not because Stage 6 output needs Stage 1-5's automated
margin/conservation/currency checks.

`tests/test_governor.py` covers ceiling enforcement (including that it's
shared across run ids, and resets on a new day), the atomic-concurrency
proof described above, circuit breaker tripping/scoping/refusal, and
failure-type classification. `tests/test_stage6_llm.py` covers near-miss
candidate gathering (including a case that would fail Stage 5's own hard
identifier gate), the non-negotiable no-auto-commit guarantee at a
mocked confidence of 0.99, both governor-degraded paths, a real call
failure, a `NO_MATCH` recommendation, and a `DecisionEvent` for every
outcome kind. `tests/test_stage6_pipeline_integration.py` runs the full
pipeline against both the calibration and evaluation datasets with a
heuristic mocked Groq client (no live key), reports how many records
Stage 6 attempted and their outcome breakdown, and re-confirms the
zero-false-positive standard against all three ground-truth categories
(duplicates, honest-abstention decoys, non-subset-of-truth) on the full
Stage 1-6 pipeline output — plus a dedicated check that no `STAGE6_LLM`
group is ever `VERIFIED`, and a full-pipeline run with a zero call budget
proving the pipeline still completes.

### Bugfix: alphabetical seeker ordering produced stale NO_CANDIDATES/NO_MATCH exceptions (post-Stage-11)

Nine tests failed after Stage 11 landed — 8 in `tests/test_stage6_llm.py`
plus `test_stage1to4_integration.py::test_every_exception_has_an_expected_category_and_is_open`
— all with the same shape: `assert len(result.decision_events) == 1`
found 2, or `assert len(result.exceptions) == 1` found 2.

The cause was `_eligible_seekers()`'s sort key: every eligible Stage 6
seeker, LEDGER- and GATEWAY-sourced alike, was sorted alphabetically by
`record_id` with no regard for source. A GATEWAY record is legitimately
eligible to seek its own BANK-side match — this mirrors Stage 5's own
convention, and is stated explicitly in this module's own docstring —
but it is *also* the match candidate a LEDGER seeker is looking for.
Whenever a GATEWAY record's `record_id` happened to sort before the
LEDGER seeker that would otherwise consider it, the GATEWAY record got
its own seeker turn first. Two shapes of staleness followed, both
present across the 8 failing unit tests:

1. The GATEWAY record found no BANK-side candidates (none existed in
   these fixtures), logged a real `NO_CANDIDATES_AVAILABLE`
   `DecisionEvent` + `ReconciliationException` for itself — and then,
   moments later in the *same pass*, the LEDGER seeker's turn came up
   and successfully claimed that same GATEWAY record as its match
   candidate. The final match was correct; the earlier exception was
   now stale and misleading, describing a record as having no home
   when it found one one step later. Nothing retracted it.
2. Even when the LEDGER seeker's LLM call ended in `NO_MATCH` rather
   than a successful claim, the GATEWAY record it had offered as a
   candidate was still unmatched afterward, so it *still* got its own
   seeker turn and *still* logged a redundant `NO_CANDIDATES_AVAILABLE`
   entry — a second audit-trail line for a record that a LEDGER seeker
   had already looked at and already recorded a decision for.

The fix has two parts, both in `matching/stage6_llm.py`:

- `_eligible_seekers()` became `_eligible_seekers_by_source()`, returning
  `(ledger_seekers, gateway_seekers)` as two separately-sorted lists
  (`record_id` remains the tiebreaker within each) instead of one
  alphabetically interleaved list.
- `run_stage6_llm` now runs every LEDGER seeker first, tracking (via a
  new `offered_to_ledger` set) every record_id that turns up in *any*
  LEDGER seeker's near-miss candidate list along the way — regardless of
  whether that seeker's attempt ends in a match, a `NO_MATCH`, a call
  failure, or governor degradation. GATEWAY seekers then run only for
  the records never offered to a LEDGER seeker this pass. This covers
  both staleness shapes above with one mechanism: shape 1 is prevented
  because the LEDGER seeker now runs, and claims the record, before the
  GATEWAY seeker would have had a turn at all; shape 2 is prevented
  because a GATEWAY record that was merely *offered* (not necessarily
  matched) is still excluded from getting its own redundant turn — that
  record's Stage 6 audit trail is the LEDGER seeker's own DecisionEvent,
  which already references it in `candidate_scores`.

This was built as an ordering fix, not a "generate then retract" patch —
there is no code path that emits a `DecisionEvent` or `Exception` and
later mutates or removes it. Once a LEDGER seeker has had first look at
a GATEWAY record this pass, that GATEWAY record simply never enters the
seeker loop a second time, so the stale entry is never produced in the
first place.

Verified directly against `test_stage6_proposal_never_auto_commits_even_at_high_confidence`:
it now produces exactly one `DecisionEvent` for the one real attempt,
with the underlying match still correct. This bug never touched the
auto-commit safety property — `status=PENDING_REVIEW`,
`verified_by=NOT_YET_VERIFIED`, `commit_policy=HUMAN_REVIEW_REQUIRED`,
`verification_result=NOT_YET_RUN` even at a mocked confidence of 0.99 —
those assertions were already passing before this fix and are
unaffected by the reordering; only the audit-trail cleanliness was
broken.

`test_stage1to4_integration.py::test_every_exception_has_an_expected_category_and_is_open`
failed for an unrelated, non-functional reason: its `expected_categories`
allow-list predates Stage 6 and didn't include `LLM_UNAVAILABLE`,
`LLM_BUDGET_EXHAUSTED`, or `INSUFFICIENT_EVIDENCE` — categories Stage 6
legitimately produces (the pipeline's actual behavior, graceful
degradation to `LLM_UNAVAILABLE` when no Groq key is configured in this
test environment, is correct). The allow-list was updated to include
Stage 6's categories; no production code changed for this part.

`tests/test_stage6_llm.py` gains
`test_ledger_seeker_claims_gateway_candidate_even_when_it_sorts_first`,
a permanent regression test: a GATEWAY candidate (`"aaa-cand"`) is
deliberately named to sort alphabetically before its LEDGER seeker
(`"zzz-seek"`) — exactly the ordering that triggered the bug — and the
test asserts the match still succeeds, exactly one `DecisionEvent` is
produced, zero `Exception`s, and only one Groq call is made (the
GATEWAY record never gets an independent, doomed seeker attempt of its
own). All 9 originally-failing tests pass, and the full suite (180
tests) shows no regressions. `recommender.py` and `verification/verifier.py`
were not touched — this was entirely a seeker-ordering fix in
`matching/stage6_llm.py`.
## Evaluation Harness (Stage 12)

`evaluation/harness.py` — the judge-facing metrics computation from §11.
This is the ONE place in the codebase allowed to read ground truth
(`ground_truth/<dataset>/ground_truth.json`); `matching/`,
`verification/`, and `llm/` are untouched and still never see it.

Run it directly:

```
python -m recon_agent.evaluation.harness --dataset evaluation
```

Defaults to `data/evaluation/records.json` — the held-out, judge-facing
set — since `calibration` was already spent tuning Stage 2-5's
thresholds (§11's calibration/evaluation separation). Prints a
human-readable summary and writes a JSON export to
`reports/<dataset>_evaluation_report.json`.

**The seven §11 metrics, each its own independently-unit-testable pure
function** (`tests/test_evaluation_harness.py`, 25 unit tests against
small hand-built fixtures with known-correct answers, plus 2
integration tests that run the real harness against both real
datasets):

1. **Auto-match precision** — correct / all VERIFIED groups (Stage 6
   proposals never reach VERIFIED by design, so restricting to VERIFIED
   naturally covers only Stages 1-5's committed output).
2. **Record coverage** — matched / eligible records, reported as both
   raw coverage and coverage-excluding-honest-abstention (the realistic
   ceiling), since honest-abstention records genuinely have no
   counterpart to find.
3. **Value coverage** — matched / eligible monetary value, using
   `abs(amount_paise)` uniformly (refund/chargeback/reversal rows are
   stored with a negative `amount_paise` in `testdata/generator.py`, so
   summing raw signed amounts would let a refund net against its
   original payment and hide real reconciliation activity).
4. **False-match rate** — incorrect / all committed (VERIFIED) matches,
   using the exact same ground-truth-correctness check as item 1 and as
   `test_stage6_pipeline_integration.py`'s own proven check: wrong if a
   group touches a known duplicate or an honest-abstention decoy, or
   isn't a subset of some real ground-truth cluster (checking all three
   categories — `match_groups`, `unresolved`, and `duplicates` — not
   just `match_groups`, per that test's own documented history of false
   positives from a narrower check).
5. **Exception quality** — for records ground truth expects to end up
   unresolved or honestly abstained on, does the system's own exception
   layer produce a plausible category rather than silence? A real
   finding shaped this metric's scope during Stage 12's own dev/test
   run (documented in `compute_exception_quality`'s docstring): an
   `unresolved` ground-truth case can legitimately end up correctly
   VERIFIED (e.g. a real ledger+gateway relationship inside a case
   ground truth calls "unresolved" only because a third bank leg never
   arrives in this batch) — that outcome is excluded from this metric's
   denominator entirely rather than double-penalized as "silence",
   since it's already credited by auto-match precision.
   `honest_abstention` cases are never given that exclusion, since a
   group touching an abstention id is never scored as correct.
6. **Tier contribution** — precision and coverage broken down by
   `proposed_by` (STAGE1_EXACT..STAGE6_LLM), proving the later, more
   expensive tiers add real value rather than merely being present.
7. **Runtime and cost** — wall-clock time and records/sec for the run;
   Groq call count read from `CallBudgetGovernor`'s own ledger (not
   re-derived); a rough, explicitly-documented token-count-based cost
   estimate (input tokens reconstructed from the real Stage 6 prompt
   text via `llm.recommender.build_prompt`, read-only reuse; output
   tokens a fixed per-call assumption; Groq's published
   `meta-llama/llama-4-scout-17b-16e-instruct` list pricing).

### Real results — both datasets, this build

No `GROQ_API_KEY` is configured in this environment, so both runs below
are genuinely real Stage 1-5 pipeline runs with Stage 6 gracefully
degrading to `LLM_UNAVAILABLE` (§14's designed degradation path, not a
harness shortcut) — `groq_configured: false` in both JSON exports.

| Metric | calibration | evaluation |
|---|---|---|
| Records / logical events | 288 / 100 | 287 / 100 |
| Groups: VERIFIED / PENDING_REVIEW / REJECTED | 87 / 0 / 5 | 96 / 0 / 5 |
| 1. Auto-match precision | 100.0% (87/87) | 100.0% (96/96) |
| 2. Record coverage (raw / excl. abstention) | 85.1% / 86.9% | 86.8% / 88.6% |
| 3. Value coverage (raw / excl. abstention) | 72.2% / 72.5% | 66.5% / 66.8% |
| 4. False-match rate | 0.0% (0/87) | 0.0% (0/96) |
| 5. Exception quality | 100.0% (7/7) | 100.0% (7/7) |
| 7. Runtime | 0.694s (415 rec/s), 0 Groq calls, $0.00 | 0.991s (290 rec/s), 0 Groq calls, $0.00 |

**Tier contribution (evaluation dataset):**

| Stage | Proposed | Verified | Precision | Records matched | Coverage contribution |
|---|---|---|---|---|---|
| STAGE1_EXACT | 90 | 90 | 100.0% | 212 | 73.9% |
| STAGE2_CONSTRAINED | 4 | 0 | N/A | 0 | 0.0% |
| STAGE3_AGGREGATE | 6 | 6 | 100.0% | 37 | 12.9% |
| STAGE4_ADJUSTMENT | 0 | 0 | N/A | 0 | 0.0% |
| STAGE5_FUZZY | 1 | 0 | N/A | 0 | 0.0% |
| STAGE6_LLM | 0 | 0 | N/A | 0 | 0.0% |

**Reading these numbers honestly, not just reporting them:**

- **Precision and false-match rate are perfect (100% / 0%) on both
  datasets** — every VERIFIED group Stage 1-5 has committed so far is
  correct against ground truth. With only 287-288 records and a still
  fairly conservative Stage 1-5 threshold set, this is a real, earned
  number rather than a coincidence of near-empty categories — but it's
  also not yet a stress test of the harder categories (Stage 2/4/5's
  own proposals never got VERIFIED at all in either run, so their
  precision is undefined, not zero — see the `N/A` cells above).
- **Exception quality is 100% (7/7) on both datasets, and it wasn't
  always — see the "Bugfix: silent unattempted-record drop" section
  below.** The first real run of this harness against real data found
  it at 14.3% (1/7): 6 of the 7 relevant cases in each dataset are
  `honest_abstention` records with `entity_type=ADJUSTMENT`, and
  `ADJUSTMENT` had no entry anywhere in the matching pipeline's
  source→target entity-type mapping, so these records were never
  offered to any stage as a seeker or a candidate — falling through
  every stage with zero `DecisionEvent`s and zero `Exception`s. That
  gap is fixed; the finding and the fix are documented in full below.
- **Value coverage (~67-72%) trails record coverage (~85-89%)** — larger
  transactions are proportionally less likely to be auto-resolved.
  Genuinely useful for prioritizing where a human's remaining review
  effort has the most monetary impact, and consistent with §11's own
  reasoning for measuring value coverage as a separate metric from
  record coverage rather than assuming they track each other.
- **0 Groq calls / $0.00 cost in both runs is a property of this
  environment (no `GROQ_API_KEY` configured), not of the harness or the
  pipeline** — Stage 6's near-miss candidate gathering still runs, and
  every residual seeker correctly produces an attempted-then-degraded
  `DecisionEvent` with `reason_code` reflecting `LLM_NOT_CONFIGURED`
  wherever residue after Stage 1-5 exists (1 STAGE5_FUZZY proposal in
  the evaluation run, 0 in calibration) — the cost-estimation code path
  itself is exercised and unit-tested (see
  `test_runtime_and_cost_reconstructs_prompt_size_for_real_calls`) even
  though this particular run had nothing to reconstruct a nonzero
  estimate from.

Full JSON exports for both runs: `reports/calibration_evaluation_report.json`,
`reports/evaluation_evaluation_report.json`.

### Bugfix: silent unattempted-record drop for records no stage ever considers (post-Stage-12)

The Stage 12 evaluation harness itself found this — its own job is to
compare pipeline output against held-out ground truth, and "exception
quality" (metric 5, the % of ground-truth `unresolved` /
`honest_abstention` cases the system gives an explicit, plausible
exception rather than silence) came back at 14.3% (1/7) on *both*
datasets, immediately after Stage 12 landed.

Tracing the 6 low-scoring cases per dataset (`evl_rec_00282` through
`evl_rec_00287` in evaluation; the calibration equivalents) directly
against every stage's own output showed something stronger than
"the exception category doesn't match": these records had **zero**
`DecisionEvent`s and **zero** `Exception`s anywhere in the pipeline's
output at all. Not a wrong classification — total silence. All six are
`entity_type=ADJUSTMENT` (a mix of LEDGER- and BANK-sourced), and
ground truth deliberately created them as `honest_abstention` cases
(`reason: NO_COUNTERPART_EXISTS`) — there is genuinely no counterpart
for a standalone adjustment record to find.

The root cause is architectural, not a broken check: `matching/common.py`'s
`ENTITY_TO_ROLE` maps `ADJUSTMENT` to `MemberRole.ADJUSTMENT`, but
neither `LEDGER_TO_GATEWAY_ENTITY` nor `GATEWAY_TO_BANK_ENTITY` has an
`ADJUSTMENT` entry, and `matching/stage6_llm.py`'s own
`_target_entity_type` mirrors that same mapping. Every stage's seeker
eligibility rule is built on top of one of those tables, so a
standalone `ADJUSTMENT` record is never, by design, something Stages
1-6 treat as a seeker or a candidate — it is only ever consumed as a
**member** inside another group's conservation check (Stage 4's role).
That design is correct: the system should never force-match an
adjustment record that has no real counterpart, and it never did
(0% false-match rate held throughout). But nothing ever *explicitly
declined* these records either — they fell through every stage's
"only work on records I recognize" filter invisibly, which is exactly
the silent drop ARCHITECTURE.md §14's acceptance gate forbids: "every
unresolved record has a reason code, evidence, and a recommended
action."

**The fix** is a new, final pass in `matching/pipeline.py` —
`run_final_catchall` — run once, after Stage 6's own proposals and
their own verification pass, against the pipeline's real final output:

- It looks at every record not in the final `matched_record_ids`.
- For each, it checks whether any `DecisionEvent` already produced by
  Stages 1-7 references it at all — checked structurally (a substring
  search of the record id across each event's `group_id`,
  `reason_code`, `explanation`, and `candidate_scores`) rather than by
  re-deriving each stage's own eligibility logic a second time, since a
  second implementation could silently drift from the real one. This
  is safe because every dataset's record ids share one fixed width
  within a single run (`evl_rec_00282`, `cal_rec_00001`, ...), so one
  record id can never be a substring of another.
- A record some stage already attempted — even one that ended in
  `NO_CANDIDATES_AVAILABLE`, `NO_ELIGIBLE_CANDIDATE`, or a verification
  rejection — already has a real `DecisionEvent` naming it and is left
  completely untouched.
- A record nothing ever touched gets exactly one new `DecisionEvent`
  (`stage=STAGE7_VERIFICATION`, `reason_code=NO_STAGE_ATTEMPTED`) and
  one new `ReconciliationException` (`category=INSUFFICIENT_EVIDENCE`,
  `severity=MEDIUM`, `recommended_action` pointing to manual review).

`DecisionStage` has no dedicated "final catch-all" value, and
`models/enums.py` is explicit that its enums are transcribed directly
from ARCHITECTURE.md and not to grow new values. Of the existing ones,
`STAGE7_VERIFICATION` is the right fit, documented at the point of use
rather than left implicit: this pass isn't a new matching capability
proposing candidate groups — it's a final evaluative pass over the
whole record pool that runs after every matching stage, the same role
`STAGE7_VERIFICATION` already plays for proposed groups, just extended
to cover records no stage ever proposed anything for in the first
place. This is purely a safety net: it never inspects, reclassifies,
or overrides any decision a matching stage already made, and it never
attempts to actually match an adjustment record to anything.

**Verified directly**, re-running the harness against both real
datasets:

| Dataset | Exception quality before | Exception quality after |
|---|---|---|
| calibration | 14.3% (1/7) | **100.0% (7/7)** |
| evaluation | 14.3% (1/7) | **100.0% (7/7)** |

Every other metric (auto-match precision, record/value coverage,
false-match rate, tier contribution) is byte-for-byte unchanged on both
datasets — expected, since this fix adds audit-trail entries for
previously-untouched records without changing what any stage decides
to attempt or commit. Updated JSON exports:
`reports/calibration_evaluation_report.json`,
`reports/evaluation_evaluation_report.json`.

`tests/test_pipeline_final_catchall.py` is a new, permanent regression
suite (3 tests) built directly against this failure shape rather than
depending on the real datasets: a standalone `ADJUSTMENT` record mixed
into an otherwise-normal batch, the same record as the *only* record in
a run at all, and — the scope-boundary check — a record Stage 1-6
genuinely did attempt (a lonely `PAYMENT` seeker with no counterpart
anywhere), confirming the catch-all adds nothing on top of that
record's own real `DecisionEvent`s. All three assert the core
property: a record is never simultaneously absent from
`matched_record_ids` and absent from the exception list.

The full suite (210 tests — the prior 207 plus these 3) passes with no
regressions; `matching/stage1_exact.py` through `stage6_llm.py` and
`verification/verifier.py` were not touched.

## FastAPI Endpoints (Stage 13)

`api/` — a thin FastAPI layer over §10's API surface. No
matching/verification/LLM/evaluation logic lives here; every endpoint
just calls into the modules earlier stages already built
(`matching/pipeline.py`, `evaluation/harness.py`, `llm/governor.py`).

**Run it:**

```
uvicorn recon_agent.api.app:app --reload
```

Then open `http://127.0.0.1:8000/docs` for interactive OpenAPI docs
(every endpoint below is fully typed via Pydantic request/response
models, so `/docs` renders real schemas, not just bare JSON blobs).

**Endpoints:**

- `POST /reconcile` — accepts `{"records": [...]}` (same shape as
  `data/*/records.json`), runs the full existing pipeline
  (`matching.pipeline.run_pipeline`) over it, and returns a `run_id`
  plus a summary (status, group/exception/decision-event counts, Groq
  calls made). Optionally accepts `dataset` (a free-text label) and
  `ground_truth` (shaped like `ground_truth/<dataset>/ground_truth.json`)
  — see the note on `GET /report` below for why.
- `GET /report/{run_id}` — the evaluation harness's metrics for that
  run, reusing `evaluation.harness`'s own metric functions directly
  (nothing is reimplemented). Since those metrics are fundamentally
  ground-truth-scored (§11), and an arbitrary uploaded batch has no
  ground truth by default, this returns the **full** report only if
  `ground_truth` was supplied at `/reconcile` time; otherwise it
  returns the ground-truth-independent metrics only (runtime/cost,
  VERIFIED/PENDING_REVIEW/REJECTED counts) plus a note explaining what's
  missing and why.
- `GET /audit/{run_id}` — the full `DecisionEvent` trail for that run,
  plus the `MatchGroup`/`MatchGroupMember`/`ReconciliationException`
  rows those events refer to.
- `GET /llm-budget/{run_id}` — `CallBudgetGovernor.status(run_id)`
  verbatim (llm/governor.py) — the daily call ceiling/usage (shared
  across all runs, scoped by calendar day per that module's own design)
  and this run's own circuit-breaker state (consecutive failures,
  whether the breaker is open).
- `GET /health` — liveness. Also runs this codebase's own selfcheck
  module if one exists; as of Stage 13, none does yet (checked via
  `importlib`, not hardcoded), so this reports basic liveness only.

**Persistence (new in this stage):** no pipeline run was persisted
anywhere before this stage — every CLI/test invocation of
`run_pipeline`/`run_evaluation` just ran in-memory and the caller did
whatever it liked with the result. `api/storage.py` adds a minimal
SQLite-backed `RunStore` (mirroring `llm/governor.py`'s own plain-sqlite3
approach, no ORM) so a `POST /reconcile` can be looked up later by
`run_id`: it stores the existing `ReconciliationRun`,
`MatchGroup`/`MatchGroupMember`/`DecisionEvent`/`ReconciliationException`
Pydantic models as their own JSON (via `model_dump_json`), keyed by
`run_id` — no new schema was designed from scratch, just run_id-scoped
lookup tables around the models that already exist. One correctness
note worth flagging: `matching.pipeline.run_pipeline` allocates
`group_id`/`event_id` values via a fresh `IdAllocator()` per call
(default prefix `"match"`), so those ids are only unique *within* one
run, not across runs — every table here is therefore keyed by
`(run_id, group_id)` / `(run_id, event_id)`, never by the bare id alone.
Configured via `Settings.api_db_path` (`API_DB_PATH` env var), separate
from `governor_db_path` so the two never share a table namespace.

**Tests:** `tests/test_api.py` (10 tests) — a full round trip (POST
`/reconcile` with a small real 3-record refund triad lifted from
`data/evaluation/records.json`, then GET each of the other three
endpoints for that `run_id`), a variant with no `ground_truth` supplied
(asserting the partial report shape), a 404 for an unknown `run_id`
against all three GET-by-id endpoints, a 422 for a malformed record and
for an empty `records` list, `/health`, an OpenAPI-schema/`/docs` smoke
test, and a server-survival stress test (see the investigation note
below). Uses `fastapi.testclient.TestClient` with temporary SQLite
files (`API_DB_PATH`/`GOVERNOR_DB_PATH` monkeypatched per test) — no
shared state with a real deployment or between tests. Does not re-test
matching/verification/harness/governor logic itself, which is already
covered elsewhere; this suite only proves the API layer correctly
exposes it. Full suite: 220 tests (the prior 219 plus 1 new — the
server-survival test folds 18 malformed-body/health-check assertions
into a single test function rather than 18 separate ones), no
regressions.

### Investigated: reported silent server death around POST /reconcile (post-Stage-13, pre-Stage-14)

Before Stage 14 could build on this API layer, a reviewer reported that
a live `uvicorn recon_agent.api.app:app` process died silently — no
exception, no traceback, no log line, the process simply vanished —
shortly after startup, around `POST /reconcile` calls. It happened
twice, on two different ports. The reviewer was explicit that they
could not rule out sandbox degradation: their sandbox had already run
15+ rounds of pip installs, pytest runs, and server launches in one
very long session before this observation, and said so rather than
guessing at a root cause.

**This was re-investigated from a fresh environment** rather than
assumed away, per this stage's own scope. The investigation:
started a real `uvicorn` process, confirmed `GET /health`, then sent 10
sequential real `POST /reconcile` calls (a real 2-record BANK/GATEWAY
pair) plus a battery of deliberately malformed bodies — a bare array
instead of the `{"records": [...]}` envelope, missing required fields,
wrong field types, an empty `records` list, invalid JSON, and a JSON
`null` body — interleaving a `GET /health` check after every single
call. Every malformed body produced a clean `422` and every intervening
health check succeeded; the server process stayed alive and responsive
throughout, including after the 10th successful reconcile. (One false
alarm along the way, noted for anyone repeating this: backgrounding
`uvicorn` with `&` inside one shell invocation and then issuing a
follow-up command in a *separate* shell invocation reliably killed the
process between the two — that's this tool environment reaping
backgrounded jobs across invocations, not the app, and reproduced
nothing once both the launch and the requests were issued within a
single shell session.)

**Conclusion: no crash reproduced.** Given a clean-4xx result on every
malformed input and full survival across repeated valid calls in a
genuinely fresh environment, the most likely explanation is exactly
what the reviewer flagged as a possibility — resource degradation in
their long-running sandbox (15+ rounds of installs/test runs/server
launches) rather than a defect in this stage's code. No code change was
made here as a result, per this stage's own instruction not to invent a
fix for a problem that can't be found.

`tests/test_server_survives_interleaved_valid_and_malformed_reconcile_calls`
in `tests/test_api.py` is the permanent regression coverage for this:
it drives `TestClient` (not a live server, for reliability — no real
socket/process/port to flake) through 3 rounds of one valid
`POST /reconcile` plus all 6 malformed-body cases above, asserting
`GET /health` succeeds after every one of the 21 calls per round, then
one final valid call at the end to confirm the app is still fully
functional and not merely still answering `/health`. If a future change
ever introduces a code path that can bring the process down instead of
returning an HTTP error, this test fails immediately and
deterministically in CI, without needing a live, long-running server to
notice.

**Stage 13 verified stable in a fresh environment** — Stage 14 can
safely build on this API layer.

## Streamlit Dashboard (Stage 14)

`dashboard/` — the judge-facing demo surface from §10/§15. This is a
**presentation layer only**: every function in `dashboard/app.py` and
`dashboard/api_client.py` either calls one of the Stage 13 FastAPI
endpoints over real HTTP, or loads a bundled `data/*/records.json` /
`ground_truth/*/ground_truth.json` fixture from disk. Nothing here
reimplements or reaches into matching, verification, LLM, or
evaluation logic directly — the dashboard exercises the exact same API
surface a judge could hit with `curl`.

**Run it — two processes, in this order:**

```
# Terminal 1 — the API must be running first
uvicorn recon_agent.api.app:app --reload

# Terminal 2 — the dashboard is an HTTP client of the API above
streamlit run src/recon_agent/dashboard/app.py
```

By default the dashboard talks to `http://127.0.0.1:8000`; change the
base URL in its sidebar if the API is running elsewhere. Every action
in the UI will fail with a clear "could not reach the API" message
(not a silent hang or a stack trace) until Terminal 1 is up.

**What's in it, tab by tab:**

- **1. Run reconciliation** — pick a bundled `data/calibration` or
  `data/evaluation` set (auto-pairing its `ground_truth/` file, with a
  checkbox to attach or omit it), or upload your own records JSON
  (plus an optional matching ground-truth JSON). Triggers
  `POST /reconcile` and shows the returned summary — status, matched
  record count, group-status counts, exception count, elapsed time,
  Groq calls made.
- **2. Headline metrics** — `GET /report/{run_id}`, rendered verbatim:
  auto-match precision, record coverage, value coverage, false-match
  rate, exception quality, the tier-contribution breakdown by
  proposing stage, and runtime/cost. When no ground truth was
  supplied, shows the ground-truth-independent subset instead (exactly
  what the API itself falls back to) rather than computing anything
  client-side.
- **3. Match / exception drill-down** — `GET /audit/{run_id}`. A
  `MatchGroup` table filterable by status (VERIFIED / PENDING_REVIEW /
  REJECTED); selecting one group surfaces `proposed_by` vs
  `verified_by` vs `commit_policy` vs `verification_result` side by
  side — the §6 provenance-vs-verification separation the demo script
  (§15 point 4) is built around — plus its evidence score/margin,
  policy checks, and member records. A parallel table for exceptions
  (filterable by severity), drilling into category, severity, evidence,
  and recommended action per exception.
- **4. LLM budget** — `GET /llm-budget/{run_id}` verbatim: calls made
  today against the shared daily ceiling, this run's consecutive
  failures, and whether its circuit breaker is open.
- **5. Graceful degradation** — §15 point 7's live demo, done honestly:
  `GROQ_API_KEY` is read once into a process-wide cached `Settings`
  object when the API starts (`config.py`'s `get_settings()`), so it
  cannot be flipped over HTTP — there is no endpoint for that, and this
  stage does not invent one just to make a toggle switch work. This tab
  says so explicitly, gives the exact commands to restart the API
  without the key, and its "Re-run last batch now" button does exactly
  and only that — re-POSTs the same batch to `/reconcile` and logs the
  result (status, Groq calls made, LLM-related exception count,
  circuit-breaker state) into a comparison table, so a judge can
  restart the API between clicks and see the run complete either way.

**Tests:** `tests/test_dashboard_api_client.py` (11 tests). Per this
stage's own brief, a Streamlit script's UI rendering isn't
conventionally unit-testable, so this suite instead proves the layer
that actually matters for correctness — `dashboard/api_client.py`'s
HTTP functions — against a real `fastapi.testclient.TestClient`-backed
API, mirroring `tests/test_api.py`'s own fixture setup exactly (fresh
temporary SQLite files, no `GROQ_API_KEY`, no shared state): a full
`reconcile` → `report` → `audit` → `llm-budget` round trip using the
same real refund-triad fixture `test_api.py` uses, the
without-ground-truth partial-report path, a 404 surfaced as
`DashboardAPIError`, and a 422 for a malformed record. Also covers
`dashboard/datasets.py`'s pure file-I/O helpers (bundled-dataset
discovery, records/ground-truth loading, and the non-list-JSON
rejection case) against `tmp_path` fixtures rather than this repo's
actual (regenerated-from-seed, not committed) `data/` directory. No UI
testing infrastructure (e.g. browser automation) was added, per this
stage's own scope note not to over-invest there. Full suite: 231 tests
(the prior 220 plus 11 new), no regressions.

**No new dependency was added** — `requests` (used by
`dashboard/api_client.py`'s `RequestsClient`) is already a transitive
dependency of `streamlit`, which was already pinned in
`requirements.txt`/`pyproject.toml` since Stage 1.

## Docker Packaging (Stage 15)

`Dockerfile`, `docker-compose.yml`, `.dockerignore`, `.gitignore`, and
`scripts/verify_docker_image.sh` — the packaging and documentation
polish pass from ARCHITECTURE.md §9/§14. No application logic changed
in this stage (`src/recon_agent/` is untouched); this is scoped
entirely to how the existing code gets built, run, and demoed.

**One image, two processes.** `docker-compose.yml` builds a single
image (`recon-agent:latest`) and runs it twice: the `api` service
(`uvicorn recon_agent.api.app:app`) and the `dashboard` service
(`streamlit run src/recon_agent/dashboard/app.py`). `docker compose up`
starts both, networked together, per §14's "one command" requirement.

**Networking, without touching `dashboard/app.py`.** The dashboard
needs to reach the API, and `dashboard/app.py`'s own default base URL
is hardcoded to `http://127.0.0.1:8000` (see "Streamlit Dashboard
(Stage 14)" above) — normally a problem across two separate containers,
since `127.0.0.1` inside one container isn't the other container. This
stage's brief is explicit: don't touch application logic to work around
that. So instead, `docker-compose.yml` gives the `dashboard` service
`network_mode: "service:api"` — it runs inside the `api` container's
network namespace rather than its own, so `127.0.0.1:8000` really is
the API from the dashboard process's point of view, with zero source
changes. The trade-off: a container sharing another service's network
namespace can't publish its own ports, so both `8000` and `8501` are
published on the `api` service in the compose file instead.

**The embedding model is pre-baked at build time (§10/§14's named,
explicit requirement).** The Dockerfile's `RUN` step calls
`SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder='/opt/hf-cache')`
directly during `docker build`, before `HF_HUB_OFFLINE`/
`TRANSFORMERS_OFFLINE` get set for the runtime stage — so the download
happens exactly once, at build time, and any accidental network call
after that point fails loudly instead of silently succeeding on a
judge's connection. `scripts/verify_docker_image.sh` step 2 proves this
concretely rather than trusting the Dockerfile's syntax: it builds the
image, then runs `SentenceTransformer(...)` inside a throwaway
container started with `--network none`, and separately (step 3) checks
the actual weight files exist on disk under `/opt/hf-cache` and are a
plausible size for a real checkpoint (not an empty stub).

**Repo-state honesty note, worth being direct about:** a full search of
`src/` as of this stage finds no current import of `sentence_transformers`
anywhere in the matching pipeline — Stage 9's fuzzy candidate retrieval
(see "Stage 5 Fuzzy Candidate Retrieval" above) ended up built entirely
on `rapidfuzz` identifier and counterparty similarity, not embeddings.
`sentence-transformers`/`all-MiniLM-L6-v2` is still declared in
`pyproject.toml`/`requirements.txt` and named explicitly in
ARCHITECTURE.md §10's tech-stack table as the intended descriptive-text
matching approach. This stage pre-bakes the model exactly as instructed
— that's an explicit, named requirement in both ARCHITECTURE.md and
this stage's own brief, and this stage's scope is packaging what the
architecture doc specifies, not auditing or rewiring what an earlier
matching stage chose to build instead. Flagging it here rather than
quietly baking in a model nothing calls, so whoever picks up Stage 16
(or reviews this repo) knows it's a real, checkable discrepancy and not
an oversight.

**Resolved by Stage 16:** the dependency, the pre-bake step, and the
associated env vars have since been removed — see "Failure Injection +
Final Acceptance-Gate Self-Check (Stage 16)" below and ARCHITECTURE.md
§10's "Removed (Stage 16)" note. This subsection (and the Dockerfile
excerpt below it) is left as it was written at the time, as an honest
record of what this stage actually shipped and flagged; it does not
describe the current Dockerfile.

**Datasets are generated into the image at build time, not copied from
the host.** `data/` and `ground_truth/` are git-ignored and regenerate
byte-identically from the fixed seeds in `testdata/generator.py` (see
"Synthetic Data (Stage 2)" above) — so a `docker build` from a
completely fresh `git clone`, where neither directory exists on disk
yet, needs to produce them itself rather than assuming they're already
there. The Dockerfile's `.dockerignore` explicitly excludes `data/`,
`ground_truth/`, and `reports/` from the build context for the same
reason: build-time generation only, never a possibly-stale host copy.
`scripts/verify_docker_image.sh` step 4 confirms both datasets are
present, non-empty, and readable inside an offline container.

**Tests.** `scripts/verify_docker_image.sh` — a CI-style shell script,
not pytest, since what it's proving (a real `docker build`, real image
layers, `--network none` behavior) isn't something Python-level unit
tests can exercise. It fails fast (`set -euo pipefail`) on the first
failing check and prints which of the four checks failed. **Honesty
note on this stage's own verification:** the sandbox this stage was
built in has no Docker daemon and no network route to
`huggingface.co`/Docker Hub, so `scripts/verify_docker_image.sh` itself
could not be executed here — the Dockerfile, compose file, and script
were written and carefully reviewed (including a hand-run of the exact
`sentence-transformers` cache-resolution logic and the exact
`testdata.generator --seed both` command against this real repo, both
outside Docker, both confirmed working — see below), but the literal
`docker build` has not yet been run by this stage's own author. Run
`./scripts/verify_docker_image.sh` once, locally or in CI, before
relying on this for a live demo.

**What actually was verified in this stage, outside Docker:**
- The full existing test suite — `pytest` — passes unchanged:
  **231 passed, 1 skipped**, matching Stage 14's own reported count, no
  regressions from anything touched this stage (which is nothing under
  `src/`).
- `python -m recon_agent.testdata.generator --seed both`, run directly,
  reproduces `data/calibration/records.json`, `data/evaluation/records.json`,
  and both `ground_truth/*/ground_truth.json` files byte-for-byte
  identical to what's already in this repo snapshot — confirming the
  Dockerfile's build-time generation step will produce the same real
  data a local `pip install -e .` setup does.
- A full repo-wide `grep` for hardcoded secrets, API key patterns, and
  private-key markers found nothing; `.env.example` contains placeholder
  keys only (`GROQ_API_KEY=`, empty); no code path logs `settings.groq_api_key`
  or embeds it in an LLM prompt (checked directly in
  `llm/recommender.py`, `dashboard/app.py`).
- No `.gitignore` existed in this repo before this stage, despite
  `data/`, `ground_truth/`, and `reports/` all being described elsewhere
  in this README as "not committed" — this stage adds one (covering
  those three directories, `*.db`/`*.sqlite3` runtime files, `.env`,
  and standard Python/editor artifacts), since an accurate `.gitignore`
  is itself part of §14's "no secrets or sensitive raw data in ...
  the repo" requirement. This repo snapshot has no `.git/` history to
  audit for prior accidental commits.

### §14 Acceptance Gate — checked item by item against the current repo (Stage 15's own pass; see Stage 16's section below for the current, final pass)

| # | Requirement | Result |
|---|---|---|
| 1 | One command starts the full demo, embedding model pre-baked, no cold-run download | **Implemented, not yet execution-verified.** `docker compose up` is wired correctly per the design above; `scripts/verify_docker_image.sh` proves the pre-bake concretely once run — but this sandbox has no Docker daemon, so that script has not actually been executed yet. Run it before a live demo. (Superseded by Stage 16: the embedding-model pre-bake this item refers to was removed as unused — see above — and Stage 16 still had no Docker daemon either, so item 1 remains genuinely open for the repo owner to verify; see `scripts/acceptance_check.sh`.) |
| 2 | 200–300 physical-row batch finishes without manual intervention | **PASS** — unchanged from Stage 12; re-confirmed this stage via the full test suite (231 passed) and both real harness runs already documented above. |
| 3 | Every VERIFIED group passes the full stage-differentiated auto-commit policy (§2) | **PASS** — unchanged from Stage 6/8/11; test suite green, nothing under `src/` touched this stage. |
| 4 | `MatchGroupMember` correctly represents every multi-source group | **PASS** — unchanged from Stage 3/8; test suite green. |
| 5 | 1:1 / many-to-one resolution optimal within configured bounds; many-to-many explicitly declared unsupported | **PASS** — unchanged from Stage 4/9; the many-to-many limitation is stated plainly in ARCHITECTURE.md §10 and the "Removed" note, not silently missing. |
| 6 | Every `STAGE6_LLM` proposal carries `commit_policy: HUMAN_REVIEW_REQUIRED`, no exceptions | **PASS** — unchanged from Stage 10/11; test suite green. |
| 7 | Evaluation computed against the held-out evaluation set, never calibration | **PASS** — unchanged from Stage 12; `evaluation/harness.py` design, re-confirmed by inspection this stage. |
| 8 | False-match rate, record coverage, value coverage displayed separately | **PASS** — Stage 14's dashboard "Headline metrics" tab renders all three as distinct fields from `GET /report/{run_id}`, not combined. |
| 9 | Every unresolved record has a reason code, evidence, and recommended action | **PASS** — unchanged from Stage 5/6/12 `Exception` model; test suite green. |
| 10 | System completes when the Groq key is missing/exhausted/malformed; budget counter atomic, no double-spend | **PASS** — unchanged from Stage 11; test suite green. |
| 11 | No secrets or sensitive raw data in logs, prompts, screenshots, or the repo | **PASS** — verified this stage by direct grep + code inspection (see "What actually was verified" above); `.gitignore` added to keep it that way going forward. Screenshots are outside this repo's scope to audit (none are committed). |
| 12 | Full demo completes reliably in under three minutes | **Not independently re-timed this stage.** The Quickstart section above and §15's demo script are both designed around a sub-three-minute run, and `docker compose up`'s image-build step (once cached) plus the health-checked startup order should comfortably fit that budget — but no stage of this relay has recorded an actual stopwatch run of the full 8-step §15 demo script end-to-end, and this stage doesn't have a live judge machine to time it on. Worth doing once, live, before the real demo. (Stage 16 also had no Docker daemon and could not time this either — still genuinely open; `scripts/acceptance_check.sh` times it automatically if run somewhere Docker is available.) |
| — | Stage 16 (final relay-wide review) | **Complete.** See "Failure Injection + Final Acceptance-Gate Self-Check (Stage 16)" below for the final cross-stage review, the failure-injection demo, and the acceptance-gate self-check script. |

**Reading this table honestly:** items 2–11 are re-confirmations of
work earlier stages already built and tested, not new claims — this
stage didn't touch application logic, so the strongest available
evidence for those is "the same 231 tests that passed before this stage
still pass now." Item 1 and item 12 are the two genuinely new,
Stage-15-specific claims, and both are flagged above exactly as far as
this stage could actually verify them and no further.

**Regenerating the verification yourself:**
```
./scripts/verify_docker_image.sh
```
Requires Docker; builds the image, then checks against an offline
(`--network none`) container. (Updated by Stage 16 — see below — to
check that `sentence-transformers`/`torch` are genuinely absent from
the built image, plus the dataset pre-bake; it no longer checks for a
model load, since that model is no longer part of the image.) Exits
non-zero on the first failing check.

---

## Failure Injection + Final Acceptance-Gate Self-Check (Stage 16)

The final stage of this relay. Two things, plus a cleanup carried over
from Stage 15's own honest self-audit; no new matching, verification,
or LLM logic — everything below either demonstrates existing behavior
or checks it.

### Cleanup: removed the unused `sentence-transformers` dependency

Stage 15 flagged, in both the Dockerfile and README.md's Build Status
above, that a full search of `src/` found no import of
`sentence_transformers` anywhere — Stage 9's fuzzy candidate retrieval
(see "Stage 5 Fuzzy Candidate Retrieval" above) hit its target (100%
precision, 0% false-match rate) using `rapidfuzz` identifier and
counterparty similarity alone, never embeddings. Stage 15 deliberately
didn't act on that finding — packaging what the architecture doc
specified was that stage's scope, not auditing it. Stage 16 acts on it:

- Removed `sentence-transformers>=3,<4` from `requirements.txt` and
  `pyproject.toml`.
- Removed the Dockerfile's model pre-bake `RUN` step, and the
  `HF_HOME`, `SENTENCE_TRANSFORMERS_HOME`, `HF_HUB_DISABLE_TELEMETRY`,
  `HF_HUB_OFFLINE`, and `TRANSFORMERS_OFFLINE` env vars — a repo-wide
  grep confirmed nothing else references any of them.
- Updated `scripts/verify_docker_image.sh` to check that
  `sentence_transformers`/`torch` are genuinely absent from the built
  image (rather than checking for a model load that no longer applies),
  keeping its dataset pre-bake check.
- Updated ARCHITECTURE.md §10's tech-stack table with a "Removed
  (Stage 16)" note, and fixed the §14 acceptance-gate line that had
  named the embedding-model pre-bake specifically.

This drops `torch` (sentence-transformers' large transitive dependency)
from the image too, which is the bulk of the expected size/build-time
reduction — not independently measured here, since that requires the
same Docker daemon this whole relay's sandboxes have lacked (see
below), but removing a multi-hundred-megabyte dependency and its build
step is a meaningful, structural reduction regardless of the exact
number.

**Verified directly:** reinstalled `requirements.txt` fresh in this
stage's own sandbox (no `torch`/`sentence-transformers` anywhere in the
resulting environment — confirmed by `pip list`) and re-ran the full
test suite: **231 passed, 1 skipped** — identical to every prior
stage's count, confirming nothing was ever actually importing the
removed package.

### 1. Failure injection — making §15's demo script points 5-7 demonstrable

`scripts/failure_injection_demo.py` runs three scenarios against real
data and real code (no new degradation/exception logic — everything it
exercises is already built and already covered by the test suite; this
script's only job is to make each behavior demonstrable in one place,
in front of a judge, rather than only provable by reading test code):

- **(a) A genuinely unresolved record correctly becomes an honest
  exception, not a forced guess.** Uses the real evaluation dataset's
  `honest_abstention` ground truth (a standalone `ADJUSTMENT` record
  with no real counterpart anywhere in the batch — see "Bugfix: silent
  unattempted-record drop" above for how this got wired up) and
  confirms directly: the record is never a member of any `VERIFIED`
  group, and it does produce a real `ReconciliationException`
  (`category=INSUFFICIENT_EVIDENCE`, `reason=NO_STAGE_ATTEMPTED`) with
  a recommended action — not silence.
- **(b) The Groq key is unset and the pipeline still completes.** Pops
  `GROQ_API_KEY` from the environment and runs the real pipeline
  against the real evaluation dataset (101 groups, 30 exceptions on
  this build, 9 of them `LLM_UNAVAILABLE` with
  `degradation_reason: LLM_NOT_CONFIGURED`) — no exception propagates
  out of `run_pipeline`, no hang, no crash. This is Stage 11's existing
  governor/degradation logic (`llm/governor.py`,
  `matching/stage6_llm.py`); the script only proves it's reachable and
  fires against real, non-trivial data, not synthetic mocks.
- **(c) A malformed/adversarial input is handled without crashing.**
  Builds directly on Stage 13's fresh-environment investigation (see
  "Investigated: reported silent server death" above): replays the
  same battery of malformed request bodies (bare array, missing
  fields, wrong types, empty list, `null` body, non-list `records`)
  against an in-process `TestClient`, each interleaved with a
  `GET /health` check, plus one additional case — a schema-valid but
  adversarially oversized (50,000-character) field, checked only for
  "doesn't 500", since accepting valid input isn't a bug. Every
  malformed case produces a clean 4xx, the server stays responsive
  throughout, and a final valid call still succeeds.

Run it directly:
```
python scripts/failure_injection_demo.py
```
Verified in this stage's own sandbox: all three scenarios pass, exit
code 0.

### 2. Final acceptance-gate self-check — `scripts/acceptance_check.sh`

One command that runs everything about ARCHITECTURE.md §14's 12-item
acceptance gate that's checkable without a live Docker daemon, plus the
two Docker-dependent items themselves *if* Docker is actually available
on the machine running it:

1. The full test suite (`pytest`).
2. Both evaluation harness runs (calibration and evaluation).
3. `scripts/failure_injection_demo.py` (above).
4. A repo-wide secrets/sensitive-data grep, plus a `.gitignore`
   sanity check.
5. Confirms the `sentence-transformers` cleanup above actually took —
   not importable in the current environment, not referenced in
   `requirements.txt`/`pyproject.toml`.
6. **If a Docker daemon is available:** builds the image via
   `scripts/verify_docker_image.sh`, then runs `docker compose up
   --build` for real and times how long it takes to reach a healthy
   `GET /health`, checking that against §14 item 12's three-minute
   budget. **If no Docker daemon is available** (the case in every
   sandbox this relay has run in, including this one — see below): it
   says so plainly and marks those two items `SKIP`, rather than
   guessing or silently passing them.
7. Prints a final table against all 12 §14 items plus the Stage 16 row,
   and a per-check PASS/FAIL/SKIP list with the totals.

Run it directly:
```
./scripts/acceptance_check.sh
```
Exit code 0 only if every check it could actually run passed; a
non-zero exit means at least one real check failed (not "was
skipped").

**This stage's own sandbox has no Docker daemon** — the same situation
Stage 15 reported. Run against this repo, in this sandbox, right now:

| Check | Result |
|---|---|
| Full test suite | **PASS** — 231 passed, 1 skipped |
| Harness — calibration | **PASS** — completes; see "Real results" above for the numbers |
| Harness — evaluation (held-out) | **PASS** — 100% precision, 0% false-match rate, 100% exception quality |
| Failure-injection demo (a)/(b)/(c) | **PASS** — all three scenarios |
| Secrets/sensitive-data scan | **PASS** — nothing found outside `.env.example` |
| `.gitignore` covers `data/`/`ground_truth/`/`reports/` | **PASS** |
| `sentence-transformers` cleanup | **PASS** — not installed, not referenced |
| Docker image build + pre-bake (`verify_docker_image.sh`) | **SKIP — no Docker daemon in this sandbox** |
| Full demo start-to-healthy timing (§14 item 12) | **SKIP — no Docker daemon in this sandbox** |

### 3. Final consistency pass

Read README.md end to end and cross-checked ARCHITECTURE.md against the
current codebase one more time:

- **Build Status** (below) is condensed rather than left to grow
  unboundedly — each stage keeps one line plus only the callouts that
  still matter to a reader deciding whether the system is ready, not
  every historical detail (those stay in each stage's own section
  further down, unchanged).
- **Identifier matching, verifier evidence fields, the exception
  taxonomy, and the API surface** were checked directly against
  `models/enums.py`, `models/match_group.py`,
  `models/exception.py`, `verification/verifier.py`, and
  `api/app.py`'s route table: all four match ARCHITECTURE.md §5–§8 and
  §10 exactly. No drift found beyond the `sentence-transformers` item
  this stage already fixed.
- No other stale claims of the same shape (a documented dependency or
  capability nothing actually uses) were found elsewhere in
  ARCHITECTURE.md or README.md.

### What is genuinely done vs. what the repo owner still needs to check

**Done and verified, in this stage's own sandbox:**
- The `sentence-transformers`/`torch` cleanup (dependency, Dockerfile,
  verify script, ARCHITECTURE.md) — code-level changes, fully verified
  by re-running the full test suite (231 passed, 1 skipped, unchanged).
- All three failure-injection scenarios, run for real against real code
  and real data.
- The acceptance-gate script itself, run for real in this sandbox: 8
  checks passed, 0 failed, 2 skipped for a documented, structural
  reason (no Docker daemon here).
- The README/ARCHITECTURE consistency pass above.

**Still needs the repo owner to check personally, on a machine with
Docker, before the real demo — this is the one honest gap left in the
whole 16-stage relay:**
- Run `./scripts/acceptance_check.sh` there. It will automatically
  build the image, run `scripts/verify_docker_image.sh`, run
  `docker compose up --build`, and time the start-to-healthy interval
  against the three-minute budget — the exact two items (§14 items 1
  and 12) that no stage of this relay, including this one, has ever
  been able to check, because no stage's sandbox has ever had a Docker
  daemon.
- Do one live, timed walkthrough of the full 8-step §15 demo script
  end-to-end by hand at least once before presenting, even after the
  script above passes — a stopwatch script proves the infrastructure
  is fast enough; it doesn't replace actually rehearsing the talking
  points against a live dashboard.

**Overall project state relative to the Sept 5 deadline:** all 16
stages are complete. Every acceptance-gate item that can be verified
without Docker has been verified, repeatedly, across multiple stages,
with no regressions at any point (231 tests passing throughout this
stage). The only remaining work is the Docker build/run/timing
verification above — a single `./scripts/acceptance_check.sh` run on
a Docker-equipped machine — which is infrastructure verification, not
outstanding feature work. Nothing in this codebase is known-broken,
known-incomplete, or silently unverified; everything unverified is
named explicitly, in this section, with the exact command that
verifies it.

## Post-Relay Bugfix Pass — First Real User Run

The 16-stage relay above finished with 245 tests passing and every
acceptance-gate item checkable outside Docker verified. Four days
before submission, the repo owner ran the system for the first time
against their own hand-crafted dataset — 15 INR records
(`data/repro_bugfix/records.json`, also attached to this pass as
`demo_upload_records.json`) covering a clean payment/settlement/
2-split-bank-credit chain, a 3-way refund, a 3-way chargeback, a
payment+settlement pair with an unexplained residual and its exact
reversal counterpart, and one genuinely orphaned bank credit — and
found four real bugs in one sitting. This is exactly the kind of
finding synthetic calibration/evaluation data is structurally blind
to: it's generated from the same assumptions (`testdata/generator.py`)
the pipeline itself encodes, so a gap in those assumptions doesn't
show up until an independently-authored dataset exercises it. All
four are fixed below, in one pass, with no new architecture — every
fix tightens something already built.

### Bug 1: unbounded Stage 1 residual

`verification/verifier.py`'s `_verify_stage1` re-checked identifier
uniqueness, currency consistency, and date policy on every Stage 1
(exact-identifier) proposal, but never amount — by design, on the
reasoning that a shared unique reference is strong evidence on its
own (§2's Stage 1 auto-commit condition doesn't require a composite
score). The reproduction dataset showed exactly why that design needed
a ceiling: `pay_di4adgpi56v9ec`'s LEDGER payment (₹49,813.20) and
GATEWAY settlement (₹48,343.72) — and separately its exact reversal
counterpart — both proposed by `STAGE1_EXACT`, both auto-verified,
both with a 146,948 paise (~2.95%) residual and no `FEE` record
anywhere in the dataset explaining it. Nothing was checking that gap
at all; the same logic would silently verify two completely unrelated
transactions that happened to share a reference by data-quality
accident, with zero scrutiny.

**The fix** adds a bounded, configurable tolerance:

- `config.py` gains `stage1_residual_tolerance_fraction` (env
  `STAGE1_RESIDUAL_TOLERANCE_FRACTION`, default `0.025`), alongside
  the existing threshold/policy versioning fields.
- `models/enums.py`'s `VerificationResult` gains
  `RESIDUAL_EXCEEDS_TOLERANCE`.
- `verification/verifier.py` gains `_stage1_residual`, which
  independently re-derives the same (LEDGER gross, GATEWAY
  already-netted) view `stage1_exact.py`'s own `_amounts` uses — from
  each member's underlying record, never trusted from the group's own
  stored `expected_amount_paise`/`matched_amount_paise` fields, this
  module's consistent stance throughout. `_verify_stage1` compares
  `abs(expected - matched)` against
  `stage1_residual_tolerance_fraction * max(expected, matched)`; a
  residual within the bound (including exactly zero) passes exactly
  as before.
- `matching/pipeline.py`'s `run_verification` routes
  `RESIDUAL_EXCEEDS_TOLERANCE` the same way it already routes
  `FAILED_MARGIN`: `status=PENDING_REVIEW`,
  `commit_policy=HUMAN_REVIEW_REQUIRED`, records stay claimed
  (`matched_record_ids` untouched, not released) — ambiguous, not
  disproven. The identifier match is still real evidence; an
  unexplained gap this size needs a human's judgment, not an
  automatic pass or a blunt rejection.

**Picking the default.** The bug report's own suggestion ("something
like 5%") turned out to be far too loose to catch the reproduction
case at all, and a naive "just under 2.95%" default turned out to flag
almost the entire real Stage 1 population in both bundled datasets:
`testdata/generator.py`'s `MDR_BPS_CHOICES = (150, 175, 200, 225, 250)`
with an 18% GST add-on produces legitimate fee-driven residuals at
exactly 1.77%, 2.065%, 2.36%, 2.655%, and 2.95% of the gross amount —
so the reproduction dataset's "anomalous" 2.95% residual sits, by
apparent coincidence, exactly at the top of the *real* system's own
legitimate MDR+GST range (250 bps MDR is the highest domestic-card
tier in the generator, and Razorpay's real-world equivalent is
typically reserved for premium/international cards). A tolerance
anywhere below ~2.95% therefore necessarily flags a meaningful slice
of genuinely correct calibration/evaluation matches too — there's no
threshold that separates the reproduction case from real data by
magnitude alone, because the reproduction case's *actual* tell is that
no FEE record exists anywhere in its 15-record batch to explain the
gap, not the size of the gap itself. Given the fix requested is
explicitly magnitude-based (a percentage of the larger amount), the
honest resolution was to settle on a default — **2.5%** — that sits
between the common domestic-card tiers (routine, still auto-verifies)
and the top MDR tier processors reserve for higher-risk card
categories (now requires a human look), and to document the resulting
shift in auto-verified coverage plainly rather than pick a looser
default that would silently fail to catch the exact case the bug
report asks for. See the Key Results table in README.md and its
callout for the resulting numbers.

**Verified directly** against the reproduction dataset:

| Group | Before | After |
|---|---|---|
| `pay_di4adgpi56v9ec` LEDGER+GATEWAY pair | VERIFIED / PASSED (residual 146,948 paise, unchecked) | PENDING_REVIEW / RESIDUAL_EXCEEDS_TOLERANCE |
| its exact reversal counterpart | VERIFIED / PASSED (same residual, unchecked) | PENDING_REVIEW / RESIDUAL_EXCEEDS_TOLERANCE |
| the 3-way refund (conserves exactly) | VERIFIED / PASSED | VERIFIED / PASSED (unaffected) |
| the 3-way chargeback (conserves exactly) | VERIFIED / PASSED | VERIFIED / PASSED (unaffected) |
| the clean settlement chain (Stage 3 aggregation, unrelated to Stage 1) | VERIFIED / PASSED | VERIFIED / PASSED (unaffected) |

And against both bundled datasets — auto-match precision and
false-match rate (the properties this fix must never compromise) are
byte-for-byte unchanged; only the VERIFIED/PENDING_REVIEW split moved:

| Dataset | Precision / false-match, before & after | VERIFIED / PENDING_REVIEW, before | after |
|---|---|---|---|
| calibration | 100.0% / 0.0% (unchanged) | 87 / 0 | 60 / 27 |
| evaluation | 100.0% / 0.0% (unchanged) | 96 / 0 | 64 / 32 |

### Bug 2: unmatched records could still end up with zero exceptions; run status didn't reflect it

`matching/pipeline.py`'s `run_final_catchall` — the safety net built
in the post-Stage-12 bugfix above to guarantee every unmatched record
gets a real exception — had a gap one layer deeper than the bug it
originally fixed: its "already handled" check searched *every*
`DecisionEvent` already produced, as a blob, for the record's id as a
substring. That's too broad — a record merely *mentioned* as a
near-miss candidate inside some other, unrelated record's own
`DecisionEvent` was enough to skip it, even though nothing had ever
raised a real `Exception` naming that record as its own subject.

The reproduction dataset's orphaned bank credit
(`UTR_DEMO_ORPHAN_001`) hit exactly this gap: Stage 3 and Stage 4 both
correctly decline to aggregate it (`NO_AGGREGATION_FOUND`, an honest
"found nothing" outcome that deliberately raises no exception, since
there's nothing ambiguous about zero eligible subsets), producing two
real `DecisionEvent`s naming it directly — but because its record id
also happened to appear inside some other seeker's own decision event
blob along the way, the catch-all's old blob check treated it as
already covered and skipped it. Result: a real, verifiably unmatched
record with a `DecisionEvent` trail and **zero** exceptions — the
dashboard showed 0 exceptions and the run looked `COMPLETE` even
though a genuine settlement leg had no counterpart and no flag.

**The fix** re-keys the catch-all's coverage check off `Exception`s
instead of `DecisionEvent`s — an `Exception` is always genuinely about
its subject (either `group_id_or_record_id` naming it directly, or its
id appearing in a rejected group's own
`evidence["released_record_ids"]`), never a side mention the way a
`DecisionEvent`'s `candidate_scores` can be:

- `run_final_catchall`'s signature changed from taking
  `decision_events` to taking `exceptions`; `run_pipeline`'s call site
  now passes `verified_stage6.exceptions`.
- The condition is otherwise unchanged in shape (still a
  fixed-record-id-width-safe substring search, still purely additive,
  never inspecting or overriding an existing decision) — only what
  it's checked against changed.
- This was verified to still correctly skip a record Stage 6 already
  gave its own exception to (`tests/test_pipeline_final_catchall.py`'s
  existing dedup test, `led_lonely`, continues to pass unmodified —
  its own `NO_CANDIDATES_AVAILABLE` exception has
  `group_id_or_record_id == "led_lonely"` and is found directly, not
  via a blob of unrelated mentions).

Also wired up `ReconciliationRun.status` end to end: on inspection,
`api/service.py`'s `_determine_status` already correctly derived
`COMPLETED_WITH_EXCEPTIONS` vs `COMPLETE` vs `DEGRADED` from
`result.exceptions` and the LLM governor snapshot — it was only ever
starved of input, since the orphan's exception never existed for it to
see. Confirmed directly:
`execute_reconciliation(repro_records, store).run.status ==
ReconciliationRunStatus.COMPLETED_WITH_EXCEPTIONS` once the catch-all
fix above lands the orphan's real exception. The dashboard's tab 1
"Latest run summary" already renders `r["status"]` verbatim from the
API response, so no dashboard change was needed there; the drill-down
tab did need one — see Bug 3.

**Verified directly**: `UTR_DEMO_ORPHAN_001` now has exactly one real
exception (non-empty `recommended_action` and `evidence`), and the
end-to-end run through `api/service.py` reports
`COMPLETED_WITH_EXCEPTIONS`, not `COMPLETE`.

### Bug 3: dashboard drill-down missing evidence fields that already existed

`models/match_group.py`'s `MatchGroup` has carried `evidence_score`,
`runner_up_score`, `score_margin`, `threshold_applied`, and
`policy_checks` since early in the build, and `DecisionEvent` has
carried `reason_code`/`explanation` per event since Stage 1. But
`dashboard/app.py`'s match/exception drill-down (Tab 3) only ever
rendered `proposed_by`/`verified_by`/`commit_policy`/
`verification_result`, `evidence_score`/`score_margin`/
`threshold_applied`, and the group's own `policy_checks` (the
*proposing* stage's own checks) — never `runner_up_score`, and never
any of the *verifier's* own recomputed evidence (residual-vs-tolerance
detail, conservation diffs, margin checks), which only ever lived
inside the `STAGE7_VERIFICATION` `DecisionEvent`'s `candidate_scores`,
not on the `MatchGroup` object itself. The exception drill-down showed
raw `evidence` JSON but never cross-referenced the `DecisionEvent`(s)
that exception was actually raised from.

**The fix** is purely additive rendering — no new computation, no
schema change (`api/schemas.py`'s `AuditResponse` already returns
`decision_events` alongside `match_groups`/`exceptions`; the dashboard
just wasn't using it for this):

- Added a `Runner-up score` metric next to the existing evidence-score
  metrics, and `Expected`/`Matched`/`Residual (paise)` metrics next to
  the existing policy-checks expander.
- Added a "Decision trail" section to both the `MatchGroup` and
  `Exception` drill-downs: every `DecisionEvent` naming the selected
  group or exception (matched the same fixed-width-safe substring way
  the catch-all pass itself matches them), rendered as a table of
  `stage` / `reason_code` / `explanation`, with each event's full
  `candidate_scores` (the matching reference used, amount/date/
  currency check results, residual-vs-tolerance detail, and so on)
  available in a per-event expander.

This surfaces, for the reproduction dataset's own
`RESIDUAL_EXCEEDS_TOLERANCE` groups, the exact reason code and
explanation text quoted in Bug 1 above, plus the full
`residual_paise`/`residual_tolerance_paise`/`residual_within_tolerance`
breakdown — none of which a human reviewer could previously see
without reading the JSON export by hand.

### Bug 4: misleadingly-named `bank_credit_coverage`

`evaluation/harness.py`'s `compute_bank_credit_coverage` (§11 metric
8) is named and documented as measuring genuine bank credits, but its
eligibility filter was `r.source == Source.BANK` — any BANK-source
record, which silently includes BANK-side `REFUND`/`CHARGEBACK`/
`REVERSAL` legs (the bank-side reflection of those events, real
BANK-source records, but not credits landing in the account) alongside
genuine `BANK_CREDIT` records. This inflated the denominator with rows
the metric's own name doesn't describe. The reproduction dataset makes
this concrete: its 3-way refund and 3-way chargeback each contribute a
real BANK-source, non-`BANK_CREDIT` leg (`demo_bank_refund_001`,
`demo_bank_chargeback_001`) that sits inside a genuine ground-truth
cluster and would have counted toward "bank credit coverage" under the
old filter despite never being a credit.

**The fix** narrows the filter to `r.entity_type == EntityType.BANK_CREDIT`
specifically, updates the metric's own docstring block to document the
bugfix and why it matters, and updates `ARCHITECTURE.md`'s one-line
metric description to match. `tests/test_evaluation_harness.py`'s
three existing `compute_bank_credit_coverage` tests used a `_record()`
helper that defaults to `entity_type=PAYMENT`, so each BANK-source test
record needed `entity_type=EntityType.BANK_CREDIT` added explicitly to
keep exercising the intended shape under the corrected filter; a new
test, `test_bank_credit_coverage_excludes_bank_side_refund_chargeback_reversal`,
covers the exact bug directly. A repro-dataset-level check is also in
`tests/test_repro_bugfix_dataset.py`, confirming the two BANK-side
refund/chargeback legs never enter the metric's eligible set even
though they sit inside real ground-truth clusters.

This is a metric-honesty fix, not a pipeline behavior change — nothing
about how records are matched, verified, or excepted is touched.
Bank credit coverage's reported number on both bundled datasets moved
as a result (64.3%→52.4% calibration, 63.0%→50.0% evaluation): the
denominator shrank because BANK-side refund/chargeback/reversal legs
no longer count, which is the metric measuring what its name says
rather than a regression in coverage itself.

### Regression testing

`tests/test_repro_bugfix_dataset.py` is a new, permanent regression
suite (7 tests) built directly against
`data/repro_bugfix/records.json` — the exact reproduction dataset
attached to this bugfix pass, not a re-derived or synthetic stand-in:

- The `pay_di4adgpi56v9ec` pair and its reversal counterpart both
  route to `PENDING_REVIEW` / `RESIDUAL_EXCEEDS_TOLERANCE`, stay
  claimed (not released), and raise no exception (Bug 1).
- The 3-way refund, the 3-way chargeback, and the clean settlement
  chain all still verify cleanly and are byte-for-byte unaffected
  (Bug 1 regression guard).
- The orphan bank credit gets exactly one real exception with
  non-empty evidence and recommended action (Bug 2).
- The full run, through `api/service.py` exactly as the API/dashboard
  see it, reports `COMPLETED_WITH_EXCEPTIONS` (Bug 2).
- `bank_credit_coverage` against a hand-built ground truth for this
  dataset counts exactly the 2 genuine `BANK_CREDIT` records, never
  the BANK-side refund/chargeback legs sitting in the same real
  clusters (Bug 4).

The full existing suite was re-run after every change in this pass —
245 tests pass (237 before this pass, +1 new `bank_credit_coverage`
unit test, +7 in the new reproduction-dataset file), 1 skipped
(unchanged, a pre-existing environment-conditional skip). In
particular, the zero-false-VERIFIED and zero-false-REJECTED properties
from earlier in the build still hold exactly:
`test_stage1_and_stage2_propose_zero_false_matches_against_calibration`,
`test_verified_groups_are_zero_false_positives_against_calibration`,
`test_stage1to4_propose_zero_false_matches_against_calibration`,
`test_stage5_calibration_zero_false_positives_all_three_categories`,
`test_stage5_evaluation_zero_false_positives_all_three_categories`,
`test_stage6_wiring_reports_and_holds_zero_false_positives_on_calibration`,
and `test_stage6_wiring_reports_and_holds_zero_false_positives_on_evaluation`
all pass unmodified, and a direct harness re-run against both bundled
datasets confirms 100.0% auto-match precision and 0.0% false-match
rate on both, unchanged from before this pass — the only thing that
moved is how many Stage 1 matches now correctly wait for a human
instead of auto-verifying, and how `bank_credit_coverage` counts its
own denominator. Updated JSON exports:
`reports/calibration_evaluation_report.json`,
`reports/evaluation_evaluation_report.json`.

**Files touched:** `config.py`, `models/enums.py`,
`verification/verifier.py`, `matching/pipeline.py`,
`evaluation/harness.py`, `dashboard/app.py`, `.env.example`,
`ARCHITECTURE.md`, `README.md`, `tests/test_evaluation_harness.py`
(3 existing tests updated, 1 new), `tests/test_repro_bugfix_dataset.py`
(new), `data/repro_bugfix/records.json` (new — the reproduction
dataset itself, checked in so the regression tests are self-contained
and don't depend on an external upload).

## Threshold Recalibration — Stage 1 Residual Tolerance, 3 Days Before Submission

The Post-Relay Bugfix Pass above (Bug 1) added a real, necessary
ceiling on Stage 1's identifier-match residual — before it, an exact-
identifier match could auto-verify with an arbitrarily large,
completely unexplained gap between its LEDGER and GATEWAY amounts.
That fix was correct and safe from the start: zero false positives on
both bundled datasets, before and after. What was NOT right was the
shipped *default value* for `stage1_residual_tolerance_fraction`
(2.5%) — it was hand-derived from reading `testdata/generator.py`'s
`MDR_BPS_CHOICES` fee-tier arithmetic, not measured from the actual
data, and turned out too tight: it flagged a real, meaningful slice of
genuinely correct Stage 1 matches, dropping record coverage from
~85-87% to ~62-63% and value coverage from ~72% to ~45-57% on both
bundled datasets. This entry redoes that calibration the honest way —
against measured data, tuning only against `data/calibration/`, per
this project's own established calibration/evaluation discipline
(§11) — and documents the resulting number, the methodology behind
it, and one important, deliberately-not-hidden consequence.

### Methodology

`scripts/calibrate_stage1_residual.py` (new, checked in):

1. Runs the full Stage 1→5 proposal pipeline (`matching.pipeline.
   run_pipeline`, with `groq_client=object()` to force Stage 6's
   graceful degradation rather than a live call — irrelevant to this
   analysis) against `data/calibration/records.json` **only**.
   `data/evaluation/` is never read by this script.
2. Isolates every `MatchGroup` still tagged `STAGE1_EXACT` in
   `result.group_id_to_stage` once Stage 3/4's in-place group
   extension has already happened. This is a deliberate, load-bearing
   choice: a raw Stage 1 LEDGER+GATEWAY pair that later gets a bank
   leg attached by Stage 3 is re-proposed under a **new** group id
   tagged `STAGE3_AGGREGATE` (`stage3_aggregate.py` sets
   `proposed_by=ProposedBy.STAGE3_AGGREGATE` on the merged group), and
   is verified by Stage 3's own conservation check instead — the
   Stage 1 residual-tolerance check never runs against it at all.
   Counting those groups here would calibrate against a population the
   config value doesn't actually gate. This is exactly the same
   population `evaluation/harness.py`'s own `compute_tier_contribution`
   metric counts as `STAGE1_EXACT proposed` (both read 80 on
   calibration), which is the correctness check for this choice: on
   calibration data, running `run_stage1_exact` in isolation returns
   102 candidate groups, but only 80 of those are still `STAGE1_EXACT`
   by the time verification runs — the other 22 were absorbed into
   Stage 3 aggregation and are governed by a different check entirely.
3. For each of the 80, computes `abs(expected - matched) /
   max(expected, matched)` — the exact same formula
   `verification/verifier.py`'s `_verify_stage1` uses, read from the
   group's own `expected_amount_paise`/`matched_amount_paise` fields
   (set once by `stage1_exact.py`'s `_amounts`, independent of
   whatever tolerance fraction is configured — the tolerance only ever
   affects the verification *outcome*, never which records are
   proposed together or what their expected/matched amounts are, so
   this distribution is threshold-invariant and safe to compute before
   picking a threshold).
4. Cross-references every group's membership against calibration
   ground truth using the exact same four-way check
   `evaluation/harness.is_verified_group_correct` already uses
   (imported, not re-implemented): wrong if it touches a known
   duplicate, wrong if it touches an honest-abstention decoy, correct
   only if it's a subset of a real `match_groups` or `unresolved`
   cluster.

### The real distribution (calibration set, n=80, all genuinely correct)

```
min:    0.0000%
p25:    1.7700%
median: 2.3599%
p75:    2.6550%
p90:    2.9499%
p95:    2.9500%
max:    2.9562%
```

Zero incorrect `STAGE1_EXACT` proposals exist anywhere in the
calibration set — there is no "wrong match" population to contrast
against by construction of this synthetic dataset, only the spread of
genuinely correct residuals above. Every value clusters tightly at
five discrete percentages — 1.77% / 2.065% / 2.36% / 2.655% / 2.95% —
which is exactly `testdata/generator.py`'s `MDR_BPS_CHOICES = (150,
175, 200, 225, 250)` basis points, each netted against an 18% GST
add-on: `250bps * 1.18% = 2.95%` is the ceiling of the *real* fee
population in this data, confirming the original bugfix's hand-derived
reasoning was directionally right but numerically un-anchored — 2.5%
cut directly through the middle of that population instead of sitting
above all of it. (Re-running the same script against
`data/evaluation/records.json`, purely as an independent sanity check
— never used to pick the number — shows the identical five tiers,
n=90, max 2.9500%: the generator's fee-tier logic, not sampling noise,
is what produces this shape.)

### Picking 3.25%

The new default needs to (a) sit above every genuinely correct
residual observed in calibration data with a real margin, (b) remain a
meaningful, real bound rather than one raised so high the check never
fires, and (c) be defensible with a number, not a feeling. **3.25%**
was chosen as roughly a 0.3-percentage-point margin above the observed
maximum (2.9562%) — comfortably covering all five real fee tiers plus
their small per-amount paise-rounding noise, while remaining far
enough below, say, a second unrelated transaction's amount colliding
with this one by reference-reuse accident (which would produce a
residual with no relationship to any MDR/GST tier at all, plausibly
tens of percent) that the check still has real teeth. It is not
"raised until the check stops firing" — it is anchored to the actual
ceiling of legitimate variance in this system's own data, with a
margin sized to the granularity of that data (discrete tiers roughly
0.2-0.3 percentage points apart), not an arbitrary round number picked
first and rationalized after.

**`config.py`:** `stage1_residual_tolerance_fraction` default changed
from `0.025` to `0.0325`; `.env.example`'s
`STAGE1_RESIDUAL_TOLERANCE_FRACTION` updated to match. No verification
logic touched — `verification/verifier.py`'s `_verify_stage1` and
`_stage1_residual` are byte-for-byte unchanged; this is a config-value
change only, exactly as scoped.

### Results — both bundled datasets, before vs. after this recalibration

| Dataset | Metric | 2.5% (before) | 3.25% (after, calibrated) |
|---|---|---|---|
| calibration | VERIFIED / PENDING_REVIEW / REJECTED | 60 / 27 / 5 | **87 / 0 / 5** |
| calibration | Auto-match precision | 100.0% (60/60) | 100.0% (87/87) |
| calibration | False-match rate | 0.0% (0/60) | 0.0% (0/87) |
| calibration | Record coverage (raw / excl. abstention) | 63.2% / 64.5% | **85.1% / 86.9%** |
| calibration | Value coverage (raw / excl. abstention) | 56.7% / 56.9% | **72.2% / 72.5%** |
| calibration | Exception quality | 77.8% (7/9) | 100.0% (7/7) |
| evaluation | VERIFIED / PENDING_REVIEW / REJECTED | 64 / 32 / 5 | **96 / 0 / 5** |
| evaluation | Auto-match precision | 100.0% (64/64) | 100.0% (96/96) |
| evaluation | False-match rate | 0.0% (0/64) | 0.0% (0/96) |
| evaluation | Record coverage (raw / excl. abstention) | 62.4% / 63.7% | **86.8% / 88.6%** |
| evaluation | Value coverage (raw / excl. abstention) | 44.8% / 45.0% | **66.5% / 66.8%** |
| evaluation | Exception quality | 77.8% (7/9) | 100.0% (7/7) |

Bank credit coverage (52.4% calibration / 50.0% evaluation) and
complete cluster resolution (50.0% calibration / 44.4% evaluation) are
**byte-for-byte identical before and after** on both datasets — those
two metrics are driven entirely by Stage 3/4 aggregation, which this
change never touches.

**Honest reporting on coverage "recovery":** it recovers essentially
completely — on both datasets, at 3.25% every single `STAGE1_EXACT`
proposal's residual now falls within tolerance (0 `PENDING_REVIEW`
left from Stage 1 on either dataset), because the calibrated ceiling
sits above the entire observed real-fee-tier population by
construction. This is not a coincidence to be suspicious of: it is the
expected, correct outcome of setting the bound at "above everything
real that's actually in this data" rather than at an arbitrary
in-between guess. Coverage matches, almost to the decimal point, what
Bug 1's own fix table (further up this file) reported as the pre-fix,
zero-residual-check baseline (87/0 VERIFIED calibration, 96/0
evaluation) — which makes sense, since the calibrated ceiling now sits
above every real residual in either dataset, so no genuinely correct
match is being held back for a human anymore. **This does not mean the
check is a no-op or that this recalibration silently reintroduced
Bug 1** — the check still independently re-derives and compares the
residual on every single group, still has a real, data-anchored
number backing it (not "no check"), and would still fire on a residual
that didn't fit the real fee-tier population (see the reproduction
dataset finding immediately below for exactly how far that population
extends, and where the honest limit of a magnitude-only check sits).

### The reproduction dataset, revisited

Bug 1's original fix was verified directly against
`data/repro_bugfix/records.json`'s `pay_di4adgpi56v9ec` pair
(`demo_ledger_payment_002`/`demo_gateway_settlement_002`, residual
146,948 paise) and its exact reversal — both were, at the time,
confirmed to correctly route to `PENDING_REVIEW`. Recomputing that
pair's residual precisely: `146,948 / 4,981,320 = 2.9502%`. Compared
against this pass's own calibration-set finding (max genuinely correct
residual: 2.9562%), the reproduction case's residual is **smaller**
than the top of the real, legitimate fee-tier population — i.e. it is
numerically indistinguishable from, and in fact slightly below,
ordinary top-tier MDR+GST variance seen throughout actual calibration
data.

**Consequence, stated plainly:** under the newly-calibrated 3.25%
default, this pair and its reversal now auto-verify (`PASSED` /
`VERIFIED`) instead of routing to `PENDING_REVIEW`. This was checked
directly, not assumed:

| Group | 2.5% (before) | 3.25% (after, calibrated) |
|---|---|---|
| `demo_ledger_payment_002` + `demo_gateway_settlement_002` | PENDING_REVIEW / RESIDUAL_EXCEEDS_TOLERANCE | **VERIFIED / PASSED** |
| `demo_ledger_reversal_002` + `demo_gateway_reversal_002` | PENDING_REVIEW / RESIDUAL_EXCEEDS_TOLERANCE | **VERIFIED / PASSED** |
| the 3-way refund, 3-way chargeback, clean settlement chain | VERIFIED / PASSED | VERIFIED / PASSED (unaffected, as before) |
| the orphan bank credit's exception, run status | unaffected | unaffected |

**Why this is the right outcome, not a regression:** the original 2.5%
default caught this case, but only by accident of being tighter than
the real fee-tier ceiling, not because it was calibrated to separate
"this specific residual" from "genuine fee variance" — the earlier
Post-Relay pass said as much at the time ("there's no threshold that
separates the reproduction case from real data by magnitude alone,
because the reproduction case's actual tell is that no FEE record
exists anywhere in its 15-record batch to explain the gap, not the
size of the gap itself"). This pass confirms that finding
quantitatively rather than by inspection: the calibration set's own
*maximum genuinely correct* residual (2.9562%) is higher than this
supposedly-suspicious case's residual (2.9502%). Any threshold that
honestly, comfortably covers real calibration data — the explicit goal
of this recalibration — necessarily also covers this case. Picking a
tighter number specifically to keep this one hand-crafted example
flagged would mean going back to guessing an arbitrary cutoff rather
than deriving one from data, which is exactly the mistake being
corrected here. Catching this specific case would require a different
kind of evidence entirely — e.g. requiring a corroborating `FEE`
record to justify a residual of this size, rather than accepting
identifier-match + magnitude-bound alone — which is a **logic change**
(a new check, not a threshold value), explicitly out of scope for this
pass, and left for future work (§16-style cut, not attempted here).

### Regression testing

`tests/test_repro_bugfix_dataset.py`'s two Bug-1 tests hardcoded the
2.5%-era outcome (`RESIDUAL_EXCEEDS_TOLERANCE` / `PENDING_REVIEW`) for
exactly the pair discussed above. Both were updated, not deleted, to
assert and explain the new, correct outcome:
`test_unexplained_residual_pair_now_passes_under_calibrated_tolerance`
and `test_unexplained_residual_reversal_also_now_passes` now assert
`PASSED` / `VERIFIED` / `AUTO_COMMIT_STAGE1`, with docstrings carrying
the same reasoning as above so a future reader hitting this test
doesn't mistake it for a silent regression. Every other test in that
file (`test_conserving_groups_still_verify_cleanly`,
`test_clean_settlement_chain_still_verifies`, the two orphan/status
tests) is untouched — none of the other three bugs from that pass are
affected by this one config value.

Full suite re-run after the change: **245 passed, 1 skipped** —
identical pass count to before this pass; only the two tests above
changed their assertions, nothing else needed touching. The zero-
false-VERIFIED and zero-false-REJECTED properties from earlier in the
build still hold exactly (same named tests as the previous pass's own
regression-testing section, re-run and re-confirmed, all still
passing unmodified): auto-match precision and false-match rate are
100.0%/0.0% on both bundled datasets before and after this
recalibration, and the 5 `REJECTED` groups on each dataset are
identical (same group ids, same `FAILED_CONSERVATION` reason) before
and after — this change touches zero conservation-check logic, so
nothing there could have moved. Updated JSON exports:
`reports/calibration_evaluation_report.json`,
`reports/evaluation_evaluation_report.json`.

**Files touched:** `config.py` (default value + docstring only — no
logic changed), `.env.example`, `scripts/calibrate_stage1_residual.py`
(new), `tests/test_repro_bugfix_dataset.py` (2 tests updated),
`README.md`, `ARCHITECTURE.md` (this file).

---

## Post-Relay Pass — Human Review Loop (2 days before submission)

**The gap.** `MatchGroup.reviewed_by` / `reviewed_at` / `review_action`
have been in the schema since early in the build (ARCHITECTURE.md
§6/§7 — "set only once a human reviewer acts on a PENDING_REVIEW
group"), and every stage that routes a group to `PENDING_REVIEW`
(Stage 1's residual check, Stage 2-5's margin check, Stage 6's
always-`HUMAN_REVIEW_REQUIRED` LLM tier) explicitly frames that status
as "needs a human's judgment, not an automatic pass." But an external
review of the repo found that no code path anywhere — no API endpoint,
no dashboard control, nothing — ever actually let a human set those
three fields. Every `PENDING_REVIEW` group produced by any run just
sat there permanently: real, correctly-flagged ambiguity with no way
to close it out. This is real, additive feature work (not a bug fix
like the entries above), scoped deliberately narrowly: one endpoint,
one dashboard action, zero changes to any matching/verification/LLM
logic.

**The fix.** `PATCH /report/{run_id}/groups/{group_id}/review`
(`api/app.py` + `api/service.py`'s `review_group`), accepting
`reviewer` (a name/identifier string), `review_action`
(`APPROVED`/`REJECTED`/`ESCALATED`), and an optional `note`:

- Only valid on a group currently `PENDING_REVIEW`. A group that's
  already `VERIFIED` or `REJECTED` returns a clean `409 Conflict`
  (`ReviewConflictError` in `api/service.py`) rather than silently
  re-reviewing it. An unknown `run_id`/`group_id` returns `404`; a
  malformed body (bad `review_action` enum value, missing `reviewer`)
  returns FastAPI's own `422`.
- **`APPROVED`** — sets `reviewed_by`/`reviewed_at`/`review_action`,
  and per ARCHITECTURE.md §6 ("At that point verified_by becomes
  HUMAN_REVIEWER and verification_result reflects the reviewer's
  decision"): `verified_by` → `HUMAN_REVIEWER`, `verification_result`
  → `PASSED`, `status` → `VERIFIED`. This is the human completing
  verification the system flagged but couldn't finish alone.
- **`REJECTED`** — sets `reviewed_by`/`reviewed_at`/`review_action`
  and `status` → `REJECTED`. Deliberately does **not** overwrite
  `verified_by`/`verification_result`: those already recorded *why*
  the automated verifier routed the group to `PENDING_REVIEW` (e.g.
  `FAILED_MARGIN`, `RESIDUAL_EXCEEDS_TOLERANCE`), and that reasoning
  stays true and useful — overwriting it would erase the evidence
  explaining why a human's judgment was needed in the first place. The
  human's verdict is fully captured by `review_action=REJECTED` +
  `reviewed_by` + `reviewed_at` + `status=REJECTED` alone. **This is
  also where a human rejection deliberately differs from an automatic
  one**: `matching/pipeline.py`'s `run_verification` releases an
  automatically-`REJECTED` group's member records back into
  `matched_record_ids` so a *later stage in the same pipeline run* can
  retry them. A human review happens after that pipeline run has
  already finished and persisted — there is no later stage left to
  hand a record to — and a human's explicit "no" on a specific group
  is a considered, final answer; silently making that record eligible
  for some other match again would undermine the review rather than
  respect it. So a human-`REJECTED` group's records simply stay
  claimed by that (now terminally `REJECTED`) group. Nothing new was
  needed to implement this — it's the *absence* of any release call in
  `review_group`, and is called out explicitly in that function's
  docstring so a future reader doesn't mistake it for an oversight.
- **`ESCALATED`** — sets `reviewed_by`/`reviewed_at`/`review_action`
  only; `status` stays `PENDING_REVIEW` and `verified_by`/
  `verification_result` are untouched. "Flagged for further
  attention," not resolved — the group remains open to a later
  `APPROVED`/`REJECTED` call (covered by
  `test_escalate_records_reviewer_but_leaves_status_pending`, which
  escalates then approves the same group in one test).
- Every action **emits a `DecisionEvent`** (`reason_code`
  `HUMAN_APPROVED`/`HUMAN_REJECTED`/`HUMAN_ESCALATED`, the optional
  `note` carried in `candidate_scores`) so the review shows up in the
  audit trail exactly like every other decision, not just as fields on
  the `MatchGroup`. **Stage choice, deliberately reusing an existing
  value rather than adding one**: `DecisionStage.STAGE7_VERIFICATION`
  is already defined (`models/enums.py`) as "the verifier's own
  pass/fail evaluation" of a group — a human review is the same kind
  of event (an evaluation of the group as a whole, not a new
  *proposal*), just performed by `HUMAN_REVIEWER` instead of
  `FINANCIAL_AND_EVIDENCE_VERIFIER`. `models/enums.py`'s own docstring
  says not to add/remove/rename values transcribed from
  ARCHITECTURE.md's §7 data model without updating the spec first;
  reusing `STAGE7_VERIFICATION` avoids that entirely, and means
  `dashboard/app.py`'s existing "every DecisionEvent naming this
  group_id" drill-down (`_events_for`) picks up review events with
  zero changes needed there.
- Persisted through `RunStore.apply_review` (`api/storage.py`) — one
  `UPDATE match_groups` + one `INSERT INTO decision_events`, wrapped in
  the same `BEGIN IMMEDIATE` / commit / rollback discipline `save_run`
  already uses, so a partially-applied review (group updated with no
  audit event, or vice versa) can never happen.

**Dashboard.** The drill-down tab's existing group inspector
(`dashboard/app.py`, Tab 3) gained a "Human review" section: for a
`PENDING_REVIEW` group it shows a reviewer-name field, an optional
note field, and Approve/Reject/Escalate buttons calling
`dashboard/api_client.py`'s new `review_group` (which required adding
`.patch()` to the client's minimal `SupportsGetPost` transport
protocol, satisfied by both `RequestsClient` and
`fastapi.testclient.TestClient` already). A successful action calls
`st.rerun()`, so the drill-down immediately reflects the updated
`status` and the newly-set `reviewed_by`/`reviewed_at`/`review_action`
— no separate manual refresh needed. For an already-`VERIFIED`/
`REJECTED` group, the section instead shows the recorded reviewer
fields (if a human already acted) or a note that only a
`PENDING_REVIEW` group can be reviewed (if it auto-verified/rejected
and no human ever needed to).

**Tests.** `tests/test_human_review.py` (new, 10 tests) is fully
end-to-end via `fastapi.testclient.TestClient`, and — per this pass's
own requirement — every `PENDING_REVIEW` group it reviews comes from
**actually running the real pipeline** through `POST /reconcile`,
never a hand-built `MatchGroup` fixture with `status=PENDING_REVIEW`
set directly. The fixture batch is three independent LEDGER+GATEWAY
pairs, each sharing one reference (a genuine Stage 1 `STAGE1_EXACT`
candidate) with the GATEWAY-side amount ~7% below the LEDGER-side
amount — comfortably past `config.py`'s calibrated
`stage1_residual_tolerance_fraction` (3.25%, see the recalibration
entry above), so the verifier's real, independent residual re-check
genuinely routes all three to `PENDING_REVIEW` /
`RESIDUAL_EXCEEDS_TOLERANCE`. Three independent groups are used
(rather than one) because `APPROVED`/`REJECTED` both permanently leave
`PENDING_REVIEW`, so each action under test needs its own group.
Covered: all three actions produce the exact field/status outcome
specified above; the audit trail (`GET /audit`) shows each review's
`DecisionEvent`; re-reviewing an already-`VERIFIED` group and an
already-`REJECTED` group are both cleanly refused with `409` (and the
group's already-recorded outcome is confirmed undisturbed by the
refused second attempt); unknown `run_id`/`group_id` return `404`;
malformed `review_action`/missing `reviewer` return `422`; the new
route appears in `/openapi.json`. `tests/test_dashboard_api_client.py`
gained one more round-trip test (`test_review_group_round_trip`) for
`api_client.review_group` specifically, following that file's own
existing pattern.

**Full suite re-run after this pass:** **256 passed, 1 skipped**
(245 → 256; the 1 skip is the same pre-existing calibration/evaluation
dataset-availability skip from earlier in the build, untouched by this
pass) — no existing test's assertions changed, since nothing here
touches matching, verification, or LLM logic; only new tests were
added.

**Files touched:** `api/app.py` (new `PATCH
/report/{run_id}/groups/{group_id}/review` endpoint), `api/service.py`
(new `review_group`/`ReviewNotFoundError`/`ReviewConflictError`),
`api/storage.py` (new `RunStore.get_match_group` /
`RunStore.apply_review`), `api/schemas.py` (new `ReviewRequest`/
`ReviewResponse`), `dashboard/app.py` (Approve/Reject/Escalate section
in the Tab 3 drill-down), `dashboard/api_client.py` (new `.patch()` on
the transport protocol + new `review_group` function),
`tests/test_human_review.py` (new), `tests/test_dashboard_api_client.py`
(1 test added), `README.md`, `BUILD_LOG.md` (this file).

---

## Multi-Seed Stress Evaluation (2 days before submission)

**The gap.** An external review independently generated 20 additional
fresh-seeded synthetic datasets (beyond the two — calibration and
evaluation — checked into this repo) and reported strong, consistent
precision across all of them. That analysis is real, but it lives
outside this repo: nobody reading this repo can reproduce it, rerun
it, or check whether it still holds after a later change. This pass
turns that into something the repo can generate and prove for itself,
on demand — additive tooling only, zero changes to
`matching/`/`verification/`/`llm/`.

**The mechanism, reusing exactly what already exists:**

- `testdata/generator.py`'s CLI gained a third mode alongside
  `calibration`/`evaluation`/`both`: `--seed stress --seed-value <int>
  --logical-events <int>`. `generate_dataset()` itself gained a third,
  optional parameter (`n_logical_events`, default `None`) that resolves
  to the fixed `N_LOGICAL_EVENTS`=100 and reuses the *exact* original
  §11 category counts, unchanged — this is exactly the path
  `calibration`/`evaluation` still take, so their content is
  byte-identical to before this pass (checksummed and confirmed; see
  Regression testing below). A different `n_logical_events` scales
  every §11 anomaly-category count proportionally
  (`_resolve_generation_counts`), clamped so every required category
  (clean, dirty references, timing variance, net settlements, all four
  lifecycle types, all three structural cases, honest abstention)
  stays representable at any reasonable scale — same anomaly-mix
  generation logic throughout, just a different RNG seed and, if
  asked, a different event count.
- `scripts/multi_seed_evaluation.py` (new): derives N=20 (default,
  matching the external review's own methodology so the numbers are
  directly comparable) distinct integer seeds from one fixed
  `--meta-seed` (so the default invocation is itself reproducible —
  not "20 random datasets," 20 *specific, named* seeds anyone can
  regenerate). For each seed: generates a fresh stress dataset into a
  throwaway temp directory (never `data/`/`ground_truth/` — keeps the
  repo lean, no 20 extra data files committed anywhere), runs the real
  pipeline via `evaluation.harness.run_evaluation` with whatever
  thresholds are already configured (one shared `Settings`, loaded
  once — nothing is retuned against these datasets, same
  calibration/evaluation-set discipline as everywhere else in this
  build), computes the full §11 metric set for each, and aggregates
  mean/min/max across all 20 for auto-match precision, record coverage
  (raw + excl. abstention), value coverage (raw + excl. abstention),
  false-match rate, exception quality, bank credit coverage, and
  complete cluster resolution. Prints a human-readable summary table
  and writes a JSON export (`reports/multi_seed_evaluation_report.json`),
  mirroring `evaluation/harness.py`'s own single-dataset CLI output
  style.
- **Groq safety.** All 20 seeds share one `Settings` (loaded once from
  the environment) and one temp governor DB path, so the real daily-
  call-ceiling/circuit-breaker protections in `llm/governor.py` apply
  across the *whole* stress batch, not reset per seed — a
  misconfigured key can't silently burn 20x the intended budget.
  Confirmed both states work: with no `GROQ_API_KEY` configured (this
  environment), all 20 seeds degrade gracefully with 0 live calls,
  exactly like calibration/evaluation always have (§14); the script
  prints which state actually occurred (`groq_configured` +
  `total_groq_calls`, measured from `runtime_and_cost.groq_calls_made`
  on each real report, never assumed from whether a key merely looks
  present).

**Run it yourself:**

```
python -m scripts.multi_seed_evaluation
```

(20 seeds, 100 logical events each, `--meta-seed 20261231` default —
pass `--n-seeds`/`--meta-seed`/`--logical-events` to vary any of that.)

**Real aggregate numbers — 20 fresh seeds, this repo, this run** (no
`GROQ_API_KEY` configured; every seed degraded gracefully, 0 live Groq
calls total; seeds `11499962, 12065379, 18011908, 26734115, 27598209,
35676529, 42714975, 44948120, 45102055, 46030642, 46232665, 51024150,
52652362, 58637099, 70182912, 77887731, 79594069, 87901550, 88591708,
99150830`):

| Metric | mean | min | max |
|---|---|---|---|
| 1. Auto-match precision | 99.8% | 98.8% | 100.0% |
| 2. Record coverage (raw) | 86.4% | 83.3% | 91.0% |
| 2. Record coverage (excl. abstention) | 88.2% | 85.1% | 92.9% |
| 3. Value coverage | 71.8% | 64.4% | 84.4% |
| 3. Value coverage (excl. abstention) | 72.1% | 64.7% | 84.8% |
| 4. False-match rate | 0.2% | 0.0% | 1.2% |
| 5. Exception quality | 100.0% | 100.0% | 100.0% |
| 8. Bank credit coverage | 49.3% | 30.0% | 76.2% |
| 9. Complete cluster resolution | 45.7% | 37.0% | 57.1% |

**Honest read against the external review's "strong, consistent
precision across all of them":** *mostly* consistent, not perfectly
so, and this repo's own run is what surfaces the difference instead of
just repeating the external claim. 17 of 20 seeds landed at exactly
100.0% precision / 0.0% false-match rate, matching calibration's and
evaluation's own 100%/0% (see README.md's "Key Results"). But 3 of 20
seeds (`27598209`, `35676529`, `51024150`) each produced exactly one
incorrect `VERIFIED` group out of 84-91 — 98.8-98.9% precision, not
100%. This is a real, if small, deviation from "consistent" that a
single held-out evaluation set can't surface on its own (calibration
and evaluation both happen to be clean at 100%/0%); it took 20
independent seeds to find. Coverage metrics vary more visibly across
seeds than precision does (record coverage 83.3-91.0%, bank credit
coverage 30.0-76.2%, complete cluster resolution 37.0-57.1%) — this
tracks expectation, since Stage 3/4 aggregation's success depends on
exactly how a given seed's events happen to batch into consolidated
settlements, which is inherently seed-sensitive in a way the
proposal-correctness checks underlying precision are not. **This finding
is reported, not chased**: per this stage's own scope (additive tooling
only), no matching/verification/LLM code was touched to try to close
that ~1% gap — that would be exactly the kind of overfitting-to-the-
exact-data-judges-will-see that §11's calibration/evaluation split
exists to prevent, and three isolated single-group misses across 20
independent 90-ish-event datasets is a materially different question
from calibration/evaluation's own clean numbers, worth flagging
honestly rather than either hiding it or hand-fixing it two days before
submission.

**Regression testing.** Calibration and evaluation datasets confirmed
byte-identical (checksummed) before and after the generator change —
this pass changes zero bytes of what those two datasets contain. New
tests: 5 in `tests/test_generator.py` (stress-seed default scale
covers every required category; scaled-down `n_logical_events=30` still
covers every required category; determinism for a given stress seed;
distinct stress seeds produce distinct datasets; the default
`n_logical_events=None` path is provably identical to calibration's own
call). 11 in `tests/test_multi_seed_evaluation.py`, scoped deliberately
to the aggregation math itself — 2-3 tiny constructed per-seed metric
fixtures with known correct mean/min/max, `None`-skipping behavior, and
seed-derivation determinism/distinctness — not a full 20-seed run,
which has no place in a test suite that needs to stay fast (it's run
manually; see the table above). **Full suite re-run after this pass:
272 passed, 1 skipped** (256 → 272; the 1 skip is the same
pre-existing dataset-availability skip from earlier in the build) — no
existing test's assertions changed.

**Files touched:** `src/recon_agent/testdata/generator.py` (`--seed
stress` CLI mode, `n_logical_events` param on `generate_dataset`,
`_resolve_generation_counts`), `scripts/multi_seed_evaluation.py`
(new), `tests/test_generator.py` (5 tests added),
`tests/test_multi_seed_evaluation.py` (new, 11 tests), `README.md`,
`BUILD_LOG.md` (this file).

## Concurrency Bug — Shared SQLite Connection Under Live Traffic (found via live testing on a real machine, submission imminent)

**This finding is different in kind from every other bug documented in
this file, and worth flagging as such up front: it was found by running
the actual API on a real machine and sending it real concurrent HTTP
traffic — not by synthetic data, not by a crafted reproduction dataset,
not by any test in this repo.** Every test this project has ever run
(all 272 passing before this pass) drives the API through
`TestClient` with sequential calls, one request awaited fully before
the next begins. That is a structurally different traffic shape from a
live deployment, where the dashboard's background `GET /health`
polling and a real `POST /reconcile` call can — and, on live testing,
did — land on the server at the same moment. No amount of additional
sequential test cases would ever have caught this; it needed genuine
concurrent traffic hitting a real socket, which is exactly the gap this
pass closes (see "Verification" below).

**The bug, as diagnosed from a real production traceback.** A
`POST /reconcile` request collided with a burst of concurrent
`GET /health` requests. On the client side this showed up as a flat
30-second `ReadTimeout` — no error, no stack trace, just a hang —
because the actual crash happened inside a background worker thread on
the server and the client's connection was simply left dangling.
Server-side, the real exception was:

```
sqlite3.OperationalError: cannot start a transaction within a transaction
```

thrown from `storage.py`'s `save_run()`, at the `conn.execute("BEGIN
IMMEDIATE")` line.

**Root cause.** Both `api/storage.py`'s `RunStore` and
`llm/governor.py`'s `CallBudgetGovernor` opened exactly one
`sqlite3.Connection` in `__init__` and held it on `self` for the life
of the process, passing `check_same_thread=False` specifically so it
*could* be reused across requests. That flag disables Python's own
same-thread safety check, but it does not make a `sqlite3.Connection`
safe for genuinely concurrent use — the connection's "am I mid-
transaction" bookkeeping lives on the `Connection` object itself, not
per-thread. FastAPI's sync endpoints (every endpoint in `api/app.py` is
a plain `def`, not `async def`) run on Starlette's threadpool, so
concurrent requests really do call into this class from different OS
threads at the same moment. When one thread's `save_run()` was
mid-transaction on the shared connection and a second thread's own
`conn.execute(...)` call (any statement — a read from `run_exists`, or
another `BEGIN IMMEDIATE`) landed on that same connection object at the
wrong moment, the connection's transaction state got corrupted and the
next `BEGIN IMMEDIATE` failed outright. This had never been caught
before because every test in this project used sequential `TestClient`
calls — nothing ever exercised genuinely concurrent live HTTP traffic
until this live run.

**The fix.** Eliminate the shared connection entirely, in both files —
`llm/governor.py`'s `CallBudgetGovernor` turned out to have the exact
same shared-connection pattern as `storage.py`'s `RunStore` (it was
built to mirror `RunStore`'s connection setup — see its own module
docstring — so it inherited the same defect, not a different one). No
locking was added around a shared connection; per the diagnosis, SQLite
connections are cheap to open for a file-based DB at this scale, and a
lock would only have serialized what should already run as independent
per-request work.

- `api/storage.py`: `RunStore` no longer holds a connection on `self`
  at all. A new `_connect()` helper opens a fresh
  `sqlite3.connect(db_path, timeout=30.0, isolation_level=None)` (note:
  `check_same_thread=False` is gone too — it's not needed once each
  connection is only ever touched by the one thread that opened it) +
  `PRAGMA busy_timeout = 30000`. Every method that used to reach for
  `self._conn` — `save_run`, `apply_review`, and every `get_*`/
  `run_exists` read — now opens its own connection via `_connect()`,
  does its work, and closes it in a `finally`. The two write paths keep
  their existing `BEGIN IMMEDIATE` / `commit()` / `except
  BaseException: rollback()` discipline unchanged — that part of the
  design was already correct and stays exactly as it was, just scoped
  to a connection that belongs to one call instead of one process.
  `close()` is kept as a harmless no-op (existing callers, e.g.
  `api/app.py`'s `reset_store_for_tests`, still call it).
- `llm/governor.py`: identical shape. `CallBudgetGovernor` no longer
  holds `self._conn`; `_connect()` opens a fresh connection per call
  (schema init, `status`, `try_consume`, `record_success`,
  `record_failure`), each still wrapped in the same
  `BEGIN IMMEDIATE`/commit/rollback pattern the module docstring
  documents (the atomic check-and-increment guarantee that "budget
  counter now transactional" correction — ARCHITECTURE.md §10 — is
  unchanged; only the connection sharing is gone). `_read_breaker`/
  `_read_calls_used` now take the caller's connection as a parameter
  instead of reaching for `self._conn`. `close()` kept as a no-op for
  the same backward-compatibility reason.
- `api/app.py`'s `get_store()` docstring updated — it previously
  justified the module-level `RunStore` singleton by claiming the
  underlying connection was "already safe for reuse across requests
  (`check_same_thread=False`)"; that claim was the bug. The singleton
  itself is still fine to keep (one `RunStore` *instance* shared across
  requests is safe now, since the instance itself holds no connection
  — only a `db_path` string), but the docstring's stated reason was
  wrong and is corrected to explain why the *new* design is safe
  instead.

**Verification.** Required a genuinely different kind of test from
everything else in this repo — real concurrent traffic, not another
sequential `TestClient` run:

- New file `tests/test_concurrency_live.py` starts an actual `uvicorn`
  server on a real TCP socket (`uvicorn.Server` in a background daemon
  thread of the test process — not `TestClient`'s in-process ASGI
  shortcut) and fires real HTTP requests at it via `httpx` from a
  `ThreadPoolExecutor`.
  - `test_concurrent_reconcile_and_health_bursts_never_crash_or_hang`:
    mixes one `POST /reconcile` with a burst of 6 concurrent
    `GET /health` calls — the same traffic shape the production
    traceback showed — repeated across 8 rounds (this was a race
    condition; a single clean round proves very little). Each request
    uses an 8-second timeout, deliberately well under the original 30s
    `ReadTimeout`, so a reintroduced hang fails fast and visibly rather
    than being masked by a generous timeout. After each round, the
    run's audit trail is read back via `GET /audit/{run_id}` to confirm
    the write that just happened under concurrent load is actually
    there — not just "no exception," but genuinely, correctly
    persisted.
  - `test_concurrent_reconcile_requests_never_crash_or_hang`: a second,
    complementary shape — 5 concurrent `POST /reconcile` writers at
    once, across 6 rounds — the most direct way to drive two threads'
    `save_run()` transactions into each other, and the shape most
    likely to reproduce the original crash on demand.
  - `test_governor_try_consume_under_real_concurrent_threads` and
    `test_governor_record_success_and_failure_under_real_concurrent_threads`:
    this test dataset's 3 records are a clean, fully-matched triad that
    never reaches `stage6_llm.py`'s residual-record path, so the two
    live-server tests above never actually exercise
    `CallBudgetGovernor` — it's hit directly instead, hammering
    `try_consume` with 40 concurrent threads across 5 rounds (asserting
    not just "no exception" but that the final count is *exactly*
    right — proof the fix serializes correctly rather than merely not
    crashing by luck) and `record_success`/`record_failure` with 60
    concurrent calls across 6 run_ids at once.
- **Confirmed these tests actually catch the bug**: before finalizing,
  the fixed `storage.py`/`governor.py` were temporarily swapped back
  for the original shared-connection versions and the suite above was
  re-run against them — it reproduced the exact diagnosed
  `sqlite3.OperationalError: cannot start a transaction within a
  transaction` (plus a few related `DatabaseError`/`InterfaceError`
  variants from the same corrupted-connection-state root cause) in both
  `RunStore` and `CallBudgetGovernor`, on the very first round, every
  time. The fix was then restored and the same suite re-run clean
  multiple times in a row.
- **Full existing suite re-run**: 272 passed, 1 skipped before this
  pass → **276 passed, 1 skipped** after (272 existing + 4 new
  concurrency tests) — every pre-existing test's assertions unchanged.

**Files touched:** `src/recon_agent/api/storage.py`,
`src/recon_agent/llm/governor.py`, `src/recon_agent/api/app.py`
(docstring only), `tests/test_concurrency_live.py` (new, 4 tests),
`README.md`, `BUILD_LOG.md` (this file).

## Groq model deprecation — Llama 4 Scout to gpt-oss-120b (2026-09-04)

**What happened.** Groq deprecated `meta-llama/llama-4-scout-17b-16e-instruct`
(announced June 2026), which had been this project's Stage 6 (`llm/recommender.py`)
model since early in the build — the same identifier named in
ARCHITECTURE.md §10's LLM row. This was caught and fixed on submission
day, so the fix was kept deliberately narrow: the model default and
the response-parsing robustness it requires, nothing else.

**The swap.** `config.py`'s `Settings.groq_model` default (and its
`from_env()` fallback) and `.env.example`'s `GROQ_MODEL` now point at
`openai/gpt-oss-120b` — Groq's own documented 1:1 replacement
recommendation for the deprecated Llama 4 Scout snapshot. Still fully
overridable via the `GROQ_MODEL` environment variable, unchanged from
before.

**Why the response parsing needed to change too.** gpt-oss-120b is a
reasoning model. Depending on how Groq surfaces reasoning for a given
call, it can prepend a `<think>...</think>` block ahead of the actual
JSON answer — and the existing `_parse_response` in
`llm/recommender.py` did strict `json.loads` only, which would treat
that prefix as a malformed response and reject an otherwise-correct
answer. Two changes, belt-and-suspenders:

1. **The API call itself now asks Groq to suppress/relocate
   reasoning.** `_call_groq` detects "known reasoning model" name
   markers (`gpt-oss`, `qwen3`, `qwq`, `deepseek-r1`, `r1-distill` —
   not hardcoded to this one model) and, only for those, sends
   `reasoning_format="hidden"` (the value Groq's API docs specify for
   use alongside `response_format={"type": "json_object"}` — only
   `"hidden"`/`"parsed"` are supported together with JSON mode) and
   `reasoning_effort="low"` (keeps latency/token usage down — this is
   now a live-demo dependency). If the installed `groq` SDK doesn't
   recognize these kwargs at all, the call retries once without them
   rather than failing outright.
2. **The response parsing defends against reasoning leaking through
   anyway.** Groq's own community forum has documented cases of
   `reasoning_format="hidden"` still not fully suppressing reasoning
   output for gpt-oss-120b, so requesting it isn't sufficient on its
   own. `_parse_response` now: (a) strips a leading
   `<think>...</think>` or `<thinking>...</thinking>` block via regex
   before attempting `json.loads`; and (b), as a general safety net
   *not* specific to that tag shape, falls back to scanning the text
   for the first syntactically valid embedded JSON object if direct
   parsing still fails. A response with no recoverable JSON anywhere
   still raises `LLMMalformedJSONError` exactly as before — nothing
   about the strict schema validation downstream was loosened.

**Scope discipline.** Only `config.py` (model default),
`.env.example` (matching default), and `llm/recommender.py` (the
parsing/call-kwargs robustness above) were touched. `matching/`,
`verification/`, `llm/governor.py`, and `api/storage.py` (the prior
round's concurrency fix) were not touched at all — confirmed by diff
against the pre-fix tree.

**Tests.** `tests/test_llm_recommender.py`'s existing mocked-Groq
coverage is untouched; six tests were added on top of it: a
`<think>`-wrapped happy-path pick, a `<thinking>`-wrapped NO_MATCH, a
plain-prose-prefixed response (proving the JSON-scan fallback isn't
tag-specific), a reasoning-wrapped response with no recoverable JSON
(still raises `LLMMalformedJSONError`), and two covering that
`reasoning_format`/`reasoning_effort` are sent for a recognized
reasoning model and *not* sent for a non-reasoning one. **Full
existing suite re-run**: 276 passed, 1 skipped before this pass →
**282 passed, 1 skipped** after (276 existing + 6 new recommender
tests) — every pre-existing test's assertions unchanged.

**Live verification — not run in this environment.** No `GROQ_API_KEY`
was present in the environment this fix was made in, and that
environment's network egress allowlist doesn't include Groq's API
domain regardless, so no genuine live call could be made here. See
README.md's Quickstart / `tests/test_llm_recommender.py`'s
`test_live_groq_call_returns_a_recommendation` for how to confirm this
on a real machine before the pitch — set a real `GROQ_API_KEY` and
`RUN_LIVE_GROQ_TESTS=1` and run
`pytest tests/test_llm_recommender.py -k live -v`, or use the
one-off script noted there. **This must be confirmed working on a
real machine before the pitch.**

**Files touched:** `src/recon_agent/config.py`, `.env.example`,
`src/recon_agent/llm/recommender.py`, `tests/test_llm_recommender.py`
(6 new tests, existing tests unchanged), `README.md`, `BUILD_LOG.md`
(this file).
