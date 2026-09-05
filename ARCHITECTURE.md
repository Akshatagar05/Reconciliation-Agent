# Reconciliation Agent — System Architecture
### Track 04: AI Finance Controller — Razorpay AI Buildathon 2026
### Product statement: An exception-aware multi-source settlement reconciliation agent that matches bank credits, gateway settlements and internal ledger entries; verifies money conservation; and produces a measurable, auditable close report.

---

## 1. Scope Decision

| Capability | Status |
|---|---|
| Multi-source reconciliation | **CORE** |
| Exception workflow + audit | **CORE** |
| LLM arbitration | **CONTROLLED** — recommend-only, defaults to `PENDING_REVIEW` |
| Cash forecast | **BONUS** — after acceptance gate, sourced from verified ledger |
| Tax-line matcher | **CUT** |
| Settlement Q&A / RAG | **CUT** |

**New scope constraint from this review:** matching cardinality is explicitly limited to **1:1 and many-to-one**. Arbitrary many-to-many resolution is declared **unsupported** rather than attempted with an unreliable greedy method. This is a deliberate, stated limitation — a smaller exact claim is stronger than a broad unreliable one.

---

## 2. Verification Policy — Stage-Differentiated Auto-Commit

Passing conservation, currency, and uniqueness checks is **necessary but not sufficient** — two different candidate combinations can produce the same monetary total, and both can conserve money while only one is factually correct. "Financial Verifier" is renamed **Financial and Evidence Verifier** throughout, and auto-commit is now stage-differentiated rather than a single pass/fail gate:

| Stage | Auto-commit condition |
|---|---|
| Stage 1 (exact) | Exact normalized identifier match AND identifier is unique AND currency/date policy passes |
| Stage 2 (constrained) | **Not a lighter policy than Stage 1** — Stage 2 exists precisely because there's no clean shared identifier, so it uses the same evidence bar as Stages 3–5: amount/date/currency/counterparty composite score, unique candidate, above the calibrated threshold, sufficient runner-up margin, conservation check passes |
| Stages 3–5 (aggregate, adjustment, fuzzy) | Candidate is unique, scores above the calibrated threshold, AND is separated from the runner-up candidate by a sufficient margin — no auto-commit on a close call |
| Stage 6 (LLM recommendation) | **Always `commit_policy: HUMAN_REVIEW_REQUIRED`, no exceptions.** There is no auto-commit path for LLM output, ever — see the reasoning below |

The Financial and Evidence Verifier evaluates every candidate group against this table before anything is marked `VERIFIED`. A group that conserves money but fails the margin/uniqueness test still routes to `PENDING_REVIEW` or `Exception` — passing the money-conservation check alone is never sufficient.

**Why Stage 6 has no auto-commit path at all, not even a corroborated one:** an earlier draft allowed the LLM's output to auto-commit if non-LLM evidence independently satisfied the Stage 3–5 policy — but that's incoherent. Stage 6 only ever receives residue that Stages 1–5 already failed to safely commit. If independent evidence had been sufficient, it would have committed *at* Stage 3–5 and never reached Stage 6 at all. So a "corroborated auto-commit" path can never actually fire — it was dead logic. The real rule is simpler and stronger: **the LLM can shorten investigation time, classify the likely cause, and recommend a candidate — but it never increases automatic financial authority.** That's also a cleaner line for the judges: *LLM output improves the review queue, not the false-match risk.*

---

## 3. Group Membership Model

`MatchAllocation` (source_record_id / target_record_id / allocated_amount_paise) is **pairwise** and breaks the moment a group spans bank + gateway + ledger records simultaneously, or contains several records from one source — links become ambiguous and duplicated. Replaced with group membership:

```
MatchGroupMember
─────────────────────────────────────────────────────────
group_id
record_id
source              enum   BANK | GATEWAY | LEDGER
role                enum   GROSS | CREDIT | FEE | TAX | REFUND | CHARGEBACK | REVERSAL | ADJUSTMENT
signed_amount_paise  int     positive for credits, negative for debits/fees
allocated_amount_paise int
```

The verifier now evaluates the **entire group's membership set** against the conservation equation, not a chain of pairwise links. Cardinality is corrected accordingly:

```
MatchGroup.cardinality: ONE_TO_ONE | ONE_TO_MANY | MANY_TO_ONE | MANY_TO_MANY
```

(`MANY_TO_MANY` remains in the enum for schema completeness and future work, but per §1 the resolver does not attempt to *produce* `MANY_TO_MANY` groups in this build — see §4.)

---

## 4. Global Conflict Resolution

A priority-ordered greedy pass is order-dependent and can miss the globally best assignment — exactly the kind of thing that gets challenged live by anyone with an algorithms background. Corrected approach, scoped to what's reliably buildable in the remaining time:

| Case | Method |
|---|---|
| 1:1 conflicts | **Hungarian algorithm** (`scipy.optimize.linear_sum_assignment`) — globally optimal with respect to the configured score function, for bipartite 1:1 assignment |
| Many-to-one aggregation (several ledger entries → one bank credit, or the reverse) | **Aggregation windows are partitioned to be disjoint by merchant/account, currency, and settlement window *before* any candidate search runs.** *(Scope note: `NormalizedRecord` carries no merchant/account field — this build models one company's own reconciliation, not a multi-tenant platform, so "merchant/account" collapses to a single implicit scope covering the whole dataset. The operative partition keys here are `currency` and settlement window; don't invent a constant merchant field just to satisfy the general phrasing.)* Per-target subset-sum is only exact when candidate pools can't overlap — two bank credits both able to claim the same ledger entry is exactly the case that breaks a naive per-target search. Within a disjoint window, bounded subset-sum/DP finds the best subset. A window that can't be cleanly partitioned (a genuine overlap) routes directly to `Exception: AMBIGUOUS_AGGREGATION` instead of being resolved by search |
| Arbitrary many-to-many | **Explicitly unsupported.** Declared as a stated limitation in the README and the pitch, not silently absent |
| Bounding | Hard limit on candidates per window, hard limit on max group size, and a timeout per window — see §9 for the documented defaults |
| Reproducibility | Deterministic tie-breaking (e.g., lowest `record_id` wins ties) so the same input always produces the same resolution |

**Why not ILP:** the review offers ILP/min-cost flow/weighted set-packing as the fuller solution, but also explicitly names the fallback — descope to 1:1 + many-to-one and state N:M as unsupported — as the stronger move if ILP doesn't fit the timeline. Taking that fallback deliberately, with disjoint windowing to keep it honest: this is **optimal with respect to the configured score function and bounded, disjoint candidate set — not a claim of factual correctness.** Algorithmic optimality and financial correctness are different things; every result, optimal or not, still has to clear the Financial and Evidence Verifier's auto-commit policy (§2) before it can be `VERIFIED`. "We explicitly don't support arbitrary many-to-many, here's why" is a more defensible answer under judge questioning than a broader claim resting on an unverified heuristic.

---

## 5. Fuzzy Matching Scope — Candidate Retrieval Only

Stage 5 previously used edit-distance on identifiers as if it were matching evidence — but a one-character difference in a payment ID can point at a completely different transaction. Fuzzy identifier similarity is now **retrieval only, never sufficient by itself**:

```
Stage 5 acceptance requires ALL of:
  fuzzy identifier match
  + same currency
  + amount compatibility (within tolerance)
  + date compatibility (within policy window)
  + counterparty evidence (name/account match, not just proximity)
  + a unique score margin over the runner-up candidate
```

A fuzzy identifier hit alone produces a **candidate**, never a commit — it still has to clear the full stage 3–5 auto-commit bar from §2.

---

## 6. Proposal Provenance vs. Verification

`resolved_by = STAGE6_LLM_RECOMMENDED` was misleading — the LLM never resolved anything, it proposed. `MatchGroup` now carries four distinct fields instead of one:

```
proposed_by         enum    STAGE1_EXACT | STAGE2_CONSTRAINED | STAGE3_AGGREGATE |
                            STAGE4_ADJUSTMENT | STAGE5_FUZZY | STAGE6_LLM
verified_by          enum    NOT_YET_VERIFIED | FINANCIAL_AND_EVIDENCE_VERIFIER | HUMAN_REVIEWER
commit_policy        enum    AUTO_COMMIT_STAGE1 | AUTO_COMMIT_STAGE2_5_THRESHOLD |
                             HUMAN_REVIEW_REQUIRED
verification_result   enum    PASSED | FAILED_CONSERVATION | FAILED_UNIQUENESS |
                             FAILED_CURRENCY | FAILED_MARGIN |
                             RESIDUAL_EXCEEDS_TOLERANCE | NOT_YET_RUN
```

`RESIDUAL_EXCEEDS_TOLERANCE` is a post-launch addition (bugfix,
BUILD_LOG.md's "Post-Relay Bugfix Pass" §Bug 1): Stage 1's own bounded-
residual analog of `FAILED_MARGIN` — a real identifier match whose
LEDGER-vs-GATEWAY residual exceeds `config.py`'s
`stage1_residual_tolerance_fraction`, routed to `PENDING_REVIEW` /
`HUMAN_REVIEW_REQUIRED` rather than auto-verified or rejected.

`LLM_CORROBORATED_AUTO_COMMIT` is not a valid value here — it would describe a path that can never actually fire (see §2). `AUTO_COMMIT_STAGE1` and `AUTO_COMMIT_STAGE2_5_THRESHOLD` reflect Stage 2 sharing Stage 3–5's evidence bar rather than Stage 1's own.

Example — a Stage 6 proposal, **before** verification has run:

```
proposed_by:         STAGE6_LLM
verified_by:          NOT_YET_VERIFIED
commit_policy:        HUMAN_REVIEW_REQUIRED
verification_result:   NOT_YET_RUN
```

`verified_by` only changes away from `NOT_YET_VERIFIED` once a verification step has actually run — either the automated verifier or a human reviewer.

When a human reviewer later acts on a `PENDING_REVIEW` group, three more fields record it:

```
reviewed_by     str        the human reviewer's identifier
reviewed_at     datetime
review_action    enum    APPROVED | REJECTED | ESCALATED
```

At that point `verified_by` becomes `HUMAN_REVIEWER` and `verification_result` reflects the reviewer's decision.

This makes the audit trail defensible under direct questioning: it shows *who suggested* a match separately from *who approved* it, *under what policy*, and — for anything that needed a human — *who actually made the call and when*.

---

## 7. Full Data Model

```
ReconciliationRun
─────────────────────────────────────────────────────────
run_id             str
input_hash          str
rules_version       str
model_version        str
threshold_version     str      NEW — thresholds are versioned separately from rules
policy_version       str      NEW — auto-commit policy is versioned separately
started_at          datetime
status              enum    RUNNING | COMPLETE | COMPLETED_WITH_EXCEPTIONS | DEGRADED | FAILED

NormalizedRecord
─────────────────────────────────────────────────────────
record_id           str
source              enum    BANK | GATEWAY | LEDGER
entity_type          enum    PAYMENT | BANK_CREDIT | GATEWAY_SETTLEMENT | LEDGER_ENTRY |
                             FEE | TAX | REFUND | CHARGEBACK | REVERSAL | ADJUSTMENT
amount_paise         int
currency             str
reference            str
counterparty          str      NEW — name/account label as it appears on that source's record; required
                              by Stage 2's and Stage 5's composite scoring (§6), which named
                              "counterparty evidence" as a signal without this field ever existing.
                              Deliberately NOT normalized at rest — same non-mutation rule as `reference`
                              (§ normalization section) — matching stages normalize/fuzzy-compare it,
                              never overwrite the stored value.
occurred_at          date
raw_hash             str

MatchGroup
─────────────────────────────────────────────────────────
group_id            str
cardinality          enum    ONE_TO_ONE | ONE_TO_MANY | MANY_TO_ONE | MANY_TO_MANY
expected_amount_paise  int
matched_amount_paise   int
residual_amount_paise  int
status               enum    VERIFIED | PENDING_REVIEW | REJECTED
evidence_score         float|null   the winning candidate's composite score
runner_up_score        float|null   NEW — second-best candidate's score, for margin proof
score_margin           float|null   NEW — evidence_score − runner_up_score
threshold_applied       float        NEW — the actual threshold this group was checked against
threshold_version       str          NEW — links back to ReconciliationRun.threshold_version
policy_checks          json         NEW — itemized pass/fail per check (conservation, currency, uniqueness, margin)
proposed_by           enum    (see §6)
verified_by            enum    (see §6)
commit_policy         enum    (see §6)
verification_result    enum    (see §6)
reviewed_by            str|null     NEW — set only once a human reviewer acts
reviewed_at            datetime|null NEW
review_action          enum|null    NEW — APPROVED | REJECTED | ESCALATED

MatchGroupMember   — replaces MatchAllocation, see §3
─────────────────────────────────────────────────────────
group_id / record_id / source / role / signed_amount_paise / allocated_amount_paise

DecisionEvent
─────────────────────────────────────────────────────────
event_id            str
group_id             str
stage                enum    STAGE1_EXACT | STAGE2_CONSTRAINED | STAGE3_AGGREGATE |
                            STAGE4_ADJUSTMENT | STAGE5_FUZZY | STAGE6_LLM | STAGE7_VERIFICATION
candidate_scores      json
reason_code           str
explanation           str
timestamp             datetime

Exception
─────────────────────────────────────────────────────────
*(Implemented in code as `ReconciliationException`, not `Exception` — Python's builtin `Exception` class means naming a Pydantic model `Exception` shadows it, forcing an aliased import everywhere the model is used. Renamed once, here, rather than carrying that friction through every later stage. The doc keeps calling it "Exception" as the conceptual term; only the code identifier differs.)*
group_id_or_record_id str
category              enum    (§8 taxonomy)
severity              enum    LOW | MEDIUM | HIGH
evidence              json
recommended_action      str
review_status           enum   OPEN | REVIEWED
```

---

## 8. Exception Taxonomy

`DUPLICATE_RECORD` · `MISSING_SETTLEMENT` · `ORPHAN_BANK_CREDIT` · `PARTIAL_SETTLEMENT` · `FEE_GST_MISMATCH` · `REFUND_CHARGEBACK_MISMATCH` · `CURRENCY_MISMATCH` · `REUSED_COUNTERPART` · `AMBIGUOUS_AGGREGATION` · `DATE_OUTSIDE_POLICY_WINDOW` · `INSUFFICIENT_EVIDENCE` · `LLM_UNAVAILABLE` · `LLM_OUTPUT_REJECTED_BY_VERIFIER`

---

## 9. Aggregation Search — Documented Limits

| Parameter | Default | Purpose |
|---|---|---|
| Max candidates per window | 20 | Keeps the many-to-one subset-sum search bounded and exact rather than heuristic |
| Max group size | 8 records | Reflects realistic consolidated-settlement sizes; larger claims are routed to `Exception: AMBIGUOUS_AGGREGATION` |
| Per-window timeout | 2 seconds | If exceeded, the window is abandoned and its records fall through to `Exception`, never left half-resolved |

(Tune against real synthetic-data density during Days 5–6; these are documented starting defaults, not hardcoded magic numbers.)

---

## 10. Tech Stack

| Layer | Choice |
|---|---|
| Language | Python 3.11+ |
| API | FastAPI 0.115+ |
| Validation | Pydantic v2 |
| Storage | SQLite, with a transactional budget counter — atomic increment via `BEGIN IMMEDIATE` or equivalent, so concurrent requests can't double-count or under-count Groq call usage |
| Amounts | integer paise, `Decimal` at parse boundary only |
| Identifier matching | `rapidfuzz` — retrieval only, never sufficient alone (§5) |
| Descriptive-text matching | `rapidfuzz` counterparty similarity, folded into the same identifier-retrieval pass — sentence-transformer embeddings were evaluated but never needed, since retrieval hit its precision/false-match target on `rapidfuzz` alone (see "Removed" note below) |
| 1:1 assignment | `scipy.optimize.linear_sum_assignment` (Hungarian), scoped to 1:1 only |
| Many-to-one aggregation | disjoint-windowed bounded subset-sum/DP, hard-limited per §9 — windows partitioned disjoint before search (§4) |
| Many-to-many | **not implemented — stated limitation** |
| LLM | Groq — Llama 4 Scout, recommend-only, defaults to `PENDING_REVIEW`, commit path narrowed per §2 |
| Forecasting (bonus) | **not implemented.** Gated behind the acceptance gate passing (§12) and never reached; the `statsmodels` dependency it would have used was removed rather than left unused (see "Removed" note below) |
| Dashboard | Streamlit |
| Packaging | Docker + docker-compose, datasets pre-baked |

**Removed:** PostgreSQL + pgvector. Listed at one point as a "scale path" but not actually used anywhere in this build (vector search only ever existed for the now-cut Q&A feature) — removed from the stack description rather than left as unused aspirational text.

**Removed:** `sentence-transformers` / `all-MiniLM-L6-v2`. Named at one point as the intended descriptive-text matching approach and briefly pre-baked into the Docker image, but a full search of `src/` found it was never actually imported anywhere — fuzzy candidate retrieval was built entirely on `rapidfuzz` and met its precision/false-match targets without it. Removed from `requirements.txt`/`pyproject.toml`, the Dockerfile's pre-bake step, and the now-unneeded HF/sentence-transformers env vars. No matching behavior changed — nothing importable was ever removed, only a dependency nothing called.

**Removed:** `sqlalchemy` and `statsmodels`. Both were listed in `requirements.txt`/`pyproject.toml` but a full search of `src/` found neither genuinely used anywhere — `sqlalchemy` because storage is plain SQLite (see the Storage row above), `statsmodels` because the forecasting bonus feature it would have backed was never built. Removed from both dependency files; the full test suite was re-verified passing in a clean environment with both absent.

---

## 11. Evaluation Plan

**Dataset size — one consistent target, replacing the earlier inconsistent phrasing:** 100 logical payment events, producing **200–300 physical rows** across the three sources (bank, gateway, ledger) — reflecting that one logical event often generates multiple physical records (a payment, its fee, its GST line, a possible refund).

**Calibration/evaluation separation:** two independently seeded datasets —
- **Calibration set** — used during Day 8 to tune Stage 2–5 thresholds and margins (Stage 2 shares Stage 3–5's evidence policy — see §2), and, in a later pass (BUILD_LOG.md's "Threshold recalibration"), to derive Stage 1's own `stage1_residual_tolerance_fraction` from the real residual distribution of genuinely correct Stage 1 matches (`scripts/calibrate_stage1_residual.py`) rather than a hand-guessed number.
- **Evaluation/demo set** — never touched during tuning, used only for the acceptance gate and the live demo.

This prevents thresholds from being implicitly overfit to the exact data judges will see, which would make the reported precision numbers meaningless.

**Anomaly mix:** exact/clean, dirty references, timing variance, net settlements, lifecycle events, structural cases, honest abstention (≥5 cases).

**Judge-facing metrics:**

| Metric | Definition |
|---|---|
| Auto-match precision | correct auto-matches / all auto-matches |
| Record coverage | matched records / eligible records |
| Value coverage | matched monetary value / eligible monetary value — **uses absolute value for refunds and chargebacks**, so netting a refund against its original payment never hides genuine reconciliation activity from the metric |
| False-match rate | incorrect committed matches / committed matches |
| Exception quality | correctly categorized unresolved cases / all unresolved cases |
| Tier contribution | precision + coverage broken down by proposing stage |
| Runtime and cost | records/sec, Groq calls made, estimated run cost |
| Bank credit coverage | of BANK-source, `entity_type=BANK_CREDIT` records that are genuinely part of a real settlement event, fraction that end up in some VERIFIED group (BANK-side REFUND/CHARGEBACK/REVERSAL legs are real BANK-source records but not credits landing in the account, so they're excluded from the denominator — see BUILD_LOG.md) |
| Complete cluster resolution | of all real settlement events, fraction whose full record set is covered EXACTLY — not merely as a correct subset — by a single VERIFIED group |

**On precision vs. closure:** the first seven metrics above measure whether a *claimed* relationship is correct — a VERIFIED group's own membership is a genuine subset of some real settlement event. A system can score 100% auto-match precision and 0% false-match rate while still leaving most settlement loops only partially closed, because a correct partial group and a fully-closed one both count as "correct" under those metrics. Bank credit coverage and complete cluster resolution measure closure directly and are always reported alongside precision/false-match rate, never as a replacement for them — see README.md's "Real results" section for current numbers on both datasets.

---

## 12. Cash Forecast (Bonus — gated)

Built only after the acceptance gate passes. Sourced from `MatchGroup` rows where `status = VERIFIED`. Holt-Winters, zero LLM calls, explicit confidence band and `trained_on_n`. **Status: not implemented** — the acceptance gate work occupied the full build, so this bonus was never reached; the `statsmodels` dependency it would have used was removed rather than left unused (§10).

---

## 13. Revised 13-Day Build Sequence

| Days | Deliverable |
|---|---|
| 1–2 | Ground-truth generator (calibration + evaluation sets, seeded separately), schemas, normalizers |
| 3–4 | Stage 1–2 (exact, constrained) matching; `DecisionEvent` + `Exception` model |
| 5–6 | Stage 3–4 (aggregation, adjustment rules) against calibration set; tune §9 limits |
| 7 | Financial and Evidence Verifier + stage-differentiated auto-commit policy engine (§2) |
| 8 | Hungarian (1:1) + disjoint-windowed many-to-one resolver; N:M explicitly stubbed as unsupported; Stage 2–5 threshold calibration finalized against the calibration set |
| 9 | Stage 5 fuzzy retrieval (never sufficient alone, §5); Stage 6 governed Groq recommendation, defaults to `PENDING_REVIEW`, zero-LLM degradation path |
| 10 | Evaluation harness against the untouched evaluation set — precision/coverage/false-match metrics, adversarial cases |
| 11 | Streamlit judge dashboard, proposal-vs-verification drill-down |
| 12 | Demo rehearsal, documentation, Docker (model pre-baked), failure injection |
| 13 | Buffer — forecast bonus only if every acceptance check in §14 passes |

---

## 14. Acceptance Gate

- One command starts the full demo on a clean machine — **including both calibration/evaluation datasets, pre-baked into the image, no cold-run generation needed.**
- A 200–300 physical-row batch finishes without manual intervention.
- Every `VERIFIED` group passes the full stage-differentiated auto-commit policy in §2, not just conservation.
- `MatchGroupMember` correctly represents every group spanning multiple sources — no pairwise-allocation ambiguity.
- 1:1 and many-to-one resolution is optimal with respect to the configured score function and bounded, disjoint candidate windows — not claimed as factual correctness; many-to-many is explicitly declared unsupported, not silently missing.
- Every stage-6 (`STAGE6_LLM`) proposal carries `commit_policy: HUMAN_REVIEW_REQUIRED` with no exceptions — there is no path by which an LLM proposal auto-commits.
- Evaluation is computed against the held-out evaluation set, never the calibration set.
- False-match rate, record coverage, value coverage displayed separately.
- Every unresolved record has a reason code, evidence, and recommended action.
- The system completes when the Groq key is missing, exhausted, or returns malformed JSON — budget counter is atomic, no race-condition double-spend of the call quota.
- No secrets or sensitive raw data in logs, prompts, screenshots, or the repo.
- Full demo completes reliably in under three minutes.

---

## 15. Three-Minute Demo Script

1. Upload a messy 200–300 row, three-source batch — show validation stats.
2. Run reconciliation — show exact, many-to-one, and adjustment-aware matches.
3. Show precision, record coverage, value coverage, false-match rate, runtime.
4. Drill into one hard `VERIFIED` group — show `proposed_by` vs `verified_by` and the conservation check.
5. Open a genuinely unresolved exception — explain why the system refused to guess.
6. Show a `STAGE6_LLM` proposal sitting in `PENDING_REVIEW` — explain why the LLM doesn't get to decide alone.
7. Disable the Groq key, rerun — prove graceful degradation live.
8. Close on the held-out evaluation numbers and the explicit many-to-many limitation, stated plainly.

**Closing line:** *It does not maximize the number of matches. It maximizes trustworthy matches, and proves every decision — including which ones it declined to make alone.*

---

## 16. Cut From Scope — Future Work

- **Tax-line matcher** — real reuse needs genuine classification/rule-evaluation logic, not a forced pass through the reconciliation engine.
- **Settlement Q&A / RAG** — a filterable structured view over `DecisionEvent`/`Exception` covers the demo need without added surface area.
- **Arbitrary many-to-many resolution** — would need ILP/min-cost flow properly implemented and tested; explicitly out of scope for this build, named here rather than left implicit.
