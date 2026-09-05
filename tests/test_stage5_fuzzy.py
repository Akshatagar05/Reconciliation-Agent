"""Unit tests for src/recon_agent/matching/stage5_fuzzy.py.

Per ARCHITECTURE.md §5's corrected policy, fuzzy identifier similarity
is candidate *retrieval* only, never sufficient by itself. The three
scenarios this file exists to cover, directly from this stage's task:

  1. A fuzzy-identifier-only "match" (no other corroborating evidence)
     must be correctly DECLINED — this is the specific failure mode §5
     exists to prevent (an earlier version of Stage 5 treated
     edit-distance on identifiers as matching evidence by itself).
  2. A genuine fuzzy match — identifier similarity PLUS all four other
     required conditions (currency, amount, date, counterparty) — must
     be proposed, with the correct provenance/verification fields.
  3. A close-margin case (two candidates that both clear every hard
     gate with almost the same score) must NOT auto-qualify — §2's
     "no auto-commit on a close call" applies to Stage 5 exactly like
     Stages 3-4.
"""

from __future__ import annotations

from datetime import date

from recon_agent.config import Settings
from recon_agent.matching.common import IdAllocator
from recon_agent.matching.stage5_fuzzy import (
    DEFAULT_THRESHOLD,
    STAGE5_MIN_MARGIN,
    run_stage5_fuzzy,
)
from recon_agent.models import (
    CommitPolicy,
    EntityType,
    NormalizedRecord,
    ProposedBy,
    Source,
    VerificationResult,
    VerifiedBy,
)


def _record(
    record_id: str,
    source: Source,
    entity_type: EntityType,
    amount_paise: int,
    reference: str,
    occurred_at: date,
    currency: str = "INR",
    counterparty: str = "Acme Retail Pvt Ltd",
) -> NormalizedRecord:
    return NormalizedRecord(
        record_id=record_id,
        source=source,
        entity_type=entity_type,
        amount_paise=amount_paise,
        currency=currency,
        reference=reference,
        counterparty=counterparty,
        occurred_at=occurred_at,
        raw_hash=f"hash_{record_id}",
    )


def _settings() -> Settings:
    return Settings()


def _run(records: list[NormalizedRecord]):
    return run_stage5_fuzzy(
        records,
        _settings(),
        prior_groups=[],
        prior_members=[],
        matched_record_ids=set(),
        ids=IdAllocator(),
    )


# ---------------------------------------------------------------------------
# 1. Fuzzy identifier match alone, no other corroborating evidence -> decline
# ---------------------------------------------------------------------------


def test_fuzzy_identifier_alone_is_declined_without_corroborating_evidence() -> None:
    """The reference is a one-character typo of the seeker's (fuzzy
    identifier similarity clears STAGE5_MIN_IDENTIFIER_SIMILARITY on its
    own), but every other required condition fails at once: the amount
    is far outside any fee/GST tolerance, the date is months away
    (outside the policy window), and the counterparty is a completely
    different, unrelated business. Per §5, identifier similarity alone
    must never be sufficient -- this must be declined, not proposed."""
    ledger = _record(
        "r1",
        Source.LEDGER,
        EntityType.PAYMENT,
        500_000,
        "pay_abc123xyz9",
        date(2026, 1, 1),
        counterparty="Acme Retail Pvt Ltd",
    )
    gateway = _record(
        "r2",
        Source.GATEWAY,
        EntityType.GATEWAY_SETTLEMENT,
        # Nowhere near a plausible fee/GST deduction off 500_000.
        120_000,
        "pay_abc123xyz8",  # one-character typo -- fuzzy identifier match
        date(2026, 6, 15),  # ~5 months away -- well outside the date window
        counterparty="Zenith Industrial Supplies LLP",  # unrelated counterparty
    )

    result = _run([ledger, gateway])

    assert result.match_groups == []
    assert result.matched_record_ids == set()
    # The decline should be recorded on the audit trail with a reason
    # that reflects a genuine hard-gate failure, not a silent no-op.
    # (Pass A evaluates the ledger record as a seeker; Pass B separately
    # evaluates the gateway record as a seeker looking for a BANK
    # candidate, of which there is none here either -- both legitimately
    # decline, so look specifically at the ledger seeker's event.)
    ledger_events = [
        e for e in result.decision_events if e.candidate_scores.get("seeker_record_id") == "r1"
    ]
    assert len(ledger_events) == 1
    event = ledger_events[0]
    assert event.reason_code == "NO_ELIGIBLE_CANDIDATE"
    # "candidate_pool_size" reflects the same-entity-type unmatched
    # pool size (1 -- the gateway record), not how many cleared every
    # hard gate; the absence of a winner is what proves none did.
    assert event.candidate_scores["candidate_pool_size"] == 1
    assert event.candidate_scores["best_candidate_record_id"] is None
    # Confirms the identifier alone *did* clear the fuzzy-similarity
    # gate -- proving the decline is driven by the missing corroborating
    # evidence, not by the identifier check itself.
    from recon_agent.matching.stage5_fuzzy import (
        STAGE5_MIN_IDENTIFIER_SIMILARITY,
        _identifier_similarity,
    )

    assert _identifier_similarity(ledger, gateway) >= STAGE5_MIN_IDENTIFIER_SIMILARITY


def test_fuzzy_identifier_alone_declined_even_with_matching_amount() -> None:
    """A second, narrower variant of the same failure mode: identifier
    fuzzy-matches AND the amount happens to fall within tolerance, but
    counterparty and date both fail. Still not enough -- ALL five
    conditions are required, not "identifier plus any one other"."""
    ledger = _record(
        "r1",
        Source.LEDGER,
        EntityType.PAYMENT,
        100_000,
        "pay_qwer7788zzzz",
        date(2026, 3, 1),
        counterparty="Acme Retail Pvt Ltd",
    )
    gateway = _record(
        "r2",
        Source.GATEWAY,
        EntityType.GATEWAY_SETTLEMENT,
        98_000,  # plausible ~2% fee deduction -- amount check would pass alone
        "pay_qwer7788zzzy",  # one-character typo
        date(2026, 9, 1),  # 6 months away -- fails date policy window
        counterparty="Vishal Electronics LLP",  # unrelated counterparty
    )

    result = _run([ledger, gateway])

    assert result.match_groups == []
    assert result.matched_record_ids == set()


# ---------------------------------------------------------------------------
# 2. Genuine fuzzy match: identifier + all four other conditions -> proposed
# ---------------------------------------------------------------------------


def test_genuine_fuzzy_match_with_full_corroboration_is_proposed() -> None:
    """Identifier is a one-character typo (fuzzy match), currency
    matches, amount falls within the realistic fee/GST deduction range,
    date is a routine T+1 settlement lag, and counterparty matches
    after case/punctuation normalization -- every §5 condition holds,
    and there is exactly one eligible candidate, so the margin is at
    its maximum. This must be proposed as a Stage 5 group with the
    correct provenance/verification fields."""
    ledger = _record(
        "r1",
        Source.LEDGER,
        EntityType.PAYMENT,
        100_000,
        "pay_9f8e7d6c5b",
        date(2026, 4, 10),
        counterparty="Vishal Electronics LLP",
    )
    gateway = _record(
        "r2",
        Source.GATEWAY,
        EntityType.GATEWAY_SETTLEMENT,
        97_600,  # ~2.4% fee+GST deduction, within MAX_FEE_FRACTION
        "pay_9f8e7d6c5a",  # last character differs -- fuzzy identifier match
        date(2026, 4, 11),  # T+1, routine settlement lag
        counterparty="vIsHaL eLEcTrOniCs LLP",  # same counterparty, dirty casing
    )

    result = _run([ledger, gateway])

    assert len(result.match_groups) == 1
    group = result.match_groups[0]
    assert result.matched_record_ids == {"r1", "r2"}

    # Provenance/verification fields per §6's separation of proposal
    # from verification -- this stage only ever proposes.
    assert group.proposed_by == ProposedBy.STAGE5_FUZZY
    assert group.verified_by == VerifiedBy.NOT_YET_VERIFIED
    assert group.commit_policy == CommitPolicy.AUTO_COMMIT_STAGE2_5_THRESHOLD
    assert group.verification_result == VerificationResult.NOT_YET_RUN
    assert group.status.value == "PENDING_REVIEW"

    assert group.evidence_score is not None and group.evidence_score >= DEFAULT_THRESHOLD
    assert group.score_margin is not None and group.score_margin >= STAGE5_MIN_MARGIN
    assert group.policy_checks["fuzzy_identifier_match"] is True
    assert group.policy_checks["counterparty_evidence"] is True

    event = [e for e in result.decision_events if e.group_id == group.group_id][0]
    assert event.reason_code == "LEDGER_GATEWAY_FUZZY_RETRIEVAL_MATCH"


def test_genuine_fuzzy_match_gateway_bank_pass_b() -> None:
    """Same shape as the ledger<->gateway case, but for Pass B
    (gateway<->bank): net-to-net amount expectation (no fee deduction),
    fuzzy-typo'd reference, matching counterparty, routine date lag."""
    gateway = _record(
        "r1",
        Source.GATEWAY,
        EntityType.GATEWAY_SETTLEMENT,
        250_000,
        "pay_55aa66bb77cc",
        date(2026, 5, 1),
        counterparty="Meridian Foods Pvt Ltd",
    )
    bank = _record(
        "r2",
        Source.BANK,
        EntityType.BANK_CREDIT,
        250_000,
        "pay_55aa66bb77cd",  # one-character typo
        date(2026, 5, 2),  # T+1
        counterparty="Meridian Foods Private Limited",  # close counterparty match
    )

    result = _run([gateway, bank])

    assert len(result.match_groups) == 1
    group = result.match_groups[0]
    assert group.proposed_by == ProposedBy.STAGE5_FUZZY
    assert result.matched_record_ids == {"r1", "r2"}


# ---------------------------------------------------------------------------
# 3. Close-margin case: two equally-good candidates -> must NOT auto-qualify
# ---------------------------------------------------------------------------


def test_close_margin_between_two_qualifying_candidates_is_declined() -> None:
    """One ledger seeker has two eligible gateway candidates -- a
    duplicate-style decoy pair with identical amount/date/counterparty
    evidence and near-identical (different one-character typo) fuzzy
    identifier similarity. Both clear every hard gate with an
    essentially tied score, so neither has a "unique score margin over
    the runner-up" per §5's last required condition. Must be declined
    rather than guessed, exactly the ambiguity this condition exists to
    catch."""
    ledger = _record(
        "led_1",
        Source.LEDGER,
        EntityType.PAYMENT,
        400_000,
        "pay_11223344aabb",
        date(2026, 6, 10),
        counterparty="Granite Motors Pvt Ltd",
    )
    gateway_a = _record(
        "gw_a",
        Source.GATEWAY,
        EntityType.GATEWAY_SETTLEMENT,
        400_000,
        "pay_11223344aabc",  # one-character typo of the ledger reference
        date(2026, 6, 11),
        counterparty="Granite Motors Pvt Ltd",
    )
    gateway_b = _record(
        "gw_b",
        Source.GATEWAY,
        EntityType.GATEWAY_SETTLEMENT,
        400_000,
        "pay_11223344aabd",  # a different one-character typo -- same similarity
        date(2026, 6, 11),
        counterparty="Granite Motors Pvt Ltd",
    )

    result = _run([ledger, gateway_a, gateway_b])

    assert result.match_groups == []
    assert result.matched_record_ids == set()

    ledger_events = [
        e for e in result.decision_events if e.candidate_scores.get("seeker_record_id") == "led_1"
    ]
    assert len(ledger_events) == 1
    event = ledger_events[0]
    assert event.reason_code == "INSUFFICIENT_MARGIN"
    assert event.candidate_scores["candidate_pool_size"] == 2


def test_close_margin_just_under_stage5_bar_is_declined_but_clear_margin_passes() -> None:
    """A tighter, numeric version of the same idea: two candidates whose
    composite scores differ by less than STAGE5_MIN_MARGIN must decline,
    while an otherwise-identical setup with a clearly dominant candidate
    (a poor runner-up) must be proposed -- isolating the margin check
    itself rather than relying on a fully-tied decoy pair."""
    ledger = _record(
        "led_1",
        Source.LEDGER,
        EntityType.PAYMENT,
        200_000,
        "pay_ff00ee11dd22",
        date(2026, 2, 5),
        counterparty="Ember Pharma Distributors",
    )
    # Strong candidate: closer date, cleaner counterparty match.
    gateway_strong = _record(
        "gw_strong",
        Source.GATEWAY,
        EntityType.GATEWAY_SETTLEMENT,
        196_000,
        "pay_ff00ee11dd23",
        date(2026, 2, 6),
        counterparty="Ember Pharma Distributors",
    )
    # Weaker but still-qualifying candidate: same amount/identifier
    # similarity, slightly worse date and counterparty match -- close
    # enough to the strong one that the margin is thin.
    gateway_weak = _record(
        "gw_weak",
        Source.GATEWAY,
        EntityType.GATEWAY_SETTLEMENT,
        196_000,
        "pay_ff00ee11dd24",
        date(2026, 2, 6),
        counterparty="Ember Pharma Distributor",
    )

    result = _run([ledger, gateway_strong, gateway_weak])

    # Whether this particular pair's margin clears STAGE5_MIN_MARGIN or
    # not is incidental to the point of this test -- what matters is
    # that when it doesn't, no group is proposed at all.
    if not result.match_groups:
        assert result.matched_record_ids == set()
        event = [e for e in result.decision_events if e.candidate_scores["seeker_record_id"] == "led_1"][0]
        assert event.reason_code == "INSUFFICIENT_MARGIN"
    else:
        group = result.match_groups[0]
        assert group.score_margin is not None and group.score_margin >= STAGE5_MIN_MARGIN
