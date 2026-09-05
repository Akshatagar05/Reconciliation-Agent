"""Shared helpers for the Stage 1 (exact) and Stage 2 (constrained)
matchers — ARCHITECTURE.md §6.

Kept here rather than duplicated in both stage modules: id allocation,
the entity-type -> MatchGroupMember.role mapping (§3/§7), the
ledger<->gateway and gateway<->bank entity-type pairing tables that
define which records are even eligible to be matched against each
other, and small constructors for MatchGroupMember / DecisionEvent.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional, Sequence

from recon_agent.models import (
    DecisionEvent,
    DecisionStage,
    EntityType,
    MatchGroupMember,
    MemberRole,
    NormalizedRecord,
    Source,
)

# ---------------------------------------------------------------------------
# Entity-type -> MatchGroupMember.role (§3/§7). Mirrors the mapping used by
# the Stage 2 generator so a proposed group's roles line up with how the
# data was constructed.
# ---------------------------------------------------------------------------

ENTITY_TO_ROLE: dict[EntityType, MemberRole] = {
    EntityType.PAYMENT: MemberRole.GROSS,
    EntityType.GATEWAY_SETTLEMENT: MemberRole.GROSS,
    EntityType.LEDGER_ENTRY: MemberRole.GROSS,
    EntityType.BANK_CREDIT: MemberRole.CREDIT,
    EntityType.FEE: MemberRole.FEE,
    EntityType.TAX: MemberRole.TAX,
    EntityType.REFUND: MemberRole.REFUND,
    EntityType.CHARGEBACK: MemberRole.CHARGEBACK,
    EntityType.REVERSAL: MemberRole.REVERSAL,
    EntityType.ADJUSTMENT: MemberRole.ADJUSTMENT,
}

# ---------------------------------------------------------------------------
# Which LEDGER entity_type is the expected counterpart of which GATEWAY
# entity_type, and which GATEWAY entity_type is the expected counterpart of
# which BANK entity_type. REVERSAL has no bank leg by design (a reversed
# payment never reaches the bank) so it's absent from the second table.
# ---------------------------------------------------------------------------

LEDGER_TO_GATEWAY_ENTITY: dict[EntityType, EntityType] = {
    EntityType.PAYMENT: EntityType.GATEWAY_SETTLEMENT,
    EntityType.REFUND: EntityType.REFUND,
    EntityType.CHARGEBACK: EntityType.CHARGEBACK,
    EntityType.REVERSAL: EntityType.REVERSAL,
}

GATEWAY_TO_BANK_ENTITY: dict[EntityType, EntityType] = {
    EntityType.GATEWAY_SETTLEMENT: EntityType.BANK_CREDIT,
    EntityType.REFUND: EntityType.REFUND,
    EntityType.CHARGEBACK: EntityType.CHARGEBACK,
}


class IdAllocator:
    """Sequential id generator shared across the Stage 1/2 pipeline run.

    Plain incrementing counters are sufficient (and preferable) here:
    unlike the Stage 2 generator, matching is a deterministic function of
    its input data, not something that needs to be independently
    reproducible from a random seed — so there's no reason to reach for
    anything fancier than a counter.
    """

    def __init__(self, prefix: str = "match") -> None:
        self._prefix = prefix
        self._group_n = 0
        self._event_n = 0

    def next_group_id(self) -> str:
        self._group_n += 1
        return f"{self._prefix}_grp_{self._group_n:05d}"

    def next_event_id(self) -> str:
        self._event_n += 1
        return f"{self._prefix}_evt_{self._event_n:05d}"


def make_member(group_id: str, record: NormalizedRecord) -> MatchGroupMember:
    """Build the MatchGroupMember row for one record's participation in a
    group. allocated_amount_paise equals signed_amount_paise here since
    Stage 1/2 only ever propose whole-record-into-one-group matches; a
    record split across multiple groups (partial allocation) isn't
    something these two stages produce.
    """
    return MatchGroupMember(
        group_id=group_id,
        record_id=record.record_id,
        source=record.source,
        role=ENTITY_TO_ROLE[record.entity_type],
        signed_amount_paise=record.amount_paise,
        allocated_amount_paise=record.amount_paise,
    )


def make_decision_event(
    ids: IdAllocator,
    group_id: str,
    stage: DecisionStage,
    candidate_scores: dict[str, Any],
    reason_code: str,
    explanation: str,
    timestamp: Optional[datetime] = None,
) -> DecisionEvent:
    return DecisionEvent(
        event_id=ids.next_event_id(),
        group_id=group_id,
        stage=stage,
        candidate_scores=candidate_scores,
        reason_code=reason_code,
        explanation=explanation,
        timestamp=timestamp if timestamp is not None else datetime.now(),
    )


def date_gap_days(a: NormalizedRecord, b: NormalizedRecord) -> int:
    return abs((a.occurred_at - b.occurred_at).days)


def currencies_match(a: NormalizedRecord, b: NormalizedRecord) -> bool:
    return a.currency == b.currency


# Sources, for readability at call sites.
BANK = Source.BANK
GATEWAY = Source.GATEWAY
LEDGER = Source.LEDGER


# ---------------------------------------------------------------------------
# Shared conservation-equation arithmetic — used by Stage 3 (aggregate),
# Stage 4 (adjustment), and the Financial and Evidence Verifier
# (verification/verifier.py) so all three ultimately agree on what
# "conserves" means for a group's raw MatchGroupMember rows, instead of
# each recomputing its own version. (Historically the verifier had its
# own, separately-written, and subtly wrong reimplementation here — see
# the Stage 6 bugfix note in README.md.)
#
# Two things a naive "sum every non-CREDIT signed_amount_paise" misses,
# both confirmed against real calibration/evaluation data:
#
#  1. A pure refund/chargeback/reversal event surfaces as up to three
#     MatchGroupMember rows (ledger, gateway, bank), all sharing the
#     SAME role and the SAME signed_amount_paise — testdata/generator.py
#     stores that one real money movement identically on every source
#     leg, per its own "role/amount is fixed by construction" note. Such
#     a group has no CREDIT and no GROSS leg at all to net the rest
#     against (it's a pure outflow, not a payment), so summing "the
#     rest" against nothing is meaningless — but three-way agreement on
#     one amount trivially IS conservation.
#
#  2. Per testdata/generator.py's own design note, "the settlement
#     amount is always net of MDR fee + GST, whether or not the FEE/TAX
#     rows are separately broken out for that event." That means, for
#     one real settlement event:
#       - its LEDGER-source GROSS (PAYMENT) row carries the pre-fee
#         gross amount;
#       - its GATEWAY-source GROSS (GATEWAY_SETTLEMENT) row carries the
#         amount *after* fee/GST/adjustment are already netted out —
#         whether or not FEE/TAX/ADJUSTMENT rows for that same event
#         are separately present in the group;
#     so a GATEWAY-source GROSS row and its LEDGER-source counterpart
#     are two views of the SAME money, not two legs to add together —
#     and any GATEWAY-source FEE/TAX/ADJUSTMENT breakout rows riding
#     alongside that settlement are already folded into its amount, not
#     an additional deduction to apply on top. A group only ever reaches
#     this state already carrying both a LEDGER GROSS and a GATEWAY
#     GROSS leg for the same event (Stage 1/2 always propose that pair
#     together; Stage 3/4 only ever add a *standalone* item when its
#     source is GATEWAY and it independently reached BANK_CREDIT/REFUND/
#     CHARGEBACK — see stage3_aggregate.py's/stage4_adjustment.py's own
#     ``GATEWAY_TO_BANK_ENTITY`` / ``_NETTABLE_ENTITY_TYPES`` gating), so
#     this exclusion never drops a leg that was the only evidence of its
#     own money. Naively summing both sides (as the verifier's old
#     ``_conservation_diff`` did) double-counts that one settlement's
#     money and reads as a large, spurious conservation failure — this
#     is exactly the shape Stage 3/4's own ``expected_amount_paise`` /
#     ``matched_amount_paise`` bookkeeping never falls into, since they
#     track one net amount per settlement unit rather than summing raw
#     member rows by role.
# ---------------------------------------------------------------------------

CONSERVATION_TOLERANCE_PAISE = 100
CONSERVATION_TOLERANCE_FRACTION = 0.001

# Roles a pure non-payment outflow (refund/chargeback/reversal) can
# legitimately appear as on every one of its source legs (§3/§7: "signed
# per its role by construction"). Mirrors verifier.py's own
# ``_NEGATIVE_ROLES`` (kept in sync, not imported, so this module has no
# dependency in that direction).
_SAME_LEG_OUTFLOW_ROLES = frozenset(
    {
        MemberRole.FEE,
        MemberRole.TAX,
        MemberRole.REFUND,
        MemberRole.CHARGEBACK,
        MemberRole.REVERSAL,
    }
)

# GATEWAY-source roles that, per the generator's own design note, are
# always already folded into a same-group GATEWAY-source GROSS
# (GATEWAY_SETTLEMENT) row's net amount, so must not also be subtracted
# separately when that anchor is present.
_GATEWAY_BREAKOUT_ROLES = frozenset({MemberRole.FEE, MemberRole.TAX, MemberRole.ADJUSTMENT})


def conservation_tolerance_for(amount_paise: int) -> int:
    """Shared tolerance policy: a small fixed floor plus a proportional
    allowance, mirroring Stage 3/4's own ``_tolerance_fn``."""
    return max(
        CONSERVATION_TOLERANCE_PAISE,
        int(CONSERVATION_TOLERANCE_FRACTION * abs(amount_paise)),
    )


def recompute_group_conservation(
    members: Sequence[MatchGroupMember],
) -> Optional[tuple[int, int]]:
    """Independently recompute a group's conservation diff straight from
    its raw ``MatchGroupMember`` rows (role, source, signed_amount_paise)
    — no group-level stored totals, no proposing stage's own bookkeeping.

    Returns ``(diff, tolerance)`` where ``diff`` is
    ``sum(non-CREDIT side) - sum(CREDIT side)`` (or the max pairwise
    spread for the same-leg-outflow special case below), or ``None`` if
    there is nothing to reconcile against at all (no CREDIT and no
    usable GROSS leg).
    """
    roles_present = {m.role for m in members}
    if len(roles_present) == 1 and next(iter(roles_present)) in _SAME_LEG_OUTFLOW_ROLES:
        # Case 1 above: one refund/chargeback/reversal event reflected
        # identically across up to three source legs — no CREDIT/GROSS
        # leg exists to net against, and none is needed: agreement
        # across legs on the same amount already *is* conservation.
        amounts = [m.signed_amount_paise for m in members]
        reference = amounts[0]
        diff = max(abs(a - reference) for a in amounts)
        return diff, conservation_tolerance_for(reference)

    non_credit = [m for m in members if m.role != MemberRole.CREDIT]

    # Case 2 above only arises when the group carries BOTH views of the
    # same settlement event — a LEDGER-source GROSS (pre-fee) row and a
    # GATEWAY-source GROSS (already-net) row for it. That LEDGER-source
    # row's presence is exactly the signal that a same-group
    # GATEWAY-source GROSS is the *net* figure with fee/GST/adjustment
    # already folded in (per the generator's design note), so it (and
    # any GATEWAY-source FEE/TAX/ADJUSTMENT breakout riding alongside
    # it) must not also be netted separately. Without a LEDGER-source
    # GROSS present at all, there is no such duplicate view to worry
    # about — a lone GATEWAY-source GROSS plus its own standalone
    # FEE/TAX/REFUND/CHARGEBACK/ADJUSTMENT legs is exactly Stage 4's
    # genuine role-based netting shape (§4), and must still be summed
    # in full.
    has_ledger_gross = any(
        m.role == MemberRole.GROSS and m.source == Source.LEDGER for m in non_credit
    )
    if has_ledger_gross:

        def _already_reflected_in_gateway_settlement(member: MatchGroupMember) -> bool:
            if member.role == MemberRole.GROSS and member.source == Source.LEDGER:
                return True
            if member.role in _GATEWAY_BREAKOUT_ROLES and member.source == Source.GATEWAY:
                return True
            return False

        effective_non_credit = [
            m for m in non_credit if not _already_reflected_in_gateway_settlement(m)
        ]
    else:
        effective_non_credit = non_credit

    credit_members = [m for m in members if m.role == MemberRole.CREDIT]
    if credit_members:
        target = sum(m.signed_amount_paise for m in credit_members)
        rest = effective_non_credit
    else:
        gross_candidates = [m for m in effective_non_credit if m.role == MemberRole.GROSS]
        if not gross_candidates:
            return None
        reference_member = gross_candidates[0]
        target = reference_member.signed_amount_paise
        rest = [m for m in effective_non_credit if m is not reference_member]

    diff = sum(m.signed_amount_paise for m in rest) - target
    tolerance = conservation_tolerance_for(target)
    return diff, tolerance
