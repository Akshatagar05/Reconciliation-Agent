"""Financial and Evidence Verifier — core policy checks (ARCHITECTURE.md §2).

Stage 5 of this (now 15-stage) relay build — see README.md's Build
Status. This module implements only the stage-differentiated
auto-commit policy table from §2, as a pure function:

    verify_match_group(group, members, records_by_id, settings=None)
        -> VerificationOutcome

Given a MatchGroup and the MatchGroupMembers proposed for it, decide
whether the group's evidence actually clears the bar its proposing
stage (``group.proposed_by``) claims — PASSED, or the specific
FAILED_* reason it missed by. This module deliberately:

  - does NOT mutate the MatchGroup it's given. ``status``,
    ``verified_by``, ``commit_policy``, and ``verification_result``
    on the input ``group`` are left untouched; ``VerificationOutcome``
    is a separate return value. Applying that outcome back onto a real
    MatchGroup (and updating ``reviewed_by``/``reviewed_at`` etc. for
    human-reviewed ones) is ``matching.pipeline.run_verification``'s
    job (Stage 6), not this module's.
  - has exactly one dependency on the ``matching`` package: the shared
    conservation arithmetic in ``matching.common`` (see the Stage 6
    bugfix note below). Nothing else here is imported by, or imports,
    any matching stage's own proposal logic.

§2's whole point is that conservation, currency, and margin checks are
"necessary but not sufficient" — two different candidate combinations
can land on the same monetary total, and both can conserve money while
only one is factually correct. So this is built as a genuine *second*
check, not a rubber stamp: Stage 2-5's margin and conservation are
recomputed directly from ``MatchGroupMember.role`` /
``signed_amount_paise``, never read back off the proposing stage's own
``evidence_score`` / ``policy_checks`` as if those were ground truth —
that would just be re-trusting the same math this module exists to
independently re-check.

On currency specifically: ``MatchGroupMember`` doesn't carry a
currency field at all (only ``NormalizedRecord`` does — see
``models/normalized_record.py`` and ``models/match_group_member.py``).
A genuinely independent currency check therefore needs each member's
underlying record, not just the group's own ``policy_checks`` dict
(reading that dict for the answer would be exactly the "trust the
proposing stage's own math" shortcut this module exists to avoid). For
the same reason, Stage 1's identifier-slot-uniqueness and date-window
checks are also re-derived from the underlying records here, rather
than by re-reading the group's stored ``policy_checks`` values.

Stage 6 bugfix note: the conservation arithmetic below
(``matching.common.recompute_group_conservation``) is now a *shared*
function also usable by Stage 3/4, rather than a second,
separately-written formula living only in this module — see that
function's docstring for the two conservation shapes (pure refund/
chargeback/reversal reflected across source legs; a settlement's
LEDGER-side pre-fee gross vs. its GATEWAY-side already-net counterpart)
a naive "sum every non-CREDIT role" reimplementation gets wrong. This
is the one place this module now depends on ``matching`` — a narrow,
one-directional dependency on shared *arithmetic*, not on any matching
stage's own proposal logic or stored evidence, so it doesn't reopen the
"trust the proposing stage's own math" shortcut this module otherwise
avoids throughout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from recon_agent.config import Settings, get_settings
from recon_agent.matching.common import (
    CONSERVATION_TOLERANCE_FRACTION,
    CONSERVATION_TOLERANCE_PAISE,
    recompute_group_conservation,
)
from recon_agent.models import (
    EntityType,
    MatchGroup,
    MatchGroupMember,
    MemberRole,
    NormalizedRecord,
    ProposedBy,
    Source,
    VerificationResult,
    VerifiedBy,
)

# ---------------------------------------------------------------------------
# Stage 1 policy mirror (ARCHITECTURE.md §2 / matching/stage1_exact.py's own
# STAGE1_DATE_POLICY_WINDOW_DAYS). Independently defined here rather than
# imported from matching.stage1_exact — this module's one dependency on the
# matching package is the shared conservation arithmetic above, not Stage
# 1's own proposal logic, so this constant is still duplicated on purpose:
# a small price for keeping that narrower boundary real rather than nominal.
# ---------------------------------------------------------------------------
STAGE1_DATE_POLICY_WINDOW_DAYS = 14

# Mirrors stage1_exact.py's own ``_GROSS_GATEWAY_ENTITY_TYPES`` — the
# GATEWAY-side entity types that represent a "primary claim" (as opposed
# to a FEE/TAX/ADJUSTMENT refinement) for the purposes of the informal
# expected/matched amounts a Stage 1 cluster carries. Independently
# defined here rather than imported, for the same "one dependency on
# matching.common's shared arithmetic only" reason given in the module
# docstring above.
_GROSS_GATEWAY_ENTITY_TYPES = frozenset(
    {
        EntityType.GATEWAY_SETTLEMENT,
        EntityType.REFUND,
        EntityType.CHARGEBACK,
        EntityType.REVERSAL,
    }
)

# Conservation tolerance now lives in matching.common
# (CONSERVATION_TOLERANCE_PAISE / CONSERVATION_TOLERANCE_FRACTION,
# imported above) alongside the shared conservation arithmetic itself —
# re-exported here so any existing external reference to
# ``verifier.CONSERVATION_TOLERANCE_PAISE`` /
# ``verifier.CONSERVATION_TOLERANCE_FRACTION`` keeps working.

# Roles whose signed_amount_paise sign is fixed by construction (§3/§7:
# "positive for credits, negative for debits/fees"; testdata/generator.py:
# "GROSS/CREDIT positive, FEE/TAX/REFUND/CHARGEBACK/REVERSAL negative,
# ADJUSTMENT signed either way"). Used by check (d), "no member's
# role/amount is internally contradictory" — MemberRole.ADJUSTMENT is
# intentionally absent from both sets.
_POSITIVE_ROLES = frozenset({MemberRole.GROSS, MemberRole.CREDIT})
_NEGATIVE_ROLES = frozenset(
    {
        MemberRole.FEE,
        MemberRole.TAX,
        MemberRole.REFUND,
        MemberRole.CHARGEBACK,
        MemberRole.REVERSAL,
    }
)


@dataclass(frozen=True)
class VerificationOutcome:
    """What the verifier decided about one MatchGroup.

    Deliberately not applied onto the MatchGroup itself — see the
    module docstring. A later stage reads this and writes
    ``verified_by`` / ``commit_policy`` / ``verification_result`` /
    ``status`` (and, for human review, ``reviewed_by`` etc.) onto the
    real MatchGroup.
    """

    group_id: str
    verification_result: VerificationResult
    verified_by: VerifiedBy
    reason_code: str
    explanation: str
    checks: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.verification_result == VerificationResult.PASSED


def _outcome(
    group: MatchGroup,
    result: VerificationResult,
    verified_by: VerifiedBy,
    reason_code: str,
    explanation: str,
    checks: dict[str, Any],
) -> VerificationOutcome:
    return VerificationOutcome(
        group_id=group.group_id,
        verification_result=result,
        verified_by=verified_by,
        reason_code=reason_code,
        explanation=explanation,
        checks=checks,
    )


def _min_score_margin(settings: Settings, threshold_version: str) -> float:
    return settings.min_score_margin_by_threshold_version.get(
        threshold_version, settings.default_min_score_margin
    )


# ---------------------------------------------------------------------------
# Stage 1 — re-check, don't just trust, identifier uniqueness + currency/date
# policy (§2's single combined "currency/date policy passes" condition).
# ---------------------------------------------------------------------------


def _stage1_recompute(
    members: Sequence[MatchGroupMember],
    records_by_id: Mapping[str, NormalizedRecord],
) -> Optional[tuple[bool, bool, bool]]:
    """Returns (identifier_unique, currency_ok, date_ok), independently
    re-derived from each member's underlying NormalizedRecord — or None
    if a member's record isn't available to verify against at all.

    "Identifier is unique" mirrors stage1_exact.py's own definition:
    no two members may occupy the same (source, entity_type) slot.
    """
    records: list[NormalizedRecord] = []
    for member in members:
        record = records_by_id.get(member.record_id)
        if record is None:
            return None
        records.append(record)

    seen_slots: set[tuple[Any, Any]] = set()
    identifier_unique = True
    for record in records:
        slot = (record.source, record.entity_type)
        if slot in seen_slots:
            identifier_unique = False
        seen_slots.add(slot)

    currencies = {record.currency for record in records}
    currency_ok = len(currencies) == 1

    dates = [record.occurred_at for record in records]
    date_ok = (max(dates) - min(dates)).days <= STAGE1_DATE_POLICY_WINDOW_DAYS

    return identifier_unique, currency_ok, date_ok


def _stage1_residual(records: Sequence[NormalizedRecord]) -> tuple[int, int]:
    """Independently recompute the same informational (expected, matched)
    view stage1_exact.py's own ``_amounts`` derives — a LEDGER-side gross
    amount vs. a GATEWAY-side already fee/GST-netted amount for the same
    event — re-derived here from each member's underlying record rather
    than trusted from the group's own stored
    expected_amount_paise/matched_amount_paise fields (this module's
    consistent "recompute, don't trust the proposing stage's own math"
    stance — see module docstring).

    Returns (expected, matched); when no LEDGER record is present in the
    cluster, expected is defined to equal matched (residual 0) — mirrors
    stage1_exact.py exactly, and keeps this a pure amount-of-evidence
    check, never a hard requirement that a LEDGER leg exist.
    """
    ledger_record = next((r for r in records if r.source == Source.LEDGER), None)
    gateway_gross_record = next(
        (
            r
            for r in records
            if r.source == Source.GATEWAY and r.entity_type in _GROSS_GATEWAY_ENTITY_TYPES
        ),
        None,
    )
    expected = abs(ledger_record.amount_paise) if ledger_record else 0
    matched = abs(gateway_gross_record.amount_paise) if gateway_gross_record else 0
    if ledger_record is None:
        expected = matched
    return expected, matched


def _verify_stage1(
    group: MatchGroup,
    members: Sequence[MatchGroupMember],
    records_by_id: Mapping[str, NormalizedRecord],
    settings: Settings,
) -> VerificationOutcome:
    recomputed = _stage1_recompute(members, records_by_id)
    if recomputed is None:
        return _outcome(
            group,
            VerificationResult.FAILED_UNIQUENESS,
            VerifiedBy.FINANCIAL_AND_EVIDENCE_VERIFIER,
            "MISSING_RECORD_EVIDENCE",
            "One or more members have no matching NormalizedRecord to "
            "independently verify identifier uniqueness and currency/date "
            "policy against; failing closed rather than trusting the "
            "group's own stored policy_checks.",
            {"records_available": False},
        )

    identifier_unique, currency_ok, date_ok = recomputed
    checks = {
        "identifier_unique": identifier_unique,
        "currency_consistent": currency_ok,
        "date_policy_window": date_ok,
    }

    if not identifier_unique:
        return _outcome(
            group,
            VerificationResult.FAILED_UNIQUENESS,
            VerifiedBy.FINANCIAL_AND_EVIDENCE_VERIFIER,
            "NON_UNIQUE_IDENTIFIER",
            "Independent re-check found two or more members occupying the "
            "same (source, entity_type) slot; the identifier does not "
            "uniquely pick a candidate for that slot.",
            checks,
        )

    if not (currency_ok and date_ok):
        # §2 names currency and date as one combined "currency/date policy"
        # condition for Stage 1, and VerificationResult has no separate
        # FAILED_DATE value — both failure modes of that combined condition
        # map onto FAILED_CURRENCY.
        reason = "CURRENCY_MISMATCH" if not currency_ok else "DATE_OUTSIDE_POLICY_WINDOW"
        return _outcome(
            group,
            VerificationResult.FAILED_CURRENCY,
            VerifiedBy.FINANCIAL_AND_EVIDENCE_VERIFIER,
            reason,
            f"Independent re-check of Stage 1's combined currency/date "
            f"policy failed ({reason}).",
            checks,
        )

    # §2's Stage 1 auto-commit condition is "exact identifier, unique,
    # currency/date policy passes" — deliberately no composite score,
    # because a shared unique reference is strong evidence on its own.
    # But without a ceiling, that same logic would silently verify two
    # completely unrelated transactions that happen to share a
    # reference by data-quality accident, with zero further scrutiny.
    # This bounded residual check is that ceiling: independently
    # re-derived from each member's own record (never the group's
    # stored expected/matched/residual fields), and bounded rather than
    # exact — realistic MDR + GST fee variance between a LEDGER-side
    # gross amount and its GATEWAY-side already-netted counterpart is
    # expected and healthy, only an *unexplained* gap beyond that band
    # is suspicious. A residual within the bound never fails this check
    # (including exactly zero) — it is purely a widening of what Stage
    # 1 already, correctly, treats as sufficient evidence.
    member_records = [records_by_id[m.record_id] for m in members]
    expected, matched = _stage1_residual(member_records)
    residual = abs(expected - matched)
    tolerance = int(
        settings.stage1_residual_tolerance_fraction * max(expected, matched)
    )
    checks["residual_paise"] = residual
    checks["residual_tolerance_paise"] = tolerance
    checks["residual_tolerance_fraction"] = settings.stage1_residual_tolerance_fraction
    checks["expected_amount_paise"] = expected
    checks["matched_amount_paise"] = matched
    within_tolerance = residual <= tolerance
    checks["residual_within_tolerance"] = within_tolerance

    if not within_tolerance:
        return _outcome(
            group,
            VerificationResult.RESIDUAL_EXCEEDS_TOLERANCE,
            VerifiedBy.FINANCIAL_AND_EVIDENCE_VERIFIER,
            "RESIDUAL_EXCEEDS_TOLERANCE",
            f"Identifier uniqueness and currency/date policy passed, but "
            f"the residual between the LEDGER-side amount ({expected} "
            f"paise) and the GATEWAY-side amount ({matched} paise) is "
            f"{residual} paise, which exceeds the configured tolerance "
            f"of {tolerance} paise "
            f"({settings.stage1_residual_tolerance_fraction:.1%} of the "
            "larger amount). The identifier match is still real "
            "evidence, but an unexplained gap this large needs a "
            "human's judgment, not an automatic pass.",
            checks,
        )

    return _outcome(
        group,
        VerificationResult.PASSED,
        VerifiedBy.FINANCIAL_AND_EVIDENCE_VERIFIER,
        "STAGE1_IDENTIFIER_VERIFIED",
        "Identifier uniqueness and currency/date policy independently "
        "re-derived from member records and confirmed; residual between "
        "LEDGER and GATEWAY legs is within the configured tolerance.",
        checks,
    )


# ---------------------------------------------------------------------------
# Stage 2-5 — margin, conservation (recomputed from raw member data), and
# currency, all independently re-checked rather than trusted from the
# proposing stage's own evidence_score / policy_checks.
#
# Stage 5 wiring note: §2's table names one shared evidence bar for
# "Stages 3-5 (aggregate, adjustment, fuzzy)", and Stage 2 shares that
# same bar per the v4.1 pre-coding correction (see this module's own
# earlier note and stage2_constrained.py's docstring) -- so Stage 5
# fuzzy-retrieval groups are policy-identical to Stage 2/3/4 groups here:
# same margin re-check, same conservation arithmetic, same currency
# re-derivation. This function was written to anticipate that (see the
# original "only STAGE5_FUZZY ... doesn't exist yet" note this replaces)
# and needed no new logic to cover it -- only the dispatch in
# ``verify_match_group`` below needed to actually route STAGE5_FUZZY
# here now that Stage 5 exists.
# ---------------------------------------------------------------------------


def _role_contradictions(members: Sequence[MatchGroupMember]) -> list[str]:
    """Check (d): "no member's role/amount is internally contradictory."

    A GROSS/CREDIT member with a non-positive amount, or a
    FEE/TAX/REFUND/CHARGEBACK/REVERSAL member with a non-negative
    amount, contradicts the fixed sign convention those roles carry by
    construction (§3/§7) — such a member would silently corrupt the
    conservation arithmetic below rather than genuinely fail it, so
    it's caught explicitly instead.
    """
    problems: list[str] = []
    for member in members:
        if member.role in _POSITIVE_ROLES and member.signed_amount_paise <= 0:
            problems.append(
                f"{member.record_id}: role {member.role.value} must be "
                f"positive, got {member.signed_amount_paise}"
            )
        elif member.role in _NEGATIVE_ROLES and member.signed_amount_paise >= 0:
            problems.append(
                f"{member.record_id}: role {member.role.value} must be "
                f"negative, got {member.signed_amount_paise}"
            )
    return problems


def _currency_check(
    members: Sequence[MatchGroupMember],
    records_by_id: Mapping[str, NormalizedRecord],
) -> Optional[tuple[bool, dict[str, Optional[str]]]]:
    """Independently re-derives currency per member from its underlying
    NormalizedRecord (MatchGroupMember itself carries no currency
    field). Returns (currency_ok, {record_id: currency}), or None if a
    member's record is missing.
    """
    currencies: dict[str, Optional[str]] = {}
    for member in members:
        record = records_by_id.get(member.record_id)
        if record is None:
            return None
        currencies[member.record_id] = record.currency
    currency_ok = len(set(currencies.values())) == 1
    return currency_ok, currencies


def _verify_stage2_to_5(
    group: MatchGroup,
    members: Sequence[MatchGroupMember],
    records_by_id: Mapping[str, NormalizedRecord],
    settings: Settings,
) -> VerificationOutcome:
    checks: dict[str, Any] = {}

    # (a) margin — independently compared against the verifier's own
    # configured minimum, not merely "the proposing stage said it was
    # sufficient" (that judgment already lives in group.score_margin, but
    # the bar it's measured against here is the verifier's own).
    min_margin = _min_score_margin(settings, group.threshold_version)
    margin_ok = group.score_margin is not None and group.score_margin >= min_margin
    checks["score_margin"] = group.score_margin
    checks["min_score_margin_required"] = min_margin
    checks["margin_sufficient"] = margin_ok
    if not margin_ok:
        return _outcome(
            group,
            VerificationResult.FAILED_MARGIN,
            VerifiedBy.FINANCIAL_AND_EVIDENCE_VERIFIER,
            "INSUFFICIENT_MARGIN",
            f"score_margin {group.score_margin!r} does not clear the "
            f"verifier's minimum margin {min_margin} for threshold_version "
            f"{group.threshold_version!r}.",
            checks,
        )

    # (d) role/amount contradictions. Caught before (b) because a
    # contradiction would otherwise silently corrupt the conservation sum
    # rather than genuinely fail it; reported as FAILED_CONSERVATION since
    # VerificationResult has no separate value for this specific check.
    contradictions = _role_contradictions(members)
    checks["role_amount_consistent"] = not contradictions
    if contradictions:
        checks["role_amount_contradictions"] = contradictions
        return _outcome(
            group,
            VerificationResult.FAILED_CONSERVATION,
            VerifiedBy.FINANCIAL_AND_EVIDENCE_VERIFIER,
            "CONTRADICTORY_MEMBER_ROLE_AMOUNT",
            "One or more members' role and signed_amount_paise sign "
            "contradict each other (e.g. a FEE/REFUND-role member with a "
            "non-negative amount): " + "; ".join(contradictions),
            checks,
        )

    # (b) conservation — recomputed directly from raw member data, never
    # from the group's own evidence_score/policy_checks (§2's "necessary
    # but not sufficient"), using the shared arithmetic in
    # matching.common (also used by Stage 3/4) so this module and the
    # proposing stages can never silently disagree about what
    # "conserves" means — see Stage 6 bugfix note in README.md.
    conservation = recompute_group_conservation(members)
    if conservation is None:
        checks["conservation_computable"] = False
        return _outcome(
            group,
            VerificationResult.FAILED_CONSERVATION,
            VerifiedBy.FINANCIAL_AND_EVIDENCE_VERIFIER,
            "NOTHING_TO_RECONCILE",
            "Group has no CREDIT-role member and no usable GROSS-role "
            "member to net the rest of the group against; conservation "
            "cannot be computed.",
            checks,
        )
    diff, tolerance = conservation
    conserves = abs(diff) <= tolerance
    checks["conservation_diff_paise"] = diff
    checks["conservation_tolerance_paise"] = tolerance
    checks["conservation_balances"] = conserves
    if not conserves:
        return _outcome(
            group,
            VerificationResult.FAILED_CONSERVATION,
            VerifiedBy.FINANCIAL_AND_EVIDENCE_VERIFIER,
            "CONSERVATION_DOES_NOT_BALANCE",
            f"Recomputed conservation diff {diff} paise exceeds tolerance "
            f"{tolerance} paise.",
            checks,
        )

    # (c) currency — re-derived from each member's own NormalizedRecord,
    # never read back off the group's stored policy_checks.
    currency = _currency_check(members, records_by_id)
    if currency is None:
        checks["currency_computable"] = False
        return _outcome(
            group,
            VerificationResult.FAILED_CURRENCY,
            VerifiedBy.FINANCIAL_AND_EVIDENCE_VERIFIER,
            "MISSING_RECORD_EVIDENCE",
            "One or more members have no matching NormalizedRecord to "
            "independently verify currency against; failing closed rather "
            "than trusting the group's own stored policy_checks.",
            checks,
        )
    currency_ok, currencies = currency
    checks["currency_consistent"] = currency_ok
    checks["member_currencies"] = currencies
    if not currency_ok:
        return _outcome(
            group,
            VerificationResult.FAILED_CURRENCY,
            VerifiedBy.FINANCIAL_AND_EVIDENCE_VERIFIER,
            "CURRENCY_MISMATCH",
            f"Independent re-check found inconsistent currencies across "
            f"members: {currencies}.",
            checks,
        )

    return _outcome(
        group,
        VerificationResult.PASSED,
        VerifiedBy.FINANCIAL_AND_EVIDENCE_VERIFIER,
        "MARGIN_CONSERVATION_CURRENCY_VERIFIED",
        "Margin, conservation, and currency all independently "
        "recomputed from raw member/record data and confirmed.",
        checks,
    )


# ---------------------------------------------------------------------------
# Stage 6 — no auto-commit path exists, ever (§2). Implemented now so a
# later stage that actually builds Stage 6 doesn't need to touch this file
# again; not testable end-to-end yet since Stage 6 doesn't exist.
# ---------------------------------------------------------------------------


def _verify_stage6(group: MatchGroup) -> VerificationOutcome:
    return _outcome(
        group,
        VerificationResult.NOT_YET_RUN,
        VerifiedBy.NOT_YET_VERIFIED,
        "STAGE6_ALWAYS_HUMAN_REVIEW",
        "Per ARCHITECTURE.md §2, Stage 6 (LLM recommendation) has no "
        "auto-commit path at all, regardless of evidence — the Financial "
        "and Evidence Verifier does not run automated margin/conservation/"
        "currency policy checks against LLM-proposed groups; a Stage 6 "
        "group's commit_policy is already HUMAN_REVIEW_REQUIRED at "
        "proposal time. verified_by is left NOT_YET_VERIFIED here (not "
        "set to FINANCIAL_AND_EVIDENCE_VERIFIER) because, per §6, only a "
        "verification step that actually evaluated the group changes "
        "verified_by away from NOT_YET_VERIFIED — and for Stage 6 that "
        "step can only ever be a human reviewer.",
        {},
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def verify_match_group(
    group: MatchGroup,
    members: Sequence[MatchGroupMember],
    records_by_id: Mapping[str, NormalizedRecord],
    settings: Optional[Settings] = None,
) -> VerificationOutcome:
    """Apply ARCHITECTURE.md §2's stage-differentiated auto-commit policy
    to one MatchGroup. Pure — does not mutate ``group`` or ``members``.

    Args:
        group: the candidate MatchGroup to evaluate.
        members: every MatchGroupMember belonging to ``group.group_id``.
        records_by_id: a ``record_id -> NormalizedRecord`` lookup covering
            every member's ``record_id``, used for the data
            MatchGroupMember itself doesn't carry (currency; and, for
            Stage 1, entity_type/occurred_at) so those checks are
            independently re-derived rather than read back off the
            group's own stored policy_checks.
        settings: defaults to ``config.get_settings()``; pass an explicit
            ``Settings`` to control ``min_score_margin_by_threshold_version``
            in tests without touching process-wide/env state.

    Raises:
        NotImplementedError: for any ``proposed_by`` this stage of the
            build doesn't implement a policy branch for yet. As of
            Stage 5's build, every ``ProposedBy`` value is covered;
            this only guards against a *future* enum addition reaching
            here with no policy branch. Raising rather than silently
            mis-verifying is the fail-closed choice consistent with §2.
    """
    settings = settings if settings is not None else get_settings()

    if group.proposed_by == ProposedBy.STAGE1_EXACT:
        return _verify_stage1(group, members, records_by_id, settings)

    if group.proposed_by in (
        ProposedBy.STAGE2_CONSTRAINED,
        ProposedBy.STAGE3_AGGREGATE,
        ProposedBy.STAGE4_ADJUSTMENT,
        ProposedBy.STAGE5_FUZZY,
    ):
        return _verify_stage2_to_5(group, members, records_by_id, settings)

    if group.proposed_by == ProposedBy.STAGE6_LLM:
        return _verify_stage6(group)

    raise NotImplementedError(
        f"verify_match_group has no policy branch for proposed_by="
        f"{group.proposed_by!r} yet — only STAGE1_EXACT, "
        "STAGE2_CONSTRAINED, STAGE3_AGGREGATE, STAGE4_ADJUSTMENT, "
        "STAGE5_FUZZY, and STAGE6_LLM are implemented at this stage of "
        "the build."
    )
