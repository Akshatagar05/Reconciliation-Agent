"""Stage 1 — exact match — ARCHITECTURE.md §6, §2.

§2's Stage 1 auto-commit condition: "Exact normalized identifier match
AND identifier is unique AND currency/date policy passes." This is the
only stage that doesn't need a composite score (§2) — an exact, unique
identifier is sufficient by itself.

What "candidate" means here: records (of any source) that share the same
*normalized* reference (see ``normalization.reference``) cluster
together — in this dataset that recovers a payment event's LEDGER row
together with its GATEWAY-side rows (settlement, and fee/tax/adjustment
when broken out), since gateway-side rows are generated sharing one
reference per event and bank credits never share an identifier with
either (they carry a UTR-style batch reference — see §2's own note that
this is *exactly* why Stage 2 exists). So a Stage 1 proposal never
includes a BANK record; that leg, when findable at all, is always
Stage 2's job.

"Identifier is unique" is enforced at the (source, entity_type) slot
level: if two records in a reference-cluster occupy the *same* slot
(e.g. two GATEWAY_SETTLEMENT rows with the same reference — exactly
what the generator's injected duplicates look like), the identifier
doesn't uniquely pick a candidate for that slot, so the whole cluster is
declined rather than guessing which one is real. This is deliberately
conservative: a smaller exact claim is stronger than a broad unreliable
one (§1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from recon_agent.config import Settings
from recon_agent.matching.common import (
    IdAllocator,
    currencies_match,
    date_gap_days,
    make_decision_event,
    make_member,
)
from recon_agent.models import (
    Cardinality,
    CommitPolicy,
    DecisionStage,
    EntityType,
    MatchGroup,
    MatchGroupMember,
    MatchGroupStatus,
    NormalizedRecord,
    ProposedBy,
    Source,
    VerificationResult,
    VerifiedBy,
)
from recon_agent.normalization import normalize_reference

# Maximum allowed spread between the earliest and latest occurred_at
# within one Stage 1 cluster. Generous relative to the generator's actual
# T+0..T+3 (or month-boundary) gaps, deliberately: Stage 1's job is to
# catch clusters that clearly belong together and reject collisions, not
# to enforce a tight timing model — that nuance belongs to Stage 2/3.
STAGE1_DATE_POLICY_WINDOW_DAYS = 14

# Entity types that represent a "primary claim" (as opposed to a
# refinement like FEE/TAX/ADJUSTMENT) — used to compute the informational
# expected/matched/residual amounts on a proposed group.
_GROSS_GATEWAY_ENTITY_TYPES = frozenset(
    {
        EntityType.GATEWAY_SETTLEMENT,
        EntityType.REFUND,
        EntityType.CHARGEBACK,
        EntityType.REVERSAL,
    }
)

# A refund/chargeback/reversal legitimately carries the *same* reference
# as its original payment (it's the same underlying transaction) but is a
# financially distinct event that must never be merged into the original
# payment's group. Clustering therefore keys on (normalized_reference,
# entity family) rather than reference alone, so "same reference, opposite
# financial event" doesn't collapse into one cluster.
_REFUND_LIKE = frozenset({EntityType.REFUND, EntityType.CHARGEBACK, EntityType.REVERSAL})


def _entity_family(entity_type: EntityType) -> str:
    if entity_type in _REFUND_LIKE:
        return entity_type.value
    return "principal"  # PAYMENT, GATEWAY_SETTLEMENT, BANK_CREDIT, FEE, TAX, ADJUSTMENT, LEDGER_ENTRY


@dataclass
class Stage1Result:
    match_groups: list[MatchGroup] = field(default_factory=list)
    match_group_members: list[MatchGroupMember] = field(default_factory=list)
    decision_events: list = field(default_factory=list)
    matched_record_ids: set[str] = field(default_factory=set)


def _slot_unique(cluster: list[NormalizedRecord]) -> bool:
    """False if two records in the cluster occupy the same (source,
    entity_type) slot — the "identifier is unique" condition."""
    seen: set[tuple[Source, EntityType]] = set()
    for record in cluster:
        slot = (record.source, record.entity_type)
        if slot in seen:
            return False
        seen.add(slot)
    return True


def _date_policy_passes(cluster: list[NormalizedRecord]) -> bool:
    dates = [r.occurred_at for r in cluster]
    return (max(dates) - min(dates)).days <= STAGE1_DATE_POLICY_WINDOW_DAYS


def _currency_policy_passes(cluster: list[NormalizedRecord]) -> bool:
    currencies = {r.currency for r in cluster}
    return len(currencies) == 1


def _amounts(cluster: list[NormalizedRecord]) -> tuple[int, int]:
    """(expected_amount_paise, matched_amount_paise) — informational only;
    proving conservation is the not-yet-built verifier's job, not ours."""
    ledger_record = next((r for r in cluster if r.source == Source.LEDGER), None)
    gateway_gross_record = next(
        (
            r
            for r in cluster
            if r.source == Source.GATEWAY and r.entity_type in _GROSS_GATEWAY_ENTITY_TYPES
        ),
        None,
    )
    expected = abs(ledger_record.amount_paise) if ledger_record else 0
    matched = abs(gateway_gross_record.amount_paise) if gateway_gross_record else 0
    if ledger_record is None:
        expected = matched
    return expected, matched


def run_stage1_exact(
    records: list[NormalizedRecord],
    settings: Settings,
    ids: IdAllocator,
) -> Stage1Result:
    result = Stage1Result()

    clusters: dict[tuple[str, str], list[NormalizedRecord]] = {}
    for record in records:
        norm_ref = normalize_reference(record.reference)
        key = (norm_ref, _entity_family(record.entity_type))
        clusters.setdefault(key, []).append(record)

    # Sorted for deterministic output order given a fixed input list.
    for cluster_key in sorted(clusters):
        norm_ref, _family = cluster_key
        cluster = clusters[cluster_key]
        if len(cluster) < 2:
            continue  # nothing to match against — not a candidate at all

        cluster = sorted(cluster, key=lambda r: r.record_id)
        slot_unique = _slot_unique(cluster)
        currency_ok = _currency_policy_passes(cluster)
        date_ok = _date_policy_passes(cluster) if slot_unique else False

        candidate_group_id = ids.next_group_id()
        record_ids = [r.record_id for r in cluster]
        policy_checks = {
            "identifier_unique": slot_unique,
            "currency_consistent": currency_ok,
            "date_policy_window": date_ok,
        }

        if slot_unique and currency_ok and date_ok:
            expected, matched = _amounts(cluster)
            group = MatchGroup(
                group_id=candidate_group_id,
                cardinality=Cardinality.ONE_TO_ONE,
                expected_amount_paise=expected,
                matched_amount_paise=matched,
                residual_amount_paise=expected - matched,
                status=MatchGroupStatus.PENDING_REVIEW,
                evidence_score=None,
                runner_up_score=None,
                score_margin=None,
                threshold_applied=1.0,
                threshold_version=settings.threshold_version,
                policy_checks=policy_checks,
                proposed_by=ProposedBy.STAGE1_EXACT,
                verified_by=VerifiedBy.NOT_YET_VERIFIED,
                commit_policy=CommitPolicy.AUTO_COMMIT_STAGE1,
                verification_result=VerificationResult.NOT_YET_RUN,
            )
            result.match_groups.append(group)
            for record in cluster:
                result.match_group_members.append(make_member(candidate_group_id, record))
                result.matched_record_ids.add(record.record_id)

            result.decision_events.append(
                make_decision_event(
                    ids,
                    candidate_group_id,
                    DecisionStage.STAGE1_EXACT,
                    candidate_scores={
                        "normalized_reference": norm_ref,
                        "cluster_record_ids": record_ids,
                        "policy_checks": policy_checks,
                    },
                    reason_code="EXACT_NORMALIZED_REFERENCE_MATCH",
                    explanation=(
                        f"{len(cluster)} records share normalized reference "
                        f"'{norm_ref}'; identifier unique per (source, entity_type) "
                        "slot; currency and date policy checks passed."
                    ),
                )
            )
        else:
            reason = (
                "NON_UNIQUE_IDENTIFIER"
                if not slot_unique
                else "CURRENCY_MISMATCH"
                if not currency_ok
                else "DATE_OUTSIDE_POLICY_WINDOW"
            )
            result.decision_events.append(
                make_decision_event(
                    ids,
                    candidate_group_id,
                    DecisionStage.STAGE1_EXACT,
                    candidate_scores={
                        "normalized_reference": norm_ref,
                        "cluster_record_ids": record_ids,
                        "policy_checks": policy_checks,
                    },
                    reason_code=reason,
                    explanation=(
                        f"{len(cluster)} records share normalized reference "
                        f"'{norm_ref}' but failed a Stage 1 policy check "
                        f"({policy_checks}); declined rather than guessing."
                    ),
                )
            )

    return result
