"""Synthetic ground-truth data generator — ARCHITECTURE.md §11.

Produces two independently seeded datasets — CALIBRATION and EVALUATION —
of synthetic bank / gateway / ledger ``NormalizedRecord`` rows (§7),
plus a hidden ground-truth mapping of which records truly belong
together. The matching pipeline built in later stages must only ever
read the ``NormalizedRecord`` batch files under ``data/``; the ground
truth under ``ground_truth/`` exists solely for the Day-10 evaluation
harness (§11, §13) and is never touched by the matching pipeline itself.

Design notes on how the required §11 anomaly mix maps to generation:

- Every one of the 100 logical payment events gets a LEDGER ``PAYMENT``
  row and a GATEWAY ``GATEWAY_SETTLEMENT`` row (the settlement amount is
  always net of MDR fee + GST, whether or not the FEE/TAX rows are
  separately broken out for that event).
- Most events' settlement amounts flow into periodic BANK ``BANK_CREDIT``
  payout batches (several events' net settlements consolidated into one
  bank credit) rather than a dedicated bank row per event — this is what
  real gateway payouts look like, and it is also what keeps the total
  physical-row count inside the 200-300 target for 100 logical events
  (100 ledger rows + 100 gateway rows + ~12-15 consolidated bank rows is
  already close to budget, before any anomaly extras). A handful of
  events are pulled out of the shared payout pool to build the other
  required cases (lifecycle events, partial settlements, missing rows).
- "Dirty reference", "timing variance", "net settlement breakout", and
  "duplicate" are applied as independent modifiers layered on top of
  whichever structural type an event has, rather than as separate event
  types — this lets one event carry more than one anomaly at once (as
  real messy data does) without inflating the event count.
- "Consolidated many-to-one settlement" is not a separate event type
  either — it falls out naturally: any payout batch with more than one
  member *is* a many-to-one consolidated settlement, and batches are
  sized 5-9 members, so this case appears constantly.

Regenerate both datasets deterministically via the CLI::

    python -m recon_agent.testdata.generator --seed calibration
    python -m recon_agent.testdata.generator --seed evaluation
    python -m recon_agent.testdata.generator --seed both   # default

An arbitrary on-demand stress-test dataset (same anomaly-mix logic, a
different RNG seed, never touching calibration/evaluation's fixed
seeds or content) can also be generated::

    python -m recon_agent.testdata.generator --seed stress --seed-value 12345 --logical-events 100

See ``scripts/multi_seed_evaluation.py`` for running many of these at
once and aggregating metrics across them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

from recon_agent.models import EntityType, NormalizedRecord, Source

# ---------------------------------------------------------------------------
# Seeds — independently fixed and distinct per §11's calibration/evaluation
# separation. Never derive one from the other and never let a value chosen
# while generating one dataset influence the other.
# ---------------------------------------------------------------------------

CALIBRATION_SEED = 20260501
EVALUATION_SEED = 20260930

DATASET_SEEDS: dict[str, int] = {
    "calibration": CALIBRATION_SEED,
    "evaluation": EVALUATION_SEED,
}
DATASET_PREFIXES: dict[str, str] = {
    "calibration": "cal",
    "evaluation": "evl",
}

# ---------------------------------------------------------------------------
# Anomaly-category labels — the vocabulary used to tag ground-truth groups
# and records. REQUIRED_CATEGORIES is exactly the §11 anomaly mix (plus
# "clean" for exact/clean matches); tests assert every one of these appears
# at least once per dataset.
# ---------------------------------------------------------------------------

CATEGORY_CLEAN = "clean"
CATEGORY_DIRTY_REFERENCE = "dirty_reference"
CATEGORY_TIMING_VARIANCE = "timing_variance"
CATEGORY_NET_SETTLEMENT = "net_settlement"
CATEGORY_REFUND = "lifecycle_refund"
CATEGORY_CHARGEBACK = "lifecycle_chargeback"
CATEGORY_REVERSAL = "lifecycle_reversal"
CATEGORY_PARTIAL_SETTLEMENT = "lifecycle_partial_settlement"
CATEGORY_CONSOLIDATED = "structural_consolidated"
CATEGORY_DUPLICATE = "structural_duplicate"
CATEGORY_MISSING_ROW = "structural_missing_row"
CATEGORY_HONEST_ABSTENTION = "honest_abstention"

REQUIRED_CATEGORIES: tuple[str, ...] = (
    CATEGORY_CLEAN,
    CATEGORY_DIRTY_REFERENCE,
    CATEGORY_TIMING_VARIANCE,
    CATEGORY_NET_SETTLEMENT,
    CATEGORY_REFUND,
    CATEGORY_CHARGEBACK,
    CATEGORY_REVERSAL,
    CATEGORY_PARTIAL_SETTLEMENT,
    CATEGORY_CONSOLIDATED,
    CATEGORY_DUPLICATE,
    CATEGORY_MISSING_ROW,
    CATEGORY_HONEST_ABSTENTION,
)

# ---------------------------------------------------------------------------
# Tunable generation targets. N_LOGICAL_EVENTS must stay at 100 and the
# resulting physical-row count must land in [200, 300] per §11 — both are
# checked by tests/test_generator.py. Every *_idx modifier count is a fixed
# integer count (not a probability) so both datasets carry the same amount
# of each anomaly category regardless of seed; only *which* events draw
# each modifier, and the concrete dirty/timing/amount values, vary by seed.
# ---------------------------------------------------------------------------

N_LOGICAL_EVENTS = 100

N_REVERSAL = 4
N_PARTIAL_SETTLEMENT = 4
N_MISSING_ROW = 4
N_REFUND = 4
N_CHARGEBACK = 3
N_STANDARD = N_LOGICAL_EVENTS - (
    N_REVERSAL + N_PARTIAL_SETTLEMENT + N_MISSING_ROW + N_REFUND + N_CHARGEBACK
)

N_DIRTY_REFERENCE = 16
N_TIMING_VARIANCE = 14
N_NET_SETTLEMENT_BREAKOUT = 12
N_NET_SETTLEMENT_ADJUSTMENT = 4  # subset of N_NET_SETTLEMENT_BREAKOUT
N_DUPLICATE = 4

N_HONEST_ABSTENTION = 6  # additional standalone rows, outside the 100 events

BATCH_SIZE_MIN = 5
BATCH_SIZE_MAX = 9

GROSS_AMOUNT_MIN_PAISE = 50_000  # ₹500
GROSS_AMOUNT_MAX_PAISE = 5_000_000  # ₹50,000
MDR_BPS_CHOICES = (150, 175, 200, 225, 250)
GST_RATE_PCT = 18

WINDOW_START = date(2026, 7, 1)
WINDOW_DAYS = 80
MONTH_END_DATES = (date(2026, 7, 31), date(2026, 8, 31))

REFERENCE_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"

# ---------------------------------------------------------------------------
# Counterparty name generation (NormalizedRecord.counterparty, corrected §7).
# One counterparty "identity" per logical event, expressed with realistic,
# consistent-but-not-identical labels across sources — e.g. "ACME Retail
# Pvt Ltd" on the ledger vs "ACME RETAIL PVT LTD" or "ACME Retail" on the
# gateway settlement. Deliberately layered onto the existing anomaly
# categories rather than a new one: dirty_reference events always get a
# strong (guaranteed non-identical) name variant, and clean events get a
# realistic mix of exact/light variance, same as real merchant-name data
# looks in practice.
# ---------------------------------------------------------------------------

COUNTERPARTY_ROOTS = (
    "Acme", "Bluepeak", "Silverline", "Crimson", "Northgate", "Meridian",
    "Vertex", "Solstice", "Everline", "Ganges", "Sundar", "Kavery",
    "Orion", "Falcon", "Ember", "Harbor", "Lotus", "Zenith", "Cobalt",
    "Amber", "Nimbus", "Anand", "Vishal", "Prakash", "Deccan", "Konkan",
    "Malabar", "Saffron", "Marigold", "Copper", "Ivory", "Granite",
)
COUNTERPARTY_TYPES = (
    "Retail", "Traders", "Textiles", "Foods", "Logistics", "Enterprises",
    "Solutions", "Apparel", "Electronics", "Hardware", "Agro", "Exports",
    "Fashions", "Distributors", "Industries", "Ventures", "Foods & Beverages",
    "Home Essentials", "Motors", "Pharma",
)
COUNTERPARTY_SUFFIXES = ("Pvt Ltd", "Private Limited", "LLP")

# The consolidated bank-payout leg is credited by the payment aggregator
# itself, not by any single underlying customer — one company's own
# reconciliation (§4 scope note), so this is a fixed constant rather than
# drawn per event.
AGGREGATOR_COUNTERPARTY_NAME = "Razorpay Settlements Pvt Ltd"

_TYPE_ABBREVIATIONS = {
    "Enterprises": "Ent",
    "Distributors": "Distrib",
    "Industries": "Inds",
    "Electronics": "Elec",
    "Solutions": "Soln",
    "Logistics": "Logi",
}


def _counterparty_base(rng: random.Random) -> str:
    root = rng.choice(COUNTERPARTY_ROOTS)
    biz_type = rng.choice(COUNTERPARTY_TYPES)
    suffix = rng.choice(COUNTERPARTY_SUFFIXES)
    return f"{root} {biz_type} {suffix}"


def _counterparty_variant(name: str, rng: random.Random, strong: bool) -> str:
    """Produce a consistent-but-not-identical label for the same
    counterparty on a different source's record.

    ``strong=True`` (dirty_reference events) guarantees a non-identical
    variant. ``strong=False`` (clean and other events) still frequently
    varies the label — real merchant names rarely appear byte-identical
    across a ledger and a payment gateway — but sometimes leaves it exact.
    """
    kinds = ["upper", "drop_suffix", "abbreviate_type", "no_suffix_period", "casing_swap"]
    if not strong and rng.random() < 0.3:
        return name  # left byte-identical

    kind = rng.choice(kinds)
    parts = name.split(" ")
    # Last one or two tokens are the legal suffix ("Pvt Ltd" / "Private
    # Limited" / "LLP"); everything before that is root + business type.
    if parts[-1] == "LLP":
        core, suffix_tokens = parts[:-1], [parts[-1]]
    else:
        core, suffix_tokens = parts[:-2], parts[-2:]

    if kind == "upper":
        return name.upper()
    if kind == "drop_suffix":
        return " ".join(core)
    if kind == "abbreviate_type":
        if len(core) >= 2 and core[-1] in _TYPE_ABBREVIATIONS:
            core = core[:-1] + [_TYPE_ABBREVIATIONS[core[-1]]]
        return " ".join(core + suffix_tokens)
    if kind == "no_suffix_period":
        dotted = ".".join(list(suffix_tokens[0])) if len(suffix_tokens[0]) <= 4 else suffix_tokens[0]
        return " ".join(core + [dotted] + suffix_tokens[1:]) if len(suffix_tokens) > 1 else " ".join(core + [dotted])
    # casing_swap: mixed random casing, character by character (excludes
    # spaces so words stay recognisable, unlike the reference dirtying).
    return " ".join(
        "".join(c.upper() if rng.random() < 0.5 else c.lower() for c in word)
        for word in parts
    )

_ENTITY_TO_ROLE = {
    EntityType.PAYMENT: "GROSS",
    EntityType.GATEWAY_SETTLEMENT: "GROSS",
    EntityType.BANK_CREDIT: "CREDIT",
    EntityType.FEE: "FEE",
    EntityType.TAX: "TAX",
    EntityType.REFUND: "REFUND",
    EntityType.CHARGEBACK: "CHARGEBACK",
    EntityType.REVERSAL: "REVERSAL",
    EntityType.ADJUSTMENT: "ADJUSTMENT",
}


# ---------------------------------------------------------------------------
# Small deterministic helpers
# ---------------------------------------------------------------------------


class _IdCounter:
    """Sequential, deterministic id generator — never uuid4/wall-clock."""

    def __init__(self, prefix: str) -> None:
        self._prefix = prefix
        self._record_n = 0
        self._group_n = 0

    def next_record_id(self) -> str:
        self._record_n += 1
        return f"{self._prefix}_rec_{self._record_n:05d}"

    def next_group_id(self) -> str:
        self._group_n += 1
        return f"{self._prefix}_grp_{self._group_n:05d}"


def _raw_hash(*parts: Any) -> str:
    """Deterministic stand-in for a raw-source hash (never real time/uuid)."""
    joined = "|".join(str(p) for p in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def _canonical_reference(rng: random.Random) -> str:
    suffix = "".join(rng.choice(REFERENCE_ALPHABET) for _ in range(14))
    return f"pay_{suffix}"


def _dirty_reference(ref: str, rng: random.Random) -> str:
    """Apply exactly one of the §11 'dirty reference' techniques."""
    kind = rng.choice(["whitespace", "casing", "separator", "truncation", "typo"])
    if kind == "whitespace":
        if rng.random() < 0.5:
            return f"  {ref}  "
        mid = len(ref) // 2
        return f"{ref[:mid]} {ref[mid:]}"
    if kind == "casing":
        return "".join(
            c.upper() if rng.random() < 0.5 else c.lower() for c in ref
        )
    if kind == "separator":
        return ref[:4] + "-" + ref[4:9] + "-" + ref[9:]
    if kind == "truncation":
        cut = max(6, len(ref) - rng.randint(2, 5))
        return ref[:cut]
    # typo: single-character substitution, never touching the "pay_" prefix
    pos = rng.randint(4, len(ref) - 1)
    other = rng.choice([c for c in REFERENCE_ALPHABET if c != ref[pos]])
    return ref[:pos] + other + ref[pos + 1 :]


def _random_date(rng: random.Random) -> date:
    return WINDOW_START + timedelta(days=rng.randint(0, WINDOW_DAYS))


def _role_for(entity_type: EntityType) -> str:
    return _ENTITY_TO_ROLE[entity_type]


# ---------------------------------------------------------------------------
# Generation result container
# ---------------------------------------------------------------------------


@dataclass
class GeneratedDataset:
    name: str
    seed: int
    n_logical_events: int = N_LOGICAL_EVENTS
    records: list[NormalizedRecord] = field(default_factory=list)
    match_groups: list[dict[str, Any]] = field(default_factory=list)
    duplicates: list[dict[str, Any]] = field(default_factory=list)
    unresolved: list[dict[str, Any]] = field(default_factory=list)
    honest_abstention: list[dict[str, Any]] = field(default_factory=list)

    def category_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {c: 0 for c in REQUIRED_CATEGORIES}
        for group in self.match_groups:
            for cat in group["categories"]:
                counts[cat] = counts.get(cat, 0) + 1
        for _dup in self.duplicates:
            counts[CATEGORY_DUPLICATE] = counts.get(CATEGORY_DUPLICATE, 0) + 1
        for _un in self.unresolved:
            counts[CATEGORY_MISSING_ROW] = counts.get(CATEGORY_MISSING_ROW, 0) + 1
        for _ab in self.honest_abstention:
            counts[CATEGORY_HONEST_ABSTENTION] = counts.get(CATEGORY_HONEST_ABSTENTION, 0) + 1
        return counts

    def ground_truth_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.name,
            "seed": self.seed,
            "num_logical_events": self.n_logical_events,
            "num_physical_records": len(self.records),
            "match_groups": self.match_groups,
            "duplicates": self.duplicates,
            "unresolved": self.unresolved,
            "honest_abstention": self.honest_abstention,
            "category_counts": self.category_counts(),
        }


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------


def _resolve_generation_counts(n_logical_events: int) -> dict[str, int]:
    """Resolve the per-category event/modifier counts for a given event
    total.

    At the fixed default (``N_LOGICAL_EVENTS`` = 100) this returns the
    exact original integer constants, untouched by any arithmetic — this
    is what ``calibration``/``evaluation`` always use, so their content
    can never drift from a rounding difference. For any other
    ``n_logical_events`` (arbitrary stress-test seeds, §11 note on
    on-demand multi-seed validation), every count is scaled
    proportionally to the same ratios and clamped so every §11 anomaly
    category stays representable at any reasonable size.
    """
    if n_logical_events == N_LOGICAL_EVENTS:
        return {
            "reversal": N_REVERSAL,
            "partial_settlement": N_PARTIAL_SETTLEMENT,
            "missing_row": N_MISSING_ROW,
            "refund": N_REFUND,
            "chargeback": N_CHARGEBACK,
            "standard": N_STANDARD,
            "dirty_reference": N_DIRTY_REFERENCE,
            "timing_variance": N_TIMING_VARIANCE,
            "net_settlement_breakout": N_NET_SETTLEMENT_BREAKOUT,
            "net_settlement_adjustment": N_NET_SETTLEMENT_ADJUSTMENT,
            "duplicate": N_DUPLICATE,
            "honest_abstention": N_HONEST_ABSTENTION,
        }

    if n_logical_events < 20:
        raise ValueError(
            f"n_logical_events={n_logical_events} is too small to carry "
            "the full §11 anomaly mix (every required category needs "
            "room for at least one event) — use >=20."
        )

    scale = n_logical_events / N_LOGICAL_EVENTS

    def scaled(base: int, minimum: int = 1) -> int:
        return max(minimum, round(base * scale))

    reversal = scaled(N_REVERSAL)
    partial_settlement = scaled(N_PARTIAL_SETTLEMENT)
    missing_row = scaled(N_MISSING_ROW)
    refund = scaled(N_REFUND)
    chargeback = scaled(N_CHARGEBACK)
    structural_fixed = reversal + partial_settlement + missing_row + refund + chargeback
    standard = n_logical_events - structural_fixed
    if standard < 1:
        raise ValueError(
            f"n_logical_events={n_logical_events} leaves no room for "
            "'standard' events once every structural category's minimum "
            "is reserved — use a larger --logical-events value."
        )

    batch_pool_size = standard + refund + chargeback
    dirty_reference = min(n_logical_events, scaled(N_DIRTY_REFERENCE))
    timing_variance = min(n_logical_events, scaled(N_TIMING_VARIANCE))
    net_settlement_breakout = min(batch_pool_size, scaled(N_NET_SETTLEMENT_BREAKOUT))
    net_settlement_adjustment = min(
        net_settlement_breakout, scaled(N_NET_SETTLEMENT_ADJUSTMENT, minimum=0)
    )
    duplicate = min(batch_pool_size, scaled(N_DUPLICATE))
    honest_abstention = max(5, scaled(N_HONEST_ABSTENTION))

    return {
        "reversal": reversal,
        "partial_settlement": partial_settlement,
        "missing_row": missing_row,
        "refund": refund,
        "chargeback": chargeback,
        "standard": standard,
        "dirty_reference": dirty_reference,
        "timing_variance": timing_variance,
        "net_settlement_breakout": net_settlement_breakout,
        "net_settlement_adjustment": net_settlement_adjustment,
        "duplicate": duplicate,
        "honest_abstention": honest_abstention,
    }


def generate_dataset(
    name: str, seed: int, n_logical_events: Optional[int] = None
) -> GeneratedDataset:
    """Generate one fully self-contained dataset for the given seed.

    Deterministic: the same (name, seed, n_logical_events) always
    produces byte-identical output. Uses only a single seeded
    ``random.Random`` instance — never the global ``random`` module,
    wall-clock time, or ``uuid4`` — so two calls, or two datasets, never
    leak state into each other.

    ``n_logical_events`` defaults to ``None``, which resolves to the
    fixed ``N_LOGICAL_EVENTS`` (100) and reuses the exact original §11
    category counts — this is the path ``calibration``/``evaluation``
    always take, so their content is completely unaffected by this
    parameter's existence. Passing an explicit value (e.g. for an
    on-demand stress-test seed) reuses the identical anomaly-mix
    generation logic below with proportionally scaled counts — see
    ``_resolve_generation_counts``.
    """
    n_events = n_logical_events if n_logical_events is not None else N_LOGICAL_EVENTS
    counts = _resolve_generation_counts(n_events)

    rng = random.Random(seed)
    prefix = DATASET_PREFIXES.get(name, name[:3])
    ids = _IdCounter(prefix)

    dataset = GeneratedDataset(name=name, seed=seed, n_logical_events=n_events)

    # 1. Assign a structural type to each logical event, in a shuffled
    #    but deterministic order so types aren't clustered by index.
    structural_types = (
        ["reversal"] * counts["reversal"]
        + ["partial_settlement"] * counts["partial_settlement"]
        + ["missing_row"] * counts["missing_row"]
        + ["refund"] * counts["refund"]
        + ["chargeback"] * counts["chargeback"]
        + ["standard"] * counts["standard"]
    )
    assert len(structural_types) == n_events
    event_order = list(range(n_events))
    rng.shuffle(event_order)
    event_type_by_index: dict[int, str] = {}
    for idx, etype in zip(event_order, structural_types):
        event_type_by_index[idx] = etype

    batch_pool_indices = [
        i
        for i in range(n_events)
        if event_type_by_index[i] in ("standard", "refund", "chargeback")
    ]

    # 2. Independently sample which events carry which modifiers. Fixed
    #    counts (not probabilities) guarantee both datasets carry the same
    #    amount of each category; sampling from a list (never a set) keeps
    #    the draw deterministic for a given seed.
    dirty_idx = set(rng.sample(range(n_events), counts["dirty_reference"]))
    timing_idx = set(rng.sample(range(n_events), counts["timing_variance"]))
    net_settlement_idx = set(rng.sample(batch_pool_indices, counts["net_settlement_breakout"]))
    adjustment_idx = set(
        rng.sample(sorted(net_settlement_idx), counts["net_settlement_adjustment"])
    )
    duplicate_idx = set(rng.sample(batch_pool_indices, counts["duplicate"]))

    pending_contributions: list[dict[str, Any]] = []

    for i in range(n_events):
        etype = event_type_by_index[i]

        gross = rng.randint(GROSS_AMOUNT_MIN_PAISE, GROSS_AMOUNT_MAX_PAISE)
        mdr_bps = rng.choice(MDR_BPS_CHOICES)
        fee = (gross * mdr_bps) // 10_000
        gst = (fee * GST_RATE_PCT) // 100

        if i in timing_idx and rng.random() < 0.5:
            ledger_date = rng.choice(MONTH_END_DATES)
            gw_offset = rng.choice([1, 2])
        elif i in timing_idx:
            ledger_date = _random_date(rng)
            gw_offset = rng.choice([2, 3])
        else:
            ledger_date = _random_date(rng)
            gw_offset = rng.choice([0, 1])
        gateway_date = ledger_date + timedelta(days=gw_offset)

        canonical_ref = _canonical_reference(rng)
        gateway_ref = _dirty_reference(canonical_ref, rng) if i in dirty_idx else canonical_ref

        # Counterparty identity for this event: the ledger keeps the
        # canonical form; the gateway-side rows for this event share a
        # consistent-but-not-identical variant (§7's corrected schema).
        # dirty_reference events guarantee a real variant; every other
        # event (including clean) still frequently varies the label.
        counterparty_base = _counterparty_base(rng)
        gateway_counterparty = _counterparty_variant(
            counterparty_base, rng, strong=(i in dirty_idx)
        )

        categories: list[str] = []
        if i in dirty_idx:
            categories.append(CATEGORY_DIRTY_REFERENCE)
        if i in timing_idx:
            categories.append(CATEGORY_TIMING_VARIANCE)
        if i in net_settlement_idx:
            categories.append(CATEGORY_NET_SETTLEMENT)
        if not categories:
            categories.append(CATEGORY_CLEAN)

        # --- ledger PAYMENT row (always present) ---
        ledger_id = ids.next_record_id()
        dataset.records.append(
            NormalizedRecord(
                record_id=ledger_id,
                source=Source.LEDGER,
                entity_type=EntityType.PAYMENT,
                amount_paise=gross,
                currency="INR",
                reference=canonical_ref,
                counterparty=counterparty_base,
                occurred_at=ledger_date,
                raw_hash=_raw_hash("LEDGER", "PAYMENT", ledger_id, gross, canonical_ref),
            )
        )

        # --- gateway settlement leg (+ optional FEE/TAX/ADJUSTMENT breakout) ---
        adjustment_value = 0
        if i in net_settlement_idx and i in adjustment_idx:
            adjustment_value = rng.choice([-500, -250, 250, 500, 750])
        net = gross - fee - gst + adjustment_value

        settlement_id = ids.next_record_id()
        dataset.records.append(
            NormalizedRecord(
                record_id=settlement_id,
                source=Source.GATEWAY,
                entity_type=EntityType.GATEWAY_SETTLEMENT,
                amount_paise=net,
                currency="INR",
                reference=gateway_ref,
                counterparty=gateway_counterparty,
                occurred_at=gateway_date,
                raw_hash=_raw_hash("GATEWAY", "SETTLEMENT", settlement_id, net, gateway_ref),
            )
        )
        settlement_member_ids = [ledger_id, settlement_id]

        if i in net_settlement_idx:
            fee_id = ids.next_record_id()
            dataset.records.append(
                NormalizedRecord(
                    record_id=fee_id,
                    source=Source.GATEWAY,
                    entity_type=EntityType.FEE,
                    amount_paise=-fee,
                    currency="INR",
                    reference=gateway_ref,
                    counterparty=gateway_counterparty,
                    occurred_at=gateway_date,
                    raw_hash=_raw_hash("GATEWAY", "FEE", fee_id, fee),
                )
            )
            tax_id = ids.next_record_id()
            dataset.records.append(
                NormalizedRecord(
                    record_id=tax_id,
                    source=Source.GATEWAY,
                    entity_type=EntityType.TAX,
                    amount_paise=-gst,
                    currency="INR",
                    reference=gateway_ref,
                    counterparty=gateway_counterparty,
                    occurred_at=gateway_date,
                    raw_hash=_raw_hash("GATEWAY", "TAX", tax_id, gst),
                )
            )
            settlement_member_ids += [fee_id, tax_id]
            if i in adjustment_idx:
                adj_id = ids.next_record_id()
                dataset.records.append(
                    NormalizedRecord(
                        record_id=adj_id,
                        source=Source.GATEWAY,
                        entity_type=EntityType.ADJUSTMENT,
                        amount_paise=adjustment_value,
                        currency="INR",
                        reference=gateway_ref,
                        counterparty=gateway_counterparty,
                        occurred_at=gateway_date,
                        raw_hash=_raw_hash("GATEWAY", "ADJUSTMENT", adj_id, adjustment_value),
                    )
                )
                settlement_member_ids.append(adj_id)

        # --- optional duplicate distractor (never enters a match group) ---
        if i in duplicate_idx:
            dup_id = ids.next_record_id()
            dataset.records.append(
                NormalizedRecord(
                    record_id=dup_id,
                    source=Source.GATEWAY,
                    entity_type=EntityType.GATEWAY_SETTLEMENT,
                    amount_paise=net,
                    currency="INR",
                    reference=gateway_ref,
                    counterparty=gateway_counterparty,
                    occurred_at=gateway_date,
                    raw_hash=_raw_hash("GATEWAY", "SETTLEMENT", "DUP", dup_id, net, gateway_ref),
                )
            )
            dataset.duplicates.append(
                {
                    "record_id": dup_id,
                    "duplicate_of": settlement_id,
                    "categories": [CATEGORY_DUPLICATE],
                }
            )

        # --- structural-type-specific handling ---
        if etype == "reversal":
            ledger_rev_id = ids.next_record_id()
            dataset.records.append(
                NormalizedRecord(
                    record_id=ledger_rev_id,
                    source=Source.LEDGER,
                    entity_type=EntityType.REVERSAL,
                    amount_paise=-gross,
                    currency="INR",
                    reference=canonical_ref,
                    counterparty=counterparty_base,
                    occurred_at=gateway_date,
                    raw_hash=_raw_hash("LEDGER", "REVERSAL", ledger_rev_id, gross),
                )
            )
            gateway_rev_id = ids.next_record_id()
            dataset.records.append(
                NormalizedRecord(
                    record_id=gateway_rev_id,
                    source=Source.GATEWAY,
                    entity_type=EntityType.REVERSAL,
                    amount_paise=-net,
                    currency="INR",
                    reference=gateway_ref,
                    counterparty=gateway_counterparty,
                    occurred_at=gateway_date,
                    raw_hash=_raw_hash("GATEWAY", "REVERSAL", gateway_rev_id, net),
                )
            )
            group_ids = settlement_member_ids + [ledger_rev_id, gateway_rev_id]
            roles = {rid: _role_for(EntityType.PAYMENT) for rid in [ledger_id]}
            dataset.match_groups.append(
                {
                    "group_id": ids.next_group_id(),
                    "cardinality": "ONE_TO_ONE",
                    "record_ids": group_ids,
                    "categories": sorted(set(categories + [CATEGORY_REVERSAL])),
                    "total_ledger_gross_paise": gross,
                    "total_settlement_paise": net,
                    "total_bank_paise": 0,
                    "notes": "Payment fully reversed before any bank payout.",
                }
            )
            continue  # never enters the batch pool

        if etype == "missing_row":
            dataset.unresolved.append(
                {
                    "record_ids": settlement_member_ids,
                    "reason": "MISSING_SETTLEMENT",
                    "categories": sorted(set(categories + [CATEGORY_MISSING_ROW])),
                    "notes": "Bank credit for this settlement never arrives in this batch.",
                }
            )
            continue  # net amount deliberately excluded from any payout batch

        if etype == "partial_settlement":
            split_first = int(net * rng.choice([0.5, 0.55, 0.6, 0.65]))
            split_second = net - split_first
            bank_id_1 = ids.next_record_id()
            bank_date_1 = gateway_date + timedelta(days=1)
            dataset.records.append(
                NormalizedRecord(
                    record_id=bank_id_1,
                    source=Source.BANK,
                    entity_type=EntityType.BANK_CREDIT,
                    amount_paise=split_first,
                    currency="INR",
                    reference=f"UTR{bank_date_1.strftime('%Y%m%d')}P1{i:04d}",
                    counterparty=AGGREGATOR_COUNTERPARTY_NAME,
                    occurred_at=bank_date_1,
                    raw_hash=_raw_hash("BANK", "PARTIAL1", bank_id_1, split_first),
                )
            )
            bank_id_2 = ids.next_record_id()
            bank_date_2 = gateway_date + timedelta(days=2)
            dataset.records.append(
                NormalizedRecord(
                    record_id=bank_id_2,
                    source=Source.BANK,
                    entity_type=EntityType.BANK_CREDIT,
                    amount_paise=split_second,
                    currency="INR",
                    reference=f"UTR{bank_date_2.strftime('%Y%m%d')}P2{i:04d}",
                    counterparty=AGGREGATOR_COUNTERPARTY_NAME,
                    occurred_at=bank_date_2,
                    raw_hash=_raw_hash("BANK", "PARTIAL2", bank_id_2, split_second),
                )
            )
            group_ids = settlement_member_ids + [bank_id_1, bank_id_2]
            dataset.match_groups.append(
                {
                    "group_id": ids.next_group_id(),
                    "cardinality": "ONE_TO_MANY",
                    "record_ids": group_ids,
                    "categories": sorted(set(categories + [CATEGORY_PARTIAL_SETTLEMENT])),
                    "total_ledger_gross_paise": gross,
                    "total_settlement_paise": net,
                    "total_bank_paise": split_first + split_second,
                    "notes": "Settlement payout split across two bank credits (T+1/T+2).",
                }
            )
            continue  # doesn't join the shared payout batch pool

        # standard / refund / chargeback: base payment joins the batch pool
        pending_contributions.append(
            {
                "event_index": i,
                "record_ids": settlement_member_ids,
                "amount": net,
                "gross": gross,
                "settlement_date": gateway_date,
                "categories": list(categories),
            }
        )

        if etype == "refund":
            fraction = rng.choice([1.0, 0.5, 0.3])
            refund_amount = int(gross * fraction)
            refund_date = gateway_date + timedelta(days=rng.randint(3, 20))
            r_ledger = ids.next_record_id()
            dataset.records.append(
                NormalizedRecord(
                    record_id=r_ledger,
                    source=Source.LEDGER,
                    entity_type=EntityType.REFUND,
                    amount_paise=-refund_amount,
                    currency="INR",
                    reference=canonical_ref,
                    counterparty=counterparty_base,
                    occurred_at=refund_date,
                    raw_hash=_raw_hash("LEDGER", "REFUND", r_ledger, refund_amount),
                )
            )
            r_gateway = ids.next_record_id()
            dataset.records.append(
                NormalizedRecord(
                    record_id=r_gateway,
                    source=Source.GATEWAY,
                    entity_type=EntityType.REFUND,
                    amount_paise=-refund_amount,
                    currency="INR",
                    reference=canonical_ref,
                    counterparty=gateway_counterparty,
                    occurred_at=refund_date,
                    raw_hash=_raw_hash("GATEWAY", "REFUND", r_gateway, refund_amount),
                )
            )
            r_bank = ids.next_record_id()
            r_bank_date = refund_date + timedelta(days=1)
            dataset.records.append(
                NormalizedRecord(
                    record_id=r_bank,
                    source=Source.BANK,
                    entity_type=EntityType.REFUND,
                    amount_paise=-refund_amount,
                    currency="INR",
                    reference=f"UTR{r_bank_date.strftime('%Y%m%d')}RF{i:04d}",
                    counterparty=AGGREGATOR_COUNTERPARTY_NAME,
                    occurred_at=r_bank_date,
                    raw_hash=_raw_hash("BANK", "REFUND", r_bank, refund_amount),
                )
            )
            dataset.match_groups.append(
                {
                    "group_id": ids.next_group_id(),
                    "cardinality": "ONE_TO_ONE",
                    "record_ids": [r_ledger, r_gateway, r_bank],
                    "categories": [CATEGORY_REFUND],
                    "total_ledger_gross_paise": refund_amount,
                    "total_settlement_paise": refund_amount,
                    "total_bank_paise": refund_amount,
                    "notes": "Refund reconciliation triad, independent of the original payment.",
                }
            )

        if etype == "chargeback":
            chargeback_date = gateway_date + timedelta(days=rng.randint(5, 30))
            c_ledger = ids.next_record_id()
            dataset.records.append(
                NormalizedRecord(
                    record_id=c_ledger,
                    source=Source.LEDGER,
                    entity_type=EntityType.CHARGEBACK,
                    amount_paise=-gross,
                    currency="INR",
                    reference=canonical_ref,
                    counterparty=counterparty_base,
                    occurred_at=chargeback_date,
                    raw_hash=_raw_hash("LEDGER", "CHARGEBACK", c_ledger, gross),
                )
            )
            c_gateway = ids.next_record_id()
            dataset.records.append(
                NormalizedRecord(
                    record_id=c_gateway,
                    source=Source.GATEWAY,
                    entity_type=EntityType.CHARGEBACK,
                    amount_paise=-gross,
                    currency="INR",
                    reference=canonical_ref,
                    counterparty=gateway_counterparty,
                    occurred_at=chargeback_date,
                    raw_hash=_raw_hash("GATEWAY", "CHARGEBACK", c_gateway, gross),
                )
            )
            c_bank = ids.next_record_id()
            c_bank_date = chargeback_date + timedelta(days=1)
            dataset.records.append(
                NormalizedRecord(
                    record_id=c_bank,
                    source=Source.BANK,
                    entity_type=EntityType.CHARGEBACK,
                    amount_paise=-gross,
                    currency="INR",
                    reference=f"UTR{c_bank_date.strftime('%Y%m%d')}CB{i:04d}",
                    counterparty=AGGREGATOR_COUNTERPARTY_NAME,
                    occurred_at=c_bank_date,
                    raw_hash=_raw_hash("BANK", "CHARGEBACK", c_bank, gross),
                )
            )
            dataset.match_groups.append(
                {
                    "group_id": ids.next_group_id(),
                    "cardinality": "ONE_TO_ONE",
                    "record_ids": [c_ledger, c_gateway, c_bank],
                    "categories": [CATEGORY_CHARGEBACK],
                    "total_ledger_gross_paise": gross,
                    "total_settlement_paise": gross,
                    "total_bank_paise": gross,
                    "notes": "Chargeback reconciliation triad, independent of the original payment.",
                }
            )

    # 3. Consolidate the batch pool's pending contributions into periodic
    #    bank-credit payout batches (§4's "several ledger entries -> one
    #    bank credit, or the reverse"). Sorting by (settlement_date, index)
    #    before chunking keeps this fully deterministic for a given seed.
    pending_contributions.sort(key=lambda c: (c["settlement_date"], c["event_index"]))
    batch_num = 0
    pos = 0
    while pos < len(pending_contributions):
        remaining = len(pending_contributions) - pos
        size = rng.randint(BATCH_SIZE_MIN, min(BATCH_SIZE_MAX, remaining))
        if remaining - size < BATCH_SIZE_MIN and remaining - size > 0:
            size = remaining  # avoid stranding a too-small trailing batch
        batch_members = pending_contributions[pos : pos + size]
        pos += size
        batch_num += 1

        payout_date = max(c["settlement_date"] for c in batch_members) + timedelta(
            days=rng.choice([1, 2])
        )
        total_amount = sum(c["amount"] for c in batch_members)
        bank_id = ids.next_record_id()
        bank_ref = f"UTR{payout_date.strftime('%Y%m%d')}{batch_num:04d}"
        dataset.records.append(
            NormalizedRecord(
                record_id=bank_id,
                source=Source.BANK,
                entity_type=EntityType.BANK_CREDIT,
                amount_paise=total_amount,
                currency="INR",
                reference=bank_ref,
                counterparty=AGGREGATOR_COUNTERPARTY_NAME,
                occurred_at=payout_date,
                raw_hash=_raw_hash("BANK", "CREDIT", bank_id, total_amount, bank_ref),
            )
        )

        member_record_ids: list[str] = []
        member_categories: set[str] = set()
        total_gross = 0
        total_settlement = 0
        for member in batch_members:
            member_record_ids.extend(member["record_ids"])
            member_categories.update(member["categories"])
            total_gross += member["gross"]
            total_settlement += member["amount"]
        member_record_ids.append(bank_id)

        cardinality = "ONE_TO_ONE" if len(batch_members) == 1 else "MANY_TO_ONE"
        if len(batch_members) > 1:
            member_categories.add(CATEGORY_CONSOLIDATED)

        dataset.match_groups.append(
            {
                "group_id": ids.next_group_id(),
                "cardinality": cardinality,
                "record_ids": member_record_ids,
                "categories": sorted(member_categories),
                "total_ledger_gross_paise": total_gross,
                "total_settlement_paise": total_settlement,
                "total_bank_paise": total_amount,
                "notes": (
                    f"Payout batch of {len(batch_members)} settlement(s) consolidated "
                    "into one bank credit."
                    if len(batch_members) > 1
                    else "Single settlement, single bank credit."
                ),
            }
        )

    # 4. Honest-abstention decoys: standalone records with no true
    #    counterpart anywhere in the dataset, added on top of the 100
    #    logical events. At least 5 required per §11; 6 generated here.
    decoy_profiles = [
        (Source.BANK, EntityType.ADJUSTMENT, "bank_interest_credit"),
        (Source.BANK, EntityType.ADJUSTMENT, "unrelated_inter_account_transfer"),
        (Source.GATEWAY, EntityType.ADJUSTMENT, "gateway_wallet_topup_unrelated"),
        (Source.LEDGER, EntityType.ADJUSTMENT, "manual_ledger_correction_no_counterpart"),
    ]
    for _ in range(counts["honest_abstention"]):
        source, entity_type, note = decoy_profiles[rng.randrange(len(decoy_profiles))]
        amount = rng.randint(GROSS_AMOUNT_MIN_PAISE, GROSS_AMOUNT_MAX_PAISE // 4)
        occurred = _random_date(rng)
        ref = _canonical_reference(rng)
        decoy_counterparty = _counterparty_base(rng)
        rec_id = ids.next_record_id()
        dataset.records.append(
            NormalizedRecord(
                record_id=rec_id,
                source=source,
                entity_type=entity_type,
                amount_paise=amount,
                currency="INR",
                reference=ref,
                counterparty=decoy_counterparty,
                occurred_at=occurred,
                raw_hash=_raw_hash("ABSTENTION", rec_id, note, amount),
            )
        )
        dataset.honest_abstention.append(
            {
                "record_id": rec_id,
                "reason": "NO_COUNTERPART_EXISTS",
                "categories": [CATEGORY_HONEST_ABSTENTION],
                "notes": note,
            }
        )

    return dataset


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    # src/recon_agent/testdata/generator.py -> repo root is 3 parents up.
    return Path(__file__).resolve().parents[3]


def write_dataset(
    dataset: GeneratedDataset,
    data_dir: Optional[Path] = None,
    ground_truth_dir: Optional[Path] = None,
) -> tuple[Path, Path]:
    """Write a generated dataset's records and ground truth to disk.

    Records go under ``data_dir/<name>/records.json`` (the only file the
    matching pipeline should ever read); ground truth goes under a
    completely separate ``ground_truth_dir/<name>/ground_truth.json`` that
    only the evaluation harness should read. Both default to the repo's
    ``data/`` and ``ground_truth/`` directories.
    """
    root = _repo_root()
    data_dir = data_dir if data_dir is not None else root / "data"
    ground_truth_dir = ground_truth_dir if ground_truth_dir is not None else root / "ground_truth"

    records_out_dir = data_dir / dataset.name
    gt_out_dir = ground_truth_dir / dataset.name
    records_out_dir.mkdir(parents=True, exist_ok=True)
    gt_out_dir.mkdir(parents=True, exist_ok=True)

    records_path = records_out_dir / "records.json"
    gt_path = gt_out_dir / "ground_truth.json"

    records_payload = [r.model_dump(mode="json") for r in dataset.records]
    with records_path.open("w", encoding="utf-8") as f:
        json.dump(records_payload, f, indent=2, sort_keys=False)
        f.write("\n")

    with gt_path.open("w", encoding="utf-8") as f:
        json.dump(dataset.ground_truth_dict(), f, indent=2, sort_keys=False)
        f.write("\n")

    return records_path, gt_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate the calibration/evaluation synthetic datasets, or an "
            "arbitrary on-demand stress-test dataset, reusing the same §11 "
            "anomaly-mix generation logic (ARCHITECTURE.md §11)."
        )
    )
    parser.add_argument(
        "--seed",
        choices=["calibration", "evaluation", "both", "stress"],
        default="both",
        help=(
            "Which dataset(s) to regenerate. 'calibration'/'evaluation'/'both' "
            "use this repo's fixed, checked-in seeds (unaffected by this "
            "change). 'stress' generates one arbitrary fresh-seeded dataset "
            "for on-demand multi-seed validation — pass --seed-value and "
            "optionally --logical-events with it. Default: both."
        ),
    )
    parser.add_argument(
        "--seed-value",
        type=int,
        default=None,
        help="Integer RNG seed to use with --seed stress. Required for --seed stress.",
    )
    parser.add_argument(
        "--logical-events",
        type=int,
        default=N_LOGICAL_EVENTS,
        help=(
            "Number of logical payment events for --seed stress (default: "
            f"{N_LOGICAL_EVENTS}, matching calibration/evaluation's scale so "
            "the already-calibrated thresholds stay meaningful against it). "
            "Ignored for calibration/evaluation/both."
        ),
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help=(
            "Dataset directory name for --seed stress (under data/<name>/ and "
            "ground_truth/<name>/). Default: 'stress_<seed-value>'. Ignored "
            "for calibration/evaluation/both."
        ),
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override the output data/ directory (default: repo's data/).",
    )
    parser.add_argument(
        "--ground-truth-dir",
        default=None,
        help="Override the output ground_truth/ directory (default: repo's ground_truth/).",
    )
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir) if args.data_dir else None
    ground_truth_dir = Path(args.ground_truth_dir) if args.ground_truth_dir else None

    if args.seed == "stress":
        if args.seed_value is None:
            parser.error("--seed stress requires --seed-value <int>")
        name = args.output_name or f"stress_{args.seed_value}"
        dataset = generate_dataset(name, args.seed_value, n_logical_events=args.logical_events)
        records_path, gt_path = write_dataset(dataset, data_dir, ground_truth_dir)
        print(
            f"[{name}] seed={dataset.seed} logical_events={dataset.n_logical_events} "
            f"records={len(dataset.records)} -> {records_path}"
        )
        print(f"[{name}] ground truth -> {gt_path}")
        return

    names = ["calibration", "evaluation"] if args.seed == "both" else [args.seed]
    for name in names:
        dataset = generate_dataset(name, DATASET_SEEDS[name])
        records_path, gt_path = write_dataset(dataset, data_dir, ground_truth_dir)
        print(
            f"[{name}] seed={dataset.seed} "
            f"records={len(dataset.records)} -> {records_path}"
        )
        print(f"[{name}] ground truth -> {gt_path}")


if __name__ == "__main__":
    main()
