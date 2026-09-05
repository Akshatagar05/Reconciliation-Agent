"""NormalizedRecord — ARCHITECTURE.md §7.

A single physical row from one of the three sources (bank, gateway,
ledger) after normalization. amount_paise is an int per §10 ("Amounts:
integer paise, Decimal at parse boundary only") — Decimal is used only
transiently while parsing raw source values, never as the stored type.

``counterparty`` is a corrected v4.1 addition (§7): the name/account
label as it appears on that source's own record, required by Stage 2's
and Stage 5's composite scoring (§6), which named "counterparty
evidence" as a signal without this field ever existing. Like
``reference``, it is deliberately NOT normalized at rest — matching
stages normalize/fuzzy-compare it, never overwrite the stored value.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict

from recon_agent.models.enums import EntityType, Source


class NormalizedRecord(BaseModel):
    model_config = ConfigDict(use_enum_values=False)

    record_id: str
    source: Source
    entity_type: EntityType
    amount_paise: int
    currency: str
    reference: str
    counterparty: str
    occurred_at: date
    raw_hash: str
