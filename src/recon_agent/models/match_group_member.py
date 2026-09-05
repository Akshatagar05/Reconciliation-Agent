"""MatchGroupMember — ARCHITECTURE.md §3 and §7.

Replaces MatchAllocation (source_record_id / target_record_id /
allocated_amount_paise), which was pairwise and broke the moment a group
spanned bank + gateway + ledger records simultaneously, or contained
several records from one source. The Financial and Evidence Verifier
evaluates the entire group's membership set against the conservation
equation, not a chain of pairwise links (§3).

signed_amount_paise and allocated_amount_paise are int per §3 (and §10's
"integer paise, Decimal at parse boundary only").
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from recon_agent.models.enums import MemberRole, Source


class MatchGroupMember(BaseModel):
    model_config = ConfigDict(use_enum_values=False)

    group_id: str
    record_id: str
    source: Source
    role: MemberRole
    signed_amount_paise: int
    allocated_amount_paise: int
