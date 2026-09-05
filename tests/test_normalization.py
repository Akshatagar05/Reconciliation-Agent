"""Unit tests for src/recon_agent/normalization/reference.py."""

from __future__ import annotations

from recon_agent.normalization import normalize_reference


def test_identical_references_normalize_equal() -> None:
    ref = "pay_6qh09u7670lndb"
    assert normalize_reference(ref) == normalize_reference(ref)


def test_whitespace_dirtying_recovers_equality() -> None:
    clean = "pay_6qh09u7670lndb"
    dirty_leading_trailing = "  pay_6qh09u7670lndb  "
    dirty_internal = "pay_6qh0 9u7670lndb"
    assert normalize_reference(clean) == normalize_reference(dirty_leading_trailing)
    assert normalize_reference(clean) == normalize_reference(dirty_internal)


def test_casing_dirtying_recovers_equality() -> None:
    clean = "pay_6qh09u7670lndb"
    dirty = "PAY_6QH09u7670LNDB"
    assert normalize_reference(clean) == normalize_reference(dirty)


def test_separator_dirtying_recovers_equality() -> None:
    clean = "pay_6qh09u7670lndb"
    dirty = "pay_-6qh09-u7670lndb"
    assert normalize_reference(clean) == normalize_reference(dirty)


def test_combined_whitespace_casing_separator_recovers_equality() -> None:
    clean = "pay_6qh09u7670lndb"
    dirty = "  PAY-6QH_09U-7670lndb  "
    assert normalize_reference(clean) == normalize_reference(dirty)


def test_truncation_does_not_recover_equality() -> None:
    clean = "pay_6qh09u7670lndb"
    truncated = clean[:-4]
    assert normalize_reference(clean) != normalize_reference(truncated)


def test_typo_does_not_recover_equality() -> None:
    clean = "pay_6qh09u7670lndb"
    typo = "pay_6qh09u7670lnda"  # last char changed
    assert normalize_reference(clean) != normalize_reference(typo)


def test_unrelated_references_do_not_collide() -> None:
    a = "pay_6qh09u7670lndb"
    b = "pay_zzzzzzzzzzzzzz"
    assert normalize_reference(a) != normalize_reference(b)


def test_normalize_is_idempotent() -> None:
    ref = "  PAY_-6QH09-u7670LNDB  "
    once = normalize_reference(ref)
    twice = normalize_reference(once)
    assert once == twice


def test_normalize_does_not_mutate_input() -> None:
    ref = "  PAY_6QH09u7670LNDB  "
    original = str(ref)
    normalize_reference(ref)
    assert ref == original
