"""Bundled-dataset discovery for the Stage 14 dashboard.

Purely file I/O over the existing ``data/`` and ``ground_truth/``
directories that earlier stages already produce (``testdata.generator``,
per ARCHITECTURE.md §2/§11) — this module does not generate, mutate, or
validate that data in any new way. It exists only so the dashboard's
"pick one of the bundled sets" option (item 1 of this stage's brief)
doesn't require the judge to locate and paste a 200-300 row JSON file
by hand; a judge could equally well ``cat data/evaluation/records.json``
themselves and paste it into the "upload a file" path instead — this is
a convenience, not a new data path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple, Optional


class BundledDataset(NamedTuple):
    name: str
    records_path: Path
    ground_truth_path: Optional[Path]


def _repo_root() -> Path:
    """The project root, found relative to this file
    (``src/recon_agent/dashboard/datasets.py`` -> repo root is three
    parents up), rather than assumed to be the process's current
    working directory — ``streamlit run`` may be invoked from anywhere.
    """
    return Path(__file__).resolve().parents[3]


def list_bundled_datasets(root: Optional[Path] = None) -> list[BundledDataset]:
    """Every ``data/<name>/records.json`` found on disk, paired with its
    ``ground_truth/<name>/ground_truth.json`` if one exists. Returns an
    empty list (never raises) if no ``data/`` directory exists at all —
    the dashboard falls back to upload-only in that case."""
    root = root if root is not None else _repo_root()
    data_dir = root / "data"
    if not data_dir.is_dir():
        return []

    found: list[BundledDataset] = []
    for child in sorted(data_dir.iterdir()):
        records_path = child / "records.json"
        if not records_path.is_file():
            continue
        gt_path = root / "ground_truth" / child.name / "ground_truth.json"
        found.append(
            BundledDataset(
                name=child.name,
                records_path=records_path,
                ground_truth_path=gt_path if gt_path.is_file() else None,
            )
        )
    return found


def load_records(path: Path) -> list[dict]:
    """Load a records JSON file (either a bundled ``data/*/records.json``
    or a judge-uploaded file of the same shape) as plain dicts — no
    validation beyond ``json.load`` happens here; the API's own
    Pydantic models are the single source of truth for what makes a
    valid ``NormalizedRecord`` (§7), so this deliberately does not
    duplicate that validation client-side."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(
            "Expected a JSON array of records (same shape as "
            "data/*/records.json), got a top-level "
            f"{type(data).__name__}."
        )
    return data


def load_records_from_bytes(raw: bytes) -> list[dict]:
    """Same as ``load_records`` but for an in-memory upload (Streamlit's
    ``UploadedFile`` gives bytes, not a path)."""
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, list):
        raise ValueError(
            "Expected a JSON array of records (same shape as "
            "data/*/records.json), got a top-level "
            f"{type(data).__name__}."
        )
    return data


def load_ground_truth(path: Path) -> dict:
    """Load a ``ground_truth.json`` file as a plain dict, shaped exactly
    like the API's ``GroundTruthPayload`` (api/schemas.py) — passed
    through to ``POST /reconcile`` unchanged."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_ground_truth_from_bytes(raw: bytes) -> dict:
    return json.loads(raw.decode("utf-8"))
