"""The result row: build it from measured pieces, validate it, write one JSON file.

Per CLAUDE.md, every number comes from a script run, and one result row is one
JSON file with exactly these fields.
"""

from __future__ import annotations

import json
import numbers
from pathlib import Path

FIELDS = (
    "model",
    "checkpoint",
    "gpu",
    "technique",
    "p50_ms",
    "p99_ms",
    "jitter_ms",
    "hz",
    "peak_mb",
    "success",
    "ci_low",
    "ci_high",
    "episodes",
    "date",
    "lerobot_version",
    "git_sha",
)

_SPEED_FIELDS = ("p50_ms", "p99_ms", "jitter_ms", "hz", "peak_mb")


def build_row(
    *,
    model: str,
    checkpoint: str,
    gpu: str,
    technique: str,
    speed: dict,
    success: dict | None,
    date: str,
    lerobot_version: str,
    git_sha: str,
) -> dict:
    """Assemble a row. `success` is None when the eval was skipped (episodes=0)."""
    row = {
        "model": model,
        "checkpoint": checkpoint,
        "gpu": gpu,
        "technique": technique,
        "p50_ms": speed["p50_ms"],
        "p99_ms": speed["p99_ms"],
        "jitter_ms": speed["jitter_ms"],
        "hz": speed["hz"],
        "peak_mb": speed["peak_mb"],
        "success": success["success"] if success else None,
        "ci_low": success["ci_low"] if success else None,
        "ci_high": success["ci_high"] if success else None,
        "episodes": success["episodes"] if success else 0,
        "date": date,
        "lerobot_version": lerobot_version,
        "git_sha": git_sha,
    }
    validate_row(row)
    return row


def validate_row(row: dict) -> dict:
    missing = [f for f in FIELDS if f not in row]
    if missing:
        raise ValueError(f"row missing fields: {missing}")
    extra = [k for k in row if k not in FIELDS]
    if extra:
        raise ValueError(f"row has unexpected fields: {extra}")

    for field in _SPEED_FIELDS:
        if not isinstance(row[field], numbers.Real):
            raise ValueError(f"{field} must be numeric, got {row[field]!r}")
    for field in ("model", "technique", "gpu", "date", "git_sha"):
        if not row[field]:
            raise ValueError(f"{field} must be a non-empty string")
    return row


def default_filename(row: dict) -> str:
    stem = f"{row['model']}_{row['checkpoint']}_{row['technique']}_{row['gpu']}_{row['date']}"
    return stem.replace("/", "-").replace(" ", "_") + ".json"


def write_row(row: dict, results_dir: str | Path) -> Path:
    validate_row(row)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / default_filename(row)
    path.write_text(json.dumps(row, indent=2) + "\n")
    return path
