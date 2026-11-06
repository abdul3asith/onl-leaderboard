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
    "denoise_steps",
    "n_action_steps",
    "compile_s",
    "p50_ms",
    "p99_ms",
    "jitter_ms",
    "hz",
    "peak_mb",
    "success",
    "ci_low",
    "ci_high",
    "episodes",
    "success_gpu",
    "success_git_sha",
    "aif_count",
    "aif_rate",
    "aif_upper",
    "certified",
    "date",
    "lerobot_version",
    "git_sha",
)

_SPEED_FIELDS = ("p50_ms", "p99_ms", "jitter_ms", "hz", "peak_mb")
# Certification against native over paired episodes; scripts/certify_rows.py fills them.
_AIF_FIELDS = ("aif_count", "aif_rate", "aif_upper", "certified")

# Speed is only comparable on one card. Modal's "A10" served both an A10 and an A10G,
# about 27% apart on the same pi05 work, so rows are pinned to one type and checked.
PINNED_GPU = "NVIDIA L40S"


def require_pinned_gpu(name: str) -> str:
    """Return the GPU name, or stop the run before it measures on the wrong card."""
    if name != PINNED_GPU:
        raise RuntimeError(f"landed on {name!r}, rows are pinned to {PINNED_GPU!r}")
    return name


def build_row(
    *,
    model: str,
    checkpoint: str,
    gpu: str,
    technique: str,
    denoise_steps: int,
    n_action_steps: int,
    compile_s: float | None,
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
        "denoise_steps": denoise_steps,
        "n_action_steps": n_action_steps,
        "compile_s": compile_s,
        "p50_ms": speed["p50_ms"],
        "p99_ms": speed["p99_ms"],
        "jitter_ms": speed["jitter_ms"],
        "hz": speed["hz"],
        "peak_mb": speed["peak_mb"],
        "success": success["success"] if success else None,
        "ci_low": success["ci_low"] if success else None,
        "ci_high": success["ci_high"] if success else None,
        "episodes": success["episodes"] if success else 0,
        "success_gpu": gpu if success else None,
        "success_git_sha": git_sha if success else None,
        **dict.fromkeys(_AIF_FIELDS),
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
    for field in ("denoise_steps", "n_action_steps"):
        if not isinstance(row[field], int) or row[field] < 1:
            raise ValueError(f"{field} must be a positive int, got {row[field]!r}")
    # A number on a compiled row only, null otherwise: 0.0 would read as a measurement.
    if row["technique"].startswith("compile"):
        if not isinstance(row["compile_s"], numbers.Real) or row["compile_s"] <= 0:
            raise ValueError(f"compile_s must be positive, got {row['compile_s']!r}")
    elif row["compile_s"] is not None:
        raise ValueError(f"compile_s must be null when uncompiled, got {row['compile_s']!r}")
    # gpu/git_sha describe the speed run; these say where the success numbers came from.
    evaluated = row["episodes"] > 0
    for field in ("success_gpu", "success_git_sha"):
        if evaluated and not row[field]:
            raise ValueError(f"{field} must be set on an evaluated row")
        if not evaluated and row[field] is not None:
            raise ValueError(f"{field} must be null without an eval, got {row[field]!r}")
    _validate_aif(row, evaluated)
    for field in ("model", "technique", "gpu", "date", "git_sha"):
        if not row[field]:
            raise ValueError(f"{field} must be a non-empty string")
    return row


def _validate_aif(row: dict, evaluated: bool) -> None:
    """All four null, or all four set and consistent. Native has nothing to be paired with."""
    values = [row[f] for f in _AIF_FIELDS]
    if all(v is None for v in values):
        return
    if any(v is None for v in values):
        raise ValueError(f"{_AIF_FIELDS} must be all set or all null, got {values}")
    if row["technique"] == "native" or not evaluated:
        raise ValueError("only an evaluated, non-native row can carry certification fields")
    count, rate, upper, certified = values
    if not isinstance(count, int) or isinstance(count, bool) or not 0 <= count <= row["episodes"]:
        raise ValueError(f"aif_count must be an int in [0, episodes], got {count!r}")
    if rate != count / row["episodes"]:
        raise ValueError(f"aif_rate {rate!r} is not aif_count / episodes")
    if not isinstance(upper, numbers.Real) or not rate <= upper <= 1:
        raise ValueError(f"aif_upper must be in [aif_rate, 1], got {upper!r}")
    if not isinstance(certified, bool):
        raise ValueError(f"certified must be a bool, got {certified!r}")


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
