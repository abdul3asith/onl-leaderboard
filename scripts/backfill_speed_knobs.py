"""Add denoise_steps, n_action_steps and compile_s to rows written before they existed.

Those rows ran each checkpoint's stored config, so the step counts are read from that
checkpoint's config.json on the Hub (applying any ModelSpec override), never typed.
compile_s is null: they ran uncompiled, and validate_row rejects a compiled row without
a measured value. Leaves everything already in a row untouched; complete rows are skipped.

    python scripts/backfill_speed_knobs.py results/
"""

from __future__ import annotations

import json
import sys
import types
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.models import ModelSpec, get_model, speed_knobs  # noqa: E402
from bench.row import FIELDS, default_filename, validate_row  # noqa: E402


def stored_knobs(spec: ModelSpec) -> dict:
    url = f"https://huggingface.co/{spec.policy_path}/resolve/main/config.json"
    with urllib.request.urlopen(url) as resp:
        config = types.SimpleNamespace(**json.load(resp))
    if spec.n_action_steps is not None:
        config.n_action_steps = spec.n_action_steps
    return speed_knobs(spec, config)


def backfill(row: dict, knobs: dict) -> dict:
    """The row with any missing `knobs` added, in schema order."""
    merged = {"compile_s": None, **knobs, **row}
    return validate_row({field: merged[field] for field in FIELDS})


def main(results_dir: str) -> None:
    for path in sorted(Path(results_dir).glob("*.json")):
        row = json.loads(path.read_text())
        if set(FIELDS) <= set(row):
            print(f"skip {path.name}")
            continue
        new = backfill(row, stored_knobs(get_model(row["model"])))
        if default_filename(new) != path.name:
            raise ValueError(f"{path.name} does not match its row: {default_filename(new)}")
        path.write_text(json.dumps(new, indent=2) + "\n")
        added = {k: new[k] for k in FIELDS if k not in row}
        print(f"backfilled {path.name}: {added}")


if __name__ == "__main__":
    main(sys.argv[1])
