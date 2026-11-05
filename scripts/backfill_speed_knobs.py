"""Add denoise_steps + n_action_steps to rows written before those fields existed.

Those rows ran each checkpoint's stored config, so the values are read from that
checkpoint's config.json on the Hub (applying any ModelSpec override), never typed.
Leaves measured fields and git_sha untouched; rows that already have the fields are
skipped.

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
    """The row with `knobs` added, in schema order."""
    merged = {**row, **knobs}
    return validate_row({field: merged[field] for field in FIELDS})


def main(results_dir: str) -> None:
    for path in sorted(Path(results_dir).glob("*.json")):
        row = json.loads(path.read_text())
        if "denoise_steps" in row:
            print(f"skip {path.name}")
            continue
        new = backfill(row, stored_knobs(get_model(row["model"])))
        if default_filename(new) != path.name:
            raise ValueError(f"{path.name} does not match its row: {default_filename(new)}")
        path.write_text(json.dumps(new, indent=2) + "\n")
        print(f"backfilled {path.name}: {new['denoise_steps']=}, {new['n_action_steps']=}")


if __name__ == "__main__":
    main(sys.argv[1])
