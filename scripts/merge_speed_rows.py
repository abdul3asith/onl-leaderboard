"""Combine speed-only rows from the pinned GPU with the success of earlier evals.

Each speed-only row (`episodes` 0) must have exactly one evaluated partner with the same
model, checkpoint and technique, and both must have run the same settings. The merged row
takes its success fields from the partner and everything else from the speed run, then
replaces both files. Nothing is written or removed unless every pair is valid.

    python scripts/merge_speed_rows.py results/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.row import require_pinned_gpu, validate_row, write_row  # noqa: E402

_KEY = ("model", "checkpoint", "technique")
_MUST_MATCH = ("denoise_steps", "n_action_steps", "lerobot_version")
_SUCCESS = ("success", "ci_low", "ci_high", "episodes")


def merge_all(results_dir: str | Path) -> None:
    rows = {path: json.loads(path.read_text()) for path in Path(results_dir).glob("*.json")}
    speed_rows = {p: r for p, r in rows.items() if r["episodes"] == 0}
    evaluated = {p: r for p, r in rows.items() if r["episodes"] > 0}

    plan = []
    for speed_path, speed in sorted(speed_rows.items()):
        require_pinned_gpu(speed["gpu"])
        key = tuple(speed[k] for k in _KEY)
        partners = [(p, r) for p, r in evaluated.items() if tuple(r[k] for k in _KEY) == key]
        if len(partners) != 1:
            raise ValueError(f"{speed_path.name}: {len(partners)} evaluated partners, wanted 1")
        partner_path, partner = partners[0]
        different = [k for k in _MUST_MATCH if speed[k] != partner[k]]
        if different:
            raise ValueError(f"{speed_path.name} and {partner_path.name} differ on {different}")
        merged = validate_row({**speed, **{k: partner[k] for k in _SUCCESS}})
        plan.append((speed_path, partner_path, merged))

    for speed_path, partner_path, merged in plan:
        speed_path.unlink()
        partner_path.unlink()
        path = write_row(merged, results_dir)
        print(f"{partner_path.name} + {speed_path.name} -> {path.name}")


if __name__ == "__main__":
    merge_all(sys.argv[1])
