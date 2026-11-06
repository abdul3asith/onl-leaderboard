"""Fill each row's success and certification fields from results/episodes/.

A row with an episodes file takes its success fields from it. A row that is not native
also takes its certification fields, from its episodes paired with native's. Every other
row keeps its success and gets null certification fields. Nothing is written unless
every row is valid.

    python scripts/certify_rows.py results/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.episodes import (  # noqa: E402
    AIF_FIELDS,
    aif_fields,
    episodes_id,
    read_episodes,
    success_fields,
)
from bench.models import get_model  # noqa: E402
from bench.row import FIELDS, validate_row  # noqa: E402


def _episodes(results_dir: Path, row: dict, technique: str) -> list[dict] | None:
    suite = row.get("suite") or get_model(row["model"]).suite
    name = episodes_id(row["model"], row["checkpoint"], technique, suite)
    path = results_dir / "episodes" / f"{name}.jsonl"
    return read_episodes(path) if path.exists() else None


def certify_all(results_dir: str | Path) -> None:
    results_dir = Path(results_dir)
    plan = []
    for path in sorted(results_dir.glob("*.json")):
        row = json.loads(path.read_text())
        row.update(dict.fromkeys(AIF_FIELDS))
        own = _episodes(results_dir, row, row["technique"])
        native = _episodes(results_dir, row, "native")
        if own is not None:
            row.update(success_fields(own))
            if row["technique"] != "native" and native is not None:
                row.update(aif_fields(native, own))
        plan.append((path, validate_row({field: row[field] for field in FIELDS})))

    for path, row in plan:
        path.write_text(json.dumps(row, indent=2) + "\n")
        certified = "no paired episodes" if row["certified"] is None else row["certified"]
        print(f"{path.name}: certified = {certified}")


if __name__ == "__main__":
    certify_all(sys.argv[1])
