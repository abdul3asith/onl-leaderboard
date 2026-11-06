"""Compare two evals episode by episode: do the same scene and seed give the same outcome?

Run on two native evals, this is the noise floor of paired evaluation. Any outcome that
flips between them would be counted as an acceleration-induced failure of a technique.

    python scripts/match_rate.py results/episodes/A.jsonl results/episodes/B.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.episodes import match_report, read_episodes  # noqa: E402

if __name__ == "__main__":
    a, b = (read_episodes(path) for path in sys.argv[1:3])
    print(json.dumps(match_report(a, b), indent=2))
