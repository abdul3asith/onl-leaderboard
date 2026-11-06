"""Per-episode outcomes: one JSON line per episode in results/episodes/<row_id>.jsonl.

An episode is identified by its task, the LIBERO initial state it started from and its
seed. Two runs are paired on that key, which is what lets a technique be compared with
native episode by episode instead of only by success rate.

Each record also carries three fingerprints, so "same scene and seed" is checked rather
than assumed:

- `init_state_hash`: the simulator state the episode was reset to.
- `obs_hash`: the first observation the policy saw, after the settle steps and rendering.
- `noise_hash`: the first flow-matching noise the policy started from.
"""

from __future__ import annotations

import json
from pathlib import Path

from .certify import ALPHA, DELTA, certify, wilson_interval

EPISODE_FIELDS = (
    "task_id",
    "task",
    "episode_ix",
    "init_state_id",
    "seed",
    "success",
    "init_state_hash",
    "obs_hash",
    "noise_hash",
)
RUN_FIELDS = ("gpu", "batch_size", "git_sha")
FIELDS = EPISODE_FIELDS + RUN_FIELDS
# What a pair must share to count as the same scene and seed. `obs_hash` is reported but
# not required: it also covers rendering, which the policy's input need not reproduce.
_MUST_MATCH = ("init_state_hash", "noise_hash", "batch_size")
AIF_FIELDS = ("aif_count", "aif_rate", "aif_upper", "certified")


def episodes_id(model: str, checkpoint: str, technique: str, suite: str, tag: str = "") -> str:
    """File stem for one eval's episodes. `tag` names a repeat of an eval that already exists."""
    stem = f"{model}_{checkpoint}_{technique}_{suite}"
    if tag:
        stem += f"__{tag}"
    return stem.replace("/", "-").replace(" ", "_")


def key(episode: dict) -> tuple[int, int, int]:
    return episode["task_id"], episode["init_state_id"], episode["seed"]


def validate_episodes(episodes: list[dict]) -> list[dict]:
    if not episodes:
        raise ValueError("no episodes")
    for ep in episodes:
        if tuple(ep) != FIELDS:
            raise ValueError(f"episode fields {tuple(ep)} are not {FIELDS}")
        if not isinstance(ep["success"], bool):
            raise ValueError(f"success must be a bool, got {ep['success']!r}")
    if len({key(ep) for ep in episodes}) != len(episodes):
        raise ValueError("two episodes share a task, initial state and seed")
    for field in RUN_FIELDS:
        if len({ep[field] for ep in episodes}) != 1:
            raise ValueError(f"episodes of one eval disagree on {field}")
    return episodes


def read_partial(path: str | Path) -> list[dict]:
    """The records the eval wrote, before the run's own fields are added."""
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def read_episodes(path: str | Path) -> list[dict]:
    return validate_episodes(read_partial(path))


def write_episodes(episodes: list[dict], path: str | Path) -> Path:
    validate_episodes(episodes)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(ep) + "\n" for ep in episodes))
    return path


def success_fields(episodes: list[dict]) -> dict:
    """A row's success fields, computed from the episodes of its eval."""
    n = len(episodes)
    successes = sum(ep["success"] for ep in episodes)
    low, high = wilson_interval(successes, n)
    return {
        "success": 100.0 * successes / n,
        "ci_low": 100.0 * low,
        "ci_high": 100.0 * high,
        "episodes": n,
        "success_gpu": episodes[0]["gpu"],
        "success_git_sha": episodes[0]["git_sha"],
    }


def pair(a: list[dict], b: list[dict]) -> list[tuple[dict, dict]]:
    """Episodes of `a` with their partners in `b`. Both must hold exactly the same episodes."""
    by_key = {key(ep): ep for ep in b}
    if {key(ep) for ep in a} != set(by_key):
        raise ValueError("the two evals did not run the same tasks, initial states and seeds")
    return [(ep, by_key[key(ep)]) for ep in a]


def aif_fields(
    native: list[dict], technique: list[dict], alpha: float = ALPHA, delta: float = DELTA
) -> dict:
    """Certification fields for a technique against native, over paired episodes.

    Refuses pairs that did not start from the same simulator state and the same noise:
    a failure there could not be blamed on the technique.
    """
    pairs = pair(native, technique)
    for nat, tech in pairs:
        different = [f for f in _MUST_MATCH if nat[f] != tech[f]]
        if different:
            raise ValueError(f"episode {key(nat)} differs from native on {different}")
    count = sum(nat["success"] and not tech["success"] for nat, tech in pairs)
    return certify(count, len(pairs), alpha, delta)


def match_report(a: list[dict], b: list[dict]) -> dict:
    """How far two evals of the same episodes agree, on outcomes and on each fingerprint."""
    pairs = pair(a, b)
    n = len(pairs)
    flips = [x for x, y in pairs if x["success"] != y["success"]]
    report = {
        "episodes": n,
        "outcome_matches": n - len(flips),
        "outcome_match_rate": (n - len(flips)) / n,
        "a_only_successes": sum(x["success"] and not y["success"] for x, y in pairs),
        "b_only_successes": sum(y["success"] and not x["success"] for x, y in pairs),
        "a_success": 100.0 * sum(x["success"] for x, _ in pairs) / n,
        "b_success": 100.0 * sum(y["success"] for _, y in pairs) / n,
    }
    for field in ("init_state_hash", "obs_hash", "noise_hash"):
        report[f"{field}_matches"] = sum(x[field] == y[field] for x, y in pairs)
    report["flipped"] = [
        {"task_id": x["task_id"], "init_state_id": x["init_state_id"], "seed": x["seed"]}
        for x in flips
    ]
    return report
