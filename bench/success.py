"""Success benchmark: run a LIBERO eval via lerobot-eval and read the result.

Runs in a subprocess so the heavy LIBERO/MuJoCo environment never shares a
process with the speed-profiling policy, and so the eval's precision handling
(see `_eval_entry`) is applied cleanly. LIBERO's first-run prompt is answered
"N" via stdin, matching the old dpc-two eval.

`episodes_per_task` is lerobot's per-task `--eval.n_episodes`; LIBERO-Object has
10 tasks, so the default of 10 yields 100 total episodes. The row's `episodes`
field is read back from the eval's overall count (100), not assumed.

The eval also writes one record per episode (see `episodes`). They are accepted only if
they add up to the totals lerobot reported, and the file appears under its final name
only then: until the eval finishes it is `<name>.partial`.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from .episodes import read_partial, write_episodes
from .models import ModelSpec
from .techniques import Technique


def run_success(
    spec: ModelSpec,
    technique: Technique,
    *,
    episodes_per_task: int,
    seed: int,
    batch_size: int,
    device: str,
    output_dir: str | Path,
    episodes_path: str | Path,
    gpu: str,
    git_sha: str,
    task_ids: str = "",
) -> dict:
    """Launch the eval, write its episodes, and return {success, ci_low, ci_high, episodes}.

    `success`, `ci_low`, `ci_high` are percentages (0-100), matching lerobot's
    Wilson interval. `episodes` is the total episode count across the suite.
    `task_ids` limits the eval to some of the suite's tasks, for a cheap smoke test.
    """
    output_dir = Path(output_dir)
    episodes_path = Path(episodes_path)
    if episodes_path.exists():
        raise FileExistsError(f"{episodes_path} exists; an eval is rerun only under a new tag")
    partial = episodes_path.with_name(episodes_path.name + ".partial")
    cmd = [
        sys.executable,
        "-m",
        "bench._eval_entry",
        "--technique",
        technique.name,
        "--policy-path",
        spec.policy_path,
        "--suite",
        spec.suite,
        "--episodes-per-task",
        str(episodes_per_task),
        "--batch-size",
        str(batch_size),
        "--seed",
        str(seed),
        "--device",
        device,
        "--output-dir",
        str(output_dir),
        "--episodes-out",
        str(partial),
    ]
    if task_ids:
        cmd += ["--task-ids", task_ids]
    if spec.n_action_steps is not None:
        cmd += ["--n-action-steps", str(spec.n_action_steps)]
    subprocess.run(cmd, check=True, input="N\n", text=True)

    info = json.loads((output_dir / "eval_info.json").read_text())
    overall = info["overall"]
    ci_low, ci_high = overall["pc_success_ci95"]

    run = {"gpu": gpu, "batch_size": batch_size, "git_sha": git_sha}
    episodes = [{**ep, **run} for ep in read_partial(partial)]
    recorded = (len(episodes), sum(ep["success"] for ep in episodes))
    reported = (int(overall["n_episodes"]), int(overall["n_success"]))
    if recorded != reported:
        raise RuntimeError(f"recorded (episodes, successes) {recorded}, lerobot says {reported}")
    write_episodes(episodes, episodes_path)
    partial.unlink()

    return {
        "success": float(overall["pc_success"]),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "episodes": int(overall["n_episodes"]),
    }
