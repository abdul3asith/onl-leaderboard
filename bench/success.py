"""Success benchmark: run a LIBERO eval via lerobot-eval and read the result.

Runs in a subprocess so the heavy LIBERO/MuJoCo environment never shares a
process with the speed-profiling policy, and so the eval's precision handling
(see `_eval_entry`) is applied cleanly. LIBERO's first-run prompt is answered
"N" via stdin, matching the old dpc-two eval.

`episodes_per_task` is lerobot's per-task `--eval.n_episodes`; LIBERO-Object has
10 tasks, so the default of 10 yields 100 total episodes. The row's `episodes`
field is read back from the eval's overall count (100), not assumed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

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
) -> dict:
    """Launch the eval and return {success, ci_low, ci_high, episodes}.

    `success`, `ci_low`, `ci_high` are percentages (0-100), matching lerobot's
    Wilson interval. `episodes` is the total episode count across the suite.
    """
    output_dir = Path(output_dir)
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
    ]
    if spec.n_action_steps is not None:
        cmd += ["--n-action-steps", str(spec.n_action_steps)]
    subprocess.run(cmd, check=True, input="N\n", text=True)

    info = json.loads((output_dir / "eval_info.json").read_text())
    overall = info["overall"]
    ci_low, ci_high = overall["pc_success_ci95"]
    return {
        "success": float(overall["pc_success"]),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "episodes": int(overall["n_episodes"]),
    }
