"""Modal entrypoint for onl-bench.

    modal run modal_app.py --model smolvla --technique fp32

runs the speed benchmark and the LIBERO success eval on a GPU and writes one
JSON row to results/. Use --episodes-per-task 0 to skip the (paid) success eval
and get a speed-only row for a cheap smoke test.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import modal

GPU = "A10G"
HERE = Path(__file__).parent

# Image mirrors the old dpc-two eval image: lerobot from git with the smolvla and
# libero extras, plus the MuJoCo/EGL system libs LIBERO needs for headless render.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "git",
        "ffmpeg",
        "build-essential",
        "cmake",
        "pkg-config",
        "libegl1",
        "libegl1-mesa-dev",
        "libgl1",
        "libgl1-mesa-dev",
        "libgles2-mesa-dev",
        "libglvnd-dev",
        "libosmesa6",
        "libosmesa6-dev",
        "libglib2.0-0",
    )
    .env(
        {
            "CMAKE_POLICY_VERSION_MINIMUM": "3.5",  # let new CMake build old deps (eglprobe)
            "MUJOCO_GL": "egl",
        }
    )
    .pip_install("pip", "setuptools", "wheel")
    .pip_install("lerobot[smolvla,libero] @ git+https://github.com/huggingface/lerobot.git")
    .add_local_dir(
        HERE,
        remote_path="/root/app",
        ignore=[".venv", "__pycache__", "*.pyc", "results", "*.mp4", ".git"],
    )
)

app = modal.App("onl-bench", image=image)

# Downloaded model weights survive between runs.
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
# Result rows persist here so a detached run's output survives the local client
# disconnecting (otherwise only the local entrypoint writes the row, and it is
# lost if that process is killed mid-run).
results_vol = modal.Volume.from_name("onl-bench-results", create_if_missing=True)
RESULTS_VOL_DIR = "/results"


@app.function(
    gpu=GPU,
    timeout=7200,  # 2h: a 100-episode eval is ~35 min; leave generous headroom
    volumes={"/root/.cache/huggingface": hf_cache, RESULTS_VOL_DIR: results_vol},
)
def benchmark(
    model: str,
    technique: str,
    episodes_per_task: int,
    seed: int,
    batch_size: int,
    git_sha: str,
) -> str:
    """Profile speed, run the LIBERO success eval, return the assembled row as JSON."""
    import datetime as dt
    import os
    import sys
    from importlib.metadata import version

    os.chdir("/root/app")
    sys.path.insert(0, "/root/app")

    import torch

    from bench.models import get_model
    from bench.policy import load_policy
    from bench.row import build_row, write_row
    from bench.speed import run_speed
    from bench.success import run_success
    from bench.techniques import get_technique

    spec = get_model(model)
    tech = get_technique(technique)
    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else GPU

    bundle = load_policy(spec, device, tech)
    speed = run_speed(bundle, tech, device)

    success = None
    if episodes_per_task > 0:
        success = run_success(
            spec,
            tech,
            episodes_per_task=episodes_per_task,
            seed=seed,
            batch_size=batch_size,
            device=device,
            output_dir=f"/tmp/eval/{spec.suite}",
        )
    hf_cache.commit()

    row = build_row(
        model=spec.name,
        checkpoint=spec.checkpoint,
        gpu=gpu_name,
        technique=tech.name,
        speed=speed,
        success=success,
        date=dt.datetime.now(dt.UTC).strftime("%Y-%m-%d"),
        lerobot_version=version("lerobot"),
        git_sha=git_sha,
    )

    # Persist to the Volume so the row survives a detached run losing its client.
    write_row(row, RESULTS_VOL_DIR)
    results_vol.commit()
    return json.dumps(row)


@app.function(gpu=GPU, timeout=1800, volumes={"/root/.cache/huggingface": hf_cache})
def breakdown(model: str, techniques: str, iters: int, warmup: int) -> str:
    """Per-stage (reset/pre/model/post) latency + device report for each technique."""
    import os
    import sys

    os.chdir("/root/app")
    sys.path.insert(0, "/root/app")

    import torch

    from bench.breakdown import breakdown as run_breakdown
    from bench.models import get_model
    from bench.policy import load_policy
    from bench.techniques import get_technique

    spec = get_model(model)
    device = "cuda"

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else GPU
    results = {"gpu": gpu_name, "runs": []}
    for name in techniques.split(","):
        tech = get_technique(name.strip())
        try:
            # Load fresh per technique so each cast starts from the native weights.
            bundle = load_policy(spec, device, tech)
            results["runs"].append(run_breakdown(bundle, tech, device, iters=iters, warmup=warmup))
        except Exception as exc:  # one bad technique shouldn't lose the others
            err = f"{type(exc).__name__}: {exc}"
            results["runs"].append({"technique": tech.name, "error": err})
    hf_cache.commit()
    return json.dumps(results)


@app.local_entrypoint()
def breakdown_main(
    model: str = "smolvla",
    techniques: str = "native,fp32,fp16",
    iters: int = 50,
    warmup: int = 10,
):
    data = json.loads(breakdown.remote(model, techniques, iters, warmup))
    print(f"GPU: {data['gpu']}\n")
    for run in data["runs"]:
        if "error" in run:
            print(f"=== {run['technique']}  FAILED: {run['error']} ===\n")
            continue
        s = run["stages"]
        total = sum(v["p50_ms"] for v in s.values())
        params = f"{run['model_dtype']} on {run['model_device']}"
        print(f"=== {run['technique']}  (model params: {params}) ===")
        print(f"  flow steps: {run['num_flow_steps']}  |  actions/chunk: {run['n_action_steps']}")
        for stage in ("reset", "pre", "model", "post"):
            ms = s[stage]["p50_ms"]
            print(f"  {stage:<6} {ms:8.2f} ms  ({100 * ms / total:4.1f}%)")
        print(f"  {'TOTAL':<6} {total:8.2f} ms")
        print(f"  amortized per control step: {run['amortized_per_action_ms']:.2f} ms")
        cpu = run["cpu_tensors_in_batch"]
        print(f"  CPU tensors fed to model: {[c['key'] for c in cpu] or 'none'}")
        print()


@app.local_entrypoint()
def main(
    model: str = "smolvla",
    technique: str = "native",  # as the checkpoint ships; fp32/fp16 are extra rows
    episodes_per_task: int = 10,  # x10 LIBERO-Object tasks = 100 episodes
    seed: int = 1000,
    batch_size: int = 10,
):
    from bench.row import write_row

    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=HERE,
        capture_output=True,
        text=True,
    )
    git_sha = sha.stdout.strip() or "unknown"

    row_json = benchmark.remote(model, technique, episodes_per_task, seed, batch_size, git_sha)
    row = json.loads(row_json)
    path = write_row(row, HERE / "results")
    print(json.dumps(row, indent=2))
    print(f"\nwrote {path}")
