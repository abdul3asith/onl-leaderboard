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
    # Everything lerobot[pi] adds on top. A separate layer, so the cached lerobot layer
    # above is kept and pi05 runs the same lerobot commit as the SmolVLA rows.
    .pip_install("sentencepiece>=0.2.0,<0.3.0")
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
# pi05's PaliGemma tokenizer is a gated Hub repo.
hf_secret = modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])


@app.function(
    gpu=GPU,
    timeout=7200,  # 2h: a 100-episode eval is ~35 min; leave generous headroom
    volumes={"/root/.cache/huggingface": hf_cache, RESULTS_VOL_DIR: results_vol},
    secrets=[hf_secret],
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

    from bench.models import get_model, speed_knobs
    from bench.policy import free_gpu, load_policy
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
    knobs = speed_knobs(spec, bundle["policy"].config)

    # The eval is a subprocess with its own copy of the policy: free this one first, or
    # a large model holds the GPU twice and an OOM says nothing about the model itself.
    del bundle
    print(f"GPU memory still allocated before eval: {free_gpu():.0f} MB")

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
        **knobs,
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


@app.function(
    gpu=GPU,
    timeout=3600,
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[hf_secret],
)
def breakdown(model: str, techniques: str, iters: int, warmup: int, compile_model: str) -> str:
    """Per-stage (reset/pre/model/post) latency + device report for each technique."""
    import dataclasses
    import os
    import sys
    from importlib.metadata import distribution

    os.chdir("/root/app")
    sys.path.insert(0, "/root/app")

    import torch

    from bench.breakdown import breakdown as run_breakdown
    from bench.models import get_model
    from bench.policy import free_gpu, load_policy
    from bench.techniques import get_technique

    spec = get_model(model)
    if compile_model != "stored":
        spec = dataclasses.replace(spec, compile_model=compile_model == "on")
    device = "cuda"

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else GPU
    source = json.loads(distribution("lerobot").read_text("direct_url.json"))
    results = {"gpu": gpu_name, "lerobot_commit": source["vcs_info"]["commit_id"], "runs": []}
    for name in techniques.split(","):
        tech = get_technique(name.strip())
        bundle = None
        try:
            # Load fresh per technique so each cast starts from the native weights.
            bundle = load_policy(spec, device, tech)
            results["runs"].append(run_breakdown(bundle, tech, device, iters=iters, warmup=warmup))
        except Exception as exc:  # one bad technique shouldn't lose the others
            err = f"{type(exc).__name__}: {exc}"
            results["runs"].append({"technique": tech.name, "error": err})
        del bundle
        results["runs"][-1]["gpu_mb_after_free"] = free_gpu()
    hf_cache.commit()
    return json.dumps(results)


@app.local_entrypoint()
def breakdown_main(
    model: str = "smolvla",
    techniques: str = "native,fp32,fp16",
    iters: int = 50,
    warmup: int = 10,
    gpu: str = GPU,
    compile_model: str = "stored",  # stored | on | off
):
    fn = breakdown.with_options(gpu=gpu)
    data = json.loads(fn.remote(model, techniques, iters, warmup, compile_model))
    print(f"GPU: {data['gpu']}  |  lerobot commit: {data['lerobot_commit']}\n")
    for run in data["runs"]:
        if "error" in run:
            print(f"=== {run['technique']}  FAILED: {run['error']} ===\n")
            continue
        s = run["stages"]
        total = sum(v["p50_ms"] for v in s.values())
        print(f"=== {run['technique']}  (on {run['model_device']}) ===")
        print(f"  weights GB by dtype: {run['weights_gb_by_dtype']}")
        print(f"  built precision: {run['built_precision']}")
        print(f"  compile_model: {run['compile_model']}")
        print(f"  warmup model ms: {[round(ms) for ms in run['warmup_model_ms']]}")
        print(f"  actions finite: {run['actions_finite']}")
        print(f"  denoise steps: {run['denoise_steps']}  |  actions/chunk: {run['n_action_steps']}")
        for stage in ("reset", "pre", "model", "post"):
            ms = s[stage]["p50_ms"]
            print(f"  {stage:<6} {ms:8.2f} ms  ({100 * ms / total:4.1f}%)")
        print(f"  {'TOTAL':<6} {total:8.2f} ms")
        print(f"  amortized per control step: {run['amortized_per_action_ms']:.2f} ms")
        cpu = run["cpu_tensors_in_batch"]
        print(f"  CPU tensors fed to model: {[c['key'] for c in cpu] or 'none'}")
        print(f"  GPU MB still allocated after free: {run['gpu_mb_after_free']:.0f}")
        print()


@app.local_entrypoint()
def main(
    model: str = "smolvla",
    technique: str = "native",  # as the checkpoint ships; fp32/fp16 are extra rows
    episodes_per_task: int = 10,  # x10 LIBERO-Object tasks = 100 episodes
    seed: int = 1000,
    batch_size: int = 10,
    gpu: str = GPU,  # the row records the GPU that actually ran, read from torch
):
    from bench.row import write_row

    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=HERE,
        capture_output=True,
        text=True,
    )
    git_sha = sha.stdout.strip() or "unknown"

    fn = benchmark.with_options(gpu=gpu)
    row_json = fn.remote(model, technique, episodes_per_task, seed, batch_size, git_sha)
    row = json.loads(row_json)
    path = write_row(row, HERE / "results")
    print(json.dumps(row, indent=2))
    print(f"\nwrote {path}")
