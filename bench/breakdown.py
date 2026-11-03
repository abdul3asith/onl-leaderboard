"""Per-stage latency breakdown: reset / preprocess / model / postprocess.

Answers "where do the milliseconds go, and is anything on the CPU that
shouldn't be?" by timing each stage of `act` separately with CUDA events and
reporting the device + dtype of every tensor the preprocessor hands the model.

Note on semantics: `act` calls `policy.reset()` every step, so each timed step
runs a *full* chunk generation (VLM encode + `num_steps` flow-matching
iterations producing `n_action_steps` actions), not a single control step. The
amortized per-control-step cost is `model_ms / n_action_steps`.
"""

from __future__ import annotations

import time
from contextlib import nullcontext

import numpy as np
import torch

from .policy import make_observation
from .techniques import Technique


def _time_ms(fn, device: str):
    if device == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end), out
    t0 = time.perf_counter()
    out = fn()
    return (time.perf_counter() - t0) * 1000, out


def _describe_batch(batch: dict) -> list[dict]:
    rows = []
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            rows.append(
                {
                    "key": key,
                    "shape": tuple(value.shape),
                    "dtype": str(value.dtype),
                    "device": str(value.device),
                }
            )
    return rows


def _stats(xs: list[float]) -> dict:
    a = np.array(xs)
    return {
        "p50_ms": float(np.percentile(a, 50)),
        "p99_ms": float(np.percentile(a, 99)),
        "mean_ms": float(a.mean()),
    }


def breakdown(bundle: dict, technique: Technique, device: str, iters: int = 50, warmup: int = 10):
    """`bundle` weights are already cast by `load_policy`; autocast is applied here."""
    policy, pre, post = bundle["policy"], bundle["pre"], bundle["post"]
    obs = make_observation(bundle, device)
    ac = technique.autocast_dtype
    autocast = (lambda: torch.autocast(device_type=device, dtype=ac)) if ac else nullcontext

    stages: dict[str, list[float]] = {"reset": [], "pre": [], "model": [], "post": []}
    batch_desc: list[dict] | None = None

    for i in range(warmup + iters):
        record = i >= warmup
        with torch.no_grad():
            dt, _ = _time_ms(policy.reset, device)
            if record:
                stages["reset"].append(dt)

            dt, batch = _time_ms(lambda: pre(obs), device)
            if record:
                stages["pre"].append(dt)

            with autocast():
                dt, action = _time_ms(lambda b=batch: policy.select_action(b), device)
            if record:
                stages["model"].append(dt)

            dt, _ = _time_ms(lambda a=action: post(a), device)
            if record:
                stages["post"].append(dt)

        if batch_desc is None:
            batch_desc = _describe_batch(batch)

    param = next(policy.parameters())
    model_ms = _stats(stages["model"])["p50_ms"]
    n_action_steps = int(policy.config.n_action_steps)
    return {
        "technique": technique.name,
        "model_device": str(param.device),
        "model_dtype": str(param.dtype),
        "n_action_steps": n_action_steps,
        "num_flow_steps": int(getattr(policy.config, "num_steps", -1)),
        "batch": batch_desc,
        "cpu_tensors_in_batch": [r for r in (batch_desc or []) if r["device"] == "cpu"],
        "stages": {k: _stats(v) for k, v in stages.items()},
        "amortized_per_action_ms": model_ms / n_action_steps,
    }
