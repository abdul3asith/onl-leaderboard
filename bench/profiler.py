"""Step-time profiler. Reused from dpc-two, pinned to onl-bench's protocol:
batch 1, 10 warmup + 200 timed steps (see CLAUDE.md)."""

import time

import numpy as np
import torch

WARMUP = 10
TIMED = 200


def time_one_step(step_fn, device: str) -> float:
    """Wall time of one step in milliseconds. Uses CUDA events on GPU so the
    timing is not fooled by async kernel launches."""
    if device == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        step_fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)

    t0 = time.perf_counter()
    step_fn()
    return (time.perf_counter() - t0) * 1000


def profile(step_fn, device: str, steps: int = TIMED, warmup: int = WARMUP) -> dict:
    for _ in range(warmup):
        step_fn()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    timings = np.array([time_one_step(step_fn, device) for _ in range(steps)])
    p50 = np.percentile(timings, 50)
    p99 = np.percentile(timings, 99)
    peak_mb = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else 0.0

    return {
        "p50_ms": float(p50),  # typical step time
        "p99_ms": float(p99),  # worst-case step time (except rare outliers)
        "jitter_ms": float(timings.std()),  # spread: low = steady rhythm
        "hz": float(1000 / p99),  # steps/s the robot can count on (worst case)
        "peak_mb": float(peak_mb),
    }
