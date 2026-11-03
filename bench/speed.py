"""Speed benchmark: profile one policy under one technique.

A technique's precision comes from two levers (see techniques.py): a weight cast
applied at load time by `load_policy`, and/or an autocast context around the
step. `make_step_fn` applies the autocast half.
"""

from __future__ import annotations

import torch

from .policy import act, make_observation
from .profiler import profile
from .techniques import Technique


def make_step_fn(bundle: dict, technique: Technique, device: str):
    dtype = technique.autocast_dtype
    if dtype is None:
        return lambda obs: act(bundle, obs)

    def step(obs):
        with torch.autocast(device_type=device, dtype=dtype):
            return act(bundle, obs)

    return step


def run_speed(bundle: dict, technique: Technique, device: str) -> dict:
    """Returns {p50_ms, p99_ms, jitter_ms, hz, peak_mb}. `bundle` is already cast."""
    obs = make_observation(bundle, device)
    step = make_step_fn(bundle, technique, device)
    return profile(lambda: step(obs), device)
