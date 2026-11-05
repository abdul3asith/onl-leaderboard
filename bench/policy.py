"""Load a policy and build a synthetic observation for speed profiling.

Reused from the old dpc-two profiler code, generalized over the model registry.
The loader is generic: every lerobot policy follows the same
`from_pretrained` + `make_pre_post_processors` pattern, so a new model needs no
code here, only a `ModelSpec` entry.
"""

from __future__ import annotations

import gc
import importlib

import torch

from .models import ModelSpec
from .techniques import COMPILE_MODE, Technique, cast_weights, check_compile


def load_policy(spec: ModelSpec, device: str, technique: Technique) -> dict:
    """Load the policy plus its pre/post-processors, built as the technique says.

    A `native` technique leaves the weights exactly as the checkpoint ships. Load
    fresh per technique rather than re-casting one bundle, so e.g. fp16 is cast from
    the native weights, not from an already-fp32 copy.
    """
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors

    module = importlib.import_module(spec.policy_module)
    policy_cls = getattr(module, spec.policy_class)

    config = PreTrainedConfig.from_pretrained(spec.policy_path)
    config.compile_model = technique.compile_model
    config.compile_mode = COMPILE_MODE
    if spec.n_action_steps is not None:
        config.n_action_steps = spec.n_action_steps

    policy = policy_cls.from_pretrained(spec.policy_path, config=config)
    check_compile(policy, technique)
    policy.to(device)
    if technique.cast_dtype is not None:
        cast_weights(policy, technique.cast_dtype)
    policy.eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=spec.policy_path)
    return {"policy": policy, "pre": pre, "post": post, "spec": spec}


def free_gpu() -> float:
    """Release a dropped policy's GPU memory; returns the MB still allocated.

    torch.compile's global caches keep a compiled model's weights alive, so dropping
    the last reference is not enough.
    """
    torch._dynamo.reset()
    gc.collect()
    torch.cuda.empty_cache()
    return torch.cuda.memory_allocated() / 1e6


def make_observation(bundle: dict, device: str) -> dict:
    """Synthetic observation with exactly the shapes the policy expects
    (cameras + state), plus the task text. Empty cameras stay absent, as in the
    LIBERO env: the policy pads and masks them itself."""
    obs = {}
    for key, feature in bundle["policy"].config.input_features.items():
        if "empty_camera" in key:
            continue
        kind = str(feature.type)
        if "VISUAL" in kind or "STATE" in kind:
            obs[key] = torch.rand(1, *feature.shape, device=device)
    obs["task"] = [bundle["spec"].task]
    return obs


def act(bundle: dict, obs: dict):
    """One full step: preprocess -> model -> postprocess, at batch 1.

    `reset()` empties the action queue so each call is a real forward pass rather
    than a dequeue from a cached action chunk.
    """
    with torch.no_grad():
        bundle["policy"].reset()
        batch = bundle["pre"](obs)
        action = bundle["policy"].select_action(batch)
        return bundle["post"](action)
