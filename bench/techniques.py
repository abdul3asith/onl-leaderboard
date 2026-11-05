"""Precision techniques.

SmolVLA mixes dtypes internally — its processor emits float32 observations and
its forward hardcodes `.to(torch.float32)` on numerically sensitive parts (flow-
matching noise/outputs). A blanket `policy.half()` therefore collides with those
float32 tensors (`mat1 and mat2 must have the same dtype`). So the three
techniques use the lever that actually works for each precision:

- ``native`` — exactly as the checkpoint ships (bf16 here): no cast, no autocast.
- ``fp32``   — upcast the weights with ``policy.to(torch.float32)``. Inputs are
  already float32, so this runs cleanly as true full precision.
- ``fp16``   — ``torch.autocast(float16)`` over the native weights: the standard
  mixed-precision inference path, since weight-casting to fp16 is unsupported.
- ``compile`` — ``native`` plus lerobot's ``compile_model`` switch, which wraps
  ``sample_actions`` in ``torch.compile`` when the policy is built.

Only ``compile`` compiles. Every other technique forces ``compile_model`` off, even
though pi05's checkpoint stores it on, so one row never mixes compiled and uncompiled
numbers: the flag comes from the technique in both the speed path and the eval, and
``check_compile`` stops a run whose policy does not match its label.

The same ``cast_dtype`` + ``autocast_dtype`` drive both the speed benchmark and
the success eval, so the two numbers in a row describe the same arithmetic.

pi05 ships mixed: bf16 language/expert, float32 vision path and norms. In eval mode
it runs its vision tower in a nested bf16 autocast, which survives the outer fp16
autocast, so pi05's ``fp16`` row is fp16 outside the vision tower and bf16 inside.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Technique:
    name: str
    cast_dtype: torch.dtype | None  # weight cast via policy.to(...); None = as shipped
    autocast_dtype: torch.dtype | None  # compute via torch.autocast(...); None = no autocast
    compile_model: bool = False  # torch.compile sample_actions at build time


COMPILE_MODE = "max-autotune"  # lerobot's default for both models

TECHNIQUES: dict[str, Technique] = {
    "native": Technique("native", None, None),
    "fp32": Technique("fp32", torch.float32, None),
    "fp16": Technique("fp16", None, torch.float16),
    "compile": Technique("compile", None, None, compile_model=True),
}


def cast_weights(policy, dtype: torch.dtype) -> None:
    """`policy.to(dtype)`, asserting every floating-point weight took the cast.

    pi05 caches its build-time precision and picks the vision tower's autocast from
    it, so `.to()` alone would leave vision running bf16 under an fp32 label.
    """
    policy.to(dtype)
    expert = getattr(getattr(policy, "model", None), "paligemma_with_expert", None)
    if expert is not None:
        expert.precision = dtype
    got = {p.dtype for p in policy.parameters() if p.is_floating_point()}
    if got != {dtype}:
        raise RuntimeError(f"weight cast failed: wanted {dtype}, got {got}")


def check_compile(policy, technique: Technique) -> None:
    """Raise unless the built policy is compiled exactly when its technique says so."""
    compiled = hasattr(policy.model.sample_actions, "_torchdynamo_orig_callable")
    if compiled != technique.compile_model:
        raise RuntimeError(
            f"{technique.name}: wanted compile={technique.compile_model}, "
            f"policy has compile={compiled}"
        )


def get_technique(name: str) -> Technique:
    try:
        return TECHNIQUES[name]
    except KeyError:
        raise KeyError(f"unknown technique {name!r}; registered: {sorted(TECHNIQUES)}") from None
