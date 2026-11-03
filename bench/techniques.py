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

The same ``cast_dtype`` + ``autocast_dtype`` drive both the speed benchmark and
the success eval, so the two numbers in a row describe the same arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Technique:
    name: str
    cast_dtype: torch.dtype | None  # weight cast via policy.to(...); None = as shipped
    autocast_dtype: torch.dtype | None  # compute via torch.autocast(...); None = no autocast


TECHNIQUES: dict[str, Technique] = {
    "native": Technique("native", None, None),
    "fp32": Technique("fp32", torch.float32, None),
    "fp16": Technique("fp16", None, torch.float16),
}


def get_technique(name: str) -> Technique:
    try:
        return TECHNIQUES[name]
    except KeyError:
        raise KeyError(
            f"unknown technique {name!r}; registered: {sorted(TECHNIQUES)}"
        ) from None
