"""Model registry.

Adding a model is one `ModelSpec` entry: the Hub repo id (used both by the speed
loader and by `lerobot-eval --policy.path`), the policy class to import, the task
text for the synthetic speed observation, and the LIBERO suite + checkpoint label
that get recorded in the result row.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str  # registry key + row "model" field
    policy_path: str  # HF repo id — loader and lerobot-eval --policy.path
    policy_module: str  # import path of the policy class
    policy_class: str  # class exposing .from_pretrained
    task: str  # natural-language task for the synthetic speed observation
    suite: str = "libero_object"  # LIBERO suite for success eval
    checkpoint: str = "main"  # recorded in the row


MODELS: dict[str, ModelSpec] = {
    "smolvla": ModelSpec(
        name="smolvla",
        policy_path="HuggingFaceVLA/smolvla_libero",
        policy_module="lerobot.policies.smolvla.modeling_smolvla",
        policy_class="SmolVLAPolicy",
        task="pick up the ketchup and place it in the basket",
        suite="libero_object",
        checkpoint="smolvla_libero",
    ),
}


def get_model(name: str) -> ModelSpec:
    try:
        return MODELS[name]
    except KeyError:
        raise KeyError(f"unknown model {name!r}; registered: {sorted(MODELS)}") from None
