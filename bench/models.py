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
    denoise_attr: str = "num_steps"  # policy-config attribute holding the denoise step count
    n_action_steps: int | None = None  # overrides the checkpoint's value; None = as stored


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
    "pi05": ModelSpec(
        name="pi05",
        policy_path="lerobot/pi05_libero_finetuned_v044",
        policy_module="lerobot.policies.pi05.modeling_pi05",
        policy_class="PI05Policy",
        task="pick up the ketchup and place it in the basket",
        suite="libero_object",
        checkpoint="pi05_libero_finetuned_v044",
        denoise_attr="num_inference_steps",
        # The checkpoint stores 50; LeRobot's published LIBERO scores pass 10 on the CLI.
        n_action_steps=10,
    ),
}


def speed_knobs(spec: ModelSpec, config) -> dict:
    """The two row fields that drive speed, read from the policy config that runs."""
    return {
        "denoise_steps": int(getattr(config, spec.denoise_attr)),
        "n_action_steps": int(config.n_action_steps),
    }


def get_model(name: str) -> ModelSpec:
    try:
        return MODELS[name]
    except KeyError:
        raise KeyError(f"unknown model {name!r}; registered: {sorted(MODELS)}") from None
