"""Wrapper entrypoint around lerobot-eval that applies a technique's precision.

lerobot-eval has no CLI lever that casts SmolVLA's weights (`--policy.dtype` is a
no-op for SmolVLA), and its autocast is always the device-default dtype. So this
wrapper reproduces, in the eval, the same two levers the speed path uses:

- weight cast (fp32): patch the eval module's `make_policy` to `policy.to(dtype)`
  right after the policy is built, asserting the cast took. The same patch checks
  every technique's compile state.
- autocast (fp16): set the global autocast dtype and run the official loop with
  `--policy.use_amp=true` (its autocast reads the global default at construction).

`native` applies neither. Everything else in the official eval loop is unchanged.

Run as: `python -m bench._eval_entry --technique native --policy-path ... ...`
"""

from __future__ import annotations

import argparse
import sys

import torch

from .techniques import Technique, cast_weights, check_compile, get_technique


def _install_policy_hook(technique: Technique) -> None:
    import lerobot.scripts.lerobot_eval as evalmod

    original_make_policy = evalmod.make_policy

    def make_policy_for_technique(*args, **kwargs):
        policy = original_make_policy(*args, **kwargs)
        check_compile(policy, technique)
        if technique.cast_dtype is not None:
            cast_weights(policy, technique.cast_dtype)
            policy.eval()
        return policy

    evalmod.make_policy = make_policy_for_technique


def _apply_autocast_dtype(device: str, dtype: torch.dtype) -> None:
    torch.set_autocast_dtype(device, dtype)
    active = torch.get_autocast_dtype(device)
    if active != dtype:
        raise RuntimeError(f"autocast dtype not set for {device!r}: wanted {dtype}, got {active}")


def build_argv(args: argparse.Namespace, technique: Technique) -> list[str]:
    use_amp = technique.autocast_dtype is not None
    argv = [
        "lerobot-eval",
        f"--policy.path={args.policy_path}",
        f"--policy.device={args.device}",
        f"--policy.use_amp={'true' if use_amp else 'false'}",
        f"--policy.compile_model={'true' if technique.compile_model else 'false'}",
        "--env.type=libero",
        f"--env.task={args.suite}",
        f"--eval.n_episodes={args.episodes_per_task}",
        f"--eval.batch_size={args.batch_size}",
        f"--seed={args.seed}",
        f"--output_dir={args.output_dir}",
    ]
    if technique.compile_model:
        argv.append(f"--policy.compile_mode={technique.compile_mode}")
    if args.n_action_steps is not None:
        argv.append(f"--policy.n_action_steps={args.n_action_steps}")
    return argv


def main() -> None:
    ap = argparse.ArgumentParser(description="lerobot-eval wrapper with per-technique precision")
    ap.add_argument("--technique", required=True)
    ap.add_argument("--policy-path", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--episodes-per-task", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--n-action-steps", type=int, default=None)
    args = ap.parse_args()

    technique = get_technique(args.technique)
    _install_policy_hook(technique)
    if technique.autocast_dtype is not None:
        _apply_autocast_dtype(args.device, technique.autocast_dtype)

    sys.argv = build_argv(args, technique)
    from lerobot.scripts.lerobot_eval import main as lerobot_eval_main

    lerobot_eval_main()


if __name__ == "__main__":
    main()
