"""Wrapper entrypoint around lerobot-eval that applies a technique's precision.

lerobot-eval has no CLI lever that casts SmolVLA's weights (`--policy.dtype` is a
no-op for SmolVLA), and its autocast is always the device-default dtype. So this
wrapper reproduces, in the eval, the same two levers the speed path uses:

- weight cast (fp32): patch the eval module's `make_policy` to `policy.to(dtype)`
  right after the policy is built, asserting the cast took. The same patch checks
  every technique's compile state.
- autocast (fp16): set the global autocast dtype and run the official loop with
  `--policy.use_amp=true` (its autocast reads the global default at construction).

`native` applies neither.

The eval is also paired: a technique must meet the same episodes as native, so that a
failure can be attributed to the technique. The official loop seeds the environment per
episode but draws the policy's noise from one stream for the whole eval, so one episode
ending a step earlier shifts the noise of every later episode. Two changes fix that, for
every technique alike:

- the noise stream is reseeded at the start of each rollout from its task and seed;
- each plan's starting noise is drawn in eager mode and passed to `sample_actions`.
  torch.compile draws random numbers with its own generator, so from the same seed a
  compiled policy would otherwise start from different noise than native.

`EpisodeRecorder` then writes one record per episode with the scene it started from, its
seed and its outcome. Everything else in the official eval loop is unchanged.

Run as: `python -m bench._eval_entry --technique native --policy-path ... ...`
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import torch

from .episodes import EPISODE_FIELDS
from .techniques import Technique, cast_weights, check_compile, get_technique


def noise_seed(task_id: int, seed: int) -> int:
    """Seed of the policy's noise stream for one rollout: fixed per episode, distinct per task."""
    return 1_000_003 * (task_id + 1) + seed


def _digest(*arrays) -> str:
    h = hashlib.sha1()
    for array in arrays:
        h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()[:16]


def _leaves(obs):
    """The arrays of a possibly nested observation dict, in a fixed order."""
    if isinstance(obs, dict):
        for name in sorted(obs):
            yield from _leaves(obs[name])
    else:
        yield np.asarray(obs)


def obs_hashes(obs, n_envs: int) -> list[str]:
    """One fingerprint per sub-env of a batched observation."""
    leaves = list(_leaves(obs))
    return [_digest(*(leaf[i] for leaf in leaves)) for i in range(n_envs)]


def pin_noise(policy, on_noise=None) -> None:
    """Draw each plan's starting noise in eager mode, outside any compiled code."""
    model = policy.model
    inner = model.sample_actions

    def sample_actions(*args, noise=None, **kwargs):
        if noise is None:
            tokens = args[2]  # language tokens, in both SmolVLA's and pi05's signature
            shape = (tokens.shape[0], model.config.chunk_size, model.config.max_action_dim)
            noise = model.sample_noise(shape, tokens.device)
            if on_noise is not None:
                on_noise(noise)
        return inner(*args, noise=noise, **kwargs)

    model.sample_actions = sample_actions


class EpisodeRecorder:
    """Writes one JSON line per episode as each task finishes.

    Wraps three functions of lerobot's eval module: `run_one` to know the task, `rollout`
    to reseed the noise and fingerprint how each episode starts, and `eval_policy` to join
    those starts with the outcomes lerobot computed.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")
        self._task_id: int | None = None
        self._rollouts: list[list[dict]] = []
        self._noise: list[str] | None = None

    def saw_noise(self, noise: torch.Tensor) -> None:
        if self._noise is None:  # only a rollout's first plan is fingerprinted
            self._noise = [_digest(row) for row in noise.detach().float().cpu().numpy()]

    @staticmethod
    def _scenes(env) -> list[dict]:
        """The task and initial state each sub-env is about to reset to."""
        scenes = []
        ids, states, tasks = (env.call(name) for name in ("init_state_id", "_init_states", "task"))
        for init_id, init_states, task in zip(ids, states, tasks, strict=True):
            init_id = int(init_id) % len(init_states)  # as LiberoEnv.reset indexes them
            scenes.append(
                {
                    "task": task,
                    "init_state_id": init_id,
                    "init_state_hash": _digest(init_states[init_id]),
                }
            )
        return scenes

    def install(self, evalmod) -> None:
        run_one, rollout, eval_policy = evalmod.run_one, evalmod.rollout, evalmod.eval_policy
        rollout_signature = inspect.signature(rollout)

        def run_one_recorded(task_group, task_id, *args, **kwargs):
            self._task_id = int(task_id)
            return run_one(task_group, task_id, *args, **kwargs)

        def rollout_recorded(*args, **kwargs):
            call = rollout_signature.bind(*args, **kwargs).arguments
            env, seeds = call["env"], call.get("seeds")
            if not seeds:
                raise RuntimeError("a paired eval needs seeded rollouts")
            scenes = self._scenes(env)
            torch.manual_seed(noise_seed(self._task_id, seeds[0]))
            self._noise = None

            first_obs = []
            reset = env.reset

            def reset_recorded(*reset_args, **reset_kwargs):
                out = reset(*reset_args, **reset_kwargs)
                first_obs.append(out[0])
                return out

            env.reset = reset_recorded
            try:
                data = rollout(*args, **kwargs)
            finally:
                del env.reset

            if self._noise is None:
                raise RuntimeError("the policy drew no noise through pin_noise in this rollout")
            starts = zip(
                scenes, seeds, obs_hashes(first_obs[0], env.num_envs), self._noise, strict=True
            )
            self._rollouts.append(
                [
                    {**scene, "seed": int(seed), "obs_hash": obs, "noise_hash": noise}
                    for scene, seed, obs, noise in starts
                ]
            )
            return data

        def eval_policy_recorded(*args, **kwargs):
            self._rollouts = []
            info = eval_policy(*args, **kwargs)
            outcomes = info["per_episode"]
            # The last batch can run more envs than episodes remain; lerobot drops the extras.
            starts = [ep for batch in self._rollouts for ep in batch][: len(outcomes)]
            with self.path.open("a") as f:
                for ix, (start, outcome) in enumerate(zip(starts, outcomes, strict=True)):
                    if outcome["seed"] != start["seed"]:
                        raise RuntimeError(f"episode {ix}: seed {outcome['seed']} != {start}")
                    record = {
                        **start,
                        "task_id": self._task_id,
                        "episode_ix": ix,
                        "success": bool(outcome["success"]),
                    }
                    f.write(json.dumps({field: record[field] for field in EPISODE_FIELDS}) + "\n")
            return info

        evalmod.run_one = run_one_recorded
        evalmod.rollout = rollout_recorded
        evalmod.eval_policy = eval_policy_recorded


def _install_policy_hook(technique: Technique, recorder: EpisodeRecorder) -> None:
    import lerobot.scripts.lerobot_eval as evalmod

    original_make_policy = evalmod.make_policy

    def make_policy_for_technique(*args, **kwargs):
        policy = original_make_policy(*args, **kwargs)
        check_compile(policy, technique)
        if technique.cast_dtype is not None:
            cast_weights(policy, technique.cast_dtype)
            policy.eval()
        pin_noise(policy, recorder.saw_noise)
        return policy

    evalmod.make_policy = make_policy_for_technique
    recorder.install(evalmod)


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
    if args.task_ids:
        argv.append(f"--env.task_ids=[{args.task_ids}]")
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
    ap.add_argument("--episodes-out", required=True, help="JSONL file, one line per episode")
    ap.add_argument("--task-ids", default="", help="comma-separated subset of the suite's tasks")
    args = ap.parse_args()

    technique = get_technique(args.technique)
    _install_policy_hook(technique, EpisodeRecorder(args.episodes_out))
    if technique.autocast_dtype is not None:
        _apply_autocast_dtype(args.device, technique.autocast_dtype)

    sys.argv = build_argv(args, technique)
    from lerobot.scripts.lerobot_eval import main as lerobot_eval_main

    lerobot_eval_main()


if __name__ == "__main__":
    main()
