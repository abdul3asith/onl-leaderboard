# onl-bench

Speed + LIBERO success benchmarks for VLA policies. One run → one JSON row in `results/`.

```bash
# full row (speed + 100-episode LIBERO-Object success eval, ~$1 of GPU)
modal run modal_app.py --model smolvla --technique native

# cheap speed-only smoke (skips the paid eval)
modal run modal_app.py --model smolvla --technique native --episodes-per-task 0

# per-stage latency breakdown across techniques (speed only, no eval)
modal run modal_app.py::breakdown_main --techniques native,fp32,fp16,compile

# a large model: pick the model and GPU; the row records the GPU that really ran
modal run --detach modal_app.py --model pi05 --technique fp32 --gpu L40S
```

Needs a Modal secret `huggingface` holding `HF_TOKEN`, from an account that has accepted
the license of [google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224)
(pi05's gated tokenizer).

Techniques are **weight dtypes**: `native` (as the checkpoint ships — bf16 for
the SmolVLA LIBERO checkpoint), `fp32`, `fp16`. Each casts the policy with
`policy.to(dtype)` so the labelled precision is the precision that runs. `compile` is
`native` plus `torch.compile` (`compile_model=true`, mode `max-autotune`).

**A row is either fully compiled or fully uncompiled.** The technique sets
`compile_model` in both the speed path and the eval, overriding the checkpoint: pi05
stores `true`, yet its `native`, `fp32` and `fp16` rows run uncompiled. A policy that
does not match its label stops the run.

## Layout

| file | role |
|------|------|
| `bench/models.py` | model registry — adding a model is one `ModelSpec` entry |
| `bench/techniques.py` | precision techniques (autocast dtype per technique) |
| `bench/policy.py` | load policy + build synthetic observation (from dpc-two) |
| `bench/profiler.py` | step timing: batch 1, 10 warmup + 200 timed |
| `bench/speed.py` | run the speed benchmark under one technique |
| `bench/success.py` | run the LIBERO eval, read `eval_info.json` |
| `bench/_eval_entry.py` | wrapper that drives lerobot-eval precision per technique |
| `bench/row.py` | build + validate + write the 18-field result row |
| `modal_app.py` | Modal app + `modal run` entrypoint |

## Precision & lerobot-eval

`lerobot-eval` has no CLI lever that casts SmolVLA's weights: `--policy.dtype` is
a no-op for SmolVLA, and `--policy.use_amp` only toggles autocast (a near no-op on
already-bf16 weights). So `_eval_entry` patches the eval module's `make_policy` to
`policy.to(dtype)` right after the policy is built, then runs the official eval
loop unchanged (`native` runs it with no patch). It asserts the cast took, so a
run can't silently report the wrong precision — matching how the speed path casts.

pi05 caches its build-time precision and picks its vision autocast from it, so
`cast_weights` updates that too. Its `fp16` row is only fp16 outside the vision tower: the
policy's own nested bf16 autocast wins inside it.

`denoise_steps` and `n_action_steps` are recorded in every row because they drive speed.
`p50_ms`/`hz` time one full chunk generation, so one model's `hz` is a control rate only
when `n_action_steps` is 1. pi05 runs at `n_action_steps=10` (the checkpoint stores 50),
as LeRobot's published LIBERO scores do.

`--eval.n_episodes` is **per task**; LIBERO-Object has 10 tasks, so
`--episodes-per-task 10` → 100 total episodes (seed 1000), per CLAUDE.md.

## Dev

```bash
uv sync
uv run ruff check .
uv run pytest        # GPU-free plumbing tests
```
