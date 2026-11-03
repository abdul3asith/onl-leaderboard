# onl-bench

Speed + LIBERO success benchmarks for VLA policies. One run → one JSON row in `results/`.

```bash
# full row (speed + 100-episode LIBERO-Object success eval, ~$1 of GPU)
modal run modal_app.py --model smolvla --technique native

# cheap speed-only smoke (skips the paid eval)
modal run modal_app.py --model smolvla --technique native --episodes-per-task 0

# per-stage latency breakdown across techniques (speed only, no eval)
modal run modal_app.py::breakdown_main --techniques native,fp32,fp16
```

Techniques are **weight dtypes**: `native` (as the checkpoint ships — bf16 for
the SmolVLA LIBERO checkpoint), `fp32`, `fp16`. Each casts the policy with
`policy.to(dtype)` so the labelled precision is the precision that runs.

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
| `bench/row.py` | build + validate + write the 16-field result row |
| `modal_app.py` | Modal app + `modal run` entrypoint |

## Precision & lerobot-eval

`lerobot-eval` has no CLI lever that casts SmolVLA's weights: `--policy.dtype` is
a no-op for SmolVLA, and `--policy.use_amp` only toggles autocast (a near no-op on
already-bf16 weights). So `_eval_entry` patches the eval module's `make_policy` to
`policy.to(dtype)` right after the policy is built, then runs the official eval
loop unchanged (`native` runs it with no patch). It asserts the cast took, so a
run can't silently report the wrong precision — matching how the speed path casts.

`--eval.n_episodes` is **per task**; LIBERO-Object has 10 tasks, so
`--episodes-per-task 10` → 100 total episodes (seed 1000), per CLAUDE.md.

## Dev

```bash
uv sync
uv run ruff check .
uv run pytest        # GPU-free plumbing tests
```
