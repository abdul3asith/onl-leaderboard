# onl-bench workflows

How the results are produced, from a GPU run to the public leaderboard. Measured values live
in `results/*.json` and the README's "Latest results"; the rules live in `CLAUDE.md`.

## Terminology

| Term | Meaning |
|------|---------|
| VLA policy | Vision-language-action model: camera images + a task sentence → robot actions |
| π0.5 (`pi05`), SmolVLA | The two policies benchmarked; both are LeRobot checkpoints |
| LIBERO-Object | Simulated manipulation suite with 10 tasks, used to measure success |
| Row | One JSON file in `results/`: one model + one technique, with its speed and success |
| Technique | How the policy is run: `native`, `fp32`, `fp16`, `compile`, `compile_autotune` |
| Plan / chunk | One model inference, producing a sequence of actions |
| `n_action_steps` | Actions used from each plan ("actions per chunk"): π0.5 10, SmolVLA 1 |
| `denoise_steps` | Flow-matching iterations inside one plan |
| p50 / p99 | Median and 99th-percentile time for one plan, in ms |
| `jitter_ms` | Standard deviation of plan time; lower means a steadier rhythm |
| `hz` | `1000 / p99`: plans per second you can count on, not actions per second |
| Actions per second | `n_action_steps × 1000 / p50`; computed by the board, not stored in rows |
| `peak_mb` | Peak GPU memory allocated during the timed steps |
| `compile_s` | Wall time of the first call on a compiled row; null when uncompiled |
| 95% CI | Wilson interval on success, reported by LeRobot's eval |
| Pinned GPU | The one card all speed is measured on: NVIDIA L40S |
| Modal | Cloud GPU platform that runs all GPU work; a Volume is its persistent storage |
| Space | Hugging Face app hosting; the leaderboard is a Gradio Space |

## Concepts

**Techniques**

| Technique | What it does |
|-----------|--------------|
| `native` | As the checkpoint ships: no cast, no autocast, uncompiled |
| `fp32` | Casts all weights with `policy.to(float32)` |
| `fp16` | `torch.autocast(float16)` over native weights; a weight cast crashes SmolVLA |
| `compile` | `native` + `torch.compile` in `default` mode |
| `compile_autotune` | `native` + `torch.compile` in `max-autotune` mode |

- **Same arithmetic in both paths.** A technique drives the speed run and the eval alike, so a
  row's two numbers describe the same setup. A policy that doesn't match its label stops the run.
- **Never mixed.** `compile_model` always comes from the technique. π0.5's checkpoint stores it
  on, so uncompiled techniques must override it.
- **π0.5 `fp16` is partial.** Its vision tower keeps its own bf16 autocast.

**Speed** runs a synthetic observation at batch 1, including pre- and post-processing:
10 warmup steps, then 200 timed. `reset()` before each step forces a real inference, so p50 and
p99 time a full plan rather than popping a cached action.

**Success** runs the official `lerobot-eval` loop in a subprocess: 10 episodes × 10 tasks,
seed 1000. `--eval.n_episodes` is per task, so pass 10 and read back 100.

**One GPU.** Speed is only comparable on one card. Modal's `A10` served both A10 and A10G, which
differed noticeably on the same work, so rows are pinned and a run on any other card stops.
Success doesn't depend on the card, so a row can reuse an earlier eval: `gpu` and `git_sha` say
where its speed ran, `success_gpu` and `success_git_sha` where its success ran.

**No typed numbers.** Rows come from runs. The board and the README's results section are
generated from rows, and tests fail if the README falls behind or a row is invalid.

## Workflows

Run from the repo root. Only the success agent launches LIBERO evals (about $1 each).

**1. Set up**

```bash
uv sync
modal token new
modal secret create huggingface HF_TOKEN=<read token>
```

The token's account must have accepted the
[PaliGemma license](https://huggingface.co/google/paligemma-3b-pt-224), which π0.5's tokenizer needs.

**2. Add a model:** add one `ModelSpec` in `bench/models.py`. Set `n_action_steps` only to
override the checkpoint, as π0.5 does to match LeRobot's published LIBERO scores.

**3. Check speed cheaply** (no eval, no success values):

```bash
modal run modal_app.py::main --model smolvla --technique native --episodes-per-task 0
```

**4. See where the time goes** (per-stage timing, dtypes, stray CPU tensors; no row):

```bash
modal run modal_app.py::breakdown_main --techniques native,fp32,fp16,compile,compile_autotune
```

**5. Produce a full row** (speed + eval, writes to `results/`):

```bash
modal run modal_app.py::main --model pi05 --technique native
```

For a long run, add `--detach`, then copy the row from the Volume:

```bash
modal volume get onl-bench-results <row>.json results/
```

Watch the logs as well as the status. π0.5 `fp32` on A10 kept retrying LIBERO renderers without
raising: progress stuck at 0/1 with repeated `MjRenderContext.__del__` errors.

**6. Rerun speed on the pinned GPU, keeping earlier success.** Run step 3 for the technique, then
merge. Rows missing newer fields need the backfill first:

```bash
uv run python scripts/backfill_speed_knobs.py results/
uv run python scripts/merge_speed_rows.py results/
```

The merge refuses mismatched settings, an off-pin speed row, or anything but exactly one evaluated
partner, and writes nothing unless every pair is valid.

**7. Update and check**

```bash
uv run python scripts/update_readme.py
uv run ruff check .
uv run pytest
uv run python board/app.py   # http://127.0.0.1:7860
```

**8. Publish the board** with a write token. This sends only what the Space needs and drops rows
no longer in `results/`:

```bash
hf auth login
hf upload basithcodes/onl-bench-leaderboard . . --repo-type space \
  --include README.md --include requirements.txt --include CLAUDE.md \
  --include board/app.py --include "results/*.json" --delete "results/*.json"
```

The first time, create it with
`hf repos create basithcodes/onl-bench-leaderboard --repo-type space --space-sdk gradio --public`.

## How the current results were built

1. Built the harness and ran SmolVLA `native`, `fp32` and `fp16`.
2. Added `denoise_steps`, `n_action_steps` and `compile_s` to rows, since they drive speed.
3. Added π0.5 and the compile techniques, with compile state forced per technique.
4. Pinned everything to L40S after seeing the A10/A10G gap; reran speed and merged it with the
   earlier success, recording where each came from.
5. Built the leaderboard and generated the README's results from the rows.

## Reading the results

- **Actions per second favors longer plans** by design: it assumes every action in a plan is used,
  and π0.5 uses more per plan than SmolVLA.
- **Compare CIs before calling a win.** Within each model, all techniques' CIs currently overlap,
  so success differences between them aren't meaningful.
- **`compile_s` is one-time** and covers only the speed path at batch 1. The eval compiles again
  at its own batch size.
- **Success GPU varies by row** (see the column). Speed never does.
