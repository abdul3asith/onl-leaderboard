# onl-bench rules

## Scope (v1 paper) — do not expand without asking
- Models: smolvla, pi05
- Techniques: native, fp16, compile_autotune, int8, int4, int8_compile, int4_compile
- GPUs: L40S (main), L4, H100. Pin the GPU; a run on the wrong GPU writes nothing.
- Suites: libero_object (all rows), libero_spatial (key rows: native, compile_autotune, best int)

## Measurement rules
- Every number in results/ comes from a script run. Never type numbers by hand.
- Speed: batch 1, includes pre/post, 10 warmup + 200 timed steps.
- Success: 100 episodes per suite (10 tasks x 10), seed 1000, eval batch 1 where feasible.
- Paired evaluation: technique and native run the SAME scenes and seeds.
  Store per-episode outcomes in results/episodes/<row_id>.jsonl.
- Acceleration-induced failure (AIF) = native succeeds AND technique fails.
- Certification (CARE-style): risk budget alpha=0.05, confidence 95%,
  Hoeffding-Bentkus bound. Fields: aif_count, aif_rate, aif_upper, certified (bool).

## Row schema (one JSON per row)
model, checkpoint, technique, suite, gpu, success_gpu, p50_ms, p99_ms, jitter_ms,
hz, actions_per_chunk, actions_per_s, peak_mb, compile_s, denoise_steps,
success, ci_low, ci_high, episodes, aif_count, aif_rate, aif_upper, certified,
date, lerobot_version, torch_version, torchao_version, git_sha, success_git_sha

## Process
- Python 3.12, uv, ruff. Modal for all GPU work, always `modal run --detach`,
  results saved to the Modal Volume, 2h timeout.
- Speed + breakdown first; paid success evals only after I approve.
- Report anything surprising; never hide or "fix" a number.
- Small commits; tests pass before review. Max 2 agents launching GPU runs at once.

## Budget and cost control
- Budget cap: $15 total. Report estimated cost before every paid run and wait for my OK.
- Cache compiled artifacts on the Volume; never compile the same config twice.
- Speed runs before evals. No eval reruns without asking.
- Stop any app that isn't progressing within 10 minutes.
