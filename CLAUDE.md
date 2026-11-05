# onl-bench rules
- Every number in results/ comes from a script run. Never type numbers by hand.
- One result row = one JSON file: model, checkpoint, gpu, technique,
  denoise_steps, n_action_steps, p50_ms, p99_ms, jitter_ms, hz, peak_mb, success, ci_low, ci_high,
  episodes, date, lerobot_version, git_sha.
- Speed: batch 1, includes pre/post-processing, 10 warmup + 200 timed steps.
- Success: LIBERO-Object, 100 episodes (10 tasks x 10), seed 1000.
- Python 3.12, uv, ruff. Modal for all GPU work.
- Only the success agent launches LIBERO evals (they cost ~$1 each).
- Small commits. Tests must pass before asking for review.
