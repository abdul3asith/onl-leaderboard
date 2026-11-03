"""onl-bench: speed + LIBERO success benchmarks for VLA policies.

Kept import-light on purpose: the top-level package pulls in no torch/lerobot so
that the Modal local entrypoint can `from bench.row import write_row` without a
GPU-side dependency installed locally.
"""
