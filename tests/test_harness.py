"""Fast, GPU-free tests for the harness plumbing.

These exercise everything that does not need a real policy or GPU: the profiler
loop on CPU, the result-row schema, the technique/model registries, and the
lerobot-eval argv the wrapper builds per technique.
"""

import argparse

import pytest
import torch

from bench import profiler, row
from bench._eval_entry import build_argv
from bench.models import get_model
from bench.techniques import get_technique


def test_profiler_cpu_shape():
    calls = {"n": 0}

    def step():
        calls["n"] += 1

    stats = profiler.profile(step, "cpu", steps=5, warmup=3)
    assert calls["n"] == 8  # warmup + timed
    assert set(stats) == {"p50_ms", "p99_ms", "jitter_ms", "hz", "peak_mb"}
    assert all(isinstance(v, float) for v in stats.values())
    assert stats["peak_mb"] == 0.0  # no CUDA on cpu


def test_profiler_protocol_defaults():
    assert profiler.WARMUP == 10
    assert profiler.TIMED == 200


def _speed():
    return {"p50_ms": 1.0, "p99_ms": 2.0, "jitter_ms": 0.1, "hz": 500.0, "peak_mb": 42.0}


def test_build_row_with_success():
    r = row.build_row(
        model="smolvla",
        checkpoint="smolvla_libero",
        gpu="NVIDIA A10G",
        technique="fp32",
        speed=_speed(),
        success={"success": 78.0, "ci_low": 68.0, "ci_high": 85.0, "episodes": 100},
        date="2026-11-02",
        lerobot_version="0.6.2",
        git_sha="abc1234",
    )
    assert set(r) == set(row.FIELDS)
    assert r["success"] == 78.0 and r["episodes"] == 100
    assert r["hz"] == 500.0


def test_build_row_speed_only():
    r = row.build_row(
        model="smolvla",
        checkpoint="smolvla_libero",
        gpu="NVIDIA A10G",
        technique="fp32",
        speed=_speed(),
        success=None,
        date="2026-11-02",
        lerobot_version="0.6.2",
        git_sha="abc1234",
    )
    assert r["success"] is None and r["episodes"] == 0


def test_validate_row_rejects_missing_and_nonnumeric():
    with pytest.raises(ValueError):
        row.validate_row({"model": "x"})
    bad = row.build_row(
        model="smolvla",
        checkpoint="c",
        gpu="g",
        technique="fp32",
        speed=_speed(),
        success=None,
        date="2026-11-02",
        lerobot_version="0.6.2",
        git_sha="sha",
    )
    bad["p50_ms"] = "not-a-number"
    with pytest.raises(ValueError):
        row.validate_row(bad)


def test_default_filename_is_safe():
    r = row.build_row(
        model="smolvla",
        checkpoint="smolvla_libero",
        gpu="NVIDIA A10G",
        technique="bf16",
        speed=_speed(),
        success=None,
        date="2026-11-02",
        lerobot_version="0.6.2",
        git_sha="abc1234",
    )
    name = row.default_filename(r)
    assert " " not in name and name.endswith(".json")
    assert name == "smolvla_smolvla_libero_bf16_NVIDIA_A10G_2026-11-02.json"


def test_write_row_roundtrip(tmp_path):
    import json

    r = row.build_row(
        model="smolvla",
        checkpoint="smolvla_libero",
        gpu="A10G",
        technique="fp32",
        speed=_speed(),
        success={"success": 78.0, "ci_low": 68.0, "ci_high": 85.0, "episodes": 100},
        date="2026-11-02",
        lerobot_version="0.6.2",
        git_sha="abc1234",
    )
    path = row.write_row(r, tmp_path)
    assert json.loads(path.read_text()) == r


def test_technique_precision_levers():
    native = get_technique("native")
    assert native.cast_dtype is None and native.autocast_dtype is None  # as shipped
    fp32 = get_technique("fp32")
    assert fp32.cast_dtype is torch.float32 and fp32.autocast_dtype is None  # weight upcast
    fp16 = get_technique("fp16")
    assert fp16.cast_dtype is None and fp16.autocast_dtype is torch.float16  # autocast
    with pytest.raises(KeyError):
        get_technique("int4")


def _args(technique):
    return argparse.Namespace(
        technique=technique,
        policy_path="HuggingFaceVLA/smolvla_libero",
        suite="libero_object",
        episodes_per_task=10,
        batch_size=10,
        seed=1000,
        device="cuda",
        output_dir="/tmp/eval/libero_object",
    )


@pytest.mark.parametrize("tech,use_amp", [("native", False), ("fp32", False), ("fp16", True)])
def test_build_argv_amp_matches_technique(tech, use_amp):
    # fp32 upcasts weights (no AMP); fp16 uses autocast (AMP); native neither.
    t = get_technique(tech)
    argv = build_argv(_args(tech), t)
    assert f"--policy.use_amp={'true' if use_amp else 'false'}" in argv
    assert "--env.task=libero_object" in argv
    assert "--eval.n_episodes=10" in argv  # per-task; 10 tasks => 100 total
    assert "--seed=1000" in argv


def test_model_registry_one_entry():
    spec = get_model("smolvla")
    assert spec.policy_path == "HuggingFaceVLA/smolvla_libero"
    assert spec.suite == "libero_object"
    with pytest.raises(KeyError):
        get_model("nope")
