"""Paired evaluation without a GPU: episode files, pairing, the recorder and the scripts."""

import json
import sys
import types

import numpy as np
import pytest
import torch

from bench import episodes, row, success
from bench._eval_entry import EpisodeRecorder, noise_seed, obs_hashes, pin_noise
from bench.models import get_model
from bench.techniques import get_technique
from scripts.certify_rows import certify_all

RUN = {"gpu": "NVIDIA L40S", "batch_size": 1, "git_sha": "abc1234"}


def _episodes(outcomes, **overrides):
    """One episode per outcome: task 0, initial states 0, 1, 2, ..., seeds 1000, 1001, ..."""
    return [
        {
            "task_id": 0,
            "task": "pick_up_the_ketchup",
            "episode_ix": i,
            "init_state_id": i,
            "seed": 1000 + i,
            "success": ok,
            "init_state_hash": f"state{i}",
            "obs_hash": f"obs{i}",
            "noise_hash": f"noise{i}",
            **RUN,
            **overrides,
        }
        for i, ok in enumerate(outcomes)
    ]


# --- episode files


def test_episodes_id_names_the_eval_and_its_repeats():
    name = episodes.episodes_id("smolvla", "smolvla_libero", "native", "libero_object")
    assert name == "smolvla_smolvla_libero_native_libero_object"
    repeat = episodes.episodes_id("smolvla", "smolvla_libero", "native", "libero_object", "repeat")
    assert repeat == name + "__repeat"


def test_write_and_read_round_trip_one_line_per_episode(tmp_path):
    eps = _episodes([True, False, True])
    path = episodes.write_episodes(eps, tmp_path / "episodes" / "x.jsonl")
    assert len(path.read_text().splitlines()) == 3
    assert episodes.read_episodes(path) == eps
    assert tuple(eps[0]) == episodes.FIELDS


@pytest.mark.parametrize(
    "broken",
    [
        [],
        _episodes([True]) + _episodes([False]),  # the same task, state and seed twice
        _episodes([True, 1]),  # success must be a bool
        [{k: v for k, v in _episodes([True])[0].items() if k != "noise_hash"}],
        _episodes([True]) + _episodes([True, True], gpu="NVIDIA A10")[1:],  # two GPUs in one eval
    ],
)
def test_invalid_episodes_are_refused(broken):
    with pytest.raises(ValueError):
        episodes.validate_episodes(broken)


def test_success_fields_come_from_the_episodes():
    fields = episodes.success_fields(_episodes([True] * 75 + [False] * 25))
    assert fields == {
        "success": 75.0,
        "ci_low": pytest.approx(65.6955364519384),  # what lerobot reported for 75 of 100
        "ci_high": pytest.approx(82.45478863771231),
        "episodes": 100,
        "success_gpu": "NVIDIA L40S",
        "success_git_sha": "abc1234",
    }


# --- pairing and certification


def test_aif_counts_only_native_success_with_technique_failure():
    native = _episodes([True, True, False, False] + [True] * 96)
    technique = _episodes([True, False, True, False] + [True] * 96)
    fields = episodes.aif_fields(native, technique)
    assert fields["aif_count"] == 1  # episode 1; episode 2 is a gain, episode 3 fails in both
    assert fields["aif_rate"] == 0.01
    assert fields["certified"] is False


def test_aif_with_no_new_failures_certifies_at_100_episodes():
    native = _episodes([True] * 75 + [False] * 25)
    technique = _episodes([True] * 100)
    fields = episodes.aif_fields(native, technique)
    assert fields["aif_count"] == 0 and fields["certified"] is True
    assert fields["aif_upper"] == pytest.approx(1 - 0.05**0.01)


def test_pairing_ignores_file_order():
    native = _episodes([True, False, True])
    technique = list(reversed(_episodes([False, False, True])))
    assert episodes.aif_fields(native, technique)["aif_count"] == 1


@pytest.mark.parametrize("field", ["init_state_hash", "noise_hash", "batch_size"])
def test_aif_refuses_pairs_that_did_not_start_the_same(field):
    native = _episodes([True, True])
    technique = _episodes([True, True], **{field: "other"})
    with pytest.raises(ValueError, match=field):
        episodes.aif_fields(native, technique)


def test_aif_does_not_require_the_rendered_observation_to_match():
    native = _episodes([True, True])
    technique = _episodes([True, False], obs_hash="rendered differently")
    assert episodes.aif_fields(native, technique)["aif_count"] == 1


def test_pairing_refuses_evals_of_different_episodes():
    with pytest.raises(ValueError, match="same tasks"):
        episodes.pair(_episodes([True, True]), _episodes([True]))
    with pytest.raises(ValueError, match="same tasks"):
        episodes.pair(_episodes([True]), _episodes([True], seed=7))


def test_match_report_counts_flips_in_both_directions():
    a = _episodes([True, True, False, False])
    b = _episodes([True, False, True, False], obs_hash="x")
    report = episodes.match_report(a, b)
    assert report["episodes"] == 4
    assert report["outcome_matches"] == 2 and report["outcome_match_rate"] == 0.5
    assert report["a_only_successes"] == 1 and report["b_only_successes"] == 1
    assert report["a_success"] == 50.0 and report["b_success"] == 50.0
    assert report["init_state_hash_matches"] == 4 and report["noise_hash_matches"] == 4
    assert report["obs_hash_matches"] == 0
    assert report["flipped"] == [
        {"task_id": 0, "init_state_id": 1, "seed": 1001},
        {"task_id": 0, "init_state_id": 2, "seed": 1002},
    ]


def test_match_report_of_an_eval_with_itself_is_perfect():
    a = _episodes([True, False, True])
    report = episodes.match_report(a, a)
    assert report["outcome_match_rate"] == 1.0 and report["flipped"] == []


# --- row fields


def _row(technique="compile_autotune", episodes_run=100, **aif):
    r = row.build_row(
        model="smolvla",
        checkpoint="smolvla_libero",
        gpu="NVIDIA L40S",
        technique=technique,
        denoise_steps=10,
        n_action_steps=1,
        compile_s=12.0 if technique.startswith("compile") else None,
        speed={"p50_ms": 1.0, "p99_ms": 2.0, "jitter_ms": 0.1, "hz": 500.0, "peak_mb": 42.0},
        success=(
            {"success": 75.0, "ci_low": 65.0, "ci_high": 82.0, "episodes": episodes_run}
            if episodes_run
            else None
        ),
        date="2026-11-05",
        lerobot_version="0.6.2",
        git_sha="abc1234",
    )
    r.update(aif)
    return r


def test_a_new_row_has_null_certification_fields():
    r = _row()
    assert [r[f] for f in episodes.AIF_FIELDS] == [None] * 4
    assert row.validate_row(r) is r


_GOOD = {"aif_count": 1, "aif_rate": 0.01, "aif_upper": 0.056, "certified": False}


@pytest.mark.parametrize(
    "kwargs,ok",
    [
        ({**_GOOD}, True),
        ({"aif_count": 0, "aif_rate": 0.0, "aif_upper": 0.03, "certified": True}, True),
        ({**_GOOD, "certified": None}, False),  # all four or none
        ({**_GOOD, "aif_rate": 0.02}, False),  # not count / episodes
        ({**_GOOD, "aif_upper": 0.005}, False),  # an upper bound below the observed rate
        ({**_GOOD, "aif_count": 1.0}, False),
        ({**_GOOD, "certified": "yes"}, False),
        ({**_GOOD, "technique": "native"}, False),  # native is the reference, never certified
        ({**_GOOD, "episodes_run": 0}, False),  # no eval, nothing to certify
    ],
)
def test_validate_row_checks_certification_fields(kwargs, ok):
    r = _row(
        technique=kwargs.pop("technique", "compile_autotune"),
        episodes_run=kwargs.pop("episodes_run", 100),
        **kwargs,
    )
    if ok:
        row.validate_row(r)
    else:
        with pytest.raises(ValueError):
            row.validate_row(r)


# --- the recorder, against a stand-in for lerobot's eval module


class _Env:
    """Two sub-envs that, like LiberoEnv, move on by `num_envs` initial states per reset."""

    num_envs = 2

    def __init__(self):
        self.states = np.arange(50 * 3, dtype=np.float64).reshape(50, 3)
        self.ids = [0, 1]

    def call(self, name):
        values = {"init_state_id": self.ids, "_init_states": [self.states] * 2, "task": ["t"] * 2}
        return tuple(values[name])

    def reset(self, seed=None, options=None):
        obs = {
            "pixels": {"image": np.stack([self.states[i] for i in self.ids])},
            "agent_pos": np.array(seed, dtype=np.float64)[:, None],
        }
        self.ids = [i + self.num_envs for i in self.ids]
        return obs, {}


def _policy():
    model = types.SimpleNamespace(
        config=types.SimpleNamespace(chunk_size=4, max_action_dim=3),
        sample_noise=lambda shape, device: torch.randn(shape, device=device),
    )
    model.sample_actions = lambda images, masks, tokens, noise=None: noise
    return types.SimpleNamespace(model=model)


def _fake_evalmod(plans_per_rollout):
    """rollout / eval_policy / run_one with lerobot's call shapes, looked up on the module."""
    mod = types.SimpleNamespace()

    def rollout(env, policy, seeds=None):
        env.reset(seed=seeds, options={})
        for _ in range(plans_per_rollout.pop(0)):
            policy.model.sample_actions(None, None, torch.zeros(env.num_envs, 5))
        return {}

    def eval_policy(env, policy, n_episodes, start_seed):
        seeds = []
        for batch in range(-(-n_episodes // env.num_envs)):
            batch_seeds = [start_seed + batch * env.num_envs + i for i in range(env.num_envs)]
            mod.rollout(env=env, policy=policy, seeds=batch_seeds)
            seeds += batch_seeds
        return {"per_episode": [{"seed": s, "success": s % 2 == 0} for s in seeds[:n_episodes]]}

    def run_one(task_group, task_id, env, **kwargs):
        return task_group, task_id, mod.eval_policy(env, **kwargs)

    mod.rollout, mod.eval_policy, mod.run_one = rollout, eval_policy, run_one
    return mod


def _record(path, plans_per_rollout):
    recorder = EpisodeRecorder(path)
    mod = _fake_evalmod(list(plans_per_rollout))
    recorder.install(mod)
    policy = _policy()
    pin_noise(policy, recorder.saw_noise)
    envs = []
    for task_id in (0, 1):
        envs.append(_Env())
        mod.run_one(
            "libero_object", task_id, envs[-1], policy=policy, n_episodes=3, start_seed=1000
        )
    return episodes.read_partial(path), envs


def test_recorder_writes_one_record_per_episode(tmp_path):
    records, envs = _record(tmp_path / "e.jsonl", [1, 1, 1, 1])
    assert len(records) == 6  # 3 per task: the second batch's spare env is dropped, as lerobot does
    assert all(tuple(r) == episodes.EPISODE_FIELDS for r in records)
    assert [r["task_id"] for r in records] == [0, 0, 0, 1, 1, 1]
    assert [r["episode_ix"] for r in records] == [0, 1, 2, 0, 1, 2]
    assert [r["init_state_id"] for r in records] == [0, 1, 2, 0, 1, 2]
    assert [r["seed"] for r in records] == [1000, 1001, 1002] * 2
    assert [r["success"] for r in records] == [True, False, True] * 2
    assert "reset" not in vars(envs[0])  # the env's own reset is back in place


def test_recorder_fingerprints_tell_scenes_apart_and_repeat_exactly(tmp_path):
    first, _ = _record(tmp_path / "a.jsonl", [1, 1, 1, 1])
    again, _ = _record(tmp_path / "b.jsonl", [1, 1, 1, 1])
    assert first == again
    for field in ("init_state_hash", "obs_hash", "noise_hash"):
        assert len({r[field] for r in first[:3]}) == 3  # three episodes, three fingerprints
    # Both tasks share initial-state ids and seeds here, but never the noise.
    assert first[0]["init_state_hash"] == first[3]["init_state_hash"]
    assert first[0]["noise_hash"] != first[3]["noise_hash"]


def test_noise_of_an_episode_does_not_depend_on_how_long_earlier_episodes_ran(tmp_path):
    short, _ = _record(tmp_path / "a.jsonl", [1, 1, 1, 1])
    long, _ = _record(tmp_path / "b.jsonl", [7, 2, 5, 3])  # more plans drawn before each next one
    assert [r["noise_hash"] for r in short] == [r["noise_hash"] for r in long]


def test_noise_seed_is_distinct_across_tasks_and_seeds():
    seeds = {noise_seed(task, seed) for task in range(10) for seed in range(1000, 1050)}
    assert len(seeds) == 500


def test_recorder_refuses_unseeded_rollouts(tmp_path):
    recorder = EpisodeRecorder(tmp_path / "e.jsonl")
    mod = _fake_evalmod([1])
    recorder.install(mod)
    with pytest.raises(RuntimeError, match="seeded"):
        mod.rollout(env=_Env(), policy=_policy(), seeds=None)


def test_obs_hashes_are_per_env_and_cover_nested_arrays():
    obs = {"pixels": {"image": np.zeros((2, 4, 4, 3), np.uint8)}, "agent_pos": np.zeros((2, 8))}
    a, b = obs_hashes(obs, 2)
    assert a == b
    obs["pixels"]["image"][1, 0, 0, 0] = 1  # one pixel of the second env
    a2, b2 = obs_hashes(obs, 2)
    assert a2 == a and b2 != b


def test_pin_noise_hands_eager_noise_to_the_wrapped_function():
    policy = _policy()
    got = []
    policy.model.sample_actions = lambda *args, noise=None: got.append(noise)
    pin_noise(policy)
    tokens = torch.zeros(2, 5)

    torch.manual_seed(7)
    policy.model.sample_actions(None, None, tokens)
    torch.manual_seed(7)
    expected = torch.randn(2, 4, 3)
    assert torch.equal(got[0], expected)  # drawn from the global stream, shaped (batch, chunk, dim)

    given = torch.ones(2, 4, 3)
    policy.model.sample_actions(None, None, tokens, noise=given)
    assert got[1] is given  # a caller's own noise is passed through untouched


# --- run_success: episodes are accepted only if they add up to lerobot's totals


def _fake_eval(n_success_reported):
    def run(cmd, **kwargs):
        out = cmd[cmd.index("--output-dir") + 1]
        partial = cmd[cmd.index("--episodes-out") + 1]
        records = [{f: ep[f] for f in episodes.EPISODE_FIELDS} for ep in _episodes([True, False])]
        with open(partial, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in records)
        overall = {
            "pc_success": 50.0,
            "pc_success_ci95": [9.0, 91.0],
            "n_episodes": 2,
            "n_success": n_success_reported,
        }
        with open(f"{out}/eval_info.json", "w") as f:
            json.dump({"overall": overall}, f)

    return run


def _run_success(tmp_path):
    (tmp_path / "out").mkdir(exist_ok=True)
    return success.run_success(
        get_model("smolvla"),
        get_technique("native"),
        episodes_per_task=1,
        seed=1000,
        batch_size=1,
        device="cuda",
        output_dir=tmp_path / "out",
        episodes_path=tmp_path / "episodes" / "e.jsonl",
        **{k: v for k, v in RUN.items() if k != "batch_size"},
    )


def test_run_success_writes_episodes_with_the_runs_fields(tmp_path, monkeypatch):
    (tmp_path / "episodes").mkdir()
    monkeypatch.setattr(success.subprocess, "run", _fake_eval(n_success_reported=1))
    assert _run_success(tmp_path)["episodes"] == 2
    assert episodes.read_episodes(tmp_path / "episodes" / "e.jsonl") == _episodes([True, False])
    assert not (tmp_path / "episodes" / "e.jsonl.partial").exists()


def test_run_success_refuses_episodes_that_disagree_with_lerobot(tmp_path, monkeypatch):
    (tmp_path / "episodes").mkdir()
    monkeypatch.setattr(success.subprocess, "run", _fake_eval(n_success_reported=2))
    with pytest.raises(RuntimeError, match="lerobot says"):
        _run_success(tmp_path)
    assert not (tmp_path / "episodes" / "e.jsonl").exists()


def test_run_success_never_overwrites_an_evals_episodes(tmp_path, monkeypatch):
    (tmp_path / "episodes").mkdir()
    (tmp_path / "episodes" / "e.jsonl").write_text("kept")
    monkeypatch.setattr(success.subprocess, "run", lambda *a, **k: pytest.fail("eval was launched"))
    with pytest.raises(FileExistsError):
        _run_success(tmp_path)
    assert (tmp_path / "episodes" / "e.jsonl").read_text() == "kept"


# --- scripts/certify_rows.py


def _write_episodes(results, technique, outcomes, **overrides):
    name = episodes.episodes_id("smolvla", "smolvla_libero", technique, "libero_object")
    episodes.write_episodes(
        _episodes(outcomes, **overrides), results / "episodes" / f"{name}.jsonl"
    )


def test_certify_rows_fills_success_and_certification_from_episodes(tmp_path):
    native_path = row.write_row(_row("native"), tmp_path)
    fast_path = row.write_row(_row("compile_autotune"), tmp_path)
    fp16_path = row.write_row(_row("fp16"), tmp_path)  # evaluated earlier, no episodes kept
    _write_episodes(tmp_path, "native", [True] * 80 + [False] * 20)
    _write_episodes(tmp_path, "compile_autotune", [False] + [True] * 99, git_sha="def5678")

    certify_all(tmp_path)

    native, fast, fp16 = (json.loads(p.read_text()) for p in (native_path, fast_path, fp16_path))
    assert native["success"] == 80.0 and native["success_git_sha"] == "abc1234"
    assert [native[f] for f in episodes.AIF_FIELDS] == [None] * 4
    assert fast["success"] == 99.0 and fast["success_git_sha"] == "def5678"
    assert (fast["aif_count"], fast["aif_rate"], fast["certified"]) == (1, 0.01, False)
    assert fp16["success"] == 75.0  # untouched
    assert [fp16[f] for f in episodes.AIF_FIELDS] == [None] * 4
    assert all(list(r) == list(row.FIELDS) for r in (native, fast, fp16))


def test_certify_rows_writes_nothing_when_a_pair_is_invalid(tmp_path):
    native_path = row.write_row(_row("native"), tmp_path)
    row.write_row(_row("compile_autotune"), tmp_path)
    _write_episodes(tmp_path, "native", [True, True])
    _write_episodes(tmp_path, "compile_autotune", [True, False], noise_hash="other")
    before = native_path.read_text()
    with pytest.raises(ValueError, match="noise_hash"):
        certify_all(tmp_path)
    assert native_path.read_text() == before


def test_eval_entry_requires_an_episodes_file(monkeypatch):
    from bench import _eval_entry

    argv = ["x", "--technique", "native", "--policy-path", "p", "--suite", "s"]
    monkeypatch.setattr(sys, "argv", argv + ["--episodes-per-task", "1", "--output-dir", "o"])
    with pytest.raises(SystemExit):
        _eval_entry.main()
