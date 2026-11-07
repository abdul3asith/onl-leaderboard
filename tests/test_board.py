"""The leaderboard shows what the rows say: nothing on the page is typed in."""

import json
import math
from importlib.metadata import version
from pathlib import Path

import pytest

from board import app
from scripts import update_readme

RULES = """# rules
- Every number in results/ comes from a script run.
- One result row = one JSON file: model,
  checkpoint.
- Speed: batch 3, includes pre/post-processing.
- One GPU type for every row.
- Success: Some-Suite, 40 episodes (4 tasks x 10), seed 42.
- Small commits. Tests must pass before asking for review.
"""


def _row(**overrides):
    r = {
        "model": "pi05",
        "gpu": "NVIDIA L40S",
        "technique": "native",
        "denoise_steps": 10,
        "n_action_steps": 10,
        "compile_s": None,
        "p50_ms": 200.0,
        "p99_ms": 250.0,
        "peak_mb": 1234.5,
        "success": 90.0,
        "ci_low": 80.0,
        "ci_high": 95.0,
        "episodes": 100,
        "success_gpu": "NVIDIA L40S",
        "date": "2026-11-05",
    }
    return {**r, **overrides}


def _speed_only(**overrides):
    gone = dict.fromkeys(("success", "ci_low", "ci_high", "success_gpu"))
    return _row(episodes=0, **gone, **overrides)


def _shown(rows, header):
    t = app.table(rows)
    i = t["headers"].index(header)
    return [r[i] for r in t["metadata"]["display_value"]], [r[i] for r in t["data"]]


def test_actions_per_second_is_chunk_times_1000_over_p50():
    assert app.actions_per_second(_row(n_action_steps=10, p50_ms=200.0)) == 50.0
    assert app.actions_per_second(_row(n_action_steps=1, p50_ms=200.0)) == 5.0


def test_table_has_every_requested_column_in_order():
    assert app.table([_row()])["headers"] == [
        "model",
        "technique",
        "success % (95% CI)",
        "p50 ms",
        "p99 ms",
        "actions per chunk",
        "actions per second",
        "peak MB",
        "compile_s",
        "denoise steps",
        "speed GPU",
        "success GPU",
        "date",
    ]


def test_table_shows_a_dash_for_missing_and_keeps_it_null_for_sorting():
    rows = [_speed_only()]
    for header in ("success % (95% CI)", "compile_s", "success GPU"):
        shown, sorted_on = _shown(rows, header)
        assert shown == ["–"] and sorted_on == [None]


def test_success_sorts_on_the_number_and_shows_the_interval():
    shown, sorted_on = _shown([_row()], "success % (95% CI)")
    assert shown == ["90.0% (80.0–95.0)"]
    assert sorted_on == [90.0]


def test_table_is_every_file_in_results():
    files = list(app.RESULTS_DIR.glob("*.json"))
    rows = app.load_rows()
    assert files and len(app.table(rows)["data"]) == len(files)
    on_disk = {json.loads(p.read_text())["p50_ms"] for p in files}
    assert set(_shown(rows, "p50 ms")[1]) == on_disk


def test_load_rows_puts_best_first_and_speed_only_last(tmp_path):
    rows = {
        "low": _row(success=50.0),
        "none": _speed_only(),
        "high_slow": _row(success=90.0, p50_ms=400.0, technique="slow"),
        "high_fast": _row(success=90.0, p50_ms=100.0, technique="fast"),
    }
    for name, r in rows.items():
        (tmp_path / f"{name}.json").write_text(json.dumps(r))
    assert [r["technique"] for r in app.load_rows(tmp_path)][:2] == ["fast", "slow"]
    assert app.load_rows(tmp_path)[-1]["success"] is None


def test_notes_read_as_written_for_the_current_kind_of_rows():
    rows = [
        _row(success_gpu="NVIDIA A10"),
        _row(success_gpu="NVIDIA A10G"),
        _row(model="smolvla", n_action_steps=1),
    ]
    rules = app.RULES_FILE.read_text()
    assert app.notes(rows, rules) == [
        "All speed measured on NVIDIA L40S, batch 1",
        "p50/p99 time one full plan (chunk); π0.5 uses 10 actions per plan, SmolVLA 1",
        "Success = 100 episodes per suite, seed 1000; "
        "some success runs were on A10/A10G (see column)",
        "Compile time is one-time, first call only",
    ]


def test_notes_follow_the_rows_and_the_rules_instead_of_hardcoding():
    rows = [
        _row(model="other", gpu="NVIDIA X", n_action_steps=7, episodes=40, success_gpu="NVIDIA X")
    ]
    speed, plan, success, _ = app.notes(rows, RULES)
    assert speed == "All speed measured on NVIDIA X, batch 3"
    assert plan.endswith("other uses 7 actions per plan")
    assert success == "Success = Some-Suite, 40 episodes, seed 42"  # no off-GPU clause


def test_notes_with_only_speed_rows_or_no_rows():
    assert not any("Success" in n for n in app.notes([_speed_only()], RULES))
    assert app.notes([], RULES) == []


def test_chart_axes_error_bars_colors_and_labels():
    rows = [
        _row(technique="native", success=90.0, ci_low=80.0, ci_high=95.0),
        _row(model="smolvla", technique="fp32", n_action_steps=1),
    ]
    fig = app.chart(rows)
    assert fig.layout.xaxis.type == "log"
    pi05, smolvla = fig.data
    assert (pi05.name, smolvla.name) == ("π0.5", "SmolVLA")
    assert pi05.marker.color != smolvla.marker.color
    assert list(pi05.x) == [50.0] and list(pi05.y) == [90.0]
    assert list(pi05.text) == ["native"]
    assert list(pi05.error_y.array) == [5.0] and list(pi05.error_y.arrayminus) == [10.0]


def test_chart_keeps_a_ci_bound_of_zero_and_skips_rows_without_success():
    fig = app.chart([_row(success=2.0, ci_low=0.0, ci_high=9.0), _speed_only(technique="speed")])
    assert len(fig.data) == 1 and list(fig.data[0].text) == ["native"]
    assert list(fig.data[0].error_y.arrayminus) == [2.0]


def test_chart_range_fits_every_point_and_stays_valid_when_empty():
    rows = [_row(p50_ms=1000.0), _row(p50_ms=10.0)]
    low, high = app.chart(rows).layout.xaxis.range
    speeds = [math.log10(app.actions_per_second(r)) for r in rows]
    assert low < min(speeds) and high > max(speeds)
    assert len(app.chart([]).data) == 0


def test_labels_go_right_unless_the_next_point_crowds_them():
    assert app._label_positions([3.58, 3.67, 30.0]) == [
        "middle left",
        "middle right",
        "middle right",
    ]


def test_methodology_comes_from_claude_md_without_the_repo_chores():
    text = app.methodology(RULES)
    assert "- Speed: batch 3, includes pre/post-processing." in text
    assert "- One result row = one JSON file: model, checkpoint." in text
    assert "Small commits" not in text


def test_methodology_stops_when_a_rule_disappears():
    with pytest.raises(ValueError, match="Success:"):
        app.methodology(RULES.replace("- Success:", "- Outcome:"))


def test_committed_rules_file_has_every_rule_the_page_reads():
    rules = app.RULES_FILE.read_text()
    assert app.methodology(rules)
    assert "Success = " in app.notes(app.load_rows(), rules)[2]


@pytest.mark.parametrize("results", ["committed", "empty"])
def test_the_app_builds(results, tmp_path):
    kwargs = {} if results == "committed" else {"results_dir": tmp_path}
    assert app.build_demo(**kwargs)


def test_space_pins_match_the_locked_versions():
    readme = Path("README.md").read_text()
    assert readme.startswith("---\n")
    assert f"sdk_version: {version('gradio')}\n" in readme
    assert f"plotly=={version('plotly')}" in Path("requirements.txt").read_text()


def test_readme_results_are_current():
    readme = Path("README.md").read_text()
    assert update_readme.render(app.load_rows(), app.RULES_FILE.read_text()) in readme


def test_readme_results_follow_the_rows():
    rows = [
        _row(technique="native", p50_ms=300.0, success=70.0),
        _row(technique="compile_autotune", p50_ms=100.0, compile_s=50.0, success=71.0),
        _row(technique="fp32", p50_ms=200.0, success=72.0),
    ]
    text = update_readme.render(rows, app.RULES_FILE.read_text())
    assert "π0.5 70–72%" in text
    assert "π0.5 300 → 100 ms (3.0× speed, 50 s one-time compile, success 70% → 71%)" in text
    assert "no faster than native" not in text  # fp32 beat native here


def test_readme_says_no_faster_only_when_it_is_true():
    rows = [_row(technique="native", p50_ms=100.0), _row(technique="fp16", p50_ms=200.0)]
    assert "no faster than native" in update_readme.render(rows, app.RULES_FILE.read_text())
