"""onl-bench leaderboard: a Gradio Space. Every number on the page comes from results/*.json."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import gradio as gr
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
RULES_FILE = ROOT / "CLAUDE.md"
REPO_URL = "https://github.com/abdul3asith/onl-leaderboard"
CONTACT_URL = f"{REPO_URL}/issues"

MISSING = "–"
MODEL_NAMES = {"pi05": "π0.5", "smolvla": "SmolVLA"}
# Categorical slots, assigned to models in sorted order and never cycled.
MODEL_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
# Points closer than this on the log axis (in decades) would put a label on their neighbor.
LABEL_CROWDED = 0.06
# Log axis ticks, labelled in full: 1, 2, 5, 10, 20, 50, ...
X_TICKS = [m * 10**e for e in range(-1, 5) for m in (1, 2, 5)]
# The CLAUDE.md rules that describe the measurement; the rest are for people changing the repo.
METHODOLOGY_RULES = ("Every number", "One result row", "Speed:", "One GPU", "Success:")


def load_rows(results_dir: Path = RESULTS_DIR) -> list[dict]:
    """Every result row, best success first, then fastest. Speed-only rows sort last."""
    rows = [json.loads(p.read_text()) for p in sorted(Path(results_dir).glob("*.json"))]
    return sorted(
        rows,
        key=lambda r: (
            r.get("success") is None,
            -(r.get("success") or 0),
            -(actions_per_second(r) or 0),
        ),
    )


def actions_per_second(row: dict) -> float | None:
    """Actions delivered per second: one plan yields n_action_steps actions in p50_ms."""
    if row.get("n_action_steps") is None or row.get("p50_ms") is None:
        return None
    return row["n_action_steps"] * 1000 / row["p50_ms"]


def model_name(model: str) -> str:
    return MODEL_NAMES.get(model, model)


def _short_gpu(gpu: str | None) -> str | None:
    return gpu.removeprefix("NVIDIA ") if gpu else None


def _success_text(row: dict) -> str:
    if row.get("success") is None:
        return MISSING
    if row.get("ci_low") is None or row.get("ci_high") is None:
        return f"{row['success']:.1f}%"
    return f"{row['success']:.1f}% ({row['ci_low']:.1f}–{row['ci_high']:.1f})"


def _column(compute, digits: int | None = None):
    """A column's sort value and displayed text; `digits` shows it as a fixed-point number."""

    def text(row: dict) -> str:
        v = compute(row)
        if v is None:
            return MISSING
        return str(v) if digits is None else f"{v:,.{digits}f}"

    return compute, text


def _field(key: str, digits: int | None = None):
    return _column(lambda r: r.get(key), digits)


# (header, Gradio datatype, sort value, displayed text). Sorting uses the value, so "100.0% (...)"
# ranks above "75.0% (...)" and a missing value stays missing rather than becoming a string.
COLUMNS = (
    ("model", "str", *_column(lambda r: model_name(r["model"]))),
    ("technique", "str", *_field("technique")),
    ("success % (95% CI)", "number", lambda r: r.get("success"), _success_text),
    ("p50 ms", "number", *_field("p50_ms", 1)),
    ("p99 ms", "number", *_field("p99_ms", 1)),
    ("actions per chunk", "number", *_field("n_action_steps", 0)),
    ("actions per second", "number", *_column(actions_per_second, 1)),
    ("peak MB", "number", *_field("peak_mb", 0)),
    ("compile_s", "number", *_field("compile_s", 1)),
    ("denoise steps", "number", *_field("denoise_steps", 0)),
    ("speed GPU", "str", *_column(lambda r: _short_gpu(r.get("gpu")))),
    ("success GPU", "str", *_column(lambda r: _short_gpu(r.get("success_gpu")))),
    ("date", "str", *_field("date")),
)


COLUMN_WIDTHS = [
    "90px", "180px", "205px", "80px", "80px", "90px", "90px", "85px", "105px", "85px", "75px",
    "85px", "115px",
]  # fmt: skip


def table(rows: list[dict]) -> dict:
    """The table as Gradio's data + display_value pair: numbers to sort, text to show."""
    return {
        "headers": [c[0] for c in COLUMNS],
        "data": [[value(r) for _, _, value, _ in COLUMNS] for r in rows],
        "metadata": {
            "display_value": [[text(r) for _, _, _, text in COLUMNS] for r in rows],
            "styling": None,
        },
    }


def _ci(row: dict, bound: str) -> float:
    """A CI bound, or the point itself on a row without one, so its error bar has no length."""
    return row["success"] if row.get(bound) is None else row[bound]


def _label_positions(speeds: list[float]) -> list[str]:
    """Label to the right, unless the next faster point sits where that label would go."""
    gaps = [math.log10(b / a) for a, b in zip(speeds, speeds[1:], strict=False)] + [math.inf]
    return ["middle left" if gap < LABEL_CROWDED else "middle right" for gap in gaps]


def chart(rows: list[dict]) -> go.Figure:
    """Success against actions per second, one color per model, one label per technique."""
    fig = go.Figure()
    plotted = [r for r in rows if r.get("success") is not None and actions_per_second(r)]
    for model, color in zip(sorted({r["model"] for r in plotted}), MODEL_COLORS, strict=False):
        points = sorted((r for r in plotted if r["model"] == model), key=actions_per_second)
        fig.add_trace(
            go.Scatter(
                x=[actions_per_second(r) for r in points],
                y=[r["success"] for r in points],
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": [_ci(r, "ci_high") - r["success"] for r in points],
                    "arrayminus": [r["success"] - _ci(r, "ci_low") for r in points],
                    "color": color,
                    "thickness": 1.5,
                    "width": 4,
                },
                mode="markers+text",
                name=model_name(model),
                text=[r["technique"] for r in points],
                textposition=_label_positions([actions_per_second(r) for r in points]),
                marker={"color": color, "size": 10, "line": {"color": "#fcfcfb", "width": 2}},
                customdata=[_success_text(r) for r in points],
                hovertemplate=(
                    "<b>%{text}</b><br>%{x:.1f} actions/s<br>%{customdata}"
                    f"<extra>{model_name(model)}</extra>"
                ),
            )
        )
    fig.update_layout(
        xaxis={"title": "Actions per second (log scale)", "type": "log", "tickvals": X_TICKS},
        yaxis={"title": "Success % (95% CI)"},
        legend={"orientation": "h", "y": 1.08, "x": 0},
        paper_bgcolor="#fcfcfb",
        plot_bgcolor="#fcfcfb",
        font={"color": "#52514e"},
        margin={"l": 60, "r": 20, "t": 40, "b": 50},
        height=460,
        hoverlabel={"namelength": -1},
    )
    fig.update_xaxes(gridcolor="#e6e5e0", zeroline=False)
    fig.update_yaxes(gridcolor="#e6e5e0", zeroline=False)
    if plotted:
        speeds = [math.log10(actions_per_second(r)) for r in plotted]
        lows = [_ci(r, "ci_low") for r in plotted]
        # Room for the outermost labels, which Plotly does not count when autoranging.
        fig.update_xaxes(range=[min(speeds) - 0.15, max(speeds) + 0.45])
        fig.update_yaxes(range=[math.floor(min(lows) / 5) * 5 - 5, 103])
    return fig


def _rule(rules_text: str, prefix: str) -> str:
    """One bullet of CLAUDE.md, joined back onto a single line."""
    for bullet in rules_text.split("\n- "):
        if bullet.startswith(prefix):
            return " ".join(bullet.split())
    raise ValueError(f"CLAUDE.md has no rule starting {prefix!r}")


def _slash(values) -> str:
    return "/".join(str(v) for v in sorted(set(values)))


def notes(rows: list[dict], rules_text: str) -> list[str]:
    """The notes under the table: GPUs and counts from the rows, the protocol from CLAUDE.md."""
    if not rows:
        return []
    speed = re.search(r"batch (\d+)", _rule(rules_text, "Speed:"))
    success = re.fullmatch(r"Success: ([^,]+), .*seed (\d+)\.", _rule(rules_text, "Success:"))
    suite, seed = success.groups()

    first, *others = [
        (model_name(m), _slash(r["n_action_steps"] for r in rows if r["model"] == m))
        for m in sorted({r["model"] for r in rows})
    ]
    chunk_note = ", ".join(
        [f"{first[0]} uses {first[1]} actions per plan", *(f"{m} {n}" for m, n in others)]
    )
    speed_gpus = {r["gpu"] for r in rows}
    result = [
        f"All speed measured on {_slash(speed_gpus)}, batch {speed.group(1)}",
        f"p50/p99 time one full plan (chunk); {chunk_note}",
    ]

    evaluated = [r for r in rows if r.get("episodes")]
    if evaluated:
        success_note = (
            f"Success = {suite}, {_slash(r['episodes'] for r in evaluated)} episodes, seed {seed}"
        )
        elsewhere = {r["success_gpu"] for r in evaluated} - speed_gpus
        if elsewhere:
            names = _slash(map(_short_gpu, elsewhere))
            success_note += f"; some success runs were on {names} (see column)"
        result.append(success_note)

    return [*result, "Compile time is one-time, first call only"]


def methodology(rules_text: str) -> str:
    rules = "\n".join(f"- {_rule(rules_text, p)}" for p in METHODOLOGY_RULES)
    return (
        "The rules every row is measured under, read from the repo's "
        f"[CLAUDE.md]({REPO_URL}/blob/main/CLAUDE.md):\n\n{rules}\n\n"
        "Actions per second is derived on this page: "
        "actions per chunk × 1000 ÷ p50 ms."
    )


ABOUT = (
    "onl-bench measures what vision-language-action policies cost to run and how well they "
    "still work: the speed of each policy under each precision and compile technique, next to "
    "its task success in simulation. Every number here comes from a script run, and the code "
    f"and raw result rows are [public]({REPO_URL}). "
    f"Questions or a policy you want benchmarked? [Get in touch]({CONTACT_URL})."
)


def build_demo(results_dir: Path = RESULTS_DIR, rules_file: Path = RULES_FILE) -> gr.Blocks:
    rows = load_rows(results_dir)
    rules_text = rules_file.read_text()
    with gr.Blocks(title="onl-bench leaderboard", fill_width=True) as demo:
        gr.Markdown("# onl-bench leaderboard")
        with gr.Tabs():
            with gr.Tab("Leaderboard"):
                gr.Dataframe(
                    value=table(rows),
                    datatype=[c[1] for c in COLUMNS],
                    interactive=False,
                    wrap=True,
                    column_widths=COLUMN_WIDTHS,
                    show_label=False,
                )
                gr.Markdown("\n".join(f"- {n}" for n in notes(rows, rules_text)))
                gr.Plot(chart(rows), show_label=False)
            with gr.Tab("Methodology"):
                gr.Markdown(methodology(rules_text))
            with gr.Tab("About"):
                gr.Markdown(ABOUT)
    return demo


if __name__ == "__main__":
    build_demo().launch()
