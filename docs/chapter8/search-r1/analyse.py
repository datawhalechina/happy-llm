"""绘制 Search-R1 评测的 Macro EM 与格式正确率。

从 Happy-LLM 仓库根目录运行：

    python docs/chapter8/search-r1/analyse.py
    python docs/chapter8/search-r1/analyse.py --preset deepseek

每项指标都读取相应评测 JSONL 中最后一条 ``type=summary`` 记录。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULT_DIR = SCRIPT_DIR / "eval_result"
DEFAULT_OUTPUT = DEFAULT_RESULT_DIR / "checkpoint_em_format.png"
DEFAULT_DEEPSEEK_OUTPUT = SCRIPT_DIR / "images" / "deepseek_checkpoint_em_format.png"

ZHIHU_CHECKPOINTS = (
    ("Base", "eval_results.jsonl"),
    ("Step 20", "eval_results_rl_step_20.jsonl"),
    ("Step 50", "eval_results_rl_step_50.jsonl"),
    ("Step 100", "eval_results_rl_step_100.jsonl"),
    ("Step 150", "eval_results_rl_step_150.jsonl"),
    ("Step 200", "eval_results_rl_step_200.jsonl"),
)
DEEPSEEK_CHECKPOINTS = (
    ("Base", "eval_results_base_deepseek_search.jsonl"),
    ("Step 20", "eval_results_rl_step_20_deepseek_search.jsonl"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=("zhihu", "deepseek"),
        default="zhihu",
        help="Evaluation set to plot; the default preserves the original Zhihu figure",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
        help="Directory containing eval_results*.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path; each preset has its own default",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Raster export resolution",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open an interactive preview after saving",
    )
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, Any]:
    """Return the last summary record in an evaluation JSONL file."""
    summary: dict[str, Any] | None = None
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from error
            if record.get("type") == "summary":
                summary = record

    if summary is None:
        raise ValueError(f"No type=summary record found in {path}")
    return summary


def load_metrics(
    result_dir: Path,
    checkpoints: tuple[tuple[str, str], ...],
    *,
    expected_backend: str | None = None,
) -> tuple[list[str], list[float], list[float]]:
    labels: list[str] = []
    macro_em: list[float] = []
    format_rate: list[float] = []
    evaluation_sizes: set[int] = set()
    search_configs: set[tuple[str, int, float, str]] = set()

    for label, filename in checkpoints:
        path = result_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing evaluation result: {path}")

        summary = load_summary(path)
        if expected_backend is not None:
            backend = summary.get("search_backend")
            if backend != expected_backend:
                raise ValueError(
                    f"Expected search_backend={expected_backend!r} in {path}, got {backend!r}"
                )
            try:
                evaluation_sizes.add(int(summary["evaluated_examples"]))
                search_configs.add(
                    (
                        str(backend),
                        int(summary["search_concurrency"]),
                        float(summary["search_timeout"]),
                        str(summary["search_model"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Search provenance is incomplete in {path}") from error

        metrics = summary.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError(f"Summary metrics are missing in {path}")

        try:
            em_value = float(metrics["em/macro"])
            format_value = float(metrics["format/rate"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Expected numeric em/macro and format/rate metrics in {path}"
            ) from error

        labels.append(label)
        macro_em.append(em_value)
        format_rate.append(format_value)

    if expected_backend is not None:
        if evaluation_sizes != {70}:
            raise ValueError(
                f"DeepSeek comparison requires the same fixed 70-question set, got {evaluation_sizes}"
            )
        if len(search_configs) != 1:
            raise ValueError(
                f"DeepSeek comparison requires identical search settings, got {search_configs}"
            )

    return labels, macro_em, format_rate


def configure_style() -> None:
    """Apply a restrained paper-style Matplotlib theme."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.edgecolor": "#202020",
            "axes.linewidth": 1.0,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def add_value_labels(
    axis: plt.Axes,
    x_positions: list[int],
    values: list[float],
    *,
    offset: float,
) -> None:
    for x_position, value in zip(x_positions, values, strict=True):
        axis.text(
            x_position,
            value + offset,
            f"{value:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#202020",
        )


def make_figure(
    labels: list[str],
    macro_em: list[float],
    format_rate: list[float],
) -> plt.Figure:
    configure_style()

    x_positions = list(range(len(labels)))
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13.0, 5.4),
        sharey=True,
        gridspec_kw={"wspace": 0.08},
    )
    em_axis, format_axis = axes

    # Panel (a): absolute checkpoint comparison.
    em_colors = ["#D94A4A"] * len(labels)
    em_colors[2] = "#F28E2B"
    bars = em_axis.bar(
        x_positions,
        macro_em,
        width=0.66,
        color=em_colors,
        edgecolor="#202020",
        linewidth=1.0,
        alpha=0.96,
        zorder=3,
    )
    bars[2].set_hatch("///")
    add_value_labels(em_axis, x_positions, macro_em, offset=0.025)
    em_axis.set_title("(a) Macro Exact Match", pad=14)
    em_axis.set_ylabel("Score")

    # Panel (b): ordered checkpoint trend.
    format_axis.plot(
        x_positions,
        format_rate,
        color="#3388B8",
        marker="s",
        markersize=7,
        markerfacecolor="#57B8D2",
        markeredgecolor="#202020",
        markeredgewidth=0.9,
        linewidth=2.1,
        zorder=4,
    )
    format_axis.scatter(
        [x_positions[2]],
        [format_rate[2]],
        s=90,
        facecolor="#F28E2B",
        edgecolor="#202020",
        linewidth=1.0,
        zorder=5,
    )
    add_value_labels(format_axis, x_positions, format_rate, offset=0.025)
    format_axis.set_title("(b) Valid Answer Format Rate", pad=14)

    for axis in axes:
        axis.set_xticks(x_positions, labels)
        axis.set_xlabel("Model / Checkpoint", labelpad=10)
        axis.set_ylim(0.0, 1.08)
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        axis.grid(
            axis="y",
            color="#C7C7C7",
            linestyle="-.",
            linewidth=0.8,
            alpha=0.75,
            zorder=0,
        )
        axis.tick_params(
            axis="both",
            which="major",
            direction="in",
            top=True,
            right=True,
            length=5,
            width=0.9,
        )
        axis.margins(x=0.06)

    figure.suptitle(
        "Search-R1 Checkpoint Evaluation",
        fontsize=17,
        y=0.995,
    )
    figure.text(
        0.5,
        0.935,
        "Fixed 70-question evaluation set · values read from persisted JSONL summaries",
        ha="center",
        va="top",
        fontsize=10,
        color="#555555",
    )
    figure.text(
        0.5,
        0.015,
        "Orange indicates the best observed Macro EM checkpoint (Step 50).",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.18, top=0.84)
    return figure


def make_deepseek_figure(
    labels: list[str],
    macro_em: list[float],
    format_rate: list[float],
) -> plt.Figure:
    """Compare Base and Step 20 under the same DeepSeek Search environment."""
    configure_style()

    x_positions = list(range(len(labels)))
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13.0, 5.4),
        sharey=False,
        gridspec_kw={"wspace": 0.08},
    )
    panels = (
        (axes[0], macro_em, "(a) Macro Exact Match", (0.42, 0.52)),
        (axes[1], format_rate, "(b) Valid Answer Format Rate", (0.86, 1.00)),
    )

    for axis, values, title, y_limits in panels:
        visible_heights = [value - y_limits[0] for value in values]
        bars = axis.bar(
            x_positions,
            visible_heights,
            bottom=y_limits[0],
            width=0.48,
            color=["#A9AFB8", "#3B82F6"],
            edgecolor="#202020",
            linewidth=1.0,
            alpha=0.96,
            zorder=3,
        )
        bars[1].set_hatch("///")
        label_offset = (y_limits[1] - y_limits[0]) * 0.045
        add_value_labels(axis, x_positions, values, offset=label_offset)
        gain = values[1] - values[0]
        axis.text(
            0.5,
            0.16,
            f"{gain * 100:+.1f} pp",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="#2563EB",
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": "white",
                "edgecolor": "#93C5FD",
                "linewidth": 0.8,
                "alpha": 0.96,
            },
            zorder=5,
        )
        axis.set_title(title, pad=14)
        axis.set_xticks(x_positions, labels)
        axis.set_xlabel("Model / Checkpoint", labelpad=10)
        axis.set_ylim(*y_limits)
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        axis.grid(
            axis="y",
            color="#C7C7C7",
            linestyle="-.",
            linewidth=0.8,
            alpha=0.75,
            zorder=0,
        )
        axis.tick_params(
            axis="both",
            which="major",
            direction="in",
            top=True,
            right=True,
            length=5,
            width=0.9,
        )
        axis.margins(x=0.28)
        axis.text(
            0.02,
            0.035,
            "focused y-axis",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            color="#777777",
            style="italic",
        )

    axes[0].set_ylabel("Score")
    em_gain = macro_em[1] - macro_em[0]
    format_gain = format_rate[1] - format_rate[0]

    figure.suptitle(
        "Search-R1 · DeepSeek Search Evaluation",
        fontsize=17,
        y=0.995,
    )
    figure.text(
        0.5,
        0.935,
        (
            "Fixed 70-question evaluation set · same DeepSeek Search backend · "
            "independent focused y-axes"
        ),
        ha="center",
        va="top",
        fontsize=10,
        color="#555555",
    )
    figure.text(
        0.5,
        0.015,
        (
            f"Step 20 change: Macro EM {em_gain * 100:+.1f} pp · "
            f"valid answer format {format_gain * 100:+.1f} pp · "
            "20 training steps · "
            "search success 100% in both evaluations."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.18, top=0.84)
    return figure


def main() -> None:
    args = parse_args()
    if args.preset == "deepseek":
        labels, macro_em, format_rate = load_metrics(
            args.result_dir,
            DEEPSEEK_CHECKPOINTS,
            expected_backend="deepseek",
        )
        figure = make_deepseek_figure(labels, macro_em, format_rate)
        output = args.output or DEFAULT_DEEPSEEK_OUTPUT
    else:
        labels, macro_em, format_rate = load_metrics(
            args.result_dir,
            ZHIHU_CHECKPOINTS,
        )
        figure = make_figure(labels, macro_em, format_rate)
        output = args.output or DEFAULT_OUTPUT

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved figure: {output}")

    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
