"""Plot Average@n / Pass@n and format rate across ReTool checkpoints.

Run from the repository root:

    uv run python 05-retool/analysis.py

The figure is written to ``05-retool/images/checkpoint_avg_pass_format.png``
by default. Each metric is read from the final ``type=summary`` record of the
corresponding evaluation JSONL file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULT_DIR = SCRIPT_DIR / "eval-results"
DEFAULT_OUTPUT = SCRIPT_DIR / "images" / "checkpoint_avg_pass_format.png"

CHECKPOINTS = (
    ("Base", "aime25-retool-base.jsonl"),
    ("Step 20", "aime25-retool-step20.jsonl"),
    ("Step 50", "aime25-retool-step50.jsonl"),
    ("Step 100", "aime25-retool-step100.jsonl"),
    ("Step 150", "aime25-retool-step150.jsonl"),
    ("Step 200", "aime25-retool-step200.jsonl"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
        help="Directory containing aime25-retool-*.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output image path; the extension selects the export format",
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
) -> tuple[list[str], int, list[float], list[float], list[float]]:
    labels: list[str] = []
    val_n: int | None = None
    average_at_n: list[float] = []
    pass_at_n: list[float] = []
    format_rate: list[float] = []

    for label, filename in CHECKPOINTS:
        path = result_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing evaluation result: {path}")

        summary = load_summary(path)
        try:
            n_value = int(summary["val_n"])
            average_value = float(summary["average_at_n"])
            pass_value = float(summary["pass_at_n"])
            format_value = float(summary["format_rate"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "Expected numeric val_n / average_at_n / pass_at_n / format_rate "
                f"metrics in {path}"
            ) from error

        if val_n is None:
            val_n = n_value
        elif val_n != n_value:
            raise ValueError(
                f"Inconsistent val_n in {path}: {n_value} != {val_n}"
            )

        labels.append(label)
        average_at_n.append(average_value)
        pass_at_n.append(pass_value)
        format_rate.append(format_value)

    assert val_n is not None
    return labels, val_n, average_at_n, pass_at_n, format_rate


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
    x_positions: list[float],
    values: list[float],
    *,
    offset: float,
    fontsize: int = 9,
) -> None:
    for x_position, value in zip(x_positions, values, strict=True):
        axis.text(
            x_position,
            value + offset,
            f"{value:.1%}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color="#202020",
        )


def make_figure(
    labels: list[str],
    val_n: int,
    average_at_n: list[float],
    pass_at_n: list[float],
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
    accuracy_axis, format_axis = axes

    # Panel (a): grouped bars for Average@n and Pass@n.
    bar_width = 0.36
    average_positions = [x - bar_width / 2 for x in x_positions]
    pass_positions = [x + bar_width / 2 for x in x_positions]
    accuracy_axis.bar(
        average_positions,
        average_at_n,
        width=bar_width,
        color="#D94A4A",
        edgecolor="#202020",
        linewidth=1.0,
        alpha=0.96,
        zorder=3,
        label=f"Average@{val_n}",
    )
    accuracy_axis.bar(
        pass_positions,
        pass_at_n,
        width=bar_width,
        color="#3388B8",
        edgecolor="#202020",
        linewidth=1.0,
        alpha=0.96,
        zorder=3,
        label=f"Pass@{val_n}",
    )
    add_value_labels(
        accuracy_axis, average_positions, average_at_n, offset=0.02, fontsize=8
    )
    add_value_labels(
        accuracy_axis, pass_positions, pass_at_n, offset=0.02, fontsize=8
    )
    accuracy_axis.set_title(f"(a) Average@{val_n} & Pass@{val_n}", pad=14)
    accuracy_axis.set_ylabel("Score")
    accuracy_axis.legend(
        loc="upper left",
        frameon=False,
        fontsize=10,
        handlelength=1.4,
    )

    # Panel (b): ordered checkpoint trend of the format rate.
    format_axis.plot(
        x_positions,
        format_rate,
        color="#F28E2B",
        marker="s",
        markersize=7,
        markerfacecolor="#F2B36B",
        markeredgecolor="#202020",
        markeredgewidth=0.9,
        linewidth=2.1,
        zorder=4,
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
        "ReTool Checkpoint Evaluation",
        fontsize=17,
        y=0.995,
    )
    figure.text(
        0.5,
        0.935,
        "AIME 2025 (30 problems) · val-n 12 · temperature 1.0",
        ha="center",
        va="top",
        fontsize=10,
        color="#555555",
    )
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.13, top=0.84)
    return figure


def main() -> None:
    args = parse_args()
    labels, val_n, average_at_n, pass_at_n, format_rate = load_metrics(
        args.result_dir
    )
    figure = make_figure(labels, val_n, average_at_n, pass_at_n, format_rate)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved figure: {args.output}")

    if args.show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
