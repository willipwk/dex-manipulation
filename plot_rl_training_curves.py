#!/usr/bin/env python3
"""Plot scalar RL training curves from a TensorBoard event file."""

from __future__ import annotations

import argparse
import math
import os
import re
import tempfile
from pathlib import Path


DEFAULT_EVENT_FILE = (
    Path(__file__).resolve().parent
    / "logs"
    / "leap_hand_tracking_rl"
    / "events.out.tfevents.1776848104.cs-3dlg-20.3645764.0"
)


def load_scalar_series(event_file: Path) -> dict[str, list[tuple[int, float, float | None]]]:
    from tensorboard.backend.event_processing import event_accumulator

    accumulator = event_accumulator.EventAccumulator(str(event_file))
    accumulator.Reload()

    series: dict[str, list[tuple[int, float, float | None]]] = {}
    for tag in accumulator.Tags().get("scalars", []):
        series[tag] = [
            (event.step, event.value, event.wall_time)
            for event in accumulator.Scalars(tag)
        ]
    return series


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) < window:
        return values[:]

    smoothed: list[float] = []
    queue: list[float] = []
    total = 0.0
    for value in values:
        queue.append(value)
        total += value
        if len(queue) > window:
            total -= queue.pop(0)
        smoothed.append(total / len(queue))
    return smoothed


def safe_filename(tag: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag).strip("_")
    return name or "scalar"


def plot_one(ax, tag: str, points: list[tuple[int, float, float | None]], smooth_window: int):
    steps = [point[0] for point in points]
    values = [point[1] for point in points]

    ax.plot(steps, values, linewidth=0.8, alpha=0.35, label="raw")
    if smooth_window > 1 and len(values) >= smooth_window:
        ax.plot(
            steps,
            moving_average(values, smooth_window),
            linewidth=1.8,
            label=f"moving avg ({smooth_window})",
        )
        ax.legend(frameon=False, fontsize=8)

    ax.set_title(tag)
    ax.set_xlabel("episode")
    ax.grid(True, linewidth=0.4, alpha=0.35)


def save_overview(
    plt,
    series: dict[str, list[tuple[int, float, float | None]]],
    output_dir: Path,
    file_format: str,
    dpi: int,
    smooth_window: int,
):
    overview_tags = [
        "Train/mean_reward",
        "Train/mean_episode_length",
        "Loss/value_function",
        "Loss/surrogate",
    ]
    overview_tags = [tag for tag in overview_tags if tag in series]
    if not overview_tags:
        return

    rows = math.ceil(len(overview_tags) / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(12, 4.2 * rows), constrained_layout=True)
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for ax, tag in zip(axes, overview_tags):
        plot_one(ax, tag, series[tag], smooth_window)
    for ax in axes[len(overview_tags) :]:
        ax.axis("off")

    fig.suptitle("Leap Hand Tracking RL Training Overview")
    fig.savefig(output_dir / f"rl_training_overview.{file_format}", dpi=dpi)
    plt.close(fig)


def save_individual_figures(
    plt,
    series: dict[str, list[tuple[int, float, float | None]]],
    output_dir: Path,
    file_format: str,
    dpi: int,
    smooth_window: int,
):
    for tag in sorted(series):
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        plot_one(ax, tag, series[tag], smooth_window)
        fig.savefig(output_dir / f"{safe_filename(tag)}.{file_format}", dpi=dpi)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--event-file",
        type=Path,
        default=DEFAULT_EVENT_FILE,
        help=f"TensorBoard event file to read. Default: {DEFAULT_EVENT_FILE}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for figure files. Default: <event-file-dir>/rl_training_figures",
    )
    parser.add_argument(
        "--format",
        default="png",
        choices=["png", "pdf", "svg"],
        help="Figure file format. Default: png",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Raster output DPI. Default: 200")
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=25,
        help="Moving-average window for the smoothed line. Use 1 to disable. Default: 25",
    )
    parser.add_argument(
        "--no-overview",
        action="store_true",
        help="Only save one figure per scalar tag.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    event_file = args.event_file.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else event_file.parent / "rl_training_figures"
    )

    if not event_file.exists():
        raise FileNotFoundError(event_file)

    series = load_scalar_series(event_file)
    if not series:
        raise RuntimeError(f"No scalar data found in {event_file}")

    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )

    if not args.no_overview:
        save_overview(plt, series, output_dir, args.format, args.dpi, args.smooth_window)
    save_individual_figures(plt, series, output_dir, args.format, args.dpi, args.smooth_window)

    print(f"Wrote {len(series) + (0 if args.no_overview else 1)} figure files to {output_dir}")
    for tag in sorted(series):
        points = series[tag]
        print(
            f"{tag}: {len(points)} points, "
            f"step {points[0][0]}..{points[-1][0]}, final {points[-1][1]:.6g}"
        )


if __name__ == "__main__":
    main()
