#!/usr/bin/env python3
"""Plot averaged ROI trace heatmaps from trace-only CSV files."""

from __future__ import annotations

import argparse
import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Average trace-only CSV files into 10 s trial windows and plot "
            "ROI heatmaps with a -3 s to +7 s stimulus-aligned x axis."
        )
    )
    parser.add_argument(
        "input_dir",
        type=pathlib.Path,
        help="Directory containing *_trace_only.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help="Output directory. Defaults to <input_dir>/Trace_Heatmap_ByFOV.",
    )
    parser.add_argument(
        "--drop-seconds",
        type=float,
        default=2.0,
        help="Seconds to remove from the beginning of each trace. Default: 2.",
    )
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=10.0,
        help="Length of each segment to average. Default: 10.",
    )
    parser.add_argument(
        "--pre-stimulus",
        type=float,
        default=3.0,
        help="Seconds before stimulus shown on the x axis. Default: 3.",
    )
    return parser.parse_args()


def roi_number(trace_column: str) -> int | None:
    match = re.search(r"ROI(\d+)$", trace_column)
    return int(match.group(1)) if match else None


def trace_columns(df: pd.DataFrame) -> list[str]:
    columns = [col for col in df.columns if col.startswith("Trace_ROI")]
    return sorted(columns, key=lambda col: roi_number(col) or 0)


def estimate_frame_interval(time_values: pd.Series) -> float:
    numeric_time = pd.to_numeric(time_values, errors="coerce")
    diffs = numeric_time.diff().iloc[1:]
    positive_diffs = diffs[diffs > 0]
    if positive_diffs.empty:
        raise ValueError("Could not estimate frame interval from Time column.")
    return float(positive_diffs.median())


def averaged_trace_matrix(
    df: pd.DataFrame,
    drop_seconds: float,
    segment_seconds: float,
) -> tuple[np.ndarray, list[int], int, float]:
    columns = trace_columns(df)
    if not columns:
        raise ValueError("No Trace_ROI columns found.")
    if "Time" not in df.columns:
        raise ValueError("Time column not found.")

    frame_interval = estimate_frame_interval(df["Time"])
    start_time = float(pd.to_numeric(df["Time"], errors="coerce").iloc[0]) + drop_seconds
    retained = df[pd.to_numeric(df["Time"], errors="coerce") >= start_time].copy()

    frames_per_segment = int(round(segment_seconds / frame_interval))
    if frames_per_segment <= 0:
        raise ValueError("Segment length produced zero frames.")

    trace_values = retained[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    segment_count = trace_values.shape[0] // frames_per_segment
    if segment_count == 0:
        raise ValueError("No complete 10 second segments after dropping the initial frames.")

    trace_values = trace_values[: segment_count * frames_per_segment]
    segments = trace_values.reshape(segment_count, frames_per_segment, len(columns))
    averaged = np.nanmean(segments, axis=0).T
    rois = [roi_number(col) or index + 1 for index, col in enumerate(columns)]
    return averaged, rois, segment_count, frame_interval


def plot_heatmap(
    matrix: np.ndarray,
    rois: list[int],
    dataset_name: str,
    segment_count: int,
    frame_interval: float,
    pre_stimulus: float,
    segment_seconds: float,
    output_path: pathlib.Path,
) -> None:
    fig_height = max(7, 0.19 * len(rois))
    fig, ax = plt.subplots(figsize=(13, fig_height))

    vmax = float(np.nanpercentile(np.abs(matrix), 99))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    image = ax.imshow(
        matrix,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        extent=(-pre_stimulus, segment_seconds - pre_stimulus, len(rois) + 0.5, 0.5),
    )
    ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
    ax.set_title(f"{dataset_name}\nMean of {segment_count} segments, dt={frame_interval:.3f}s")
    ax.set_xlabel("Time from stimulus (s)")
    ax.set_ylabel("ROI")
    ax.set_xticks(np.arange(-pre_stimulus, segment_seconds - pre_stimulus + 0.1, 1.0))
    ax.set_yticks(np.arange(1, len(rois) + 1))
    ax.set_yticklabels([str(roi) for roi in rois])
    ax.tick_params(axis="y", labelsize=7)

    colorbar = fig.colorbar(image, ax=ax, pad=0.015)
    colorbar.set_label("Mean trace value")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def process_file(
    csv_path: pathlib.Path,
    output_dir: pathlib.Path,
    drop_seconds: float,
    segment_seconds: float,
    pre_stimulus: float,
) -> dict[str, object]:
    df = pd.read_csv(csv_path)
    matrix, rois, segment_count, frame_interval = averaged_trace_matrix(
        df,
        drop_seconds=drop_seconds,
        segment_seconds=segment_seconds,
    )

    stem = csv_path.stem.removesuffix("_trace_only")
    matrix_path = output_dir / f"{stem}_mean_segments.csv"
    heatmap_path = output_dir / f"{stem}_trace_heatmap.png"

    relative_time = np.linspace(
        -pre_stimulus,
        segment_seconds - pre_stimulus,
        matrix.shape[1],
        endpoint=False,
    )
    matrix_df = pd.DataFrame(matrix, index=[f"ROI{roi}" for roi in rois], columns=relative_time)
    matrix_df.index.name = "ROI"
    matrix_df.to_csv(matrix_path)

    plot_heatmap(
        matrix,
        rois,
        stem,
        segment_count,
        frame_interval,
        pre_stimulus,
        segment_seconds,
        heatmap_path,
    )

    return {
        "Dataset": stem,
        "TraceFile": str(csv_path),
        "Heatmap": str(heatmap_path),
        "MeanSegmentsCSV": str(matrix_path),
        "SegmentCount": segment_count,
        "FramesPerSegment": matrix.shape[1],
        "FrameIntervalSeconds": frame_interval,
    }


def main() -> None:
    args = parse_arguments()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else input_dir / "Trace_Heatmap_ByFOV"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.glob("*trace_only.csv"))
    if not csv_files:
        raise ValueError(f"No *trace_only.csv files found in {input_dir}")

    summaries = []
    for csv_path in csv_files:
        print(f">> Processing {csv_path.name}")
        summaries.append(
            process_file(
                csv_path,
                output_dir,
                drop_seconds=args.drop_seconds,
                segment_seconds=args.segment_seconds,
                pre_stimulus=args.pre_stimulus,
            )
        )

    summary_path = output_dir / "trace_heatmap_summary.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    print(f"Wrote {len(summaries)} heatmaps to: {output_dir}")
    print(f"Wrote summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
