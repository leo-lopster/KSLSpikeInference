#!/usr/bin/env python3
"""Plot the first N seconds of spike CSVs in a folder and save PNGs.

Usage:
    python3 scripts/plot_model_outputs.py /path/to/Model_comparison_output/400Hz450um1s_FOOT --seconds 25
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_heatmap(
    df: pd.DataFrame,
    mask: pd.Series,
    time: pd.Series,
    spike_cols: list[str],
    out_png: Path,
    title: str,
) -> None:
    # Build matrix (n_rois x n_timepoints)
    roi_nums: list[int | None] = []
    for c in spike_cols:
        m = __import__('re').search(r"ROI(\d+)", c)
        roi_nums.append(int(m.group(1)) if m else None)

    data = df.loc[mask, spike_cols].values.T
    # If ROI numbers were found, sort rows by ROI number
    if any(x is not None for x in roi_nums):
        order = [i for i, _ in sorted(enumerate(roi_nums), key=lambda x: (x[1] if x[1] is not None else float('inf')))]
        data = data[order, :]
        y_labels = [f"{roi_nums[i]}" if roi_nums[i] is not None else f"{i + 1}" for i in order]
    else:
        y_labels = list(map(str, range(1, data.shape[0] + 1)))

    fig, ax = plt.subplots(figsize=(12, max(3, 0.2 * data.shape[0] + 2)))
    t0 = float(time.iloc[0]) if len(time) else 0.0
    t1 = float(time.iloc[-1]) if len(time) else 0.0
    im = ax.imshow(
        data,
        aspect='auto',
        interpolation='nearest',
        cmap='viridis',
        origin='lower',
        extent=(t0, t1, 0.5, data.shape[0] + 0.5),
    )
    ax.set_ylabel('ROI')
    ax.set_yticks(range(1, data.shape[0] + 1))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('Time (s)')
    ax.set_ylim(0.5, data.shape[0] + 0.5)
    fig.colorbar(im, ax=ax, label='Spike probability')
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_traces(
    df: pd.DataFrame,
    mask: pd.Series,
    time: pd.Series,
    spike_cols: list[str],
    out_png: Path,
    title: str,
) -> None:
    n = len(spike_cols)
    # determine a near-square grid: cols ~ ceil(sqrt(n)), rows = ceil(n / cols)
    if n <= 1:
        rows, cols = 1, 1
    else:
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))

    fig_width = max(6, 3 * cols)
    fig_height = max(3, 1.5 * rows)
    fig, axes = plt.subplots(rows, cols, sharex=True, figsize=(fig_width, fig_height))

    # Normalize axes to a flat list for easy iteration
    if isinstance(axes, np.ndarray):
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]

    for ax, col in zip(axes_flat, spike_cols):
        ax.plot(time, df.loc[mask, col].values, linewidth=1)
        ax.set_ylabel(col)
        ax.grid(True, linestyle='--', alpha=0.35)

    # Hide any unused subplots
    for ax in axes_flat[len(spike_cols):]:
        ax.axis('off')

    # Label x-axis on bottom row
    start_bottom = (rows - 1) * cols
    for ax in axes_flat[start_bottom:start_bottom + cols]:
        ax.set_xlabel('Time (s)')

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_graph(
    df: pd.DataFrame,
    mask: pd.Series,
    time: pd.Series,
    spike_cols: list[str],
    out_png: Path,
    title: str,
    heatmap: bool = False,
) -> None:
    if heatmap:
        plot_heatmap(df, mask, time, spike_cols, out_png, title)
    else:
        plot_traces(df, mask, time, spike_cols, out_png, title)


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
        raise ValueError("No complete trace segments after dropping the initial frames.")

    trace_values = trace_values[: segment_count * frames_per_segment]
    segments = trace_values.reshape(segment_count, frames_per_segment, len(columns))
    averaged = np.nanmean(segments, axis=0).T
    rois = [roi_number(col) or index + 1 for index, col in enumerate(columns)]
    return averaged, rois, segment_count, frame_interval


def plot_trace_only_heatmap(
    matrix: np.ndarray,
    rois: list[int],
    dataset_name: str,
    segment_count: int,
    frame_interval: float,
    pre_stimulus: float,
    segment_seconds: float,
    output_path: Path,
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


def process_trace_only_file(
    csv_path: Path,
    output_dir: Path,
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

    plot_trace_only_heatmap(
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
        "TraceFile": csv_path,
        "Heatmap": heatmap_path,
        "MeanSegmentsCSV": matrix_path,
        "SegmentCount": segment_count,
        "FramesPerSegment": matrix.shape[1],
        "FrameIntervalSeconds": frame_interval,
    }


def plot_trace_only_heatmaps_in_folder(
    folder: Path,
    output_dir: Path,
    drop_seconds: float = 2.0,
    segment_seconds: float = 10.0,
    pre_stimulus: float = 3.0,
) -> list[Path]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    out_images: list[Path] = []
    csv_files = sorted(folder.glob("*trace_only.csv"))
    if not csv_files:
        raise SystemExit(f"No *trace_only.csv files found in {folder}")

    for csv_path in csv_files:
        try:
            summary = process_trace_only_file(
                csv_path,
                output_dir,
                drop_seconds=drop_seconds,
                segment_seconds=segment_seconds,
                pre_stimulus=pre_stimulus,
            )
        except Exception as e:
            print(f"Skipping {csv_path.name}: {e}")
            continue

        # Ensure we append a Path object (summary may contain plain objects/strings)
        out_images.append(Path(str(summary["Heatmap"])))
        summaries.append({
            "Dataset": summary["Dataset"],
            "TraceFile": str(summary["TraceFile"]),
            "Heatmap": str(summary["Heatmap"]),
            "MeanSegmentsCSV": str(summary["MeanSegmentsCSV"]),
            "SegmentCount": summary["SegmentCount"],
            "FramesPerSegment": summary["FramesPerSegment"],
            "FrameIntervalSeconds": summary["FrameIntervalSeconds"],
        })
        print(f"Saved {summary['Heatmap']}")

    summary_path = output_dir / "trace_heatmap_summary.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    print(f"Wrote summary CSV: {summary_path}")
    return out_images


def plot_graphs_in_folder(
    folder: Path,
    output_dir: Path,
    heatmap: bool = False,
    seconds: float = 25.0,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_images: list[Path] = []
    csv_files = sorted(folder.glob("*_spikes.csv"))
    if not csv_files:
        raise SystemExit(f"No *_spikes.csv files found in {folder}")

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Skipping {csv_path.name}: failed to read CSV ({e})")
            continue

        if 'Time' not in df.columns:
            print(f"Skipping {csv_path.name}: no 'Time' column")
            continue

        mask = df['Time'] <= seconds
        time = df.loc[mask, 'Time']

        # detect spike columns (anything starting with Spike_)
        spike_cols = [c for c in df.columns if c.startswith('Spike_')]
        if not spike_cols:
            # fallback: use all numeric columns except Frame/Time
            spike_cols = [c for c in df.select_dtypes(include='number').columns if c not in ('Frame', 'Time')]
        if not spike_cols:
            print(f"Skipping {csv_path.name}: no spike or numeric columns to plot")
            continue

        title = csv_path.name
        suffix = 'heatmap' if heatmap else 'first25s'
        out_png = output_dir / f"{csv_path.stem}_{suffix}.png"
        plot_graph(df, mask, time, spike_cols, out_png, title, heatmap=heatmap)

        out_images.append(out_png)
        print(f"Saved {out_png}")

    return out_images


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot spike CSVs or trace-only heatmaps from a folder.'
    )
    parser.add_argument('folder', type=Path, nargs='?', default=Path('.'), help='Folder containing CSV files')
    parser.add_argument('--seconds', type=float, default=25.0, help='Number of seconds to plot for spike CSV files')
    parser.add_argument('--heatmap', action='store_true', help='Plot heatmap (time x ROI) instead of individual traces for spike CSV files')
    parser.add_argument('--trace-only', action='store_true', help='Process *_trace_only.csv files into averaged ROI trace heatmaps')
    parser.add_argument('--drop-seconds', type=float, default=2.0, help='Seconds to drop at the beginning of each trace segment for trace-only heatmaps')
    parser.add_argument('--segment-seconds', type=float, default=10.0, help='Segment length in seconds for trace-only heatmaps')
    parser.add_argument('--pre-stimulus', type=float, default=3.0, help='Seconds before stimulus on the x axis for trace-only heatmaps')
    parser.add_argument('--output-dir', type=Path, default=None, help='Directory to save generated plot PNGs')
    args = parser.parse_args()

    folder = args.folder
    if not folder.exists():
        raise SystemExit(f"Folder does not exist: {folder}")

    if args.trace_only:
        output_dir = args.output_dir or folder / 'Trace_Heatmap_ByFOV'
        plot_trace_only_heatmaps_in_folder(
            folder,
            output_dir,
            drop_seconds=args.drop_seconds,
            segment_seconds=args.segment_seconds,
            pre_stimulus=args.pre_stimulus,
        )
    else:
        output_dir = args.output_dir or Path('./first_n_second_plots')
        plot_graphs_in_folder(folder, output_dir, heatmap=args.heatmap, seconds=args.seconds)
