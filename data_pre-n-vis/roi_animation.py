"""Reusable MP4 writer for ROI-overlaid calcium recordings."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib import animation
from tqdm.auto import tqdm


FrameLoader = Callable[[int, pl.DataFrame], np.ndarray]


def save_roi_animation(
    *,
    index: pl.DataFrame,
    masks: Sequence[np.ndarray],
    load_frame: FrameLoader,
    output_path: str | Path,
    channel: int = 0,
    roi_source: str = "ROI",
    frame_step: int = 1,
    playback_fps: float | None = None,
    dpi: int = 120,
    overwrite: bool = False,
) -> Path:
    """Stream frames with fixed ROI contours to an H.264 MP4."""
    if index.height == 0:
        raise ValueError("index is empty")
    if frame_step < 1:
        raise ValueError("frame_step must be >= 1")
    if not animation.writers.is_available("ffmpeg"):
        raise RuntimeError("ffmpeg is required (macOS: brew install ffmpeg).")

    output_path = Path(output_path).with_suffix(".mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0 and not overwrite:
        return output_path

    masks = [np.asarray(mask, dtype=bool) for mask in masks]
    row_ids = np.arange(0, index.height, frame_step, dtype=int)
    timepoints = index["timepoint"].to_numpy()[row_ids]
    elapsed_s = index["elapsed_s"].to_numpy()[row_ids].astype(float)

    valid_dt = np.diff(index["elapsed_s"].to_numpy().astype(float))
    valid_dt = valid_dt[np.isfinite(valid_dt) & (valid_dt > 0)]
    native_fps = 1.0 / np.median(valid_dt) if valid_dt.size else 10.0
    playback_fps = native_fps / frame_step if playback_fps is None else float(playback_fps)
    if not np.isfinite(playback_fps) or playback_fps <= 0:
        raise ValueError("playback_fps must be a positive finite number")

    first_frame = load_frame(int(timepoints[0]), index)
    for i, mask in enumerate(masks, start=1):
        if mask.shape != first_frame.shape:
            raise ValueError(f"ROI {i} shape {mask.shape} != frame shape {first_frame.shape}")

    contrast_rows = np.linspace(0, index.height - 1, min(12, index.height), dtype=int)
    contrast_pixels = np.concatenate(
        [
            load_frame(int(index["timepoint"][int(row)]), index)[::8, ::8].ravel()
            for row in contrast_rows
        ]
    )
    vmin, vmax = np.percentile(contrast_pixels, [1, 99])
    if vmax <= vmin:
        vmax = vmin + 1

    fig, ax = plt.subplots(figsize=(9.6, 7.2))
    image_artist = ax.imshow(first_frame, cmap="coolwarm", vmin=vmin, vmax=vmax)
    colors = plt.cm.tab20.colors
    for i, mask in enumerate(masks):
        ax.contour(mask, levels=[0.5], colors=[colors[i % len(colors)]], linewidths=0.65)
    title = ax.set_title("", fontsize=10)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.95)

    writer = animation.FFMpegWriter(
        fps=playback_fps,
        codec="libx264",
        bitrate=4000,
        metadata={"title": "Calcium recording with ROI overlay"},
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    try:
        with writer.saving(fig, str(output_path), dpi=dpi):
            for tp, elapsed in tqdm(
                zip(timepoints, elapsed_s), total=len(timepoints), desc="Writing ROI MP4"
            ):
                image_artist.set_data(load_frame(int(tp), index))
                title.set_text(
                    f"Ch{channel} | TP {int(tp)} | t={elapsed:.1f}s | {roi_source} ROI"
                )
                writer.grab_frame()
    finally:
        plt.close(fig)

    return output_path
