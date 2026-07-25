"""Parse LT two-photon raw image exports and save one preview per ImageData array.

The LT_2p_rawimage tree is expected to look like:

    LT_2p_rawimage/<session>/Slide*.dir/<trial>.imgdir/ImageData*.npy

This helper keeps traversal, loading, preview rendering, and logging separate so
future processing steps can plug in without rewriting the crawler.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


LOGGER = logging.getLogger("parse_lt_2p_rawimage")


@dataclass(frozen=True)
class ParseConfig:
    root_dir: Path
    output_dir: Path
    image_pattern: str
    preview_index: int
    frame_axis: int
    cmap: str
    show: bool
    dpi: int
    max_files: int | None


@dataclass(frozen=True)
class ImageDataFile:
    session_dir: Path
    slide_dir: Path
    img_dir: Path
    npy_path: Path


@dataclass
class ParseRecord:
    session: str
    slide_dir: str
    imgdir: str
    filename: str
    path: str
    shape: str | None = None
    dtype: str | None = None
    ndim: int | None = None
    preview_index: int | None = None
    preview_path: str | None = None
    preview_min: float | None = None
    preview_max: float | None = None
    status: str = "pending"
    error: str | None = None


def parse_args() -> ParseConfig:
    parser = argparse.ArgumentParser(
        description="Find LT_2p_rawimage .dir/.imgdir folders, parse ImageData*.npy files, and save image previews/logs."
    )
    parser.add_argument(
        "root_dir",
        nargs="?",
        type=Path,
        default=Path("LT_2p_rawimage"),
        help="Root folder to scan (default: LT_2p_rawimage).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output folder for previews and logs (default: <root_dir>/parse_outputs/<timestamp>).",
    )
    parser.add_argument(
        "--image-pattern",
        default="ImageData*.npy",
        help="Image .npy glob pattern inside each .imgdir folder (default: ImageData*.npy).",
    )
    parser.add_argument(
        "--preview-index",
        type=int,
        default=0,
        help="Frame/slice index to preview along the frame axis (default: 0).",
    )
    parser.add_argument(
        "--frame-axis",
        type=int,
        default=0,
        help="Axis treated as the frame/slide axis for arrays with 3+ dimensions (default: 0).",
    )
    parser.add_argument(
        "--cmap",
        default="gray",
        help="Matplotlib colormap for preview images (default: gray).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display each preview with Matplotlib in addition to saving it.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Preview PNG DPI (default: 150).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Process only the first N ImageData files, useful for smoke tests.",
    )

    args = parser.parse_args()
    root_dir = args.root_dir.expanduser().resolve()
    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = root_dir / "parse_outputs" / stamp

    return ParseConfig(
        root_dir=root_dir,
        output_dir=output_dir.expanduser().resolve(),
        image_pattern=args.image_pattern,
        preview_index=args.preview_index,
        frame_axis=args.frame_axis,
        cmap=args.cmap,
        show=args.show,
        dpi=args.dpi,
        max_files=args.max_files,
    )


def configure_logging(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir.parent / ".matplotlib"))
    log_path = output_dir / "parse_lt_2p_rawimage.log"

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))

    logging.basicConfig(level=logging.INFO, handlers=[file_handler, stream_handler], force=True)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    return log_path


def iter_slide_dirs(root_dir: Path) -> Iterable[Path]:
    """Yield folders with a .dir suffix below the LT root."""
    yield from sorted(path for path in root_dir.rglob("*.dir") if path.is_dir())


def iter_img_dirs(slide_dir: Path) -> Iterable[Path]:
    """Yield .imgdir folders immediately inside a Slide*.dir folder."""
    yield from sorted(path for path in slide_dir.iterdir() if path.is_dir() and path.name.endswith(".imgdir"))


def iter_image_data_files(root_dir: Path, image_pattern: str) -> Iterable[ImageDataFile]:
    for slide_dir in iter_slide_dirs(root_dir):
        session_dir = slide_dir.parent
        LOGGER.info("Scanning slide folder: %s", slide_dir)

        img_dirs = list(iter_img_dirs(slide_dir))
        if not img_dirs:
            LOGGER.warning("No .imgdir folders found in %s", slide_dir)
            continue

        for img_dir in img_dirs:
            image_paths = sorted(img_dir.glob(image_pattern))
            if not image_paths:
                LOGGER.warning("No %s files found in %s", image_pattern, img_dir)
                continue

            for npy_path in image_paths:
                yield ImageDataFile(
                    session_dir=session_dir,
                    slide_dir=slide_dir,
                    img_dir=img_dir,
                    npy_path=npy_path,
                )


def load_npy_image(path: Path) -> np.ndarray:
    """Load an ImageData array via NumPy's public .npy API."""
    return np.load(path, mmap_mode="r", allow_pickle=False)


def choose_preview_plane(arr: np.ndarray, preview_index: int, frame_axis: int) -> tuple[np.ndarray, int]:
    """Return a 2D preview plane and the frame index actually used."""
    if arr.ndim < 2:
        raise ValueError(f"Expected at least 2 dimensions for image data, got shape {arr.shape}")

    if arr.ndim == 2:
        return np.asarray(arr), 0

    normalized_axis = frame_axis % arr.ndim
    axis_len = arr.shape[normalized_axis]
    if axis_len == 0:
        raise ValueError(f"Frame axis {normalized_axis} is empty for shape {arr.shape}")

    actual_index = min(max(preview_index, 0), axis_len - 1)
    plane = np.take(arr, indices=actual_index, axis=normalized_axis)

    while plane.ndim > 2:
        plane = np.take(plane, indices=0, axis=0)

    return np.asarray(plane), actual_index


def robust_display_limits(image: np.ndarray) -> tuple[float, float]:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return 0.0, 1.0

    vmin, vmax = np.percentile(finite, [1, 99])
    if np.isclose(vmin, vmax):
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0

    return float(vmin), float(vmax)


def sanitize_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def build_preview_path(item: ImageDataFile, previews_dir: Path) -> Path:
    rel_parts = [
        item.session_dir.name,
        item.slide_dir.name,
        item.img_dir.name,
        item.npy_path.stem,
    ]
    return previews_dir / f"{sanitize_filename('__'.join(rel_parts))}.png"


def save_preview_image(
    image: np.ndarray,
    output_path: Path,
    title: str,
    cmap: str,
    dpi: int,
    show: bool,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_path.parent.parent / ".matplotlib"))

    if not show:
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    vmin, vmax = robust_display_limits(image)

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    fig.savefig(output_path, dpi=dpi)

    if show:
        plt.show(block=False)
        plt.pause(0.001)

    plt.close(fig)


def make_record(item: ImageDataFile) -> ParseRecord:
    return ParseRecord(
        session=item.session_dir.name,
        slide_dir=item.slide_dir.name,
        imgdir=item.img_dir.name,
        filename=item.npy_path.name,
        path=str(item.npy_path),
    )


def parse_one_image(item: ImageDataFile, config: ParseConfig, previews_dir: Path) -> ParseRecord:
    record = make_record(item)
    LOGGER.info("Parsing %s", item.npy_path)

    try:
        arr = load_npy_image(item.npy_path)
        record.shape = str(tuple(arr.shape))
        record.dtype = str(arr.dtype)
        record.ndim = int(arr.ndim)

        preview, actual_index = choose_preview_plane(arr, config.preview_index, config.frame_axis)
        record.preview_index = int(actual_index)
        record.preview_min = float(np.nanmin(preview))
        record.preview_max = float(np.nanmax(preview))

        preview_path = build_preview_path(item, previews_dir)
        title = f"{item.slide_dir.name} / {item.img_dir.name} / {item.npy_path.name} [{actual_index}]"
        save_preview_image(
            preview,
            output_path=preview_path,
            title=title,
            cmap=config.cmap,
            dpi=config.dpi,
            show=config.show,
        )

        record.preview_path = str(preview_path)
        record.status = "ok"
    except Exception as exc:
        record.status = "error"
        record.error = str(exc)
        LOGGER.exception("Failed to parse %s", item.npy_path)

    return record


def write_csv_log(records: Sequence[ParseRecord], output_dir: Path) -> Path:
    path = output_dir / "parse_records.csv"
    fieldnames = list(asdict(records[0]).keys()) if records else list(ParseRecord("", "", "", "", "").__dict__.keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
    return path


def write_jsonl_log(records: Sequence[ParseRecord], output_dir: Path) -> Path:
    path = output_dir / "parse_records.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
    return path


def limit_items(items: Iterable[ImageDataFile], max_files: int | None) -> Iterable[ImageDataFile]:
    for idx, item in enumerate(items):
        if max_files is not None and idx >= max_files:
            return
        yield item


def run(config: ParseConfig) -> list[ParseRecord]:
    if not config.root_dir.exists():
        raise SystemExit(f"Root directory does not exist: {config.root_dir}")

    previews_dir = config.output_dir / "previews"
    records = [
        parse_one_image(item, config, previews_dir)
        for item in limit_items(iter_image_data_files(config.root_dir, config.image_pattern), config.max_files)
    ]

    csv_path = write_csv_log(records, config.output_dir)
    jsonl_path = write_jsonl_log(records, config.output_dir)

    ok_count = sum(1 for record in records if record.status == "ok")
    error_count = sum(1 for record in records if record.status == "error")
    LOGGER.info("Parsed %d ImageData file(s): %d ok, %d error", len(records), ok_count, error_count)
    LOGGER.info("Saved CSV log: %s", csv_path)
    LOGGER.info("Saved JSONL log: %s", jsonl_path)

    return records


def main() -> None:
    config = parse_args()
    log_path = configure_logging(config.output_dir)
    LOGGER.info("Root directory: %s", config.root_dir)
    LOGGER.info("Output directory: %s", config.output_dir)
    LOGGER.info("Text log: %s", log_path)
    run(config)


if __name__ == "__main__":
    main()
