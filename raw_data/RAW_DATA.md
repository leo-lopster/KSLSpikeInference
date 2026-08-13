# Format for Datasets

## **TL;DR**

A dataset is a single folder holding one recording. The pictures come in one of two
arrangements, and both work: either thousands of separate image files, one per moment in
time, or a single file holding the whole recording at once. Two things have to be right.
The image files must be named `ImageData_Ch0_TP0000000.npy`,
`ImageData_Ch0_TP0000001.npy`, … — counting upwards, always with the same number of
digits, because the app puts them in order by name and uneven digits shuffle the recording
out of sequence (a one-file recording is just `…TP0000000.npy` on its own). And the folder
should contain `ElapsedTimes.yaml`, the small text file the microscope writes to record
when each image was taken; without it the app still runs, but it has to *assume* how fast
the recording went, and a wrong assumption makes every later result wrong in a way that
still looks believable. Everything else in the folder is ignored. Finally, name the folder
with the recording's name first and a hyphen after it (`Debi_DRG-Streamtodisk-…`), because
the app takes everything before the first hyphen and stamps it onto every file it produces
from then on.

What the pipeline needs in order to open a recording at **Tab 1 · Dataset**, and what it
accepts as filenames. Written against the loaders in `analysis_tools/store.py`
(`list_frames`, `load_frame`, `load_imgdir`, `load_timebase`) — those functions are the
authority, and everything below is what they actually do.

A dataset folder is a **frame dump**: one `.npy` file per timepoint per channel, plus the
metadata the acquisition software wrote beside them. The reference example in this
directory is
[`Debi_DRG-Streamtodisk-1769166699-997.imgdir/`](Debi_DRG-Streamtodisk-1769166699-997.imgdir)
— 3220 timepoints of `(1024, 1376)` uint16 at ~10 Hz, streamed to disk by a 3i SlideBook
system.

**Only two things are read.** Everything else in a stream-to-disk folder is kept for
provenance and never opened:

| File | Status | Read by |
|---|---|---|
| `ImageData_Ch<c>_TP<nnnnnnn>.npy` | **Required** | `list_frames`, `load_frame` |
| `ElapsedTimes.yaml` | Optional, strongly recommended | `load_timebase` |
| `ChannelRecord.yaml` | Ignored | — |
| `AnnotationRecord.yaml` | Ignored | — |
| `AuxData.yaml` | Ignored | — |
| `HistogramData_Ch<c>_TP<nnnnnnn>.npy` | Ignored | — |
| `HistogramSummary_Ch<c>.npy`, `ImageRecord.yaml`, `MaskRecord.yaml`, `SAPositionData.yaml`, `StagePositionData.yaml` | Ignored | — |
| `experiment_summary.json` / `.pkl` | Ignored, but see [§1](#1-metadata) | — |

A folder containing nothing but the `ImageData_*.npy` frames will load and run. It will run
on a **nominal** time axis, which is the one thing in this document that quietly corrupts
every downstream number — see [§1](#1-metadata).

---

## 1. Metadata

### `ElapsedTimes.yaml` — the acquisition timebase

The only metadata file the pipeline reads. Its format, exactly as
[the example](Debi_DRG-Streamtodisk-1769166699-997.imgdir/ElapsedTimes.yaml) writes it:

```yaml
---
theElapsedTimes: [
3220,0,99,199,299,399,500,599,699,799,900,999,1100,1200,
...
321716,321816,321917,322017,322117,322217
 ]
```

| Property | Requirement |
|---|---|
| Top-level key | `theElapsedTimes` — a single flat list. Any other key is not found and the file is treated as absent. |
| First entry | The **frame count** (`3220`), not a time. `load_timebase` drops it before doing anything else. |
| Remaining entries | Elapsed time per frame, in **milliseconds**, one per timepoint. |
| Length | `len(list) - 1` must equal the number of `ImageData_Ch<c>_TP*.npy` files for the channel being loaded. |
| Origin | Need not start at 0; times are re-zeroed against the first entry and converted to seconds. |
| Spacing | Need not be uniform. The frame rate is the **median** inter-frame interval, so occasional dropped or late frames do not shift it. |

**Frame rate is derived, never declared.** There is no field anywhere that states the
acquisition rate — it is computed from these timestamps. The example yields
`1/median(diff)` ≈ 10.0 Hz over 322.2 s.

**When the file is missing, unreadable, or the wrong length**, the loader prints a warning
and substitutes a nominal axis at the **Fallback Freq. (Hz)** value from Tab 1 (default
`10.0`). The Tab 1 readout then shows `time source: fallback` instead of
`ElapsedTimes.yaml`. This is a real failure mode, not a formality: the frame rate sets the
CASCADE resampling ratio and every phase window in Tab 6, so a wrong one produces output
that looks entirely plausible and is wrong throughout. **Check the time source line after
loading.** A length mismatch in particular means the metadata does not describe these
frames — usually a folder that was copied or truncated part-way.

### `experiment_summary.json` / `.pkl` — not read, but the best record of the run

Present in the newer `DRG_SNI-live_recordings` exports and worth reading by hand: it
states `n_frames`, `height`, `width`, `pixel_size` (µm), `FOV_size`, which channel is
green and which is red, the acquisition date/time, the stage position, and the
stimulation protocol — `stimulation_events`, `stimulation_timeframes`, `stimulation_ms`,
`repetitions`, `duty_cycle`, `stimulated_rois`. It also repeats the timebase as
`time_stamps` (ms, one per frame, no leading count entry — unlike `ElapsedTimes.yaml`).

Nothing in the pipeline opens it. Where its `n_frames`, `height` and `width` are a useful
independent check on what was loaded, the stimulation fields are the obvious source for
the Tab 6 phase table — currently typed in by hand. Both are unimplemented, not
unavailable.

### `ChannelRecord.yaml`, `AnnotationRecord.yaml`, `AuxData.yaml` — ignored

Emitted by the 3i system; no code path opens them. They carry the exposure record, ROI
annotations and auxiliary data tables, and they are worth keeping with the dataset as an
acquisition record — `ChannelRecord.yaml` is where `mExposureTime` and the plane count
live, which is the only durable statement of how the recording was made. But nothing is
validated against them, and their absence blocks nothing.

Note in particular: **the channel count is not read from `ChannelRecord.yaml`.** Which
channel gets analysed is the **Use Channel...** spin box in Tab 1, and the only evidence
that a channel exists is whether `ImageData_Ch<c>_TP*.npy` files are on disk for it.

---

## 2. Image data and channels

### Filename convention

```
ImageData_Ch<channel>_TP<timepoint>.npy
        └─ int    └─ zero-padded int
```

`list_frames` globs `ImageData_Ch{channel}_TP*.npy` and **sorts the result as text**.

| Rule | Why |
|---|---|
| `Ch<c>` — integer, no padding. `Ch0`, `Ch1`, … | Matched literally against the channel number in Tab 1. `Ch00` will not be found by a request for channel `0`. |
| `TP<n>` — **zero-padded**, 7 digits in the example (`TP0000000` … `TP0003219`) | The sort is lexicographic, so padding is what makes it temporal. Unpadded names order `TP10` before `TP9`, silently scrambling the recording into an order no error can detect. Any consistent width works; **inconsistent width does not.** |
| Numbering | Starts at 0 in the example. Nothing depends on the start value or on the numbers being contiguous — the sorted order is what matters, and a gap simply means the frame does not exist. |
| Extension | `.npy` only. |

### Array format inside each file — two layouts

Both are read; which one you have is detected from the `.npy` header of the first file,
never from the file count.

| Layout | Files per channel | Array shape | Seen in |
|---|---|---|---|
| **per-timepoint** | one per frame, `TP0000000` … `TP<N-1>` | `(1, H, W)` | `Debi_DRG-Streamtodisk-…imgdir` |
| **single-stack** | **one**, always `TP0000000` | `(T, H, W)` | `DRG_SNI-live_recordings/Slide1.dir/…imgdir` |

| Property | Requirement | Example |
|---|---|---|
| Axes | Leading axis is **frames**, then `(y, x)` | `(667, 356, 396)` = 667 frames, y=356, x=396 |
| dtype | Any numeric; converted to float32 during preprocessing | `uint16` |
| Consistency | Every frame shares the shape and dtype of the **first** | — |

The leading axis is not optional in either layout. A bare `(H, W)` array does not raise —
it is read as a stack whose "frames" are single pixel rows, and the first symptom is an
ROI-label shape mismatch several tabs later. Save frames as `arr[None, ...]` if you are
generating them yourself.

**Why the file count cannot decide the layout.** A single-stack recording read as
one-file-per-timepoint yields a **1-frame** movie: the loader takes `array[0]` and stops.
Nothing raises, because a 1-frame stack is a legal stack — the frame count silently
becomes whatever was assumed rather than what is on disk. `store.frame_layout` therefore
reads the leading axis of the first file (header only, no pixels) and reports
`per-timepoint` or `single-stack` in the log line when the dataset loads. Check it:

```
> 667 timepoints, frame (356, 396) uint16 [single-stack (1 file(s) of 667 frames)] <- …
> 3220 timepoints, frame (1024, 1376) uint16 [per-timepoint (3220 files)] <- …
```

Frames are addressed as `(file, index)` pairs — `store.frame_index` — so one code path
covers both layouts, and reads are memory-mapped: pulling frame 500 out of a 188 MB stack
costs one frame, not the file. Only headers are read when a dataset is opened; a later
frame of a different shape therefore fails at compute time — during the Zarr write or
trace extraction — not at load.

### Channels

One channel is analysed at a time. **Use Channel...** in Tab 1 selects it, the matching
frames become the stack, and frames of every other channel are ignored for that session.
To compare channels, load the dataset twice and preprocess each into its own `.zarr` store.

`Debi_DRG` is single-channel (`Ch0` only). A two-channel acquisition carries
`ImageData_Ch0_*.npy` and `ImageData_Ch1_*.npy` side by side in the same folder — set the
spin box to whichever holds the calcium indicator. In the `DRG_SNI-live_recordings`
datasets that is **`Ch0` (green)**, with `Ch1` the red reference channel; both are
`(667, 356, 396)`, and `experiment_summary.json` states the mapping outright as
`green_channel: 0, red_channel: 1`. A wrong channel number is
reported immediately and by name:

```
No 'ImageData_Ch1_TP*.npy' frames in <dir>. Check the path and the channel number
(channel is not always 0 in multi-channel acquisitions).
```

### `HistogramData_Ch<c>_TP<nnnnnnn>.npy` — ignored

One per timepoint alongside every image frame — `(1, 16384)` uint32 intensity histograms
in the example, and roughly half the file count of the folder. Never read. They are
matched by neither glob, so they cost only disk space.

### How many frames are enough

| Constraint | Minimum |
|---|---|
| Deriving a frame rate | 2 timepoints |
| `BASELINE_SLIDES` (Tab 4, default 20) | must fit inside the extraction window |
| `EXTRACT_TIMEPOINTS` (Tab 4, default 500) | clipped to what exists; a shorter recording is not an error |
| Tab 6 phases | every phase must contain at least one frame, or the analysis refuses to start |

---

## 3. Others

### Folder naming — and the `dataset_tag` trap

The dataset folder name is not merely descriptive. `store.dataset_tag` takes **everything
before the first hyphen** and that string becomes `dataset_tag`, which is embedded in every
downstream filename:

```
Debi_DRG-Streamtodisk-1769166699-997.imgdir   ->   Debi_DRG
                                                     |
              preprocessed/Debi_DRG_nlm.zarr  <──────┤
              analysis/Debi_DRG/traces.csv    <──────┤
              labels/Debi_DRG_ROI.tiff        <──────┘  (suggested name)
```

| Folder name | Resulting tag | Verdict |
|---|---|---|
| `Debi_DRG-Streamtodisk-1769166699-997.imgdir` | `Debi_DRG` | Correct — the 3i convention works as intended. |
| `Mark_DCN_RF_FOV1` | `Mark_DCN_RF_FOV1` | Fine. Underscores are not separators; the whole name is the tag. |
| `MyRecording.imgdir` | `MyRecording.imgdir` | **Bad.** No hyphen means the suffix is never stripped, so stores are named `MyRecording.imgdir_nlm.zarr`. |
| `2026-02-10_DRG_FOV1` | `2026` | **Bad.** Leading date, so every recording made that year collides on one tag. |

Rules that follow: **put the identifying name first**, use `_` inside it, and use `-` only
to separate the name from acquisition junk (session ids, timestamps). The `.imgdir`
extension is a convention, not a requirement — nothing checks it, and any directory holding
correctly named frames will load — but keep it, because it is the only thing marking the
folder as a frame dump rather than an output directory.

Two datasets that reduce to the same tag will overwrite each other's stores and analysis
folders without warning. Tags are checked for consistency in one place only: loading a
`.zarr` whose tag disagrees with the open dataset logs a warning and keeps the dataset's
tag.

### Other accepted entry points

The pipeline does not have to start at raw frames. Two other inputs are complete entry
points in their own right, with their own naming rules (full detail in the
[main README](../README.md)):

| Entry | Where | Naming convention |
|---|---|---|
| Preprocessed stack | Tab 2 · **Load from Zarr** | `<dataset_tag>_<method>.zarr` with a sibling `<dataset_tag>_<method>.params.json`. The name is parsed for the tag and the denoise chain, so `_` before the method matters. |
| dF/F traces | Tab 4 · **Import dF/F** | `.csv` (a time column named `time`/`time_s`/`t`/`seconds`…, then one column per ROI, ideally `ROI1`, `ROI2`, … so ROI ids survive the round trip), `.npz` with a `dff` key plus optional `t` and `roi_ids`, or a bare 2-D `.npy`. |
| ROI trace export | Tab 4 · **Import dF/F** | Tab-separated `*_roi_traces.txt` — `Frame`, `Time`, then `Green_Mean_ROI<x>` / `Red_Mean_ROI<x>` / `Trace_ROI<x>` per ROI. Converted to `.csv` beside the source before it is read; only `Trace_ROI<x>` is imported. Full spec: [DRG_Baseline/DATASET_FORMAT.md](DRG_Baseline/DATASET_FORMAT.md). |
| ROI labels | Tab 3 · **Load ROI labels** | uint16 `.tiff` label image, shape exactly `(H, W)` of the preprocessed stack. Suggested `<dataset_tag>_ROI.tiff`; the dialog records whatever you actually pick. |

A traces file with no time axis **must** be given `frame_rate_hz` at import — the importer
raises rather than guessing, for the reason given in §1.

### Checking a folder before loading it

```bash
D=raw_data/Debi_DRG-Streamtodisk-1769166699-997.imgdir

ls "$D"/ImageData_Ch0_TP*.npy | wc -l          # frame count for channel 0
ls "$D"/ImageData_Ch*_TP0000000.npy            # which channels exist
head -3 "$D"/ElapsedTimes.yaml                 # first entry = frame count?
python -c "import numpy as np; a=np.load('$D/ImageData_Ch0_TP0000000.npy'); print(a.shape, a.dtype)"
```

Expect the frame count and the first entry of `theElapsedTimes` to be the **same number**,
and the array shape to have three axes with a leading `1`. After loading, confirm in the
Tab 1 readout that `time axis:` reads `ElapsedTimes.yaml` rather than `fallback`, and that
`dataset_tag` is the short name you expected.
