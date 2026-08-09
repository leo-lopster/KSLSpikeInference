# GUI Options Spec — `process_using_napari.ipynb` → PyQt

Inventory of every option the notebook exposes, grouped by the panel it would become,
in preparation for a PyQt front-end. Source of truth is the notebook plus
`analysis_tools/`; where a value below disagrees with the code, the code wins.

Scope note: this document lists *options*. The I/O behind them is implemented
(`analysis_tools/store.py`); the compute kernels are not yet lifted out of the notebook
(see §9, *Gaps*).

**Legend** — Tier **P** = primary, exposed by default. Tier **A** = advanced, collapsed
by default but still editable. Tier **D** = documented here, *adjustability deliberately
not implemented yet* (plot cosmetics).

---

## 0. The five I/O points

Everything the app reads or writes, and the function behind it. All live in
`analysis_tools/store.py` unless noted.

| # | Point | Direction | Format | Function |
|---|---|---|---|---|
| 1 | Frame images from an `.imgdir` | import | `ImageData_Ch<n>_TP*.npy` | `list_frames`, `load_frame`, `load_imgdir`, `load_timebase` |
| 2 | ROI labels | import + export | `.tiff` uint16 label image | `load_roi_labels`, `save_roi_labels` |
| 3 | dF/F traces, many ROIs | import | `.csv` / `.npz` / `.npy` | `load_traces`, `import_dff` |
| 4 | dF/F traces | export | `.npz` + `.csv` + `.json` | `save_traces` |
| 5 | PCA analysis + summary | export | `.csv` tables + `.tiff` figures | `save_pca`, `save_features`, `save_groups`, `save_figure` |

Supporting I/O: `save_preprocessed` / `load_preprocessed` (Zarr cache),
`cascade_runner.save` / `cascade_runner.load` (spike inference).

---

## 1. Session modes

The app has two entry points, and which one is active determines what is even meaningful
to show.

### Full mode — enters at point 1

All panels live. Stages unlock in order as artifacts appear:

| Artifact | Unlocks |
|---|---|
| `.imgdir` selected | Preprocessing |
| `preprocessed/<tag>_<method>.zarr` | ROI panel |
| `labels/<tag>_ROI.tiff` | Trace extraction |
| `analysis/<tag>/traces.npz` | Analysis (spike inference onward) |
| `analysis/<tag>/spike_rate.npy` | Features / PCA |
| `analysis/<tag>/pca_<label>.npz` | Re-cut the tree without re-running PCA |

`store.available(analysis_dir)` returns these as a dict — use it to drive the gating
rather than catching `FileNotFoundError`.

### Traces-only mode — enters at point 3

Importing dF/F directly means there is no pixel data behind the traces, so upstream
stages are not stale, they are *absent*.

**Disabled:** dataset/image import · preprocessing panel (all of §3 below) · ROI label
import/export · the extraction window (`START_TIMEPOINT`, `EXTRACT_TIMEPOINTS`,
`BASELINE_SLIDES` — imported traces are already baselined) · the napari viewer, which has
nothing to display.

**Live:** spike inference · phases · features · PCA · grouping · every export.

**The one thing that must be got right:** frame rate. There is no `ElapsedTimes.yaml` in
this mode, so it comes from a time column in the imported file or from manual entry.
`FRAME_RATE` drives CASCADE resampling, the phase windows and the Nyquist ceiling on
`FREQ_BANDS` — a wrong value produces plausible-looking output that is wrong throughout.
`import_dff` refuses to guess: no time column and no `frame_rate_hz` raises.

Switching modes must clear downstream state rather than mix provenance. The mode is
recorded in every params sidecar written afterwards.

---

## 2. Dataset panel (point 1)

| Option | Tier | Widget | Default | Notes |
|---|---|---|---|---|
| `DATA_DIR` | P | directory picker | `./datasets/Debi_DRG-Streamtodisk-1769166699-997.imgdir` | The `.imgdir` frame dump. |
| `channel` | P | int spin / dropdown | `0` | Was hard-coded as `ImageData_Ch0_TP*.npy`; now a parameter of `list_frames`/`load_imgdir`. |
| frame count | — | read-only | derived | From `list_frames`. |
| frame shape / dtype | — | read-only | derived | Observed `(1024, 1376)` uint16. |
| `fallback_hz` | A | float | `10.0` | Used **only** when `ElapsedTimes.yaml` is missing or its length disagrees with the stack. Surface the warning; do not bury it. |
| time source | — | read-only | derived | `ElapsedTimes.yaml` or `fallback` — from `load_timebase`. |
| `dataset_tag` | — | read-only | `DATA_DIR.name.split("-")[0]` → `Debi_DRG` | Every downstream filename keys off this. `store.dataset_tag()`. |

---

## 3. Preprocessing panel

All parameters exposed for input, per the plan. Source: notebook cell "CONSTANTS" plus
the denoising cell.

| Option | Tier | Widget | Default | Range / choices |
|---|---|---|---|---|
| `DENOISE_METHOD` | P | ordered multi-select | `"nlm"` | Any subset of `gaussian`, `nlm`, applied **in sequence** (`dct` deprecated); stored comma-separated (`"gaussian,nlm"`). |
| `DENOISE_SIGMA` | P | float | `3.0` | Gaussian filter sigma, px. Applies only if `gaussian` is in the chain. |
| `NLM_PATCH_SIZE` | P | int | `5` | px. NLM only. |
| `NLM_PATCH_DISTANCE` | P | int | `6` | px search radius. NLM only. |
| `NLM_H_FACTOR` | P | float | `1.25` | Cut-off distance, multiplied by the per-frame `estimate_sigma`. NLM only. |
| ~~`DCT_THRESHOLD_FRACTION`~~ | — | *deprecated* | ~~`0.05`~~ | DCT denoising is deprecated and commented out in `analysis_tools/preprocess.py`; the control is gone from the GUI. |
| `BACKGROUND_PERCENTILE` | P | int | `4` | Per-frame percentile subtracted as background. |
| `upsample_factor` | A | int | `10` | Sub-pixel precision of `phase_cross_correlation` motion correction. |
| projection stride | A | int | `5` | `preprocessed_stack[::stride]` for the max projection. Display only. |
| `PREPROCESSED_DIR` | A | directory picker | `./preprocessed/` | Zarr cache location. |
| `method_tag` | — | read-only | derived | `store.method_tag(DENOISE_METHOD)`. |

**Trap to surface in the UI:** the output filename is `<dataset_tag>_<method_tag>.zarr`.
Changing `DENOISE_METHOD` therefore silently retargets which store is saved *and loaded* —
a user who switches the denoiser and hits "load" gets a different recording's
preprocessing, or a "not found" for work they believe they did. Show the resolved path
next to the control, and say which methods have a cached store on disk.

Save/load actions: `store.save_preprocessed` (Zarr + `.params.json` sidecar, overwrites
without prompting — the GUI should prompt) and `store.load_preprocessed`.

---

## 4. ROI panel (point 2)

| Option | Tier | Widget | Default | Notes |
|---|---|---|---|---|
| `LABELS_DIR` | A | directory picker | `./labels` | |
| `ROI_LABELS_PATH` | P | file picker | `<LABELS_DIR>/<dataset_tag>_ROI.tiff` | Load and save target. |
| `REFERENCE_IMAGE_LAYER` | P | dropdown of layers | `"preprocessed_max"` | The layer the Labels layer is sized/aligned to. `"preprocessed_std"` is the documented alternative (now archived in the notebook). |
| label dtype | — | fixed | `uint16` | |
| overlay opacity | D | slider | `0.6` | |

Actions: **new blank ROI labels** (sized to the reference layer) · **load** (`store.load_roi_labels`,
which raises on a shape mismatch against the reference — catch and show it as a dialog,
since the alternative is silently wrong traces) · **save** (`store.save_roi_labels`).

**Napari layer names are currently lookup keys** — `"ROI labels"`, `"preprocessed_max"`,
`f"preprocessed_{DENOISE_METHOD}"`, `"raw_stack"`, `"frame_0000"`. In a GUI these should
become held references, not strings the user can break by renaming a layer.

---

## 5. Trace extraction / import panel (points 3 and 4)

| Option | Tier | Widget | Default | Notes |
|---|---|---|---|---|
| `START_TIMEPOINT` | P | int | `0` | First frame of the extraction window. **See the bug note below.** |
| `EXTRACT_TIMEPOINTS` | P | int | `500` | Frames to extract (≈50 s at 10 Hz). |
| `BASELINE_SLIDES` | P | int | `20` | Leading frames averaged into F0 (≈2 s). Interacts with the `pre` phase — see §6. |
| import path | P | file picker | — | Traces-only entry. `.csv` / `.npz` / `.npy`. |
| `orientation` | P | dropdown | `auto` | `roi_rows` \| `roi_cols` \| `auto`. Auto reads a table with a time column as one column per ROI; otherwise it takes the longer axis as time. |
| `time_column` | A | text / dropdown | `auto` | Column name or index, `auto`, or none. Auto accepts a column named time/t/seconds, or a strictly increasing numeric first column in a taller-than-wide table. |
| `frame_rate_hz` | P (traces-only) | float | — | **Required** when the file carries no time axis. |

Export writes three files via `store.save_traces`: `traces.npz` (exact, incl. raw
pre-dF/F traces), `traces.csv` (portable, `time_s` + one column per ROI), and
`traces_params.json` (mode, dataset, source zarr, ROI-label path, the three options above,
frame rate, ROI/frame counts).

> **Known bug in the extraction loop, inherited from the notebook.** In
> `extract_roi_traces` the per-ROI array is allocated with length
> `n_t = min(stack.shape[0], start_t + extract_timepoints)` but written at absolute index
> `t`, which runs from `start_t`. With `START_TIMEPOINT = 0` (the only value used so far)
> this is correct; with any non-zero start it writes past the end of the array or
> misaligns the trace. A GUI that exposes `START_TIMEPOINT` will hit this immediately.
> Fix when the extraction kernel is lifted into `analysis_tools/traces.py` (§9).

---

## 6. Spike inference panel

| Option | Tier | Widget | Default | Notes |
|---|---|---|---|---|
| `CASCADE_DIR` | A | directory picker | `./CascadeTorch` | Also supplies the FFT helper used by Index B. |
| `CASCADE_MODEL_DIR` | A | directory picker | `<CASCADE_DIR>/Pretrained_models` | 24 models present locally. |
| `MODEL_NAME` | P | dropdown | `Spinal_cord_excitatory_30Hz_smoothing50ms` | Populate from `cascade_runner.available_models()`. |
| `ANALYSIS_DIR` | A | directory picker | `./analysis/<dataset_tag>` | `store.analysis_dir()`. Traces I/O only — inference and analysis outputs ask for their folder per run (below). |

The dropdown must show what `cascade_runner.check_model()` prints, inline and before the
run: training rate parsed from the **name** vs the rate in `config.yaml` (at least one
shipped model disagrees), the indicator the model was trained on (GCaMP6 vs 8 mismatch),
the resample ratio, and the pad-frames-per-end that will be NaN at each edge. Warn when
padding reaches into the `pre` window.

Model choice is a scientific claim, not a preference: the current default is a spinal
dorsal-horn model applied to DRG, which is the nearest available analogue and still an
untested transfer (CLAUDE.md §4C, App. A §8). The panel should carry that caveat where
the user picks, not only in the writeup.

Actions: **run** (`cascade_runner.run`) · **save** (`cascade_runner.save` →
`spike_rate.npy` + `spike_inference_params.json`) · **load** (`cascade_runner.load`, to
skip re-running).

Running does **not** write anything by itself. It offers to save, and saving asks for a
destination folder — it does not derive one from `ANALYSIS_DIR`. That derived path is
identical for every run of a dataset, so a second inference (different model, different
trace window) would overwrite the first with no record that it happened. `ANALYSIS_DIR`
is only where the folder dialog opens; the dialog's *new folder* button is how runs are
kept apart. Loading asks for a folder too, otherwise a run saved outside the derived path
could never be read back. Declining the save prompt keeps the rate in the session and
leaves the Save button live — it is lost only when the window closes.

---

## 7. Analysis panel (point 5)

| Option | Tier | Widget | Default | Notes |
|---|---|---|---|---|
| `PHASES` | P | editable table, name → (start, end) s | `pre (2,10)`, `stim (10,15)`, `post (15,50)` | `pre` starts at 2.0 s, not 0, because frames `0..BASELINE_SLIDES` *defined* F0 — dF/F there is ~0 by construction and would fake a silent baseline. Tie this control to `BASELINE_SLIDES` and warn if `pre` starts inside the F0 window. |
| `ACTIVE_THRESH` | P | float + auto toggle | `None` (auto) | spikes/s. Auto = baseline median + `ACTIVE_N_SIGMA` × robust sigma. |
| `ACTIVE_N_SIGMA` | P | float | `3.0` | A fixed absolute threshold does not transfer between models. |
| `FREQ_BANDS` | P | editable band table | `(0.2,0.5) (0.5,1.0) (1.0,2.0) (2.0,5.0)` Hz | Lower edge = FFT bin width of the *shortest* phase; upper edge = Nyquist of the acquisition rate. Both bounds should be recomputed and validated when `PHASES` or the frame rate changes. |
| `PCA_VAR_TARGET` | P | float 0–1 | `0.95` | Cumulative variance target selecting the PC count. |
| `MAX_K` | A | int | `8` | Largest k in the candidate-cut table. |
| `MIN_GROUP_SIZE` | A | int | `3` | A "group" of 1–2 ROIs is an outlier, not a population. |
| `MAX_GROUP_FRAC` | A | float | `0.9` | One group holding almost everything is not a partition. |
| `RATE_CUT_HEIGHT` | P | float, blank = auto | `None` | Manual dendrogram cut for Index A; blank → `grouping.suggest_cut`. |
| `FREQ_CUT_HEIGHT` | P | float, blank = auto | `None` | Same for Index B. |

Clicking **run analysis** asks for the output folder first, before any compute, for
the same reason as the inference save: re-running with different phases, thresholds or
cut heights writes the same nine filenames, and into a derived `<root>/<tag>` that
silently replaces the previous pass. `CASCADE_DIR` is also checked up front — Index B
loads its FFT helper from `CascadeTorch/scripts`, and a missing one used to fail halfway,
after the Index A CSVs were already on disk.

Exports, all into the folder chosen for that run:

- `rate_features.csv`, `freq_features.csv` — `store.save_features`
- `roi_groups.csv` — `store.save_groups`
- `pca_rate.npz` / `pca_freq.npz` + `.json` + `pca_<label>_scores.csv` — `store.save_pca`
- `figures/*.tiff` — `store.save_figure` on the figure each `plots.*` function returns:
  `pca_rate_summary`, `dendrogram_rate`, `rate_heatmap`, `rate_group_means`,
  `pca_freq_summary`, `dendrogram_freq`

`save_pca` stores scores, loadings, explained variance and the linkage matrix — enough to
re-plot and re-cut — but **not** the `StandardScaler` statistics, so it cannot project new
ROIs into a saved PCA space. The PCA is per-dataset; there is nothing to project.

---

## 8. Plotting panel — documented, adjustability not implemented

Listed so the eventual controls are known. Currently these are constants in the plot
functions and the notebook QC cells.

| Option | Current value | Where |
|---|---|---|
| dF/F heatmap colormap | `magma` | QC plot A, `plots.rate_heatmap` |
| max-projection colormap | `coolwarm` | notebook projection cells |
| robust colour limits | 1st / 99th percentile | QC plot A |
| heatmap `vmax` | `nanpercentile(99)` | `plots.rate_heatmap` |
| figure dpi | `200` | QC plot B, `store.save_figure` |
| figsize rules | `(12, 0.18·n_rois + 2)`, `(12, 1.9·n_groups + 1.2)`, `(16, 4.4)`, `(13, 4.6)` | `plots.*` |
| trace linewidth | `0.6` | QC plot B |
| `SHOW_ROI_TRACES` | `True` | draw member ROIs behind group means |
| `ROI_TRACE_ALPHA` | `0.3` | |
| stimulus shading | `PHASES["stim"]` | `plots.rate_heatmap`, `plots.group_means` |

Constraints that are **not** preferences and should stay fixed even once this panel is
built (CLAUDE.md §5): perceptually-uniform colormaps only, never `jet`; raw and inferred
traces in separate subplot rows or visually distinct; every panel showing inferred spikes
names the algorithm and its parameters in that same panel.

---

## 9. Gaps to close before the GUI can drive the pipeline

1. **Compute kernels are still notebook-resident.** The I/O is lifted; the processing is
   not. A GUI cannot call:
   - the preprocessing chain — `denoise_frame` (+ the gaussian/nlm/dct variants),
     `normalise_to_background`, `preprocess_frame` → should become
     `analysis_tools/preprocess.py`;
   - trace extraction and the F0/dF/F computation — `extract_roi_traces` and the
     `BASELINE_SLIDES` block → should become `analysis_tools/traces.py` (and fix the
     `START_TIMEPOINT` bug noted in §5 while doing it).
2. **Napari embedding.** The notebook drives a standalone `napari.Viewer()` via the
   `%gui qt6` magic. A PyQt app should embed `viewer.window._qt_window` (or use
   `napari.Viewer(show=False)` and add the widget) rather than spawn a second event loop.
3. **Long operations need to be cancellable and off the UI thread** — preprocessing writes
   3220 frames to Zarr, and trace extraction streams 500 frames × 59 ROIs. Both currently
   report progress via `ProgressBar`/`tqdm` to stdout.
4. **Stale references in the archived cells** — `PROJECTIONS_DIR` still points at
   `/Users/leolopster/Desktop/spike_inference/_projections` (the repo has since moved to
   `KSLSpikeInference`), and the archived STD-projection cell references a lowercase
   `data_dir` that no longer exists. Fix or delete before re-enabling that path.
5. **`labels/Debi_DRG_ROI.csv`** exists on disk but nothing reads or writes it. Decide
   whether it is an input format worth supporting or a leftover.
