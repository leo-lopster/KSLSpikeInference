# GUI for KSLSpikeInference

**The upstream CascadeTorch README is attached at the end of this file** ([PTRRupprecht/CascadeTorch](https://github.com/PTRRupprecht/CascadeTorch), GPL-3.0), since `CascadeTorch/` is vendored into this repository.

Everything below documents the local pipeline that *uses* it — the DRG/DCN spike-inference workflow in this
repository. It is not part of the upstream toolbox and will not be found in the
CascadeTorch or Cascade repositories.

Run it with `python spike_inference_gui.py`. A napari viewer opens alongside the control
window for the image and ROI work. Every file operation goes through `analysis_tools/`,
which the archived notebook (`archive/process_using_napari.ipynb`) also calls, so the two
share one implementation.

**Interactive Components in the GUI:**
**Legend** — Tier **P** = primary, exposed by default. Tier **A** = advanced but
editable. Tier **D** = documented, adjustability deliberately not implemented (plot
cosmetics). Tier **—** = read-only, derived, or recorded but not acted on.

---

## Contents

- [Build notes](#build-notes) — Index B, display names, I/O behaviour, closed gaps
- [What the app reads and writes](#what-the-app-reads-and-writes)
- [Session modes](#session-modes)
- [Tab 1 · Dataset](#tab-1--dataset)
- [Tab 2 · Preprocess](#tab-2--preprocess)
- [Tab 3 · ROIs](#tab-3--rois)
- [Tab 4 · Traces](#tab-4--traces)
- [Tab 5 · Spike inference](#tab-5--spike-inference)
- [Tab 6 · Analysis](#tab-6--analysis)
- [How a job runs](#how-a-job-runs)
- [Plotting](#plotting--documented-adjustability-not-implemented)
- [Remaining gaps](#remaining-gaps)
- [Scientific caveats](#scientific-caveats-that-travel-with-every-output)

---

## Build notes

Four things to know before reading the tab reference: **Index B is switched off**, what
controls are **called**, what the app **does with files**, and which of the original
spec's gaps are now **closed**.

### 1. Index B (the frequency-domain PCA) is disabled in this build

Tab 6 runs **Index A only** — the inferred-rate features. Everything belonging to Index B
(band-power and dominant-frequency features, its PCA, its dendrogram, its cut, the
rate-vs-frequency agreement / adjusted Rand index, and its two figures) is commented out
behind the marker `[Index B disabled]` in `spike_inference_gui.py`. `grep -n "\[Index B
disabled\]" spike_inference_gui.py` lists every block that has to be un-commented to bring
it back — 20-odd call sites, no deletions.

The library functions it used (`features.freq_features`, `features.describe_bands`,
`features.freq_group_summary`, `grouping.compare_partitions`) are **untouched and still
work**; only the GUI call sites are switched off. What disappears from the window and the
exports while it is off:

| Gone from the GUI | Gone from the exports |
|---|---|
| the frequency-band table and its Nyquist caption | `freq_features.csv` |
| the second k slider and `FREQ_CUT_HEIGHT` override | `pca_freq.npz` / `.json` / `pca_freq_scores.csv` |
| the Index A/Index B preview selector (hidden, single entry) | `figures/pca_freq_summary.tiff`, `figures/dendrogram_freq.tiff` |
| the FFT-bin-width note under the region slider | the `freq_group` column of `roi_groups.csv` |

Two consequences worth stating outright. `CASCADE_DIR` is **no longer required to run the
analysis** — it was demanded only because `freq_features` loads its FFT helper from
`CascadeTorch/scripts`, and Index A never needed it. And `analysis_params.json` records
`"index_b_frequency_pca": "disabled in this build"` rather than a band list, so an export
from this build can be told apart from one made with both indices.

### 2. Display names

Tabs 1 and 2 dropped the notebook's `SHOUTING_CONSTANT` style for readable labels; tabs
3–6 kept it. The underlying variable names in `analysis_tools/` are unchanged, so scripts
and `.params.json` sidecars still use the right-hand column.

| Tab | GUI label | Underlying name | Note |
|---|---|---|---|
| 1 | **Raw Data Directory** | `DATA_DIR` | |
| 1 | **Use Channel...** | `channel` | Was hard-coded `Ch0`; now a parameter. |
| 1 | **Fallback Freq. (Hz)** | `fallback_hz` | |
| 2 | **Denoise Method** | `DENOISE_METHOD` | Checkboxes, not a text field. |
| 2 | **[Gaussian] Sigma** | `DENOISE_SIGMA` | Bracket prefix marks which denoiser the knob belongs to — it does nothing unless that method is in the chain. |
| 2 | **[NLM] Patch Size** | `NLM_PATCH_SIZE` | |
| 2 | **[NLM] Patch Distance** | `NLM_PATCH_DISTANCE` | |
| 2 | **[NLM] H-Factor** | `NLM_H_FACTOR` | |
| 2 | **Preprocessed Data Directory** | `PREPROCESSED_DIR` | |
| 3 | **Labels Directory** | `LABELS_DIR` | |
| 4 | **source .zarr** | — | New; no notebook equivalent. See [Tab 4](#tab-4--traces). |
| 4–6 | unchanged | `START_TIMEPOINT`, `BASELINE_SLIDES`, `ACTIVE_THRESH`, `PCA_VAR_TARGET`, … | |

Explanatory notes live behind the **ⓘ** badges rather than as permanent prose under each
control. Hovering holds the note open for as long as the pointer stays on the badge.

**Three controls the spec listed are no longer in the GUI at all:**

| Control | Status | Value actually used |
|---|---|---|
| `DCT_THRESHOLD_FRACTION` | deprecated with the DCT denoiser | — |
| `BACKGROUND_PERCENTILE` | commented out of the panel | `4` — the `PreprocessParams` default |
| `upsample_factor` | commented out of the panel | `10` — the `PreprocessParams` default |

The last two still act on every frame; they are simply no longer adjustable from the
window. They remain fields of `preprocess.PreprocessParams` and are still written into
each store's `.params.json`, so a run stays reproducible. Uncomment the three lines in
`_build_preprocess_tab` and the matching lines in `_preprocess_params` to bring them back.
The DCT denoiser itself is commented out of `analysis_tools/preprocess.py`, so
`DENOISE_METHODS` is `("gaussian", "nlm")`.

`ROI_MASK_PATH` also went away as a *field*. The path is now internal state set by the
load/save dialogs — see [Tab 3](#tab-3--rois).

### 3. I/O behaviour

**Outputs go where you say, per run — not to a derived folder.** This is the substantive
change. Previously the analysis root plus `dataset_tag` determined every output path,
which meant every run of a dataset wrote the same filenames to the same place: a second
inference or a second analysis pass silently replaced the first.

| Action | Then | Now |
|---|---|---|
| Run CASCADE | wrote `spike_rate.npy` to `<root>/<tag>/` as a side effect | writes nothing; **asks** whether to save, then asks **where** |
| Save inference | — (implicit) | own button, own folder dialog, refuses to clobber without confirming |
| Load inference | read `<root>/<tag>/` | asks which folder to read |
| Run analysis | wrote nine files to `<root>/<tag>/` | asks **once per session** for a parent folder, then writes an **auto-named subfolder per run** |

`ANALYSIS_DIR` still exists, in tab 4, and still governs `save_traces` / `load_traces`.
For inference and analysis it is only where the folder dialog opens.

**Other file-level changes:**

- **`spike_rate.csv`** is written alongside `spike_rate.npy`. Same numbers, transposed to
  the `traces.csv` layout (`time_s`, then one column per ROI) so the inferred rate lines
  up row-for-row with the dF/F it came from. CASCADE's NaN pad frames stay empty cells —
  not zeros, which would read as "silent" rather than "not estimated". If the time axis
  and the rate disagree on length, a `frame` index column is written instead of `time_s`
  and the mismatch is logged.
- **`analysis_params.json`** is written next to every analysis run: the window mode and
  its frames/seconds, the phases, the frame rate, the pad count, the threshold **and where
  it came from**, the PCA and cut settings, and the model name.
- **Trace extraction reads a named `.zarr` store**, not whatever is in the viewer. A
  *source .zarr* field in tab 4 selects it. Extracting from the in-session lazy chain
  would re-run denoising and motion correction per frame — work already paid for — and
  would leave the traces' provenance as "whatever was loaded".
- **Building a preprocessing chain prompts to save it.** A built chain is lazy: nothing is
  computed until something asks. Since extraction reads from a store on disk, an unsaved
  chain means paying the whole preprocessing cost again later.
- **Saving a Zarr store adopts it.** After a save the session *reads the file it just
  wrote*, so nothing downstream re-runs the chain. A guard refuses to re-save an adopted
  store onto itself, which would delete the source mid-write and zero the contents.

### 4. Gaps the spec listed that are now closed

- **Compute kernels are lifted.** `analysis_tools/preprocess.py` (denoise chain,
  background normalisation, motion correction) and `analysis_tools/traces.py` (ROI
  extraction, F0/dF/F) exist and are what both the notebook and the GUI call.
- **The `START_TIMEPOINT` bug is fixed.** The notebook allocated a per-ROI array of length
  `min(n_frames, start + count)` but wrote at absolute index `t`, so any non-zero start
  misaligned or overran the trace. `traces.extract_roi_traces` writes at an index relative
  to the window.
- **Long operations run off the UI thread** on a `QThreadPool`, with a progress bar and
  stdout mirrored into the log pane.
- **napari opens alongside the window** rather than being driven by an IPython `%gui`
  magic.
- **Models can be downloaded from inside the app** — tab 5 carries a searchable picker
  over the 156-entry index. The repository ships **no weights**, only
  `Pretrained_models/available_models_CascadeTorch.yaml`.
- **The grouping can be previewed before anything is written**, and k dragged without
  recomputation — see [Tab 6](#tab-6--analysis).

---

## What the app reads and writes

Every file the app touches, which tab touches it, and the function behind it. All live in
`analysis_tools/store.py` unless noted.

| Data | Tab | Direction | Format | Function |
|---|---|---|---|---|
| Frame images from an `.imgdir` | 1 | import | `ImageData_Ch<n>_TP*.npy`, one file per frame **or** one holding the whole stack | `list_frames`, `frame_layout`, `frame_index`, `load_frame`, `load_imgdir` |
| Acquisition timebase | 1 | import | `ElapsedTimes.yaml` | `load_timebase` |
| Preprocessed stack | 2 | import + export | `.zarr` + `.params.json` | `save_preprocessed`, `load_preprocessed` |
| ROI labels | 3 | import + export | `.tiff` uint16 label image | `load_roi_labels`, `save_roi_labels` |
| dF/F traces, many ROIs | 4 | import | `.csv` / `.npz` / `.npy` | `load_traces`, `import_dff` |
| ROI trace export | 4 | import + export | `.txt` (tab-separated) → `.csv` | `convert_roi_traces_txt`, called by `import_dff` |
| dF/F traces | 4 | export | `.npz` + `.csv` + `.json` | `save_traces` |
| Pretrained model | 5 | import | `.zip` → model folder | `cascade_runner.model_index`, `download_model` |
| Inferred spike rate | 5 | import + export | `.npy` + `.csv` + `.json` | `cascade_runner.save` / `.load`, `_save_spike_rate_csv` † |
| Features + groups + PCA | 6 | export | `.csv` tables, `.npz` + `.json` | `save_features`, `save_groups`, `save_pca` |
| Group mask | 6 | export | `.tiff` uint16 label image | `save_roi_labels` (via `_export_group_mask` †) |
| Group overlay | 6 | export | `.tiff` RGBA, tab10 | `save_rgba_overlay` (via `_export_group_mask` †) |
| Run parameters | 6 | export | `analysis_params.json` | `_save_analysis_params` † |
| Figures | 6 | export | `.tiff` under `figures/` | `save_figure` |

† defined in `spike_inference_gui.py`, not in `store.py`.

`store.available(dir)` reports which of these artifacts exist on disk. The GUI does not
currently use it — it gates on live session state instead — but it is there for scripts
that want to tell an empty directory from a half-finished run.

---

## Session modes

Two entry points, and which one is active determines what is meaningful to show.

### Full mode — Import @ Tab 1

All tabs live. What each stage needs before its buttons enable:

| Action | Enabled once |
|---|---|
| Load dataset | a path is in **Raw Data Directory** |
| Build lazy stack | frames are loaded **and** a preprocessed directory is set |
| Save to Zarr… | a stack has been built or loaded |
| Load from Zarr | always (a `.zarr` is a complete entry point) |
| New / Load ROI labels | a stack or a known frame shape exists |
| Save ROI labels | labels exist in the session or in napari |
| Extract traces | ROI labels exist **and** a source `.zarr` is named |
| Save / Load traces | traces exist (save) and an analysis root is set |
| Download models | `CASCADE_DIR` is set — no traces or ROIs needed |
| Run CASCADE | traces exist **and** a model is selected |
| Save inferred spikes… | CASCADE has been run **this session** (a rate loaded from disk is already saved) |
| Load saved inference… | traces exist (tab 6 reads dF/F alongside the rate) |
| Preview / Run analysis | an inferred rate is in the session |

Tabs 1–3 are enabled only in full mode; tabs 4 and 5 are always enabled; tab 6 unlocks
when a spike rate is present. **Tab 5 stays enabled even with nothing loaded** — an empty
`Pretrained_models/` has to be fillable on a fresh clone, and a disabled tab disables its
downloader. Tab availability never tracks busy-ness; running jobs disable the action
buttons instead, because disabling the current tab makes Qt jump focus to another one.

### Traces-only mode — Import @ Tab 4

Importing dF/F directly means there is no pixel data behind the traces, so upstream stages
are not stale, they are *absent*.

**Disabled:** tabs 1–2 entirely — dataset/image import and preprocessing — plus the
extraction window in tab 4 (`START_TIMEPOINT`, `EXTRACT_TIMEPOINTS`, `BASELINE_SLIDES`;
imported traces are already baselined) and **New blank ROI labels**, which has no frame
shape to size itself against. Napari layers are cleared on import.

**Live:** ROI label import · spike inference · phases and regions · features · PCA ·
grouping · every export.

**Tab 3 stays open**, for load and save only. An imported trace table carries no geometry,
so a label image is the only way the pixel location of each ROI re-enters the session — and
without one, tab 6 cannot paint its group mask. There is no stack to check the labels
against here, so the ROI **ids** are cross-checked against the traces on load instead, and
any that appear in only one of the two are named.

**The one thing that must be got right:** frame rate. There is no `ElapsedTimes.yaml` in
this mode, so it comes from a time column in the imported file or from manual entry. It
drives CASCADE resampling and the phase windows — a wrong value produces plausible-looking
output that is wrong throughout. `import_dff` refuses to guess: no time column and no
`frame_rate_hz` raises.

Switching modes clears downstream state rather than mixing provenance. The mode is
recorded in every params sidecar written afterwards.

---

## Tab 1 · Dataset

| GUI label | Tier | Widget | Default | Notes |
|---|---|---|---|---|
| **Raw Data Directory** | P | directory picker | *(empty)* | The `.imgdir` frame dump. |
| **Use Channel...** | P | int spin (0–8) | `0` | Selects `ImageData_Ch<n>_TP*.npy`. |
| **Fallback Freq. (Hz)** | A | float | `10.0` | Used **only** when `ElapsedTimes.yaml` is missing or its length disagrees with the stack. The log says which was used. Also supplies the nominal axis when traces are extracted from a `.zarr` opened without its `.imgdir` — see [Tab 4](#tab-4--traces). |
| frame count · shape · dtype | — | read-only | derived | Observed `(1024, 1376)` uint16. |
| time source | — | read-only | derived | `ElapsedTimes.yaml` or `fallback`. |
| `dataset_tag` | — | read-only | `DATA_DIR.name.split("-")[0]` → `Debi_DRG` | Every downstream filename keys off this. |

Action: **Load dataset**. The stack is a lazy dask array, one frame per chunk — nothing is
read beyond the first frame until something computes.

Every path field starts empty. A blank field resolves to `None`, never to the working
directory — an accidental `Path("")` would silently target the repo root.

---

## Tab 2 · Preprocess

| GUI label | Tier | Widget | Default | Range / choices |
|---|---|---|---|---|
| **Denoise Method** | P | checkboxes | `nlm` | Any subset of `gaussian`, `nlm`, applied **in sequence** in that fixed order; stored comma-separated (`"gaussian,nlm"`). |
| **[Gaussian] Sigma** | P | float | `3.0` | px. Ignored unless `gaussian` is in the chain. |
| **[NLM] Patch Size** | P | int | `5` | px. |
| **[NLM] Patch Distance** | P | int | `6` | px search radius. |
| **[NLM] H-Factor** | P | float | `1.25` | Cut-off, multiplied by the per-frame `estimate_sigma`. |
| **Preprocessed Data Directory** | A | directory picker | *(empty)* | Zarr cache location. |
| resolved path | — | read-only | derived | `<dataset_tag>_<method_tag>.zarr`, which chains are already cached, or — when the path cannot be resolved — exactly which of the three prerequisites is missing. |

Not exposed, see [display names](#2-display-names): `BACKGROUND_PERCENTILE` (`4`),
`upsample_factor` (`10`), `DCT_THRESHOLD_FRACTION` (deprecated with the DCT denoiser).

### What the chain does, per frame

`preprocess.preprocess_frame`, in order: denoise (the checked methods in sequence) →
subtract the 4th-percentile background → rigid motion correction against a fixed reference
(`phase_cross_correlation`, `upsample_factor=10`, sub-pixel shift) → clip at zero, float32.
The reference is frame 0 put through the same denoise + background steps, computed once.

**The trap this panel is built around:** the output filename embeds the denoise chain.
Changing the method silently retargets which store is saved *and* loaded — switch the
denoiser and press load, and you get a different preprocessing run, or a "not found" for
work you believe you did. The resolved path sits next to the control for that reason.

Actions: **Build lazy stack** (wires the chain, computes nothing, then offers to save) ·
**Save to Zarr…** (computes every frame; confirms before overwriting) · **Load from Zarr**
(a complete entry point — available without loading a dataset first; the tag and the
denoise chain are read back out of the store's `<dataset_tag>_<method>.zarr` name and the
checkboxes are re-pointed at what the store was actually built with).

---

## Tab 3 · ROIs

| GUI label | Tier | Widget | Default | Notes |
|---|---|---|---|---|
| **Labels Directory** | A | directory picker | *(empty)* | **Only** sets where the load/save dialogs open. |
| label dtype | — | fixed | `uint16` | |
| overlay opacity | D | — | `0.6` | Hard-coded in `_add_labels_layer`. |

The ROI label path is **not** a field. Load and save each open their own dialog and record
what the dialog returned, so the recorded path always reflects the file actually used —
never a stale field someone edited by hand. It is written into `traces_params.json`. The
dialogs suggest `<dataset_tag>_ROI.tiff` and reopen at the last file used.

Actions: **New blank ROI labels** (sized to the preprocessed stack) · **Load ROI labels**
(raises on a shape mismatch against the stack — shown as a dialog, since the alternative
is silently wrong traces) · **Save ROI labels**.

Both saving and loading leave the labels live in the session; there is no second click to
"load" what you just saved. Paint on the *ROI labels* layer in napari with the brush;
press <kbd>M</kbd> for a fresh label id per ROI.

**The ROI count is never stored.** It is derived from the current label image everywhere
it appears — the tab 3 readout reads the live napari layer, so a painted ROI is counted
without saving first, and extraction derives `roi_ids` the same way (`traces.roi_ids_in`,
which counts non-zero labels rather than `len(unique) - 1`, so a label image with no
background pixel is not undercounted). Note that importing a new label set does **not**
invalidate traces already extracted from an older one; re-run extraction after changing
labels.

In traces-only mode this tab stays usable: **Load ROI labels** and **Save ROI labels**
work, **New blank ROI labels** does not (there is no frame shape to size a canvas
against), and the shape check is replaced by the id cross-check described under
[session modes](#traces-only-mode--import--tab-4).

This build renders no max/STD projection.

---

## Tab 4 · Traces

| GUI label | Tier | Widget | Default | Notes |
|---|---|---|---|---|
| **source .zarr** | P | directory picker | derived from tab 2 | The store traces are extracted from. Filled in automatically when a store is saved or loaded; override to extract from a different preprocessing run. |
| `START_TIMEPOINT` | P | int | `0` | First frame of the window. |
| `EXTRACT_TIMEPOINTS` | P | int | `500` | Frames to extract (≈50 s at 10 Hz). |
| `BASELINE_SLIDES` | P | int | `20` | Leading frames averaged into F0 (≈2 s). Interacts with the `pre` phase — see tab 6. |
| Analysis root | A | directory picker | *(empty)* | `<root>/<dataset_tag>/` for `save_traces` / `load_traces`. Also where the tab 5/6 output dialogs open. |
| import file | P | file picker | — | Traces-only entry. `.csv` / `.npz` / `.npy`. |
| `orientation` | P | dropdown | `auto` | `auto` \| `roi_rows` \| `roi_cols`. Auto reads a table with a time column as one column per ROI; otherwise the longer axis is time. |
| `time_column` | A | text | `auto` | Column name or index, `auto`, or `none`. Auto accepts a column named time/t/seconds, or a strictly increasing numeric first column in a taller-than-wide table — a `roi` id column is explicitly not mistaken for one. |
| `frame_rate_hz` | P *(traces-only)* | float | `0.0` = unset | **Required** when the file carries no time axis. |

### How the traces are computed

Three stages, in `analysis_tools/traces.py`:

1. **Spatial mean** — for ROI *r* with pixel set `P_r = {(y,x) : L(y,x) = r}` from the
   label image `L`: `F_r(t) = (1/|P_r|) · Σ I(t,y,x)`. Unweighted, no neuropil
   subtraction. Frames are materialised one at a time, so the movie is never in RAM.
2. **Baseline** — `F0_r = mean(F_r[0 : BASELINE_SLIDES])`. A fixed leading window, not a
   rolling percentile; it assumes the recording starts quiet.
3. **Normalisation** — `ΔF/F_r(t) = (F_r(t) − F0_r) / F0_r`.

`F` here is the mean over the **preprocessed** stack, so each pixel has already been
denoised, background-subtracted at the 4th percentile, motion-corrected and clipped at
zero. That subtraction shrinks `F0` and therefore inflates ΔF/F relative to the same
computation on raw camera counts — and it is why `F0 = 0` is a live failure mode, guarded
explicitly. Both `F` and ΔF/F are kept and saved.

Extraction re-checks the label shape against the store it is reading and refuses a
mismatch. If the store was opened without its `.imgdir` there is no acquisition timebase,
so the time axis is nominal at the tab 1 **Fallback Freq.** — said out loud in the log,
because that number feeds CASCADE resampling.

Export writes three files via `store.save_traces`: `traces.npz` (exact, including the raw
pre-dF/F traces), `traces.csv` (portable — `time_s` plus one column per ROI), and
`traces_params.json` (mode, dataset, source zarr, ROI-label path, the three window
options, frame rate, ROI/frame counts).

---

## Tab 5 · Spike inference

| GUI label | Tier | Widget | Default | Notes |
|---|---|---|---|---|
| `CASCADE_DIR` | A | directory picker | *(empty)* | e.g. `./CascadeTorch`. Models are read from `<CASCADE_DIR>/Pretrained_models`. |
| `MODEL_NAME` | P | dropdown | `Spinal_cord_excitatory_30Hz_smoothing50ms` *(if installed)* | From `cascade_runner.available_models()` — a folder is listed only once its `config.yaml` is there, so an interrupted download is not offered. |

The panel shows what `cascade_runner.check_model()` prints, inline and before the run:
training rate parsed from the **name** vs the rate in `config.yaml` (at least one shipped
model disagrees), the training datasets, the smoothing and window size, the resample plan
in both directions, an explicit warning that upsampling adds no information, an indicator
mismatch warning (GCaMP6 model vs GCaMP8 recording), and the pad frames that will be NaN
at each edge.

Model choice is a scientific claim, not a preference. The default is a spinal dorsal-horn
model applied to DRG — the nearest available analogue, and an untested transfer. No
DRG-specific ground truth exists. The panel carries that caveat where the user picks, not
only in the writeup.

### Downloading models

The repository ships **no weights** — only the index,
`Pretrained_models/available_models_CascadeTorch.yaml` (156 entries). **Download
pretrained models…** opens a picker over that index: filter by name or family, hide what
is installed, and sort by the columns *Model · Installed · Family · Rate (Hz) · Smoothing
(ms) · Noise*. The rate and smoothing are parsed out of the name, because the property
that decides whether a model suits a recording is buried mid-string. Nothing is fetched to
build the list. Models on disk but absent from the index (hand-placed or retrained
folders) are named at the bottom rather than left to make the two counts disagree.

The download stages into a temp folder and replaces an existing copy **only after** the
new one has extracted cleanly and its `config.yaml` is present, so a failed re-download
cannot destroy a working model. The newly installed model is then selected in `MODEL_NAME`.

### Inference

`cascade_runner.run` measures noise levels at the **native** frame rate (the metric is a
median frame-to-frame difference, and interpolated frames are correlated by construction),
resamples the traces to the model rate with polyphase `resample_poly` — NaN-preserving —
runs `cascade.predict` on CPU with `padding=np.nan`, converts spikes/frame to spikes/s by
multiplying by the grid rate **before** resampling back, and returns to the acquisition
grid.

Actions: **Run CASCADE** · **Save inferred spikes…** · **Load saved inference…**

Running writes nothing by itself. It offers to save, and saving asks for a destination
folder rather than deriving one — see [I/O behaviour](#3-io-behaviour). Declining keeps the
rate in the session with the Save button live; it is lost only when the window closes.
Loading asks for a folder too, otherwise a run saved outside the derived path could never
be read back, and it clears the in-memory result so a rate that is already on disk is not
duplicated. Adopting a rate whose ROI count disagrees with the loaded traces is refused.

Written on save: `spike_rate.npy`, `spike_rate.csv`, `spike_inference_params.json`.

---

## Tab 6 · Analysis

Split into two sub-tabs: **Setup** holds the controls, **Live preview** the canvases.

### Analysis window — phases *or* one region, never both on screen

A pair of radio buttons swaps the panel below them.

| GUI label | Tier | Widget | Default | Notes |
|---|---|---|---|---|
| **Chop up into phases** | P | editable table, name → (start, end) s | `pre (2,10)`, `injury (10,15)`, `post (15,50)` | Rows can be added and removed. `pre` starts at 2.0 s, not 0, because frames `0..BASELINE_SLIDES` *defined* F0 — dF/F there is ~0 by construction and would fake a silent baseline. The **first** phase is also the `ACTIVE_THRESH` baseline. |
| **Select a region** | P | range slider + two frame spins | whole recording | The analysis runs on this window only, as a single phase named `region_<t0>-<t1>s`. Until a handle is dragged the region tracks the entire recording. |
| region readout | — | read-only | derived | Frames, seconds and duration of the current region. |

In region mode everything is cropped to the selected frames and the NaN **pad count is
recounted on the crop** — `session.pad` describes CASCADE's pad at the ends of the *full*
trace, and a crop may exclude it entirely or land inside it. And `ACTIVE_THRESH` is
measured **before** the region, not inside it: a threshold taken from the window under test
is set by the very response it is meant to detect. A region starting at frame 0 has nothing
before it, so the threshold falls back to the region itself — which is circular, and said
so in the log.

### Thresholds, PCA and clustering

| GUI label | Tier | Widget | Default | Notes |
|---|---|---|---|---|
| `ACTIVE_THRESH` | P | checkbox + float | auto | Auto = baseline median + `ACTIVE_N_SIGMA` × robust (MAD) sigma. Unticking enables the manual spikes/s box. |
| `ACTIVE_N_SIGMA` | P | float | `3.0` | A fixed absolute threshold does not transfer between models. |
| `PCA_VAR_TARGET` | P | float 0.05–1 | `0.95` | Cumulative variance target selecting the PC count. |
| `MAX_K` | A | int | `12` | Caps the k slider and the printed candidate-cut table. |
| `MIN_GROUP_SIZE` | — | int | `3` | **Recorded into `analysis_params.json`, but not acted on in this build** — it is a parameter of `grouping.suggest_cut`, which the GUI no longer calls now that the cut is set by k or by an explicit height. |
| `MAX_GROUP_FRAC` | A | float | `0.9` | One group holding more than this share triggers the degenerate-partition warning. |
| **Index A groups (k)** | P | slider 2–`MAX_K` | `4` | Cuts the tree with `fcluster(maxclust)`. The equivalent height is shown beside the slider and written to `analysis_params.json`, so k-specified runs stay comparable with height-specified ones. |
| `RATE_CUT_HEIGHT` override | P | text, blank = use k | blank | Pins a height; takes precedence over k **in the export**. The live preview always uses k. |

### Live preview

**Preview (no files written)** computes the features, the PCA and the linkage — everything
*except* the cut — and draws three panels: the dendrogram, the ROIs in PC space coloured by
group, and the group-mean inferred rate with each member ROI behind it. Dragging k then
re-cuts the stored tree and redraws, with no recomputation (debounced to one redraw per
120 ms, since each costs ~100–200 ms).

Between the scatter and the traces sits an explainer stating what the PCA actually
measures — three numbers per ROI per phase, z-scored, distances Euclidean in the leading
PCs — which features PC1 and PC2 load on for *this* fit, and the threshold with its
provenance. The grouping is easy to over-read: the distances are between per-phase summary
statistics, not between traces.

Changing the phases, the region or `MAX_K` clears the preview and says so, rather than
leaving figures up for a window you are no longer looking at. See
[Remaining gaps](#remaining-gaps) for the settings that do **not** clear it.

### Export

**Run analysis and export (CSV + TIFF)** asks once per session for a **parent** folder,
then writes an auto-named subfolder per run: `<window>_<cut>`, e.g.
`phases_pre-injury-post_k4` or `region_12.0-30.0s_h1.85`. An existing non-empty folder of
the same name asks before it is overwritten, so two different configurations never
collide.

Into that folder:

- `rate_features.csv` — per-ROI mean rate, peak rate and active fraction per phase, joined
  with the descriptive dF/F summary and the group column (`store.save_features`)
- `roi_groups.csv` — `roi` → `rate_group` (`store.save_groups`)
- `roi_group_mask.tiff` — the grouping painted back into image space: a uint16
  label image carrying each ROI's group number (1..k) over its pixels, 0 elsewhere.
  Values, not colours — read it to count pixels or to re-colour it yourself
- `roi_group_overlay.tiff` — the same grouping as a picture: RGBA, each ROI in the
  **tab10** colour its group has in the trace panels and the PC-space scatter, background
  fully transparent. Lay it over a max projection of the recording and a group reads as
  the same colour there as in every figure of the run

  Both are skipped, with a line in the log, when no ROI label image is loaded — which
  pixels an ROI occupies cannot be recovered from the traces alone. Load the matching
  labels in tab 3 (available in traces-only mode too) and re-run
- `pca_rate.npz` + `pca_rate.json` + `pca_rate_scores.csv` (`store.save_pca`)
- `analysis_params.json` — everything needed to reproduce the run
- `figures/*.tiff` (`store.save_figure`): `pca_rate_summary`, `dendrogram_rate`,
  `rate_heatmap`, `rate_group_means`

The log also carries the candidate-cut table (k, height, merge gap, group sizes), the
per-group phase profile with its `dominant_phase` column, and the ROI membership of every
group.

`save_pca` stores scores, loadings, explained variance and the linkage matrix — enough to
re-plot and re-cut — but **not** the `StandardScaler` statistics, so it cannot project new
ROIs into a saved PCA space. The PCA is per-dataset; there is nothing to project.

Group labels are an arbitrary integer labelling, not a cell type. `dominant_phase` is an
argmax over three means — with a single trial it ranks, it does not test.

---

## How a job runs

Every long operation goes through one path, so the behaviour is the same everywhere.

- One job at a time, on a `QThreadPool` capped to a single thread — stdout capture is
  global, so a second concurrent job would interleave its log lines into the first.
- The job's stdout is redirected into the log pane, which is why `analysis_tools`
  functions communicate by `print` and everything they say ends up on screen.
- `progress` callbacks drive the progress bar (trace extraction, model download).
- On success the completion handler runs back on the UI thread; on failure the traceback
  goes to the log and its last line into a dialog. Either way the action buttons re-enable.
- Figures are drawn on the UI thread after the numeric work finishes. `analysis_tools.plots`
  builds bare `Figure` objects and never touches pyplot, so nothing leaks into pyplot's
  global figure manager or drags its thread affinity into the Qt event loop.

---

## Plotting — documented, adjustability not implemented

Listed so the eventual controls are known. These are currently constants in the plot
functions (`analysis_tools/plots.py`) or in `store.save_figure`.

| Option | Current value | Where |
|---|---|---|
| inferred-rate heatmap colormap | `magma` | `plots.rate_heatmap` |
| heatmap colour limits | `vmin=0`, `vmax=nanpercentile(99)` | `plots.rate_heatmap` |
| group colours | **tab10** (Tableau 10), cycled — colours repeat past 10 groups | `plots.GROUP_COLORS` |
| figure dpi / format | `200`, TIFF, `bbox_inches="tight"` | `store.save_figure` |
| figsize rules | `(16, 4.4)` pca_summary · `(6.4, 4.6)` pca_scatter · `(13, 4.6)` dendrogram · `(12, 0.18·n_rois + 2.2)` heatmap · `(12, 1.9·n_groups + 1.2)` group means | `plots.*` |
| `SHOW_ROI_TRACES` | `True` | member ROIs behind group means |
| `ROI_TRACE_ALPHA` | `0.3`, linewidth `0.5` | `plots.group_means` |
| group-mean linewidth / SEM alpha | `1.2` / `0.35` | `plots.group_means` |
| stimulus shading | grey `axvspan` (group means), white dashed edges (heatmap) | `plots.group_means`, `plots.rate_heatmap` |
| napari ROI overlay opacity | `0.6` | `_add_labels_layer` |
| preview canvas heights | `4.6" / 4.6" / 6.0"` at 100 dpi | `_build_preview_page` |

The shaded stimulus band is the phase named `stim`; with no such phase it is the **second**
phase in the table, or the only one in region mode. The default phase names in this build
are `pre / injury / post`, so the shaded band is `injury` unless a phase is renamed.

**Group colour is a single source of truth.** `plots.GROUP_COLORS` is pinned to
matplotlib's `tab10` rather than written as the `C0`–`C9` property-cycle shorthand. The
default cycle *is* tab10, so the two agree today — but the cycle follows `rcParams`, and a
style sheet or a `seaborn` import would repaint every figure while the exported
`roi_group_overlay.tiff`, which resolves its colours from the same list via
`plots.group_rgb`, kept the old ones. Everything that shows a group — dendrogram, PC-space
scatter, group-mean traces, exported overlay — reads through `plots.group_color`, which is
what guarantees G3 is the same green in a TIFF as it is in the trace panel. Past 10 groups
the palette repeats: the panels still agree with each other, but two groups share a colour.

Constraints that are **not** preferences and stay fixed even once this panel is built:
perceptually-uniform colormaps only, never `jet`; raw and inferred traces in separate
subplot rows or visually distinct; every panel showing inferred spikes names the algorithm
and its parameters in that same panel.

---

## Remaining gaps

1. **Index B is disabled** — see [build notes](#1-index-b-the-frequency-domain-pca-is-disabled-in-this-build). The code is commented, not deleted.
2. **Not every setting invalidates the live preview.** The phases, the region and `MAX_K`
   clear it; `PCA_VAR_TARGET`, `ACTIVE_N_SIGMA`, the manual `ACTIVE_THRESH` and the
   auto/manual toggle do **not** — the preview keeps showing the fit it was computed with
   until Preview is pressed again. (`MAX_GROUP_FRAC` is read live at each re-cut, and the
   `RATE_CUT_HEIGHT` override is ignored by the preview entirely.)
3. **`MIN_GROUP_SIZE` is inert** — recorded in `analysis_params.json`, but nothing reads it
   now that `grouping.suggest_cut` is no longer called.
4. **Napari runs as a separate top-level window**, not embedded in the Qt app. Workable,
   but the two windows can be lost behind each other.
5. **Jobs are not cancellable.** They run off the UI thread with a progress bar, but a
   long Zarr write has to be waited out.
6. **New ROI labels do not invalidate existing traces.** Importing a label set after
   extracting leaves the old `roi_ids` and dF/F in the session, and tabs 5–6 stay
   enabled. Re-extract after changing labels.
7. **No projection rendering.** The max/STD projection layers the notebook used for ROI
   reference are not built here; ROI labels are sized against the preprocessed stack.
8. **Plot adjustability** — the table above.

---

## Scientific caveats that travel with every output

- CASCADE's ground-truth corpus is **entirely CNS** (zebrafish telencephalon, CA3,
  neocortex). No DRG or DCN training set exists. The dorsal-horn model used here is the
  nearest analogue and the transfer is untested — state this wherever the output is
  quoted.
- Default CASCADE models are GCaMP6-tuned; applied to GCaMP8 they misestimate rates at
  both ends of the range.
- **ΔF/F amplitude is not a rate proxy.** The `dff_peak_*` columns in the exports are
  descriptive only and feed no PCA.
- Upsampling a recording to a faster model's rate adds **no** information; true resolution
  stays at the acquisition rate whatever the model label says.
- For DRG the honest deliverable is **burst / relative rate, not absolute spike counts**.

## AI Declaration
Scripts and documentation are created with the help of Claude Code and Codex, prompted by Bo-Yu Chen and Sun-Hsing Ho.

Contact @leolopster on Telegram for any enquiries.

---
---


[![DOI](https://zenodo.org/badge/241174650.svg)](https://zenodo.org/badge/latestdoi/241174650)
[![License](https://img.shields.io/badge/License-GPL--3.0-brightgreen)](https://github.com/PTRRupprecht/CascadeTorch/tree/master/LICENSE)
[![Size](https://img.shields.io/github/repo-size/PTRRupprecht/CascadeTorch?style=plastic)](https://img.shields.io/github/repo-size/PTRRupprecht/CascadeTorch?style=plastic)
[![Language](https://img.shields.io/github/languages/top/PTRRupprecht/CascadeTorch?style=plastic)](https://github.com/PTRRupprecht/CascadeTorch)

## CascadeTorch: Calibrated spike inference from calcium imaging data using PyTorch

<!---![Concept of supervised inference of spiking activity from calcium imaging data using deep networks](https://github.com/PTRRupprecht/CascadeTorch/tree/master/etc/Figure%20concept.png)--->
<p align="center"><img src="https://github.com/PTRRupprecht/CascadeTorch/blob/master/etc/CA1_deconvolution_CASCADE.gif "  width="75%"></p>

*Cascade* translates calcium imaging ΔF/F traces into spiking probabilities or discrete spikes.

*Cascade* is described in detail in **[the main paper](https://www.nature.com/articles/s41593-021-00895-5)**. There are follow-up papers which describe the application of Cascade to **[spinal cord data](https://www.biorxiv.org/content/10.1101/2024.07.17.603957)** and the application of Cascade to **[GCaMP8](https://www.biorxiv.org/content/10.1101/2025.03.03.641129)**.

*Cascade's* toolbox consists of

- A large and continuously updated ground truth database spanning brain regions, calcium indicators, species
- A deep network that is trained to predict spike rates from calcium data
- Procedures to resample the training ground truth such that noise levels and frame rates of calcium recordings are matched
- A large set of pre-trained deep networks for various conditions (additional models upon request)
- Tools to quantify the out-of-dataset generalization for a given model and noise level
- A tool to transform inferred spike rates into discrete spikes

Get started quickly with the following *Colaboratory Notebook*:

- **[Spike inference from calcium data (Colaboratory Notebook)](https://colab.research.google.com/github/PTRRupprecht/CascadeTorch/blob/master/Demo%20scripts/Calibrated_spike_inference_with_Cascade.ipynb)**
- Upload your calcium data, use Cascade to process the data, download the inferred spike rates.
- Spike inference with Cascade improves the temporal resolution, denoises the recording and provides an absolute spike rate estimate.
- No parameter tuning, no installation required.
- You will get started within few minutes.

## Getting started

#### Without installation

If you want to try out the algorithm, just open **[this online Colaboratory Notebook](https://colab.research.google.com/github/PTRRupprecht/CascadeTorch/blob/master/Demo%20scripts/Calibrated_spike_inference_with_Cascade.ipynb)**. With the Notebook, you can apply the algorithm to existing test datasets, or you can apply **pre-trained models** to **your own data**. No installation will be required since the entire algorithm runs in the cloud (Colaboratory Notebook hosted by Google servers; a Google account is required). The entire Notebook is designed to be used by researchers with little background in Python, but it is also the best starting point for experienced programmers. Try it out - within a couple of minutes, you can start using the algorithm!

#### With a local installation (Ubuntu/Windows)

If you want to modify the code, integrate the algorithm into your existing pipeline (e.g., with CaImAn or Suite2P), or train your own networks, you will need a local installation.

Although Cascade is based on deep networks, GPU support is not required. Model training runs smoothly on CPUs (although GPUs can speed up the process). Therefore, installation is much simpler than for typical deep learning toolboxes that depend on GPU-based processing.

Inference has been tested successfully with Torch versions between 2.4 and 2.9 on Colab, Ubuntu, and Windows. See `setup.py` for a full list of requirements, or navigate to the CascadeTorch folder in your environment and run: `pip install .`

Feedback about problems with configurations and operating systems (also positive feedback about working environments) is welcome. Please submit issues, e-mails, or pull requests.

### Updates, FAQs, further info:

Check the parent [CASCADE repository](https://github.com/HelmchenLabSoftware/Cascade). FAQs and updates are updated only there for simplicity.

### References


> Please cite as primary reference for Cascade:
>
> Rupprecht P, Carta S, Hoffmann A, Echizen M, Blot A, Kwan AC, Dan Y, Hofer SB, Kitamura K, Helmchen F\*, Friedrich RW\*, *[A database and deep learning toolbox for noise-optimized, generalized spike inference from calcium imaging](https://www.nature.com/articles/s41593-021-00895-5)*, Nature Neuroscience (2021).
> (\* = co-senior authors)
>
> And the following papers specific for models trained with GCaMP8 and spinal cord data, respectively:
>
> Rupprecht P, Rózsa M, Fang X, Svoboda K, Helmchen F. *[Spike inference from calcium imaging data acquired with GCaMP8 indicators](https://www.biorxiv.org/content/10.1101/2025.03.03.641129)*, bioRxiv (2025).
>
> Rupprecht P, Fan W, Sullivan S, Helmchen F, Sdrulla A. *[Spike rate inference from mouse spinal cord calcium imaging data](https://www.jneurosci.org/content/45/18/e1187242025)*, J Neuroscience (2025).
