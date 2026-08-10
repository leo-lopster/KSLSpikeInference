# GUI for KSLSpikeInference

**The upstream CascadeTorch README is attached at the end of this file** ([PTRRupprecht/CascadeTorch](https://github.com/PTRRupprecht/CascadeTorch), GPL-3.0), since `CascadeTorch/` is vendored into this repository. Everything below documents the local pipeline that *uses* it — the DRG/DCN spike-inference workflow in this
repository. It is not part of the upstream toolbox and will not be found in the
CascadeTorch or Cascade repositories.

**Interactive Components in the GUI:**
**Legend** — Tier **P** = primary, exposed by default. Tier **A** = advanced but
editable. Tier **D** = documented, adjustability deliberately not implemented (plot
cosmetics). Tier **—** = read-only or derived.

---

## Contents

- [Version 0.1 Notes](#version-01-notes) — display names, I/O behaviour, closed gaps
- [What the app reads and writes](#what-the-app-reads-and-writes)
- [Session modes](#session-modes)
- [Tab 1 · Dataset](#tab-1--dataset)
- [Tab 2 · Preprocess](#tab-2--preprocess)
- [Tab 3 · ROIs](#tab-3--rois)
- [Tab 4 · Traces](#tab-4--traces)
- [Tab 5 · Spike inference](#tab-5--spike-inference)
- [Tab 6 · Analysis](#tab-6--analysis)
- [Plotting](#plotting--documented-adjustability-not-implemented)
- [Remaining gaps](#remaining-gaps)

---

## Version 0.1 Notes

Three kinds of change: what controls are **called**, what the app **does with files**, and
which of the spec's open gaps are now **closed**.

### 1. Display names

Tabs 1 and 2 dropped the notebook's `SHOUTING_CONSTANT` style for readable labels; tabs
3–6 kept it. The underlying variable names in `analysis_tools/` are unchanged, so scripts
and `.params.json` sidecars still use the right-hand column.

| Tab | GUI label | Underlying name | Note |
|---|---|---|---|
| 1 | **Data Directory** | `DATA_DIR` | |
| 1 | **Calcium Indicator Channel** | `channel` | Was hard-coded `Ch0`; now a parameter. |
| 1 | **Fallback Freq. (Hz)** | `fallback_hz` | |
| 2 | **Denoise Method** | `DENOISE_METHOD` | Checkboxes, not a text field. |
| 2 | **[Gaussian] Sigma** | `DENOISE_SIGMA` | Bracket prefix marks which denoiser the knob belongs to — it does nothing unless that method is in the chain. |
| 2 | **[NLM] Patch Size** | `NLM_PATCH_SIZE` | |
| 2 | **[NLM] Patch Distance** | `NLM_PATCH_DISTANCE` | |
| 2 | **[NLM] H-Factor** | `NLM_H_FACTOR` | |
| 3–6 | unchanged | `START_TIMEPOINT`, `BASELINE_SLIDES`, `ACTIVE_THRESH`, `PCA_VAR_TARGET`, … | |

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

`ROI_MASK_PATH` also went away as a *field*. The path is now internal state set by the
load/save dialogs — see [Tab 3](#tab-3--rois).

### 2. I/O behaviour

**Outputs go where you say, per run — not to a derived folder.** This is the substantive
change. Previously the analysis root plus `dataset_tag` determined every output path,
which meant every run of a dataset wrote the same filenames to the same place: a second
inference or a second analysis pass silently replaced the first.

| Action | Then | Now |
|---|---|---|
| Run CASCADE | wrote `spike_rate.npy` to `<root>/<tag>/` as a side effect | writes nothing; **asks** whether to save, then asks **where** |
| Save inference | — (implicit) | own button, own folder dialog, refuses to clobber without confirming |
| Load inference | read `<root>/<tag>/` | asks which folder to read |
| Run analysis | wrote nine files to `<root>/<tag>/` | asks for the output folder **before** any compute |

`ANALYSIS_DIR` still exists, in tab 4, and still governs `save_traces` / `load_traces`.
For inference and analysis it is only where the folder dialog opens; the dialog's *new
folder* button is how runs are kept apart.

**Other file-level changes:**

- **`spike_rate.csv`** is written alongside `spike_rate.npy`. Same numbers, transposed to
  the `traces.csv` layout (`time_s`, then one column per ROI) so the inferred rate lines
  up row-for-row with the dF/F it came from. CASCADE's NaN pad frames stay empty cells —
  not zeros, which would read as "silent" rather than "not estimated". If the time axis
  and the rate disagree on length, a `frame` index column is written instead of `time_s`
  and the mismatch is logged.
- **Trace extraction reads a named `.zarr` store**, not whatever is in the viewer. A new
  *source .zarr* field in tab 4 selects it. Extracting from the in-session lazy chain
  would re-run denoising and motion correction per frame — work already paid for — and
  would leave the traces' provenance as "whatever was loaded".
- **Building a preprocessing chain prompts to save it.** A built chain is lazy: nothing is
  computed until something asks. Since extraction reads from a store on disk, an unsaved
  chain means paying the whole preprocessing cost again later.
- **`CASCADE_DIR` is validated before the analysis starts.** Index B loads its FFT helper
  from `CascadeTorch/scripts`; a missing one used to fail halfway, after the Index A CSVs
  were already written.

### 3. Gaps the spec listed that are now closed

- **Compute kernels are lifted.** `analysis_tools/preprocess.py` (denoise chain,
  background normalisation, motion correction) and `analysis_tools/traces.py` (ROI
  extraction, F0/dF/F) exist and are what both the notebook and the GUI call.
- **The `START_TIMEPOINT` bug is fixed.** The notebook allocated a per-ROI array of length
  `min(n_frames, start + count)` but wrote at absolute index `t`, so any non-zero start
  misaligned or overran the trace. `traces.extract_roi_traces` writes at an index relative
  to the window. Non-zero starts are covered by test.
- **Long operations run off the UI thread** on a `QThreadPool`, with a progress bar and
  stdout mirrored into the log pane.
- **napari opens alongside the window** rather than being driven by an IPython `%gui`
  magic.

---

## What the app reads and writes

Every file the app touches, which tab touches it, and the function behind it. All live in
`analysis_tools/store.py` unless noted.

| Data | Tab | Direction | Format | Function |
|---|---|---|---|---|
| Frame images from an `.imgdir` | 1 | import | `ImageData_Ch<n>_TP*.npy` | `list_frames`, `load_frame`, `load_imgdir`, `load_timebase` |
| Preprocessed stack | 2 | import + export | `.zarr` + `.params.json` | `save_preprocessed`, `load_preprocessed` |
| ROI labels | 3 | import + export | `.tiff` uint16 label image | `load_roi_labels`, `save_roi_labels` |
| dF/F traces, many ROIs | 4 | import | `.csv` / `.npz` / `.npy` | `load_traces`, `import_dff` |
| dF/F traces | 4 | export | `.npz` + `.csv` + `.json` | `save_traces` |
| Inferred spike rate | 5 | import + export | `.npy` + `.csv` + `.json` | `cascade_runner.save` / `.load`, `_save_spike_rate_csv` |
| PCA analysis + summary | 6 | export | `.csv` tables + `.tiff` figures | `save_pca`, `save_features`, `save_groups`, `save_figure` |

---

## Session modes

Two entry points, and which one is active determines what is meaningful to show.

### Full mode — Import @ Tab 1

All tabs live. Stages unlock in order as artifacts appear:

| Artifact | Unlocks |
|---|---|
| `.imgdir` selected | Preprocessing |
| `preprocessed/<tag>_<method>.zarr` | ROI panel |
| an ROI label `.tiff` | Trace extraction |
| traces in the session | Spike inference |
| an inferred rate in the session | Analysis |

`store.available(dir)` reports which artifacts exist on disk — use it to drive gating
rather than catching `FileNotFoundError`.

### Traces-only mode — Import @ Tab 4

Importing dF/F directly means there is no pixel data behind the traces, so upstream stages
are not stale, they are *absent*.

**Disabled:** tabs 1–3 entirely — dataset/image import, preprocessing, ROI labels — plus
the extraction window (`START_TIMEPOINT`, `EXTRACT_TIMEPOINTS`, `BASELINE_SLIDES`;
imported traces are already baselined). Napari layers are cleared.

**Live:** spike inference · phases · features · PCA · grouping · every export.

**The one thing that must be got right:** frame rate. There is no `ElapsedTimes.yaml` in
this mode, so it comes from a time column in the imported file or from manual entry. It
drives CASCADE resampling, the phase windows and the Nyquist ceiling on the frequency
bands — a wrong value produces plausible-looking output that is wrong throughout.
`import_dff` refuses to guess: no time column and no `frame_rate_hz` raises.

Switching modes clears downstream state rather than mixing provenance. The mode is
recorded in every params sidecar written afterwards.

---

## Tab 1 · Dataset

| GUI label | Tier | Widget | Default | Notes |
|---|---|---|---|---|
| **Data Directory** | P | directory picker | *(empty)* | The `.imgdir` frame dump. |
| **Calcium Indicator Channel** | P | int spin | `0` | Selects `ImageData_Ch<n>_TP*.npy`. |
| **Fallback Freq. (Hz)** | A | float | `10.0` | Used **only** when `ElapsedTimes.yaml` is missing or its length disagrees with the stack. The log says which was used. |
| frame count · shape · dtype | — | read-only | derived | Observed `(1024, 1376)` uint16. |
| time source | — | read-only | derived | `ElapsedTimes.yaml` or `fallback`. |
| `dataset_tag` | — | read-only | `DATA_DIR.name.split("-")[0]` → `Debi_DRG` | Every downstream filename keys off this. |

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
| `PREPROCESSED_DIR` | A | directory picker | *(empty)* | Zarr cache location. |
| resolved path | — | read-only | derived | `<dataset_tag>_<method_tag>.zarr`, plus which chains are already cached. |

Not exposed, see [display names](#1-display-names): `BACKGROUND_PERCENTILE` (`4`),
`upsample_factor` (`10`), `DCT_THRESHOLD_FRACTION` (deprecated with the DCT denoiser).

**The trap this panel is built around:** the output filename embeds the denoise chain.
Changing the method silently retargets which store is saved *and* loaded — switch the
denoiser and press load, and you get a different preprocessing run, or a "not found" for
work you believe you did. The resolved path sits next to the control for that reason.

Actions: **Build lazy stack** (wires the chain, computes nothing, then offers to save) ·
**Save to Zarr…** (computes every frame; confirms before overwriting) · **Load from Zarr**
(a complete entry point — available without loading a dataset first).

Saving adopts the store: the session afterwards *reads* the file it just wrote, so
extraction does not re-run the chain. A guard refuses to re-save an adopted store onto
itself, which would delete the source mid-write and zero the contents.

---

## Tab 3 · ROIs

| GUI label | Tier | Widget | Default | Notes |
|---|---|---|---|---|
| `LABELS_DIR` | A | directory picker | *(empty)* | **Only** sets where the load/save dialogs open. |
| label dtype | — | fixed | `uint16` | |
| overlay opacity | D | — | `0.6` | Not adjustable yet. |

The ROI label path is **not** a field. Load and save each open their own dialog and record
what the dialog returned, so the recorded path always reflects the file actually used —
never a stale field someone edited by hand. It is written into `traces_params.json`.

Actions: **New blank ROI labels** (sized to the preprocessed stack) · **Load ROI labels**
(raises on a shape mismatch against the stack — shown as a dialog, since the alternative
is silently wrong traces) · **Save ROI labels**.

Both saving and loading leave the labels live in the session; there is no second click to
"load" what you just saved. Paint on the *ROI labels* layer in napari with the brush;
press <kbd>M</kbd> for a fresh label id per ROI.

**The ROI count is never stored.** It is derived from the current label image everywhere
it appears — the tab 3 readout reads the live napari layer, so a painted ROI is counted
without saving first, and extraction derives `roi_ids` the same way. Note that importing a
new label set does **not** invalidate traces already extracted from an older one; re-run
extraction after changing labels.

This build renders no max/STD projection.

---

## Tab 4 · Traces

| GUI label | Tier | Widget | Default | Notes |
|---|---|---|---|---|
| **source .zarr** | P | directory picker | derived from tab 2 | The store traces are extracted from. Filled in automatically when a store is saved or loaded; override to extract from a different preprocessing run. |
| `START_TIMEPOINT` | P | int | `0` | First frame of the window. |
| `EXTRACT_TIMEPOINTS` | P | int | `500` | Frames to extract (≈50 s at 10 Hz). |
| `BASELINE_SLIDES` | P | int | `20` | Leading frames averaged into F0 (≈2 s). Interacts with the `pre` phase — see tab 6. |
| Analysis root | A | directory picker | *(empty)* | `<root>/<dataset_tag>/` for `save_traces` / `load_traces`. Also the starting point for the tab 5/6 output dialogs. |
| import file | P | file picker | — | Traces-only entry. `.csv` / `.npz` / `.npy`. |
| `orientation` | P | dropdown | `auto` | `roi_rows` \| `roi_cols` \| `auto`. Auto reads a table with a time column as one column per ROI; otherwise the longer axis is time. |
| `time_column` | A | text | `auto` | Column name or index, `auto`, or none. Auto accepts a column named time/t/seconds, or a strictly increasing numeric first column in a taller-than-wide table — a `roi` id column is explicitly not mistaken for one. |
| `frame_rate_hz` | P *(traces-only)* | float | — | **Required** when the file carries no time axis. |

### How the traces are computed

Three stages, in `analysis_tools/traces.py`:

1. **Spatial mean** — for ROI *r* with pixel set `P_r = {(y,x) : L(y,x) = r}` from the
   label image `L`: `F_r(t) = (1/|P_r|) · Σ I(t,y,x)`. Unweighted, no neuropil
   subtraction.
2. **Baseline** — `F0_r = mean(F_r[0 : BASELINE_SLIDES])`. A fixed leading window, not a
   rolling percentile; it assumes the recording starts quiet.
3. **Normalisation** — `ΔF/F_r(t) = (F_r(t) − F0_r) / F0_r`.

`F` here is the mean over the **preprocessed** stack, so each pixel has already been
denoised, background-subtracted at the 4th percentile, motion-corrected and clipped at
zero. That subtraction shrinks `F0` and therefore inflates ΔF/F relative to the same
computation on raw camera counts — and it is why `F0 = 0` is a live failure mode, guarded
explicitly. Both `F` and ΔF/F are kept and saved.

Export writes three files via `store.save_traces`: `traces.npz` (exact, including the raw
pre-dF/F traces), `traces.csv` (portable — `time_s` plus one column per ROI), and
`traces_params.json` (mode, dataset, source zarr, ROI-label path, the three window
options, frame rate, ROI/frame counts).

---

## Tab 5 · Spike inference

| GUI label | Tier | Widget | Default | Notes |
|---|---|---|---|---|
| `CASCADE_DIR` | A | directory picker | *(empty)* | e.g. `./CascadeTorch`. Also supplies the FFT helper used by Index B in tab 6. |
| `MODEL_NAME` | P | dropdown | `Spinal_cord_excitatory_30Hz_smoothing50ms` | From `cascade_runner.available_models()`; models are read from `<CASCADE_DIR>/Pretrained_models`. |

The panel shows what `cascade_runner.check_model()` prints, inline and before the run:
training rate parsed from the **name** vs the rate in `config.yaml` (at least one shipped
model disagrees), the indicator the model was trained on (GCaMP6 vs 8 mismatch), the
resample ratio, and the pad frames that will be NaN at each edge.

Model choice is a scientific claim, not a preference. The default is a spinal dorsal-horn
model applied to DRG — the nearest available analogue, and an untested transfer
(`CLAUDE.md` §4C, App. A §8). No DRG-specific ground truth exists. The panel carries that
caveat where the user picks, not only in the writeup.

Actions: **Run CASCADE** · **Save inferred spikes…** · **Load saved inference…**

Running writes nothing by itself. It offers to save, and saving asks for a destination
folder rather than deriving one — see [I/O behaviour](#2-io-behaviour). Declining keeps the
rate in the session with the Save button live; it is lost only when the window closes.
Loading asks for a folder too, otherwise a run saved outside the derived path could never
be read back, and it clears the in-memory result so a rate that is already on disk is not
duplicated.

Written on save: `spike_rate.npy`, `spike_rate.csv`, `spike_inference_params.json`.

---

## Tab 6 · Analysis

| GUI label | Tier | Widget | Default | Notes |
|---|---|---|---|---|
| Stimulus phases | P | editable table, name → (start, end) s | `pre (2,10)`, `stim (10,15)`, `post (15,50)` | `pre` starts at 2.0 s, not 0, because frames `0..BASELINE_SLIDES` *defined* F0 — dF/F there is ~0 by construction and would fake a silent baseline. |
| Index B frequency bands | P | editable band table | `(0.2,0.5) (0.5,1.0) (1.0,2.0) (2.0,5.0)` Hz | Upper edge guarded against Nyquist, with a tolerance: a measured 9.999999 Hz must not reject the documented 5.0 Hz edge on floating-point margin. |
| `ACTIVE_THRESH` | P | float + auto toggle | auto | spikes/s. Auto = baseline median + `ACTIVE_N_SIGMA` × robust sigma. |
| `ACTIVE_N_SIGMA` | P | float | `3.0` | A fixed absolute threshold does not transfer between models. |
| `PCA_VAR_TARGET` | P | float 0–1 | `0.95` | Cumulative variance target selecting the PC count. |
| `MAX_K` | A | int | `8` | Largest k in the candidate-cut table. |
| `MIN_GROUP_SIZE` | A | int | `3` | A "group" of 1–2 ROIs is an outlier, not a population. |
| `MAX_GROUP_FRAC` | A | float | `0.9` | One group holding almost everything is not a partition. |
| `RATE_CUT_HEIGHT` | P | float, blank = auto | auto | Manual dendrogram cut for Index A; blank → `grouping.suggest_cut`. |
| `FREQ_CUT_HEIGHT` | P | float, blank = auto | auto | Same for Index B. |

Clicking **Run analysis and export** validates `CASCADE_DIR`, then asks for the output
folder, then computes. Exports, all into the folder chosen for that run:

- `rate_features.csv`, `freq_features.csv` — `store.save_features`
- `roi_groups.csv` — `store.save_groups`
- `pca_rate.npz` / `pca_freq.npz` + `.json` + `pca_<label>_scores.csv` — `store.save_pca`
- `figures/*.tiff` — `store.save_figure` on the figure each `plots.*` function returns:
  `pca_rate_summary`, `dendrogram_rate`, `rate_heatmap`, `rate_group_means`,
  `pca_freq_summary`, `dendrogram_freq`

`save_pca` stores scores, loadings, explained variance and the linkage matrix — enough to
re-plot and re-cut — but **not** the `StandardScaler` statistics, so it cannot project new
ROIs into a saved PCA space. The PCA is per-dataset; there is nothing to project.

Figures are drawn on the UI thread after the numeric work finishes in the worker. pyplot
is not thread-safe: building figures inside the worker deadlocks against the Qt event loop
at 0% CPU and never returns.

Group labels are an arbitrary integer labelling, not a cell type.

---

## Plotting — documented, adjustability not implemented

Listed so the eventual controls are known. These are currently constants in the plot
functions.

| Option | Current value | Where |
|---|---|---|
| dF/F heatmap colormap | `magma` | `plots.rate_heatmap` |
| max-projection colormap | `coolwarm` | notebook projection cells |
| robust colour limits | 1st / 99th percentile | notebook QC plot |
| heatmap `vmax` | `nanpercentile(99)` | `plots.rate_heatmap` |
| figure dpi | `200` | `store.save_figure` |
| figsize rules | `(12, 0.18·n_rois + 2)`, `(12, 1.9·n_groups + 1.2)`, `(16, 4.4)`, `(13, 4.6)` | `plots.*` |
| trace linewidth | `0.6` | notebook QC plot |
| `SHOW_ROI_TRACES` | `True` | member ROIs behind group means |
| `ROI_TRACE_ALPHA` | `0.3` | |
| stimulus shading | the `stim` phase | `plots.rate_heatmap`, `plots.group_means` |

Constraints that are **not** preferences and stay fixed even once this panel is built
(`CLAUDE.md` §5): perceptually-uniform colormaps only, never `jet`; raw and inferred
traces in separate subplot rows or visually distinct; every panel showing inferred spikes
names the algorithm and its parameters in that same panel.

---

## Remaining gaps

1. **Napari runs as a separate top-level window**, not embedded in the Qt app. Workable,
   but the two windows can be lost behind each other.
2. **Jobs are not cancellable.** They run off the UI thread with a progress bar, but a
   long Zarr write has to be waited out.
3. **New ROI labels do not invalidate existing traces.** Importing a label set after
   extracting leaves the old `roi_ids` and dF/F in the session, and tabs 5–6 stay
   enabled. Re-extract after changing labels.
4. **No projection rendering.** The max/STD projection layers the notebook used for ROI
   reference are not built here; ROI labels are sized against the preprocessed stack.
5. **Plot adjustability** — the table above.
6. **`labels/Debi_DRG_ROI.csv`** exists on disk but nothing reads or writes it. Decide
   whether it is an input format worth supporting or a leftover.

---

## Scientific caveats that travel with every output

- CASCADE's ground-truth corpus is **entirely CNS** (zebrafish telencephalon, CA3,
  neocortex). No DRG or DCN training set exists. The dorsal-horn model used here is the
  nearest analogue and the transfer is untested — state this wherever the output is
  quoted.
- Default CASCADE models are GCaMP6-tuned; applied to GCaMP8 they misestimate rates at
  both ends of the range.
- **ΔF/F amplitude is not a rate proxy.** The `dff_peak_*` columns in the exports are
  descriptive only.
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
