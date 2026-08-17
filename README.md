# KSLSpikeInference

Spike inference toolkit for two-photon microscope imagery at KSL Lab. This toolkit uses the CASCADE model by Rupprecht et al. (2021), originally published on Nature Neuroscience.

## Workflow
<div>
  
1. **Working with .npy files**<br>
Use `process_using_napari.ipynb` for preprocessing, ROI extraction, and dF/F trace extraction directly from .npy files obtained on microscopes. Alternatively, modify `SlideBook_WF_Timelapse_Dask_Load.ipynb` for image processing (Credit: Li-Ting, KSL Lab) if you have SlideBook installed locally.

</div>
<div>
  
  2. **Working with extracted traces**<br>
  Under `/CascadeTorch/scripts`, `model_tester.py` and `model_tester_batch.py` can be found with their capabilites listed at filehead. Before spike inference can be performed, note that traces must be preprocessed to `.csv` files with the following format (designated by `model_tester.py`). Additional columns are allowed but will be ignored by python scripts.

  <div align="center">
    
  | Trace_ROI1 | Trace_ROI2 | Trace_ROI3 |
  | -------- | -------- | -------- |
  | 0.0001    | 0.0003   | 0.0005   |
  | 0.0002    | 0.0004   | 0.0006   |

  *Standard format for calcium traces (ΔF/F).*
  </div>
  

  Resampling of traces to the sampling rate of the selected pretrained model is done by `model_tester.py`. Simply indicate sampling frame rate of input traces with the option `--input-fps` when executing `model_tester.py` or `model_tester_batch.py`. 
  Some parsing examples can be found at `/CascadeTorch/scripts/parse_csv_Mark.py` and `CascadeTorch/scripts/parse_npy_LT.py`  

</div>

---

## Analysis notebook — `main/main_new.ipynb`

Downstream analysis of DRG calcium imaging **before and after spared-nerve injury (SNI)**, starting
from CASCADE output that has already been computed. The notebook does not run spike inference; §6
loads pre-computed runs from `data/pro_drg/<group>/<FOV>/<model>/spike_rate.npy`.

**Runs top to bottom in ~19 s** on the current dataset (one FOV, four recordings). Kernel: the
`Cajal` conda environment (Python 3.12, numpy ≥ 2, pandas ≥ 2, scipy, scikit-learn, matplotlib,
ipywidgets). Set `SAVE_OUTPUTS = True` in the Config cell to write PNGs to `OUTPUT_DIR`; nothing is
ever written into the dataset folders.

### 1. The question

Two questions, asked of the same cells on two days:

1. Does a cell's response **pattern** change after SNI?
2. Does its response **amplitude** go up or down?

Both are *within-cell, before-versus-after* questions. That single fact drives most of the design
decisions below: anything that rescales a cell by its own activity, or that means something
different on the two days, destroys the comparison.

### 2. The data

| | |
| --- | --- |
| preparation | mouse DRG, GCaMP, two-photon, one field of view (`FOV3`) |
| conditions | `Baseline` and `SNI-live`, imaged on separate days |
| protocols | **graze** — 5 s rest, then 12 × (5 s graze, 5 s gap); **clip** — applied at 10 s, withdrawn at 30 s |
| sampling | ~10.5 Hz (`dt` ≈ 0.095 s) |
| lengths | graze 1420 frames / 134.9 s both days; clip 734 frames / 69.7 s (Baseline), 652 / 61.9 s (SNI) |
| ROIs | 291 Baseline, 349 SNI → **291 paired by ROI id** |
| inputs | `spike_rate.npy` (CASCADE inferred rate, Hz) + the `*_roi_traces.txt` ΔF/F export it came from |

`Trace_ROI<x>` in the export is **already ΔF/F**, computed upstream against a full-trace baseline
and not re-baselinable from these files. §5 therefore does *not* recompute ΔF/F₀ — it only subtracts
each ROI's own pre-stimulus mean, so that "response amplitude" starts from zero on both days. The
shape of the signal is untouched.

### 3. First principles

These are the rules the whole notebook is built on. Everything in §7 follows from them.

#### 3.1 Detection on the inferred rate, never on ΔF/F

All event detection runs on the CASCADE inferred rate. The rate is **non-negative and
zero-inflated** — most bins are exactly 0 — which is why detection is **one-sided**: `mu + k·sd` is
meaningful, `mu − k·sd` is not. The consequence is stated in the notebook and worth repeating: this
pipeline detects *increases*. **It cannot see a cell going silent.** A cell that loses a response
appears as a cell that stops being detected, which is not the same measurement.

#### 3.2 The threshold comes from the pre-stimulus window and nowhere else

For each cell independently:

```text
mu, sd = mean, SD of the inferred rate over the FIRST segment only
thr    = mu + DETECT_K · sd                    (DETECT_K = 3)
active = rate > thr
event  = a run of >= N_CONSEC consecutive active frames   (N_CONSEC = 3)
```

Using the whole trace would be **circular**: a strongly responding cell inflates its own SD, raises
its own threshold, and censors itself. Restricting to the pre-stimulus window breaks that loop.

This is also the one thing that must not become protocol-dependent. `detect_events` is deliberately
blind to which protocol it is running on — it always uses `edges[0]`–`edges[1]`, which is the
pre-stimulus window under either segment model. Both parts of the analysis rest on that.

> **Known asymmetry, checked in §8a.** The post-SNI "baseline" window is itself post-injury
> spontaneous activity. If SNI raised resting activity, the SNI threshold sits higher, the same
> response is less likely to clear it, and cells appear to *lose* responses for arithmetic rather
> than biological reasons. §8a prints both thresholds and says whether they are comparable.

#### 3.3 Erased (0) and undefined (NaN) are different things, permanently

- Sub-threshold bins are erased to **0** — "no activity here".
- CASCADE pads 17 frames of **NaN** at each end of every recording — "nothing was measured here".

The two never mix. A NaN bin is never active and never becomes a 0. In every AUC, NaN contributes
nothing; in every figure, NaN is drawn grey while erased 0 is drawn near-black. Treating padding as
silence would manufacture quiet cells at both ends of every trace.

#### 3.4 Amplitude stays in its original scale

No per-cell normalisation, ever. Dividing each cell by its own total would divide out exactly the
biology in question: **a cell whose response grew 10× would look unchanged.** Z-scoring appears only
once, in the PCA cells, and it is applied *across the pooled population* so that features with
different units are comparable — a population rescaling, not a per-cell one.

#### 3.5 Labels are declared, not fitted

Segment boundaries, the `DETECT_K` threshold, `ON_GAP_HIGH`, `EARLY_MAX_TRIAL` — all are
**parameters written in the Config cell**, not values chosen by optimising an outcome. The reason is
comparability: a fitted boundary would mean something different on the two days, and the entire
before/after comparison assumes the label means the same thing in both. Where a declared cut matters
(§9B), the notebook prints the distribution it is cutting so the cut can be judged by eye.

#### 3.6 Chance level comes from a circular-shift null

An absolute count of responding cells means nothing on its own — with a permissive rule, cells
respond "during the stimulus" by accident. The null answers: *how often would this happen if the
cell's activity were not aligned to the stimulus?*

```text
for each surrogate:
    rotate each cell's FINITE samples by a random offset (NaN padding stays put)
    re-run the identical detection
    count what you counted for real
```

Rotation preserves each cell's own statistics exactly — its rate distribution, its burst structure,
its number of event frames — and destroys only *when* the activity happened relative to the
stimulus. That is precisely the alignment the claim is about, so it is the right thing to destroy.

Only the finite samples rotate. Rolling the raw array would drag CASCADE's NaN padding into the
baseline window and leave nothing to build a threshold from; the padding is a property of the
recording, not of the cell.

**The graze constraint (§8b).** With a 10 s cycle, an offset of about *k* whole cycles lands the
response back on top of a *different* graze. The surrogate is then still stimulus-locked, scores as
high as the real trace, and the null swallows the effect it exists to measure. For graze, offsets
within `CYCLE_EXCLUSION_FRAMES` of any multiple of the cycle period are therefore illegal. clip has
one application and no such structure, so its shift is unconstrained.

#### 3.7 Significance as a row weight

Where a per-cell "how remarkable is this" number is needed, it is an empirical p-value turned into a
weight:

```text
p      = (1 + #{surrogates scoring >= real}) / (1 + n_surrogates)
weight = -log10(p)
```

The `1 +` on both sides is the standard permutation correction — it makes `p` bounded away from 0,
so the best attainable value is `1/(n+1)`, and it is honest about the resolution the surrogate count
actually buys. Two consequences show up constantly in the figures:

- A cell with **no detected events** scores `real = 0`, every surrogate ties it, `p = 1`, `weight = 0`
  **exactly**. These cells form a large tied block.
- The weight has a **ceiling** at `-log10(1/(n+1))`. Many cells pile up there. `weight_max` is
  therefore uninformative; the useful statistic is *how many* cells reach the ceiling.

#### 3.8 Cells are paired by ROI id

The two days do not carry the same number of ROIs (291 → 349). `pair_indices` intersects the id
lists and returns each id's row index within its own recording. Surplus SNI ids have no partner and
are excluded from every paired analysis. ROI ids are read **numerically**, not as text — a string
sort puts `ROI10` before `ROI2`, and that order would silently become the row order of every matrix
downstream.

#### 3.9 The protocol fork

The two protocols are not the same experiment with different words:

| | clip | graze |
| --- | --- | --- |
| stimulus | one 20 s application | 12 × (5 s on, 5 s gap) |
| stimulus share of recording | 20 s of ~70 s | 120 s of 135 s (**89 %**) |
| window after the stimulus | ~32–40 s | **9.9 s** |
| natural label | 3 bits, `S1 S2 S3` | per-cycle: when it started, how phasic |

A 3-bit label applied to graze puts 89 % of the recording inside one `S2` bit: nearly every
responding cell becomes `010`, and a circular shift leaves the response inside `S2` almost
regardless of offset. So the notebook forks after §8 — **Part A = clip, Part B = graze** — sharing
every function and routing on `experiment`.

### 4. Section by section

#### Shared: §0 – §8c

| § | Does | Why |
| --- | --- | --- |
| **0** | Imports, then the **Config cell** — every parameter in the notebook | One place to change anything; nothing downstream invents a parameter |
| **1** | Loads `spike_rate.npy` + its source ΔF/F, matched **by provenance** through the params JSON | Baseline and SNI use identical file *names*; matching on name alone silently loads the wrong day. The whole relative path is resolved, and ROI ids are checked against the JSON |
| **2–3** | ROI positions → centroids and boxes; draws the ROI map per condition | A sanity check that the two days are the same field |
| **4** | Skipped — needs raw frame stacks, which this dataset does not ship | Stated rather than silently omitted |
| **5** | Subtracts each ROI's pre-stimulus mean from ΔF/F | Not a ΔF/F₀ computation (§2). Makes "amplitude" start from 0 on both days |
| **6** | Dropdown over the pre-computed CASCADE model runs | Only models present for *every* run are offered, so a pick is always loadable |
| **7** | ΔF/F and inferred rate heatmaps, side by side | Convention: never replace the raw trace with the inferred one in a validation figure |
| **8** | `segment_edges`, `segment_kinds`, `detect_events` | Protocol-dependent **segments**, protocol-blind **detection** (§3.2). Also prints the graze post-window check |
| **8a** | Two pre-flight checks | Is the baseline window inside CASCADE's padding? Are the two days' thresholds comparable (§3.2)? |
| **8b** | The circular-shift null, the graze constraint, the offset-pool guard | Chance level (§3.6). Counts are aggregated **by segment kind**, because graze has 26 segments and "any ON window" is the quantity of interest |
| **8c** | Shared per-ROI amplitude/shape helpers | `amplitude_totals`, `sharpness`, `event_onsets` mean the same thing under either segment model, so they are computed once rather than twice with a chance of drifting apart |

**Segments.** clip gets 3 (`S1 | S2 | S3`). graze gets **26** — `S1 | C1_on | C1_gap | … | C12_gap
| S3` — generated from the protocol constants, not hand-set. Every segment carries a **kind** in
`{pre, on, gap, post}`, and all downstream code selects by kind (`kind_of_t == "on"`) rather than by
index arithmetic that would mean different things in the two protocols.

**The graze post-window is checked, not assumed.** It is whatever is left after the 12th cycle — on
this recording, **9.9 s**. That is too short to separate a sustained response from the tail of the
last graze, so §8 prints a warning: graze **cannot support a "chronic" claim at all**. What graze can
speak about is activity in the **gap windows between cycles** (5 s after each graze, not minutes
after all of them). Those are different statements and the notebook words them differently.

**The offset-pool guard.** Constraining graze offsets removes about half of them (676 of 1386 legal
on this recording), and a pool barely larger than the surrogate count means the "null" is a few
rotations resampled. `SHIFT_POOL_FACTOR = 10` is the guard. It is **hard for graze**, where the pool
depends on `CYCLE_EXCLUSION_FRAMES` and the surrogate count and both can be changed — which is why
`SPECTRUM_N_SURROGATES_GRAZE = 60`. It **warns and continues** for the unconstrained shift, where the
pool *is* the number of valid frames and no parameter enlarges it. clip does fall short (700 offsets
for 200 surrogates); that is a property of a 70 s recording, and the only consequence is that `p` is
resolution-limited, which it already is at 1/201.

#### Part A — clip

| § | Does | Why |
| --- | --- | --- |
| **9A** | One **bit per segment**: `r[s] = any detected event in segment s` → labels like `010`. Per-segment AUC, peak, event frames and sharpness ride alongside in the original scale | The label is *defined*, so it means the same thing on both days (§3.5). `010` = responds during the clip then stops; `001` = quiet during, chronic afterwards |
| **9Ab** | Scatter of `auc_S2` against `sharpness_S2` | Checks whether the shape axis separates anything **before** it goes into the clustering. If it is one blob, sharpness separates nothing on this dataset — worth knowing early |
| **10A** | PCA + k-means on amplitude × timing, fitted **once on pooled Baseline + SNI** | Amplitude-only features are collinear and just reproduce the volcano; timing is what makes the cloud separable. Pooled fitting makes a cluster id mean the same thing on both days |
| **11A** | Transition matrix, Baseline label × SNI label | The before/after question *is* a contingency table. The diagonal is cells that did not change |
| **12A** | The spectrum: every ROI over time, rows ordered by weight | The whole dataset in one view, with one row = one cell readable straight across both days |
| **12Ab** | Spectrum with cells quiet on **both** days dropped (percentile cutoff per day) | The black mass in §12A is erased zeros, not a colour-scale problem. Dropping cells quiet in *each* recording removes those rows without touching the phenotype of interest — a cell quiet at Baseline but active after SNI is kept |
| **12Ac** | Spectrum with a **rank-sum** filter: drop when `pr_baseline + pr_sni < QUIET_PR_SUM` | A cell can survive by being strong on one day alone, which is the asymmetry a before/after question needs. Two guards: a tie diagnostic (once more than half the cells are silent the filter can become a no-op, and it says so), and `PROTECT_LATE_RESPONDERS`, which never drops a cell with an event outside `S1` — that is the `001` phenotype the analysis exists to find |

**Sharpness**, used by 9A/9Ab/10A, is `peak / mean-height-while-active` = `peak · duration / auc`.
AUC collapses amplitude × duration into one number, so a sharp transient and a low sustained plateau
can share an AUC — precisely the two phenotypes of interest. Sharpness separates them and is a
**ratio**, hence dimensionless and unchanged if the whole trace is scaled, which is what makes it
orthogonal to the amplitude axes. It is `NaN` where a cell has no events — never 0, never 1.

> Acquisition is ~10.5 Hz and the model carries a 150 ms smoothing kernel, so "sharp" has a physical
> floor. Sharpness separates a few-hundred-ms transient from a tens-of-seconds plateau; it must not
> be used to claim sub-100 ms timing differences.

#### Part B — graze

| § | Does | Why |
| --- | --- | --- |
| **9B** | **Per-cycle table** (`auc_on`, `auc_gap`, `peak_on`, `responded` per ROI per cycle) plus a per-ROI summary and a 5-way phenotype | A bit per segment would give 2²⁶ labels — a serial number, not a phenotype |
| **10B** | PCA + k-means on amplitude × **per-cycle** timing | Same method as 10A, different timing axes: `first_trial`, `trial_slope`, `on_gap_log`, `sharpness_total` |
| **11B** | Transition matrix over the 5 graze phenotypes, plus **marginals** | The two label axes move independently; a cell can keep its timing and flip its shape, and the 5 × 5 grid alone hides that |
| **12B** | graze spectrum, with per-cycle markers and a weight from the **constrained** shift over the **ON windows** | See below |

**The §9B summary columns, from first principles:**

| column | how it is computed | what it means |
| --- | --- | --- |
| `n_trials_responded` | count of cycles with a detected event in the ON window | 0–12 |
| `first_trial` | index of the first such cycle, `NaN` if none | **the direct test of "only from the 4th graze onwards"** |
| `trial_slope` | **Theil–Sen** slope of `auc_on` vs cycle index — the *median* of all pairwise slopes | Robust to the one huge cycle that would dominate a least-squares fit. > 0 = sensitising, < 0 = adapting |
| `trial_rho` | Spearman(cycle index, `auc_on`) | Rank version of the same trend, insensitive to the size of the change |
| `on_gap_ratio` | `sum(auc_on) / sum(auc_gap)` | **High = phasic; ≈ 1 or lower = tonic/sustained.** The axis a single `S2` bit made invisible, and what "persistent response" can mean for graze |
| `phase_vector_strength`, `phase_p` | each event **onset** folded onto the 10 s cycle as an angle; `R` = length of the mean unit vector; **Rayleigh** test for uniformity | `R` ≈ 1 = every event at the same point in the cycle; `R` ≈ 0 = scattered. Standard measure for periodic stimulation |

Two deliberate choices in the phase statistic: it uses one sample per **event**, not per
supra-threshold frame (frames inside one event are serially correlated, and counting each would
inflate *n* and deflate `p`); and it is reported **alongside** the shift-based weight, never instead
of it — locking to a consistent phase and responding more than chance are different claims.

**Per-cycle values are descriptive, not tested.** There is no p-value per cycle. A test per cycle per
cell is 12 × n_roi tests answering a question nobody asked; the trend across cycles (§13) and the
whole-recording weight (§12B) are where the testing belongs.

**The graze label** (declared thresholds, §3.5):

| label | rule |
| --- | --- |
| `none` | `n_trials_responded == 0` |
| `early_phasic` | `first_trial ≤ EARLY_MAX_TRIAL` and `on_gap_ratio ≥ ON_GAP_HIGH` |
| `early_tonic` | `first_trial ≤ EARLY_MAX_TRIAL` and `on_gap_ratio < ON_GAP_HIGH` |
| `late_phasic` | `first_trial > EARLY_MAX_TRIAL` and `on_gap_ratio ≥ ON_GAP_HIGH` |
| `late_tonic` | `first_trial > EARLY_MAX_TRIAL` and `on_gap_ratio < ON_GAP_HIGH` |

`on_gap_ratio` is `+inf` for a cell that responded but never fired in a gap — correctly the *most*
phasic case, and the label handles it. For clustering and plotting, `on_gap_log` is the regularised
twin, `log10((on + ON_GAP_EPS)/(gap + ON_GAP_EPS))`, finite for everyone; `ON_GAP_EPS` is declared
(roughly the AUC of one minimal detectable event), not fitted.

**§12B changes two things** relative to §12A, and prints them **applied one at a time** so the gain
is attributed rather than asserted:

| variant | window scored | offsets |
| --- | --- | --- |
| §12a (original) | whole 5–125 s block | unconstrained |
| middle | ON windows only | unconstrained |
| §12B | ON windows only | **constrained** |

Scoring the ON windows mostly moves the **median** (the block is half gap, so a cell firing only
during grazes was being scored across 60 s of its own silence); constraining the offsets mostly
moves the **tail** — the count of cells no surrogate ever beat. Both gains are real and both are
small. **The ceiling is set by the duty cycle, not by the offset pool:** the ON windows are half of
every cycle, so even a maximally wrong shift drops about half the response back inside an ON window.
No choice of offsets undoes that. graze has less power than clip here; the constraint narrows the
gap without closing it.

#### Shared tail: §12d – §15

| § | Does | Why |
| --- | --- | --- |
| **12d** | **Row index → ROI id** map, per experiment, plus `rois_for_rows()` | A row number is a property of a *figure*, not a cell — see below |
| **13** | Response probability across the 12 graze cycles, with an order-shuffle test | The population-level test of "the response only appears after the 4th graze" |
| **14** | clip phenotype accounting: group sizes with ROI ids, restricted transitions, chronic-bit latency | Turns figure-reading claims about `010`/`011`/`001` into counts |
| **15** | Co-activation and pairwise Jaccard against a shift null | Nothing else in the notebook measures cells being active *together* |

**Why §12d exists.** Claims arrive as row ranges — "rows 150–250". But §12A, §12Ab, §12Ac and §12B
each sort a *different* surviving subset, so the same row number points at different cells in each,
and the mapping moves again whenever the weight is recomputed. §12d emits one row per paired ROI with
its position in its own spectrum, both days' weights, and both days' labels, then resolves
`CLAIM_ROWS` into explicit ROI ids. **Everything downstream takes ROI ids; nothing names a row.**

> **The tied-zero block.** Cells with no detected event in the scored window score weight **0
> exactly** (§3.7), and `np.argsort` orders that whole block arbitrarily. A row range falling inside
> it **does not name a reproducible set of cells** — re-run with a different surrogate seed and the
> membership changes while the figure looks identical. §12d prints where the block starts and how
> much of each claim range lands in it. Check this before reading any row-based claim.

**§13's test.** Spearman's rho between cycle index and per-cycle responder count, against a null that
shuffles the **cycle order within each cell**. That preserves how many cycles each cell responded to
and destroys only *when*, so rho beyond the null means the **ordering** carries information — not
merely that some cells respond more than others.

**§14's chronic-bit latency** compares the time from withdrawal to the first detected `S3` event,
before against after. It is run twice — over each day's full window and over the **common span** —
because the post-withdrawal window is 39.7 s at Baseline but only 31.9 s after SNI. A cell that lost
the chronic bit and a cell whose chronic response begins after the camera stopped are
indistinguishable here, so **"delayed beyond the recording" is labelled untestable**, with both
window lengths printed beside the result.

**§15's null** shifts each cell's **event mask** independently, so every cell keeps its exact number
of event frames and only the alignment *between* cells is destroyed. A consequence worth knowing:
the co-activation **total is conserved by construction** (the printed check confirms it), so only its
distribution over time can move — which is why there is no "excess averaged over the whole recording"
column, and why the readout is the shape of the trace against the band.

> **This null does not isolate coupling.** Rotating a cell also destroys its lock to the *stimulus*,
> so common drive clears the band comfortably. A synchronised-cluster claim needs the observed trace
> above the band, but being above the band does not establish the claim. `excess_stim_off` is the
> partial answer available. Note that §15 deliberately keeps the **unconstrained** shift even for
> graze: there a cycle-aligned rotation would preserve mutual alignment, so the unconstrained version
> is the conservative choice, and constraining it would *inflate* apparent synchrony.

### 5. Figure reference

Seventeen figures. For each: what the axes are, what the colour or grid value encodes, and why the
plot is drawn that way.

---
**`3_roi_maps` — §3.** x = ROI centroid x (px) · y = ROI centroid y (px, **inverted** to match image
convention) · marks = one ellipse per ROI, sized by its bounding box, coloured by condition.
**Why:** a visual check that Baseline and SNI are the same field before any cell is paired by id.

---
**`7_trace_heatmaps` — §7.** x = time (s) · y = one row per ROI, **sorted by peak inferred rate** ·
colour = ΔF/F (`magma`, top row of panels) and inferred rate in Hz (`viridis`, bottom row), clipped
at the 2nd/99th percentile. Red bars mark stimulus on.
**Why:** raw and inferred shown together, before any thresholding. Lab convention is never to replace
the raw trace with the inferred one in a validation figure — if the two disagree, this is where it
shows.

---
**`8b_calibration` — §8b.** x = (condition, segment kind) groups · y = number of cells with ≥ 1
detected event · paired bars = **real** (blue) against **circular-shift null** (grey).
**Why:** an absolute count is uninterpretable on its own. Bars of similar height mean detection at
chance in that window. Aggregated by kind because graze has 26 segments and "any ON window" is the
quantity the claim is about.

---
**`9Ab_sharpness` — §9Ab.** x = `auc_S2`, **log** · y = `sharpness_S2`, **log** · colour = the 3-bit
label · dotted line at sharpness = 1 (a flat, square pulse).
**Why:** tests whether the shape axis does any work before it is allowed into the clustering. If
`010` and `001` sit at different heights where they overlap in AUC, shape is real; one blob means
sharpness separates nothing here. Only cells with events **in S2** can appear — sharpness is
undefined elsewhere.

---
**`10A_pca_clusters` — §10A.** x = PC1, y = PC2 (percent variance explained in the axis labels) ·
**colour = k-means cluster**, **marker shape = condition** (circle Baseline, triangle SNI), with two
separate legends.
**Why:** two things are encoded at once and must stay independent — if a cluster were only ever one
condition, that is a finding, and conflating colour with shape would hide it. Cells with no detected
events have no timing and are held out (`cluster = -1`) rather than imputed.

---
**`11A_transitions` — §11A.** x = **SNI** label · y = **Baseline** label · **grid value = number of
paired ROIs** making that transition, printed in each cell and also mapped to `viridis`.
**Why:** the before/after question is literally a contingency table. The **diagonal** is cells that
did not change; everything off it is a cell that did.

---
**`12A_spectrum` — §12A.** Two main panels (Baseline, SNI): x = time (s) · y = **one row per paired
ROI**, ordered by weight, **heaviest at the top** · **colour = the detected inferred rate in Hz, on a
log scale**. Grey = undefined (CASCADE padding); near-black = erased sub-threshold zeros, pushed
below `vmin` so they take a distinct colour and cannot be confused with "undefined". Dashed white
lines = segment boundaries; red bar = stimulus. Right strip: x = weight (Baseline drawn right, SNI
mirrored left), y = **the same row order**.
**Why:** the entire dataset in one view. Both panels share **one row order — the Baseline one — so a
given row is the same cell in both and can be read straight across.** The log scale is necessary:
detected rates run from the ~0.03 Hz threshold to ~20 Hz, three decades, and the bulk sits just above
threshold, so a linear ramp would show only the few brightest cells. The weight strip lets you see
the ordering variable itself rather than trusting the sort.

---
**`12Ab_spectrum_filtered` — §12Ab.** Identical axes to `12A_spectrum`, drawn over the subset of rows
that are **not** quiet on both days (weight ≤ the per-day percentile cutoff in *each* recording).
Colour limits are recomputed over the survivors.
**Why:** removes the black mass without touching the phenotype of interest. A cell quiet at Baseline
but active after SNI survives, because it only goes if it is quiet in **each** recording.

---
**`12Ac_spectrum_ranksum` — §12Ac.** Identical axes again, with rows dropped when
`percentile_rank(Baseline) + percentile_rank(SNI) < QUIET_PR_SUM`.
**Why:** a cell can survive by being strong on **one** day alone — the asymmetry a before/after
question needs, which the per-day cutoff of 12Ab cannot express. Read the printed tie diagnostic
first: once more than about half the cells are silent, the tied block can score above the cutoff on
its own and the filter drops nothing.

---
**`9B_graze_axes` — §9B.** *Left:* x = `log10(on_gap_ratio)` (right = phasic, left = tonic) · y =
number of cells · colour = condition · dashed vertical = the declared `ON_GAP_HIGH` cut · an
annotation reports the cells at **∞** (responded, never fired in a gap), which are off-scale right
and counted as phasic. *Right:* x = `first_trial` (jittered by condition) · y = `on_gap_log` · dashed
horizontal = the phasic cut, dotted vertical = the early/late cut.
**Why:** the cut is a *declared* parameter, so the figure shows the distribution it is cutting and
lets it be judged by eye. The right panel shows both label axes at once, which is what the 5-way
label is made of. The ∞ group is annotated rather than plotted as a bar because it is several times
the tallest finite bin and would flatten the distribution the panel exists to show.

---
**`10B_graze_pca` — §10B.** Same encoding as `10A_pca_clusters` — x = PC1, y = PC2, colour = cluster,
shape = condition — over the graze feature set. Cells that never responded in an ON window are held
out.
**Why:** `first_trial` is undefined for a cell that never responded, and filling it in would invent
the very quantity the axis exists to measure. The printed loadings say what PC1 and PC2 are actually
made of, which matters here because `trial_slope` is weak on this dataset.

---
**`11B_graze_transitions` — §11B.** x = SNI phenotype · y = Baseline phenotype · **grid value = count
of paired ROIs**, over the 5 graze labels.
**Why:** same logic as 11A, on a label that actually carries information for this protocol. The
printed marginals accompany it because the two label axes (timing, shape) move independently.

---
**`12B_graze_spectrum` — §12B.** Same axes and colour meaning as `12A_spectrum`, with two changes:
**red bars above and below the panel mark each of the 12 ON windows** (plus faint white hairlines at
each onset), and the row-ordering weight comes from the **constrained** shift scored over the **ON
windows only**.
**Why:** the 12 cycles are the structure of this protocol and a single 120 s bar hides them. The
markers stay in the **margin** deliberately — a shaded band across the data tints the heatmap and
costs exactly the contrast the figure exists to show.

---
**`13a_cycle_response` — §13.** x = graze cycle 1–12 · y = **fraction of paired ROIs with a detected
event** in that cycle · one line per condition · *left panel* = the 5 s ON window, *right panel* =
the 5 s gap window as a **control**.
**Why:** the direct test of claim 1. The gap panel is the one discriminator available: a trend
confined to the ON window is at least stimulus-locked, while a trend in both points at drift
affecting the whole recording.

---
**`13b_cycle_response_split` — §13.** Same axes, split by whether the cell responded in **any**
Baseline ON window.
**Why:** isolates the "previously low-response" group. **Read the right panel carefully:** that group
is *defined* as silent in every Baseline ON window, so its Baseline line is 0 at every cycle **by
construction** — drawn dotted and annotated to mark it. Only the SNI line there is a measurement.

---
**`14c_chronic_latency` — §14.** *Left:* x = Baseline latency to the first `S3` event (s) · y = SNI
latency (s) · one point per paired cell that has an `S3` event on **both** days · dashed diagonal =
no change, so **above the diagonal = later after SNI** · dotted red horizontal = the end of the SNI
recording. *Right:* x = SNI label · y = number of cells · **green = still chronic, grey = chronic bit
lost**.
**Why:** a paired scatter shows per-cell movement that a pair of medians would average away, and the
"end of recording" line makes the censoring visible — everything above it is unobservable, which is
exactly why "delayed beyond the recording" is untestable.

---
**`15_coactivation` — §15.** x = time (s) · y = **number of cells simultaneously inside a detected
event** · coloured line = observed · dark line = null mean · **grey band = the 2.5–97.5 percentile of
the shift null** · light red shading = stimulus on.
**Why:** a synchronised-cluster claim needs the observed trace **above the band**. A high
co-activation count on its own is expected whenever many cells respond to the same stimulus, so the
band is the whole point. Frames where nothing was measured are masked rather than drawn as 0.
Compare the exceedance percentage against the band's own chance rate (**2.5 %**), not against 0.

### 6. Parameters worth knowing

All in the Config cell. The ones that change conclusions rather than cosmetics:

| parameter | effect |
| --- | --- |
| `DETECT_K`, `N_CONSEC` | The detection rule. Lower `K` or raise `N_CONSEC` to trade height for duration — the small-and-sustained phenotype is the one a height-based rule erases |
| `SEGMENT_EDGES_S` | clip's segment boundaries. graze's are generated from the protocol constants |
| `CYCLE_EXCLUSION_FRAMES` | How close to a whole cycle a graze offset may come. `None` = half the ON window. Larger = stricter null, smaller offset pool |
| `SPECTRUM_N_SURROGATES`, `..._GRAZE` | p-value resolution (`1/(n+1)`) and, via `SHIFT_POOL_FACTOR`, whether the guard passes |
| `EARLY_MAX_TRIAL`, `ON_GAP_HIGH` | The graze phenotype cuts. Declared — judge them against the `9B_graze_axes` distribution |
| `CLUSTER_K` | Number of k-means clusters. Not selected for you |
| `FOVS` | `("FOV3",)` today. Every section already loops over `FOVS` and pairs within a field |
| `SNI_VARIANT`, `STIM_TERRITORY` | §12d. Free-text provenance printed in every caption from §12d on. **Currently `"unspecified"` — fill these in** |

### 7. What this notebook cannot answer

Stated so the figures are not read as implying otherwise.

- **Cell type / neuron subtype.** No marker channel, no genetic label, no post-hoc stain registered
  to the ROIs. A claim that an effect is specific to nociceptors, LTMRs or proprioceptors can be
  neither supported nor refuted here. ROI size is not a subtype proxy — it comes from the
  segmentation, not from soma diameter.
- **Which nerve carries which response.** Needs the SNI variant and the stimulated skin territory,
  neither of which is in the files. Both are config strings, typed in and printed, not inferred.
  **Note also that the loaded traces are named `RightDRG_FOV3_…` while `TRACE_ROOT` points at
  `…/leftDRGcut`.** That inconsistency is in the source data, not the code; confirm laterality
  against the lab record before any territory claim.
- **A chronic response in graze.** The window after the 12th cycle is 9.9 s. graze can speak about
  `on_gap_ratio` — activity persisting into the 5 s gap after each graze — and that is a different
  statement from clip's chronic response.
- **Cells going silent.** Detection is one-sided (§3.1).
- **Whether a signal was replaced by another FOV.** Needs the other fields loaded; `FOVS` is
  `("FOV3",)`.
- **Anything about mice.** One animal, one session, one field. Every percentage has *cells* for its
  denominator, and the two days are two recordings rather than two samples.
