# Project Context: Spike Inference — Sensory Afferent → DRG → DCN Pathway (Mouse)

Purpose: reference context for any coding session in this project. Covers the
biological basis of the calcium signal in this pathway, computational spike
inference methods, and data presentation conventions. Treat the "Practical
defaults" sections as starting points, not fixed rules — confirm against the
specific dataset/indicator/frame rate before locking in a method.

## 1. Biological basis of the calcium signal

- DRG primary sensory neurons are pseudounipolar; the soma does not perform
  dendritic-style synaptic integration. Somatic Ca2+ transients largely track
  backpropagating action potentials via voltage-gated Ca2+ channels (N-, T-,
  L-type), so spike-to-calcium coupling is closer to linear than in cortical
  pyramidal neurons — but burst firing in nociceptors/mechanoreceptors at high
  frequency causes nonlinear summation and indicator saturation.
- DCN (gracile/cuneate) neurons are second-order. Calcium signal there
  reflects a mix of synaptic Ca2+ entry (NMDA-R, VGCC) and AP-driven somatic
  transients — less spike-faithful, especially if imaging boutons/dendrites
  rather than soma.
- Caveat for modeling: most public ground-truth (spike + simultaneous Ca2+)
  datasets used to calibrate inference tools (Allen Institute, GENIE,
  spikefinder, CASCADE training corpus) come from cortex/hippocampus
  pyramidal cells. Kinetics and noise models tuned on those may not transfer
  cleanly to DRG/DCN — treat this as an open assumption, not a validated one.

## 2. Indicators & acquisition

- jGCaMP7/8 family: "s" variants = higher sensitivity, slower off-kinetics;
  "f" variants = faster off-kinetics, lower sensitivity. Prefer 8f/8m for
  spike-resolved inference in fast-firing afferents.
- Typical DRG/DCN in vivo frame rates: 15–30 Hz (resonant scanning), up to
  60–100 Hz with restricted ROI or line-scan acquisition for higher temporal
  fidelity on fast afferents.

## 3. Preprocessing pipeline (recommended order)

1. Motion correction — rigid + piecewise (NoRMCorre or suite2p registration).
   DRG prep is especially prone to pulsatile/respiratory motion artifact.
2. ROI segmentation — suite2p or CaImAn (CNMF).
3. Neuropil/background subtraction — important in DCN (densely packed somata);
   less critical in sparse DRG GRIN-lens preps.
4. ΔF/F0 normalization — F0 as rolling low-percentile (e.g. 8th pct) baseline
   over a 30–60 s window, to handle slow drift.
5. Detrend slow drift (photobleaching, lens settling) *before* spike inference.

## 4. Computational spike inference methods

### A. Deconvolution-based (linear, time-invariant kernel)
- **OASIS** (Friedrich, Zhou & Paninski, 2017) — constrained FOOPSI via fast
  active-set solver. Default in suite2p/CaImAn. Fast, minimal assumptions.
- **Constrained FOOPSI** (Vogelstein et al., 2010) — L1-penalized
  deconvolution with non-negativity constraint.
- Breaks down under burst/saturating dynamics common in nociceptor afferents.

### B. Biophysical / generative model-based
- **MLspike** (Deneux et al., 2016) — explicit model of Ca2+ binding and
  indicator saturation; EM/particle-filter inference. Preferred when burst
  firing causes nonlinear saturation (high-frequency mechanoreceptors).

### C. Supervised / deep learning
- **CASCADE** (Rupprecht et al., 2021, *Nat Neurosci*) — CNN ensemble
  pretrained on imaging+ephys ground truth across cell types/indicators/frame
  rates. Outputs spike *probability*, not discrete events.
- Open caveat: no DRG- or DCN-specific training set currently exists in its
  library — nearest analogues are peripheral/cortical sensory sets. Treat
  cross-cell-type generalization as untested, and say so in any writeup.

### D. Bayesian / sequential
- Particle filtering / sequential Monte Carlo (used internally by MLspike) —
  gives explicit timing-uncertainty estimates, useful for low-SNR DRG traces.

### Practical defaults for this pathway
- Sparse, high-SNR DRG soma, moderate firing rate → OASIS/FOOPSI (cheap,
  scales to large populations).
- Burst-firing nociceptive afferents / suspected saturation → MLspike.
- Cross-study comparisons needing continuous rate rather than discrete spikes
  → CASCADE, with the training-set caveat noted above stated explicitly.
- Always log method + parameters used; outputs from different algorithms are
  not directly comparable without a parameter-sensitivity check.

## 5. Data presentation conventions

### Trace-level
- Show raw F and ΔF/F together when introducing a recording — never replace
  the raw trace with inferred spikes in a validation figure.
- Inferred output rendered as either (a) continuous spike-probability trace
  overlaid on ΔF/F (lighter line weight/alpha), or (b) discrete spike raster
  after thresholding — state the threshold explicitly; it's not neutral.

### Population-level
- Heatmap of trial-aligned ΔF/F or inferred rate per neuron, sorted by
  latency or peak response.
- PSTH-style plots: convert to spike-probability/binned rate *before*
  trial-averaging (raw ΔF/F kinetics distort latency if averaged directly).
  Trial-averaged trace + shaded SEM/CI band.
- Raster plots: one row per neuron or per trial, ticks at thresholded events.

### Plotting conventions for this codebase
- matplotlib/seaborn; avoid `jet` (perceptually non-uniform) — use
  `viridis`/`magma` for heatmaps.
- Raw vs. inferred traces go in separate subplot rows, not overlaid without
  distinct linestyle/alpha.
- Every figure/legend showing inferred spikes states which algorithm +
  parameters produced it — never present inferred spikes as ground truth
  without that label in the same panel.

## 6. Open questions to flag in any analysis writeup

- Whether the linear-kernel assumption behind common deconvolution tools
  (calibrated on cortex) holds for pseudounipolar DRG somata vs. postsynaptic
  DCN neurons — largely untested directly.
- No pathway-specific simultaneous Ca2+/ephys ground-truth dataset exists for
  DRG or DCN; all current benchmark tools are calibrated on CNS pyramidal
  neuron preparations.
- No field consensus on principled threshold selection for converting
  continuous spike-probability traces into discrete event counts.

---

# Appendix A: Literature-review additions

Appended from a focused review of recent high-frequency / model-based spike
inference work. Cross-references existing sections by number. Same caveat
applies: starting points, confirm against the specific dataset/indicator/frame
rate.

## 7. Method additions (extends §4)

### §4B addition — Biophysical generative + SMC/ML (handles GCaMP8 nonlinearity)
- **BiophysSMC / BiophysML** (Broussard, Diana, Quiroz, Sermet, Lynch,
  DiGregorio*, Wang*, 2025; bioRxiv 2024.12.31.630967, "Precise calcium-to-spike
  inference using biophysical generative models").
  - Fits a **multistate GCaMP model** (bright-fast, bright-slow, and dark/slow
    non-fluorescent states). This explicitly reproduces GCaMP8f **use-dependent
    slowing** of fluorescence decay — the decay τ is history-dependent, most
    apparent after ~10 APs — which single-kernel methods absorb as noise and
    convert into false positives during/after bursts.
  - Two methods: **BiophysSMC** (Bayesian sequential Monte Carlo, unsupervised)
    and **BiophysML** (faster supervised surrogate, trained on *synthetic* data
    generated from the validated biophysical model).
  - Reaches ~4 ms median spike-time uncertainty = the theoretical (Cramér–Rao)
    limit, ~2× better than prior methods; best across all timescales for
    excitatory cells.
  - Why it matters here: the **synthetic-data route sidesteps the ground-truth
    bottleneck** — generate indicator-/cell-appropriate training data without
    paired DRG/DCN ephys (see §10). The indicator state-model is chemistry, so
    it should transfer across cell types; the cell-specific parameters are what
    must be re-fit.

### §4D addition — Particle Gibbs / fully-Bayesian, uncertainty-aware
- **PGBAR** (Diana, Sermet, Broussard, Wang, DiGregorio, 2026; *eLife*
  13:RP94723, DOI 10.7554/eLife.94723). "Particle Gibbs with ancestor sampling
  (PGAS) on a Bursting AutoRegressive (BAR) model."
  - AR(2) calcium kernel + **explicit two-state (high/low) bursting process** +
    Gaussian-random-walk baseline + Poisson spikes.
  - Headline feature: **jointly infers spike times AND all model parameters**
    (amplitude, rise/decay, noise, baseline variance, firing rates) in one
    Bayesian sweep — no separate calibration/grid-search, **no training data**,
    and returns **full posteriors → calibrated uncertainty** on both spikes and
    parameters. This is the main differentiator vs every method in §4A–C.
  - Accuracy on CASCADE benchmark (Pearson r to ground truth, 200 ms Gaussian
    kernel): mean **r = 0.75**; slightly below MLSpike (within 6%, p=0.0043,
    attributed to MLSpike modelling indicator nonlinearity); **CASCADE is best
    overall but is supervised and gives no posteriors.**
  - High-frequency resolution: resolves **inter-spike intervals down to 5 ms**,
    validated on cerebellar granule cells (GCaMP8f, ~3 kHz 2P linescan) — for a
    5.3 ms interval at low SNR (~2.4), 100% of posterior samples contained two
    spikes. Single-trial temporal accuracy 0.43 ms (boutons) / 1.39 ms (somata);
    spike-count relative error ~15%; reliable to ~200 Hz firing.
  - **Caveat:** the 5 ms capability requires kHz-class acquisition + a fast
    indicator. On the 7.5 Hz-downsampled benchmark it is the r=0.75 regime, not
    5 ms. Temporal resolution is set by acquisition rate × indicator kinetics,
    not by the algorithm.
  - Limits: AR(2) is phenomenological and **mismodels indicator nonlinearity**
    (GCaMP6f example: 30% of 1-s bins outside the posterior IQR); Gaussian-noise
    assumption → non-Gaussian fluctuations misread as spikes, inflating inferred
    burst rate; particle-MCMC **compute cost** (per-cell, does not amortize).
- Note: PGBAR and BiophysSMC/ML are **sibling methods** from overlapping groups
  — phenomenological-Bayesian (self-calibrating, AR(2)) vs biophysical-generative
  (mechanistic, multistate, needs indicator characterization). Pick per the
  trade in §10.

### §4C enrichment — CASCADE corpus & GCaMP8 retuning
- CASCADE ground-truth corpus (Rupprecht et al., 2021) = zebrafish telencephalon,
  hippocampal CA3, neocortex (S1, V1 incl. interneurons) — **all CNS, none
  peripheral/pseudounipolar.** Confirms the §1/§4C transfer caveat concretely.
- Default CASCADE models are GCaMP6-tuned; applied to **GCaMP8** they
  misestimate rates at both high and low ends → **retraining/tuning required**.
  Reported optimal integration window ~30–40 ms for GCaMP8 vs ~50–150 ms for
  GCaMP6 (relevant if targeting closed-loop / low-latency use).

### Method-selection caveats (downsides of numerical/model-based inference)
- Most-accurate ≠ most-usable: SMC/particle-Gibbs are slowest and **per-cell**;
  they do not amortize to large populations (motivates supervised surrogates
  like BiophysML).
- A misspecified generative model yields **confidently wrong** output *with*
  credible intervals — wrong cortical priors on DRG/DCN propagate faithfully.
- **Non-identifiability:** amplitude/decay/baseline/noise trade off; different
  spike trains fit nearly equally → seed/initialization-dependent results.
- Hyperparameters (sparsity penalty, threshold, particle count) have large
  effects and **no principled setting without ground truth** — which we lack
  for this pathway.
- Point-estimate methods manufacture precision: they emit a number even where
  the indicator never encoded the spikes (e.g. >50 Hz, saturation). Prefer
  uncertainty-aware output for this pathway.

## 8. Ground-truth landscape (extends §1 caveat, §6)

- **Nearest non-cortical calibration = spinal cord dorsal horn**
  (Rupprecht, Fan, Sullivan, Helmchen, Sdrulla, 2025; *J Neurosci*
  45(18):e1187242025, DOI 10.1523/JNEUROSCI.1187-24.2025; bioRxiv
  2024.07.17.603957). Simultaneous **cell-attached + 2P** of glutamatergic and
  GABAergic superficial dorsal-horn neurons (VGlut2-Cre × Ai96, GCaMP6s).
  - Finding: cortex-trained **CASCADE and OASIS generalize *well*** to both
    excitatory and inhibitory dorsal-horn cells; **retraining still improves**
    accuracy. Retrained models openly provided for variable noise/frame rates.
  - This is the **best available prior for DCN** (second-order, conventional CNS
    relay — see §9). Same VGlut2-Cre driver as the SPARC visceral-DRG sets
    (those use Ai95/GCaMP6f), so kinetics differ (6s vs 6f) — adapt, don't reuse
    blindly.
  - Signal is often **burst-dominated** (isolated transients), reminiscent of
    cortical interneurons.
- **DRG:** no paired Ca²⁺/ephys ground truth exists. Hard bound: PV⁺
  proprioceptors fire >50 Hz during stretch and **individual spikes are not
  resolvable with 2P GCaMP6s** (eNeuro 2019, DOI 10.1523/ENEURO.0349-18.2019) —
  absolute-rate inference there is currently not feasible.
- **DCN (gracile/cuneate):** no dedicated spike-inference ground truth located;
  brainstem (dorsal medulla) access is motion-/depth-limited. Feasible
  validation template: subcortical sensory-relay GECI imaging of two
  genetically-defined classes (GABAergic + non-GABAergic) with simultaneous
  **juxtacellular** confirmation (cf. inferior colliculus two-photon studies).
- **Subcortical transfer principle:** porting a method across regions works
  better when its tuning parameters are biophysically interpretable (used in
  dopamine-neuron work to re-tune rather than retrain from scratch). [confirm
  exact citation before use in a writeup]

## 9. Pathway-specific biophysics refinements (extends §1)

- **DRG T-junction filtering / target ambiguity.** The §1 "closer to linear"
  picture holds at low rates, but the **T-junction acts as a frequency-dependent
  low-pass filter**: somatic AP invasion is conditional, so somatic GCaMP can
  report a *filtered* surrogate of the peripheral spike train, diverging from it
  exactly at the high rates of interest. Consequence: be explicit about the
  **inference target** — peripheral spike rate vs somatic-invasion rate vs
  evoked depolarization (these are not the same; see §11).
- **GCaMP8f use-dependent kinetic slowing** (named failure mode, see §7/§4B):
  history-dependent decay that grows during bursts; defeats fixed-kernel
  deconvolution. Argues for MLSpike / biophysical / bursting-aware methods in
  fast-firing afferents — directly relevant given §2 prefers 8f/8m.
- **DCN = conventional multipolar CNS relay** (principal/thalamus-projecting +
  local inhibitory interneurons). Unlike DRG it has **no pseudounipolar
  compartment pathology** — soma reports somatic spikes normally, so it sits in
  the same regime as dorsal horn (§8). The dominant modelling risk shifts from
  biophysics to **firing-statistics priors**: synaptic integration, lateral
  inhibition, center-surround, and corticocuneate top-down modulation make
  trains bursty/adapting/state-modulated (not Poisson). The "interneurons"
  target overlaps the DCN local-inhibitory class — treat principal vs
  interneuron as hierarchical cell-type levels, not separate problems.

## 10. Modelling strategy (synthesis)

- **Factorize the generative model** along three axes so transferable physics is
  fit once and cell-specific physics carries informative priors:
  1. a shared **indicator submodel** (e.g. multistate GCaMP8 kinetics) — fit
     once from in-vitro characterization, frozen across cell types;
  2. a **per-cell calcium-handling submodel** (resting Ca, ΔF/spike, decay) with
     **hierarchical priors keyed to cell type** (nociceptor / proprioceptor /
     LTMR / DCN-principal / DCN-inhibitory);
  3. for DRG only, an explicit **T-junction transmission stage** (frequency-
     dependent transmission) between latent peripheral train and somatic influx
     — keeps the filtering assumption visible rather than buried.
- **Bridge the ground-truth gap with synthetic data** (BiophysSMC/ML approach):
  train an amortized network on simulations from the factorized model with
  cell-type-appropriate priors; **validate against even a small real paired
  set**, since unvalidated synthetic priors produce confident, well-calibrated,
  wrong inference.
- **Acquisition sets the ceiling:** spike-resolved / short-ISI (≤5 ms) inference
  needs kHz-class linescan + a fast indicator (GCaMP8f). At 15–30 Hz population
  rates, target binned rate / burst detection, not single spikes (see §11).

## 11. Additional open questions (extends §6)

- **What is the inference target for a DRG soma** — peripheral spike rate,
  somatic-invasion rate, or evoked depolarization? They diverge under T-junction
  filtering and are routinely conflated.
- **Is DRG difficulty the cell or the morphology+regime?** Falsifiable: a
  cortex-trained model should fail *more* on large high-rate proprioceptors
  (strong T-junction filtering) than on small nociceptors. Equal failure ⇒
  compartment story is wrong, it's just saturation.
- **Above ~50 Hz**, is absolute-rate recovery off the table for any GECI+method,
  making honest deliverable = burst detection / relative rate?
- **ΔF/F amplitude is not a reliable rate proxy** (shown in dorsal horn) — do
  not use trace amplitude/variance as a stand-in for spike rate; treat as noise.
- **Accuracy vs uncertainty / phenomenological vs biophysical:** PGBAR buys
  self-calibration + posteriors but mismodels nonlinearity; BiophysSMC/ML buys
  mechanistic accuracy but needs indicator characterization. Which cost is
  payable for DRG/DCN is unresolved.

---

# Appendix B: How the literature frames method usage (DL vs model-based)

Condensed from a citation review of deep-learning vs. model-based spike inference and
the downstream papers that use these methods. Full version + sources:
`DL_vs_modelbased_spike_inference_review.md`. Scope: DRG (target) + mouse neocortex
(reference). Same caveat: confirm against the specific dataset/indicator/frame rate.

## 12. DL vs model-based, at a glance (extends §4)

- **Deep-learning/supervised** (CASCADE, ENS2, BiophysML, GCaMP8-retuned): most
  accurate *in-sample / on matched ground truth* (CASCADE top on its benchmark; ENS2
  ~34–36% higher correlation than MLSpike). Prominent wherever paired ground truth
  exists — neocortex L2/3 V1/S1, CA3, zebrafish. **Weak point: out-of-distribution.**
  spikefinder showed plain OASIS *beats* CNNs out-of-sample; no posteriors (CASCADE
  emits probability, not uncertainty).
- **Model-based** (OASIS/FOOPSI deconvolution; MLSpike, BiophysSMC, PGBAR
  biophysical/Bayesian): close behind on accuracy (MLSpike within ~6%; PGBAR r≈0.75;
  BiophysSMC hits the ~4 ms timing limit). Deployed more freely on novel/subcortical/
  DRG preps because they need no training set; interpretable params + (for Bayesian)
  calibrated uncertainty. **Weak point:** misspecified model → confidently-wrong
  output; particle methods are slow + per-cell (don't amortize).
- **Shared failure zone = high fluorescence / high rate.** Saturation makes CASCADE
  *underestimate* high-frequency rate; GCaMP8f use-dependent slowing makes single-
  kernel deconvolution (and default CASCADE) emit *false positives* after bursts.
  Running two methods that *agree on the wrong answer* during bursts is correlated
  error, not validation.

## 13. How citing papers convince reviewers (six moves, ranked)

Ranked by how much they actually *resolve* interpretation vs. merely persuade:

1. **Ground-truth validation / retraining** (gold standard). Dorsal-horn paper
   (Rupprecht/Sdrulla 2025): cortex-trained CASCADE/OASIS "generalize well," and
   retraining "improved retrieval of high-frequency spike events, was less biased in
   absolute rate, improved relative-rate prediction." DRG **cannot** make this move —
   no paired ground truth exists.
2. **Cross-method agreement** (OASIS+CASCADE converge). Weak where both share the
   same failure (saturation/slowing).
3. **Simplicity-as-safety** (Pachitariu/Stringer/Harris 2018). Choose plain NND:
   "simplicity, efficiency, accuracy"; warns "supervised methods can introduce
   artifactual structure … leading to erroneous scientific conclusions"; simple method
   is "insensitive to parameter choices, making incorrect conclusions much less
   likely." Also gave a **ground-truth-free benchmark** (correlation across stimulus
   repeats) — usable for DRG.
4. **Hedged readout** — report "deconvolved activity / inferred rate / relative rate,"
   never absolute counts. Doesn't solve the problem; scopes the claim below it.
5. **Amplitude-as-proxy** (DRG escape hatch). Emery et al. 2018: "GCaMP6 fluorescence
   intensity provides a useful proxy for frequency of action potential firing" — but
   only "above 1 Hz" via summation, and single APs "detected by eye … likely to be
   less precise." Anchored to a controlled spike train, claim kept at modality/relative
   level. **Tension with §11**: raw amplitude is *not* a rate proxy absent that anchor.
6. **Authority transfer** (anti-pattern, most common). Cite CASCADE's "outperforms
   model-based / generalizes across cell types," apply the GCaMP6 default to GCaMP8 or
   to pseudounipolar DRG, re-validate nothing. Exactly what the GCaMP8 + dorsal-horn
   papers show must be re-earned — persuasion without resolution.

## 14. Take-home for this pathway

No published paper *resolves* high-rate/saturating inference; the credible ones scope
the claim to what the indicator provably encoded — anchor a proxy to a controlled
spike train (Emery/DRG), pick the parameter-insensitive method and justify it
(Pachitariu/cortex), or acquire in-prep ground truth and quantify the residual failure
(dorsal horn). **For DRG the honest deliverable stays burst/relative rate, not absolute
spike counts** (consistent with §10–§11). The defensible build is still the §10 hybrid
— factorized biophysical generative model + synthetic-data surrogate — validated
against even a small real paired set.
