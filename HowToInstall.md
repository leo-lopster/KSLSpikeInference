# Installing the Spike Inference GUI

`spike_inference_gui.py` is the whole application: a Qt window with the six pipeline
stages, plus a napari viewer alongside it for the image and ROI work. Everything it
needs comes from `environment.yml` — there is nothing to install separately.

## Requirements

- **Anaconda or Miniconda** on the machine.
- **A display.** napari renders through OpenGL, so the GUI cannot run headless or over a
  plain SSH session.
- **~2.5 GB of disk** for the conda environment (`torch` and the Qt stack dominate).
- The repository **including `CascadeTorch/`** — the CASCADE code is vendored there.
  The **model weights are not**: `.gitignore` excludes them, so a fresh clone has an
  empty `Pretrained_models/` holding only the index of download links. See
  [Getting the models](#getting-the-models) below.

### napari does not need to be installed separately

`environment.yml` installs `napari[pyqt6]`, so `conda env create` brings in napari, PyQt6
and the whole Qt/OpenGL stack with it. Verified on macOS: napari 0.7.1 on Qt 6.10 /
PyQt6 6.10.2, rendering through Metal, opening a viewer and taking image and label layers
from the GUI with no independent napari install anywhere on the system. If you have
napari installed elsewhere — a standalone bundle, another conda env, `pipx` — it is
neither used nor needed; the GUI only ever sees the copy inside the `cascade`
environment.

## macOS

Run the following in Terminal, in sequence:

1. Run: (replace "..." with your actual directory)

        cd .../KSLSpikeInference

2. Create the working environment using Anaconda/Miniconda (a few minutes — it downloads torch and Qt):

        conda env create -f environment.yml

3. Activate it:

        conda activate cascade

4. Run the GUI:

        python spike_inference_gui.py

   Two windows open: the pipeline window and the napari viewer.

## Windows

*[To Be Implemented]* — `environment.yml` carries no platform-specific pins, so the same
three commands are expected to work, but nothing on Windows has been tested.

## Checking the installation

With the environment active:

    napari --info

This prints the napari, Qt and OpenGL versions it resolved. If it reports a GL version
and a screen, the graphics half of the stack is fine. If it fails, the GUI will not
start either.

## First run

In **tab 5 · Spike inference**, set `CASCADE_DIR` to the `CascadeTorch` folder inside
this repository. `MODEL_NAME` then fills from `CascadeTorch/Pretrained_models/`, and
`CASCADE_DIR` is also required by the tab 6 analysis, which loads its FFT helper from
`CascadeTorch/scripts/`.

The other path fields (`PREPROCESSED_DIR`, `LABELS_DIR`, the analysis root) start empty on
purpose and are yours to point wherever the data lives.

### Getting the models

On a fresh clone `MODEL_NAME` is empty. Use **Download
pretrained models**, in the same tab:

1. Set `CASCADE_DIR` first; the list is read from
   `CascadeTorch/Pretrained_models/available_models_CascadeTorch.yaml`.
2. Pick a model and press **Download selected model**. It unpacks into
   `Pretrained_models/<name>/` and is selected in `MODEL_NAME` when it finishes.
   Models already on disk are marked `✓`, and the line underneath counts how many of
   the listed models are installed.
3. `Spinal_cord_excitatory_30Hz_smoothing50ms` is the project default — the nearest
   available analogue for DRG, and an untested transfer. See `CLAUDE.md` §4C before
   quoting anything it produces.

Downloading needs no traces or ROIs, so it works on a first launch with nothing else
loaded. Each model is a few MB to tens of MB. Re-downloading an installed model asks
before replacing it, and the local copy is only removed once the new one has downloaded
and unpacked cleanly.

## Updating

If any changes are made to `environment.yml`, update in place rather than recreating:

    conda env update -f environment.yml --prune

## Notes

- **The Jupyter notebooks are not supported on this branch.** `environment.yml` declares
  nothing on their behalf — no Jupyter, no `ipykernel`, no `SlideBook_API`. Use the GUI.
- **`cascade2p` is vendored** in `CascadeTorch/`, not installed from PyPI. Do not
  `pip install cascade2p` into this environment; it would shadow the local fork.
- **`pip` is a runtime dependency, not just an installer.** `cascade2p/config.py` imports
  `pip._internal` when it loads, so removing pip from the environment breaks CASCADE.
