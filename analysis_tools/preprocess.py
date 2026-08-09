"""Denoising, background normalisation and rigid motion correction.

Lifted out of the notebook so a GUI can drive the same chain the notebook does. Every
stage is per-frame and wired together lazily, so the full (n_timepoints, H, W) float32
stack is never held in RAM at once -- only when something computes it, one frame per
chunk.

Parameters travel as a `PreprocessParams` instance rather than module globals, so the
same values that produced a stack are the ones written into its `.params.json` sidecar.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
# from scipy.fft import dctn, idctn
from scipy.ndimage import gaussian_filter
from scipy.ndimage import shift as nd_shift
from skimage.registration import phase_cross_correlation
from skimage.restoration import denoise_nl_means, estimate_sigma

DENOISE_METHODS = (
    "gaussian",
    "nlm", 
    # "dct", (Deprecated)
    )


@dataclass
class PreprocessParams:
    """Every knob of the preprocessing chain, with the notebook's defaults."""

    denoise_method: str = "nlm"          # comma-separated, applied in sequence
    denoise_sigma: float = 3.0           # gaussian
    nlm_patch_size: int = 5              # nlm: patch size (px) for patch comparison
    nlm_patch_distance: int = 6          # nlm: max search distance (px)
    nlm_h_factor: float = 1.25           # nlm: cut-off, scaled by the estimated sigma
    # dct_threshold_fraction: float = 0.05  # dct: deprecated, see denoise_dct below
    background_percentile: float = 4      # per-frame percentile subtracted
    upsample_factor: int = 10            # sub-pixel precision of motion correction

    def methods(self):
        """The denoise chain as a validated list, e.g. 'gaussian,nlm' -> [...]."""
        names = [m.strip() for m in str(self.denoise_method).split(",")]
        names = [m for m in names if m]  # drop empties from stray/trailing commas
        if not names:
            raise ValueError(f"denoise_method must name at least one of {DENOISE_METHODS}, "
                             f"got {self.denoise_method!r}")
        unknown = [m for m in names if m not in DENOISE_METHODS]
        if unknown:
            raise ValueError(f"Unknown denoise method(s) {unknown!r}, expected one of "
                             f"{DENOISE_METHODS}")
        return names

    def to_sidecar(self):
        """The notebook's `.params.json` key names, for `store.save_preprocessed`."""
        return {
            "DENOISE_METHOD": self.denoise_method,
            "DENOISE_SIGMA": self.denoise_sigma,
            "NLM_PATCH_SIZE": self.nlm_patch_size,
            "NLM_PATCH_DISTANCE": self.nlm_patch_distance,
            "NLM_H_FACTOR": self.nlm_h_factor,
            # "DCT_THRESHOLD_FRACTION": self.dct_threshold_fraction,  # deprecated
            "BACKGROUND_PERCENTILE": self.background_percentile,
            "UPSAMPLE_FACTOR": self.upsample_factor,
        }

    def as_dict(self):
        return asdict(self)


# ----------------------------------------------------------------------- denoisers

def denoise_gaussian(frame, sigma):
    return gaussian_filter(frame, sigma=sigma)


def denoise_nlm(frame, patch_size, patch_distance, h_factor):
    # https://scikit-image.org/docs/stable/auto_examples/filters/plot_nonlocal_means.html
    sigma_est = estimate_sigma(frame, channel_axis=None)
    return denoise_nl_means(
        frame,
        h=h_factor * sigma_est,
        sigma=sigma_est,
        fast_mode=True,
        patch_size=patch_size,
        patch_distance=patch_distance,
        channel_axis=None,
    )

'''
def denoise_dct(frame, threshold_fraction):
    """Whole-frame 2D DCT, hard-thresholding small-magnitude AC coefficients.

    The DC coefficient (frame mean intensity) is excluded from the threshold scale and
    never dropped -- it is usually 100-1000x larger than any AC term, so including it
    made the threshold zero out nearly the whole AC spectrum and collapsed the frame to
    a flat constant.
    """
    coeffs = dctn(frame, norm="ortho")
    ac_max = np.abs(coeffs.flat[1:]).max()
    mask = np.abs(coeffs) < threshold_fraction * ac_max
    mask[0, 0] = False
    coeffs[mask] = 0
    return idctn(coeffs, norm="ortho")
'''


def denoise_frame(frame, params):
    """Apply the configured denoise chain to one frame, in order."""
    for name in params.methods():
        if name == "gaussian":
            frame = denoise_gaussian(frame, params.denoise_sigma)
        elif name == "nlm":
            frame = denoise_nlm(frame, params.nlm_patch_size, params.nlm_patch_distance,
                                params.nlm_h_factor)
        # else:
        #     frame = denoise_dct(frame, params.dct_threshold_fraction)
    return frame


def normalise_to_background(frame, background_percentile):
    """Subtract a per-frame low percentile as background."""
    return frame.astype(np.float32) - np.percentile(frame, background_percentile)


# ------------------------------------------------------------------- the full chain

def reference_frame(first_frame, params):
    """The fixed target for rigid motion correction -- computed once, kept in memory."""
    return normalise_to_background(
        denoise_frame(first_frame.astype(np.float32), params), params.background_percentile)


def preprocess_frame(path, reference, params):
    """Load one frame and run it through the whole chain. Returns float32, clipped at 0."""
    frame = np.load(path)[0].astype(np.float32)
    frame = denoise_frame(frame, params)
    frame = normalise_to_background(frame, params.background_percentile)
    frame_shift, _, _ = phase_cross_correlation(reference, frame,
                                                upsample_factor=params.upsample_factor)
    frame = nd_shift(frame, frame_shift)
    return np.clip(frame, 0, None).astype(np.float32)


def build_stack(image_files, params, first_frame=None):
    """Lazy (n_timepoints, H, W) float32 stack of preprocessed frames.

    Nothing is computed here: the returned dask array carries one delayed frame per
    chunk, so it can be added to napari or streamed to Zarr without materialising.
    """
    import dask.array as da
    from dask import delayed

    if first_frame is None:
        first_frame = np.load(image_files[0])[0]
    reference = reference_frame(first_frame, params)

    lazy = delayed(preprocess_frame)
    frames = [
        da.from_delayed(lazy(f, reference, params), shape=reference.shape, dtype=np.float32)
        for f in image_files
    ]
    stack = da.stack(frames, axis=0)
    print(f"> Preprocessing chain '{params.denoise_method}' over {len(image_files)} frames "
          f"-> lazy {stack.shape} {stack.dtype}")
    return stack
