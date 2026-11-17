"""All fixed numbers of the analysis, in one place.

Two kinds of settings live here:

* **Constants** of the observation and geometry (epochs, UVW axes, grid step).
  These describe the data and are not choices.
* **Analysis choices**, collected in :class:`Config`.  ``Config()`` is the
  analysis in the paper; the sensitivity analysis (notebook 06) varies one
  field at a time with ``Config(...).replace(field=value)``.
"""
from __future__ import annotations

import dataclasses
import json
import os

import numpy as np

# --------------------------------------------------------------------------- observation
#: JWST NIRCam F444W visits, in time order, with their mid-exposure MJD.
EPOCHS = (
    ('05451001001', 60541.3205653675),
    ('05451002001', 60569.0728464583),
    ('05451012001', 60583.2558483507),
    ('05451003001', 60599.204972636),
    ('05451004001', 60674.750816684),
    ('08527029001', 60870.2545367911),
    ('05451005001', 60886.4259551997),
)
N_EPOCHS = len(EPOCHS)
#: Telescope roll (PA_V3, deg) of each epoch, in the order above; sets the PSF orientation.
ROLLS_DEG = (197.8, 173.0, 163.0, 139.9, 60.8, 229.9, 217.4)
#: The 3D positions of the scattering surface are tabulated for two reference
#: dates; position at an epoch is linearly interpolated in time from MJD_BASE.
MJD_BASE = 60310
#: Image extensions of the star/galaxy-removed mosaics.
INFILL_HDUS = {'draw1': 'INFILL_DRAW_1', 'draw2': 'INFILL_DRAW_2', 'draw3': 'INFILL_DRAW_3', 'mean': 'INFILL_MEAN'}

# --------------------------------------------------------------------------- geometry
#: Unit vectors of the echo frame, in the Cas A-centred (x, y, z) frame [ly].
#: W is the normal to the light-echo paraboloid at Mab; U and V span the echo plane.
U_VEC = np.array([0.87914841, 0.11371786, 0.46278107])
V_VEC = np.array([-1.38777878e-17, -9.71110979e-01, 2.38628301e-01])
W_VEC = np.array([0.47654808, -0.20978969, -0.85375067])
#: UV grid step [ly]; equals the 60 mas mosaic pixel at Mab.
PIX_LY = 0.0032
#: Mosaic pixel scale [arcsec].
PIX_ARCSEC = 0.060

# --------------------------------------------------------------------------- regions
#: Region 1 and Region 2: centre (row, col) on the UV grid of the 150 px boxes.
REGION_CENTERS = {1: (1939, 2167), 2: (2191, 1713)}
REGION_BOX = 150
#: Region 2 is followed through depth (co-moving window); Region 1 is fixed.
REGION_TRACKED = {1: False, 2: True}


def cloud_polygon_uv():
    """Hand-drawn polygon (U, V) [ly] around the cloud, used only to pick empty sky."""
    p = os.path.join(os.path.dirname(__file__), 'cloud_polygon_uv.json')
    return np.asarray(json.load(open(p))['vertices_uv'])


# --------------------------------------------------------------------------- analysis choices
@dataclasses.dataclass(frozen=True)
class Config:
    """Analysis choices.  Defaults reproduce the paper.

    Data
        epochs        indices of the epochs used
        infill        which star/galaxy infill realisation ('draw1', 'draw2', 'draw3', 'mean', 'mask')
        norm          per-epoch flux normalisation: 'window' (rms of each window),
                      'global' (one factor per epoch over the cloud), 'none'
    Sample
        size          window side [px] (150 px = 0.48 ly)
        shift         offset (rows, cols) of the tiling grid from the Region 1 corner
        on_k          structure-to-noise threshold of the cloud region
        smooth        Gaussian smoothing [px] of the structure-to-noise map
        frac          minimum fraction of a window inside the cloud region
        centres       'tiles' (regular tiling) or 'same' (paper centres, other size)
    Fit
        rcut          maximum 3D lag used in the fit [ly]
        rmin          minimum 3D lag [ly] (3 px)
        weighting     lag weighting: '1/r', '1/r2' or None
        K             jackknife blocks per side (K x K delete-one-block)
        inner_uv      half-width [px] of the lag region passed to the fit
        stride        use every `stride`-th lag pixel in U and V
    PSF
        sigma         extra Gaussian (arcsec) added to the WebbPSF model
        calib         PSF bias calibration: 'dedup' (default) or 'published'
                      (the table as first run, with duplicated synthetic fields)
        t2min         minimum |transfer|^2 kept in the Fourier deconvolution
    Statistics
        degen         a fit 'collapses' when a2/a1 or a3/a2 < degen
        collapsed     'include' collapsed fits in statistics, 'exclude' them
    """
    epochs: tuple = tuple(range(N_EPOCHS))
    infill: str = 'draw1'
    norm: str = 'window'
    size: int = 150
    shift: tuple = (0, 0)
    on_k: float = 2.0
    smooth: float = 25
    frac: float = 0.9
    centres: str = 'tiles'
    rcut: float = 0.1
    rmin: float = 0.0
    weighting: str | None = '1/r'
    K: int = 4
    inner_uv: int = 45
    stride: int = 2
    sigma: float = 0.0325
    calib: str = 'dedup'
    t2min: float | None = 0.05
    degen: float = 0.02
    collapsed: str = 'include'

    def replace(self, **kw):
        return dataclasses.replace(self, **kw)

    def fit_tag(self):
        """Short name for the cache directory of the per-window fits.

        Only the choices that change a fit enter the tag, so configurations that
        differ only in sample or PSF calibration share fits.
        """
        d = Config()
        parts = []
        for k in ('epochs', 'infill', 'norm', 'size', 'rcut', 'weighting', 'K', 'inner_uv', 'stride'):
            v = getattr(self, k)
            if v != getattr(d, k):
                v = ''.join(map(str, v)) if isinstance(v, tuple) else str(v).replace('/', '')
                parts.append('%s-%s' % (k, v))
        return 'base' if not parts else '_'.join(parts)


#: Low-level fit settings that are not analysis choices.
FIT_KW = dict(min_same_epoch_lag_pix=4, s2_floor=10 ** -3.75, noise_scale_dex=0.1, min_n_fraction=0.1)
#: Preprocessing of the structure function (raw flux, percentile clipping, pooled mean).
S2_KW = dict(clip_percentiles=(0.002, 0.998), background=0.0, arcsinh_scale=None,
             subtract_mean='global', assume_stationary=True)
#: Window reading: erosion radius [px] around missing data, minimum epoch coverage.
READ_KW = dict(edge_mask_radius=50, min_coverage=0.25)
#: In-plane parameters held fixed in the single-epoch (dW = 0) fit.
FREEZE_2D = ('s33', 'l13', 'l23')
