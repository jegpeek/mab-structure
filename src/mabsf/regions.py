"""Regions 1 and 2, the co-moving window, depth persistence and drift.

**The two example regions** (0.48 ly boxes):
* Region 1: long, straight filaments.  Fixed window.
* Region 2: a compact clump that keeps its shape but drifts in (U, V) from
  epoch to epoch.  It is followed with a *co-moving* window whose centre moves
  by (dU/dW, dV/dW) x (W_k - W_1); the drift is fitted from the epochs that
  register coherently with epoch 1.  (Region 1's filaments do not register
  coherently, so it cannot be tracked.)

For each region, the PSF-deconvolved in-plane S2 is averaged in nine 20 deg
wedges of lag direction (the 1D structure functions of the figure), and the
cross-epoch S2 gives the variation along W ('W points').  In a co-moving
window the epoch pairs are separated by dW sqrt(1 + |v|^2) in 3D.

**Trackability** (claim C6): how easily structure can be followed between
the closely spaced epochs 2, 3 and 4 (dW = 0.024-0.047 ly) -- the mean
correlation of their 0.01-0.1 ly structure maps, each pair at its best shift
(<= 12 px).  It is compared with in-plane anisotropy, the 3D inclination and
the in-plane width of the structures.

**Drift** (claim D1): for every window, the (dU/dW, dV/dW) that maximises the
correlation of the epoch pairs 0.015-0.1 ly apart in depth.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import label
from scipy.signal import fftconvolve

from . import config as C
from . import data as D
from . import geometry as G
from . import preprocess as P
from . import psf

ANGLES = np.linspace(0, 180, 9, endpoint=False)     # wedge centres [deg from +U toward +V]
ANGLE_WIDTH = 10                                     # wedge half-width [deg]
TRACK_EPOCHS = (1, 2, 3)                             # epochs 2, 3, 4 (0-based)
TRACK_PAIRS = ((1, 2), (2, 3), (1, 3))
MAXSHIFT = 12                                        # px


def box_corner(i):
    r, c = C.REGION_CENTERS[i]
    return r - C.REGION_BOX // 2, c - C.REGION_BOX // 2


# =========================================================================== tracking
def global_shift(p1, p2):
    """Integer shift (dx, dy) of p2 relative to p1 and the peak normalised correlation (Hann-tapered)."""
    p1 = p1 - np.nanmean(p1); p2 = p2 - np.nanmean(p2)
    h, w = p1.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    p1, p2 = p1 * win, p2 * win
    cc = fftconvolve(p1, p2[::-1, ::-1], mode='same')
    nrm = np.sqrt(np.sum(p1 ** 2) * np.sum(p2 ** 2))
    if nrm > 0:
        cc /= nrm
    ymax, xmax = np.unravel_index(np.argmax(cc), cc.shape)
    cy, cx = np.array(cc.shape) // 2
    return xmax - cx, ymax - cy, cc.max()


def fit_velocity(i=2, cc_min=0.45):
    """Drift (dU/dW, dV/dW) of a region: least-squares line through the origin of the
    displacement vs depth of every epoch that registers with epoch 1 at correlation >= cc_min."""
    cube, w = D.uv_cube('draw1'), D.epoch_w()
    v_c, u_c = C.REGION_CENTERS[i]; b = C.REGION_BOX
    base = np.nan_to_num(np.asarray(cube[:, v_c - b // 2:v_c + b // 2, u_c - b // 2:u_c + b // 2], float))
    dW, du, dv, used = [], [], [], []
    for k in range(base.shape[0]):
        dx, dy, cc = global_shift(base[0], base[k])
        if cc >= cc_min or k == 0:
            dW.append(w[k] - w[0]); du.append(-dx * C.PIX_LY); dv.append(-dy * C.PIX_LY); used.append(k)
    dW, du, dv = map(np.array, (dW, du, dv))
    return float(np.sum(du * dW) / np.sum(dW ** 2)), float(np.sum(dv * dW) / np.sum(dW ** 2)), used


def region_cube(i, velocity=None):
    """(7, 150, 150) raw flux of region i (NaN -> 0); co-moving if a velocity is given.

    Returns (cube, centres) with the box centre (row, col) used at each epoch.
    """
    cube, w = D.uv_cube('draw1'), D.epoch_w()
    v_c, u_c = C.REGION_CENTERS[i]; half = C.REGION_BOX // 2
    vu, vv = velocity if velocity is not None else (0.0, 0.0)
    out = np.empty((cube.shape[0], C.REGION_BOX, C.REGION_BOX)); cen = []
    for k in range(cube.shape[0]):
        dW = w[k] - w[0]
        vk, uk = v_c + int(round(vv * dW / C.PIX_LY)), u_c + int(round(vu * dW / C.PIX_LY))
        out[k] = np.nan_to_num(np.asarray(cube[k, vk - half:vk + half, uk - half:uk + half], float))
        cen.append((vk, uk))
    return out, cen


def region_jacobian(i):
    return G.pixel_to_uv_jacobian(*box_corner(i), C.REGION_BOX)[0]


# =========================================================================== structure functions
def wedge_profiles(sf2, angles=ANGLES, width=ANGLE_WIDTH, nbins=40, r_bins=None):
    """Mean of a 2D S2 map in +-width wedges (both lobes) and log-spaced radial bins.

    Returns (r_mid, profiles (n_angles, n_bins)).  Default bins: 39 bins from
    one pixel to the largest lag along both axes.
    """
    ny, nx = sf2.shape
    u = np.linspace(-(nx // 2) * C.PIX_LY, (nx // 2) * C.PIX_LY, nx)
    v = np.linspace(-(ny // 2) * C.PIX_LY, (ny // 2) * C.PIX_LY, ny)
    LU, LV = np.meshgrid(u, v); R = np.hypot(LU, LV)
    T = np.degrees(np.arctan2(LV, LU)); T[T < 0] += 360
    if r_bins is None:
        r_bins = np.logspace(np.log10(C.PIX_LY), np.log10(min(u[-1], v[-1])), nbins)
        r_mid = 0.5 * (r_bins[1:] + r_bins[:-1])
    else:
        r_mid = np.sqrt(r_bins[1:] * r_bins[:-1])
    prof = []
    for ang in angles:
        m = (((T >= ang - width) & (T <= ang + width)) | ((T >= ang + 180 - width) & (T <= ang + 180 + width)))
        prof.append([np.mean(sf2[m & (R >= r_bins[j]) & (R < r_bins[j + 1])]) for j in range(len(r_bins) - 1)])
    return r_mid, np.array(prof)


def region_sf(i, velocity=None, t2min=psf.T2MIN, correct=True):
    """PSF-deconvolved in-plane wedge profiles and W points of region i.

    Returns dict(cube, centres, J, sf2d, r, prof, w_sep, w_s2).
    """
    cube, cen = region_cube(i, velocity)
    J = region_jacobian(i)
    sf2 = psf.sf2d_epochs(cube, J, t2min=t2min, correct=correct)
    r, prof = wedge_profiles(sf2)
    scale = 1.0 if velocity is None else float(np.sqrt(1 + np.hypot(*velocity) ** 2))
    sep, s2w = psf.w_points(cube, D.epoch_w(), J, t2min=t2min, correct=correct, sep_scale=scale)
    return dict(cube=cube, centres=cen, J=J, sf2d=sf2, r=r, prof=prof, w_sep=sep, w_s2=s2w)


def _interp_log(r, p, x):
    ok = np.isfinite(p) & (p > 0)
    return 10 ** np.interp(np.log10(x), np.log10(r[ok]), np.log10(p[ok]), left=np.nan, right=np.nan)


def region_claims(res, r_range=(0.02, 0.03)):
    """C1-C4 numbers for one region (see docs/claims_inventory.md).

    aniso_*      : max / min over the wedges of S2 in radial bins within r_range
    w_over_max   : each W point / the largest wedge S2 at the same separation
    w_over_mean  : each W point / the mean over wedges at the same separation
    """
    r, Pm, sep, sw = res['r'], res['prof'], res['w_sep'], res['w_s2']
    m = (r >= r_range[0]) & (r <= r_range[1])
    ratio = np.nanmax(Pm[:, m], 0) / np.nanmin(Pm[:, m], 0)
    mx = np.array([max(_interp_log(r, p, s) for p in Pm) for s in sep])
    mean = np.array([np.nanmean([_interp_log(r, p, s) for p in Pm]) for s in sep])
    q1, q2 = sw / mx, sw / mean
    ok = np.isfinite(q1)
    return dict(aniso_min=float(ratio.min()), aniso_max=float(ratio.max()),
                w_over_max_median=float(np.median(q1[ok])),
                w_over_max_frac_0p5_1p5=float(np.mean((q1[ok] > 0.5) & (q1[ok] < 1.5))),
                w_over_mean_median=float(np.nanmedian(q2)), w_over_mean_min=float(np.nanmin(q2)), n_w=int(ok.sum()))


# =========================================================================== trackability
def ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def best_shift_ncc(a, b, m=MAXSHIFT):
    """Largest correlation of the central part of a with b over shifts |dx|, |dy| <= m px."""
    n = a.shape[0]; core = a[m:n - m, m:n - m]
    best = -1
    for dy in range(-m, m + 1):
        for dx in range(-m, m + 1):
            best = max(best, ncc(core, b[m + dy:n - m + dy, m + dx:n - m + dx]))
    return best


def acf_shape(img):
    """In-plane major and minor length [ly]: second moments of the autocorrelation's central lobe above 0.5."""
    x = img - img.mean(); n = x.shape[0]
    F = np.fft.fft2(x, s=(2 * n, 2 * n)); ac = np.fft.fftshift(np.real(np.fft.ifft2(np.abs(F) ** 2)))
    ov = np.fft.fftshift(np.real(np.fft.ifft2(np.abs(np.fft.fft2(np.ones_like(x), s=(2 * n, 2 * n))) ** 2)))
    ac = ac / np.maximum(ov, 1); ac /= ac[n, n]
    lab, _ = label(ac > 0.5); m = lab == lab[n, n]
    yy, xx = np.nonzero(m)
    ev = np.sort(np.linalg.eigvalsh(np.cov(np.vstack([xx - n, yy - n]))))[::-1]
    return np.sqrt(ev[0]) * C.PIX_LY, np.sqrt(ev[1]) * C.PIX_LY


def trackability(row, col, size=150):
    """Epoch 2-4 correlation of the structure maps in one window (zero shift and best shift)."""
    b = {k: np.asarray(P.band(k)[row:row + size, col:col + size], float) for k in TRACK_EPOCHS}
    return dict(ncc0=float(np.mean([ncc(b[i], b[j]) for i, j in TRACK_PAIRS])),
                ncct=float(np.mean([best_shift_ncc(b[i], b[j]) for i, j in TRACK_PAIRS])),
                l_maj=acf_shape(b[2])[0], l_min=acf_shape(b[2])[1])


R_BINS_A = np.logspace(np.log10(0.0032), np.log10(0.24), 30)
ANGLES_A = np.arange(0, 180, 20)


def anisotropy(row, col, cfg=C.Config()):
    """In-plane anisotropy A [dex]: log10 of max / min over 9 wedges of the PSF-deconvolved
    in-plane S2, averaged over 0.02 <= r <= 0.1 ly (normalised window)."""
    d = P.load_window(row, col, cfg)
    J, _ = G.pixel_to_uv_jacobian(row, col, cfg.size)
    s2d = psf.sf2d_epochs(np.asarray(d['flux_epochs'], float), J, t2min=cfg.t2min or psf.T2MIN,
                          correct=cfg.t2min is not None, extra_sigma=cfg.sigma)
    r, Pm = wedge_profiles(s2d, ANGLES_A, 10, r_bins=R_BINS_A)
    sel = (r >= 0.02) & (r <= 0.1)
    amp = np.nanmean(np.where(Pm[:, sel] > 0, Pm[:, sel], np.nan), axis=1)
    return float(np.log10(np.nanmax(amp) / np.nanmin(amp)))


# =========================================================================== drift of every window
VGRID = np.round(np.arange(-2.0, 2.001, 0.1), 2)
TRACK_MARGIN = 40


def window_drift(row, col, cfg=C.Config()):
    """(dU/dW, dV/dW) on a 0.1 grid in [-2, 2] maximising the mean correlation of the
    epoch pairs 0.015-0.1 ly apart in depth, each cut at -/+ v dW / 2 around the window."""
    M, size = TRACK_MARGIN, cfg.size
    big = P.load_window(row - M, col - M, cfg.replace(size=size + 2 * M))
    f = np.asarray(big['flux_epochs'], float); w = np.asarray(big['W_values'], float)
    pairs = [(a, b) for a in range(len(w)) for b in range(a + 1, len(w)) if 0.015 <= abs(w[b] - w[a]) <= 0.1]

    def cut(img, du, dv):
        return img[M + dv:M + dv + size, M + du:M + du + size]

    def nccm(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 100:
            return np.nan
        x = a[m] - a[m].mean(); y = b[m] - b[m].mean()
        return float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum()))
    best, bv = -np.inf, (0.0, 0.0)
    for vu in VGRID:
        for vv in VGRID:
            s = []
            for a, b in pairs:
                dW = w[b] - w[a]
                du, dv = int(round(vu * dW / 2 / C.PIX_LY)), int(round(vv * dW / 2 / C.PIX_LY))
                s.append(nccm(cut(f[a], -du, -dv), cut(f[b], du, dv)))
            s = np.nanmean(s)
            if s > best:
                best, bv = s, (vu, vv)
    return dict(row=row, col=col, vU=bv[0], vV=bv[1], cc_track=best)


def window_table(windows, cfg=C.Config(), procs=1):
    """Trackability, anisotropy, drift for every window."""
    def one(rc):
        r, c = rc
        return dict(row=r, col=c, **trackability(r, c), A=anisotropy(r, c, cfg),
                    **{k: v for k, v in window_drift(r, c, cfg).items() if k not in ('row', 'col')})
    return pd.DataFrame([one(rc) for rc in windows])
