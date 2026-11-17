"""From the UV cube to analysis windows: masking, normalisation, cloud region, tiling.

* **Windows** are square cut-outs of the UV cube.  The per-epoch sky level is
  subtracted; pixels within 50 px of missing data are dropped (resampling
  near footprint edges repeats edge values); an epoch covering less than 25%
  of the window is dropped.  (`read_window`)
* **Normalisation.**  The echo brightens with depth and the epochs differ in
  overall brightness, so each epoch's flux in a window is scaled to the
  median rms of the epochs (`normalise`, Config.norm = 'window').  Only the
  morphology then enters S2.
* **Cloud region.**  Structure, not brightness, defines the cloud: band-pass
  each epoch to the 0.01-0.1 ly scales the shape fit uses, measure its local
  rms against the empty-sky rms, take the weakest epoch at each pixel, smooth
  by 0.08 ly and keep pixels at >= 2x noise; fill holes and keep the largest
  connected piece (`cloud_region`).
* **Sample.**  Non-overlapping 0.48 ly (150 px) tiles on a grid anchored at
  the Region 1 box, kept if >= 90% inside the region and covered by all 7
  epochs (`sample_windows`).
"""
from __future__ import annotations

import functools
import os

import numpy as np
from matplotlib.path import Path
from scipy.ndimage import binary_fill_holes, distance_transform_edt, gaussian_filter, label, uniform_filter

from . import config as C
from . import data as D

BAND_LO, BAND_HI = 2, 30      # band-pass Gaussian sigmas [px]: keeps ~0.01-0.1 ly
BLOCK = 25                    # block / box size [px] (0.08 ly) for local rms
EDGE_RADIUS = 50


# --------------------------------------------------------------------------- windows
@functools.lru_cache(None)
def edge_mask(radius=EDGE_RADIUS):
    """(7, ny, nx) bool: pixel valid and > radius px from any missing pixel (from the draw-1 cube)."""
    p = D.products('edge_mask_r%d.npy' % radius)
    if not os.path.exists(p):
        cube = D.uv_cube('draw1')
        m = np.empty(cube.shape, bool)
        for e in range(cube.shape[0]):
            f = np.asarray(cube[e]); bad = ~np.isfinite(f) | (f == 0)
            m[e] = distance_transform_edt(~bad) > radius if bad.any() else True
        np.save(p, m)
    return np.load(p, mmap_mode='r')


def read_window(row, col, size, epochs=None, infill='draw1', edge_mask_radius=EDGE_RADIUS, min_coverage=0.25):
    """One window: dict(flux_epochs (n_ep, size, size), U_grid, V_grid, W_values, row, col, size)."""
    ep = list(range(C.N_EPOCHS)) if epochs is None else list(epochs)
    rs, cs = slice(row, row + size), slice(col, col + size)
    if infill == 'mask':
        f = np.asarray(D.uv_cube('draw1')[:, rs, cs], float)[ep]
        flag = np.asarray(D.uv_cube('infillflag')[:, rs, cs])[ep]
        f[flag > 0] = np.nan
    else:
        f = np.asarray(D.uv_cube(infill)[:, rs, cs], float)[ep]
    f -= D.backgrounds()[ep][:, None, None]
    if edge_mask_radius > 0:
        f[~np.asarray(edge_mask(edge_mask_radius)[:, rs, cs])[ep]] = np.nan
    n = f.shape[1] * f.shape[2]
    for e in range(f.shape[0]):
        if np.isfinite(f[e]).sum() / n < min_coverage:
            f[e] = np.nan
    U, V = D.uv_grids()
    return dict(flux_epochs=f, U_grid=np.asarray(U[rs, cs]), V_grid=np.asarray(V[rs, cs]),
                W_values=D.epoch_w()[ep], row=row, col=col, size=size)


@functools.lru_cache(None)
def cloud_rms_factors():
    """Per-epoch rms of the sky-subtracted flux over the cloud region (Config.norm = 'global')."""
    cube, bg, reg = D.uv_cube('draw1'), D.backgrounds(), cloud_region()
    rr, cc = np.where(reg)
    v = np.array([np.asarray(cube[k])[rr, cc] - bg[k] for k in range(C.N_EPOCHS)])
    return v[:, np.isfinite(v).all(0)].std(1)


def normalise(d, norm='window', epochs=None):
    """Scale each epoch's flux to a common rms (in place on a copy); adds d['epoch_rms']."""
    f = np.array(d['flux_epochs'], float)
    with np.errstate(invalid='ignore'):
        sd = np.nanstd(f, axis=(1, 2))
    if norm == 'window':
        ok = np.isfinite(sd) & (sd > 0)
        f[ok] *= (float(np.median(sd[ok])) / sd[ok])[:, None, None]
    elif norm == 'global':
        g = cloud_rms_factors()[list(range(C.N_EPOCHS)) if epochs is None else list(epochs)]
        f *= (float(np.median(g)) / g)[:, None, None]
    elif norm != 'none':
        raise ValueError(norm)
    out = dict(d); out['flux_epochs'] = f; out['epoch_rms'] = sd
    return out


def load_window(row, col, cfg=C.Config()):
    """read_window + normalise under the choices of cfg."""
    d = read_window(row, col, cfg.size, epochs=cfg.epochs, infill=cfg.infill, **C.READ_KW)
    return normalise(d, cfg.norm, cfg.epochs)


# --------------------------------------------------------------------------- cloud region
def band_path(k):
    return D.products('band_%d_%d_e%d.npy' % (BAND_LO, BAND_HI, k))


def band_maps(force=False):
    """Band-pass maps G(2 px) - G(30 px) of each epoch, normalised for missing data."""
    cube = D.uv_cube('draw1')
    for k in range(C.N_EPOCHS):
        if os.path.exists(band_path(k)) and not force:
            continue
        f = np.asarray(cube[k], float); m = np.isfinite(f); f0 = np.where(m, f, 0.0)
        g = lambda s: gaussian_filter(f0, s) / np.maximum(gaussian_filter(m.astype(float), s), 1e-6)  # noqa: E731
        np.save(band_path(k), np.where(m, g(BAND_LO) - g(BAND_HI), np.nan).astype(np.float32))


@functools.lru_cache(None)
def band(k):
    if not os.path.exists(band_path(k)):
        band_maps()
    return np.load(band_path(k), mmap_mode='r')


def block_rms(b):
    ny, nx = (b.shape[0] // BLOCK) * BLOCK, (b.shape[1] // BLOCK) * BLOCK
    x = b[:ny, :nx].reshape(ny // BLOCK, BLOCK, nx // BLOCK, BLOCK)
    fin = np.isfinite(x).mean(axis=(1, 3))
    with np.errstate(invalid='ignore'):
        r = np.sqrt(np.nanmean(x ** 2, axis=(1, 3)))
    r[fin < 0.9] = np.nan
    return r


@functools.lru_cache(None)
def polygon_mask():
    """Pixels inside the hand-drawn cloud polygon (used only to find empty sky)."""
    U, V = D.uv_grids()
    u_ax, v_ax = np.asarray(U[0, :], float), np.asarray(V[:, 0], float)
    poly = C.cloud_polygon_uv()
    c0 = max(int(np.searchsorted(u_ax, poly[:, 0].min())) - 2, 0); c1 = int(np.searchsorted(u_ax, poly[:, 0].max())) + 2
    r0 = max(int(np.searchsorted(v_ax, poly[:, 1].min())) - 2, 0); r1 = int(np.searchsorted(v_ax, poly[:, 1].max())) + 2
    UU, VV = np.meshgrid(u_ax[c0:c1], v_ax[r0:r1])
    m = np.zeros(U.shape, bool)
    m[r0:r1, c0:c1] = Path(poly).contains_points(np.column_stack([UU.ravel(), VV.ravel()])).reshape(UU.shape)
    return m


@functools.lru_cache(None)
def noise_levels():
    """Per-epoch band-pass rms of empty sky: 20th percentile of 25 px block rms outside the polygon."""
    poly = polygon_mask()
    ny, nx = (poly.shape[0] // BLOCK) * BLOCK, (poly.shape[1] // BLOCK) * BLOCK
    pin = poly[:ny, :nx].reshape(ny // BLOCK, BLOCK, nx // BLOCK, BLOCK).mean(axis=(1, 3))
    out = []
    for k in range(C.N_EPOCHS):
        r = block_rms(np.asarray(band(k)))
        out.append(float(np.nanpercentile(r[(pin == 0) & np.isfinite(r)], 20)))
    return np.array(out)


@functools.lru_cache(None)
def weakest_epoch_sn():
    """Per pixel: local (25 px box) band rms / sky noise, minimum over epochs; 0 if any epoch missing."""
    p = D.products('weakest_epoch_sn.npy')
    if os.path.exists(p):
        return np.load(p, mmap_mode='r')
    nz = noise_levels(); out = None
    for k in range(C.N_EPOCHS):
        b = np.asarray(band(k), float); m = np.isfinite(b)
        w = uniform_filter(m.astype(float), BLOCK)
        r = np.sqrt(np.maximum(uniform_filter(np.where(m, b, 0) ** 2, BLOCK) / np.maximum(w, 1e-6), 0)) / nz[k]
        r[w < 0.9] = 0.0
        out = r if out is None else np.minimum(out, r)
    np.save(p, out.astype(np.float32))
    return np.load(p, mmap_mode='r')


@functools.lru_cache(None)
def cloud_region(on_k=2.0, smooth=25):
    """The cloud as one contiguous mask (see module docstring)."""
    on = binary_fill_holes(gaussian_filter(np.asarray(weakest_epoch_sn(), float), smooth) >= on_k)
    lab, _ = label(on)
    n = np.bincount(lab.ravel()); n[0] = 0
    return lab == np.argmax(n)


def epochs_covering(row, col, size):
    """Number of epochs whose band map covers >= 75% of the window's 25 px blocks."""
    n = 0
    for k in range(C.N_EPOCHS):
        r = block_rms(np.asarray(band(k)[row:row + size, col:col + size]))
        n += np.isfinite(r).mean() >= 0.75
    return int(n)


def tile_origin(size, shift=(0, 0)):
    """First row/col of the tiling grid: Region 1's box is a tile (for size 150)."""
    r1 = C.REGION_CENTERS[1]
    a = (r1[0] - C.REGION_BOX // 2 + shift[0], r1[1] - C.REGION_BOX // 2 + shift[1])
    return a[0] % size, a[1] % size


def sample_windows(cfg=C.Config()):
    """[(row, col)] of the analysis windows (top-left corners on the UV grid)."""
    if cfg.centres == 'same':
        h = (150 - cfg.size) // 2
        return [(r + h, c + h) for r, c in sample_windows(cfg.replace(size=150, centres='tiles'))
                if epochs_covering(r + h, c + h, cfg.size) == C.N_EPOCHS]
    reg = cloud_region(cfg.on_k, cfg.smooth)
    s = cfg.size
    r0, c0 = tile_origin(s, cfg.shift)
    out = []
    for r in range(r0, reg.shape[0] - s + 1, s):
        for c in range(c0, reg.shape[1] - s + 1, s):
            if reg[r:r + s, c:c + s].mean() >= cfg.frac and epochs_covering(r, c, s) == C.N_EPOCHS:
                out.append((r, c))
    return out
