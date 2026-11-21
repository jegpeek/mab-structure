"""From sky images to the light-echo frame (U, V, W).

Each mosaic pixel images dust at a definite 3D position: the light echo lies
on a paraboloid with Cas A at its focus, set by the light-travel time.  The
input cubes ``pos3d_60mas_<year>.npy`` give that position (x, y, z) [ly] in a
Cas A-centred frame for every pixel, at two reference dates.  Between them the
paraboloid moves outward, so positions at an epoch are interpolated linearly
in time.

The echo frame has

* **W**, the normal to the paraboloid at Mab (the depth direction:
  successive epochs sample successively deeper layers, W increasing);
* **U, V**, orthogonal axes spanning the echo surface.

Projecting each pixel's (x, y, z) on U and V and interpolating the flux onto a
regular (U, V) grid with step PIX_LY puts all epochs on a common grid.  Epoch k
is then a slice at depth W_k: the mean W of its pixels, which changes by only
~0.6 ly over the 7 epochs.
"""
from __future__ import annotations

import numpy as np
from astropy.io import fits
from scipy.interpolate import griddata

from . import config as C
from . import data as D


def epoch_dt(mjd):
    """Fractional year from MJD_BASE used to interpolate the two position cubes."""
    return np.float32((mjd - C.MJD_BASE) / 365.0)


def load_positions():
    """(x, y, z) of the 2024 and 2025 reference surfaces, each a tuple of (ny, nx) arrays."""
    p24 = np.load(D.pos3d_path('2024'), mmap_mode='r')
    p25 = np.load(D.pos3d_path('2025'), mmap_mode='r')
    a = tuple(np.asarray(p24[..., k], np.float32) for k in range(3))
    b = tuple(np.asarray(p25[..., k], np.float32) for k in range(3))
    return a, b


def xyz_at(pos, dt):
    """Linear interpolation of the surface positions to fractional year dt."""
    (x24, y24, z24), (x25, y25, z25) = pos
    return (x24 * (1 - dt) + x25 * dt, y24 * (1 - dt) + y25 * dt, z24 * (1 - dt) + z25 * dt)


def project(X, Y, Z):
    """(U, V, W) [ly] of points (X, Y, Z)."""
    U = X * C.U_VEC[0] + Y * C.U_VEC[1] + Z * C.U_VEC[2]
    V = X * C.V_VEC[0] + Y * C.V_VEC[1] + Z * C.V_VEC[2]
    W = X * C.W_VEC[0] + Y * C.W_VEC[1] + Z * C.W_VEC[2]
    return U, V, W


def uv_grid(pos):
    """Master UV grid covering the first epoch's footprint, step PIX_LY."""
    X0, Y0, Z0 = xyz_at(pos, epoch_dt(C.EPOCHS[0][1]))
    U0, V0, _ = project(X0, Y0, Z0)
    u = np.arange(np.nanmin(U0), np.nanmax(U0), C.PIX_LY, dtype=np.float32)
    v = np.arange(np.nanmin(V0), np.nanmax(V0), C.PIX_LY, dtype=np.float32)
    return np.meshgrid(u, v)


def read_mosaic(visit, hdu):
    with fits.open(D.mosaic_path(visit)) as h:
        return np.asarray(h[hdu].data, np.float32)


def resample_epochs(infill='draw1', write=True, verbose=True):
    """Resample the 7 epoch mosaics onto the UV grid.

    Linear interpolation (Delaunay triangulation of the pixel positions in
    (U, V)), one epoch at a time.  Writes, under data/products/:

    ``uv_cube_<infill>.npy`` (7, ny, nx) float32
        flux on the grid, NaN outside an epoch's footprint (infill='infillflag':
        the interpolated fraction of infilled pixels, used by Config(infill='mask'));
    ``U_grid.npy``, ``V_grid.npy``, ``epoch_w.npy``, ``backgrounds.npy``
        (written only for infill='draw1'): the grid, the mean W of each epoch
        and its sky level (median of the resampled flux).

    About 3 min per epoch and ~6 GB of memory.
    """
    pos = load_positions()
    U_grid, V_grid = uv_grid(pos)
    ny, nx = U_grid.shape
    cube = np.full((C.N_EPOCHS, ny, nx), np.nan, np.float32)
    w = np.zeros(C.N_EPOCHS)
    bg = np.zeros(C.N_EPOCHS)
    for i, (visit, mjd) in enumerate(C.EPOCHS):
        X, Y, Z = xyz_at(pos, epoch_dt(mjd))
        U, V, W = project(X, Y, Z)
        if infill == 'infillflag':      # 1 where the infill realisations differ (an infilled pixel)
            d1 = read_mosaic(visit, C.INFILL_HDUS['draw1'])
            img = np.where(np.isfinite(d1), (d1 != read_mosaic(visit, C.INFILL_HDUS['draw2'])).astype(np.float32), np.nan)
        else:
            img = read_mosaic(visit, C.INFILL_HDUS[infill])
        ok = np.isfinite(U) & np.isfinite(V) & np.isfinite(img)
        w[i] = float(np.nanmean(W[ok]))
        cube[i] = griddata(np.column_stack((U[ok], V[ok])), img[ok], (U_grid, V_grid), method='linear')
        bg[i] = float(np.nanmedian(cube[i]))
        if verbose:
            print('epoch %d (%s): mean W %.4f ly, sky %.4f, %d px' % (i, visit, w[i], bg[i], ok.sum()), flush=True)
    if write:
        np.save(D.uv_cube_path(infill), cube)
        if infill == 'draw1':
            np.save(D.products('U_grid.npy'), U_grid)
            np.save(D.products('V_grid.npy'), V_grid)
            np.save(D.products('epoch_w.npy'), w)
            np.save(D.products('backgrounds.npy'), bg)
    return cube, U_grid, V_grid, w, bg


# --------------------------------------------------------------------------- local geometry
def pixel_to_uv_jacobian(row, col, size, stride=2):
    """Median d(U, V)/d(mosaic col, row) [ly per 60 mas pixel] over a UV window.

    Needed to carry the sky-plane PSF into the UV plane: the sky-to-UV mapping
    is locally linear but sheared and rotated, and differs slightly across the
    field.  Uses the 2024 surface sampled every `stride` mosaic pixels.
    Returns J with rows (U, V) and columns (mosaic col, mosaic row).
    """
    U_grid, V_grid = D.uv_grids()
    Uw = U_grid[row:row + size, col:col + size]; Vw = V_grid[row:row + size, col:col + size]
    u0, u1, v0, v1 = np.nanmin(Uw), np.nanmax(Uw), np.nanmin(Vw), np.nanmax(Vw)
    U, V = _uv_coarse(stride)
    g = [np.gradient(A, axis=ax) / stride for A in (U, V) for ax in (1, 0)]
    sel = (U >= u0) & (U <= u1) & (V >= v0) & (V <= v1) & np.isfinite(sum(g))
    J = np.median(np.array([[g[0][sel], g[1][sel]], [g[2][sel], g[3][sel]]]), axis=-1)
    return J, int(sel.sum())


_POS = {}


def _uv_coarse(stride):
    """U, V of the 2024 surface every `stride` mosaic pixels (cached)."""
    if stride not in _POS:
        P = np.asarray(np.load(D.pos3d_path('2024'), mmap_mode='r')[::stride, ::stride], float)
        X, Y, Z = (P[..., i] for i in range(3))
        _POS[stride] = (X * C.U_VEC[0] + Y * C.U_VEC[1] + Z * C.U_VEC[2],
                        X * C.V_VEC[0] + Y * C.V_VEC[1] + Z * C.V_VEC[2])
    return _POS[stride]
