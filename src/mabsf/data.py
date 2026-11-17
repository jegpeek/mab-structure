"""Where the data live, and how to get them.

Layout under the data directory (``$MABSF_DATA``, default ``<repo>/data``)::

    inputs/                         from the Zenodo deposit (see data/MANIFEST.md)
        mosaics/mosaic_<visit>_F444W_modB_pixscale60_infill_v13.fits   x 7
        pos3d/pos3d_60mas_2024.npy, pos3d_60mas_2025.npy
        psf/PSF_NIRCam_in_flight_opd_filter_F444W.fits
        psf/F444W_60583_b_60_i2d.fits
    products/                       made by this package (regenerable)
        uv_cube_<infill>.npy, U_grid.npy, V_grid.npy, epoch_w.npy, backgrounds.npy, ...

Small analysis results (per-window fits, PSF calibration) are kept in the
repository under ``results/`` so the notebooks run in minutes.
"""
from __future__ import annotations

import functools
import glob
import os

import numpy as np

from . import config as C

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#: Zenodo record holding the inputs (filled in at release).
ZENODO_DOI = None


def data_dir():
    return os.environ.get('MABSF_DATA', os.path.join(REPO, 'data'))


def inputs(*p):
    return os.path.join(data_dir(), 'inputs', *p)


def products(*p):
    d = os.path.join(data_dir(), 'products')
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, *p)


def results(*p):
    """Small cached results shipped with the repository."""
    d = os.environ.get('MABSF_RESULTS', os.path.join(REPO, 'results'))
    path = os.path.join(d, *p)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def figures(*p):
    d = os.path.join(REPO, 'figures')
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, *p)


# --------------------------------------------------------------------------- inputs
def mosaic_path(visit):
    hits = sorted(glob.glob(inputs('mosaics', 'mosaic_%s_*_infill_v*.fits' % visit)))
    if not hits:
        raise FileNotFoundError('mosaic for visit %s not found under %s; see data/MANIFEST.md'
                                % (visit, inputs('mosaics')))
    return hits[0]


def pos3d_path(year):
    return inputs('pos3d', 'pos3d_60mas_%s.npy' % year)


PSF_FITS = ('psf', 'PSF_NIRCam_in_flight_opd_filter_F444W.fits')
STARS_I2D = ('psf', 'F444W_60583_b_60_i2d.fits')


def check_inputs(verbose=True):
    """True if every input file is present."""
    need = [mosaic_path(v) if glob.glob(inputs('mosaics', 'mosaic_%s_*' % v)) else inputs('mosaics', v)
            for v, _ in C.EPOCHS]
    need += [pos3d_path('2024'), pos3d_path('2025'), inputs(*PSF_FITS), inputs(*STARS_I2D)]
    missing = [p for p in need if not os.path.exists(p)]
    if verbose:
        print('%d / %d inputs present under %s' % (len(need) - len(missing), len(need), os.path.relpath(inputs(), REPO)))
        for p in missing:
            print('  missing', p)
    return not missing


# --------------------------------------------------------------------------- products on the UV grid
def uv_cube_path(infill='draw1'):
    return products('uv_cube_%s.npy' % infill)


@functools.lru_cache(None)
def uv_cube(infill='draw1', mmap=True):
    """(7, ny, nx) float32 flux on the UV grid (made by mabsf.geometry.resample_epochs)."""
    p = uv_cube_path(infill)
    if not os.path.exists(p):
        raise FileNotFoundError('%s not found: run `mabsf resample` (notebook 00)' % p)
    return np.load(p, mmap_mode='r' if mmap else None)


@functools.lru_cache(None)
def uv_grids():
    """(U_grid, V_grid) [ly], each (ny, nx), memory-mapped."""
    return (np.load(products('U_grid.npy'), mmap_mode='r'), np.load(products('V_grid.npy'), mmap_mode='r'))


@functools.lru_cache(None)
def epoch_w():
    """Mean W [ly] of each epoch (the depth of the scattering surface)."""
    return np.load(products('epoch_w.npy'))


@functools.lru_cache(None)
def backgrounds():
    """Per-epoch sky level of the UV cube (median flux)."""
    return np.load(products('backgrounds.npy'))
