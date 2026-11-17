"""Per-window ellipsoid fits with block-jackknife errors, cached as JSON.

For each window two fits are made to the structure function at 3D lags
r < rcut (0.1 ly):

* **3D**: all epoch pairs -> axis ratios a2/a1, a3/a2, slope alpha and the
  orientation of the long axis (inclination from W);
* **2D**: the same-epoch (dW = 0) planes only -> the in-plane axis ratio
  b2/b1.  It is started both from a default guess and from the 3D fit, and the
  better of the two is kept.

Errors come from a K x K delete-one-block jackknife (K = 4): the window is
cut into 16 blocks, each removed in turn from every epoch, and the fits
repeated.  SE = sqrt((n - 1)/n * sum (x_k - mean)^2).

    fit_windows(windows, cfg, procs=6)    # cached under results/window_fits/<cfg.fit_tag()>/
    table(windows, cfg)                   # one row per window, central values + SE
"""
from __future__ import annotations

import json
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from . import preprocess as P
from . import sf

TRACK = ('a2a1', 'a3a2', 'a3a1', 'alpha', 'incl', 'theta', 'phi', 'psi', 'b2b1', 'pa2d', 'du_per_dw', 'dv_per_dw')


def delete_block(data, i, j, k):
    """Copy of the window with block (i, j) of a k x k grid set to NaN in every epoch."""
    f = np.array(data['flux_epochs'], copy=True)
    _, ny, nx = f.shape
    re, ce = np.linspace(0, ny, k + 1).astype(int), np.linspace(0, nx, k + 1).astype(int)
    f[:, re[i]:re[i + 1], ce[j]:ce[j + 1]] = np.nan
    d = dict(data); d['flux_epochs'] = f
    return d


def _one(s2b, cfg, freeze, guess=None):
    """One fit, flattened to scalars; never raises."""
    try:
        f = sf.fit_s2(s2b, guess=guess, inner_uv=cfg.inner_uv, stride=cfg.stride, weighting=cfg.weighting,
                      freeze=freeze, **C.FIT_KW)
        rec = sf.scalars(f['params'])
        rec.update(fit_success=bool(f['fit'].success), chi2=f['chi2'], n_fit=f['n'],
                   rms_resid=float(np.sqrt(f['chi2'] / max(f['n'], 1))))
        return rec
    except Exception as exc:   # noqa: BLE001
        return {'error': repr(exc), 'fit_success': False}


def fit_realisation(data, cfg, r_grid=None):
    """3D and 2D fits of one realisation (the full window or a jackknife sample)."""
    s2 = sf.compute_s2(data, **C.S2_KW)
    if r_grid is None:
        r_grid = sf.lag_radius(s2)
    b3 = sf.band(s2, r_grid, cfg.rmin, cfg.rcut)
    b2 = sf.band(sf.in_plane(s2), r_grid, cfg.rmin, cfg.rcut)
    out = {'3d': _one(b3, cfg, ('A',))}
    auto = _one(b2, cfg, C.FREEZE_2D + ('A',))
    c3 = out['3d']
    if 's11' in c3:
        g = {k: c3[k] for k in ('s11', 's22', 's33', 'l12', 'l13', 'l23', 'alpha', 'A')}
        seed = _one(b2, cfg, ('s33', 'l13', 'l23', 'A'), guess=g)
        ca = auto.get('chi2', np.inf); cs = seed.get('chi2', np.inf)
        ca = ca if np.isfinite(ca) else np.inf; cs = cs if np.isfinite(cs) else np.inf
        out['2d'] = dict(seed, seeded_from_3d=True) if cs < ca else dict(auto, seeded_from_3d=False)
    else:
        out['2d'] = auto
    return out, r_grid


def fit_dir(cfg):
    return D.results('window_fits', cfg.fit_tag(), '')


def fit_path(row, col, cfg):
    return os.path.join(fit_dir(cfg), 'r%d_c%d_s%d.json' % (row, col, cfg.size))


def fit_window(row, col, cfg=C.Config(), force=False):
    """Fit one window (central + K^2 jackknife samples) and cache the result."""
    path = fit_path(row, col, cfg)
    if os.path.exists(path) and not force:
        return json.load(open(path))
    t0 = time.time()
    data = P.load_window(row, col, cfg)
    central, r_grid = fit_realisation(data, cfg)
    samples = []
    for i in range(cfg.K):
        for j in range(cfg.K):
            s, _ = fit_realisation(delete_block(data, i, j, cfg.K), cfg, r_grid)
            s['block'] = [i, j]
            samples.append(s)
    rec = dict(row=row, col=col, size=cfg.size, K=cfg.K, rcut=cfg.rcut, config=cfg.fit_tag(),
               epoch_rms=[float(x) for x in data['epoch_rms']], central=central, samples=samples,
               seconds=time.time() - t0)
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(rec, fh, default=float)
    os.replace(tmp, path)
    return rec


def _job(spec):
    r, c, cfg = spec
    fit_window(r, c, cfg)
    return r, c


def fit_windows(windows, cfg=C.Config(), procs=6, verbose=True):
    """Fit every window not yet cached (in parallel)."""
    todo = [(r, c, cfg) for r, c in windows if not os.path.exists(fit_path(r, c, cfg))]
    if verbose:
        print('%d windows, %d to fit (cache %s)' % (len(windows), len(todo), fit_dir(cfg)), flush=True)
    if not todo:
        return
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ.setdefault(v, '1')        # one thread per worker process
    t0 = time.time()
    with Pool(procs) as p:
        for k, _ in enumerate(p.imap_unordered(_job, todo), 1):
            if verbose and (k % 10 == 0 or k == len(todo)):
                print('  %d/%d  %.0f s' % (k, len(todo), time.time() - t0), flush=True)


def jk_se(values):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    if v.size < 2:
        return np.nan
    return float(np.sqrt((v.size - 1) / v.size * np.sum((v - v.mean()) ** 2)))


def table(windows, cfg=C.Config(), mode='3d'):
    """One row per window: central values and jackknife SEs of the `mode` ('3d'/'2d') fit."""
    rows = []
    for r, c in windows:
        rec = json.load(open(fit_path(r, c, cfg)))
        cen = rec['central'][mode]
        row = dict(row=r, col=c, fit_success=cen.get('fit_success', False), chi2=cen.get('chi2', np.nan))
        for q in TRACK:
            row[q] = cen.get(q, np.nan)
            row['se_' + q] = jk_se([s[mode].get(q) for s in rec['samples']])
        if mode == '3d':
            # jackknife SEs of the variance ratios f12 = (a2/a1)^alpha, f23 = (a3/a2)^alpha
            # (mabsf.shape) and of log10 a2/a1, over samples with a finite fit
            sm = [s['3d'] for s in rec['samples']]
            sm = [s for s in sm if all(np.isfinite(s.get(k, np.nan)) for k in ('a2a1', 'a3a2', 'alpha'))
                  and s['alpha'] > 0]
            row['se_f12'] = jk_se([s['a2a1'] ** s['alpha'] for s in sm])
            row['se_f23'] = jk_se([s['a3a2'] ** s['alpha'] for s in sm])
            row['se_l12'] = jk_se([np.log10(s['a2a1']) for s in sm])
            row['se_l23'] = jk_se([np.log10(s['a3a2']) for s in sm])
        rows.append(row)
    return pd.DataFrame(rows)
