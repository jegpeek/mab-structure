"""Sensitivity of every claim to every analysis choice, one choice at a time.

Each entry of CHOICES is (group, label, overrides of Config).  For each one the
sample is rebuilt, the windows are fitted (cached under results/window_fits/
<fit tag>), the matching PSF calibration is applied, and all claims are
re-evaluated (`claims`).  Groups:

  D1   epochs used               D14  window size (same centres)
  D2   flux normalisation        D15  phase of the tiling grid
  D3   star/galaxy infill        D16  cloud-region threshold
  D5   extra PSF Gaussian        D17  cloud-region smoothing
  D6   deconvolution gain cap    D18  fraction of a tile inside the region
  D8   calibration geometry      D19  fit range
  D22  jackknife blocks          D20  lag weighting
  D23  collapse handling         D24  S2 estimator (stationary = first submission)

What each claim depends on: C1-C4 (Regions 1 and 2, notebook 04) only on the
deconvolution settings (D5, D6); C5 on nothing varied here; C6 on the sample,
the fits and the deconvolution; C7-C13 on everything else.
"""
from __future__ import annotations

import functools
import os

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C, data as D, fitting as F, preprocess as P, psf, regions as RG, shape as S

CHOICES = [('base', 'paper analysis', {})]
_C = CHOICES.append
for e, lab in (((0, 1, 2, 3, 4), 'epochs 1-5 only'), ((1, 2, 3, 4, 5, 6), 'drop epoch 1'),
               ((0, 1, 2, 3, 5, 6), 'drop epoch 5'), ((0, 1, 2, 3, 4, 5), 'drop epoch 7')):
    _C(('D1', lab, dict(epochs=e)))
_C(('D2', 'no normalisation (raw flux)', dict(norm='none')))
_C(('D2', 'one cloud-wide factor per epoch', dict(norm='global')))
for k, lab in (('draw2', 'infill draw 2'), ('draw3', 'infill draw 3'), ('mean', 'infill mean'),
               ('mask', 'infilled pixels masked')):
    _C(('D3', lab, dict(infill=k)))
_C(('D5', 'extra PSF Gaussian 22.5 mas', dict(sigma=0.0225, calib_variant='sig225')))
_C(('D5', 'extra PSF Gaussian 42.5 mas', dict(sigma=0.0425, calib_variant='sig425')))
_C(('D6', 'gain cap T2 >= 0.02', dict(t2min=0.02)))
_C(('D6', 'gain cap T2 >= 0.2', dict(t2min=0.2)))
_C(('D6', 'no deconvolution', dict(t2min=None)))
_C(('D8', 'calibration geometry 150 px', dict(calib_variant='geom150')))
_C(('D14', 'windows 0.32 ly (same centres)', dict(size=100, centres='same')))
_C(('D14', 'windows 0.64 ly (same centres)', dict(size=200, centres='same')))
for s, lab in (((75, 0), 'grid shifted 1/2 tile in V'), ((0, 75), 'grid shifted 1/2 tile in U'),
               ((75, 75), 'grid shifted 1/2 tile in U and V')):
    _C(('D15', lab, dict(shift=s)))
_C(('D16', 'region threshold 1.5', dict(on_k=1.5)))
_C(('D16', 'region threshold 3', dict(on_k=3.0)))
_C(('D17', 'region smoothing 0.04 ly', dict(smooth=12)))
_C(('D17', 'region smoothing 0.16 ly', dict(smooth=50)))
_C(('D18', 'tile >= 75% inside region', dict(frac=0.75)))
_C(('D18', 'tile 100% inside region', dict(frac=1.0)))
for rc, cal in ((0.05, 'rc050'), (0.075, 'rc075'), (0.15, 'rc150')):
    _C(('D19', 'fit range r < %g ly' % rc, dict(rcut=rc, calib_variant=cal)))
_C(('D20', 'uniform lag weighting', dict(weighting=None, calib_variant='wnone')))
_C(('D20', '1/r^2 lag weighting', dict(weighting='1/r2', calib_variant='w1r2')))
_C(('D22', 'jackknife 3x3', dict(K=3)))
_C(('D22', 'jackknife 5x5', dict(K=5)))
_C(('D23', 'collapsed excluded (log stats)', dict(collapsed='exclude')))
_C(('D23', 'collapse threshold 0.01', dict(degen=0.01)))
_C(('D23', 'collapse threshold 0.05', dict(degen=0.05)))
_C(('D24', 'stationary S2 estimator', dict(stationary=True, calib_variant='stationary')))

#: The claims as tested, with the rule that decides whether each still holds.
#: (column, short name, description, test on a row)
RULES = [
    ('C1_aniso_min', 'C1', 'R1 in-plane anisotropy (min over 0.02-0.03 ly) > 3x', lambda r: r > 3),
    ('C2_aniso_max', 'C2', 'R2 in-plane anisotropy (max) < 1.5x', lambda r: r < 1.5),
    ('C3_w_over_max', 'C3', 'R1: W variation within 0.5-1.5x of the largest in-plane', lambda r: 0.5 < r < 1.5),
    ('C4_w_over_mean', 'C4', 'R2 (tracked): W variation below the in-plane mean', lambda r: r < 1),
    ('C6_rho_incl', 'C6', 'trackability follows inclination (rho < 0, p < 0.01)', None),
    ('C6_rho_A', 'C6b', 'trackability does not follow sky-plane anisotropy (|rho| < 0.3)', lambda r: abs(r) < 0.3),
    ('f12', 'C7', 'common f12 (elongation)', None),
    ('f23', 'C7', 'common f23 (flattening)', None),
    ('common_a2a1', 'C7', 'common a2/a1', None),
    ('common_a3a2', 'C7', 'common a3/a2', None),
    ('rho_incl_b2b1', 'C7c', 'sky-plane axis ratio follows inclination (rho < -0.3)', lambda r: r < -0.3),
    ('median_incl', 'C8', 'median inclination of a1 from W', None),
    ('frac_excl_iso', 'C10', 'isotropy excluded in >= 90% of windows', lambda r: r >= 0.9),
    ('prolate_sigma', 'C11', 'prolate (f12 < f23) at > 3 sigma', lambda r: r > 3),
    ('sig_f12_95', 'C12', '95% limit on scatter in f12 < 0.2', lambda r: r < 0.2),
    ('sig_f23_95', 'C12', '95% limit on scatter in f23 < 0.15', lambda r: r < 0.15),
    ('n_collapsed', 'C13', 'windows with an unbounded long axis', None),
]


def cfg_of(over):
    return C.Config().replace(**over)


def sample(cfg):
    return P.sample_windows(cfg)


def fit_choices(procs=6, groups=None, verbose=True):
    """Fit every window of every choice that changes a fit (cached)."""
    for g, lab, over in CHOICES:
        if groups and g not in groups:
            continue
        cfg = cfg_of(over)
        wins = sample(cfg)
        todo = [w for w in wins if not F.is_cached(*w, cfg)]
        if todo:
            if verbose:
                print('%-4s %-36s %d windows to fit (%s)' % (g, lab, len(todo), cfg.fit_tag()), flush=True)
            F.fit_windows(wins, cfg, procs=procs, verbose=False)


# --------------------------------------------------------------------------- C1-C6
@functools.lru_cache(None)
def _regions(t2min, sigma):
    vu, vv, _ = RG.fit_velocity(2)
    kw = dict(t2min=t2min or psf.T2MIN, correct=t2min is not None, extra_sigma=sigma)
    res = {1: RG.region_sf(1, **kw), 2: RG.region_sf(2, (vu, vv), **kw)}
    return {i: RG.region_claims(res[i]) for i in (1, 2)}


def _track_path():
    return D.results('sensitivity_trackability.csv')


def trackability_table(windows, cfg):
    """Trackability (band maps only) and anisotropy A (deconvolution settings) per window, cached."""
    key = dict(size=cfg.size, sigma=cfg.sigma, t2min=-1.0 if cfg.t2min is None else cfg.t2min,
               norm=cfg.norm, infill=cfg.infill, epochs=''.join(map(str, cfg.epochs)))
    p = _track_path()
    T = pd.read_csv(p, dtype={'epochs': str}) if os.path.exists(p) else pd.DataFrame(columns=['row', 'col', *key])
    sel = np.ones(len(T), bool)
    for k, v in key.items():
        sel &= (T[k].astype(str) == str(v)).values
    have = {(int(r), int(c)) for r, c in zip(T.row[sel], T.col[sel])}
    new = []
    for r, c in windows:
        if (r, c) in have:
            continue
        tr = RG.trackability(r, c, cfg.size)
        new.append(dict(row=r, col=c, **key, ncct=tr['ncct'], A=RG.anisotropy(r, c, cfg)))
    if new:
        T = pd.concat([T, pd.DataFrame(new)], ignore_index=True)
        T.to_csv(p, index=False)
        sel = np.r_[sel, np.ones(len(new), bool)]
    T = T[sel]
    return T[[(int(r), int(c)) in set(windows) for r, c in zip(T.row, T.col)]].astype({'row': int, 'col': int})


# --------------------------------------------------------------------------- all claims
def claims(cfg, n_boot=300):
    """Every claim's number under cfg (one dict)."""
    out, k = S.summary(cfg, n_boot=n_boot)
    vr = out['variance_ratio']
    rc = _regions(cfg.t2min, cfg.sigma)
    TW = trackability_table(list(zip(k.row, k.col)), cfg).merge(k[['row', 'col', 'incl']], on=['row', 'col'])
    ri = stats.spearmanr(TW.ncct, TW.incl); ra = stats.spearmanr(TW.ncct, TW.A)
    ul = out['upper_limits_a2a1']
    return dict(
        n=len(k), n_collapsed=int(k.collapsed.sum()),
        C1_aniso_min=rc[1]['aniso_min'], C2_aniso_max=rc[2]['aniso_max'],
        C3_w_over_max=rc[1]['w_over_max_median'], C4_w_over_mean=rc[2]['w_over_mean_median'],
        C6_rho_incl=float(ri[0]), C6_p_incl=float(ri[1]), C6_rho_A=float(ra[0]), C6_p_A=float(ra[1]),
        f12=vr['f12'], sig_f12=vr['sig_f12'], sig_f12_95=vr['sig_f12_95'],
        f23=vr['f23'], sig_f23=vr['sig_f23'], sig_f23_95=vr['sig_f23_95'],
        median_alpha=out['median_alpha'], common_a2a1=out['common_a2a1'], common_a3a2=out['common_a3a2'],
        log_common_a2a1=out['C7_common_a2a1'], log_common_a3a2=out['C7_common_a3a2'],
        rho_incl_b2b1=out['C7_rho_incl_b2b1'], median_incl=out['C8_median_incl'],
        frac_excl_iso=out['n_exclude_isotropy_2sig'] / len(k), n_prolate=out['n_prolate_f'],
        prolate_sigma=out['C11_prolate_sigma'],
        R1_a2a1_limit=ul.get('%d_%d' % S.R1_TILE, np.nan))


def run(n_boot=300, verbose=True):
    rows = []
    for g, lab, over in CHOICES:
        cfg = cfg_of(over)
        r = dict(group=g, choice=lab, **claims(cfg, n_boot))
        rows.append(r)
        if verbose:
            print('%-4s %-36s n=%2d f12 %.3f f23 %.3f a2/a1 %.3f a3/a2 %.3f prolate %4.1fs rho6 %+.2f' %
                  (g, lab, r['n'], r['f12'], r['f23'], r['common_a2a1'], r['common_a3a2'], r['prolate_sigma'],
                   r['C6_rho_incl']), flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(D.results('sensitivity_claims.csv'), index=False)
    return T


def grade(T):
    """True / False per choice and testable rule; NaN where the rule has no pass/fail."""
    G = pd.DataFrame(index=T.index)
    for col, short, desc, rule in RULES:
        if col == 'C6_rho_incl':
            G[col] = (T.C6_rho_incl < 0) & (T.C6_p_incl < 0.01)
        elif rule is not None:
            G[col] = T[col].map(rule)
    return G


# --------------------------------------------------------------------------- figure
#: Columns of the summary matrix: (column, header, format, tolerance for the colour of a
#: non-graded value: |change from the paper analysis| at which the cell is fully shaded).
MATRIX = [
    ('C1_aniso_min', 'C1  R1 anisotropy', '%.1f', None), ('C2_aniso_max', 'C2  R2 anisotropy', '%.2f', None),
    ('C3_w_over_max', 'C3  R1 W / max', '%.2f', None), ('C4_w_over_mean', 'C4  R2 W / mean', '%.2f', None),
    ('C6_rho_incl', r'C6  $\rho$(track, incl.)', '%+.2f', None), ('C6_rho_A', r'C6  $\rho$(track, A)', '%+.2f', None),
    ('f12', 'C7  $f_{12}$', '%.3f', 0.06), ('f23', 'C7  $f_{23}$', '%.3f', 0.06),
    ('common_a2a1', 'C7  $a_2/a_1$', '%.3f', 0.05), ('common_a3a2', 'C7  $a_3/a_2$', '%.3f', 0.06),
    ('rho_incl_b2b1', r'C7c  $\rho$(incl., $b_2/b_1$)', '%+.2f', None),
    ('median_incl', 'C8  median incl. [deg]', '%.0f', 8),
    ('frac_excl_iso', 'C10  not isotropic', '%.2f', None), ('prolate_sigma', r'C11  prolate [$\sigma$]', '%.1f', None),
    ('sig_f12_95', 'C12  scatter $f_{12}$, 95%', '%.3f', None),
    ('sig_f23_95', 'C12  scatter $f_{23}$, 95%', '%.3f', None),
    ('n_collapsed', 'C13  unbounded', '%d', 2), ('n', 'windows', '%d', None),
]


def plot_matrix(T, ax=None):
    """Annotated matrix: rows = choices, columns = claims.  Green / red: the claim's test passes /
    fails; blue shading: change of a value from the paper analysis (no pass/fail test)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    G = grade(T)
    nr, nc = len(T), len(MATRIX)
    if ax is None:
        _, ax = plt.subplots(figsize=(7.25, 0.155 * nr + 1.3))
    img = np.ones((nr, nc, 3))
    base = T.iloc[0]
    for j, (col, _, fmt, tol) in enumerate(MATRIX):
        for i in range(nr):
            v = T[col].iloc[i]
            if col in G and isinstance(G[col].iloc[i], (bool, np.bool_)):
                img[i, j] = to_rgb('#cfe8cf' if G[col].iloc[i] else '#f4b6b0')
            elif tol is not None and np.isfinite(v):
                x = min(abs(v - base[col]) / tol, 1.0)
                img[i, j] = (1 - x) * np.ones(3) + x * np.array(to_rgb('#6aa6d8'))
            ax.text(j, i, (fmt % v) if np.isfinite(v) else '–', ha='center', va='center', fontsize=5.2)
    ax.imshow(img, aspect='auto', interpolation='nearest')
    ax.set_xticks(range(nc))
    ax.set_xticklabels([m[1] for m in MATRIX], fontsize=5.5, rotation=55, ha='left', rotation_mode='anchor')
    ax.xaxis.tick_top()
    ax.set_yticks(range(nr))
    ax.set_yticklabels(['%s  %s' % (g if g != 'base' else '', c) for g, c in zip(T.group, T.choice)], fontsize=5.5)
    ax.tick_params(length=0, pad=1.5)
    for y in np.flatnonzero(T.group.values[1:] != T.group.values[:-1]) + 0.5:
        ax.axhline(y, color='0.55', lw=0.4)
    for x in (5.5, 9.5, 11.5, 13.5, 15.5, 16.5):
        ax.axvline(x, color='0.55', lw=0.4)
    for s in ax.spines.values():
        s.set_linewidth(0.4)
    return ax
