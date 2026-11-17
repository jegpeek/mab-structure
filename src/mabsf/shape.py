"""Population statistics of the window shapes: common shape, scatter, prolateness, orientation.

**Two descriptions of the same anisotropy.**  For the fitted power law
S2 = A (rho)^alpha, the structure function at a fixed separation r along
principal axis i is A (r / a_i)^alpha.  So

    f12 = S2(along a1) / S2(along a2) = (a2/a1)^alpha     (long vs middle axis)
    f23 = S2(along a2) / S2(along a3) = (a3/a2)^alpha     (middle vs short)

are *variance ratios* in [0, 1], read directly off the data inside the fitted
range, while the axis ratios a2/a1 = f12^(1/alpha) extrapolate along the long
axis.  With 1/alpha ~ 2, small f12 becomes a tiny a2/a1 with an enormous error
-- the "collapsed" fits.  Population statistics are therefore done on f12,
f23 (every window enters with a finite error) and converted back to axis
ratios to state the shape.

* `window_table`   per-window PSF-corrected fits, f12, f23 and jackknife errors
* `ml_lin`         ML common value and intrinsic scatter of f (linear, Gaussian)
* `log_claims`     the same statistics in log axis ratios (the paper's first form),
                   prolateness, isotropy test, orientation
* `summary`        every number quoted in the text, in one dict
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize, minimize_scalar

from . import config as C
from . import fitting as F
from . import preprocess as P
from . import psf

R1_TILE = (C.REGION_CENTERS[1][0] - C.REGION_BOX // 2, C.REGION_CENTERS[1][1] - C.REGION_BOX // 2)


# =========================================================================== per window
def usable(d, cols=('a2a1', 'a3a2', 'b2b1', 'incl')):
    """Rows with finite, positive values (and finite SEs) in every column."""
    m = d.fit_success.astype(bool)
    for c in cols:
        m &= np.isfinite(d[c]) & (d[c] > 0)
        if 'se_' + c in d:
            m &= np.isfinite(d['se_' + c])
    return m


def window_table(cfg=C.Config(), windows=None):
    """PSF-corrected 3D fits of the sample, one row per window.

    Columns: row, col, a2a1, a3a2, alpha, incl (corrected), their jackknife SEs,
    b2b1_2d (in-plane fit, corrected), f12, f23, se_f12, se_f23, se_l12,
    collapsed (a2/a1 or a3/a2 < cfg.degen).
    """
    wins = P.sample_windows(cfg) if windows is None else windows
    d3 = F.table(wins, cfg, '3d')
    d2 = F.table(wins, cfg, '2d')
    k = psf.correct(d3, cfg.calib, degen=cfg.degen)
    r2 = d2[['row', 'col', 'b2b1']].merge(d3[['row', 'col', 'incl']], on=['row', 'col'])
    r2['a2a1'] = 1.0; r2['a3a2'] = 1.0
    r2 = psf.correct(r2, cfg.calib, degen=cfg.degen)
    k = k.merge(r2[['row', 'col', 'b2b1']].rename(columns={'b2b1': 'b2b1_2d'}), on=['row', 'col'])
    k['collapsed'] = (k.a2a1 < cfg.degen) | (k.a3a2 < cfg.degen)
    a = k.alpha.clip(lower=0.05)
    k['f12'] = k.a2a1 ** a
    k['f23'] = k.a3a2.clip(upper=1) ** a
    return k


# =========================================================================== variance ratios
def ml_lin(x, s):
    """ML mean and intrinsic scatter of values x with errors s; p-value of zero scatter.

    Model x_i ~ N(mu, s_i^2 + sigma^2); the p-value is the likelihood-ratio test
    of sigma = 0 (chi2 with 1 dof).
    """
    x = np.asarray(x, float); s = np.maximum(np.asarray(s, float), 1e-3)

    def nll(p, zero=False):
        g = 0.0 if zero else np.exp(p[1]); v = s ** 2 + g ** 2
        return 0.5 * np.sum((x - p[0]) ** 2 / v + np.log(v))
    b = min((minimize(nll, [np.median(x), np.log(z)], method='Nelder-Mead') for z in (0.02, 0.08, 0.2)),
            key=lambda q: q.fun)
    b0 = minimize(lambda p: nll(p, True), [np.median(x)], method='Nelder-Mead')
    return b.x[0], np.exp(b.x[1]), stats.chi2.sf(2 * (b0.fun - b.fun), 1)


def bootstrap(x, s, n=600, seed=5):
    """(n, 2) bootstrap over windows of (mean, intrinsic scatter)."""
    rng = np.random.default_rng(seed)
    return np.array([ml_lin(x[i], s[i])[:2] for i in (rng.integers(0, len(x), len(x)) for _ in range(n))])


def variance_ratio_stats(k, n_boot=600):
    m12, s12, p12 = ml_lin(k.f12.values, k.se_f12.values)
    m23, s23, p23 = ml_lin(k.f23.values, k.se_f23.values)
    out = dict(n=len(k), f12=m12, sig_f12=s12, p0_f12=p12, f23=m23, sig_f23=s23, p0_f23=p23,
               median_alpha=float(k.alpha.median()))
    if n_boot:
        b12 = bootstrap(k.f12.values, k.se_f12.values, n_boot)
        b23 = bootstrap(k.f23.values, k.se_f23.values, n_boot)
        out.update(sig_f12_68=np.percentile(b12[:, 1], [16, 84]).tolist(), sig_f12_95=float(np.percentile(b12[:, 1], 97.5)),
                   sig_f23_68=np.percentile(b23[:, 1], [16, 84]).tolist(), sig_f23_95=float(np.percentile(b23[:, 1], 97.5)))
    return out


def to_axis_ratio(f, alpha):
    return np.asarray(f, float) ** (1.0 / np.asarray(alpha, float))


def spread_in_axis_ratio(mu, sig, alphas=None, alpha=0.5, n=200000, seed=1):
    """Spread (half the 16-84% range, dex) of a = f^(1/alpha) for f ~ N(mu, sig).

    With `alphas`, each draw uses a randomly chosen window slope (an upper
    bracket: the slope spread includes measurement noise); otherwise a single
    alpha.
    """
    rng = np.random.default_rng(seed)
    f = np.clip(rng.normal(mu, sig, n), 1e-4, 1)
    a = rng.choice(np.asarray(alphas), n) if alphas is not None else alpha
    la = np.log10(f ** (1 / a))
    return float(np.subtract(*np.percentile(la, [84, 16])) / 2)


def upper_limits(k, nsig=2):
    """nsig upper limit on a2/a1 of the collapsed windows: (f12 + nsig se)^(1/alpha), capped at 1."""
    u = k[k.collapsed]
    return {(int(r), int(c)): float(min(f + nsig * e, 1.0) ** (1 / a))
            for r, c, f, e, a in zip(u.row, u.col, u.f12, u.se_f12, u.alpha)}


# =========================================================================== log axis ratios
def _nll_log(ls, lv, sl):
    s2t = sl ** 2 + (10 ** ls) ** 2
    w = 1.0 / s2t
    mu = np.sum(w * lv) / np.sum(w)
    return 0.5 * np.sum((lv - mu) ** 2 / s2t + np.log(s2t))


def weighted_center(lv, sl, sig=0.0):
    w = 1.0 / (np.asarray(sl) ** 2 + sig ** 2)
    return float(np.sum(w * lv) / np.sum(w)), float(1.0 / np.sqrt(np.sum(w)))


def ml_log(lv, sl, bounds=(-4, 1)):
    """ML common log10 value and intrinsic scatter [dex] (scatter profiled on a bounded log grid)."""
    lv, sl = np.asarray(lv, float), np.asarray(sl, float)
    r = minimize_scalar(_nll_log, bounds=bounds, method='bounded', args=(lv, sl))
    sig = float(10 ** r.x)
    return weighted_center(lv, sl, sig)[0], sig


def profile_ci(lv, sl, bounds=(-4, 1), ngrid=4001):
    """ML scatter, its 1-sigma profile interval and the delta-chi2 of zero scatter."""
    lv, sl = np.asarray(lv, float), np.asarray(sl, float)
    r = minimize_scalar(_nll_log, bounds=bounds, method='bounded', args=(lv, sl))
    n0 = _nll_log(r.x, lv, sl)
    g = np.linspace(bounds[0], bounds[1], ngrid)
    dn = 2 * np.array([_nll_log(x, lv, sl) for x in g]) - 2 * n0
    ok = dn <= 1.0
    return float(10 ** r.x), float(10 ** g[ok].min()), float(10 ** g[ok].max()), float(dn[0])


def log_claims(k, cfg=C.Config()):
    """Shape statistics in log axis ratios, prolateness, isotropy and orientation.

    k: window_table(cfg).  Keys follow the claims inventory (docs/claims_inventory.md).
    """
    u = k[usable(k)].copy()
    col = ((u.a2a1 < cfg.degen) | (u.a3a2 < cfg.degen)).values
    out = dict(n_usable=len(u), C13_n_collapsed=int(col.sum()),
               C13_R1_collapsed=bool(any((r, c) == R1_TILE for r, c in zip(u.row[col], u.col[col]))))
    s = u if cfg.collapsed == 'include' else u[~col]
    lsl = {q: (s['se_' + q] / s[q] / np.log(10)).values for q in ('a2a1', 'a3a2')}
    for q in ('a2a1', 'a3a2'):
        lv = np.log10(s[q].values)
        mu, sig = ml_log(lv, lsl[q])
        out['C7_common_' + q] = 10 ** mu
        out['C12_sigint_' + q] = sig
        out['C10_dist_sigma_' + q] = -mu / weighted_center(lv, lsl[q], sig)[1]
        out['C12_p_zero_' + q] = stats.chi2.sf(profile_ci(lv, lsl[q])[3], 1)
    # prolateness: log(a3/a2) - log(a2/a1) > 0
    Pv = np.log10(s.a3a2.values) - np.log10(s.a2a1.values)
    sP = np.hypot(lsl['a2a1'], lsl['a3a2'])
    muP, sigP = ml_log(Pv, sP)
    seP = 1 / np.sqrt((1 / (sP ** 2 + sigP ** 2)).sum())
    out.update(C11_prolate_dex=muP, C11_prolate_sigma=muP / seP, C11_frac_prolate=float((Pv > 0).mean()))
    # isotropy: windows whose two ratios are both within 2 sigma of 1
    z1 = np.abs(np.log10(s.a2a1.values)) / lsl['a2a1']; z2 = np.abs(np.log10(s.a3a2.values)) / lsl['a3a2']
    inf = ~(((s.a2a1 < cfg.degen) | (s.a3a2 < cfg.degen)).values)
    out['C10_n_consistent_isotropic'] = int(((z1 <= 2) & (z2 <= 2) & inf).sum())
    out['C10_n_informative'] = int(inf.sum())
    # one shape, many orientations: sky-plane anisotropy follows the inclination
    j = s[['incl', 'b2b1_2d']].dropna()
    rho, p = stats.spearmanr(j.incl, j.b2b1_2d)
    out.update(C7_rho_incl_b2b1=rho, C7_rho_p=p)
    v = s.incl.dropna()
    out.update(C8_median_incl=float(v.median()), C8_ks_p=stats.kstest(np.cos(np.radians(v)), 'uniform').pvalue,
               C9_frac_30_60=float(((v >= 30) & (v <= 60)).mean()))
    return out


# =========================================================================== everything quoted
def summary(cfg=C.Config(), n_boot=600):
    k = window_table(cfg)
    vr = variance_ratio_stats(k, n_boot)
    am = vr['median_alpha']
    out = dict(log_claims(k, cfg), variance_ratio=vr, median_alpha=am,
               common_a2a1=float(vr['f12'] ** (1 / am)), common_a3a2=float(vr['f23'] ** (1 / am)),
               upper_limits_a2a1={'%d_%d' % rc: v for rc, v in upper_limits(k).items()},
               n_prolate_f=int((k.f12 < k.f23).sum()),
               n_exclude_isotropy_2sig=int(((k.f12 + 2 * k.se_f12 < 1) | (k.f23 + 2 * k.se_f23 < 1)).sum()),
               rho_f23_incl=float(stats.spearmanr(k.f23, k.incl)[0]),
               rho_f12_incl=float(stats.spearmanr(k.f12, k.incl)[0]))
    if n_boot:
        al = k.alpha.values
        for q, mu, sg, s95 in (('a2a1', vr['f12'], vr['sig_f12'], vr['sig_f12_95']),
                               ('a3a2', vr['f23'], vr['sig_f23'], vr['sig_f23_95'])):
            out['spread_dex_' + q] = dict(single_alpha=spread_in_axis_ratio(mu, sg),
                                          single_alpha_95=spread_in_axis_ratio(mu, s95),
                                          window_alpha=spread_in_axis_ratio(mu, sg, al),
                                          window_alpha_95=spread_in_axis_ratio(mu, s95, al))
    return out, k
