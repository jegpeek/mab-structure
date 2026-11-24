"""A direct PSF forward-model fit of the structure function (tested, not used).

Each epoch image is the true field blurred in the (U, V) plane by the PSF,
plus noise.  For a statistically stationary field the blurred images have the
lag covariance C_obs = G (*) C_true, where G is the PSF's lag kernel (its
autocorrelation, `psf.lag_psf`) and the convolution runs over (dU, dV) only:
each epoch is a single W slice.  Since S2 = 2 C(0) - 2 C(lag), the measured
structure function is

    S2_obs(dU, dV, dW) = [G (*) S2_true](dU, dV, dW) - [G (*) S2_true](0, 0, 0) + noise,

with noise = s_i^2 + s_j^2 for epochs i != j and 2 s_i^2 (1 - rho_n(lag)) for
i == j (rho_n: the correlation of the resampling noise, which is short-range).

`fit_forward` fits the ellipsoidal power law S2_true with exactly the points,
weights and loss of `sf.fit_s2`, replacing the model by S2_obs above.  The
returned parameters are then intrinsic (no PSF correction needed).  Notebook 02
shows why the analysis does not use it.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import fftconvolve

from . import config as C, psf, sf


def fit_points_px(s2, inner_uv, min_same_epoch_lag_pix, s2_floor, min_n_fraction, stride):
    """The points of `sf._fit_points`, also as (pair index, dV pixel, dU pixel)."""
    a, n = s2['s2'], s2['n_counts']
    _, nv, nu = a.shape
    cv, cu = nv // 2, nu // 2
    p = int(min(inner_uv, cu, cv))
    off = np.arange(2 * p) - p
    RO, CO = np.meshgrid(off, off, indexing='ij')
    near0 = (np.abs(RO) <= min_same_epoch_lag_pix) & (np.abs(CO) <= min_same_epoch_lag_pix)
    smask = np.zeros((2 * p, 2 * p), bool); smask[::stride, ::stride] = True
    K, OV, OU, Y = [], [], [], []
    for k, (i, j) in enumerate(s2['epoch_pairs']):
        plane = a[k, cv - p:cv + p, cu - p:cu + p]
        m = np.isfinite(plane) & (n[k, cv - p:cv + p, cu - p:cu + p] >= min_n_fraction * n[k].max()) & smask
        if i == j:
            m &= ~near0
        K.append(np.full(m.sum(), k)); OV.append(RO[m]); OU.append(CO[m])
        Y.append(np.log10(np.maximum(plane[m], s2_floor)))
    return np.concatenate(K), np.concatenate(OV), np.concatenate(OU), np.concatenate(Y)


class BlurredPowerLaw:
    """log10 S2_obs of an ellipsoidal power law at the fit points of one S2 result."""

    def __init__(self, s2, k, ov, ou, G, noise_sigma=None, rho_n=None):
        self.G = np.asarray(G, float)
        kh = self.G.shape[0] // 2
        du = float(s2['lag_du'][1] - s2['lag_du'][0]); dv = float(s2['lag_dv'][1] - s2['lag_dv'][0])
        reach = int(max(np.abs(ov).max(), np.abs(ou).max()) + kh)
        g = np.arange(-reach, reach + 1)
        DV, DU = np.meshgrid(g * dv, g * du, indexing='ij')
        dws = np.round(np.asarray(s2['lag_dw'], float), 9)
        udw, inv = np.unique(dws, return_inverse=True)
        self.planes = [np.column_stack([DU.ravel(), DV.ravel(), np.full(DU.size, w)]) for w in udw]
        self.shape = DU.shape
        gc = np.arange(-kh, kh + 1)
        DVc, DUc = np.meshgrid(gc * dv, gc * du, indexing='ij')
        self.grid0 = np.column_stack([DUc.ravel(), DVc.ravel(), np.zeros(DUc.size)])
        self.pi, self.rr, self.cc = inv[k], ov + reach, ou + reach
        self.noise = 0.0
        if noise_sigma is not None:
            pairs = np.asarray(s2['epoch_pairs'])[k]
            s = np.asarray(noise_sigma, float)
            nz = s[pairs[:, 0]] ** 2 + s[pairs[:, 1]] ** 2
            same = pairs[:, 0] == pairs[:, 1]
            if rho_n is not None:
                rh = rho_n.shape[0] // 2
                inside = same & (np.abs(ov) <= rh) & (np.abs(ou) <= rh)
                nz[inside] *= 1 - rho_n[ov[inside] + rh, ou[inside] + rh]
            self.noise = nz

    def s2(self, p):
        geom, prof = p[:sf.N_GEOM], p[sf.N_GEOM:]
        planes = np.stack([10 ** sf.power_law(sf.ellipsoid_radius(geom, g), prof).reshape(self.shape)
                           for g in self.planes])
        conv = fftconvolve(planes, self.G[None], mode='same', axes=(1, 2))
        c0 = float(np.sum(self.G.ravel() * 10 ** sf.power_law(sf.ellipsoid_radius(geom, self.grid0), prof)))
        return conv[self.pi, self.rr, self.cc] - c0 + self.noise


def noise_correlation(J, extra_sigma=0.0325):
    """Correlation rho_n(dV, dU) of the resampling noise (detector x drizzle x tent shape)."""
    k1 = np.fft.fftfreq(psf.NFFT, d=C.PIX_LY)
    KU, KV = np.meshgrid(k1, k1)
    g = psf.lag_kernel(psf.transfer(KU, KV, J, *psf.symmetric_sky_mtf(extra_sigma))[1])
    return g / g[psf.KHALF, psf.KHALF]


def fit_forward(s2, G, guess, noise_sigma=None, rho_n=None, freeze=('A',), inner_uv=45, stride=2,
                weighting='1/r', min_same_epoch_lag_pix=4, s2_floor=10 ** -3.75, noise_scale_dex=0.1,
                min_n_fraction=0.1, max_nfev=None):
    """`sf.fit_s2` with S2_obs (PSF + optional noise) in place of the model.

    guess : starting parameters (normally the standard fit of the same data);
    parameters in `freeze` stay at their guess value.  Returns the same
    dict(params, chi2, n) as `sf.fit_s2`, so the chi2 of the two are directly
    comparable (same points, weights and scaling).
    """
    k, ov, ou, y = fit_points_px(s2, inner_uv, min_same_epoch_lag_pix, s2_floor, min_n_fraction, stride)
    lags = np.column_stack([ou * float(s2['lag_du'][1] - s2['lag_du'][0]),
                            ov * float(s2['lag_dv'][1] - s2['lag_dv'][0]), np.asarray(s2['lag_dw'])[k]])
    model = BlurredPowerLaw(s2, k, ov, ou, G, noise_sigma, rho_n)
    sw = np.sqrt(sf._weights(lags, s2, weighting))
    names = list(sf.GEOM_KEYS) + list(sf.power_law.param_names)
    p0 = np.array([float(guess[nm]) for nm in names])
    to_q = lambda p: np.r_[np.log(np.maximum(p[:3], 1e-30)), p[3:6], np.log(np.maximum(p[6:], 1e-30))]  # noqa: E731
    from_q = lambda q: np.r_[np.exp(q[:3]), q[3:6], np.exp(q[6:])]                                       # noqa: E731
    lo = np.array([-np.inf] * 6 + [np.log(1e-3), np.log(1e-6)])
    hi = np.array([np.log(10.0)] * 3 + [np.inf] * 3 + [np.inf, np.inf])
    free = ~np.array([nm in freeze for nm in names])
    q0 = np.clip(to_q(p0), lo + 1e-9, hi - 1e-9)

    def resid(x):
        q = q0.copy(); q[free] = x
        return sw * (y - np.log10(np.maximum(model.s2(from_q(q)), 1e-30))) / noise_scale_dex

    res = least_squares(resid, q0[free], bounds=(lo[free], hi[free]), x_scale='jac', max_nfev=max_nfev)
    q = q0.copy(); q[free] = res.x
    params = dict(zip(names, from_q(q)))
    return dict(params=params, chi2=float(np.sum(res.fun ** 2)), n=int(res.fun.size), success=bool(res.success))


# =========================================================================== the test of notebook 02
SYN_ALPHA = 0.5                       # close to the slope of the data
SYN_FIELDS = [(s, t, 11) for s in psf.SHAPES for t in (30, 60, 90)]
START_ALPHAS = (0.2, 0.4, 0.6, 0.8)  # starting slopes of the multi-start test


def _kw(cfg):
    return dict(inner_uv=cfg.inner_uv, stride=cfg.stride, weighting=cfg.weighting)


def _fkw(cfg):
    return dict(_kw(cfg), min_same_epoch_lag_pix=4, s2_floor=C.FIT_KW['s2_floor'],
                noise_scale_dex=C.FIT_KW['noise_scale_dex'], min_n_fraction=C.FIT_KW['min_n_fraction'])


def _row(prefix, f):
    s = sf.scalars(f['params'])        # axis ratios are NaN if an axis underflows to 0
    return {prefix + k: s.get(k, np.nan) for k in ('alpha', 'a1', 'a2', 'a3', 'a2a1', 'a3a2', 'incl')} | \
        {prefix + 'chi2': f['chi2']}


def real_setup(row, col, cfg=C.Config()):
    """Band-limited S2 of a data window, its PSF lag kernel, noise rms and noise correlation."""
    from . import geometry as GM, preprocess as P
    data = P.load_window(row, col, cfg)
    s2 = sf.compute_s2(data, **dict(C.S2_KW, assume_stationary=cfg.stationary))
    b = sf.band(s2, sf.lag_radius(s2), cfg.rmin, cfg.rcut)
    J, _ = GM.pixel_to_uv_jacobian(row, col, cfg.size)
    return b, psf.lag_psf(J), psf.window_noise_sigma(data, J), noise_correlation(J)


def synthetic_setup(spec, cfg=C.Config(), alpha=SYN_ALPHA):
    """The same for an injected field (as in the calibration, at slope `alpha`): observed and clean S2."""
    shape, tilt, seed = spec
    geo, J, sig_noise = psf.calibration_template()
    w_vals = np.asarray(geo[200]['W_values'], float)
    real = np.asarray(geo[200]['flux_epochs'], float)
    sig_real = np.sqrt(np.nanmedian([np.nanvar(e) for e in real if np.isfinite(e).any()]))
    sl = psf.make_field(psf.SHAPES[shape], (tilt, 30 + seed, 20 + 7 * seed), alpha, seed, w_vals) * sig_real
    obs = psf.blur(sl, J, seed + 1000, sig_noise)
    data = geo[cfg.size]; n = data['flux_epochs'].shape[1]; c = (psf.NB - n) // 2
    out = []
    for img in (obs, sl):
        d = dict(data); d['flux_epochs'] = img[:, c:c + n, c:c + n].astype(np.float32)
        s2 = sf.compute_s2(d, **dict(C.S2_KW, assume_stationary=cfg.stationary))
        out.append(sf.band(s2, sf.lag_radius(s2), cfg.rmin, cfg.rcut))
    return out[0], out[1], psf.lag_psf(J), sig_noise, noise_correlation(J)


def _real_job(rc):
    cfg = C.Config()
    b, G, sig, rho = real_setup(*rc, cfg)
    std = sf.fit_s2(b, freeze=('A',), **_kw(cfg), **C.FIT_KW)
    fwd = fit_forward(b, G, std['params'], noise_sigma=sig, rho_n=rho, **_fkw(cfg))
    return dict(kind='data', row=rc[0], col=rc[1], n=std['n'], **_row('std_', std), **_row('fwd_', fwd))


def _syn_job(spec):
    cfg = C.Config()
    b, bc, G, sig, rho = synthetic_setup(spec, cfg)
    std = sf.fit_s2(b, freeze=('A',), **_kw(cfg), **C.FIT_KW)
    clean = sf.fit_s2(bc, freeze=('A',), **_kw(cfg), **C.FIT_KW)
    fwd = fit_forward(b, G, std['params'], noise_sigma=sig, rho_n=rho, **_fkw(cfg))
    a = psf.SHAPES[spec[0]]
    return dict(kind='synthetic', shape=spec[0], tilt=spec[1], seed=spec[2], n=std['n'], alpha_true=SYN_ALPHA,
                a2a1_true=a[1] / a[0], a3a2_true=a[2] / a[1], **_row('clean_', clean), **_row('std_', std),
                **_row('fwd_', fwd))


def start_at(params, alpha0, r0=0.05):
    """Starting parameters with slope alpha0 and the same S2 at r0 [ly] (A is frozen,
    so the ellipsoid is rescaled: S2 = A (r / a)^alpha)."""
    s = sf.scalars(params)
    rho0 = r0 / (s['a1'] * s['a2'] * s['a3']) ** (1 / 3)
    k = np.exp(np.log(rho0) * (1 - params['alpha'] / alpha0))
    out = dict(params)
    for key in sf.GEOM_KEYS:
        out[key] = params[key] * k
    out['alpha'] = alpha0
    return out


def _multistart_job(args):
    kind, ident, alpha0 = args
    cfg = C.Config()
    if kind == 'data':
        b, G, sig, rho = real_setup(*ident, cfg)
    else:
        b, _, G, sig, rho = synthetic_setup(ident, cfg)
    std = sf.fit_s2(b, freeze=('A',), **_kw(cfg), **C.FIT_KW)
    fwd = fit_forward(b, G, start_at(std['params'], alpha0), noise_sigma=sig, rho_n=rho, **_fkw(cfg))
    return dict(kind=kind, ident=str(ident), alpha_start=alpha0, std_chi2=std['chi2'], **_row('fwd_', fwd))


def run_test(windows, procs=8):
    """Forward vs standard fits on the data windows and on injected fields, and a
    multi-start test (every 6th window and every injected field, from 4 starting slopes).

    Writes results/psf_forward_test.csv and results/psf_forward_multistart.csv.
    """
    import os
    from multiprocessing import Pool
    import pandas as pd
    from . import data as D
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ.setdefault(v, '1')
    windows = [tuple(map(int, w)) for w in windows]
    ms = [('data', w, a) for w in windows[::6] for a in START_ALPHAS] + \
         [('synthetic', f, a) for f in SYN_FIELDS for a in START_ALPHAS]
    with Pool(procs) as p:
        multi = p.map_async(_multistart_job, ms)
        syn = p.map_async(_syn_job, SYN_FIELDS)
        real = p.map_async(_real_job, windows)
        T = pd.DataFrame(syn.get() + real.get())
        M = pd.DataFrame(multi.get())
    T.to_csv(D.results('psf_forward_test.csv'), index=False)
    M.to_csv(D.results('psf_forward_multistart.csv'), index=False)
    return T, M
