"""The 3D structure function and its ellipsoidal power-law model.

**Structure function.**  For a window with images f_i at depths W_i, the
second-order structure function at lag (dU, dV) between epochs i and j is

    S2_ij(dU, dV) = < [f_j(x + d) - f_i(x)]^2 >,      dW = W_j - W_i,

averaged over all valid pixel pairs.  Pairs i = j give the in-plane (dW = 0)
structure function; i != j sample lags with a depth component.  All pairs are
computed at once with FFT cross-correlations of the masked images, so missing
pixels are excluded exactly (`compute_s2`).

**Model.**  S2 depends on the lag only through an ellipsoidal radius,

    S2 = A * rho^alpha,     rho = | L^{-1} (dU, dV, dW) |,

where L is upper triangular (s11, s22, s33 on the diagonal, l12, l13, l23
above it).  C = L L^T is the shape matrix of the surfaces of constant S2;
its eigenvectors are the principal axes and the square roots of its
eigenvalues the semi-axes a1 >= a2 >= a3.  The overall size is degenerate
with A, so A is held fixed and only the axis ratios and orientation are
measured.  Fits are least squares in log10 S2, with weights 1/r per lag
(`fit_s2`).
"""
from __future__ import annotations

from itertools import combinations
from types import SimpleNamespace

import numpy as np
from scipy.fft import irfft2, next_fast_len, rfft2
from scipy.optimize import least_squares

GEOM_KEYS = ['s11', 's22', 's33', 'l12', 'l13', 'l23']
N_GEOM = 6


# =========================================================================== S2
def _xcorr_full(a_fft, b_fft, fft_shape, n_rows, n_cols):
    """C[dr, dc] = sum a[r, c] b[r + dr, c + dc], all lags, FFT order."""
    c = irfft2(np.conj(a_fft) * b_fft, s=fft_shape)
    fr, fc = fft_shape
    out = np.empty((2 * n_rows - 1, 2 * n_cols - 1))
    out[:n_rows, :n_cols] = c[:n_rows, :n_cols]
    out[n_rows:, :n_cols] = c[fr - n_rows + 1:fr, :n_cols]
    out[:n_rows, n_cols:] = c[:n_rows, fc - n_cols + 1:fc]
    out[n_rows:, n_cols:] = c[fr - n_rows + 1:fr, fc - n_cols + 1:fc]
    return out


def compute_s2(data, clip_percentiles=(0.002, 0.998), background=0.0, arcsinh_scale=None,
               subtract_mean='global', assume_stationary=True):
    """Structure function of one window for every epoch pair.

    data : dict with flux_epochs (n_ep, ny, nx; NaN = missing), U_grid, V_grid
           (ny, nx) [ly] and W_values (n_ep,) [ly], as from
           mabsf.preprocess.read_window.

    Steps: subtract `background`; mask each epoch's extreme pixels outside the
    given percentiles; optionally apply arcsinh(f / arcsinh_scale); subtract the
    mean over all epochs ('global'), per epoch ('epoch') or not at all.  With
    assume_stationary, S2 = 2 <f^2> - 2 <f_i f_j>(lag), with <f^2> taken per
    epoch over its valid pixels.

    Returns a dict with s2 and n_counts (n_pairs, 2ny-1, 2nx-1), centred
    (fftshifted) lag axes lag_dv, lag_du [ly], lag_dw (n_pairs,) and
    epoch_pairs: the same-epoch pairs (i, i) first, then (i, j) with i < j.
    """
    flux = np.array(data['flux_epochs'], copy=True)      # keeps the input dtype (float32 or float64)
    n_ep = flux.shape[0]
    if background is not None:
        bg = np.broadcast_to(np.asarray(background, dtype=float), (n_ep,))
        for e in range(n_ep):
            flux[e] -= bg[e]
    if clip_percentiles is not None:
        lo_f, hi_f = clip_percentiles
        for e in range(n_ep):
            plane = flux[e]
            valid = plane[np.isfinite(plane)]
            if valid.size == 0:
                continue
            lo, hi = np.percentile(valid, lo_f * 100), np.percentile(valid, hi_f * 100)
            flux[e] = np.where((plane < lo) | (plane > hi), np.nan, plane)
    if arcsinh_scale is not None:
        flux = np.arcsinh(flux / arcsinh_scale)
    if subtract_mean == 'global':
        vals = np.concatenate([flux[e][np.isfinite(flux[e])] for e in range(n_ep)])
        flux -= float(vals.mean()) if vals.size else 0.0
    elif subtract_mean == 'epoch':
        for e in range(n_ep):
            flux[e] -= np.nanmean(flux[e])
    elif subtract_mean != 'none':
        raise ValueError(subtract_mean)

    U_grid, V_grid, W = data['U_grid'], data['V_grid'], np.asarray(data['W_values'], float)
    _, n_rows, n_cols = flux.shape
    du = float(U_grid[0, 1] - U_grid[0, 0])
    dv = float(V_grid[1, 0] - V_grid[0, 0])
    lag_du = np.r_[np.arange(n_cols), np.arange(-(n_cols - 1), 0)] * du
    lag_dv = np.r_[np.arange(n_rows), np.arange(-(n_rows - 1), 0)] * dv
    pairs = [(e, e) for e in range(n_ep)] + list(combinations(range(n_ep), 2))
    lag_dw = np.array([float(W[j] - W[i]) for i, j in pairs])

    fft_shape = (next_fast_len(2 * n_rows - 1), next_fast_len(2 * n_cols - 1))
    masks = np.isfinite(flux).astype(np.float64)
    f_ = np.where(masks.astype(bool), flux, 0.0)
    M = [rfft2(masks[e], s=fft_shape) for e in range(n_ep)]
    MF = [rfft2(masks[e] * f_[e], s=fft_shape) for e in range(n_ep)]
    if assume_stationary:
        ev = [float((f_[e][masks[e].astype(bool)] ** 2).mean()) if masks[e].any() else 0.0 for e in range(n_ep)]
        MF2 = [ev[e] * M[e] for e in range(n_ep)]
    else:
        MF2 = [rfft2(masks[e] * f_[e] ** 2, s=fft_shape) for e in range(n_ep)]

    shape = (len(pairs), 2 * n_rows - 1, 2 * n_cols - 1)
    s2 = np.full(shape, np.nan, dtype=np.float32)
    n_counts = np.zeros(shape, dtype=np.int32)
    xc = lambda a, b: _xcorr_full(a, b, fft_shape, n_rows, n_cols)   # noqa: E731
    for k, (i, j) in enumerate(pairs):
        N = xc(M[i], M[j])
        dsq = xc(MF2[i], M[j]) + xc(M[i], MF2[j]) - 2.0 * xc(MF[i], MF[j])
        ok = N >= 0.5
        s2[k][ok] = dsq[ok] / N[ok]
        n_counts[k] = N
    return dict(s2=np.fft.fftshift(s2, axes=(1, 2)), n_counts=np.fft.fftshift(n_counts, axes=(1, 2)),
                lag_du=np.fft.fftshift(lag_du), lag_dv=np.fft.fftshift(lag_dv), lag_dw=lag_dw,
                epoch_pairs=pairs, w_values=W.copy())


def lag_radius(s2):
    """|(dU, dV, dW)| [ly] on the (n_pairs, n_lag_v, n_lag_u) lag grid."""
    DV, DU = np.meshgrid(s2['lag_dv'], s2['lag_du'], indexing='ij')
    return np.sqrt((DU ** 2 + DV ** 2)[None] + np.asarray(s2['lag_dw'], float)[:, None, None] ** 2)


def band(s2, r_grid, lo, hi):
    """Copy of s2 with S2 set to NaN outside lo <= |lag| < hi."""
    out = dict(s2)
    a = np.array(s2['s2'], copy=True)
    a[~((r_grid >= lo) & (r_grid < hi))] = np.nan
    out['s2'] = a
    return out


def in_plane(s2):
    """Copy of s2 keeping only the dW = 0 (same-epoch) planes."""
    out = dict(s2)
    a = np.array(s2['s2'], copy=True)
    a[np.asarray(s2['lag_dw'], float) != 0.0] = np.nan
    out['s2'] = a
    return out


# =========================================================================== model
def power_law(r, params):
    """log10 S2 = log10 A + alpha log10 rho."""
    alpha, A = params
    return np.log10(A) + alpha * np.log10(r)


power_law.n_params = 2
power_law.param_names = ['alpha', 'A']
power_law.default_guess = [0.4, None]            # None: median S2 of the data
power_law.param_bounds = [(1e-3, np.inf), (1e-6, np.inf)]


def ellipsoid_radius(geom, lags):
    """rho = |L^{-1} lag| for lags (N, 3) = (dU, dV, dW)."""
    s11, s22, s33, l12, l13, l23 = geom
    xw = lags[:, 2] / s33
    xv = (lags[:, 1] - l23 * xw) / s22
    xu = (lags[:, 0] - l12 * xv - l13 * xw) / s11
    return np.maximum(np.sqrt(xu * xu + xv * xv + xw * xw), 1e-100)


def log_s2_model(params, lags, profile=power_law):
    if isinstance(params, dict):
        geom = [params[k] for k in GEOM_KEYS]; prof = [params[k] for k in profile.param_names]
    else:
        geom, prof = params[:N_GEOM], params[N_GEOM:]
    return profile(ellipsoid_radius(geom, lags), prof)


def params_from_axes(a1, a2, a3, theta, phi, psi):
    """L-matrix parameters of an ellipsoid with semi-axes a1 >= a2 >= a3.

    theta : polar angle of a1 from W [rad]; phi : azimuth of a1 from U toward
    V [rad]; psi : roll of a2, a3 about a1 [rad].
    """
    n1 = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    n2_0 = np.array([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)])
    n3_0 = np.array([-np.sin(phi), np.cos(phi), 0.0])
    n2 = np.cos(psi) * n2_0 + np.sin(psi) * n3_0
    n3 = -np.sin(psi) * n2_0 + np.cos(psi) * n3_0
    Q = np.column_stack([n1, n2, n3])
    Cm = Q @ np.diag([a1 ** 2, a2 ** 2, a3 ** 2]) @ Q.T
    s33 = np.sqrt(Cm[2, 2]); l23 = Cm[1, 2] / s33; l13 = Cm[0, 2] / s33
    s22 = np.sqrt(Cm[1, 1] - l23 ** 2); l12 = (Cm[0, 1] - l23 * l13) / s22
    s11 = np.sqrt(Cm[0, 0] - l12 ** 2 - l13 ** 2)
    return dict(s11=s11, s22=s22, s33=s33, l12=l12, l13=l13, l23=l23)


def axes_from_params(params):
    """Semi-axes (a1 >= a2 >= a3) and orientation (theta, phi, psi) [rad] of L."""
    if isinstance(params, dict):
        s11, s22, s33, l12, l13, l23 = (params[k] for k in GEOM_KEYS)
    else:
        s11, s22, s33, l12, l13, l23 = params[:6]
    L = np.array([[s11, l12, l13], [0.0, s22, l23], [0.0, 0.0, s33]])
    ev, Q = np.linalg.eigh(L @ L.T)
    idx = np.argsort(ev)[::-1]; ev = ev[idx]; Q = Q[:, idx]
    if np.linalg.det(Q) < 0:
        Q[:, -1] *= -1
    a1, a2, a3 = np.sqrt(np.maximum(ev, 0.0))
    n1 = Q[:, 0]
    theta = np.arccos(np.clip(n1[2], -1.0, 1.0)); phi = np.arctan2(n1[1], n1[0])
    n2_0 = np.array([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)])
    n3_0 = np.array([-np.sin(phi), np.cos(phi), 0.0])
    psi = np.arctan2(Q[:, 1] @ n3_0, Q[:, 1] @ n2_0)
    return dict(a1=a1, a2=a2, a3=a3, theta=theta, phi=phi, psi=psi, n1=n1)


def in_plane_axes(params):
    """Axes b1 >= b2 and position angle of the dW = 0 slice of the ellipsoid.

    At dW = 0 only s11, s22, l12 enter, so the in-plane shape is the central
    slice C2 = L2 L2^T (not the projection) -- what a single epoch measures.
    """
    if isinstance(params, dict):
        s11, s22, l12 = params['s11'], params['s22'], params['l12']
    else:
        s11, s22, l12 = params[0], params[1], params[3]
    L2 = np.array([[s11, l12], [0.0, s22]])
    ev, Q = np.linalg.eigh(L2 @ L2.T)
    idx = np.argsort(ev)[::-1]
    b1, b2 = np.sqrt(np.maximum(ev[idx], 0.0))
    n1 = Q[:, idx[0]]
    return dict(b1=float(b1), b2=float(b2), pa=float(np.arctan2(n1[1], n1[0]) % np.pi))


def scalars(params):
    """Derived quantities of a fit: axis ratios, angles [deg], drift, in-plane shape."""
    d = {k: float(v) for k, v in params.items()}
    ax = axes_from_params(params)
    d.update(a1=ax['a1'], a2=ax['a2'], a3=ax['a3'], theta=float(np.degrees(ax['theta'])),
             phi=float(np.degrees(ax['phi'])), psi=float(np.degrees(ax['psi'])),
             du_per_dw=params['l13'] / params['s33'], dv_per_dw=params['l23'] / params['s33'])
    ax2 = in_plane_axes(params)
    d.update(b1=ax2['b1'], b2=ax2['b2'], b2b1=ax2['b2'] / ax2['b1'] if ax2['b1'] > 0 else np.nan,
             pa2d=float(np.degrees(ax2['pa'])))
    a1, a2, a3 = d['a1'], d['a2'], d['a3']
    if a1 > 0 and a2 > 0 and a3 > 0:
        d.update(a2a1=a2 / a1, a3a2=a3 / a2, a3a1=a3 / a1)
    # inclination: angle of the long axis from W folded into [0, 90] deg
    if np.isfinite(d['theta']):
        d['incl'] = float(90.0 - abs(90.0 - (d['theta'] % 180.0)))
    return d


# =========================================================================== fitting
def _fit_points(s2, inner_uv, min_same_epoch_lag_pix, s2_floor, min_n_fraction, stride):
    """Lags and log10 S2 of the points entering the fit.

    Keeps |dU|, |dV| < inner_uv pixels, lags sampled by at least
    min_n_fraction of the best-sampled lag of that pair, every `stride`-th
    pixel, and drops the smallest same-epoch lags (|d| <= 4 px), where pixel
    noise dominates.
    """
    a, n = s2['s2'], s2['n_counts']
    _, nv, nu = a.shape
    cv, cu = nv // 2, nu // 2
    p = int(min(inner_uv, cu, cv))
    vs, us = slice(cv - p, cv + p), slice(cu - p, cu + p)
    DV, DU = np.meshgrid(s2['lag_dv'][vs], s2['lag_du'][us], indexing='ij')
    off = np.abs(np.arange(2 * p) - p)
    RO, CO = np.meshgrid(off, off, indexing='ij')
    near0 = (RO <= min_same_epoch_lag_pix) & (CO <= min_same_epoch_lag_pix)
    smask = None
    if stride > 1:
        smask = np.zeros((2 * p, 2 * p), bool); smask[::stride, ::stride] = True
    L, Y = [], []
    for k, (i, j) in enumerate(s2['epoch_pairs']):
        plane = a[k, vs, us]
        m = np.isfinite(plane) & (n[k, vs, us] >= min_n_fraction * n[k].max())
        if i == j:
            m &= ~near0
        if smask is not None:
            m &= smask
        sel = m.ravel()
        L.append(np.column_stack([DU.ravel()[sel], DV.ravel()[sel], np.full(sel.sum(), s2['lag_dw'][k])]))
        Y.append(np.log10(np.maximum(plane, s2_floor)).ravel()[sel])
    return np.concatenate(L), np.concatenate(Y)


def _weights(lags, s2, weighting):
    if weighting is None:
        return np.ones(len(lags))
    r = np.maximum(np.hypot(lags[:, 0], lags[:, 1]), abs(s2['lag_du'][1] - s2['lag_du'][0]))
    return 1.0 / r if weighting == '1/r' else 1.0 / r ** 2


def _uvshift_guess(s2, inner=200):
    """dU/dW, dV/dW of the S2 minimum in the shortest-baseline cross-epoch plane."""
    du_, dv_ = s2['lag_du'], s2['lag_dv']
    cu, cv = len(du_) // 2, len(dv_) // 2
    us = slice(max(cu - inner, 0), min(cu + inner + 1, len(du_)))
    vs = slice(max(cv - inner, 0), min(cv + inner + 1, len(dv_)))

    def refine(arr, i):
        if i == 0 or i == len(arr) - 1:
            return float(i)
        a, b, c = arr[i - 1], arr[i], arr[i + 1]
        if not (np.isfinite(a) and np.isfinite(c)):
            return float(i)
        den = 2.0 * (a - 2.0 * b + c)
        return i - (c - a) / den if den != 0.0 else float(i)

    dws = s2['lag_dw']
    for k in sorted((k for k, w in enumerate(dws) if w != 0.0), key=lambda k: abs(dws[k])):
        p = s2['s2'][k][vs, us].copy(); p[~np.isfinite(p)] = np.inf
        iv, iu = np.unravel_index(np.argmin(p), p.shape)
        if iu in (0, p.shape[1] - 1) or iv in (0, p.shape[0] - 1):
            continue
        u = du_[us][0] + refine(p[iv, :], iu) * (du_[1] - du_[0])
        v = dv_[vs][0] + refine(p[:, iu], iv) * (dv_[1] - dv_[0])
        return u / dws[k], v / dws[k]
    raise ValueError('no usable cross-epoch plane')


def _geom_from_shift(du, dv, scale, elongation=5.0):
    """Starting ellipsoid: prolate, elongation 5, tilted to reproduce the drift (du, dv)."""
    phi = np.arctan2(dv, du); mag = np.hypot(du, dv); e = elongation
    if e <= 1.0 or mag == 0.0:
        theta = np.pi / 2.0
    else:
        from scipy.optimize import brentq

        def f(th):
            s, c = np.sin(th), np.cos(th)
            return s * c * (e ** 2 - 1.0) / (e ** 2 * c ** 2 + s ** 2) - mag
        th = np.linspace(1e-6, np.pi / 2 - 1e-6, 1000); fv = np.vectorize(f)(th); ip = np.argmax(fv)
        theta = th[ip] if fv[ip] <= 0 else brentq(f, 1e-6, th[ip])
    return params_from_axes(scale * e, scale, scale, theta, phi, 0.0)


def fit_s2(s2, profile=power_law, guess=None, inner_uv=45, stride=2, weighting='1/r', freeze=('A',),
           min_same_epoch_lag_pix=4, s2_floor=10 ** -3.75, noise_scale_dex=0.1, min_n_fraction=0.1,
           max_nfev=None):
    """Least-squares fit of the ellipsoidal model to log10 S2.

    Start: an ellipsoid whose tilt reproduces the drift of the S2 minimum
    between the closest epochs, at the best of four trial sizes; the profile
    (alpha) is pre-fitted with the geometry fixed, then everything is fitted
    together.  s11, s22, s33 are fitted in log space (bounded at 10 ly), so
    they stay positive.  Parameters named in `freeze` are held at their start
    value -- A for the power law (the size-amplitude degeneracy); for the
    in-plane fit also s33, l13, l23, which dW = 0 data do not constrain.

    Returns dict(params=..., fit=scipy result, n=points, chi2=...).
    """
    lags, y = _fit_points(s2, inner_uv, min_same_epoch_lag_pix, s2_floor, min_n_fraction, stride)
    cv, cu = s2['lag_dv'].size // 2, s2['lag_du'].size // 2
    inner = s2['s2'][:, cv - inner_uv:cv + inner_uv, cu - inner_uv:cu + inner_uv]
    amp = float(np.nanmedian(inner[np.isfinite(inner)]))
    g = guess or {}
    try:
        du, dv = _uvshift_guess(s2)
    except Exception:     # noqa: BLE001
        du, dv = 0.0, 0.0
    pdef = [amp if d is None else d for d in profile.default_guess]
    best, bscale = np.inf, 0.05
    for sc in (0.05, 0.1, 0.2, 0.3):
        gm = _geom_from_shift(du, dv, sc)
        c2 = float(np.mean((y - log_s2_model(np.array([gm[k] for k in GEOM_KEYS] + pdef), lags, profile)) ** 2))
        if c2 < best:
            best, bscale = c2, sc
    ag = _geom_from_shift(du, dv, bscale)
    pb = profile.param_bounds
    p0 = np.array([g.get(k, ag[k]) for k in GEOM_KEYS] +
                  [np.clip(g.get(nm, amp if d is None else d), b[0], b[1])
                   for nm, d, b in zip(profile.param_names, profile.default_guess, pb)])
    sw = np.sqrt(_weights(lags, s2, weighting))
    plog = [b[0] > 0 for b in pb]

    def to_q(p):
        q = p.copy(); q[:3] = np.log(np.maximum(p[:3], 1e-30))
        for i, lg in enumerate(plog):
            if lg:
                q[N_GEOM + i] = np.log(max(p[N_GEOM + i], 1e-30))
        return q

    def from_q(q):
        p = q.copy(); p[:3] = np.exp(q[:3])
        for i, lg in enumerate(plog):
            if lg:
                p[N_GEOM + i] = np.exp(q[N_GEOM + i])
        return p

    lo = np.array([-np.inf] * 6 + [np.log(b[0]) if lg else b[0] for b, lg in zip(pb, plog)])
    hi = np.array([np.log(10.0)] * 3 + [np.inf] * 3 +
                  [np.log(b[1]) if (lg and np.isfinite(b[1])) else b[1] for b, lg in zip(pb, plog)])
    names = list(GEOM_KEYS) + list(profile.param_names)
    bad = [f for f in freeze if f not in names]
    if bad:
        raise ValueError('unknown parameter(s) %s' % bad)
    free = ~np.array([nm in freeze for nm in names])
    q0 = to_q(p0)

    def resid(q):
        return sw * (y - log_s2_model(from_q(q), lags, profile)) / noise_scale_dex

    gq, pq = q0[:N_GEOM], q0[N_GEOM:].copy()
    pf = free[N_GEOM:]
    if pf.any():
        def rp(x):
            z = pq.copy(); z[pf] = x
            return resid(np.concatenate([gq, z]))
        pre = least_squares(rp, pq[pf], loss='linear', max_nfev=200, bounds=(lo[N_GEOM:][pf], hi[N_GEOM:][pf]))
        pq[pf] = pre.x
    q0 = np.concatenate([gq, pq])

    def embed(x):
        q = q0.copy(); q[free] = x
        return q
    fit = least_squares(lambda x: resid(embed(x)), q0[free], loss='linear', max_nfev=max_nfev,
                        bounds=(lo[free], hi[free]))
    p = from_q(embed(fit.x))
    params = {**dict(zip(GEOM_KEYS, p[:N_GEOM])), **dict(zip(profile.param_names, p[N_GEOM:]))}
    # residuals at the solution (as reported in the original pipeline)
    fun = sw * (y - log_s2_model(p, lags, profile)) / noise_scale_dex
    fit.jac = None
    return dict(params=params, fit=SimpleNamespace(x=fit.x, success=fit.success, nfev=fit.nfev, fun=fun),
                n=int(fun.size), chi2=float(np.sum(fun ** 2)))
