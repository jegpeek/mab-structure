"""The point-spread function in the echo frame, and how it is corrected.

**The PSF chain.**  A point on the sky is spread by
1. the telescope optics (in-flight WebbPSF F444W model, rotated with the
   telescope roll of each epoch),
2. the 63 mas detector pixel,
3. drizzling onto the 60 mas mosaic (box),
4. an extra Gaussian of sigma 32.5 mas (unmodelled jitter / drizzle kernel;
   set by matching the stacked stars of the with-stars mosaic), and
5. the linear interpolation onto the UV grid (tent).

Steps 1-4 live on the sky; the local Jacobian J = d(U, V)/d(col, row) maps
them into the UV plane, where the PSF has a lag FWHM of ~0.018 x 0.015 ly
(5-6 grid pixels) -- comparable to the smallest scales in the fit.

**Two corrections are used.**
* **Shape fits (axis ratios, slope, inclination).**  The PSF rounds small
  structures, which biases a2/a1, a3/a2, b2/b1 and alpha.  The bias is
  calibrated by injection: synthetic 3D power-law fields of known shape are
  sliced at the 7 epoch depths, blurred by the full chain (with the real roll
  angles) plus noise, and fitted exactly like the data.  The median
  (observed - true) as a function of the long-axis tilt is subtracted from each
  window (`correct`).
* **1D structure functions of Regions 1 and 2.**  S2 is quadratic in the image,
  so the PSF multiplies its power spectrum by T^2(k).  The noise is removed
  and the spectrum divided by max(T^2, 0.05), then transformed back
  (`sf2d_epochs`, `w_points`).  The floor caps the gain at 20x, so the result
  is the sky seen through an *effective* PSF of FWHM ~0.011 ly, not infinite
  resolution.
"""
from __future__ import annotations

import functools
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from astropy.io import fits
from matplotlib.path import Path
from scipy.fft import fft2, next_fast_len
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt, maximum_filter

from . import config as C
from . import data as D
from . import geometry as G
from . import preprocess as P
from . import sf

MOS_PX, DET_PX = 0.060, 0.063        # mosaic and detector pixel [arcsec]
KHALF = 24                          # lag kernels stored on (2 KHALF + 1)^2 pixels
NFFT = 512
K_NOISE = 120.0                     # cycles/ly; beyond this the signal transfer is < 1e-3
T2MIN = 0.05                        # deconvolution gain cap (1 / T2MIN = 20x)
APOD = 20                           # cosine edge taper [px] of the deconvolution
T2_TAIL = 1e-3


# =========================================================================== inputs
def psf_inputs_path():
    return D.products('psf_f444w_uv.npz')


def star_stack(i2d, R=20, snr_min=150):
    """Median of isolated, unsaturated stars (S/N > 150) from the with-stars mosaic.

    Background from an r = 15-20 px annulus; each star Fourier-shifted to the
    centre and normalised to its r <= 12 px flux.
    """
    with fits.open(i2d) as h:
        sci = h['SCI'].data.astype(float); err = h['ERR'].data.astype(float)
    S = np.nan_to_num(sci); E = np.where(np.isfinite(err) & (err > 0), err, np.inf)
    cand = np.argwhere((S == maximum_filter(S, size=41)) & (S / E > snr_min) & (S > 0))
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]; rr = np.hypot(yy, xx)
    ky = np.fft.fftfreq(2 * R + 1)[:, None]; kx = np.fft.fftfreq(2 * R + 1)[None, :]
    out = []
    for y, x in cand:
        if min(y, x) < R + 2 or y > S.shape[0] - R - 3 or x > S.shape[1] - R - 3:
            continue
        st = sci[y - R:y + R + 1, x - R:x + R + 1].astype(float)
        if not np.isfinite(st).all():
            continue                                         # saturated core / gap
        st = st - np.median(st[(rr >= 15) & (rr <= 20)]); pk = st[R, R]
        if np.any((maximum_filter(st, 5) == st) & (rr > 4) & (st > 0.03 * pk)):
            continue                                         # not isolated
        if st[(rr > 0.5) & (rr < 1.5)].mean() / pk > 0.8:
            continue                                         # flat-topped
        c3 = st[R - 1:R + 2, R - 1:R + 2]
        dx = 0.5 * (c3[1, 0] - c3[1, 2]) / (c3[1, 0] - 2 * c3[1, 1] + c3[1, 2])
        dy = 0.5 * (c3[0, 1] - c3[2, 1]) / (c3[0, 1] - 2 * c3[1, 1] + c3[2, 1])
        sh = np.real(np.fft.ifft2(np.fft.fft2(st) * np.exp(2j * np.pi * (ky * dy + kx * dx))))
        out.append(sh / sh[rr <= 12].sum())
    return np.median(out, 0), len(out)


def cloud_jacobian(stride=4):
    """Median sky -> UV Jacobian over the cloud polygon (for the PSF figure)."""
    Pp = np.load(D.pos3d_path('2024'), mmap_mode='r')
    X, Y, Z = (np.asarray(Pp[::stride, ::stride, i], float) for i in range(3))
    U, V, _ = G.project(X, Y, Z)
    ok = np.isfinite(U) & np.isfinite(V)
    ins = np.zeros_like(ok); ins[ok] = Path(C.cloud_polygon_uv()).contains_points(np.column_stack([U[ok], V[ok]]))
    g = [np.gradient(A, axis=ax) / stride for A in (U, V) for ax in (1, 0)]
    sel = ins & np.isfinite(sum(g))
    return np.median(np.array([[g[0][sel], g[1][sel]], [g[2][sel], g[3][sel]]]), axis=-1), int(sel.sum())


def build_psf_inputs():
    """Write data/products/psf_f444w_uv.npz: WebbPSF model, star stack, cloud Jacobian."""
    with fits.open(D.inputs(*D.PSF_FITS)) as h:
        psf_os = h['OVERDIST'].data.astype(float); pix_os = h['OVERDIST'].header['PIXELSCL']
    stack, n = star_stack(D.inputs(*D.STARS_I2D))
    J, _ = cloud_jacobian()
    np.savez_compressed(psf_inputs_path(), psf_os=psf_os / psf_os.sum(), psf_os_pix_arcsec=pix_os,
                        star_stack=stack, n_stars=n, stack_pix_arcsec=0.060, J=J)
    return psf_inputs_path()


@functools.lru_cache(None)
def psf_inputs():
    if not os.path.exists(psf_inputs_path()):
        build_psf_inputs()
    z = np.load(psf_inputs_path())
    return {k: z[k] for k in z.files}


# =========================================================================== transfer functions
def optics_otf(psf_os, pix_arcsec, n=2048):
    """Interpolator for the complex OTF of an oversampled PSF (q in cycles/arcsec)."""
    pad = np.zeros((n, n)); m = psf_os.shape[0]; o = n // 2 - m // 2
    pad[o:o + m, o:o + m] = psf_os / psf_os.sum()
    F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pad)))
    f = np.fft.fftshift(np.fft.fftfreq(n, d=pix_arcsec))
    re = RegularGridInterpolator((f, f), F.real, bounds_error=False, fill_value=0)
    im = RegularGridInterpolator((f, f), F.imag, bounds_error=False, fill_value=0)
    return lambda qx, qy: re(np.stack([qy, qx], -1)) + 1j * im(np.stack([qy, qx], -1))


@functools.lru_cache(None)
def otf():
    Z = psf_inputs()
    return optics_otf(Z['psf_os'], float(Z['psf_os_pix_arcsec']))


@functools.lru_cache(None)
def symmetric_sky_mtf(extra_sigma=0.0325, nq=600):
    """Azimuthally averaged |OTF| x detector pixel x extra Gaussian, vs |q| [cycles/arcsec].

    The 7 epochs have different roll angles, so the epoch-averaged PSF is close
    to symmetric on the sky; the deconvolution uses this symmetrised form.
    """
    o = otf()
    qg = np.linspace(0, 12, nq)
    m = np.zeros(nq)
    phis = np.linspace(0, np.pi, 90, endpoint=False)
    for ph in phis:
        qx, qy = qg * np.cos(ph), qg * np.sin(ph)
        m += np.abs(o(qx, qy)) * np.abs(np.sinc(qx * DET_PX) * np.sinc(qy * DET_PX))
    m /= len(phis)
    m *= np.exp(-2 * np.pi ** 2 * extra_sigma ** 2 * qg ** 2)
    return qg, m / m[0]


def transfer(KU, KV, J, qg, mtf):
    """Signal power transfer T^2 and noise power shape N^2 at UV frequencies (KU, KV) [cycles/ly].

    The UV frequency maps to sky frequency through J; the optics enter T^2
    only, while detector, drizzle and tent shape the noise too.
    """
    Qc = J[0, 0] * KU + J[1, 0] * KV          # cycles / mosaic px
    Qr = J[0, 1] * KU + J[1, 1] * KV
    qx, qy = Qc / MOS_PX, Qr / MOS_PX
    sym = np.interp(np.hypot(qx, qy), qg, mtf, right=0.0)
    driz = np.abs(np.sinc(Qc) * np.sinc(Qr))
    tent = np.sinc(Qc) ** 2 * np.sinc(Qr) ** 2
    det = np.abs(np.sinc(qx * DET_PX) * np.sinc(qy * DET_PX))
    return (sym * driz * tent) ** 2, (det * driz * tent) ** 2


def lag_kernel(T2):
    """Inverse FFT of a power transfer (FFT order) on the central (2 KHALF + 1)^2 lags."""
    g = np.real(np.fft.fftshift(np.fft.ifft2(T2)))
    c = T2.shape[0] // 2
    return g[c - KHALF:c + KHALF + 1, c - KHALF:c + KHALF + 1]


def fwhm_axes(g):
    """Major/minor FWHM [ly] and PA [deg from +U to +V] of a lag kernel (2nd moments above half max)."""
    yy, xx = np.mgrid[-KHALF:KHALF + 1, -KHALF:KHALF + 1] * C.PIX_LY
    w = np.where(g > 0.5 * g.max(), g, 0)
    Ixx, Iyy, Ixy = [(w * a).sum() / w.sum() for a in (xx * xx, yy * yy, xx * yy)]
    ev, evec = np.linalg.eigh([[Ixx, Ixy], [Ixy, Iyy]])
    s = np.sqrt(ev / 0.2787)          # truncated-Gaussian 2nd moment -> sigma
    return 2.3548 * s[1], 2.3548 * s[0], float(np.degrees(np.arctan2(evec[1, 1], evec[0, 1])))


def lag_psf(J, extra_sigma=0.0325, n=NFFT):
    """The PSF's lag kernel (autocorrelation of the UV-plane PSF), normalised to sum 1."""
    k1 = np.fft.fftfreq(n, d=C.PIX_LY)
    KU, KV = np.meshgrid(k1, k1)
    g = lag_kernel(transfer(KU, KV, J, *symmetric_sky_mtf(extra_sigma))[0])
    return g / g.sum()


# =========================================================================== Fourier deconvolution
def transfer_grid(J, ny, nx, extra_sigma=0.0325):
    KU, KV = np.meshgrid(np.fft.fftfreq(nx, d=C.PIX_LY), np.fft.fftfreq(ny, d=C.PIX_LY))
    return transfer(KU, KV, J, *symmetric_sky_mtf(extra_sigma))


def noise_amplitudes(F, N2, T2):
    """Per-epoch noise amplitude n_e (noise power = n_e N2(k)).

    From the MEAN over the high-k tail (T^2 < 1e-3) of |F_e|^2 - Re F_e F_nbr*:
    the cross-spectrum with a neighbouring epoch carries no noise, and
    subtracting it removes the edge leakage the two share.  (Mean, not median:
    per-pixel noise power is exponentially distributed.)
    """
    ne = F.shape[0]
    tail = (T2 < T2_TAIL) & (N2 > 0.02)
    amp = np.empty(ne)
    for e in range(ne):
        nb = e + 1 if e + 1 < ne else e - 1
        d = (np.abs(F[e]) ** 2 - np.real(F[e] * np.conj(F[nb])))[tail]
        amp[e] = max(float(np.mean(d / N2[tail])), 0.0)
    return amp


def taper(ny, nx, apod=APOD):
    """Separable cosine edge ramp of width `apod` px."""
    def ramp(n):
        t = np.ones(n)
        if apod > 0:
            x = 0.5 * (1 - np.cos(np.pi * (np.arange(apod) + 0.5) / apod))
            t[:apod] = x; t[-apod:] = x[::-1]
        return t
    return np.outer(ramp(ny), ramp(nx))


def _ac_norm(w):
    ny, nx = w.shape
    ones = np.zeros((2 * ny, 2 * nx)); ones[:ny, :nx] = w
    ov = np.fft.fftshift(np.real(np.fft.ifft2(np.abs(np.fft.fft2(ones)) ** 2)))
    ov[ov < 1e-5 * ov.max()] = np.inf
    return ov


def sf2d_epochs(imgs, J, t2min=T2MIN, correct=True, apod=APOD, extra_sigma=0.0325):
    """Epoch-averaged in-plane 2D structure function, optionally noise-removed and deconvolved.

    S2(l) = 2 [C(0) - C(l)] from the tapered autocorrelation C of each epoch,
    normalised by the taper's own autocorrelation (unbiased at every lag).
    Returns an (ny, nx) array on lags centred at (ny // 2, nx // 2).
    """
    imgs = np.nan_to_num(np.asarray(imgs, float))
    ne, ny, nx = imgs.shape
    w = taper(ny, nx, apod)
    Pd = np.zeros((ne, 2 * ny, 2 * nx))
    for e in range(ne):
        mu = np.sum(w * imgs[e]) / w.sum()
        Pd[e, :ny, :nx] = (imgs[e] - mu) * w
    F = np.fft.fft2(Pd)
    ov = _ac_norm(w)
    sy, sx = ny - ny // 2, nx - nx // 2
    acc = np.zeros((ny, nx))
    if correct:
        T2, N2 = transfer_grid(J, 2 * ny, 2 * nx, extra_sigma)
        amp = noise_amplitudes(F, N2, T2)
        gain = 1.0 / np.maximum(T2, t2min)
    for e in range(ne):
        psd = np.abs(F[e]) ** 2
        if correct:
            psd = (psd - amp[e] * N2) * gain
        ac = np.fft.fftshift(np.real(np.fft.ifft2(psd))) / ov
        ac = ac[sy:sy + ny, sx:sx + nx]
        acc += 2.0 * (ac[ny // 2, nx // 2] - ac)
    return acc / ne


def w_points(imgs, w_vals, J, t2min=T2MIN, correct=True, sep_scale=1.0, apod=APOD, extra_sigma=0.0325):
    """Cross-epoch S2 = <(a - b)^2> for every epoch pair, with its depth separation |W_b - W_a|.

    With correct=True the noise (independent between epochs, so present only in
    the auto terms) is removed and the PSF deconvolved, as for sf2d_epochs.
    sep_scale stretches the separation for a co-moving window (see regions).
    """
    imgs = np.nan_to_num(np.asarray(imgs, float))
    ne, ny, nx = imgs.shape
    w = taper(ny, nx, apod)
    Pd = np.zeros((ne, 2 * ny, 2 * nx)); Pd[:, :ny, :nx] = imgs * w
    F = np.fft.fft2(Pd)
    norm = Pd.shape[1] * Pd.shape[2] * np.sum(w ** 2)
    if correct:
        T2, N2 = transfer_grid(J, 2 * ny, 2 * nx, extra_sigma)
        amp = noise_amplitudes(F, N2, T2)
        gain = 1.0 / np.maximum(T2, t2min)
    sep, s2 = [], []
    for a in range(ne):
        for b in range(a + 1, ne):
            d = np.abs(F[a] - F[b]) ** 2
            if correct:
                d = (d - (amp[a] + amp[b]) * N2) * gain
            sep.append(abs(w_vals[b] - w_vals[a]) * sep_scale)
            s2.append(float(d.sum() / norm))
    return np.array(sep), np.array(s2)


def effective_fwhm(J, t2min=T2MIN, n=320, extra_sigma=0.0325):
    """Lag FWHM [ly] (major axis) of the real PSF and of the effective PSF after deconvolution."""
    T2, _ = transfer_grid(J, n, n, extra_sigma)
    out = []
    for t in (T2, np.minimum(1.0, T2 / t2min)):
        g = np.fft.fftshift(np.real(np.fft.ifft2(t)))
        c = n // 2
        out.append(fwhm_axes(g[c - KHALF:c + KHALF + 1, c - KHALF:c + KHALF + 1])[0])
    return tuple(out)


# =========================================================================== injection calibration
#: The two injected shapes: semi-axes (a1, a2, a3) [ly].
SHAPES = {'a2a1=0.28,a3a2=0.50': (0.3, 0.084, 0.042), 'a2a1=0.50,a3a2=0.40': (0.3, 0.15, 0.06)}
TILTS = (0, 15, 30, 45, 60, 75, 90)
#: Seeds per tilt: 3 everywhere, 9 at tilts >= 60 deg, where most windows lie.
SEEDS = {t: ((11, 12, 13, 21, 22, 23, 24, 25, 26) if t >= 60 else (11, 12, 13)) for t in TILTS}
FIELD_ALPHA = 0.8
NB, NW = 320, 256            # synthetic box [px] in UV and W (periodic, much larger than a window)
#: Window that supplies the lag geometry, depths, Jacobian and noise level of the injections.
TEMPLATE = (1466, 1217, 200)
CALIB_RMIN = 0.0096          # 3 px
CALIB_INNER_UV = 100
#: Calibration variants (sensitivity analysis): extra Gaussian [arcsec], template size, weighting, rcut.
VARIANTS = {
    'base': (0.0325, 200, '1/r', 0.10),
    'sig225': (0.0225, 200, '1/r', 0.10), 'sig425': (0.0425, 200, '1/r', 0.10),
    'geom150': (0.0325, 150, '1/r', 0.10),
    'rc050': (0.0325, 200, '1/r', 0.05), 'rc075': (0.0325, 200, '1/r', 0.075), 'rc150': (0.0325, 200, '1/r', 0.15),
    'wnone': (0.0325, 200, None, 0.10), 'w1r2': (0.0325, 200, '1/r2', 0.10),
}
CORRECTED = ('a2a1', 'a3a2', 'a3a1', 'b2b1', 'alpha')
CLIP = 0.5                   # a correction may at most halve an observed ratio


def power_spectrum_2d(img, foot, apod_px=10, pad=2):
    """Tapered, zero-padded periodogram of a masked image (sum P dk^2 = variance)."""
    valid = foot & np.isfinite(img)
    d = distance_transform_edt(foot)
    wgt = (0.5 - 0.5 * np.cos(np.pi * np.clip(d / apod_px, 0, 1))) * valid
    f = np.where(valid, img, 0.0)
    f = f - np.sum(f * wgt) / np.sum(wgt)
    ny, nx = f.shape
    sh = (next_fast_len(pad * ny), next_fast_len(pad * nx))
    Fk = fft2(f * wgt, s=sh)
    Pk = (Fk.real ** 2 + Fk.imag ** 2) * C.PIX_LY ** 2 / np.sum(wgt ** 2)
    KU, KV = np.meshgrid(np.fft.fftfreq(sh[1], d=C.PIX_LY), np.fft.fftfreq(sh[0], d=C.PIX_LY))
    return Pk, KU, KV


def ring_average(Pk, KU, KV, edges):
    k = np.hypot(KU, KV).ravel(); p = Pk.ravel()
    idx = np.digitize(k, edges) - 1
    s = (idx >= 0) & (idx < len(edges) - 1)
    num = np.bincount(idx[s], weights=p[s], minlength=len(edges) - 1)
    cnt = np.bincount(idx[s], minlength=len(edges) - 1)
    ksum = np.bincount(idx[s], weights=k[s], minlength=len(edges) - 1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return ksum / cnt, num / cnt


def window_noise_sigma(data, J, extra_sigma=0.0325):
    """Per-epoch rms of the resampling/detector noise in a window.

    The noise power has the known spectral shape N^2(k) of pixel x drizzle x
    tent; its amplitude is the median of P(k) / N^2(k) at k > 120 cycles/ly,
    where the PSF has removed the signal.  The predicted adjacent-pixel
    difference of that noise is compared with the measured one, and the noise
    is scaled down if it over-predicts (an upper bound).
    """
    qg, mtf = symmetric_sky_mtf(extra_sigma)
    k1 = np.fft.fftfreq(NFFT, d=C.PIX_LY)
    KU, KV = np.meshgrid(k1, k1)
    _, N2 = transfer(KU, KV, J, qg, mtf)
    Cn_shape = lag_kernel(N2) * NFFT ** 2 * (1.0 / (NFFT * C.PIX_LY)) ** 2
    s = data['flux_epochs'].shape[1]
    kedges = np.geomspace(1.0, np.sqrt(2) * 0.5 / C.PIX_LY, 40)
    _, KUw, KVw = power_spectrum_2d(np.zeros((s, s)), np.ones((s, s), bool))
    N2r = ring_average(transfer(KUw, KVw, J, qg, mtf)[1], KUw, KVw, kedges)[1]
    f = np.array(data['flux_epochs'], float)
    N0 = []
    for img in f:
        foot = np.isfinite(img)
        if foot.mean() < 0.2:
            N0.append(np.nan); continue
        img = np.where(foot, img - np.nanmean(img), np.nan)
        Pk, KUe, KVe = power_spectrum_2d(np.nan_to_num(img), foot)
        k, Pr = ring_average(Pk, KUe, KVe, kedges)
        hi = (k > K_NOISE) & np.isfinite(Pr) & (N2r > 0.02)
        N0.append(float(np.median(Pr[hi] / N2r[hi])))
    Cn = np.array([n0 * Cn_shape for n0 in N0])
    dx = np.nanmedian([np.nanstd(e[:, 1:] - e[:, :-1]) / np.sqrt(2) for e in f if np.isfinite(e).any()])
    pred = np.sqrt(max(np.nanmedian(Cn[:, KHALF, KHALF]) - np.nanmedian(Cn[:, KHALF, KHALF + 1]), 0))
    scale = min(1.0, dx / pred)
    return np.sqrt(Cn[:, KHALF, KHALF]) * scale


@functools.lru_cache(None)
def calibration_template():
    """Template windows (200 and 150 px), their Jacobian and noise rms for the injections."""
    r, c, s = TEMPLATE
    d200 = P.read_window(r, c, s, **C.READ_KW)
    d150 = P.read_window(r + 25, c + 25, 150, **C.READ_KW)
    J, _ = G.pixel_to_uv_jacobian(r, c, s)
    return {200: d200, 150: d150}, J, window_noise_sigma(d200, J)


def make_field(a_axes, angles_deg, alpha, seed, w_vals):
    """3D Gaussian random field with S2 = A |L^-1 d|^alpha, sliced at depths w_vals.

    Built in Fourier space on an NB x NB x NW periodic box with power
    |L^T k|^-(3 + alpha); returns (n_epochs, NB, NB), unit variance.
    """
    rng = np.random.default_rng(seed)
    g = sf.params_from_axes(*a_axes, *np.radians(angles_deg))
    Lm = np.array([[g['s11'], g['l12'], g['l13']], [0, g['s22'], g['l23']], [0, 0, g['s33']]])
    ku = np.fft.fftfreq(NB, d=C.PIX_LY); kw = np.fft.fftfreq(NW, d=C.PIX_LY)
    KU, KV, KW = np.meshgrid(ku, ku, kw, indexing='xy')
    k = np.stack([KU, KV, KW], -1) @ Lm
    kk = np.sqrt(np.sum(k ** 2, -1)); kk[0, 0, 0] = np.inf
    amp = kk ** (-(3 + alpha) / 2)
    f = np.real(np.fft.ifftn(amp * np.fft.fftn(rng.standard_normal((NB, NB, NW)))))
    f /= f.std()
    iw = np.round((np.asarray(w_vals) - np.min(w_vals)) / C.PIX_LY).astype(int) + 8
    return np.moveaxis(f[:, :, iw], -1, 0)


def blur(slices, J, seed_noise, sig_noise, extra_sigma=0.0325):
    """Blur each slice with the full (non-symmetrised) PSF chain at its epoch's roll; add noise.

    The noise is white noise shaped by detector x drizzle x tent, scaled to the
    template window's per-epoch noise rms.
    """
    o = otf()
    ku = np.fft.fftfreq(NB, d=C.PIX_LY); KU, KV = np.meshgrid(ku, ku)
    Qc = J[0, 0] * KU + J[1, 0] * KV; Qr = J[0, 1] * KU + J[1, 1] * KV
    qx, qy = Qc / MOS_PX, Qr / MOS_PX
    driz = np.sinc(Qc) * np.sinc(Qr); tent = np.sinc(Qc) ** 2 * np.sinc(Qr) ** 2
    rng = np.random.default_rng(seed_noise)
    out = np.empty_like(slices)
    for e, img in enumerate(slices):
        th = np.radians(C.ROLLS_DEG[e] - C.ROLLS_DEG[0])
        rx, ry = np.cos(th) * qx - np.sin(th) * qy, np.sin(th) * qx + np.cos(th) * qy
        det = np.sinc(rx * DET_PX) * np.sinc(ry * DET_PX)
        T = o(rx, ry) * det * np.exp(-2 * np.pi ** 2 * extra_sigma ** 2 * (qx ** 2 + qy ** 2)) * driz * tent
        im = np.real(np.fft.ifft2(np.fft.fft2(img) * T))
        nz = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal(img.shape)) * det * driz * tent))
        out[e] = im + nz * (sig_noise[e] / nz.std())
    return out


def fit_injected(img, data, weighting, rcut):
    """Fit synthetic slices exactly like a data window (S2, band r < rcut, power law, A frozen)."""
    n = data['flux_epochs'].shape[1]; c = (NB - n) // 2
    d = dict(data); d['flux_epochs'] = img[:, c:c + n, c:c + n].astype(np.float32)
    s2 = sf.compute_s2(d, **C.S2_KW)
    b = sf.band(s2, sf.lag_radius(s2), CALIB_RMIN, rcut)
    p = sf.fit_s2(b, inner_uv=CALIB_INNER_UV, stride=2, weighting=weighting, freeze=('A',),
                  min_n_fraction=0.1)['params']
    s = sf.scalars(p)
    return {k: s[k] for k in ('a2a1', 'a3a2', 'a3a1', 'b2b1', 'alpha', 'incl')}


def calibrate_field(spec, variants=('base',)):
    """One synthetic field: clean and observed fits under each calibration variant."""
    shape, tilt, seed = spec
    geo, J, sig_noise = calibration_template()
    data = geo[200]
    w_vals = np.asarray(data['W_values'], float)
    real = np.asarray(data['flux_epochs'], float)
    sig_real = np.sqrt(np.nanmedian([np.nanvar(e) for e in real if np.isfinite(e).any()]))
    sl = make_field(SHAPES[shape], (tilt, 30 + seed, 20 + 7 * seed), FIELD_ALPHA, seed, w_vals) * sig_real
    sigmas = {VARIANTS[v][0] for v in variants}
    blurred = {s: blur(sl, J, seed + 1000, sig_noise, s) for s in sigmas}
    rows = []
    for name in variants:
        sig, size, wgt, rc = VARIANTS[name]
        for which, img in (('clean', sl), ('observed', blurred[sig])):
            try:
                r = fit_injected(img, geo[size], wgt, rc)
            except Exception:   # noqa: BLE001
                continue
            rows.append(dict(variant=name, shape=shape, tilt=tilt, seed=seed, fit=which, **r))
    return rows


def calibration_fields():
    return [(s, t, k) for s in SHAPES for t in TILTS for k in SEEDS[t]]


def calibration_path():
    return D.results('psf_calibration.csv')


def _cal_job(args):
    spec, variants = args
    return calibrate_field(spec, variants)


def run_calibration(variants='all', procs=6, verbose=True):
    """Injection calibration over all fields; writes results/psf_calibration.csv (~1 h for 'all')."""
    variants = tuple(VARIANTS) if variants == 'all' else ((variants,) if isinstance(variants, str) else variants)
    specs = calibration_fields()
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ.setdefault(v, '1')        # one thread per worker process
    rows = []
    with Pool(procs) as p:
        for k, r in enumerate(p.imap_unordered(_cal_job, [(s, variants) for s in specs]), 1):
            rows += r
            if verbose and k % 10 == 0:
                print('  %d/%d fields' % (k, len(specs)), flush=True)
    R = pd.DataFrame(rows).sort_values(['variant', 'shape', 'tilt', 'seed', 'fit']).reset_index(drop=True)
    R.to_csv(calibration_path(), index=False)
    return R


def calibration_table(calib='dedup', variant='base'):
    """Per-field clean/observed fits for one calibration.

    calib='dedup'      the injection over 78 distinct fields (this package's default);
    calib='published'  the table used for the paper's first numbers, in which
                       several fields were included more than once.
    """
    if calib == 'published':
        R = pd.read_csv(D.results('psf_calibration_published.csv'))
        return R[R.band == 5].drop(columns='band')
    R = pd.read_csv(calibration_path())
    return R[R.variant == variant].drop(columns='variant')


def bias_by_tilt(table, model='add'):
    """Median bias (observed - clean) and its standard error at each injected tilt.

    model 'add': q_obs - q_true (default); 'log': in log10; 'quad': q_obs^2 - q_true^2.
    alpha is always additive.
    """
    idx = ['shape', 'tilt', 'seed']
    a = table[table.fit == 'clean'].set_index(idx)[list(CORRECTED)]
    b = table[table.fit == 'observed'].set_index(idx)[list(CORRECTED)]
    d = {'add': lambda: b - a, 'log': lambda: np.log10(b) - np.log10(a), 'quad': lambda: b ** 2 - a ** 2}[model]()
    d['alpha'] = (b - a)['alpha']
    d = d.dropna().reset_index()
    g = d.groupby('tilt')[list(CORRECTED)]
    med = g.median()
    sem = g.agg(lambda x: 1.2533 * 1.4826 * np.median(np.abs(x - np.median(x))) / np.sqrt(len(x)))
    return med, sem


def correct(df, calib='dedup', variant='base', degen=0.02, model='add'):
    """Subtract the PSF bias, interpolated at each window's inclination.

    Collapsed fits (a2/a1 or a3/a2 < degen) are passed through unchanged, so
    they stay recognisable as collapsed.  A correction may at most halve an
    observed ratio (the calibration covers ratios 0.26-0.50; smaller ratios
    would be extrapolation).  The inclination bias (< 2 deg) is not corrected.
    """
    med, _ = bias_by_tilt(calibration_table(calib, variant), model)
    tilts = med.index.values.astype(float)
    out = df.copy()
    inc = out['incl'].values.astype(float)
    m = np.isfinite(inc) & ~((out.a2a1 < degen) | (out.a3a2 < degen)).values
    for k in CORRECTED:
        if k not in out:
            continue
        bias = np.interp(inc[m], tilts, med[k].values)
        q = out[k].values.astype(float); v = q[m]
        cc = v - bias
        if k != 'alpha':
            cc = np.where(cc < CLIP * v, CLIP * v, cc)
        q[m] = cc; out[k] = q
    return out
