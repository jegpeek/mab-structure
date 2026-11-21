# %% [markdown]
# # 03 · The common shape of the substructure
#
# **Result.**  Across the cloud, the substructure has a common triaxial shape,
# leaning prolate: **a1 : a2 : a3 ~ 13 : 2 : 1**.  Its orientation varies from
# region to region; its shape varies little.
#
# **How.**  Each of the 46 windows is fitted with an ellipsoidal power law
# (notebook 01 for the sample, `mabsf.fitting` for the fit, notebook 02 for
# the PSF correction).  This notebook turns the per-window fits into the
# numbers quoted in the paper.
#
# **Two descriptions of the anisotropy.**  For S2 = A rho^alpha, the structure
# function at a fixed separation r along principal axis i is A (r/a_i)^alpha.
# - The **variance ratios** f12 = (a2/a1)^alpha and f23 = (a3/a2)^alpha
#   compare how much the brightness varies along two axes *at the same
#   separation*, inside the fitted range.  They lie between 0 and 1.
# - The **axis ratios** a2/a1 and a3/a2 compare *how far* one must go along two
#   axes to see the same variation.  Along the long axis, that distance can be
#   far beyond the 0.48 ly window: an extrapolation.
#
# Converting f to an axis ratio raises it to the power 1/alpha ~ 2, which turns
# a modestly small f12 into a tiny a2/a1 with an enormous error.  Two windows
# are like this ("collapsed" fits, a ratio < 0.02).  The statistics are
# therefore done on f12 and f23, where every window has an ordinary error bar,
# and the results are converted back to axis ratios to describe the shape.

# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from mabsf import config as C, preprocess as P, fitting as F, shape as S, plotting
from mabsf.checks import check, targets

plotting.style()
RECOMPUTE = False       # True: refit all windows (`mabsf fit`, ~1-7 min on 8 cores)
cfg = C.Config()
wins = P.sample_windows(cfg)
if RECOMPUTE or not all(F.is_cached(r, c, cfg) for r, c in wins):
    F.fit_windows(wins, cfg, procs=8)
summ, k = S.summary(cfg)
k[['row', 'col', 'a2a1', 'a3a2', 'alpha', 'incl', 'f12', 'se_f12', 'f23', 'se_f23', 'collapsed']].round(3).head(8)

# %% [markdown]
# ## 1. Why the axis ratio "collapses" and the variance ratio does not

# %%
fig, axs = plt.subplots(1, 2, figsize=(plotting.W_IN, 2.8))
a = axs[0]
r = np.logspace(-2.6, 1.2, 300); al, a2 = 0.5, 0.2
S2 = lambda r, aa: (r / aa) ** al
a.axvspan(0.0096, cfg.rcut, color='0.92'); a.axvline(0.48, color='0.6', lw=0.6, ls=':')
a.text(0.031, 0.006, 'fit range', ha='center', fontsize=6); a.text(0.52, 0.006, 'window', fontsize=6)
a.loglog(r, S2(r, a2), color='C0', label='along $a_2$')
a.loglog(r, S2(r, a2 / 0.44 ** (1 / al)), color=plotting.C_WIN, label='along $a_1$, $f_{12}$ = 0.44 (typical)')
a.loglog(r, S2(r, a2 / 0.05 ** (1 / al)), '--', color=plotting.C_WIN, label='along $a_1$, $f_{12}$ = 0.05')
a.set_xlim(0.003, 15); a.set_ylim(5e-3, 10); a.set_xlabel('separation [ly]'); a.set_ylabel('$S_2$ (model)')
a.legend(fontsize=6, loc='upper left')
a.set_title('$f_{12}$: vertical gap; $a_2/a_1$: horizontal gap', fontsize=7)
b = axs[1]
o = np.argsort(k.f12.values); ks = k.iloc[o]; x = np.arange(len(ks)); u = ks.collapsed.values
b.errorbar(x, ks.f12, ks.se_f12, fmt='none', ecolor=plotting.C_WIN, elinewidth=0.6)
b.plot(x[~u], ks.f12[~u], 'o', ms=3, color=plotting.C_WIN, label='$f_{12}$')
b.plot(x[u], ks.f12[u], 'o', ms=4, mfc='w', mec=plotting.C_WIN, label='collapsed in $a_2/a_1$')
b2 = b.twinx(); b2.spines['right'].set_visible(True)
b2.errorbar(x + 0.3, np.log10(ks.a2a1), ks.se_l12, fmt='none', ecolor='C0', elinewidth=0.6)
b2.plot(x + 0.3, np.log10(ks.a2a1), 's', ms=2.5, color='C0', label='log $a_2/a_1$')
b.set_ylim(-0.3, 1.1); b2.set_ylim(-4.2, 0.6)
b.set_xlabel('window (sorted by $f_{12}$)'); b.set_ylabel('$f_{12}$', color=plotting.C_WIN)
b2.set_ylabel('log$_{10}$ $a_2/a_1$', color='C0')
b.legend(loc='upper left', fontsize=6)
fig.tight_layout()

# %% [markdown]
# The two collapsed windows sit at the low end of a continuous f12
# distribution, with ordinary errors (~0.1).  In log a2/a1 the same windows have
# errors of 1-3 dex.

# %% [markdown]
# ## 2. The common shape and its spread (claims C7, C12)
#
# The common value and intrinsic (window-to-window) scatter of f12 and f23 are
# fitted by maximum likelihood, x_i ~ N(mu, s_i^2 + sigma^2), with s_i the
# jackknife error of each window.  Errors on sigma come from a bootstrap over
# windows.

# %%
vr = summ['variance_ratio']
pd.DataFrame({
    'common value': [vr['f12'], vr['f23']],
    'intrinsic scatter': [vr['sig_f12'], vr['sig_f23']],
    'scatter 16-84%': [vr['sig_f12_68'], vr['sig_f23_68']],
    'scatter 95% upper': [vr['sig_f12_95'], vr['sig_f23_95']],
    'p (scatter = 0)': [vr['p0_f12'], vr['p0_f23']],
}, index=['f12 (long / middle)', 'f23 (middle / short)']).round(3)

# %% [markdown]
# In words: at separations below 0.1 ly, the brightness varies 1/f12 ~ 2.7 times
# less (in variance) along the long axis than along the middle one, and
# 1/f23 ~ 1.3 times less along the middle than along the short axis.
#
# **In axis-ratio language**, converting at the median slope alpha:

# %%
am = summ['median_alpha']
sp = {q: summ['spread_dex_' + q] for q in ('a2a1', 'a3a2')}
fac = lambda d: 10 ** d
print('median alpha = %.2f' % am)
print('common shape: a2/a1 = %.3f, a3/a2 = %.3f  ->  a1 : a2 : a3 = %.1f : %.1f : 1'
      % (summ['common_a2a1'], summ['common_a3a2'], 1 / (summ['common_a2a1'] * summ['common_a3a2']), 1 / summ['common_a3a2']))
print('spread in a2/a1: factor %.2f (95%%: < %.2f) at one alpha; %.2f (< %.2f) with each window\'s alpha'
      % tuple(fac(sp['a2a1'][x]) for x in ('single_alpha', 'single_alpha_95', 'window_alpha', 'window_alpha_95')))
print('spread in a3/a2: %.0f%% (95%%: < %.0f%%) at one alpha; %.0f%% (< %.0f%%) with each window\'s alpha'
      % tuple(100 * (fac(sp['a3a2'][x]) - 1) for x in ('single_alpha', 'single_alpha_95', 'window_alpha', 'window_alpha_95')))

# %% [markdown]
# The per-window-alpha version is an upper bracket, because part of the slope
# spread is measurement noise.  The paper quotes the range: **elongation a2/a1
# varies between regions by a factor of ~1.7-2.0 (< ~2.2 at 95%), flattening
# a3/a2 by ~25% (< ~35%)**.
#
# For reference, the same statistics computed directly in log axis ratios (the
# paper's first form) give a similar common shape, but their scatter estimate
# depends strongly on how the collapsed windows are handled (notebook 06):

# %%
pd.Series({q: summ[q] for q in ('C7_common_a2a1', 'C7_common_a3a2', 'C12_sigint_a2a1', 'C12_sigint_a3a2',
                                'C12_p_zero_a2a1', 'C12_p_zero_a3a2')}).round(4)

# %% [markdown]
# ## 3. Windows with an unbounded long axis (claim C13)
#
# For the two collapsed windows the variation along the long axis is too
# small to measure within the window.  Their 2-sigma upper limit on a2/a1 is
# (f12 + 2 se)^(1/alpha):

# %%
ul = S.upper_limits(k)
pd.DataFrame([dict(window=rc, region1=(rc == S.R1_TILE), f12=k.set_index(['row', 'col']).f12[rc],
                   se_f12=k.set_index(['row', 'col']).se_f12[rc], alpha=k.set_index(['row', 'col']).alpha[rc],
                   a2a1_upper_2sigma=v) for rc, v in ul.items()]).round(3)

# %% [markdown]
# Only Region 1 gives a useful limit (a2/a1 < 0.09); the other is consistent
# with the common shape.  (With the stationary estimator a third window,
# (1864, 892), also collapses; see section 7.)

# %% [markdown]
# ## 4. Triaxial, leaning prolate; not isotropic (claims C10, C11)

# %%
pd.Series(dict(
    windows_prolate_f12_lt_f23=summ['n_prolate_f'],
    windows_excluding_isotropy_2sigma=summ['n_exclude_isotropy_2sig'],
    n=len(k),
    prolateness_dex=summ['C11_prolate_dex'],
    prolateness_significance_sigma=summ['C11_prolate_sigma'],
)).round(3)

# %% [markdown]
# f12 < f23 is the same condition as a3/a2 > a2/a1 (for one alpha): the two
# short axes are closer to each other than the middle axis is to the long one.

# %% [markdown]
# ## 5. One shape, many orientations (claims C7c, C8)
#
# If all windows share one 3D shape, the anisotropy seen in the echo plane
# should depend on how the long axis is oriented: a long axis lying in the plane
# gives an elongated in-plane slice; one tilted toward W gives a round slice.

# %%
fig, axs = plt.subplots(1, 3, figsize=(plotting.W_IN, 2.4))
axs[0].plot(k.incl, k.b2b1_2d, 'o', ms=3, color=plotting.C_WIN)
axs[0].set_xlabel('inclination of $a_1$ from W [deg]'); axs[0].set_ylabel('in-plane $b_2/b_1$')
axs[0].set_title(r'$\rho$ = %.2f' % summ['C7_rho_incl_b2b1'], fontsize=7)
th = np.linspace(0, 90, 200)
axs[1].hist(k.incl, bins=np.arange(0, 91, 10), color=plotting.C_WIN, alpha=0.8, density=True, label='windows')
axs[1].plot(th, np.sin(np.radians(th)) * np.pi / 180, 'k--', lw=0.8, label='random orientation')
axs[1].set_xlabel('inclination [deg]'); axs[1].legend(fontsize=6)
axs[1].set_title('median %.0f deg; KS p = %.3f' % (summ['C8_median_incl'], summ['C8_ks_p']), fontsize=7)
axs[2].plot(k.incl, k.f23, 'o', ms=3, color='C0', label='$f_{23}$')
axs[2].plot(k.incl, k.f12, 'o', ms=3, color=plotting.C_WIN, label='$f_{12}$')
axs[2].set_xlabel('inclination [deg]'); axs[2].legend(fontsize=6)
axs[2].set_title(r'$\rho(f_{23}$, incl) = %.2f' % summ['rho_f23_incl'], fontsize=7)
fig.tight_layout()

# %% [markdown]
# * The in-plane axis ratio follows the inclination as one shape predicts.
# * Long axes are spread over a wide range of inclinations, but concentrated
#   nearer the echo plane than random orientations would be.
# * f23 correlates with inclination, so part of its small scatter is viewing
#   geometry; f12 does not.

# %% [markdown]
# ## 6. Checks against the original analysis
#
# The original analysis used the stationary S2 estimator, so the checks use
# `C.FIRST_SUBMISSION`, with both calibrations: 'dedup' (each synthetic field
# once) and 'published' (the first table, with duplicated fields; the paper's
# first numbers).  Section 7 shows what changes with the exact estimator.

# %%
fs = C.FIRST_SUBMISSION
if not all(F.is_cached(r, c, fs) for r, c in wins):
    F.fit_windows(wins, fs, procs=8)
summ_fs, k_fs = S.summary(fs)
for calib, key in (('dedup', 'population_dedup'), ('published', 'population_base')):
    s_, k_ = (summ_fs, k_fs) if calib == 'dedup' else S.summary(fs.replace(calib=calib), n_boot=0)
    t = targets()[key]
    for q in ('f12', 'sig_f12', 'f23', 'sig_f23'):
        check('%s %s' % (calib, q), s_['variance_ratio'][q], t['variance_ratio'][q], rtol=1e-5)
    for q in ('common_a2a1', 'common_a3a2', 'median_alpha', 'n_prolate_f', 'n_exclude_isotropy_2sig', 'rho_f23_incl'):
        check('%s %s' % (calib, q), s_[q], t[q], rtol=1e-6)
    for q in ('C7_common_a2a1', 'C7_common_a3a2', 'C12_sigint_a2a1', 'C12_sigint_a3a2', 'C11_prolate_sigma',
              'C11_frac_prolate', 'C10_n_consistent_isotropic', 'C13_n_collapsed', 'C7_rho_incl_b2b1',
              'C8_median_incl', 'C8_ks_p'):
        check('%s %s' % (calib, q), s_[q], t['claims'][q], rtol=1e-6, atol=1e-9)
    ul_ = S.upper_limits(k_)
    check('%s a2/a1 upper limits' % calib, [ul_[tuple(map(int, x.split('_')))] for x in sorted(t['upper_limits_a2a1'])],
          [t['upper_limits_a2a1'][x] for x in sorted(t['upper_limits_a2a1'])], rtol=1e-6)

# %% [markdown]
# ## 7. The S2 estimator: exact (default) vs stationary (first submission)
#
# S2 between epochs i and j at lag l is the mean of (f_i(x) - f_j(x + l))^2 over
# the pixel pairs that overlap at that lag:
# <f_i^2> + <f_j^2> - 2 <f_i(x) f_j(x + l)>.
# The **exact** estimator (`Config.stationary = False`, the default) averages
# the two squared terms over the same overlapping pixels as the cross term.
# The **stationary** estimator (used in the first submission) replaces them by
# one window-wide mean square per epoch, which is right only if the
# fluctuations have the same variance everywhere in the window.
#
# On synthetic fields, which are stationary by construction, both estimators
# recover the injected shape; the exact one is about twice as precise
# (`results/stationarity_synthetic.csv`).  On the data, the exact estimator
# gives a lower f12 (a more elongated common shape).  Since the two agree on
# stationary fields, the difference reflects real variations of the
# fluctuation amplitude inside the windows, which only the exact estimator
# handles correctly.  f23, alpha and the inclinations hardly change.

# %%
rows = {}
for name, (s_, k_) in (('exact (default)', (summ, k)), ('stationary (first submission)', (summ_fs, k_fs))):
    vr = s_['variance_ratio']
    rows[name] = {'f12': vr['f12'], 'sigma(f12), 95% limit': vr['sig_f12_95'], 'f23': vr['f23'],
                  'sigma(f23), 95% limit': vr['sig_f23_95'], 'common a2/a1': s_['common_a2a1'],
                  'common a3/a2': s_['common_a3a2'], 'median alpha': s_['median_alpha'],
                  'median inclination [deg]': s_['C8_median_incl'], 'prolate [sigma]': s_['C11_prolate_sigma'],
                  'windows prolate (f12 < f23)': s_['n_prolate_f'],
                  'unbounded long axis': int(k_.collapsed.sum()),
                  'median jackknife SE of f12': float(np.median(k_.se_f12))}
pd.DataFrame(rows).round(3)
