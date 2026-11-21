# %% [markdown]
# # 06 · Sensitivity of the results to the analysis choices
#
# **Question.**  Every number in notebooks 03-05 rests on choices: which
# epochs, how the flux is normalised, which star/galaxy infill, the PSF model,
# how the cloud region and the windows are drawn, the fit range and weighting,
# the jackknife, how collapsed fits are treated, and the S2 estimator.  Do the
# claims depend on them?
#
# **How.**  One choice at a time is changed from the paper analysis
# (`Config()`), everything downstream is redone with it -- the sample, the
# per-window fits, the matching PSF injection calibration, the population
# statistics and the region measurements -- and every claim is re-evaluated.
# The choices are listed in `mabsf.sensitivity.CHOICES`; the claims and their
# tests in `mabsf.sensitivity.RULES` (claim numbers as in
# `docs/claims_inventory.md`).
#
# What each claim can depend on:
# * C1-C4 (Regions 1 and 2) use only the deconvolved region cubes, so only the
#   PSF width (D5) and gain cap (D6) change them;
# * C5 (Region 1 untrackable, Region 2 trackable) uses only the structure maps,
#   so none of these choices changes it;
# * C6 depends on the sample, the fitted inclinations and the deconvolution;
# * C7-C13 depend on everything else.

# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mabsf import config as C, data as D, fitting as F, sensitivity as SE, shape as S, plotting
from mabsf.checks import check, targets

plotting.style()
RECOMPUTE = False     # True: refit every choice (~1 h on 8 cores) and re-evaluate the claims (~10 min)
path = D.results('sensitivity_claims.csv')
if RECOMPUTE:
    SE.fit_choices(procs=8)
if RECOMPUTE or not os.path.exists(path):
    SE.run()
T = pd.read_csv(path)
pd.DataFrame(SE.CHOICES, columns=['group', 'choice', 'change from Config()']).set_index('group')

# %% [markdown]
# ## 1. All claims under all choices
#
# Each row is one choice, each column one claim.  **Green / red**: the claim's
# test (below) passes / fails.  **Blue**: a value without a pass/fail test,
# shaded by its change from the paper analysis (top row); full shade at the
# tolerance listed in `mabsf.sensitivity.MATRIX` (0.06 in f, 0.05-0.06 in an
# axis ratio, 8 deg, 2 windows).

# %%
fig, ax = plt.subplots(figsize=(plotting.W_IN, 0.155 * len(T) + 1.3))
SE.plot_matrix(T, ax)
fig.tight_layout()
fig.savefig(D.figures('sensitivity_matrix.pdf'), bbox_inches='tight')
fig.savefig(D.figures('sensitivity_matrix.png'), dpi=250, bbox_inches='tight')

# %%
pd.DataFrame([dict(claim=s, test=d) for _, s, d, _ in SE.RULES]).set_index('claim')

# %% [markdown]
# ## 2. How far each number moves

# %%
G = SE.grade(T)
rows = []
for col, short, desc, rule in SE.RULES:
    v = T[col]
    r = dict(claim=short, quantity=desc, paper=v.iloc[0], min=v.min(), max=v.max(),
             choice_min=T.choice[v.idxmin()], choice_max=T.choice[v.idxmax()])
    if col in G:
        r['fails'] = ', '.join(T.choice[~G[col].astype(bool)]) or '-'
    rows.append(r)
R = pd.DataFrame(rows).set_index('claim')
R.round(3)

# %% [markdown]
# **Reading the matrix.**
# * Every pass/fail test holds under every choice (all green).
# * C1-C6 are insensitive: only the deconvolution settings change C1-C4, and
#   none enough to change the conclusion.
# * **f12 depends on the fit range (D19, 0.26-0.43) and the lag weighting (D20),
#   but the common axis ratio a2/a1 does not (0.13-0.14 across D19).**  Both
#   choices change the slope alpha (0.41-0.68 across D19), and f12 = (a2/a1)^alpha
#   follows alpha; the shape itself is stable.  This is why the shape is stated
#   as axis ratios, and f12 and f23 are used for the statistics.
# * a3/a2 moves most with the window size (D14, 0.51-0.63) and the fit range
#   (0.53-0.63): the estimator is biased low as the fitted scales grow (see the
#   window-size test), so the flattening is the less certain of the two ratios.
# * The number of unbounded windows (C13) ranges from 0 to 6: it is a property
#   of the fit threshold, not a robust count.

# %% [markdown]
# ## 3. The S2 estimator (D24)
#
# The last row is the analysis as first submitted: the stationary S2
# estimator (notebook 03, section 7).  It gives the least elongated common
# shape of all the choices (a2/a1 = 0.19; every other choice gives 0.105-0.165),
# because the fluctuation amplitude varies within the windows and only the
# exact estimator handles that.  On stationary synthetic fields the two
# estimators agree and the exact one is twice as precise; it is therefore the
# default.

# %%
d24 = T[T.group.isin(['base', 'D24'])].set_index('choice')
d24[['n', 'n_collapsed', 'f12', 'sig_f12_95', 'f23', 'sig_f23_95', 'common_a2a1', 'common_a3a2',
     'median_alpha', 'median_incl', 'prolate_sigma', 'rho_incl_b2b1', 'C6_rho_incl']].T.round(3)

# %% [markdown]
# ## 4. Checks
#
# * The first row reproduces notebook 03 (same configuration, same code).
# * The D24 row reproduces the original analysis (values frozen in
#   `validation/`, 'dedup' calibration): the estimator is the only difference.
# * The C1-C4 values of the paper row reproduce the original regions analysis.

# %%
summ, k = S.summary(C.Config(), n_boot=300)
b = T.iloc[0]
for q in ('f12', 'f23', 'sig_f12_95', 'sig_f23_95'):
    check('paper row %s = notebook 03' % q, b[q], summ['variance_ratio'][q], rtol=1e-9)
check('paper row common shape = notebook 03', [b.common_a2a1, b.common_a3a2],
      [summ['common_a2a1'], summ['common_a3a2']], rtol=1e-9)
t = targets()['population_dedup']; fs = T[T.group == 'D24'].iloc[0]
for q in ('f12', 'f23', 'sig_f12', 'sig_f23'):
    check('D24 row %s = original analysis' % q, fs[q], t['variance_ratio'][q], rtol=1e-5)
check('D24 row common shape = original analysis', [fs.common_a2a1, fs.common_a3a2],
      [t['common_a2a1'], t['common_a3a2']], rtol=1e-6)
check('D24 row prolate sigma = original analysis', fs.prolate_sigma, t['claims']['C11_prolate_sigma'], rtol=1e-6)
check('D24 row median inclination = original analysis', fs.median_incl, t['claims']['C8_median_incl'], rtol=1e-6)
TR = targets()['region_claims']
check('paper row C1-C4 = original regions', [b.C1_aniso_min, b.C2_aniso_max, b.C3_w_over_max, b.C4_w_over_mean],
      [TR['R1']['aniso_min'], TR['R2']['aniso_max'], TR['R1']['w_over_max_median'], TR['R2']['w_over_mean_median']],
      rtol=1e-10)
