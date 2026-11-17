"""Freeze regression targets from the original (casa-fork) pipeline.

Run ONCE, inside the casa checkout's environment, to record what the published
analysis produces.  The new `mabsf` code is validated against these files by
validation/validate.py.  This script is the only place that imports casa code.

    CASA=/path/to/casa python validation/extract_targets.py

Writes validation/targets.json (scalars), validation/targets_windows.csv
(per-window table) and validation/targets_arrays.npz (array-valued products).
"""
import hashlib
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')
CASA = os.environ['CASA']
sys.path[:0] = [CASA, os.path.join(CASA, 'analysis')]
os.chdir(CASA)
HERE = os.path.dirname(os.path.abspath(__file__))

import sensitivity as S                       # noqa: E402
import variance_ratio as VR                   # noqa: E402
import cloud_occupancy as CO                  # noqa: E402
import make_seven_panel_tracked_psf as M7     # noqa: E402
import psf_deconv as PD                       # noqa: E402
from psf_deconv_injection import region_setup  # noqa: E402
import c6_trackability as CT                  # noqa: E402
import psf_correction as PC                   # noqa: E402

T, A = {}, {}


def sha(a):
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


# ---------------------------------------------------------------- 1. inputs on the UV grid
res = np.load('data/resampled_epochs_noclip.npy', mmap_mode='r')
U = np.load('data/U_grid.npy', mmap_mode='r'); V = np.load('data/V_grid.npy', mmap_mode='r')
w = np.load('data/epoch_mean_w.npy'); bg = np.load('data/noclip_backgrounds.npy')
T['grid'] = dict(shape=list(res.shape), U00=float(U[0, 0]), V00=float(V[0, 0]),
                 dU=float(U[0, 1] - U[0, 0]), dV=float(V[1, 0] - V[0, 0]),
                 epoch_mean_w=w.tolist(), backgrounds=bg.tolist(),
                 epoch_nanmean=[float(np.nanmean(res[k])) for k in range(res.shape[0])],
                 epoch_nfinite=[int(np.isfinite(res[k]).sum()) for k in range(res.shape[0])])

# ---------------------------------------------------------------- 2. preprocessing / sample
reg = S.region(S.BASE['on_k'], S.BASE['smooth'])
wins = S.sample(S.BASE)
occ = pd.read_csv('results/occupancy_windows_w150.csv')
occ_on = sorted(zip(occ[occ.on_cloud.astype(bool)].row, occ[occ.on_cloud.astype(bool)].col))
assert sorted(wins) == [tuple(map(int, x)) for x in occ_on], 'sample mismatch'
nz = CO.noise_levels()
T['sample'] = dict(n=len(wins), windows=[list(x) for x in sorted(wins)], region_npix=int(reg.sum()),
                   region_sha=sha(reg.astype(np.uint8)), noise_levels=[float(x) for x in nz],
                   R1_tile=[1864, 2092])

# ---------------------------------------------------------------- 3. per-window fits
rows = []
d = S.fit_dir(S.BASE)
for r, c in sorted(wins):
    J = json.load(open(os.path.join(d, 'sb_r%d_c%d_s150.json' % (r, c))))
    cen = J['central']['3d|powerlaw']
    rows.append(dict(row=r, col=c, raw_a2a1=cen['a2a1'], raw_a3a2=cen['a3a2'], raw_alpha=cen['alpha'],
                     raw_incl=cen['incl'], raw_b2b1=cen.get('b2b1', np.nan),
                     n_jk=len(J['samples'])))
W = pd.DataFrame(rows)
for calib in ('base', 'dedup'):
    c = S.cfg_of(dict(calib=calib))
    k = VR.load_cfg(c)
    k = k[['row', 'col', 'a2a1', 'a3a2', 'alpha', 'incl', 'f12', 'f23', 'se_f12', 'se_f23', 'se_l12', 'collapsed']]
    W = W.merge(k.add_suffix('_' + calib).rename(columns={'row_' + calib: 'row', 'col_' + calib: 'col'}),
                on=['row', 'col'])
W.to_csv(os.path.join(HERE, 'targets_windows.csv'), index=False)

# ---------------------------------------------------------------- 4. population statistics
claims = pd.read_csv('results/sensitivity_claims.csv')
vrs = pd.read_csv('results/variance_ratio_sensitivity.csv')
for calib, lab in (('base', 'Science baseline'), ('dedup', 'bias calibration de-duplicated')):
    row = claims[claims.choice == lab].iloc[0].drop(['group', 'choice']).to_dict()
    vr = vrs[vrs.choice == lab].iloc[0].drop(['group', 'choice']).to_dict()
    k = VR.load_cfg(S.cfg_of(dict(calib=calib)))
    am = float(np.median(k.alpha))
    u = k[k.collapsed]
    lim = {'%d_%d' % (r, c): float(min(f + 2 * e, 1.0) ** (1 / a))
           for r, c, f, e, a in zip(u.row, u.col, u.f12, u.se_f12, u.alpha)}
    T['population_' + calib] = dict(
        claims={a: (None if (isinstance(b, float) and not np.isfinite(b)) else (float(b) if not isinstance(b, (bool, np.bool_)) else bool(b))) for a, b in row.items()},
        variance_ratio={a: float(b) for a, b in vr.items()},
        median_alpha=am, common_a2a1=float(vr['f12'] ** (1 / am)), common_a3a2=float(vr['f23'] ** (1 / am)),
        upper_limits_a2a1=lim,
        n_prolate_f=int((k.f12 < k.f23).sum()),
        n_exclude_isotropy_2sig=int(((k.f12 + 2 * k.se_f12 < 1) | (k.f23 + 2 * k.se_f23 < 1)).sum()),
        rho_f23_incl=float(stats.spearmanr(k.f23, k.incl)[0]))

# ---------------------------------------------------------------- 5. PSF bias calibration
bt = pd.read_csv(PC.FIELD_TABLE)
med, sem = PC.bias_by_tilt('add', 5)
T['psf'] = dict(field_table_rows=len(bt), field_table_unique_fields=int(bt[['shape', 'tilt', 'seed']].drop_duplicates().shape[0]),
                bias_by_tilt={k: med[k].round(6).tolist() for k in PC.CORRECTED}, tilts=med.index.tolist())
Z = np.load('data/psf_f444w_uv.npz')
T['psf']['psf_inputs'] = {k: (sha(Z[k]) if Z[k].size > 4 else np.asarray(Z[k]).tolist()) for k in Z.files}

# ---------------------------------------------------------------- 6. Regions 1 and 2 (panels B-E)
resf = np.load('data/resampled_epochs_noclip.npy')
B = M7.BOX_SIZE
v_c2, u_c2 = M7.CENTERS_VU[1]
vu, vv, nco = M7.fit_velocity(resf, w, v_c2, u_c2, B)
T['regions'] = dict(centers_vu=[list(x) for x in M7.CENTERS_VU], box=B, R2_velocity=[float(vu), float(vv)],
                    R2_n_coherent=int(nco))
J_REG = region_setup()['J']
angles = np.linspace(0, 180, M7.NUM_UV_ANGLES, endpoint=False)
PIX = M7.DU
for i, (v_c, u_c) in enumerate(M7.CENTERS_VU):
    vs, us = v_c - B // 2, u_c - B // 2
    cube = (np.nan_to_num(resf[:, vs:vs + B, us:us + B]) if i == 0
            else M7.comoving_cube(resf, w, v_c, u_c, B, vu, vv)[0])
    sf2 = PD.sf2d_epochs(cube, J_REG[i], correct=True)
    ny, nx = sf2.shape
    u_vec = np.linspace(-(nx // 2) * PIX, (nx // 2) * PIX, nx); v_vec = np.linspace(-(ny // 2) * PIX, (ny // 2) * PIX, ny)
    LU, LV = np.meshgrid(u_vec, v_vec); LR = np.hypot(LU, LV)
    LT = np.degrees(np.arctan2(LV, LU)); LT[LT < 0] += 360
    rb = np.logspace(np.log10(PIX), np.log10(min(u_vec[-1], v_vec[-1])), 40); rc = 0.5 * (rb[1:] + rb[:-1])
    Wd = M7.ANGLE_WIDTH
    prof = []
    for ang in angles:
        m = (((LT >= ang - Wd) & (LT <= ang + Wd)) | ((LT >= ang + 180 - Wd) & (LT <= ang + 180 + Wd)))
        prof.append([np.mean(sf2[m & (LR >= rb[j]) & (LR < rb[j + 1])]) for j in range(len(rb) - 1)])
    prof = np.array(prof)
    vmag = np.hypot(vu, vv)
    sep, sw = PD.w_points(cube, w, J_REG[i], correct=True, sep_scale=(1.0 if i == 0 else np.sqrt(1 + vmag ** 2)))
    A['R%d_cube_sha' % (i + 1)] = np.array(sha(cube.astype(np.float64)))
    A['R%d_sf2d' % (i + 1)] = sf2; A['R%d_r' % (i + 1)] = rc; A['R%d_prof' % (i + 1)] = prof
    A['R%d_wsep' % (i + 1)] = np.asarray(sep); A['R%d_ws2' % (i + 1)] = np.asarray(sw)
    A['R%d_J' % (i + 1)] = np.asarray(J_REG[i])
A['angles'] = angles

# ---------------------------------------------------------------- 7. C6 trackability (all windows)
c6 = pd.read_csv('results/c6_trackability.csv')
T['trackability'] = dict(
    rho_ncct_incl=float(stats.spearmanr(c6.incl, c6.ncct)[0]),
    rho_ncct_A=float(stats.spearmanr(c6.A, c6.ncct)[0]),
    rho_ncct_lmin=float(stats.spearmanr(c6.l_min, c6.ncct)[0]),
    median_ncct=float(c6.ncct.median()))
c6w = pd.read_csv('results/sensitivity_c6_windows.csv'); c6w = c6w[c6w.choice == 'Science baseline']
T['drift'] = dict(n_minus_U=int((c6w.vU < 0).sum()), n=len(c6w))
c6.to_csv(os.path.join(HERE, 'targets_c6.csv'), index=False)

json.dump(T, open(os.path.join(HERE, 'targets.json'), 'w'), indent=1, default=float)
np.savez_compressed(os.path.join(HERE, 'targets_arrays.npz'), **A)
print('wrote targets: %d windows, R2 v = (%.3f, %.3f)' % (len(wins), vu, vv))
