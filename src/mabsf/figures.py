"""The paper figure: structure functions and shapes (Science double column, 7.25 in).

Panels
  A  epochs 2, 3, 4 as R, G, B on the UV grid; the 46 windows (dashed: long axis
     unbounded, drawn as upper limits in F); Regions 1 and 2; Region 2's drift
  B  Region 1, zoomed (fixed window)
  C  Region 1: PSF-deconvolved in-plane S2 in nine wedges, and S2 along W
  D  Region 2, zoomed (co-moving window)
  E  Region 2: the same as C, in the co-moving window
  F  a3/a2 against a2/a1 of every window; upper limits for the unbounded
     windows; the common shape (from the variance-ratio fit)

    fig = science_figure(cfg)           # cfg.calib selects the PSF calibration
"""
from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.patches import Rectangle

from . import config as C
from . import data as D
from . import preprocess as P
from . import regions as RG
from . import shape as S
from .plotting import C_R1, C_R2, C_WIN, FS_LETTER, FS_TEXT, LW, RGB_EPOCHS, RGB_RANGE, W_IN, rgb_composite, style

C_W = '#2c6fbb'                  # S2 along W
FRAME_LW = 1.3
A_MARGIN = 0.12                  # ly of sky around the windows in panel A
GLOBAL_U, GLOBAL_V = (103.8, 110.2), (55.8, 57.3)
XLIM_1D, YLIM_1D = (1e-2, 3e-1), (4e-4, 1.5e-1)
F_FLOOR = 0.02                   # lower limit of both axes of F (every bounded window is visible)


def place_free_label(fig, ax, text, kw, prefer=(0.62, 0.62)):
    """Place `text` at the free spot of `ax` (no markers, lines, legend or text under it) nearest `prefer`."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pts = []
    for ln in ax.lines:
        xy = ln.get_xydata()
        if len(xy) > 1:
            t = np.linspace(0, 1, 20)
            xy = np.concatenate([xy[i] + (xy[i + 1] - xy[i]) * t[:, None] for i in range(len(xy) - 1)])
        pts.append(ax.transData.transform(xy))
    for coll in ax.collections:
        for path in coll.get_paths():
            v = path.vertices
            if len(v) > 1:
                t = np.linspace(0, 1, 40)
                v = np.concatenate([v[i] + (v[i + 1] - v[i]) * t[:, None] for i in range(len(v) - 1)])
            pts.append(ax.transData.transform(v))
    pts = np.concatenate([p for p in pts if len(p)])
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    blocked = [t.get_window_extent(r) for t in ax.texts]
    if ax.get_legend() is not None:
        blocked.append(ax.get_legend().get_window_extent(r))
    t = ax.text(0, 0, text, transform=ax.transAxes, ha='center', va='center', **kw)
    best = None
    for fx in np.linspace(0.15, 0.9, 31):
        for fy in np.linspace(0.12, 0.9, 31):
            t.set_position((fx, fy))
            bb = t.get_window_extent(r).expanded(1.15, 1.3)
            if any(bb.overlaps(b) for b in blocked):
                continue
            if ((pts[:, 0] > bb.x0) & (pts[:, 0] < bb.x1) & (pts[:, 1] > bb.y0) & (pts[:, 1] < bb.y1)).sum():
                continue
            d = np.hypot(fx - prefer[0], fy - prefer[1])
            if best is None or d < best[0]:
                best = (d, fx, fy)
    if best is None:
        t.remove(); return None
    t.set_position(best[1:])
    return best[1:]


def letter(fig, ax, s, dx=-0.36, dy=0.04):
    b = ax.get_position(); Wf, Hf = fig.get_size_inches()
    fig.text(b.x0 + dx / Wf, b.y1 + dy / Hf, s, fontsize=FS_LETTER, fontweight='bold', ha='left', va='bottom')


def shape_panel(ax, k, summ):
    """Panel F: bounded windows with jackknife errors, upper limits, common shape."""
    d = k[S.usable(k) & ~k.collapsed]
    rng = (F_FLOOR, 1.0)
    ax.plot(rng, rng, '-', color='0.55', lw=0.9, zorder=2)
    s = d[(d.a2a1 >= rng[0]) & (d.a3a2 >= rng[0])]
    xlo = np.maximum(s.a2a1 - s.se_a2a1, rng[0] * 1.001); ylo = np.maximum(s.a3a2 - s.se_a3a2, rng[0] * 1.001)
    ax.errorbar(s.a2a1, s.a3a2, xerr=[s.a2a1 - xlo, s.se_a2a1], yerr=[s.a3a2 - ylo, s.se_a3a2], fmt='o',
                ms=np.sqrt(26) * 0.62, color=C_WIN, ecolor=C_WIN, elinewidth=0.5, capsize=0, mew=0, zorder=6)
    for (r, c), xu in S.upper_limits(k).items():
        y = min(max(k.set_index(['row', 'col']).a3a2[(r, c)], rng[0] * 1.3), 1.0)
        ax.annotate('', xy=(max(xu / 1.55, rng[0] * 1.06), y), xytext=(xu, y), zorder=12, annotation_clip=False,
                    arrowprops=dict(arrowstyle='-|>', color=C_WIN, lw=1.1, mutation_scale=7, shrinkA=0, shrinkB=0))
        ax.plot([xu], [y], marker='|', ms=7, mew=1.3, color=C_WIN, zorder=12, clip_on=False)
    ax.plot([summ['common_a2a1']], [summ['common_a3a2']], 'D', ms=6.0, mfc=C_WIN, mec='black', mew=0.6, zorder=16)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_aspect('equal')
    ax.set_xlim(rng[0], 1.0); ax.set_ylim(rng[0], 1.0)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(mticker.FixedLocator([0.02, 0.05, 0.1, 0.2, 0.5, 1.0]))
        axis.set_major_formatter(mticker.FixedFormatter(['0.02', '0.05', '0.1', '0.2', '0.5', '1']))
        axis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel(r'$a_2/a_1$', labelpad=1.5); ax.set_ylabel(r'$a_3/a_2$', labelpad=1)
    lab_kw = dict(fontsize=FS_TEXT, color='0.35', zorder=20,
                  bbox=dict(boxstyle='square,pad=0.15', fc='white', ec='none', alpha=0.8))
    ax.text(0.04, 0.42, 'prolate\n' r'$a_3/a_2 > a_2/a_1$', transform=ax.transAxes, ha='left', va='center', **lab_kw)
    ax.text(0.96, 0.10, 'oblate\n' r'$a_3/a_2 < a_2/a_1$', transform=ax.transAxes, ha='right', va='center', **lab_kw)
    keep = [(mlines.Line2D([], [], ls='none', marker='o', ms=np.sqrt(26) * 0.62, color=C_WIN, mew=0),
             'On-cloud windows (%d)' % len(s)),
            (mlines.Line2D([], [], ls='none', marker=r'$\leftarrow$', ms=7, color=C_WIN, mew=0.3),
             'Long axis unbounded\n(2$\\sigma$ upper limit)'),
            (mlines.Line2D([], [], ls='none', marker='D', ms=5, mfc=C_WIN, mec='black', mew=0.6), 'Common cloud shape')]
    ax.legend([q[0] for q in keep], [q[1] for q in keep], loc='lower left', frameon=True, framealpha=0.9,
              edgecolor='none', fancybox=False, handlelength=1.6, handletextpad=0.4, borderaxespad=0.3,
              labelspacing=0.3).set_zorder(20)
    ax.text(0.97, 0.97, 'spherical', transform=ax.transAxes, ha='right', va='top', **lab_kw)
    ax.plot([1.0], [1.0], 'o', ms=4, mfc='white', mec='0.35', mew=0.7, clip_on=False, zorder=21)
    place_free_label(ax.figure, ax, 'triaxial', lab_kw)
    return s


def common_footprint():
    """Boolean (ny, nx): pixels with data (finite flux) in all epochs of the UV cube."""
    cube = D.uv_cube('draw1')
    m = np.ones(cube.shape[1:], bool)
    for k in range(cube.shape[0]):
        m &= np.isfinite(np.asarray(cube[k]))
    return m


def science_figure(cfg=C.Config(), footprint=False):
    """footprint=True: also outline (light dashed) the area with data in all 7 epochs in panel A."""
    style()
    cube = D.uv_cube('draw1'); Ug, Vg = D.uv_grids(); w = D.epoch_w()
    summ, k = S.summary(cfg, n_boot=0)
    vu, vv, _ = RG.fit_velocity(2)
    reg = {1: RG.region_sf(1), 2: RG.region_sf(2, (vu, vv))}
    wins = P.sample_windows(cfg)

    # ---- layout [in]
    up = (GLOBAL_U[1] - GLOBAL_U[0]) * 0.2; vp = (GLOBAL_V[1] - GLOBAL_V[0]) * 0.2
    ulim = [GLOBAL_U[0] - up, GLOBAL_U[1] + up]; vlim = [GLOBAL_V[0] - vp, GLOBAL_V[1] + vp]
    r0 = np.array([r for r, _ in wins]); c0 = np.array([c for _, c in wins]); s_ = cfg.size
    wu = np.r_[Ug[r0, c0], Ug[r0 + s_ - 1, c0 + s_ - 1]]; wv = np.r_[Vg[r0, c0], Vg[r0 + s_ - 1, c0 + s_ - 1]]
    ulim = (min(ulim[0], wu.min() - A_MARGIN), max(ulim[1], wu.max() + A_MARGIN))
    vlim = (min(vlim[0], wv.min() - A_MARGIN), max(vlim[1], wv.max() + A_MARGIN))
    L, R = 0.40, 0.09
    w_top = W_IN - L - R
    h_top = w_top * (vlim[1] - vlim[0]) / (ulim[1] - ulim[0])
    H_blk, gap_r = 2.80, 0.20
    h_row = (H_blk - gap_r) / 2
    g_1d, g_sp = 0.44, 0.47
    w_1d = W_IN - L - h_row - g_1d - g_sp - H_blk - R
    bot, gap_tb, top = 0.33, 0.50, 0.17
    H_IN = bot + H_blk + gap_tb + h_top + top
    fig = plt.figure(figsize=(W_IN, H_IN))

    def ax_in(x, y, ww, hh):
        return fig.add_axes([x / W_IN, y / H_IN, ww / W_IN, hh / H_IN])
    ax_top = ax_in(L, bot + H_blk + gap_tb, w_top, h_top)
    rows_y = [bot + h_row + gap_r, bot]
    ax_img = [ax_in(L, y, h_row, h_row) for y in rows_y]
    ax_1d = [ax_in(L + h_row + g_1d, y, w_1d, h_row) for y in rows_y]
    ax_sp = ax_in(L + h_row + g_1d + w_1d + g_sp, bot, H_blk, H_blk)

    # ---- A
    ax_top.imshow(rgb_composite(cube), origin='lower', aspect='equal', interpolation='antialiased',
                  extent=[np.nanmin(Ug), np.nanmax(Ug), np.nanmin(Vg), np.nanmax(Vg)])
    ax_top.set_xlim(*ulim); ax_top.set_ylim(*vlim)
    ax_top.set_xlabel('U [ly]', labelpad=1.5); ax_top.set_ylabel('V [ly]', labelpad=2)
    ax_top.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
    B = C.REGION_BOX
    for i in (1, 2):
        vs, us = RG.box_corner(i)
        u_g = Ug[vs:vs + B, us:us + B]; v_g = Vg[vs:vs + B, us:us + B]
        rc = (C_R1, C_R2)[i - 1]
        for lw, ec in ((2.0, 'black'), (1.1, rc)):
            ax_top.add_patch(Rectangle((u_g.min(), v_g.min()), u_g.max() - u_g.min(), v_g.max() - v_g.min(),
                                       lw=lw, ec=ec, fc='none', zorder=6))
        ax_top.text(u_g.min(), v_g.min() - 0.05, 'Region %d' % i, color=rc, fontsize=FS_TEXT, fontweight='bold',
                    ha='left', va='top', zorder=7, path_effects=[pe.withStroke(linewidth=1.5, foreground='black')])
    col = set(zip(k.row[k.collapsed], k.col[k.collapsed]))
    inset = 3
    for rr0, cc0 in wins:
        rr = np.array([rr0 + inset, rr0 + inset, rr0 + s_ - inset, rr0 + s_ - inset, rr0 + inset])
        cc = np.array([cc0 + inset, cc0 + s_ - inset, cc0 + s_ - inset, cc0 + inset, cc0 + inset])
        ok = (rr0, cc0) not in col
        ax_top.plot(Ug[rr, cc], Vg[rr, cc], color=C_WIN, lw=0.6 if ok else 0.7, ls='-' if ok else (0, (2, 1.2)),
                    zorder=3)
    hs = [mlines.Line2D([], [], color=C_WIN, lw=1.0), mlines.Line2D([], [], color=C_WIN, lw=1.0, ls=(0, (2, 1.2)))]
    labs = ['Shown in F (%d)' % len(wins), 'Upper limit in F (%d)' % len(col)]
    if footprint:
        st = 4
        fp = common_footprint()[::st, ::st].astype(float)
        ax_top.contour(np.asarray(Ug[0, ::st]), np.asarray(Vg[::st, 0]), fp, [0.5], colors='white',
                       linewidths=0.7, linestyles=[(0, (3, 2))], alpha=0.8, zorder=4)
        hs.append(mlines.Line2D([], [], color='white', lw=0.7, ls=(0, (3, 2)), alpha=0.8))
        labs.append('Data in all 7 epochs')
    ax_top.legend(hs, labs, loc='upper right',
                  labelcolor='white', handlelength=1.6, borderaxespad=0.3, frameon=True, facecolor='black',
                  edgecolor='none', framealpha=0.75, fancybox=False).set_zorder(20)
    v_c2, u_c2 = C.REGION_CENTERS[2]; dW = w[-1] - w[0]
    u0, v0 = Ug[v_c2, u_c2], Vg[v_c2, u_c2]
    ax_top.annotate('', xy=(u0 + vu * dW, v0 + vv * dW), xytext=(u0, v0),
                    arrowprops=dict(arrowstyle='-|>', color=C_R2, lw=1.0, mutation_scale=7), zorder=7)

    # ---- B-E
    angles = RG.ANGLES
    acol = plt.cm.viridis(np.linspace(0.0, 0.85, len(angles)))
    for row, i in enumerate((1, 2)):
        R_ = reg[i]
        ny, nx = R_['sf2d'].shape
        u_vec = np.linspace(-(nx // 2) * C.PIX_LY, (nx // 2) * C.PIX_LY, nx)
        v_vec = np.linspace(-(ny // 2) * C.PIX_LY, (ny // 2) * C.PIX_LY, ny)
        a = ax_img[row]
        a.imshow(rgb_composite(R_['cube'], RGB_EPOCHS, RGB_RANGE), origin='lower', interpolation='antialiased',
                 extent=[u_vec[0], u_vec[-1], v_vec[0], v_vec[-1]])
        a.set_xlim(u_vec[0], u_vec[-1]); a.set_ylim(v_vec[0], v_vec[-1])
        a.xaxis.set_major_locator(mticker.MultipleLocator(0.1)); a.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
        a.set_ylabel(r'$\Delta V$ [ly]', labelpad=1)
        if row == 0:
            a.tick_params(labelbottom=False)
        else:
            a.set_xlabel(r'$\Delta U$ [ly]', labelpad=1.5)
        a.text(0.04, 0.96, 'Region %d' % i + (' (tracked)' if i == 2 else ''), transform=a.transAxes, ha='left',
               va='top', fontsize=FS_TEXT, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=(C_R1, C_R2)[row], lw=0.8, alpha=0.9))
        b = ax_1d[row]
        for j in range(len(angles)):
            p = R_['prof'][j]
            b.loglog(R_['r'], np.where(p > 0, p, np.nan), color=acol[j], lw=0.9, alpha=0.9)
        b.loglog(R_['w_sep'], R_['w_s2'], 'o', color=C_W, ms=3.6, mew=0, label='W' + (' (tracked)' if i == 2 else ''),
                 zorder=5)
        b.set_xlim(XLIM_1D); b.set_ylim(YLIM_1D)
        b.xaxis.set_major_locator(mticker.FixedLocator([0.01, 0.02, 0.05, 0.1, 0.2]))
        b.xaxis.set_major_formatter(mticker.FixedFormatter(['0.01', '0.02', '0.05', '0.1', '0.2']))
        b.xaxis.set_minor_formatter(mticker.NullFormatter())
        b.set_ylabel(r'$S_2$', labelpad=1)
        if row == 0:
            b.tick_params(labelbottom=False)
        else:
            b.set_xlabel('Separation [ly]', labelpad=1.5)
        b.legend(loc='lower right', handletextpad=0.3, borderaxespad=0.3)
        if row == 0:
            cax = b.inset_axes([0.06, 0.86, 0.36, 0.045])
            cm = mcolors.ListedColormap(acol); dA = angles[1] - angles[0]
            nb = mcolors.BoundaryNorm(np.r_[angles - dA / 2, angles[-1] + dA / 2], cm.N)
            tk = [angles[0], angles[len(angles) // 2], angles[-1]]
            cb = fig.colorbar(plt.cm.ScalarMappable(norm=nb, cmap=cm), cax=cax, orientation='horizontal', ticks=tk)
            cb.ax.set_xticklabels(['%.0f\u00b0' % x for x in tk])
            cb.ax.tick_params(labelsize=FS_TEXT - 1, length=1.5, width=LW, pad=1)
            cb.outline.set_linewidth(LW)
            cb.ax.set_title('angle from +U', fontsize=FS_TEXT - 1, pad=1.5)
        for ax in (a, b):
            for sp in ax.spines.values():
                sp.set_edgecolor((C_R1, C_R2)[row]); sp.set_linewidth(FRAME_LW)

    # ---- F
    drawn = shape_panel(ax_sp, k, summ)
    for ax, s in zip([ax_top, ax_img[0], ax_1d[0], ax_img[1], ax_1d[1], ax_sp], 'ABCDEF'):
        letter(fig, ax, s, dx=-0.36 if ax is not ax_top else -0.38)
    fig.panel_data = dict(summary=summ, windows=k, drawn=drawn, regions=reg, velocity=(vu, vv), size_in=(W_IN, H_IN))
    return fig


def save(fig, name='science_fig_sf_shape'):
    p = D.figures(name)
    fig.savefig(p + '.pdf', dpi=450)
    fig.savefig(p + '.png', dpi=300)
    return p + '.pdf', p + '.png'
