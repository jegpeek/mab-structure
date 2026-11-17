"""Plot style and image helpers shared by the notebooks and the paper figure."""
from __future__ import annotations

import os

import numpy as np

#: Science double-column width [in].
W_IN = 7.25
FS_LETTER, FS_LABEL, FS_TEXT = 10, 8, 7
LW = 0.6
#: Colours: sample windows, Region 1, Region 2 (Okabe-Ito).
C_WIN, C_R1, C_R2 = '#E69F00', '#CC79A7', '#56B4E9'
#: Epochs shown in the colour composites (0-based: epochs 2, 3, 4) and their flux range.
RGB_EPOCHS = (1, 2, 3)
RGB_RANGE = (0.3, 1.1)


def style():
    """Science-style rcParams (Arial/Helvetica if available, 7-8 pt, thin lines)."""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in ('Arial.ttf', 'Arial Bold.ttf'):
        p = os.path.join('/System/Library/Fonts/Supplemental', f)
        if os.path.exists(p):
            font_manager.fontManager.addfont(p)
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': FS_TEXT, 'axes.labelsize': FS_LABEL, 'xtick.labelsize': FS_TEXT,
        'ytick.labelsize': FS_TEXT, 'legend.fontsize': FS_TEXT, 'axes.titlesize': FS_LABEL,
        'axes.linewidth': LW, 'xtick.major.width': LW, 'ytick.major.width': LW,
        'xtick.minor.width': 0.5, 'ytick.minor.width': 0.5, 'xtick.major.size': 2.5,
        'ytick.major.size': 2.5, 'xtick.minor.size': 1.5, 'ytick.minor.size': 1.5,
        'xtick.direction': 'out', 'ytick.direction': 'out', 'lines.linewidth': 0.9,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'legend.frameon': False, 'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial',
        'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold', 'figure.dpi': 110,
    })


def rgb_composite(cube, epochs=RGB_EPOCHS, rng=RGB_RANGE, colors=np.eye(3)):
    """Colour composite of three epochs (raw flux, linear stretch over `rng`).

    Structure that is the same in the three epochs appears grey/white; structure
    that moves or changes between them appears in colour.  Pixels missing in
    any of the epochs are black.
    """
    lo, hi = rng
    rgb = np.zeros(np.shape(cube)[1:] + (3,))
    for c, k in zip(colors, epochs):
        rgb += np.clip((np.nan_to_num(np.asarray(cube[k], float)) - lo) / (hi - lo), 0, 1)[..., None] * c
    rgb = np.clip(rgb, 0, 1)
    rgb[~np.all([np.isfinite(np.asarray(cube[k])) for k in epochs], axis=0)] = 0.0
    return rgb
