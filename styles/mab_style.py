"""Helpers that go with mab_science.mplstyle (things a style sheet cannot set).

    import mab_style
    mab_style.use()                                   # register Arial if present, apply the style
    mab_style.log_ticks(ax.xaxis, [0.01, 0.02, 0.05, 0.1, 0.2])
    mab_style.letter(ax, 'A')
"""
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import font_manager

STYLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mab_science.mplstyle')
#: Colours of the paper figure (Okabe-Ito).
C_WIN, C_R1, C_R2, C_W = '#E69F00', '#CC79A7', '#56B4E9', '#2c6fbb'
W_DOUBLE, W_SINGLE = 7.25, 3.5      # Science column widths [in]


def use():
    for f in ('Arial.ttf', 'Arial Bold.ttf', 'Arial Italic.ttf'):
        p = os.path.join('/System/Library/Fonts/Supplemental', f)
        if os.path.exists(p):
            font_manager.fontManager.addfont(p)
    plt.style.use(STYLE)


def log_ticks(axis, ticks):
    """Label only `ticks` on a log axis, as plain numbers ('0.02', '0.1', '1'); no minor labels."""
    axis.set_major_locator(mticker.FixedLocator(ticks))
    axis.set_major_formatter(mticker.FixedFormatter(['%g' % t for t in ticks]))
    axis.set_minor_formatter(mticker.NullFormatter())


def letter(ax, s, dx=-0.15, dy=1.02):
    """Bold 10 pt panel letter at the upper left, outside the axes."""
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=10, fontweight='bold', ha='left', va='bottom')
