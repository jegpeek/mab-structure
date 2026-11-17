"""Compare results with the values frozen from the original analysis (validation/).

Each notebook ends with a few `check(...)` calls; scripts/validate.py collects
every check into validation/REPORT.md.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .data import REPO

VDIR = os.path.join(REPO, 'validation')
_LOG = []


def targets():
    return json.load(open(os.path.join(VDIR, 'targets.json')))


def target(path):
    """Value at a '/'-separated path in targets.json, e.g. 'population_dedup/variance_ratio/f12'."""
    v = targets()
    for k in path.split('/'):
        v = v[k]
    return v


def check(name, value, expected, rtol=1e-6, atol=0.0, quiet=False):
    """Record and print one comparison (see log() for the outcome)."""
    if isinstance(expected, str):
        expected = target(expected)
    a, b = np.asarray(value, float), np.asarray(expected, float)
    ok = a.shape == b.shape and bool(np.all(np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)))
    diff = float(np.nanmax(np.abs(a - b))) if a.shape == b.shape and a.size else np.nan
    _LOG.append(dict(name=name, ok=ok, value=a.tolist(), expected=b.tolist(), max_abs_diff=diff, rtol=rtol, atol=atol))
    if not quiet:
        show = (lambda x: np.array2string(x, precision=5) if x.size > 1 else '%.6g' % x)
        print('%-4s %-48s %s  (expected %s)' % ('ok' if ok else 'FAIL', name, show(a) if a.size <= 8 else
                                               '[%d values, max |diff| %.2g]' % (a.size, diff), show(b) if b.size <= 8 else ''))


def log():
    return list(_LOG)
