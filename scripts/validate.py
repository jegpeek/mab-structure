"""Collect every check from the executed notebooks into validation/REPORT.md.

    python scripts/run_notebooks.py          # execute (stores outputs)
    python scripts/validate.py               # summarise

Each notebook ends with `check(...)` calls comparing its results with the values
frozen from the original analysis (validation/targets*.json/csv/npz).  This
script reads the printed outcome of every check from the stored outputs.
"""
import glob
import os
import re
import sys

import nbformat

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
LINE = re.compile(r'^(ok|FAIL)\s+(.+?)\s{2,}(.*)$')


def collect():
    out = []
    for p in sorted(glob.glob(os.path.join(ROOT, 'notebooks', '[0-9][0-9]_*.ipynb'))):
        nb = nbformat.read(p, 4)
        text = ''.join(o.get('text', '') for c in nb.cells if c.cell_type == 'code'
                       for o in c.get('outputs', []) if o.output_type == 'stream')
        errors = [o.ename + ': ' + o.evalue for c in nb.cells if c.cell_type == 'code'
                  for o in c.get('outputs', []) if o.output_type == 'error']
        rows = [LINE.match(l).groups() for l in text.split('\n') if LINE.match(l)]
        out.append((os.path.basename(p), rows, errors))
    return out


def main():
    res = collect()
    n_ok = sum(r[0] == 'ok' for _, rows, _ in res for r in rows)
    n_fail = sum(r[0] == 'FAIL' for _, rows, _ in res for r in rows)
    n_err = sum(len(e) for _, _, e in res)
    L = ['# Validation report', '',
         'Every result of the notebooks compared with the original analysis '
         '(values frozen by `validation/extract_targets.py`).', '',
         '**%d checks passed, %d failed; %d notebook errors.**' % (n_ok, n_fail, n_err), '',
         '| Notebook | Passed | Failed |', '|---|---|---|']
    for name, rows, errs in res:
        L.append('| %s | %d | %d%s |' % (name, sum(r[0] == 'ok' for r in rows), sum(r[0] == 'FAIL' for r in rows),
                                        (' (+%d errors)' % len(errs)) if errs else ''))
    for name, rows, errs in res:
        L += ['', '## ' + name, '', '| | Check | Result |', '|---|---|---|']
        L += ['| %s | %s | %s |' % ('✓' if s == 'ok' else '**✗**', n, v.replace('|', '/')) for s, n, v in rows]
        L += [''] + ['**Error:** ' + e for e in errs]
    L += ['', '## Known, accepted differences', '',
          '* PSF injection calibration (`results/psf_calibration.csv`): 1321 of 1404 fitted rows are bit-identical '
          'to the original; the remaining 83 differ by at most 2e-5 in an axis ratio (0.0005 deg in inclination), '
          'from thread-dependent floating-point rounding in the optimiser.  The bias table and every corrected '
          'window value agree to < 1e-4 relative; all population numbers agree to the precision quoted.',
          '* The original PSF bias table (`psf_calibration_published.csv`) contains duplicated synthetic fields '
          '(1368 rows, 78 distinct).  `Config.calib = \'dedup\'` (default) uses each field once; '
          '`\'published\'` reproduces the first-submitted numbers.']
    open(os.path.join(ROOT, 'validation', 'REPORT.md'), 'w').write('\n'.join(L) + '\n')
    print('%d passed, %d failed, %d errors' % (n_ok, n_fail, n_err))
    return n_fail + n_err


if __name__ == '__main__':
    sys.exit(1 if main() else 0)
