"""Command-line entry point for the expensive steps.

    mabsf resample [--infill draw1]        sky mosaics -> UV cube            (~20 min)
    mabsf fit [--procs 6] [--set k=v ...]  per-window fits of the sample     (~1 min per 10 windows/proc)
    mabsf calibrate [--procs 6]            PSF bias injection calibration   (~1 h)
    mabsf sensitivity [--procs 6]          all sensitivity configurations   (hours)

Results are cached; each step skips work already done.
"""
from __future__ import annotations

import argparse
import ast


def _cfg(sets):
    from . import config as C
    kw = {}
    for s in sets or []:
        k, v = s.split('=', 1)
        try:
            kw[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            kw[k] = v
    return C.Config(**kw)


def main(argv=None):
    ap = argparse.ArgumentParser(prog='mabsf')
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('resample'); a.add_argument('--infill', default='draw1')
    a = sub.add_parser('fit'); a.add_argument('--procs', type=int, default=6); a.add_argument('--set', nargs='*')
    a = sub.add_parser('calibrate'); a.add_argument('--procs', type=int, default=6)
    a.add_argument('--variant', default='all')
    a = sub.add_parser('sensitivity'); a.add_argument('--procs', type=int, default=6)
    args = ap.parse_args(argv)
    if args.cmd == 'resample':
        from . import geometry
        geometry.resample_epochs(args.infill)
    elif args.cmd == 'fit':
        from . import fitting, preprocess
        cfg = _cfg(args.set)
        fitting.fit_windows(preprocess.sample_windows(cfg), cfg, procs=args.procs)
    elif args.cmd == 'calibrate':
        from . import psf
        psf.run_calibration(args.variant, procs=args.procs)
    elif args.cmd == 'sensitivity':
        from . import sensitivity
        sensitivity.run(procs=args.procs)


if __name__ == '__main__':
    main()
