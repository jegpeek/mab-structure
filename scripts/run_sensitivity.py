"""Fit every window of every sensitivity choice, then evaluate all claims.

    python scripts/run_sensitivity.py fit [procs]     # ~1-2 h on 8 cores, cached
    python scripts/run_sensitivity.py claims          # writes results/sensitivity_claims.csv
"""
import sys
from mabsf import sensitivity as SE

if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'fit'
    if what == 'fit':
        SE.fit_choices(procs=int(sys.argv[2]) if len(sys.argv) > 2 else 8)
        print('fits done', flush=True)
    else:
        SE.run()
