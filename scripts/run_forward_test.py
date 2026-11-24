"""The PSF forward-model test of notebook 02 (~15 min on 8 cores).

    python scripts/run_forward_test.py [procs]
"""
import sys
from mabsf import config as C, preprocess as P, psf_forward as PF

if __name__ == '__main__':
    PF.run_test(P.sample_windows(C.Config()), procs=int(sys.argv[1]) if len(sys.argv) > 1 else 8)
    print('done')
