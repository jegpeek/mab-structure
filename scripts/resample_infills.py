"""Resample the alternative infills (sensitivity choice D3) onto the UV grid."""
import sys
from mabsf import geometry as G
for k in (sys.argv[1:] or ['draw2', 'draw3', 'mean', 'infillflag']):
    G.resample_epochs(k, verbose=False)
    print('wrote', k, flush=True)
