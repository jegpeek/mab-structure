"""Stage the input files for the Zenodo deposit and record their checksums.

    python scripts/prepare_zenodo.py OUTDIR

Copies (or hard-links, when on the same volume) every file listed in
data/MANIFEST.md from $MABSF_DATA/inputs into OUTDIR, preserving the
sub-directory layout, and writes data/zenodo_checksums.txt (sha256, size,
path) for mabsf.data to verify downloads against.
"""
import hashlib
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from mabsf import config as C, data as D   # noqa: E402


def files():
    out = [D.mosaic_path(v) for v, _ in C.EPOCHS]
    out += [D.pos3d_path('2024'), D.pos3d_path('2025'), D.inputs(*D.PSF_FITS), D.inputs(*D.STARS_I2D)]
    return out


def sha256(p, block=1 << 24):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(block), b''):
            h.update(b)
    return h.hexdigest()


def main(outdir):
    root = os.path.realpath(D.inputs())
    lines = []
    for p in files():
        rel = os.path.relpath(p, D.inputs())
        dst = os.path.join(outdir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        src = os.path.realpath(p)
        if not os.path.exists(dst):
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        lines.append('%s  %d  %s' % (sha256(src), os.path.getsize(src), rel))
        print(lines[-1], flush=True)
    with open(os.path.join(D.REPO, 'data', 'zenodo_checksums.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('staged %d files under %s (from %s)' % (len(lines), outdir, root))


if __name__ == '__main__':
    main(sys.argv[1])
