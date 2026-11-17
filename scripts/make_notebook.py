"""Build a notebook from a plain-text source with `# %%` cell markers.

    python scripts/make_notebook.py notebooks/src/00_data_and_geometry.py

Cells start with `# %%` (code) or `# %% [markdown]` (markdown; each line's
leading '# ' is stripped).  The notebook is written next to the source's
parent directory: notebooks/src/X.py -> notebooks/X.ipynb.  The .py sources
are the files to edit and review; the .ipynb files carry the executed outputs.
"""
import os
import re
import sys

import nbformat


def parse(text):
    cells = []
    for block in re.split(r'^# %%', text, flags=re.M)[1:]:
        head, _, body = block.partition('\n')
        body = body.strip('\n')
        if '[markdown]' in head:
            lines = [ln[2:] if ln.startswith('# ') else ln.lstrip('#') for ln in body.split('\n')]
            cells.append(nbformat.v4.new_markdown_cell('\n'.join(lines)))
        else:
            cells.append(nbformat.v4.new_code_cell(body))
    return cells


def build(src):
    nb = nbformat.v4.new_notebook()
    nb.cells = parse(open(src).read())
    nb.metadata['kernelspec'] = dict(name='python3', display_name='Python 3', language='python')
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(src))),
                       os.path.splitext(os.path.basename(src))[0] + '.ipynb')
    nbformat.write(nb, out)
    return out


if __name__ == '__main__':
    for s in sys.argv[1:]:
        print(build(s))
