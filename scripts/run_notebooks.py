"""Execute the notebooks in order and store their outputs.

    python scripts/run_notebooks.py [00 01 ...]     # default: all, in order

Each notebook runs in its own kernel with the working directory notebooks/.
Outputs are written back into the .ipynb so the rendered notebooks on GitHub
show every figure and number.
"""
import glob
import os
import sys
import time

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

HERE = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(HERE, '..', 'notebooks')


def run(path, timeout=7200):
    nb = nbformat.read(path, as_version=4)
    t0 = time.time()
    ExecutePreprocessor(timeout=timeout, kernel_name='python3').preprocess(nb, {'metadata': {'path': NB}})
    nbformat.write(nb, path)
    print('%-45s %5.0f s' % (os.path.basename(path), time.time() - t0), flush=True)


def run_inprocess(path):
    """Execute a notebook in this Python process (no Jupyter kernel).

    For environments where a kernel cannot be started.  Captures stdout,
    the value of each cell's last expression (HTML for DataFrames) and every
    matplotlib figure, and stores them as cell outputs.
    """
    import ast
    import base64
    import contextlib
    import io
    import traceback
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    nb = nbformat.read(path, as_version=4)
    cwd = os.getcwd(); os.chdir(NB)
    ns = {'__name__': '__main__'}
    t0 = time.time(); n = 0
    try:
        for cell in nb.cells:
            if cell.cell_type != 'code':
                continue
            n += 1
            outs = []
            buf = io.StringIO()
            try:
                tree = ast.parse(cell.source)
                last = tree.body[-1] if tree.body and isinstance(tree.body[-1], ast.Expr) else None
                body = tree.body[:-1] if last is not None else tree.body
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
                    exec(compile(ast.Module(body=body, type_ignores=[]), '<cell %d>' % n, 'exec'), ns)
                    val = eval(compile(ast.Expression(last.value), '<cell %d>' % n, 'eval'), ns) if last else None
            except Exception as e:     # noqa: BLE001
                if buf.getvalue():
                    outs.append(nbformat.v4.new_output('stream', name='stdout', text=buf.getvalue()))
                outs.append(nbformat.v4.new_output('error', ename=type(e).__name__, evalue=str(e),
                                                   traceback=traceback.format_exc().split('\n')))
                cell.outputs = outs; cell.execution_count = n
                nbformat.write(nb, path)
                raise
            if buf.getvalue():
                outs.append(nbformat.v4.new_output('stream', name='stdout', text=buf.getvalue()))
            for num in plt.get_fignums():
                fig = plt.figure(num); b = io.BytesIO()
                fig.savefig(b, format='png', dpi=110, bbox_inches='tight')
                outs.append(nbformat.v4.new_output('display_data', data={
                    'image/png': base64.b64encode(b.getvalue()).decode(), 'text/plain': '<Figure>'}))
            plt.close('all')
            quiet = cell.source.rstrip().endswith(';')
            if val is not None and not quiet and 'matplotlib' not in str(type(val)):
                data = {'text/plain': repr(val)}
                if hasattr(val, '_repr_html_'):
                    data['text/html'] = val._repr_html_()
                outs.append(nbformat.v4.new_output('execute_result', data=data, execution_count=n))
            cell.outputs = outs; cell.execution_count = n
    finally:
        os.chdir(cwd)
    nbformat.write(nb, path)
    print('%-45s %5.0f s (in-process)' % (os.path.basename(path), time.time() - t0), flush=True)


if __name__ == '__main__':
    args = sys.argv[1:]
    inproc = '--inprocess' in args
    want = [a for a in args if not a.startswith('--')]
    for p in sorted(glob.glob(os.path.join(NB, '[0-9][0-9]_*.ipynb'))):
        if not want or os.path.basename(p)[:2] in want:
            run_inprocess(p) if inproc else run(p)
