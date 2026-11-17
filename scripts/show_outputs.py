"""Print the text outputs and errors of executed notebooks (for quick review)."""
import sys, nbformat
for p in sys.argv[1:]:
    nb = nbformat.read(p, 4)
    for i, c in enumerate(nb.cells):
        if c.cell_type != 'code':
            continue
        for o in c.outputs:
            if o.output_type == 'stream':
                print('[%d]' % i, o.text.rstrip()[:2500])
            elif o.output_type == 'error':
                print('[%d] ERROR' % i, o.ename, o.evalue)
            elif o.output_type == 'execute_result':
                print('[%d]' % i, o.data.get('text/plain', '')[:1500])
            elif o.output_type == 'display_data':
                print('[%d] <figure>' % i)
