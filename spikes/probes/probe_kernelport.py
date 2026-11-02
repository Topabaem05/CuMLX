"""Does kernelport verify Metal ports, and does it fail a wrong one? Usage: probe_kernelport.py <kernelport cli> <zoo dir>

Gate: every stored Metal port passes, and vector_add fails after its plus sign becomes a minus sign.
Originals stay UNVERIFIED here because they need a CUDA GPU.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

CLI, ZOO = sys.argv[1], Path(sys.argv[2])


def zoo(work, *extra):
    env = {**os.environ, 'COLUMNS': '160', 'NO_COLOR': '1'}
    done = subprocess.run([CLI, 'zoo', '--backend', 'metal', '--no-bench', *extra], cwd=work, env=env, capture_output=True, text=True, timeout=500)
    rows = {}
    for line in done.stdout.splitlines():
        cells = [cell.strip() for cell in line.split('│')]
        if len(cells) > 5 and cells[1] and cells[1] != 'kernel':
            rows[cells[1]] = {'original': cells[2], 'port': cells[4], 'cases': cells[5]}
    return done.returncode, rows


with tempfile.TemporaryDirectory() as temp:
    work = Path(temp)
    shutil.copytree(ZOO, work / 'zoo')
    code, rows = zoo(work)
    print(json.dumps({'probe': 'kernelport', 'exit': code, 'rows': rows}), flush=True)
    port = work / 'zoo/vector_add/ports/metal.py'
    source = port.read_text()
    assert 'out[i] = a[i] + b[i];' in source
    port.write_text(source.replace('out[i] = a[i] + b[i];', 'out[i] = a[i] - b[i];'))
    wrong_code, wrong = zoo(work, '--kernel', 'vector_add')
    print(json.dumps({'probe': 'kernelport', 'control': 'wrong vector_add', 'exit': wrong_code, 'rows': wrong}), flush=True)
    passed = bool(rows) and all(row['port'] == 'PASS' for row in rows.values()) and wrong.get('vector_add', {}).get('port') == 'FAIL'
    print(json.dumps({'probe': 'kernelport', 'gate': 'all stored ports pass; the wrong port fails', 'pass': passed}), flush=True)
