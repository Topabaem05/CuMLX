"""Loaded by each Python process that ``cumlx run`` starts, through PYTHONPATH.

The compatibility layer is loaded by file path, so that the script's process does not import cumlx
or numpy before the script does.
"""
import importlib.util
import os
from pathlib import Path
import sys

if os.environ.get('CUMLX_RUN_TARGET'):
    _path = Path(__file__).resolve().parent.parent / 'torch_cuda.py'
    _spec = importlib.util.spec_from_file_location('_cumlx_torch_cuda', _path)
    _module = importlib.util.module_from_spec(_spec)
    sys.modules['_cumlx_torch_cuda'] = _module
    _spec.loader.exec_module(_module)
    _module.activate()
