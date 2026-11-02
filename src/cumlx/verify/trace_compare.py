"""Compare two golden traces of one model, such as a CUDA capture and a CPU capture.

This measures how much the reference itself moves between devices, or between two runs on one device. If that
number is above a threshold, a port can fail for a reason that has nothing to do with the port.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..errors import CumlxError
from .metrics import compare
from .trace import Trace

SCHEMA = 'cumlx.trace-compare.v1'
DEFAULT_REL_L2 = 1e-4  # the float32 threshold of `cumlx diff`


def compare_traces(a: str, b: str, *, rel_l2: float = DEFAULT_REL_L2) -> dict[str, Any]:
    first, second = Trace(a), Trace(b)
    threshold = {'rel_l2': float(rel_l2), 'zero_atol': float(rel_l2)}
    rows: list[dict[str, Any]] = []
    if len(first.manifest.samples) != len(second.manifest.samples):
        raise CumlxError('E_INPUT_INVALID', 'the traces have a different number of samples',
                         hint='capture both with the same inputs')
    for sa, sb in zip(first.manifest.samples, second.manifest.samples):
        if not np.array_equal(first.token_ids(sa), second.token_ids(sb)):
            raise CumlxError('E_INPUT_INVALID', f'sample {sa.id} has different token ids in the two traces',
                             hint='capture both with the same inputs')
        other = {r.key: r for r in sb.records}
        for r in sa.records:
            twin = other.get(r.key)
            if twin is None or r.output is None or twin.output is None:
                rows.append({'sample': sa.id, 'key': r.key, 'verdict': 'missing'})
                continue
            rows.append({'sample': sa.id, 'key': r.key, 'type': r.type, 'phase': r.phase,
                         **compare(first.tensor(sa, r.output.tensor), second.tensor(sb, twin.output.tensor), threshold)})
    values = sorted((x['rel_l2'] for x in rows if x.get('rel_l2') is not None), reverse=True)
    failed = [x for x in rows if x['verdict'] == 'fail']
    missing = [x for x in rows if x['verdict'] == 'missing']
    verdict = 'diverged' if failed else ('incomplete' if missing or not values else 'ok')
    return {'schema': SCHEMA, 'verdict': verdict, 'a': {'path': a, 'source': first.manifest.source},
            'b': {'path': b, 'source': second.manifest.source}, 'threshold_rel_l2': float(rel_l2),
            'summary': {'records': len(rows), 'compared': len(values), 'above_threshold': len(failed), 'missing': len(missing),
                        'worst_rel_l2': values[0] if values else None,
                        'median_rel_l2': values[len(values) // 2] if values else None},
            'worst': sorted([x for x in rows if x.get('rel_l2') is not None], key=lambda x: -x['rel_l2'])[:20]}
