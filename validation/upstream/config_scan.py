#!/usr/bin/env python3
"""Find published MLX ports whose config mlx-lm reads differently from transformers, without downloading weights.

    python validation/upstream/config_scan.py --limit 2000 -o validation/upstream/config-scan.json

For each recent model with the `mlx` library tag, it downloads config.json only, builds mlx-lm's own ModelArgs with
ModelArgs.from_dict, and compares the RoPE settings that mlx-lm will use with the ones transformers resolves from the
same config (AutoConfig.for_model; the raw `rope_parameters` when transformers cannot build the config):

* rope_base:    the reference rope_theta differs from ModelArgs.rope_theta, for example because the config has no
                top-level rope_theta and ModelArgs falls back to its default, or because the two disagree.
* rope_scaling: the reference rope_type is not "default" and ModelArgs has no rope_scaling.

Each flagged row also holds what ModelArgs would read with the change of ml-explore/mlx-lm#1964 (pr1964_logic.py).

A model whose ModelArgs has a `rope_parameters` field reads it itself and is not flagged.

A flagged port is a candidate. Each one must still be confirmed with `cumlx diff` against the original before it is
reported; this scan does not run any model.
"""
from __future__ import annotations

import argparse
import dataclasses
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys


def mlx_args(config: dict):
    from mlx_lm.utils import _get_classes

    _, args_class = _get_classes(config)
    return args_class.from_dict(config)


def transformers_rope(config: dict) -> dict | None:
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.for_model(config['model_type'], **{k: v for k, v in config.items() if k != 'model_type'})
    except BaseException:  # an architecture this transformers does not know
        return None
    rp = getattr(cfg, 'rope_parameters', None)
    return dict(rp) if isinstance(rp, dict) else None


def check(repo: str, config: dict) -> dict | None:
    if not isinstance(config.get('rope_parameters'), dict):
        return None
    reference = transformers_rope(config)
    rp = reference or config['rope_parameters']
    if isinstance(rp.get('full_attention'), dict):  # per-layer-type parameters (gemma3 style)
        rp = rp['full_attention']
    row = {'repo': repo, 'model_type': config.get('model_type'), 'transformers_version': config.get('transformers_version'),
           'top_level_rope_theta': config.get('rope_theta'), 'rope_parameters': config.get('rope_parameters'),
           'quantization': (config.get('quantization') or {}).get('bits'),
           'reference': 'transformers' if reference else 'config', 'transformers_rope_parameters': reference, 'findings': []}
    try:
        args = mlx_args(config)
    except BaseException as error:  # a config mlx-lm cannot read is a result here
        row['load'] = f'{type(error).__name__}: {str(error)[:200]}'
        return row
    if 'rope_parameters' in {f.name for f in dataclasses.fields(args)}:
        row['reads_rope_parameters'] = True  # this model's ModelArgs reads rope_parameters itself
        return row
    theta = rp.get('rope_theta')
    used = getattr(args, 'rope_theta', None)
    row['mlx_rope_theta'] = used
    if theta is not None and used is not None and not math.isclose(float(theta), float(used), rel_tol=1e-9):
        row['findings'].append('rope_base')
    kind = rp.get('rope_type') or rp.get('type')
    row['mlx_rope_scaling'] = getattr(args, 'rope_scaling', 'no attribute')
    if kind not in (None, 'default') and not getattr(args, 'rope_scaling', None):
        row['findings'].append('rope_scaling')
    if row['findings']:  # what mlx-lm would read with the change of ml-explore/mlx-lm#1964
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from mlx_lm.utils import _get_classes
        from pr1964_logic import from_dict

        fixed = from_dict.__func__(_get_classes(config)[1], config)
        fixed_theta = getattr(fixed, 'rope_theta', None)
        row['with_pr1964'] = {'rope_theta': fixed_theta, 'rope_scaling': getattr(fixed, 'rope_scaling', None),
                              'base_fixed': theta is None or (fixed_theta is not None and math.isclose(float(theta), float(fixed_theta), rel_tol=1e-9))}
    return row


def fetch(repo: str, cache: Path) -> dict | None:
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(repo, 'config.json', cache_dir=str(cache))
        return json.loads(Path(path).read_text())
    except BaseException:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--limit', type=int, default=2000)
    ap.add_argument('--cache', type=Path, default=Path('.config-scan-cache'))
    ap.add_argument('-o', '--output', type=Path, required=True)
    args = ap.parse_args()
    from huggingface_hub import HfApi
    from importlib import metadata

    repos = [m.id for m in HfApi().list_models(filter='mlx', sort='last_modified', limit=args.limit)]
    with ThreadPoolExecutor(8) as pool:
        configs = list(pool.map(lambda r: fetch(r, args.cache), repos))
    rows, read, with_rp = [], 0, 0
    for repo, config in zip(repos, configs):
        if not config:
            continue
        read += 1
        row = check(repo, config)
        if row is None:
            continue
        with_rp += 1
        if row['findings'] or 'load' in row:
            rows.append(row)
    doc = {'schema': 'cumlx.config-scan.v1', 'date': datetime.now(timezone.utc).strftime('%Y-%m-%d'),
           'mlx_lm': metadata.version('mlx-lm'), 'repos_listed': len(repos), 'configs_read': read,
           'configs_with_rope_parameters': with_rp,
           'flagged': [r for r in rows if r['findings']], 'not_loadable': [r for r in rows if 'load' in r],
           'does_not_show': ['that a flagged port gives wrong outputs: confirm each with cumlx diff',
                             'ports without the mlx tag, or older than the listed window']}
    args.output.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps({k: doc[k] for k in ('repos_listed', 'configs_read', 'configs_with_rope_parameters')}
                     | {'flagged': len(doc['flagged']), 'not_loadable': len(doc['not_loadable'])}), file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
