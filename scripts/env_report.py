#!/usr/bin/env python3
"""Report physical capabilities without presenting untested features as supported."""
import json
from cumlx.mlx_backend import environment_report

if __name__ == '__main__':
    print(json.dumps(environment_report(probe=True), indent=2, default=str))
