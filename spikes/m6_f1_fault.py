#!/usr/bin/env python3
"""Copy an MLX port with one weight scaled by 1.01. cumlx diff must reject the copy, or a pass of the right port says nothing."""
import argparse
from pathlib import Path
import shutil

import mlx.core as mx

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('port', type=Path)
parser.add_argument('copy', type=Path)
args = parser.parse_args()
shutil.copytree(args.port, args.copy)
files = sorted(args.copy.glob('*.safetensors'))
weights, metadata = mx.load(str(files[0]), return_metadata=True)
name = next(key for key in sorted(weights) if key.endswith('mlp.up_proj.weight'))
weights[name] = weights[name] * 1.01
mx.save_safetensors(str(files[0]), weights, metadata=metadata)
print(name)
