"""The port under review. CUMLX calls this function and compares the module it returns."""
from cumlx.verify.mlx_port import load_port


def load(path: str):
    return load_port(path)
