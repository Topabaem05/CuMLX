"""Agent-friendly command line for the CUMLX porting verifier. See ``cumlx describe``."""
from .main import build_parser, entrypoint, main

__all__ = ['build_parser', 'entrypoint', 'main']
