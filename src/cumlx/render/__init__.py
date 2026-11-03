"""Render sensor images on Apple Silicon with Mitsuba on Metal. `compare` needs only NumPy."""
from .compare import compare_images, psnr

__all__ = ['compare_images', 'psnr']
