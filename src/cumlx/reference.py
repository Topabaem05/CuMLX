"""High-precision CPU oracles; explicitly requested, never an implicit GPU fallback."""
from __future__ import annotations
import numpy as np
from .quant import FP4Matrix, dequantize, _finite_float


def silu(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(-np.abs(x))
    sigmoid = np.where(x >= 0, 1/(1+e), e/(1+e))
    return x*sigmoid


def _finish(y: np.ndarray, bias: np.ndarray | None, epilogue: str) -> np.ndarray:
    if bias is not None:
        b=_finite_float(bias,"bias")
        if b.shape != (y.shape[-1],): raise ValueError("bias must have shape (N,)")
        y=y+b
    if epilogue == "silu": y=silu(y)
    elif epilogue != "none": raise ValueError("epilogue must be none or silu")
    return y.astype(np.float32)


def linear(x: np.ndarray, w: FP4Matrix, *, bias: np.ndarray | None = None, epilogue: str = "none") -> np.ndarray:
    """FP16/FP32 inputs and decoded weights, float64 oracle dot, FP32 output."""
    a=_finite_float(x,"x")
    if a.ndim!=2 or min(a.shape)<=0 or a.shape[1]!=w.shape[1]: raise ValueError("x must be (M,K), M>0")
    y=a.astype(np.float64) @ dequantize(w).astype(np.float64).T
    return _finish(y,bias,epilogue)


def linear_fp4(a: FP4Matrix, w: FP4Matrix, *, bias: np.ndarray | None = None, epilogue: str = "none") -> np.ndarray:
    if a.shape[1]!=w.shape[1]: raise ValueError("packed operands must share K")
    y=dequantize(a).astype(np.float64) @ dequantize(w).astype(np.float64).T
    return _finish(y,bias,epilogue)


def silu_mul(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    g,u=_finite_float(gate,"gate"),_finite_float(up,"up")
    if g.shape!=u.shape: raise ValueError("gate/up shapes must match")
    return (silu(g)*u).astype(np.float32)


def rms_norm(x: np.ndarray, weight: np.ndarray, *, eps: float = 1e-5, residual: np.ndarray | None = None) -> np.ndarray:
    a=_finite_float(x,"x").astype(np.float64); w=_finite_float(weight,"weight")
    if a.ndim!=2 or min(a.shape)<=0 or w.shape!=(a.shape[-1],): raise ValueError("expected x[M,K] and weight[K]")
    if not np.isfinite(eps) or eps<=0: raise ValueError("eps must be finite and positive")
    if residual is not None:
        r=_finite_float(residual,"residual")
        if r.shape!=a.shape: raise ValueError("residual shape mismatch")
        # Kernel forms the residual sum in float32, not input dtype.
        a=(a+r).astype(np.float32).astype(np.float64)
    return (a/np.sqrt(np.mean(a*a,axis=-1,keepdims=True)+eps)*w).astype(np.float32)
