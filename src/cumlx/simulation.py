"""CPU launch-schedule simulations; NOT execution, compilation, or emulation of GPU source.

They independently check output coverage, pair packing, tail masking, and tree
reduction layouts against the high-precision oracle. GPU integration is separate.
"""
from __future__ import annotations
import numpy as np
from .planning import plan_linear
from .quant import FP4Matrix, dequantize


def simulate_linear(x: np.ndarray, w: FP4Matrix, *, algorithm: str='auto') -> tuple[np.ndarray,np.ndarray]:
    x=np.asarray(x,np.float32); m,k=x.shape; n=w.shape[0]
    if k!=w.shape[1]: raise ValueError('K mismatch')
    plan=plan_linear(m,n,k,algorithm=algorithm)
    # Diagnostic-only dense buffer; the device kernels do not use this simulation.
    dense=dequantize(w)
    y=np.full((m,n),np.nan,np.float32); coverage=np.zeros((m,n),np.uint16)
    if plan.algorithm in ('gemv','small_m'):
        bm=plan.tile_m
        for group in range(plan.grid[0]//128):
            col=group%n; m0=(group//n)*bm
            scratch=np.zeros((bm,128),np.float32)
            for lane in range(128):
                for kk in range(2*lane,k,256):
                    for row in range(bm):
                        if m0+row<m:
                            scratch[row,lane]+=np.float32(x[m0+row,kk]*dense[col,kk])
                            if kk+1<k:
                                scratch[row,lane]+=np.float32(x[m0+row,kk+1]*dense[col,kk+1])
            for step in (64,32,16,8,4,2,1):
                scratch[:,:step]+=scratch[:,step:2*step]
            for row in range(bm):
                if m0+row<m:
                    y[m0+row,col]=scratch[row,0]
                    coverage[m0+row,col]+=1
    else:
        tn=(n+31)//32
        for group in range(plan.grid[0]//128):
            m0=(group//tn)*8; n0=(group%tn)*32
            am=min(8,m-m0); an=min(32,n-n0)
            accum=np.zeros((8,32),np.float32)
            for k0 in range(0,k,32):
                ak=min(32,k-k0)
                a=np.zeros((8,32),np.float32); b=np.zeros((32,32),np.float32)
                a[:am,:ak]=x[m0:m0+am,k0:k0+ak]
                b[:an,:ak]=dense[n0:n0+an,k0:k0+ak]
                for kk in range(32):
                    accum+=a[:,kk,None]*b[None,:,kk]
            # A 128-lane group, each lane owns two columns of one row.
            for lane in range(128):
                r,c=lane//16,lane%16
                for cc in (c,c+16):
                    if m0+r<m and n0+cc<n:
                        y[m0+r,n0+cc]=accum[r,cc]
                        coverage[m0+r,n0+cc]+=1
    return y,coverage
