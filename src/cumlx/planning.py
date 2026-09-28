"""Deterministic launch geometry. These are conservative candidates, not tuned winners."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numbers


@dataclass(frozen=True)
class KernelPlan:
    algorithm: str
    m: int
    n: int
    k: int
    tile_m: int
    tile_n: int
    tile_k: int
    grid: tuple[int,int,int]
    threadgroup: tuple[int,int,int]
    threadgroup_memory_bytes: int
    accumulation: str = 'float32'
    full_weight_dequantization: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


def plan_linear(m: int, n: int, k: int, *, algorithm: str = 'auto') -> KernelPlan:
    dims=(m,n,k)
    if any(isinstance(v,bool) or not isinstance(v,numbers.Integral) or v<=0 for v in dims):
        raise ValueError('M, N, K must be positive integers')
    m,n,k=map(int,dims)
    if max(m*n,m*k,n*k)>2**31-1:
        raise ValueError('logical element counts exceed the int32 kernel profile')
    if algorithm=='auto': algorithm='gemv' if m==1 else ('small_m' if m<=8 else 'tiled')
    if algorithm not in ('gemv','small_m','tiled'): raise ValueError('unknown kernel algorithm')
    if algorithm=='gemv' and m!=1: raise ValueError('gemv requires M=1')
    if algorithm=='small_m' and m>8: raise ValueError('small_m profile requires M<=8')
    if algorithm in ('gemv','small_m'):
        bm=1 if algorithm=='gemv' else 4
        groups=n*((m+bm-1)//bm)
        bn,bk,smem=1,256,bm*128*4
    else:
        bm,bn,bk=8,32,32
        groups=((m+bm-1)//bm)*((n+bn-1)//bn)
        smem=(bm*bk+bn*bk)*4
    if groups*128>2**31-1: raise ValueError('launch grid exceeds the int32 profile')
    return KernelPlan(algorithm,m,n,k,bm,bn,bk,(groups*128,1,1),(128,1,1),smem)
