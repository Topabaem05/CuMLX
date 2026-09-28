"""Canonical FP4 codecs. This module is CPU reference/packing code, not GPU code.

Only rowwise NVFP4 and MXFP4 are accepted. External swizzled layouts, NF4,
NxFP4 and TransformerEngine training recipes require separate adapters.
All scales are reconstruction multipliers, never inverse scales.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import hmac
import json
from pathlib import Path
import zipfile
from typing import Literal
import numpy as np

Format = Literal["nvfp4", "mxfp4"]
LAYOUT = "row_major_k_low_nibble_v1"
CODEC = "cumlx.e2m1.v1"
RECIPE = "cumlx.maxabs.rne.rowwise.v1"
_MAG = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)


def normalize_format(fmt: str) -> Format:
    if not isinstance(fmt, str) or fmt.lower() not in ("nvfp4", "mxfp4"):
        raise ValueError(f"unsupported FP4 format {fmt!r}; only nvfp4/mxfp4; NxFP4 and NF4 are not aliases")
    return fmt.lower()  # type: ignore[return-value]


def group_size(fmt: str) -> int:
    return 16 if normalize_format(fmt) == "nvfp4" else 32


def _uint8(x: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(x)
    if a.dtype != np.uint8:
        raise TypeError(f"{name} must be uint8, not {a.dtype}")
    return a


def _finite_float(x: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(x)
    if a.dtype.kind != "f":
        raise TypeError(f"{name} must have a floating dtype")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains NaN or infinity")
    if np.any(np.abs(a.astype(np.float64)) > np.finfo(np.float32).max):
        raise ValueError(f"{name} exceeds float32 range")
    return a.astype(np.float32, copy=False)


def decode_e2m1(codes: np.ndarray) -> np.ndarray:
    c = _uint8(codes, "codes")
    if np.any(c > 15):
        raise ValueError("E2M1 codes must be in 0..15")
    return np.copysign(_MAG[c & 7], np.where(c & 8, np.float32(-1), np.float32(1)))


def _nearest_even_positive(x: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Nearest finite code; exact midpoint selects an even code LSB."""
    hi = np.searchsorted(table, x, side="left").clip(0, len(table)-1)
    lo = np.maximum(hi - 1, 0)
    dh = np.abs(table[hi].astype(np.float64) - x.astype(np.float64))
    dl = np.abs(table[lo].astype(np.float64) - x.astype(np.float64))
    choose_hi = (dh < dl) | ((dh == dl) & ((hi & 1) == 0))
    return np.where(choose_hi, hi, lo).astype(np.uint8)


def encode_e2m1(values: np.ndarray) -> np.ndarray:
    x = _finite_float(values, "values")
    code = _nearest_even_positive(np.minimum(np.abs(x), np.float32(6)), _MAG)
    return code | (np.signbit(x).astype(np.uint8) << 3)


def pack_nibbles(codes: np.ndarray) -> np.ndarray:
    c = _uint8(codes, "codes")
    if c.ndim < 1 or c.shape[-1] <= 0 or np.any(c > 15):
        raise ValueError("codes need a nonempty last axis and values in 0..15")
    out = c[..., ::2].copy()
    count = c.shape[-1] // 2
    out[..., :count] |= c[..., 1::2] << 4
    return out


def unpack_nibbles(payload: np.ndarray, k: int) -> np.ndarray:
    p = _uint8(payload, "payload")
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k <= 0:
        raise ValueError("k must be a positive integer")
    if p.ndim < 1 or p.shape[-1] != (k + 1)//2:
        raise ValueError("payload shape does not match k")
    out = np.empty(p.shape[:-1] + (p.shape[-1]*2,), np.uint8)
    out[..., ::2], out[..., 1::2] = p & 15, p >> 4
    return out[..., :k]


def decode_e4m3_scale(codes: np.ndarray) -> np.ndarray:
    """Unsigned finite E4M3FN scale subset: raw codes 0..126, max 448."""
    c = _uint8(codes, "E4M3 scales")
    if np.any(c > 126):
        raise ValueError("E4M3 scale must be nonnegative finite (codes 0..126)")
    e, m = (c >> 3).astype(np.int32), (c & 7).astype(np.float32)
    return np.where(e == 0, m * np.float32(2**-9), np.ldexp(1 + m/8, e-7)).astype(np.float32)


def decode_e8m0_scale(codes: np.ndarray) -> np.ndarray:
    c = _uint8(codes, "E8M0 scales")
    if np.any(c == 255):
        raise ValueError("E8M0 scale code 255 is NaN")
    return np.exp2(c.astype(np.float64)-127).astype(np.float32)


_E4_TABLE = decode_e4m3_scale(np.arange(127, dtype=np.uint8))


def encode_e4m3_scale(values: np.ndarray, *, nonzero_floor: bool = False) -> np.ndarray:
    x = _finite_float(values, "scale")
    if np.any(x < 0):
        raise ValueError("scale cannot be negative")
    c = _nearest_even_positive(x, _E4_TABLE)
    if nonzero_floor:
        c = np.where((x > 0) & (c == 0), 1, c).astype(np.uint8)
    return c


def _immutable(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a)
    # A bytes owner prevents setflags(write=True), unlike a mutable owning array.
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


@dataclass(frozen=True, eq=False)
class FP4Matrix:
    format: Format
    shape: tuple[int, int]
    payload: np.ndarray = field(repr=False)
    scales: np.ndarray = field(repr=False)
    global_scale: float = 1.0
    layout: str = LAYOUT
    codec: str = CODEC
    checksum: str = field(init=False)

    def __post_init__(self) -> None:
        fmt = normalize_format(self.format)
        if self.layout != LAYOUT or self.codec != CODEC:
            raise ValueError("unsupported layout or codec; normalize in an explicit producer adapter")
        if len(self.shape) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v <= 0 for v in self.shape):
            raise ValueError("shape must be positive (N,K)")
        n, k = (int(v) for v in self.shape)
        if n*k > 2**31-1:
            raise ValueError("this release limits logical element counts to signed int32")
        p, s = _uint8(self.payload, "payload"), _uint8(self.scales, "scales")
        g = group_size(fmt)
        if p.shape != (n, (k+1)//2) or s.shape != (n, (k+g-1)//g):
            raise ValueError("canonical payload/scales shapes do not match logical shape")
        if k % 2 and np.any(p[:, -1] >> 4):
            raise ValueError("unused high-nibble padding must be zero")
        gs = float(self.global_scale)
        if not np.isfinite(gs) or gs <= 0 or gs > np.finfo(np.float32).max:
            raise ValueError("global_scale must be a positive finite float32 multiplier")
        gs32 = np.float32(gs)
        if gs32 <= 0:
            raise ValueError("global_scale underflows float32")
        if fmt == "mxfp4" and gs32 != 1:
            raise ValueError("MXFP4 has no extra tensor scale; global_scale must be 1")
        bs = decode_e4m3_scale(s) if fmt == "nvfp4" else decode_e8m0_scale(s)
        # Inspect only blocks with huge scales; do not inflate ordinary weights.
        possible = bs.astype(np.float64) * float(gs32) * 6 > np.finfo(np.float32).max
        for row, block in np.argwhere(possible):
            lo, hi = int(block)*g, min((int(block)+1)*g, k)
            vals = decode_e2m1(unpack_nibbles(p[row:row+1], k)[0, lo:hi])
            if np.any(np.abs(vals.astype(np.float64))*float(bs[row,block])*float(gs32) > np.finfo(np.float32).max):
                raise ValueError("reconstructed weights exceed float32 range")
        object.__setattr__(self, "format", fmt)
        object.__setattr__(self, "shape", (n,k))
        object.__setattr__(self, "global_scale", float(gs32))
        object.__setattr__(self, "payload", _immutable(p))
        object.__setattr__(self, "scales", _immutable(s))
        meta = json.dumps(self.metadata(), sort_keys=True, separators=(",", ":")).encode()
        object.__setattr__(self, "checksum", sha256(meta + p.tobytes(order="C") + s.tobytes(order="C")).hexdigest())

    @property
    def group_size(self) -> int:
        return group_size(self.format)

    @property
    def storage_nbytes(self) -> int:
        return self.payload.nbytes + self.scales.nbytes + (4 if self.format == "nvfp4" else 0)

    def metadata(self) -> dict:
        return {"version": 1, "format": self.format, "shape": list(self.shape), "global_scale": self.global_scale,
                "layout": self.layout, "codec": self.codec}


def quantize(weight: np.ndarray, fmt: str = "nvfp4", *, global_scale: float | None = None) -> FP4Matrix:
    """Deterministic CUMLX max-abs/RNE rowwise recipe, NOT the TE training recipe.

    NVFP4 uses maxabs/(6*448) as the automatic tensor reconstruction scale,
    rounded to FP32 and floored at FP32 tiny. E4M3 nonzero block scales are
    floored to their smallest positive code. MXFP4 chooses a ceiling power-of-2
    scale for blockmax/6. Incomplete final blocks are zero padded.
    """
    fmt = normalize_format(fmt)
    x = _finite_float(weight, "weight")
    if x.ndim != 2 or min(x.shape) <= 0:
        raise ValueError("weight must be a nonempty 2D matrix")
    n,k = x.shape; g = group_size(fmt); ng = (k+g-1)//g
    if global_scale is None:
        gs = max(float(np.max(np.abs(x))) / 2688.0, float(np.finfo(np.float32).tiny)) if fmt == "nvfp4" else 1.0
        if not np.any(x): gs = 1.0
    else:
        gs = float(global_scale)
    if not np.isfinite(gs) or gs <= 0 or gs > np.finfo(np.float32).max or (fmt == "mxfp4" and gs != 1):
        raise ValueError("invalid global_scale for format")
    gs32 = np.float32(gs)
    if gs32 <= 0: raise ValueError("global_scale underflows float32")
    codes = np.zeros((n,k),np.uint8); scales = np.empty((n,ng),np.uint8)
    # Process one block column at a time, keeping auxiliary memory O(N*G).
    for b in range(ng):
        lo,hi = b*g,min((b+1)*g,k)
        block = x[:,lo:hi]
        amax = np.max(np.abs(block),axis=1)
        # FP32 operation order is shared with the device encoder. Overflow in
        # normalization is intentional saturation, never an unbounded cast.
        with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
            if fmt == "nvfp4":
                raw = np.clip((amax / gs32) / np.float32(6), 0, 448)
                sc = encode_e4m3_scale(raw, nonzero_floor=True)
                bs = decode_e4m3_scale(sc)
            else:
                ratio = amax / np.float32(6)
                table = decode_e8m0_scale(np.arange(255, dtype=np.uint8))
                sc = np.searchsorted(table, ratio, side="left").clip(0,254).astype(np.uint8)
                sc = np.where(amax == 0, 127, sc).astype(np.uint8)
                bs = decode_e8m0_scale(sc)
            scaled = block / gs32
            normalized = np.zeros_like(block)
            np.divide(scaled,bs[:,None],out=normalized,where=bs[:,None]!=0)
            normalized = np.clip(normalized,-6,6).astype(np.float32)
        normalized = np.copysign(normalized, block)
        codes[:,lo:hi]=encode_e2m1(normalized); scales[:,b]=sc
    return FP4Matrix(fmt,(n,k),pack_nibbles(codes),scales,float(gs32))


def dequantize(q: FP4Matrix) -> np.ndarray:
    """Full CPU diagnostic reconstruction. GPU fast paths do NOT call this."""
    values = decode_e2m1(unpack_nibbles(q.payload,q.shape[1]))
    bs = decode_e4m3_scale(q.scales) if q.format == "nvfp4" else decode_e8m0_scale(q.scales)
    scale = np.repeat(bs,q.group_size,axis=1)[:,:q.shape[1]]
    # Same explicit multiplication order as the kernel helper.
    with np.errstate(over="raise",invalid="raise"):
        values = (values*scale).astype(np.float32) * np.float32(q.global_scale)
    return values.astype(np.float32)


def save_fp4(path: str | Path, q: FP4Matrix) -> None:
    meta=q.metadata(); meta["checksum"]=q.checksum
    with Path(path).open("wb") as f:
        np.savez(f,payload=q.payload,scales=q.scales,metadata=np.array(json.dumps(meta,sort_keys=True)))


def load_fp4(path: str | Path, *, max_bytes: int = 8*1024**3) -> FP4Matrix:
    """Load only the CUMLX container; never load pickle or arbitrary vendor layouts."""
    with zipfile.ZipFile(path) as z:
        if sum(i.file_size for i in z.infolist()) > max_bytes:
            raise ValueError("FP4 container exceeds uncompressed size limit")
        if sorted(z.namelist()) != ["metadata.npy","payload.npy","scales.npy"]:
            raise ValueError("unexpected container members")
    with np.load(path,allow_pickle=False) as d:
        meta=json.loads(str(d["metadata"].item()))
        keys={"version","format","shape","global_scale","layout","codec","checksum"}
        if set(meta)!=keys or meta["version"]!=1: raise ValueError("unsupported metadata schema")
        q=FP4Matrix(meta["format"],tuple(meta["shape"]),d["payload"],d["scales"],meta["global_scale"],meta["layout"],meta["codec"])
        if not isinstance(meta["checksum"],str) or not hmac.compare_digest(meta["checksum"],q.checksum):
            raise ValueError("FP4 checksum mismatch")
        return q
