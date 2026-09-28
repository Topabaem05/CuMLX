"""Compile the actual CUDA math helpers as HOST C++; not CUDA device validation."""
import ctypes
from pathlib import Path
import shutil
import subprocess
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def helper(tmp_path_factory):
    cxx=shutil.which('g++') or shutil.which('clang++')
    if not cxx: pytest.skip('C++ host compiler unavailable')
    path=tmp_path_factory.mktemp('host_helpers')
    header=(ROOT/'src/cumlx/kernels/common_cuda.h').as_posix()
    (path/'test.cpp').write_text(f'''
#define __device__
#define __forceinline__ inline
#include "{header}"
extern "C" float dec(unsigned int c,int fmt){{
    return fmt==0?cumlx_e2m1(c):(fmt==1?cumlx_e4m3(c):cumlx_e8m0(c));
}}
extern "C" unsigned int enc(float x,int fmt){{
    return fmt==0?cumlx_encode_e2m1(x):(fmt==1?cumlx_encode_e4m3(x):cumlx_encode_e8m0_ceil(x));
}}
''')
    proc=subprocess.run([cxx,'-std=c++17','-O2','-fPIC','-shared',str(path/'test.cpp'),'-o',str(path/'helpers.so')],capture_output=True,text=True)
    assert proc.returncode==0,proc.stderr
    lib=ctypes.CDLL(str(path/'helpers.so'))
    lib.dec.argtypes=[ctypes.c_uint,ctypes.c_int]; lib.dec.restype=ctypes.c_float
    lib.enc.argtypes=[ctypes.c_float,ctypes.c_int]; lib.enc.restype=ctypes.c_uint
    return lib

@pytest.mark.parametrize('fmt,size',[(0,16),(1,127),(2,255)])
def test_host_shader_decode_exhaustive(helper,fmt,size):
    from cumlx.quant import decode_e2m1,decode_e4m3_scale,decode_e8m0_scale
    expected=[decode_e2m1,decode_e4m3_scale,decode_e8m0_scale][fmt](np.arange(size,dtype=np.uint8))
    actual=np.array([helper.dec(c,fmt) for c in range(size)],np.float32)
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(np.signbit(actual),np.signbit(expected))

@pytest.mark.parametrize('fmt',[0,1])
def test_host_shader_encode_midpoints(helper,fmt):
    from cumlx.quant import decode_e2m1,decode_e4m3_scale,encode_e2m1,encode_e4m3_scale
    table=(decode_e2m1(np.arange(8,dtype=np.uint8)) if fmt==0 else decode_e4m3_scale(np.arange(127,dtype=np.uint8)))
    mid=(table[1:]+table[:-1])/2
    values=np.concatenate([table,mid,np.nextafter(mid,np.float32(0)),np.nextafter(mid,np.float32(np.inf)),np.array([448,1000],np.float32)])
    if fmt==0: values=np.concatenate([values,-values])
    expected=encode_e2m1(values) if fmt==0 else encode_e4m3_scale(values,nonzero_floor=True)
    actual=np.array([helper.enc(float(v),fmt) for v in values],np.uint8)
    np.testing.assert_array_equal(actual,expected)


def test_host_shader_e8_scale_ceiling(helper):
    for e in range(-126,126):
        value=np.float32(2.**e)
        assert helper.enc(float(value),2)==e+127
        assert helper.enc(float(np.nextafter(value,np.float32(np.inf))),2)==e+128


def test_host_e2m1_extreme_finite_values_saturate(helper):
    v=float(np.finfo(np.float32).max)
    assert helper.enc(v,0)==7
    assert helper.enc(-v,0)==15
