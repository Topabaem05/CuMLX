import importlib
import numpy as np
import pytest


def modules():
    assert importlib.util.find_spec("cumlx.reference") is not None, "reference not implemented"
    return importlib.import_module("cumlx.quant"), importlib.import_module("cumlx.reference")


@pytest.mark.parametrize("fmt",["nvfp4","mxfp4"])
@pytest.mark.parametrize("m,n,k",[(1,1,1),(1,7,33),(2,17,65),(4,31,16),(7,33,67),(11,65,127)])
def test_linear_reference(fmt,m,n,k):
    q,r = modules()
    rng=np.random.default_rng(42)
    x=rng.normal(size=(m,k)).astype(np.float16)
    w=q.quantize(rng.normal(size=(n,k)).astype(np.float32),fmt)
    bias=rng.normal(size=n).astype(np.float32)
    actual=r.linear(x,w,bias=bias)
    expected=(x.astype(np.float64)@q.dequantize(w).astype(np.float64).T+bias).astype(np.float32)
    np.testing.assert_allclose(actual,expected,atol=3e-5,rtol=3e-5)


def test_w4a4_is_not_w4a16():
    q,r=modules(); rng=np.random.default_rng(8)
    x=rng.normal(size=(5,33)).astype(np.float32)
    w=q.quantize(rng.normal(size=(7,33)).astype(np.float32),'nvfp4')
    a=q.quantize(x,'nvfp4')
    actual=r.linear_fp4(a,w)
    np.testing.assert_allclose(actual,q.dequantize(a)@q.dequantize(w).T,atol=2e-5,rtol=2e-5)
    assert not np.allclose(actual,r.linear(x,w))


def test_stable_silu_and_rms_norm():
    _,r=modules()
    gate=np.array([[-1000,0,1,1000]],np.float32)
    out=r.silu_mul(gate,np.ones_like(gate))
    assert np.isfinite(out).all() and out[0,0]==0 and out[0,-1]==1000
    x=np.array([[3,4,0]],np.float32); w=np.array([1,2,3],np.float32)
    expected=x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+1e-5)*w
    np.testing.assert_allclose(r.rms_norm(x,w,eps=1e-5),expected,rtol=1e-6)


def test_shape_and_epilogue_failures():
    q,r=modules(); w=q.quantize(np.ones((3,16),np.float32),'nvfp4')
    for x,bias,epilogue in [(np.ones((2,15),np.float32),None,'none'),(np.ones((2,16),np.float32),np.ones(2,np.float32),'none'),(np.ones((2,16),np.float32),None,'unknown')]:
        with pytest.raises((ValueError,TypeError)):
            r.linear(x,w,bias=bias,epilogue=epilogue)
