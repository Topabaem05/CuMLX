"""These tests must run on physical hardware before any DEVICE_VALIDATED claim."""
import numpy as np
import pytest

pytestmark=pytest.mark.gpu


@pytest.mark.parametrize('fmt',['nvfp4','mxfp4'])
@pytest.mark.parametrize('m,n,k',[(1,1,1),(1,7,17),(1,33,65),(2,17,31),(3,33,67),(7,5,9),(9,35,63),(17,65,129)])
def test_gpu_linear(gpu,fmt,m,n,k):
    from cumlx import quantize, reference
    mx=gpu.mx; rng=np.random.default_rng(33)
    x=rng.normal(size=(m,k)).astype(np.float32)
    w=quantize(rng.normal(size=(n,k)).astype(np.float32),fmt)
    bias=rng.normal(size=n).astype(np.float32)
    qw=gpu.prepare(w)
    for algorithm in ['auto','tiled']:
        y=gpu.linear(mx.array(x),qw,bias=mx.array(bias),algorithm=algorithm)
        mx.eval(y)
        np.testing.assert_allclose(np.array(y),reference.linear(x,w,bias=bias),atol=2e-4,rtol=3e-4)


@pytest.mark.parametrize('fmt',['nvfp4','mxfp4'])
def test_gpu_decode_all_bytes(gpu,fmt):
    from cumlx import FP4Matrix, dequantize
    mx=gpu.mx
    payload=np.arange(256,dtype=np.uint8)[:,None]
    scales=np.full((256,1),64 if fmt=='nvfp4' else 128,np.uint8)
    q=FP4Matrix(fmt,(256,2),payload,scales,.25 if fmt=='nvfp4' else 1.)
    y=gpu.decode(gpu.prepare(q)); mx.eval(y)
    np.testing.assert_array_equal(np.array(y),dequantize(q))
    np.testing.assert_array_equal(np.signbit(np.array(y)),np.signbit(dequantize(q)))


@pytest.mark.parametrize('af,wf',[('nvfp4','nvfp4'),('mxfp4','mxfp4'),('nvfp4','mxfp4')])
def test_gpu_w4a4(gpu,af,wf):
    from cumlx import quantize,reference
    rng=np.random.default_rng(8); mx=gpu.mx
    a=quantize(rng.normal(size=(9,67)).astype(np.float32),af)
    w=quantize(rng.normal(size=(35,67)).astype(np.float32),wf)
    y=gpu.linear_fp4(gpu.prepare(a),gpu.prepare(w)); mx.eval(y)
    np.testing.assert_allclose(np.array(y),reference.linear_fp4(a,w),atol=2e-4,rtol=3e-4)


@pytest.mark.parametrize('fmt',['nvfp4','mxfp4'])
def test_gpu_quantizer(gpu,fmt):
    from cumlx import quantize,FP4Matrix,dequantize
    mx=gpu.mx; rng=np.random.default_rng(92)
    a=rng.normal(size=(3,67)).astype(np.float32)
    # Pinned global scale separates encoder parity from global-amax rounding.
    gs=.125 if fmt=='nvfp4' else 1.
    expected=quantize(a,fmt,global_scale=gs)
    actual=gpu.quantize(mx.array(a),fmt,global_scale=gs)
    mx.eval(actual.payload,actual.scales,actual.global_scale)
    np.testing.assert_array_equal(np.array(actual.payload),expected.payload)
    np.testing.assert_array_equal(np.array(actual.scales),expected.scales)
    decoded=gpu.decode(actual); mx.eval(decoded)
    np.testing.assert_allclose(np.array(decoded),dequantize(expected),atol=0,rtol=0)


@pytest.mark.parametrize('dtype',['float16','float32','bfloat16'])
def test_gpu_fused_ops(gpu,dtype):
    from cumlx import reference
    mx=gpu.mx; rng=np.random.default_rng(12)
    x=mx.array(rng.normal(size=(3,65)).astype(np.float32)).astype(getattr(mx,dtype))
    u=mx.array(rng.normal(size=(3,65)).astype(np.float32)).astype(getattr(mx,dtype))
    w=mx.array(rng.normal(size=65).astype(np.float32))
    # Use actually rounded operands in the oracle, especially for BF16.
    xx=np.array(x.astype(mx.float32)); uu=np.array(u.astype(mx.float32))
    y=gpu.silu_mul(x,u); z=gpu.rms_norm(x,w,residual=u,eps=1e-5)
    mx.eval(y,z)
    np.testing.assert_allclose(np.array(y),reference.silu_mul(xx,uu),atol=2e-5,rtol=2e-5)
    np.testing.assert_allclose(np.array(z),reference.rms_norm(xx,np.array(w),residual=uu,eps=1e-5),atol=3e-5,rtol=3e-5)


def test_gpu_noncontiguous_inputs(gpu):
    from cumlx import quantize,reference
    mx=gpu.mx; rng=np.random.default_rng(19)
    data=rng.normal(size=(17,3)).astype(np.float32)
    x=mx.array(data).T
    w=quantize(rng.normal(size=(7,17)).astype(np.float32),'nvfp4')
    y=gpu.linear(x,gpu.prepare(w)); mx.eval(y)
    np.testing.assert_allclose(np.array(y),reference.linear(data.T,w),atol=3e-5,rtol=3e-5)


def test_gpu_bias_silu_epilogue(gpu):
    from cumlx import quantize,reference
    mx=gpu.mx; rng=np.random.default_rng(191)
    x=rng.normal(size=(7,33)).astype(np.float32)
    w=quantize(rng.normal(size=(11,33)).astype(np.float32),'nvfp4')
    bias=rng.normal(size=11).astype(np.float32)
    for algorithm in ['small_m','tiled']:
        y=gpu.linear(mx.array(x),gpu.prepare(w),bias=mx.array(bias),epilogue='silu',algorithm=algorithm)
        mx.eval(y)
        np.testing.assert_allclose(np.array(y),reference.linear(x,w,bias=bias,epilogue='silu'),atol=1e-4,rtol=1e-4)


def test_gpu_failures_are_explicit(gpu):
    from cumlx import quantize
    mx=gpu.mx; q=gpu.prepare(quantize(np.ones((3,17),np.float32)))
    with pytest.raises(ValueError): gpu.linear(mx.ones((1,16)),q)
    with pytest.raises(TypeError): gpu.linear(mx.ones((1,17),dtype=mx.int32),q)
    with pytest.raises(ValueError): gpu.quantize(mx.array([[float('nan')]]),'nvfp4')
    with pytest.raises(ValueError): gpu.quantize(mx.ones((1,16)),'nxfp4')


@pytest.mark.parametrize('dtype',['float16','bfloat16'])
@pytest.mark.parametrize('algorithm',['auto','tiled'])
def test_gpu_linear_low_precision_inputs(gpu,dtype,algorithm):
    from cumlx import quantize,reference
    mx=gpu.mx; rng=np.random.default_rng(110)
    x=mx.array(rng.normal(size=(3,67)).astype(np.float32)).astype(getattr(mx,dtype))
    w=quantize(rng.normal(size=(35,67)).astype(np.float32))
    actual=np.array(x.astype(mx.float32))
    y=gpu.linear(x,gpu.prepare(w),algorithm=algorithm); mx.eval(y)
    np.testing.assert_allclose(np.array(y),reference.linear(actual,w),atol=2e-4,rtol=3e-4)


@pytest.mark.parametrize('fmt',['nvfp4','mxfp4'])
def test_gpu_zero_encoder_padding_and_signed_zero(gpu,fmt):
    mx=gpu.mx
    x=np.zeros((2,17),np.float32); x[1]=-x[1]
    q=gpu.quantize(mx.array(x),fmt); y=gpu.decode(q); mx.eval(y,q.payload)
    np.testing.assert_array_equal(np.array(y),x)
    np.testing.assert_array_equal(np.signbit(np.array(y)),np.signbit(x))
    assert np.all(np.array(q.payload)[:,-1]>>4==0)
