import importlib
import numpy as np
import pytest


def modules():
    assert importlib.util.find_spec('cumlx.planning') is not None, 'kernel planner not implemented'
    return importlib.import_module('cumlx.planning')


@pytest.mark.parametrize('m,name,bm',[(1,'gemv',1),(2,'small_m',4),(4,'small_m',4),(8,'small_m',4),(9,'tiled',8),(64,'tiled',8)])
def test_dispatch(m,name,bm):
    p=modules().plan_linear(m,33,67)
    assert p.algorithm==name and p.tile_m==bm
    assert p.threadgroup==(128,1,1) and p.grid[0]%128==0
    assert p.accumulation=='float32'
    assert p.full_weight_dequantization is False
    assert p.grid[0]>0


@pytest.mark.parametrize('shape',[(0,3,4),(1,0,4),(1,3,0),(-1,3,4),(True,3,4),(1.2,3,4),(2**31,1,1)])
def test_invalid_shape(shape):
    with pytest.raises((ValueError,TypeError)):
        modules().plan_linear(*shape)


def test_wrong_algorithm():
    p=modules()
    with pytest.raises(ValueError): p.plan_linear(2,3,4,algorithm='gemv')
    with pytest.raises(ValueError): p.plan_linear(1,3,4,algorithm='tensorcore')


@pytest.mark.parametrize('m,n,k',[(1,1,1),(1,5,17),(2,3,33),(3,7,67),(4,5,31),(7,3,9),(9,33,35)])
@pytest.mark.parametrize('fmt',['nvfp4','mxfp4'])
def test_kernel_schedule_simulator(m,n,k,fmt):
    p=modules()
    assert importlib.util.find_spec('cumlx.simulation') is not None, 'schedule simulator not implemented'
    from cumlx.simulation import simulate_linear
    from cumlx.quant import quantize
    from cumlx.reference import linear
    rng=np.random.default_rng(17)
    x=rng.normal(size=(m,k)).astype(np.float32)
    w=quantize(rng.normal(size=(n,k)).astype(np.float32),fmt)
    for alg in [p.plan_linear(m,n,k).algorithm,'tiled']:
        y,coverage=simulate_linear(x,w,algorithm=alg)
        np.testing.assert_array_equal(coverage,np.ones((m,n),np.uint16))
        np.testing.assert_allclose(y,linear(x,w),atol=5e-5,rtol=5e-5)
