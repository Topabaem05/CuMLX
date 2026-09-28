import importlib
import numpy as np
import pytest


def module():
    assert importlib.util.find_spec("cumlx.quant") is not None, "FP4 codec module not implemented"
    return importlib.import_module("cumlx.quant")


def test_all_e2m1_codes():
    q = module()
    codes = np.arange(16, dtype=np.uint8)
    expected = np.array([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6], np.float32)
    out = q.decode_e2m1(codes)
    np.testing.assert_array_equal(out, expected)
    assert np.signbit(out[8]) and not np.signbit(out[0])


def test_all_packed_bytes_roundtrip():
    q = module()
    p = np.arange(256, dtype=np.uint8)[:, None]
    codes = q.unpack_nibbles(p, 2)
    np.testing.assert_array_equal(codes[:, 0], p[:, 0] & 15)
    np.testing.assert_array_equal(codes[:, 1], p[:, 0] >> 4)
    np.testing.assert_array_equal(q.pack_nibbles(codes), p)


@pytest.mark.parametrize("k", [1, 3, 15, 16, 17, 31, 32, 33, 127])
def test_odd_tail_packing(k):
    q = module()
    codes = np.random.default_rng(k).integers(0, 16, (3, k), dtype=np.uint8)
    p = q.pack_nibbles(codes)
    assert p.shape == (3, (k + 1)//2)
    np.testing.assert_array_equal(q.unpack_nibbles(p, k), codes)
    if k % 2:
        assert np.all(p[:, -1] >> 4 == 0)


def test_rne_midpoints_and_saturation():
    q = module()
    x = np.array([.25,.75,1.25,1.75,2.5,3.5,5.,100.,-0.,-.25], np.float32)
    expected = np.array([0,2,2,4,4,6,6,7,8,8], np.uint8)
    np.testing.assert_array_equal(q.encode_e2m1(x), expected)


def test_scale_known_values():
    q = module()
    e4 = np.array([0,1,7,8,56,64,119,126],np.uint8)
    exp = np.array([0,2**-9,7*2**-9,2**-6,1,2,240,448],np.float32)
    np.testing.assert_array_equal(q.decode_e4m3_scale(e4), exp)
    e8 = np.array([0,1,126,127,128,254],np.uint8)
    np.testing.assert_array_equal(q.decode_e8m0_scale(e8), np.exp2(e8.astype(np.float64)-127).astype(np.float32))


@pytest.mark.parametrize("code", [127,128,255])
def test_bad_nv_scales(code):
    with pytest.raises(ValueError):
        module().decode_e4m3_scale(np.array([code],np.uint8))


def test_bad_mx_scale():
    with pytest.raises(ValueError):
        module().decode_e8m0_scale(np.array([255],np.uint8))


@pytest.mark.parametrize("fmt,g", [("nvfp4",16),("mxfp4",32)])
@pytest.mark.parametrize("k", [1, 15, 16, 17, 31, 32, 33, 65])
def test_quantize_roundtrip_and_storage(fmt,g,k):
    q = module()
    w = np.random.default_rng(k).normal(size=(3,k)).astype(np.float32)
    packed = q.quantize(w, fmt)
    decoded = q.dequantize(packed)
    assert decoded.shape == w.shape and decoded.dtype == np.float32
    assert packed.payload.shape == (3,(k+1)//2)
    assert packed.scales.shape == (3,(k+g-1)//g)
    assert packed.storage_nbytes == packed.payload.nbytes + packed.scales.nbytes + (4 if fmt == "nvfp4" else 0)
    assert np.linalg.norm(decoded-w)/np.linalg.norm(w) < .3
    assert not packed.payload.flags.writeable
    with pytest.raises(ValueError):
        packed.payload.setflags(write=True)


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp4"])
def test_zero_tensor(fmt):
    q = module()
    p = q.quantize(np.zeros((2,17),np.float32),fmt)
    assert np.all(q.dequantize(p)==0)
    assert np.isfinite(p.global_scale) and p.global_scale > 0


def test_nonunit_scale_not_dropped():
    q = module()
    codes = np.array([[2,3,4,5,6,7,0,8,9,10,11,12,13,14,15,1]],np.uint8)
    w = q.FP4Matrix("nvfp4",(1,16),q.pack_nibbles(codes),np.array([[64]],np.uint8),np.float32(.25))
    np.testing.assert_array_equal(q.dequantize(w),q.decode_e2m1(codes)*np.float32(.5))


@pytest.mark.parametrize("fmt", ["NXFP4","nxfp4","nf4","int4","suno"])
def test_unknown_formats_are_not_aliased(fmt):
    with pytest.raises(ValueError, match="format"):
        module().quantize(np.ones((1,32),np.float32),fmt)


def test_reject_corrupt_padding():
    q = module()
    with pytest.raises(ValueError,match="padding"):
        q.FP4Matrix("nvfp4",(1,1),np.array([[0xf2]],np.uint8),np.array([[56]],np.uint8),1.)


@pytest.mark.parametrize("value",[0.,-1.,np.inf,np.nan])
def test_bad_global_scale(value):
    q = module()
    with pytest.raises(ValueError):
        q.FP4Matrix("nvfp4",(1,16),np.zeros((1,8),np.uint8),np.ones((1,1),np.uint8),value)


@pytest.mark.parametrize("bad",[np.nan,np.inf,-np.inf])
def test_nonfinite_inputs_rejected(bad):
    with pytest.raises(ValueError):
        module().quantize(np.array([[bad]],np.float32),"nvfp4")


def test_safe_serialization(tmp_path):
    q = module()
    p = q.quantize(np.arange(51,dtype=np.float32).reshape(3,17)-20,"nvfp4")
    path = tmp_path/"weight.npz"
    q.save_fp4(path,p)
    restored = q.load_fp4(path)
    assert restored.checksum == p.checksum
    np.testing.assert_array_equal(q.dequantize(restored),q.dequantize(p))


def test_invalid_nibble_and_dtype():
    q = module()
    for x in [np.array([[16]],np.uint8), np.array([[-1]],np.int8), np.array([[1.2]],np.float32)]:
        with pytest.raises((TypeError,ValueError)):
            q.pack_nibbles(x)


def test_tiny_explicit_scale_saturates_without_overflow_warning():
    import warnings
    from cumlx import quantize, dequantize
    with warnings.catch_warnings():
        warnings.simplefilter('error',RuntimeWarning)
        q=quantize(np.array([[1.,-1.]],np.float32),'nvfp4',global_scale=float(np.finfo(np.float32).smallest_subnormal))
    assert np.isfinite(dequantize(q)).all()
    np.testing.assert_array_equal(q.payload,np.array([[0xf7]],np.uint8))


def test_e2m1_extreme_finite_values_saturate_to_six():
    from cumlx.quant import encode_e2m1
    x=np.array([np.finfo(np.float32).max,-np.finfo(np.float32).max],np.float32)
    np.testing.assert_array_equal(encode_e2m1(x),np.array([7,15],np.uint8))
