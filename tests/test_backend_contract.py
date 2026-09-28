import importlib
import pytest


def backend_module():
    assert importlib.util.find_spec('cumlx.mlx_backend') is not None, 'MLX launch backend not implemented'
    return importlib.import_module('cumlx.mlx_backend')


def test_unvalidated_execution_requires_opt_in():
    with pytest.raises(RuntimeError,match='unvalidated'):
        backend_module().MLXBackend('metal')


def test_unknown_backend_not_silently_cpu():
    with pytest.raises(ValueError,match='backend'):
        backend_module().MLXBackend('cpu',allow_unvalidated=True)


def test_kernel_registry_has_real_sources():
    b=backend_module()
    for name in b.KERNELS:
        for kind in ['metal','cuda']:
            header,source,digest=b.kernel_source(name,kind)
            assert len(digest)==64 and len(header)>100 and len(source)>150
            assert '@' not in source


def test_capabilities_do_not_claim_cuda_facade():
    report=backend_module().environment_report()
    assert report['cuda_facing_pytorch'] is False
    assert report['native_fp4_matrix_instruction'] is False
    assert report['device_validation']=='NOT_RUN'
