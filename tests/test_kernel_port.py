"""`cumlx kernel port`: the pre-translation layer and the translator adapter. No GPU and no third-party code."""
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

from cumlx.cli.main import main
from cumlx.errors import CumlxError, ExitCode
from cumlx.verify import kernel_port
from cumlx.verify.kernel_port import PTX_BUILTINS, preprocess_file, split_declarations

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests/kernel_fixtures'
LAB = ROOT / 'src/cumlx/kernels'
LAB_KERNELS = ['fp4_decode', 'fp4_gemv', 'fp4_small_m', 'fp4_tiled', 'fp4_w4a4', 'fp4_quantize', 'rms_norm', 'silu_mul']
STUB = 'python:kernel_stub_translator:'


def run(capsys, *argv):
    code = main([*argv, '--format', 'json'])
    return code, json.loads(capsys.readouterr().out)


def write(tmp_path, text, name='k.cu'):
    p = tmp_path / name
    p.write_text(text)
    return p


# ---------------------------------------------------------------- the four walls of M0
def test_template_is_instantiated_from_the_host_launch():
    pre = preprocess_file(FIXTURES / 'vector_add.cu')
    assert pre.form == 'complete' and pre.entry_points == ['vector_add']
    assert 'extern "C" __global__ void vector_add(const float* a, const float* b, float* out)' in pre.source
    assert 'template' not in pre.source and 'typename' not in pre.source
    assert pre.record['templates'] == {'vector_add<float>': 'vector_add'}


def test_host_code_is_separated():
    pre = preprocess_file(FIXTURES / 'vector_add.cu')
    assert pre.record['host_removed'] == ['main', 'report']
    for word in ('main', 'std::', 'cudaMalloc', 'iostream', '<<<'):
        assert word not in pre.source
    assert pre.record['headers']['host_dropped'] == ['iostream', 'vector']


def test_framework_header_and_host_function_go_and_device_code_stays():
    pre = preprocess_file(FIXTURES / 'norm_mixed.cu')
    assert pre.record['headers']['host_dropped'] == ['ATen/ATen.h'] and pre.record['host_removed'] == ['run_on_tensor']
    assert 'at::' not in pre.source and 'ATen' not in pre.source
    assert pre.record['headers']['inlined'] == ['norm_constants.h']
    assert '#define NORM_SCALE 2.5f' in pre.source and 'NOT_USED_BY_THE_KERNEL' not in pre.source  # only what device code needs
    assert 'scale_of__float(x[' in pre.source and 'template' not in pre.source  # helper instance from an explicit call
    assert '__launch_bounds__' not in pre.source
    assert 'namespace' not in pre.source


def test_builtins_are_rewritten_with_helpers_that_stay_in_the_output():
    pre = preprocess_file(FIXTURES / 'builtin_map.cu')
    rewritten = pre.record['builtins_rewritten']
    assert rewritten['rsqrtf'] == 'cumlx_rsqrtf' and rewritten['__clz'] == 'cumlx_clz' and rewritten['__fadd_rn'] == 'cumlx_fadd_rn'
    tokens = set(re.findall(r'[A-Za-z_]\w*', pre.source))
    for name in rewritten:
        assert name not in tokens, name
    for helper in set(rewritten.values()):
        assert f'{helper}(' in pre.source
    assert 'cumlx_exp10f' not in pre.source  # a helper that is not used is not emitted
    assert pre.record['builtins_direct']['threadIdx'] == 'thread_position_in_threadgroup'


def test_standard_headers_are_replaced(tmp_path):
    p = write(tmp_path, '''
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cfloat>
#include <climits>
#include <limits>
__global__ void k(const float* a, float* out) {
    uint32_t i = threadIdx.x;
    float m = std::numeric_limits<float>::max();
    out[i] = std::min(a[i] + FLT_EPSILON, m) + (float)INT_MAX;
}
''')
    pre = preprocess_file(p)
    assert '#include <math.h>' in pre.source and '#include <stdint.h>' in pre.source
    assert 'cfloat' not in pre.source and 'cuda_runtime' not in pre.source
    assert '#define FLT_EPSILON' in pre.source and '#define INT_MAX' in pre.source and '#define FLT_MAX' in pre.source
    assert 'std::' not in pre.source and 'FLT_MAX' in pre.source and 'min(a[i]' in pre.source


# ---------------------------------------------------------------- templates
def test_two_instances_get_two_entry_points(tmp_path):
    p = write(tmp_path, '''
template <typename T> __global__ void k(const T* a, T* out) { out[threadIdx.x] = a[threadIdx.x]; }
void host() { k<float><<<1, 4>>>(nullptr, nullptr); k<int><<<1, 4>>>(nullptr, nullptr); }
''')
    pre = preprocess_file(p)
    assert sorted(pre.entry_points) == ['k__float', 'k__int']
    assert 'k__float(const float* a, float* out)' in pre.source and 'k__int(const int* a, int* out)' in pre.source


def test_instantiate_flag_wins_and_non_type_parameters_work(tmp_path):
    p = write(tmp_path, '''
template <typename T, int N> __global__ void k(const T* a, T* out) { out[threadIdx.x] = a[threadIdx.x] * N; }
void host() { k<double, 3><<<1, 4>>>(nullptr, nullptr); }
''')
    pre = preprocess_file(p, instantiate=['k<float, 2>'])
    assert pre.entry_points == ['k'] and 'const float* a' in pre.source and '* 2;' in pre.source


def test_default_template_arguments_are_enough(tmp_path):
    pre = preprocess_file(write(tmp_path, 'template <typename T = float> __global__ void k(const T* a, T* out) { out[threadIdx.x] = a[0]; }\n'))
    assert pre.entry_points == ['k'] and 'const float* a' in pre.source


def test_a_template_with_no_concrete_instance_is_refused(tmp_path):
    p = write(tmp_path, '''
template <typename T> __global__ void k(const T* a, T* out) { out[threadIdx.x] = a[0]; }
template <typename U> void host(U* a, U* b) { k<U><<<1, 4>>>(a, b); }
''')
    with pytest.raises(CumlxError) as exc:
        preprocess_file(p)
    assert exc.value.code == 'E_KERNEL_UNPORTABLE' and exc.value.exit_code == ExitCode.INPUT
    refused = exc.value.details['refused'][0]
    assert refused['feature'] == 'uninstantiated-template' and 'k<float>' in refused['reason'] and 'k<U>' in refused['reason']


def test_deduced_helper_stays_a_template_and_says_so(tmp_path):
    p = write(tmp_path, '''
template <typename T> __device__ T twice(T x) { return x + x; }
__global__ void k(const float* a, float* out) { out[threadIdx.x] = twice(a[threadIdx.x]); }
''')
    pre = preprocess_file(p)
    assert 'template <typename T> __device__ T twice' in pre.source
    assert [w['code'] for w in pre.warnings] == ['W_KERNEL_TEMPLATE_KEPT']


def test_instantiate_rejects_what_is_not_a_template(tmp_path):
    p = write(tmp_path, '__global__ void k(const float* a, float* out) { out[threadIdx.x] = a[0]; }\n')
    with pytest.raises(CumlxError, match='not a template'):
        preprocess_file(p, instantiate=['k<float>'])
    with pytest.raises(CumlxError, match='NAME<ARGS>'):
        preprocess_file(p, instantiate=['nonsense'])
    with pytest.raises(CumlxError, match='plain type'):
        preprocess_file(p, instantiate=['k<evil; system("x")>'])


# ---------------------------------------------------------------- refusals
@pytest.mark.parametrize('body,feature,token', [
    ('unsigned int m = __match_any_sync(__activemask(), 3); out[0] = m;', 'warp-level', '__match_any_sync'),
    ('out[0] = __shfl_down_sync(0xffffffffu, a[0], 1);', 'warp-level', '__shfl_down_sync'),
    ('__syncwarp(); out[0] = a[0];', 'warp-level', '__syncwarp'),
    ('out[0] = __frcp_rz(a[0]);', 'rounding-mode', '__frcp_rz'),
    ('out[0] = __fadd_ru(a[0], a[1]);', 'rounding-mode', '__fadd_ru'),
    ('double d = a[0]; out[0] = (float)d;', 'fp64', 'double'),
    ('printf("x"); out[0] = a[0];', 'no-metal-equivalent', 'printf'),
    ('__threadfence(); out[0] = a[0];', 'no-metal-equivalent', '__threadfence'),
    ('out[0] = erff(a[0]);', 'no-metal-equivalent', 'erff'),
    ('asm("mov.f32 %0, %1;" : "=f"(out[0]) : "f"(a[0]));', 'no-metal-equivalent', 'asm'),
    ('out[0] = __new_builtin_nobody_knows(a[0]);', 'unknown-builtin', '__new_builtin_nobody_knows'),
    ('other<<<1, 4>>>(a, out);', 'dynamic-parallelism', '<<<'),
    ('std::vector<float> v; out[0] = v[0];', 'host-symbol-in-device', 'std::vector'),
    ('cooperative_groups::thread_block b; out[0] = a[0];', 'no-metal-equivalent', 'cooperative_groups'),
])
def test_features_with_no_metal_equivalent_are_refused_with_a_reason(tmp_path, body, feature, token):
    p = write(tmp_path, f'__global__ void k(const float* a, float* out) {{\n    {body}\n}}\n')
    with pytest.raises(CumlxError) as exc:
        preprocess_file(p)
    assert exc.value.code == 'E_KERNEL_UNPORTABLE'
    refused = exc.value.details['refused']
    match = next(r for r in refused if r['token'] == token)
    assert match['feature'] == feature and len(match['reason']) > 20 and match['line'] == 2


def test_every_refusal_is_reported_not_only_the_first(tmp_path):
    p = write(tmp_path, '__global__ void k(const float* a, float* out) { out[0] = __frcp_rz(a[0]) + __ballot_sync(1u, true) + __fsqrt_rd(a[1]); }\n')
    with pytest.raises(CumlxError) as exc:
        preprocess_file(p)
    assert {r['token'] for r in exc.value.details['refused']} == {'__frcp_rz', '__ballot_sync', '__fsqrt_rd'}


def test_a_word_in_a_comment_or_a_string_is_not_a_feature(tmp_path):
    p = write(tmp_path, '''// __match_any_sync and double are only named here
__global__ void k(const float* a, float* out) {
    /* __frcp_rz */ out[threadIdx.x] = a[0];
}
''')
    assert preprocess_file(p).entry_points == ['k']


def test_warp_code_in_the_host_part_does_not_matter(tmp_path):
    p = write(tmp_path, '''
__global__ void k(const float* a, float* out) { out[threadIdx.x] = a[0]; }
void host() { double d = 1.0; printf("%f", d); }
''')
    assert preprocess_file(p).record['host_removed'] == ['host']


def test_device_code_that_calls_a_host_function_is_refused(tmp_path):
    p = write(tmp_path, '''
float host_only(float x) { return x; }
__global__ void k(const float* a, float* out) { out[threadIdx.x] = host_only(a[0]); }
''')
    with pytest.raises(CumlxError) as exc:
        preprocess_file(p)
    assert exc.value.details['refused'][0]['feature'] == 'host-symbol-in-device'


def test_mutable_device_global_and_recursion_are_refused(tmp_path):
    with pytest.raises(CumlxError) as exc:
        preprocess_file(write(tmp_path, '__device__ int counter;\n__global__ void k(float* out) { out[0] = counter; }\n'))
    assert exc.value.details['refused'][0]['feature'] == 'device-global'
    with pytest.raises(CumlxError) as exc:
        preprocess_file(write(tmp_path, '''
__device__ int g(int n);
__device__ int f(int n) { return n ? g(n - 1) : 0; }
__device__ int g(int n) { return f(n); }
__global__ void k(int* out) { out[0] = f(3); }
'''))
    assert exc.value.details['refused'][0]['feature'] == 'recursion'


FLOAT2_OPERATORS = '''
__device__ __forceinline__ float2& operator*=(float2& a, const float2& b) { a.x *= b.x; a.y *= b.y; return a; }
__device__ __forceinline__ float2& operator+=(float2& a, float b) { a.x += b; a.y += b; return a; }
__device__ __forceinline__ bool operator!=(const float2& a, const float2& b) { return a.x != b.x || a.y != b.y; }
__device__ __forceinline__ float2 operator*(const float2& a, float b) { return make_float2(a.x * b, a.y * b); }
struct Accum { float v; };
__device__ __forceinline__ Accum& operator-=(Accum& a, float b) { a.v -= b; return a; }
__device__ float scale(float x) { return x * 2.0f; }
'''


def test_a_compound_assignment_operator_is_a_function_and_not_a_device_global(tmp_path):
    """The `=` of `operator*=` read as an initializer: a global named `operator`, and the next declaration inside it."""
    source = FLOAT2_OPERATORS + '__global__ void k(float* out) { out[0] = scale(1.0f); }\n'
    pre = preprocess_file(write(tmp_path, source))
    assert pre.entry_points == ['k']
    chunks = split_declarations(source)
    operators = [c for c in chunks if 'operator' in c.text]
    assert [c.kind for c in operators] == ['function'] * 5
    assert all(c.device for c in operators)
    # Each stays its own declaration, so the one after an operator is not swallowed.
    assert [c.name for c in chunks if c.kind == 'function' and c.name] == ['scale', 'k']
    for token in ('operator*=', 'operator+=', 'operator!=', 'operator-='):
        assert token in pre.source


@pytest.mark.parametrize('declaration, token', [
    ('__device__ int counter;', 'counter'),
    ('__device__ int counter = 3;', 'counter'),
    ('__device__ int operator_count = 3;', 'operator_count'),
    ('__device__ float2 origin = make_float2(0.0f, 0.0f);', 'origin'),
])
def test_a_real_device_global_is_still_refused_with_its_name(tmp_path, declaration, token):
    with pytest.raises(CumlxError) as exc:
        preprocess_file(write(tmp_path, f'{declaration}\n__global__ void k(float* out) {{ out[0] = 1.0f; }}\n'))
    assert [(r['feature'], r['token']) for r in exc.value.details['refused']] == [('device-global', token)]


def test_an_overload_that_calls_another_arity_of_its_name_is_not_recursion(tmp_path):
    """A wrapper of a CUDA built-in has the same name, and the built-in is not in this source."""
    for overloads in ('''
__device__ __forceinline__ float4 make_float4(const float3& a, float w) { return make_float4(a.x, a.y, a.z, w); }
''', '''
__device__ __forceinline__ float4 make_float4(const float2& a) { return make_float4(a.x, a.y, 0.0f, 1.0f); }
__device__ __forceinline__ float4 make_float4(const float3& a, float w) { return make_float4(a.x, a.y, a.z, w); }
''', '''
__device__ __forceinline__ float2 pair(float a, float b) { return make_float2(a, b); }
__device__ __forceinline__ float2 pair(float a) { return pair(a, a); }
'''):
        pre = preprocess_file(write(tmp_path, overloads + '__global__ void k(float* out) { out[0] = 1.0f; }\n'))
        assert pre.entry_points == ['k']


@pytest.mark.parametrize('source, token', [
    # Direct, as before.
    ('__device__ int f(int n) { return n ? f(n - 1) : 0; }\n', 'f -> f'),
    # The first overload recurses and the second does not: only the last definition of a name used to be looked at.
    ('__device__ int f(int n) { return n ? f(n - 1) : 0; }\n__device__ int f(int a, int b) { return a + b; }\n', 'f -> f'),
    # Two overloads that call each other. No function calls its own arity.
    ('__device__ int f(int a, int b);\n__device__ int f(int a) { return f(a, 1); }\n__device__ int f(int a, int b) { return b ? f(a) : a; }\n', 'f -> f -> f'),
    # The same arity: types decide, and this layer does not read them, so it does not call this safe.
    ('__device__ int f(float a) { return a; }\n__device__ int f(int a) { return f(a - 1); }\n', 'f -> f'),
    # A default argument and a variadic list both accept this call.
    ('__device__ int f(int a, int b = 0) { return f(a); }\n', 'f -> f'),
    ('__device__ int f(int a, ...) { return f(a, 1, 2); }\n', 'f -> f'),
    # `<` in an argument must not hide the call by miscounting it.
    ('__device__ int f(int a, int b) { return f(a < b, a > b); }\n', 'f -> f'),
    # Through a name with other overloads, across two functions.
    ('__device__ int g(int a);\n__device__ int f(int a, int b) { return g(a) + b; }\n__device__ int g(int a) { return f(a, 1); }\n', 'f -> g -> f'),
])
def test_real_recursion_is_still_refused(tmp_path, source, token):
    with pytest.raises(CumlxError) as exc:
        preprocess_file(write(tmp_path, source + '__global__ void k(int* out) { out[0] = 1; }\n'))
    assert [(r['feature'], r['token']) for r in exc.value.details['refused']] == [('recursion', token)]


def test_constant_global_stays(tmp_path):
    pre = preprocess_file(write(tmp_path, '__constant__ float table[2] = {1.0f, 2.0f};\n__global__ void k(float* out) { out[threadIdx.x] = table[1]; }\n'))
    assert '__constant__ float table[2]' in pre.source


def test_an_include_that_cannot_be_found_is_refused_and_include_dir_finds_it(tmp_path):
    p = write(tmp_path, '#include "elsewhere.h"\n__global__ void k(float* out) { out[threadIdx.x] = VALUE; }\n')
    with pytest.raises(CumlxError) as exc:
        preprocess_file(p)
    assert exc.value.details['refused'][0]['feature'] == 'unresolved-include'
    other = tmp_path / 'other'
    other.mkdir()
    (other / 'elsewhere.h').write_text('#define VALUE 3.0f\n')
    pre = preprocess_file(p, include_dirs=[str(other)])
    assert '#define VALUE 3.0f' in pre.source and pre.record['headers']['inlined'] == ['elsewhere.h']


def test_a_quoted_header_means_the_local_file_even_with_a_standard_name(tmp_path):
    (tmp_path / 'limits.h').write_text('#define VALUE 3.0f\n')
    pre = preprocess_file(write(tmp_path, '#include "limits.h"\n__global__ void k(float* out) { out[threadIdx.x] = VALUE; }\n'))
    assert '#define VALUE 3.0f' in pre.source and pre.record['headers']['inlined'] == ['limits.h'] and pre.record['headers']['replaced'] == {}
    pre = preprocess_file(write(tmp_path, '#include "stdint.h"\n__global__ void k(float* out) { out[threadIdx.x] = 1.0f; }\n', name='j.cu'))
    assert pre.record['headers']['replaced'] == {'stdint.h': '<stdint.h>'}  # no such local file, so the standard one


def test_unknown_system_header_is_dropped_and_stated(tmp_path):
    pre = preprocess_file(write(tmp_path, '#include <some_library.h>\n__global__ void k(float* out) { out[threadIdx.x] = 1.0f; }\n'))
    assert pre.record['headers']['unknown_dropped'] == ['some_library.h']
    assert 'W_KERNEL_HEADER_DROPPED' in [w['code'] for w in pre.warnings]


def test_fp64_demotion_is_explicit_and_stated(tmp_path):
    p = write(tmp_path, '__global__ void k(const double* a, double* out) { double x = a[0] * 0.5; out[threadIdx.x] = x; }\n')
    pre = preprocess_file(p, demote_fp64=True)
    assert 'double' not in pre.source and 'const float* a' in pre.source and '0.5f' in pre.source
    assert 'W_KERNEL_FP64_DEMOTED' in [w['code'] for w in pre.warnings]


def test_unsuffixed_literal_is_stated(tmp_path):
    pre = preprocess_file(write(tmp_path, '__global__ void k(const float* a, float* out) { out[threadIdx.x] = a[0] * 0.5; }\n'))
    warning = next(w for w in pre.warnings if w['code'] == 'W_KERNEL_FP64_LITERAL')
    assert warning['count'] == 1


def test_a_file_that_is_not_a_kernel_is_refused(tmp_path):
    with pytest.raises(CumlxError) as exc:
        preprocess_file(write(tmp_path, 'int main() { return 0; }\n'))
    assert exc.value.details['refused'][0]['feature'] == 'no-entry-point'


def test_builtin_table_covers_the_walls_of_m0():
    assert PTX_BUILTINS['__match_any_sync'].status == 'refuse' and PTX_BUILTINS['__frcp_rz'].status == 'refuse'
    assert PTX_BUILTINS['rsqrtf'].status == 'rewrite'
    assert PTX_BUILTINS['threadIdx'].metal == 'thread_position_in_threadgroup'
    for name, item in PTX_BUILTINS.items():
        assert item.status in ('direct', 'rewrite', 'refuse'), name
        assert item.status != 'refuse' or item.reason, name
    from cumlx.verify import bf16
    assert all(item.helper in kernel_port._HELPERS or item.helper.endswith('f') or item.helper in bf16.NAMES
               for item in PTX_BUILTINS.values() if item.status == 'rewrite')
    assert all(f' {name}(' in bf16.METAL_HEADER or f'struct {name}' in bf16.METAL_HEADER for name in bf16.NAMES)
    assert kernel_port.builtin_table()['__frcp_rz']['status'] == 'refuse'


def test_scanner_handles_the_shapes_of_declarations():
    chunks = split_declarations('''
#define A 1
struct P { float x; __device__ float f() const { return x; } };
typedef unsigned int u32;
using f32 = float;
enum Kind { One, Two };
static const float limit = 3.0f;
__device__ float g(float x) { return x; }
__global__ void k(float* out) { P p; out[0] = g(p.x); }
int main(int argc, char** argv) { return 0; }
''')
    assert [(c.kind, c.name) for c in chunks] == [
        ('directive', 'A'), ('type', 'P'), ('type', 'u32'), ('type', 'f32'), ('type', 'Kind'), ('variable', 'limit'),
        ('function', 'g'), ('function', 'k'), ('function', 'main')]
    assert [c.name for c in chunks if c.device] == ['g', 'k'] and [c.name for c in chunks if c.is_global] == ['k']


# ---------------------------------------------------------------- function bodies
@pytest.mark.parametrize('kernel', LAB_KERNELS)
def test_lab_bodies_stay_bodies(kernel):
    pre = preprocess_file(LAB / f'{kernel}.cu')
    assert pre.form == 'body' and pre.entry_points == [kernel] and '__global__' not in pre.source
    assert 'cumlx_restore' in pre.header or kernel in ('silu_mul', 'rms_norm')
    assert 'threadIdx' in pre.source or 'blockIdx' in pre.source


def test_body_wrapped_with_a_case_is_a_complete_kernel():
    cases = str(ROOT / 'validation/kernel-cases/cases.json')
    pre = preprocess_file(LAB / 'rms_norm.cu', cases_file=cases, case_id='rms_norm.plain.4x129')
    assert pre.form == 'complete' and pre.record['case'] == 'rms_norm.plain.4x129'
    assert 'extern "C" __global__ void rms_norm(const float* x, const float* residual, const float* weight, const float* epsilon, float* y)' in pre.source
    assert '#define K 129' in pre.source and '#define HAS_RESIDUAL 0' in pre.source


def test_case_flags_are_checked():
    cases = str(ROOT / 'validation/kernel-cases/cases.json')
    with pytest.raises(CumlxError, match='go together'):
        preprocess_file(LAB / 'rms_norm.cu', cases_file=cases)
    with pytest.raises(CumlxError, match='no case'):
        preprocess_file(LAB / 'rms_norm.cu', cases_file=cases, case_id='nope')
    with pytest.raises(CumlxError, match='already a complete kernel'):
        preprocess_file(FIXTURES / 'builtin_map.cu', cases_file=cases, case_id='rms_norm.plain.4x129')


# ---------------------------------------------------------------- the prepared source is valid C++
@pytest.mark.skipif(shutil.which('clang++') is None, reason='no host C++ compiler to check the syntax')
@pytest.mark.parametrize('kernel', ['vector_add', 'norm_mixed', 'builtin_map'])
def test_prepared_source_parses_in_a_host_compiler(tmp_path, kernel):
    shim = tmp_path / 'shim.h'
    shim.write_text('''
#define __global__
#define __device__
#define __forceinline__ inline
#define __shared__ static
struct cumlx_dim { unsigned int x, y, z; };
static cumlx_dim threadIdx, blockIdx, blockDim;
static void __syncthreads() {}
struct __half { __half() {} explicit __half(float) {} explicit operator float() const { return 0.0f; } };
''')
    fake = tmp_path / 'fake'
    fake.mkdir()
    (fake / 'cuda_fp16.h').write_text('')
    source = tmp_path / 'prepared.cpp'
    source.write_text(preprocess_file(FIXTURES / f'{kernel}.cu').source)
    done = subprocess.run(['clang++', '-std=c++14', '-fsyntax-only', '-I', str(fake), '-include', str(shim), str(source)],
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr


# ---------------------------------------------------------------- the adapter
def test_missing_translator_is_an_environment_error_never_a_pass(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr(kernel_port.importlib.util, 'find_spec', lambda _n: None)
    out = tmp_path / 'vector_add.metal'
    code, doc = run(capsys, 'kernel', 'port', str(FIXTURES / 'vector_add.cu'), '-o', str(out))
    assert code == ExitCode.ENVIRONMENT and doc['status'] == 'error' and not doc['ok']
    assert doc['errors'][0]['code'] == 'E_KERNEL_TRANSLATOR_MISSING' and 'metal-woof' in doc['errors'][0]['hint']
    assert not out.exists()


@pytest.mark.parametrize('name,code_expected,error', [
    ('MissingTranslator', ExitCode.ENVIRONMENT, 'E_KERNEL_TRANSLATOR_MISSING'),
    ('FailingTranslator', ExitCode.INPUT, 'E_KERNEL_TRANSLATE_FAILED'),
    ('EmptyTranslator', ExitCode.INPUT, 'E_KERNEL_TRANSLATE_FAILED'),
    ('DropsEntryTranslator', ExitCode.INPUT, 'E_KERNEL_TRANSLATE_FAILED'),
    ('ReturnsNonsense', ExitCode.INPUT, 'E_KERNEL_TRANSLATE_FAILED'),
])
def test_a_translator_that_fails_writes_nothing(capsys, tmp_path, monkeypatch, name, code_expected, error):
    monkeypatch.syspath_prepend(str(ROOT / 'tests'))
    out = tmp_path / 'vector_add.metal'
    code, doc = run(capsys, 'kernel', 'port', str(FIXTURES / 'vector_add.cu'), '-o', str(out), '--translator', STUB + name)
    assert code == code_expected and doc['errors'][0]['code'] == error
    assert not out.exists() and not out.with_suffix('.port.json').exists()


def test_refused_kernel_never_reaches_the_translator(capsys, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'tests'))
    p = write(tmp_path, '__global__ void k(float* out) { out[0] = __match_any_sync(1u, 2); }\n')
    code, doc = run(capsys, 'kernel', 'port', str(p), '-o', str(tmp_path / 'k.metal'), '--translator', STUB + 'StubTranslator')
    assert code == ExitCode.INPUT and doc['errors'][0]['code'] == 'E_KERNEL_UNPORTABLE'
    assert 'warp' in doc['errors'][0]['details']['refused'][0]['reason'].lower()
    assert not (tmp_path / 'k.metal').exists()


def test_translator_that_needs_a_complete_kernel_says_how_to_get_one(capsys, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'tests'))
    code, doc = run(capsys, 'kernel', 'port', str(LAB / 'silu_mul.cu'), '-o', str(tmp_path / 'silu_mul.metal'), '--translator', STUB + 'NoBody')
    assert code == ExitCode.INPUT and doc['errors'][0]['code'] == 'E_KERNEL_TRANSLATE_FAILED' and '--case' in doc['errors'][0]['hint']


def test_translate_writes_metal_and_a_record(capsys, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'tests'))
    out = tmp_path / 'vector_add.metal'
    code, doc = run(capsys, 'kernel', 'port', str(FIXTURES / 'vector_add.cu'), '-o', str(out), '--translator', STUB + 'StubTranslator')
    assert code == 0 and doc['data']['grade'] == 'unverified'
    assert 'W_KERNEL_NOT_VERIFIED' in [w['code'] for w in doc['warnings']]
    assert 'kernel void vector_add(' in out.read_text()
    side = json.loads((tmp_path / 'vector_add.port.json').read_text())
    assert side['translator']['name'] == 'test-stub' and side['schema'] == kernel_port.PORT_SCHEMA and side['preprocess']['host_removed'] == ['main', 'report']
    code, doc = run(capsys, 'kernel', 'port', str(FIXTURES / 'vector_add.cu'), '-o', str(out), '--translator', STUB + 'StubTranslator')
    assert code == ExitCode.USAGE and doc['errors'][0]['code'] == 'E_OUTPUT_EXISTS'
    code, _ = run(capsys, 'kernel', 'port', str(FIXTURES / 'vector_add.cu'), '-o', str(out), '--translator', STUB + 'StubTranslator', '--overwrite')
    assert code == 0


def test_preprocess_only_needs_no_translator_and_says_it_translated_nothing(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr(kernel_port.importlib.util, 'find_spec', lambda _n: None)
    out = tmp_path / 'vector_add.pre.cu'
    code, doc = run(capsys, 'kernel', 'port', str(FIXTURES / 'vector_add.cu'), '--preprocess-only', '-o', str(out))
    assert code == 0 and 'W_KERNEL_NOT_TRANSLATED' in [w['code'] for w in doc['warnings']]
    assert 'extern "C" __global__' in out.read_text() and doc['data']['translator'] is None


def test_bad_translator_specs(capsys, tmp_path):
    out = str(tmp_path / 'x.metal')
    source = str(FIXTURES / 'vector_add.cu')
    code, doc = run(capsys, 'kernel', 'port', source, '-o', out, '--translator', 'other')
    assert code == ExitCode.USAGE
    code, doc = run(capsys, 'kernel', 'port', source, '-o', out, '--translator', 'python:only_one_part')
    assert code == ExitCode.USAGE
    code, doc = run(capsys, 'kernel', 'port', source, '-o', out, '--translator', 'python:no_such_module_xyz:Thing')
    assert code == ExitCode.ENVIRONMENT and doc['errors'][0]['code'] == 'E_KERNEL_TRANSLATOR_MISSING'
    code, doc = run(capsys, 'kernel', 'port', source, '-o', out, '--translator', 'python:json:dumps')
    assert code != 0


def test_woof_adapter_calls_the_names_seen_in_m0(tmp_path, monkeypatch):
    """A stub package with the names that spikes/probes/probe_woof.py uses. This is not woof."""
    package = tmp_path / 'woof'
    (package / 'backend').mkdir(parents=True)
    (package / '__init__.py').write_text('')
    (package / 'backend' / '__init__.py').write_text('')
    (package / 'backend' / 'translate.py').write_text('''
class Result:
    msl = 'kernel void vector_add() {}\\n'
    kernels = {'vector_add': None}
calls = []
def translate(source, unit, include_dirs=()):
    calls.append((source, unit, include_dirs))
    return Result()
''')
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in [m for m in sys.modules if m == 'woof' or m.startswith('woof.')]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setenv('WOOF_BACKEND', 'mlx')
    adapter = kernel_port.load_translator('woof')
    assert adapter.unavailable() is None and adapter.accepts_body is False
    pre = preprocess_file(FIXTURES / 'vector_add.cu')
    result = kernel_port.translate(pre, adapter, ['inc'])
    assert result.form == 'complete' and result.entry_points == ['vector_add'] and result.translator == 'woof'
    calls = sys.modules['woof.backend.translate'].calls
    assert calls[0][1] == 'vector_add' and calls[0][2] == ('inc',) and 'extern "C" __global__' in calls[0][0]
    for name in [m for m in sys.modules if m == 'woof' or m.startswith('woof.')]:
        del sys.modules[name]
