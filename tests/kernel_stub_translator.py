"""A stub translator, written for tests of the adapter interface. It is NOT a translator and says nothing about woof.

It rewrites the small CUDA dialect of the CUMLX kernels and test kernels into Metal with a fixed table. It proves that the
pipeline of `cumlx kernel port` and `cumlx kernel verify` is wired correctly. It cannot translate anything else.
"""
import re

from cumlx.verify.kernel_port import TranslationResult
from cumlx.verify.kernel_verify import ATTRIBUTES

_CALLS = {'ldexpf': 'metal::ldexp', 'copysignf': 'metal::copysign', 'signbit': 'metal::signbit', 'fabsf': 'metal::abs',
          'expf': 'metal::exp', 'fmaxf': 'metal::max', 'fminf': 'metal::min', 'sqrtf': 'metal::sqrt', 'fmaf': 'metal::fma'}
_TYPES = {'float', 'int', 'unsigned int', '__half'}


def _common(text):
    text = re.sub(r'#include\s*<(?:math|stdint|cuda_fp16)\.h>\n', '', text)
    text = text.replace('(blockIdx.x * blockDim.x + threadIdx.x)', 'thread_position_in_grid.x')
    text = text.replace('blockIdx.x * blockDim.x + threadIdx.x', 'thread_position_in_grid.x')
    for old, new in {'threadIdx': 'thread_position_in_threadgroup', 'blockIdx': 'threadgroup_position_in_grid',
                     'blockDim': 'threads_per_threadgroup', '__shared__': 'threadgroup',
                     '__syncthreads()': 'threadgroup_barrier(mem_flags::mem_threadgroup)',
                     '__device__ __forceinline__': 'inline', '__half': 'half'}.items():
        text = text.replace(old, new)
    for old, new in _CALLS.items():
        text = re.sub(rf'(?<![\w:]){old}\b', new, text)
    return text


def _complete(text):
    translated = _common(text)

    def signature(match):
        name, params = match.group(1), match.group(2)
        out = []
        for i, param in enumerate(p.strip() for p in params.split(',')):
            kind, _, var = param.rpartition('*')
            kind = kind.strip()
            if kind.startswith('const '):
                out.append(f'const constant {kind[6:]}* {var.strip()} [[buffer({i})]]')
            else:
                out.append(f'device {kind}* {var.strip()} [[buffer({i})]]')
        body = translated[match.end():]
        out += [f'{t} {a} [[{a}]]' for a, t in ATTRIBUTES.items() if re.search(rf'\b{a}\b', body)]
        return f"kernel void {name}({', '.join(out)})"

    return re.sub(r'extern "C" __global__ void (\w+)\(([^)]*)\)', signature, translated)


class StubTranslator:
    name = 'test-stub'
    accepts_body = True

    def unavailable(self):
        return None

    def translate(self, request):
        if request.form == 'body':
            return TranslationResult(_common(request.source), 'body', self.name, '0', header=_common(request.header), entry_points=[request.name])
        return TranslationResult(_complete(request.source), 'complete', self.name, '0', entry_points=list(request.entry_points))


class MissingTranslator(StubTranslator):
    name = 'test-missing'

    def unavailable(self):
        return 'it is a test double that reports itself as not installed'


class FailingTranslator(StubTranslator):
    name = 'test-failing'

    def translate(self, request):
        raise ValueError('template instance not found')


class EmptyTranslator(StubTranslator):
    name = 'test-empty'

    def translate(self, request):
        return TranslationResult('   \n', 'complete', self.name, entry_points=list(request.entry_points))


class DropsEntryTranslator(StubTranslator):
    name = 'test-drops'

    def translate(self, request):
        return TranslationResult('kernel void other() {}\n', 'complete', self.name, entry_points=['other'])


class ReturnsNonsense(StubTranslator):
    name = 'test-nonsense'

    def translate(self, request):
        return {'msl': 'not a result'}


class NoBody(StubTranslator):
    name = 'test-no-body'
    accepts_body = False
