"""Regenerate our deliberately portable Metal/CUDA implementations.

This is a source maintenance utility for eight hand-designed kernels; it is
NOT a translator for arbitrary CUDA, PTX, warp intrinsics, or CUDA libraries.
"""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/'src/cumlx/kernels'
COMMON=r'''
@INLINE@ float cumlx_e2m1(unsigned int c) {
    unsigned int e = (c & 7u) >> 1u;
    unsigned int m = c & 1u;
    float v = e == 0u ? 0.5f * float(m) : @LDEXP@(1.0f + 0.5f * float(m), int(e) - 1);
    return @COPYSIGN@(v, (c & 8u) ? -1.0f : 1.0f);
}
@INLINE@ float cumlx_e4m3(unsigned int c) {
    if (c > 126u) return NAN;
    unsigned int e = c >> 3u, m = c & 7u;
    return e == 0u ? float(m) * 0.001953125f : @LDEXP@(1.0f + float(m) * 0.125f, int(e) - 7);
}
@INLINE@ float cumlx_e8m0(unsigned int c) {
    return c == 255u ? NAN : @LDEXP@(1.0f, int(c) - 127);
}
@INLINE@ float cumlx_scale(unsigned int c, int fmt) {
    return fmt == 0 ? cumlx_e4m3(c) : cumlx_e8m0(c);
}
@INLINE@ float cumlx_restore(unsigned int c, unsigned int s, float global, int fmt) {
    // Explicit order matches the reference decoder. No half-precision weight rounding.
    return (cumlx_e2m1(c) * cumlx_scale(s, fmt)) * global;
}
@INLINE@ unsigned int cumlx_encode_e2m1(float x) {
    float a = @MIN@(@ABS@(x), 6.0f); // saturate before distance comparisons
    unsigned int best = 0u;
    float distance = a;
    for (unsigned int c = 1u; c < 8u; ++c) {
        float d = @ABS@(a - cumlx_e2m1(c));
        if (d < distance || (d == distance && (c & 1u) == 0u)) {
            distance = d; best = c;
        }
    }
    return best | (@SIGNBIT@(x) ? 8u : 0u);
}
@INLINE@ unsigned int cumlx_encode_e4m3(float x) {
    if (x <= 0.0f) return 0u;
    if (x >= 448.0f) return 126u;
    unsigned int lo = 0u, hi = 126u;
    while (lo < hi) {
        unsigned int mid = (lo + hi) / 2u;
        if (cumlx_e4m3(mid) < x) lo = mid + 1u; else hi = mid;
    }
    unsigned int right = lo, left = right == 0u ? 0u : right - 1u;
    float dl = @ABS@(x - cumlx_e4m3(left));
    float dr = @ABS@(cumlx_e4m3(right) - x);
    unsigned int result = dr < dl || (dr == dl && (right & 1u) == 0u) ? right : left;
    // Pin the CUMLX nonzero-block scale floor; not an implicit vendor recipe.
    return result == 0u ? 1u : result;
}
@INLINE@ unsigned int cumlx_encode_e8m0_ceil(float x) {
    unsigned int lo = 0u, hi = 254u;
    while (lo < hi) {
        unsigned int mid = (lo + hi) / 2u;
        if (cumlx_e8m0(mid) < x) lo = mid + 1u; else hi = mid;
    }
    return lo;
}
@INLINE@ float cumlx_silu(float x) {
    float z = @EXP@(-@ABS@(x));
    float sigmoid = x >= 0.0f ? 1.0f / (1.0f + z) : z / (1.0f + z);
    return x * sigmoid;
}
'''

DECODE=r'''
// Diagnostic full reconstruction only. Packed GEMV/GEMM never invoke this kernel.
unsigned int i = @GID@;
if (i >= N * K) return;
unsigned int n = i / K, k = i % K;
unsigned int p = wq[n * WP + k / 2u];
unsigned int code = (p >> ((k & 1u) * 4u)) & 15u;
y[i] = cumlx_restore(code, ws[n * WSG + k / WG], wglobal[0], WF);
'''

GEMV=r'''
// BM=1: GEMV. BM=4: reuse the decoded pair across up to four activation rows.
// One 128-lane workgroup owns one output column and BM rows. No warp assumptions.
unsigned int lane = @LID@;
unsigned int group = @GROUP@;
unsigned int n = group % N;
unsigned int m0 = (group / N) * BM;
@SHARED@ float partial[BM * 128];
float accum[BM];
for (unsigned int r = 0u; r < BM; ++r) accum[r] = 0.0f;
for (unsigned int k = lane * 2u; k < K; k += 256u) {
    unsigned int pair = wq[n * WP + k / 2u];
    unsigned int scale = ws[n * WSG + k / WG];
    float w0 = cumlx_restore(pair & 15u, scale, wglobal[0], WF);
    float w1 = k + 1u < K ? cumlx_restore(pair >> 4u, scale, wglobal[0], WF) : 0.0f;
    for (unsigned int r = 0u; r < BM; ++r) {
        if (m0 + r < M) {
            float a0 = float(x[(m0 + r) * K + k]);
            accum[r] = @FMA@(a0, w0, accum[r]);
            if (k + 1u < K) {
                float a1 = float(x[(m0 + r) * K + k + 1u]);
                accum[r] = @FMA@(a1, w1, accum[r]);
            }
        }
    }
}
for (unsigned int r = 0u; r < BM; ++r) partial[r * 128u + lane] = accum[r];
@BARRIER@;
// All 128 lanes execute every barrier, including lanes with no K contribution.
for (unsigned int step = 64u; step > 0u; step >>= 1u) {
    if (lane < step) {
        for (unsigned int r = 0u; r < BM; ++r)
            partial[r * 128u + lane] += partial[r * 128u + lane + step];
    }
    @BARRIER@;
}
if (lane == 0u) {
    for (unsigned int r = 0u; r < BM; ++r) {
        if (m0 + r < M) {
            float result = partial[r * 128u];
            if (HAS_BIAS) result += bias[n];
            if (APPLY_SILU) result = cumlx_silu(result);
            y[(m0 + r) * N + n] = result;
        }
    }
}
'''

TILED=r'''
// BM=8, BN=32, BK=32, 128 lanes; each lane accumulates two output columns.
// Shared memory: (8*32 + 32*32)*4 = 5120 bytes, independent of full model size.
unsigned int lane = @LID@;
unsigned int group = @GROUP@;
unsigned int tiles_n = (N + 31u) / 32u;
unsigned int m0 = (group / tiles_n) * 8u;
unsigned int n0 = (group % tiles_n) * 32u;
unsigned int row = lane / 16u, col = lane % 16u;
@SHARED@ float tile_a[8 * 32];
@SHARED@ float tile_w[32 * 32];
float acc0 = 0.0f, acc1 = 0.0f;
for (unsigned int k0 = 0u; k0 < K; k0 += 32u) {
    for (unsigned int t = lane; t < 8u * 32u; t += 128u) {
        unsigned int r = t / 32u, kk = t % 32u;
        float value = 0.0f;
        if (m0 + r < M && k0 + kk < K) {
            @LOAD_A@
        }
        tile_a[t] = value;
    }
    for (unsigned int t = lane; t < 32u * 32u; t += 128u) {
        unsigned int n = n0 + t / 32u, k = k0 + t % 32u;
        float value = 0.0f;
        if (n < N && k < K) {
            unsigned int pair = wq[n * WP + k / 2u];
            unsigned int code = (pair >> ((k & 1u) * 4u)) & 15u;
            value = cumlx_restore(code, ws[n * WSG + k / WG], wglobal[0], WF);
        }
        tile_w[t] = value;
    }
    @BARRIER@;
    // There is no early return on partial M/N tiles: all threads must synchronize.
    for (unsigned int kk = 0u; kk < 32u; ++kk) {
        float a = tile_a[row * 32u + kk];
        acc0 = @FMA@(a, tile_w[col * 32u + kk], acc0);
        acc1 = @FMA@(a, tile_w[(col + 16u) * 32u + kk], acc1);
    }
    @BARRIER@; // Finish all reads before the next K tile overwrites shared memory.
}
if (m0 + row < M) {
    if (n0 + col < N) {
        if (HAS_BIAS) acc0 += bias[n0 + col];
        if (APPLY_SILU) acc0 = cumlx_silu(acc0);
        y[(m0 + row) * N + n0 + col] = acc0;
    }
    if (n0 + col + 16u < N) {
        if (HAS_BIAS) acc1 += bias[n0 + col + 16u];
        if (APPLY_SILU) acc1 = cumlx_silu(acc1);
        y[(m0 + row) * N + n0 + col + 16u] = acc1;
    }
}
'''
LOAD_FLOAT='value = float(x[(m0 + r) * K + k0 + kk]);'
LOAD_PACKED=r'''
unsigned int m = m0 + r, k = k0 + kk;
unsigned int pair = aq[m * AP + k / 2u];
unsigned int code = (pair >> ((k & 1u) * 4u)) & 15u;
value = cumlx_restore(code, ascales[m * ASG + k / AG], aglobal[0], AF);
'''

QUANTIZE=r'''
// One scalar lane owns one block and its scale byte; writes never overlap.
// The global-amax reduction is supplied by the MLX graph, outside this kernel.
unsigned int block = @GID@;
if (block >= N * SG) return;
unsigned int n = block / SG, b = block % SG, k0 = b * G;
float amax = 0.0f;
for (unsigned int j = 0u; j < G && k0 + j < K; ++j)
    amax = @MAX@(amax, @ABS@(float(x[n * K + k0 + j])));
unsigned int sc;
if (F == 0) {
    float raw = (amax / global[0]) / 6.0f;
    sc = cumlx_encode_e4m3(raw);
} else {
    sc = amax == 0.0f ? 127u : cumlx_encode_e8m0_ceil(amax / 6.0f);
}
outs[block] = sc;
float divisor = cumlx_scale(sc, F);
for (unsigned int j = 0u; j < G && k0 + j < K; j += 2u) {
    float v0 = float(x[n * K + k0 + j]);
    float z0 = divisor > 0.0f ? (v0 / global[0]) / divisor : @COPYSIGN@(0.0f, v0);
    unsigned int c0 = cumlx_encode_e2m1(@MIN@(6.0f, @MAX@(-6.0f, z0)));
    unsigned int c1 = 0u; // canonical zero high nibble for an odd K tail
    if (k0 + j + 1u < K) {
        float v1 = float(x[n * K + k0 + j + 1u]);
        float z1 = divisor > 0.0f ? (v1 / global[0]) / divisor : @COPYSIGN@(0.0f, v1);
        c1 = cumlx_encode_e2m1(@MIN@(6.0f, @MAX@(-6.0f, z1)));
    }
    outq[n * P + (k0 + j) / 2u] = c0 | (c1 << 4u);
}
'''

RMS=r'''
// One workgroup per row. Residual addition and accumulation take place in FP32.
unsigned int lane = @LID@, row = @GROUP@;
@SHARED@ float square_sum[128];
float partial = 0.0f;
for (unsigned int k = lane; k < K; k += 128u) {
    float v = float(x[row * K + k]);
    if (HAS_RESIDUAL) v += float(residual[row * K + k]);
    partial = @FMA@(v, v, partial);
}
square_sum[lane] = partial;
@BARRIER@;
for (unsigned int step = 64u; step > 0u; step >>= 1u) {
    if (lane < step) square_sum[lane] += square_sum[lane + step];
    @BARRIER@;
}
float inverse_rms = 1.0f / @SQRT@(square_sum[0] / float(K) + epsilon[0]);
for (unsigned int k = lane; k < K; k += 128u) {
    float v = float(x[row * K + k]);
    if (HAS_RESIDUAL) v += float(residual[row * K + k]);
    y[row * K + k] = (v * inverse_rms) * float(weight[k]);
}
'''
SILU=r'''
unsigned int i = @GID@;
if (i >= SIZE) return;
float g = float(gate[i]), u = float(up[i]);
y[i] = cumlx_silu(g) * u;
'''


def render(text, backend):
    if backend=='metal':
        rep={'INLINE':'inline','LDEXP':'metal::ldexp','COPYSIGN':'metal::copysign','SIGNBIT':'metal::signbit',
             'ABS':'metal::abs','EXP':'metal::exp','MAX':'metal::max','MIN':'metal::min','SQRT':'metal::sqrt',
             'FMA':'metal::fma','SHARED':'threadgroup','BARRIER':'threadgroup_barrier(mem_flags::mem_threadgroup)',
             'LID':'thread_position_in_threadgroup.x','GID':'thread_position_in_grid.x',
             'GROUP':'threadgroup_position_in_grid.x'}
    else:
        rep={'INLINE':'__device__ __forceinline__','LDEXP':'ldexpf','COPYSIGN':'copysignf','SIGNBIT':'signbit',
             'ABS':'fabsf','EXP':'expf','MAX':'fmaxf','MIN':'fminf','SQRT':'sqrtf','FMA':'fmaf',
             'SHARED':'__shared__','BARRIER':'__syncthreads()',
             'LID':'threadIdx.x','GID':'(blockIdx.x * blockDim.x + threadIdx.x)','GROUP':'blockIdx.x'}
    for k,v in rep.items(): text=text.replace('@'+k+'@',v)
    assert '@' not in text
    return '// CUMLX v0.3. Inference-only candidate; physical GPU validation required.\n'+text.strip()+'\n'


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    bodies={'fp4_decode':DECODE,'fp4_gemv':GEMV,'fp4_small_m':GEMV,
            'fp4_tiled':TILED.replace('@LOAD_A@',LOAD_FLOAT),
            'fp4_w4a4':TILED.replace('@LOAD_A@',LOAD_PACKED),
            'fp4_quantize':QUANTIZE,'rms_norm':RMS,'silu_mul':SILU}
    for backend,ext in [('metal','metal'),('cuda','cu')]:
        prefix='#include <metal_stdlib>\nusing namespace metal;\n' if backend=='metal' else '#include <math.h>\n#include <stdint.h>\n'
        (ROOT/f'common_{backend}.h').write_text(prefix+render(COMMON,backend))
        for name,body in bodies.items(): (ROOT/f'{name}.{ext}').write_text(render(body,backend))


if __name__=='__main__': main()
