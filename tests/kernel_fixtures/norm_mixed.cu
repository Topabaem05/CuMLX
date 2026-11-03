// Written for CUMLX tests. Device code among host code that needs a large framework header: the second wall that M0 found.
// It also has a device template helper that is called with explicit arguments, a local include, and PTX built-ins.
#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include "norm_constants.h"

namespace cumlx_test {

template <typename T>
__device__ __forceinline__ T scale_of(T x) {
    return x * NORM_SCALE;
}

__global__ __launch_bounds__(4) void norm_mixed(const float* x, float* out) {
    __shared__ float partial[4];
    unsigned int lane = threadIdx.x;
    partial[lane] = scale_of<float>(x[blockIdx.x * 4u + lane]);
    __syncthreads();
    float sum = 0.0f;
    for (unsigned int i = 0u; i < 4u; ++i) sum += partial[i] * partial[i];
    out[blockIdx.x * 4u + lane] = partial[lane] * rsqrtf(sum);
}

}  // namespace cumlx_test

at::Tensor run_on_tensor(const at::Tensor& input) {
    at::Tensor output = at::zeros_like(input);
    cumlx_test::norm_mixed<<<1, 4>>>(input.data_ptr<float>(), output.data_ptr<float>());
    return output;
}
