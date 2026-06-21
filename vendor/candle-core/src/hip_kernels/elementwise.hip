// Elementwise HIP kernels for the candle-core HIP backend (Phase 45 / ADR 073).
//
// Each `extern "C"` wrapper launches its kernel and synchronizes, returning 0 on
// success. `src/hip_backend.rs` dlopen's this (built by build.rs) and calls the
// wrappers with device pointers. f32 + contiguous for the first pass; strided /
// f16 / bf16 follow. Op codes are mirrored by the Rust dispatch on `B::NAME`.
#include <hip/hip_runtime.h>
#include <math.h>

// candle's `Gelu` is the tanh approximation (its `gelu_erf` is the exact one).
__device__ inline float gelu_tanh(float x) {
    const float c = 0.7978845608028654f; // sqrt(2/pi)
    return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
}

__global__ void affine_f32(const float* x, float* y, size_t n, float mul, float add) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] * mul + add;
}

// op: 0 neg, 1 exp, 2 silu, 3 gelu, 4 sqr, 5 sqrt, 6 recip, 7 relu, 8 tanh, 9 abs
__global__ void unary_f32(const float* x, float* y, size_t n, int op) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    switch (op) {
        case 0: y[i] = -v; break;
        case 1: y[i] = expf(v); break;
        case 2: y[i] = v / (1.0f + expf(-v)); break; // silu
        case 3: y[i] = gelu_tanh(v); break;
        case 4: y[i] = v * v; break;
        case 5: y[i] = sqrtf(v); break;
        case 6: y[i] = 1.0f / v; break;
        case 7: y[i] = v > 0.0f ? v : 0.0f; break;
        case 8: y[i] = tanhf(v); break;
        case 9: y[i] = fabsf(v); break;
        default: y[i] = v;
    }
}

// op: 0 add, 1 sub, 2 mul, 3 div
__global__ void binary_f32(const float* a, const float* b, float* c, size_t n, int op) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    switch (op) {
        case 0: c[i] = a[i] + b[i]; break;
        case 1: c[i] = a[i] - b[i]; break;
        case 2: c[i] = a[i] * b[i]; break;
        case 3: c[i] = a[i] / b[i]; break;
        default: c[i] = a[i];
    }
}

extern "C" {

int akh_affine_f32(const float* x, float* y, size_t n, float mul, float add) {
    if (n == 0) return 0;
    int t = 256;
    size_t blocks = (n + t - 1) / t;
    hipLaunchKernelGGL(affine_f32, dim3(blocks), dim3(t), 0, 0, x, y, n, mul, add);
    if (hipGetLastError() != hipSuccess) return 1;
    return hipDeviceSynchronize() == hipSuccess ? 0 : 2;
}

int akh_unary_f32(const float* x, float* y, size_t n, int op) {
    if (n == 0) return 0;
    int t = 256;
    size_t blocks = (n + t - 1) / t;
    hipLaunchKernelGGL(unary_f32, dim3(blocks), dim3(t), 0, 0, x, y, n, op);
    if (hipGetLastError() != hipSuccess) return 1;
    return hipDeviceSynchronize() == hipSuccess ? 0 : 2;
}

int akh_binary_f32(const float* a, const float* b, float* c, size_t n, int op) {
    if (n == 0) return 0;
    int t = 256;
    size_t blocks = (n + t - 1) / t;
    hipLaunchKernelGGL(binary_f32, dim3(blocks), dim3(t), 0, 0, a, b, c, n, op);
    if (hipGetLastError() != hipSuccess) return 1;
    return hipDeviceSynchronize() == hipSuccess ? 0 : 2;
}

} // extern "C"
