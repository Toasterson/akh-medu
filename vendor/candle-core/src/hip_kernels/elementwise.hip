// Elementwise HIP kernels for the candle-core HIP backend (Phase 45 / ADR 073).
//
// Each `extern "C"` wrapper launches its kernel and synchronizes, returning 0 on
// success. `src/hip_backend.rs` dlopen's this (built by build.rs) and calls the
// wrappers with device pointers. f32 + contiguous for the first pass; strided /
// f16 / bf16 follow. Op codes are mirrored by the Rust dispatch on `B::NAME`.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <math.h>
#include <stdint.h>

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

// ---- reduce (Phase 45d) ----------------------------------------------------
// One output element per thread. `op`: 0 sum, 1 min, 2 max. `dims`/`strides`/
// `reduce_mask` are DEVICE arrays of length `num_dims` (input is contiguous, so
// strides are row-major). Each thread reconstructs the kept-dim coords of its
// output index, forms the base input offset, then loops over the cartesian
// product of the reduced dims accumulating per `op`.
__global__ void reduce_f32(const float* x, float* y, size_t out_numel,
                           int num_dims, const int* dims, const long* strides,
                           const int* reduce_mask, size_t reduce_numel, int op) {
    size_t out_i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (out_i >= out_numel) return;
    size_t base = 0;
    size_t rem = out_i;
    for (int d = num_dims - 1; d >= 0; --d) {
        if (reduce_mask[d]) continue;
        int sz = dims[d];
        int coord = (int)(rem % (size_t)sz);
        rem /= (size_t)sz;
        base += (size_t)coord * (size_t)strides[d];
    }
    float acc = 0.0f;
    bool first = true;
    for (size_t r = 0; r < reduce_numel; ++r) {
        size_t off = base;
        size_t rr = r;
        for (int d = num_dims - 1; d >= 0; --d) {
            if (!reduce_mask[d]) continue;
            int sz = dims[d];
            int coord = (int)(rr % (size_t)sz);
            rr /= (size_t)sz;
            off += (size_t)coord * (size_t)strides[d];
        }
        float v = x[off];
        if (first) { acc = v; first = false; }
        else if (op == 0) { acc += v; }
        else if (op == 1) { acc = v < acc ? v : acc; }
        else { acc = v > acc ? v : acc; }
    }
    y[out_i] = acc;
}

// ---- index_select (Phase 45d) ----------------------------------------------
// Source viewed as [left, dim_size, right]; output [left, n_idx, right].
// out[(l*n_idx + j)*right + r] = src[(l*dim_size + ids[j])*right + r].
__global__ void index_select_u32_f32(
    const float* src, const unsigned int* ids, float* out,
    size_t left, size_t dim_size, size_t n_idx, size_t right) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = left * n_idx * right;
    if (i >= total) return;
    size_t r = i % right;
    size_t j = (i / right) % n_idx;
    size_t l = i / (right * n_idx);
    size_t idx = (size_t)ids[j];
    out[i] = src[(l * dim_size + idx) * right + r];
}
__global__ void index_select_i64_f32(
    const float* src, const long long* ids, float* out,
    size_t left, size_t dim_size, size_t n_idx, size_t right) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = left * n_idx * right;
    if (i >= total) return;
    size_t r = i % right;
    size_t j = (i / right) % n_idx;
    size_t l = i / (right * n_idx);
    size_t idx = (size_t)ids[j];
    out[i] = src[(l * dim_size + idx) * right + r];
}

// ---- where_cond (Phase 45d): out[i] = cond[i] ? t[i] : f[i] -----------------
__global__ void where_cond_u8_f32(const unsigned char* cond, const float* t,
                                  const float* f, float* out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = cond[i] ? t[i] : f[i];
}
__global__ void where_cond_u32_f32(const unsigned int* cond, const float* t,
                                   const float* f, float* out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = cond[i] ? t[i] : f[i];
}

// ---- strided copy (Phase 45d): powers .contiguous() ------------------------
// dst[o] = src[strided_index(o)]; `info` is [dims[0..rank], strides[0..rank]]
// uploaded to device; src/dst are already advanced to their start offsets.
__device__ inline size_t strided_src_index(size_t o, size_t rank, const size_t* dims, const size_t* strides) {
    size_t src = 0;
    for (size_t d = rank; d-- > 0;) {
        size_t dim = dims[d];
        size_t coord = o % dim;
        o /= dim;
        src += coord * strides[d];
    }
    return src;
}
__global__ void ucopy_f32(const float* src, float* dst, size_t n, size_t rank, const size_t* info) {
    size_t o = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= n) return;
    const size_t* dims = info;
    const size_t* strides = info + rank;
    dst[o] = src[strided_src_index(o, rank, dims, strides)];
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

// reduce: dims/strides/reduce_mask are DEVICE pointers (caller uploads + frees).
int akh_reduce_f32(const float* x, float* y, size_t out_numel, int num_dims,
                   const int* dims, const long* strides, const int* reduce_mask,
                   size_t reduce_numel, int op) {
    if (out_numel == 0) return 0;
    int t = 256;
    size_t blocks = (out_numel + t - 1) / t;
    hipLaunchKernelGGL(reduce_f32, dim3(blocks), dim3(t), 0, 0, x, y, out_numel,
                       num_dims, dims, strides, reduce_mask, reduce_numel, op);
    if (hipGetLastError() != hipSuccess) return 1;
    return hipDeviceSynchronize() == hipSuccess ? 0 : 2;
}

int akh_index_select_u32_f32(
    const float* src, const unsigned int* ids, float* out,
    size_t left, size_t dim_size, size_t n_idx, size_t right) {
    size_t total = left * n_idx * right;
    if (total == 0) return 0;
    int t = 256;
    size_t blocks = (total + t - 1) / t;
    hipLaunchKernelGGL(index_select_u32_f32, dim3(blocks), dim3(t), 0, 0,
                       src, ids, out, left, dim_size, n_idx, right);
    if (hipGetLastError() != hipSuccess) return 1;
    return hipDeviceSynchronize() == hipSuccess ? 0 : 2;
}
int akh_index_select_i64_f32(
    const float* src, const long long* ids, float* out,
    size_t left, size_t dim_size, size_t n_idx, size_t right) {
    size_t total = left * n_idx * right;
    if (total == 0) return 0;
    int t = 256;
    size_t blocks = (total + t - 1) / t;
    hipLaunchKernelGGL(index_select_i64_f32, dim3(blocks), dim3(t), 0, 0,
                       src, ids, out, left, dim_size, n_idx, right);
    if (hipGetLastError() != hipSuccess) return 1;
    return hipDeviceSynchronize() == hipSuccess ? 0 : 2;
}

int akh_where_cond_u8_f32(const unsigned char* cond, const float* t, const float* f,
                          float* out, size_t n) {
    if (n == 0) return 0;
    int tpb = 256;
    size_t blocks = (n + tpb - 1) / tpb;
    hipLaunchKernelGGL(where_cond_u8_f32, dim3(blocks), dim3(tpb), 0, 0, cond, t, f, out, n);
    if (hipGetLastError() != hipSuccess) return 1;
    return hipDeviceSynchronize() == hipSuccess ? 0 : 2;
}
int akh_where_cond_u32_f32(const unsigned int* cond, const float* t, const float* f,
                           float* out, size_t n) {
    if (n == 0) return 0;
    int tpb = 256;
    size_t blocks = (n + tpb - 1) / tpb;
    hipLaunchKernelGGL(where_cond_u32_f32, dim3(blocks), dim3(tpb), 0, 0, cond, t, f, out, n);
    if (hipGetLastError() != hipSuccess) return 1;
    return hipDeviceSynchronize() == hipSuccess ? 0 : 2;
}

// strided gather-copy: src/dst already advanced to start offsets; info is a device
// buffer of 2*rank size_t [dims..., strides...].
int akh_ucopy_f32(const float* src, float* dst, size_t n, size_t rank, const size_t* info) {
    if (n == 0) return 0;
    int t = 256;
    size_t blocks = (n + t - 1) / t;
    hipLaunchKernelGGL(ucopy_f32, dim3(blocks), dim3(t), 0, 0, src, dst, n, rank, info);
    if (hipGetLastError() != hipSuccess) return 1;
    return hipDeviceSynchronize() == hipSuccess ? 0 : 2;
}

} // extern "C"

// ---- dtype casts (Phase 45d) -----------------------------------------------
// One templated elementwise cast; `Src`/`Dst` are the natural device types
// (float, __half, hip_bfloat16, uint32_t, int64_t). `static_cast<Dst>(Src)`
// routes the half types through HIP's converting ctors/operators (via f32),
// matching candle's CPU cast within rounding. NOTE: this ROCm exposes the legacy
// `hip_bfloat16` struct (round-to-nearest f32->bf16, zero-extend bf16->f32) in
// <hip/hip_bfloat16.h> — not `__hip_bfloat16` (that name is in <hip/hip_bf16.h>).
template <typename Src, typename Dst>
__global__ void cast_kernel(const Src* x, Dst* y, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = static_cast<Dst>(x[i]);
}

#define AKH_CAST_LAUNCH(SRC, DST, xptr, yptr, n)                                \
    do {                                                                       \
        if ((n) == 0) return 0;                                                \
        int t = 256;                                                           \
        size_t blocks = ((n) + t - 1) / t;                                     \
        hipLaunchKernelGGL((cast_kernel<SRC, DST>), dim3(blocks), dim3(t), 0,  \
                           0, (xptr), (yptr), (n));                            \
        if (hipGetLastError() != hipSuccess) return 1;                         \
        return hipDeviceSynchronize() == hipSuccess ? 0 : 2;                   \
    } while (0)

// f16 <-> bf16 route through f32 explicitly (no direct converting ctor).
__global__ void cast_f16_bf16_kernel(const __half* x, hip_bfloat16* y, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = static_cast<hip_bfloat16>(__half2float(x[i]));
}
__global__ void cast_bf16_f16_kernel(const hip_bfloat16* x, __half* y, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = __float2half(static_cast<float>(x[i]));
}

extern "C" {

int akh_cast_f32_f16(const float* x, __half* y, size_t n) { AKH_CAST_LAUNCH(float, __half, x, y, n); }
int akh_cast_f16_f32(const __half* x, float* y, size_t n) { AKH_CAST_LAUNCH(__half, float, x, y, n); }
int akh_cast_f32_bf16(const float* x, hip_bfloat16* y, size_t n) { AKH_CAST_LAUNCH(float, hip_bfloat16, x, y, n); }
int akh_cast_bf16_f32(const hip_bfloat16* x, float* y, size_t n) { AKH_CAST_LAUNCH(hip_bfloat16, float, x, y, n); }
int akh_cast_f16_bf16(const __half* x, hip_bfloat16* y, size_t n) {
    if (n == 0) return 0;
    int t = 256;
    size_t blocks = (n + t - 1) / t;
    hipLaunchKernelGGL(cast_f16_bf16_kernel, dim3(blocks), dim3(t), 0, 0, x, y, n);
    if (hipGetLastError() != hipSuccess) return 1;
    return hipDeviceSynchronize() == hipSuccess ? 0 : 2;
}
int akh_cast_bf16_f16(const hip_bfloat16* x, __half* y, size_t n) {
    if (n == 0) return 0;
    int t = 256;
    size_t blocks = (n + t - 1) / t;
    hipLaunchKernelGGL(cast_bf16_f16_kernel, dim3(blocks), dim3(t), 0, 0, x, y, n);
    if (hipGetLastError() != hipSuccess) return 1;
    return hipDeviceSynchronize() == hipSuccess ? 0 : 2;
}
int akh_cast_f32_u32(const float* x, uint32_t* y, size_t n) { AKH_CAST_LAUNCH(float, uint32_t, x, y, n); }
int akh_cast_u32_f32(const uint32_t* x, float* y, size_t n) { AKH_CAST_LAUNCH(uint32_t, float, x, y, n); }
int akh_cast_f32_i64(const float* x, int64_t* y, size_t n) { AKH_CAST_LAUNCH(float, int64_t, x, y, n); }
int akh_cast_i64_f32(const int64_t* x, float* y, size_t n) { AKH_CAST_LAUNCH(int64_t, float, x, y, n); }

} // extern "C"
