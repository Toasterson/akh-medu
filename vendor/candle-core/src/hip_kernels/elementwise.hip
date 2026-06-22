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

// Phase 45g: fixed-rank metadata passed BY VALUE in the kernel launch params (rides
// along with the launch — no per-op hipMalloc + *synchronous* hipMemcpy + hipFree of
// a device dims/strides array, which stalled the stream ~4000×/token). Rank ≤ 8 covers
// every tensor the decode path produces (gemma's deepest is rank 5).
#define AKH_MAXR 8
struct AkhI64x8 { long v[AKH_MAXR]; };

// candle's `Gelu` is the tanh approximation (its `gelu_erf` is the exact one).
__device__ inline float gelu_tanh(float x) {
    const float c = 0.7978845608028654f; // sqrt(2/pi)
    return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
}

__global__ void affine_f32(const float* x, float* y, size_t n, float mul, float add) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] * mul + add;
}

// op: 0 neg, 1 exp, 2 silu, 3 gelu, 4 sqr, 5 sqrt, 6 recip, 7 relu, 8 tanh, 9 abs,
//     10 cos, 11 sin, 12 log
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
        case 10: y[i] = cosf(v); break;
        case 11: y[i] = sinf(v); break;
        case 12: y[i] = logf(v); break;
        default: y[i] = v;
    }
}

// Strided unary (Phase 45f): input may be non-contiguous (e.g. `neg` on a narrow
// view inside rotate_half / rope_slow). `dims`/`x_str` are DEVICE long arrays of
// length `rank`; `x` already advanced to its start offset; out contiguous. Same op
// codes as unary_f32.
__global__ void unary_strided_f32(const float* x, float* y, size_t n, size_t rank,
                                  AkhI64x8 dims, AkhI64x8 x_str, int op) {
    size_t o = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= n) return;
    size_t x_off = 0, rem = o;
    for (size_t d = rank; d-- > 0;) {
        size_t dim = (size_t)dims.v[d];
        size_t coord = rem % dim;
        rem /= dim;
        x_off += coord * (size_t)x_str.v[d];
    }
    float v = x[x_off];
    switch (op) {
        case 0: y[o] = -v; break;
        case 1: y[o] = expf(v); break;
        case 2: y[o] = v / (1.0f + expf(-v)); break;
        case 3: y[o] = gelu_tanh(v); break;
        case 4: y[o] = v * v; break;
        case 5: y[o] = sqrtf(v); break;
        case 6: y[o] = 1.0f / v; break;
        case 7: y[o] = v > 0.0f ? v : 0.0f; break;
        case 8: y[o] = tanhf(v); break;
        case 9: y[o] = fabsf(v); break;
        case 10: y[o] = cosf(v); break;
        case 11: y[o] = sinf(v); break;
        case 12: y[o] = logf(v); break;
        default: y[o] = v;
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

// ---- strided binary (Phase 45f): broadcasting add/sub/mul/div --------------
// Like `binary_f32` but each operand carries its own strides, so broadcast layouts
// (stride-0 dims from `broadcast_mul`/`broadcast_div`, e.g. inside rms_norm/softmax/
// rope) work without a prior `.contiguous()`. `dims`/`a_str`/`b_str` are DEVICE long
// arrays of length `rank`; `a`/`b` already advanced to start offsets; out contiguous.
// op: 0 add, 1 sub, 2 mul, 3 div.
__global__ void binary_strided_f32(const float* a, const float* b, float* c, size_t n,
                                   size_t rank, AkhI64x8 dims, AkhI64x8 a_str,
                                   AkhI64x8 b_str, int op) {
    size_t o = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= n) return;
    size_t a_off = 0, b_off = 0, rem = o;
    for (size_t d = rank; d-- > 0;) {
        size_t dim = (size_t)dims.v[d];
        size_t coord = rem % dim;
        rem /= dim;
        a_off += coord * (size_t)a_str.v[d];
        b_off += coord * (size_t)b_str.v[d];
    }
    float av = a[a_off], bv = b[b_off];
    switch (op) {
        case 0: c[o] = av + bv; break;
        case 1: c[o] = av - bv; break;
        case 2: c[o] = av * bv; break;
        case 3: c[o] = av / bv; break;
        default: c[o] = av;
    }
}

// ---- reduce (Phase 45d) ----------------------------------------------------
// One output element per thread. `op`: 0 sum, 1 min, 2 max. `dims`/`strides`/
// `reduce_mask` are DEVICE arrays of length `num_dims` (input is contiguous, so
// strides are row-major). Each thread reconstructs the kept-dim coords of its
// output index, forms the base input offset, then loops over the cartesian
// product of the reduced dims accumulating per `op`.
__global__ void reduce_f32(const float* x, float* y, size_t out_numel,
                           int num_dims, AkhI64x8 dims, AkhI64x8 strides,
                           AkhI64x8 reduce_mask, size_t reduce_numel, int op) {
    size_t out_i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (out_i >= out_numel) return;
    size_t base = 0;
    size_t rem = out_i;
    for (int d = num_dims - 1; d >= 0; --d) {
        if (reduce_mask.v[d]) continue;
        int sz = (int)dims.v[d];
        int coord = (int)(rem % (size_t)sz);
        rem /= (size_t)sz;
        base += (size_t)coord * (size_t)strides.v[d];
    }
    float acc = 0.0f;
    bool first = true;
    for (size_t r = 0; r < reduce_numel; ++r) {
        size_t off = base;
        size_t rr = r;
        for (int d = num_dims - 1; d >= 0; --d) {
            if (!reduce_mask.v[d]) continue;
            int sz = (int)dims.v[d];
            int coord = (int)(rr % (size_t)sz);
            rr /= (size_t)sz;
            off += (size_t)coord * (size_t)strides.v[d];
        }
        float v = x[off];
        if (first) { acc = v; first = false; }
        else if (op == 0) { acc += v; }
        else if (op == 1) { acc = v < acc ? v : acc; }
        else { acc = v > acc ? v : acc; }
    }
    y[out_i] = acc;
}

// ---- argmin/argmax (Phase 45f): reduce -> u32 index ------------------------
// One output element per thread; single reduced dim (candle restricts arg* to one
// dim), so the running index `r` IS the coordinate along that dim. op: 0 argmin,
// 1 argmax. Same dims/strides/reduce_mask metadata as reduce_f32. Ties resolve to
// the first occurrence (matches candle's CPU `<`/`>` accumulation).
__global__ void reduce_arg_f32(const float* x, unsigned int* y, size_t out_numel,
                               int num_dims, AkhI64x8 dims, AkhI64x8 strides,
                               AkhI64x8 reduce_mask, size_t reduce_numel, int op) {
    size_t out_i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (out_i >= out_numel) return;
    size_t base = 0;
    size_t rem = out_i;
    for (int d = num_dims - 1; d >= 0; --d) {
        if (reduce_mask.v[d]) continue;
        int sz = (int)dims.v[d];
        int coord = (int)(rem % (size_t)sz);
        rem /= (size_t)sz;
        base += (size_t)coord * (size_t)strides.v[d];
    }
    float best = 0.0f;
    unsigned int best_idx = 0;
    bool first = true;
    for (size_t r = 0; r < reduce_numel; ++r) {
        size_t off = base;
        size_t rr = r;
        for (int d = num_dims - 1; d >= 0; --d) {
            if (!reduce_mask.v[d]) continue;
            int sz = (int)dims.v[d];
            int coord = (int)(rr % (size_t)sz);
            rr /= (size_t)sz;
            off += (size_t)coord * (size_t)strides.v[d];
        }
        float v = x[off];
        bool better = first || (op == 1 ? (v > best) : (v < best));
        if (better) {
            best = v;
            best_idx = (unsigned int)r;
            first = false;
        }
    }
    y[out_i] = best_idx;
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

// ---- cmp (Phase 45f): strided elementwise compare -> u8 --------------------
// Both inputs may be broadcast/non-contiguous (e.g. `mask.eq(0)` where the mask
// is a broadcast view and the scalar rhs has all-zero strides), so each thread
// reconstructs the output coords and forms each operand's offset from its own
// strides. `dims`/`a_str`/`b_str` are DEVICE arrays of length `rank`; `a`/`b`
// are already advanced to their layout start offsets. op: 0 eq 1 ne 2 le 3 ge
// 4 lt 5 gt (matches candle's `CmpOp`). Output is contiguous u8 (1/0).
__global__ void cmp_f32(const float* a, const float* b, unsigned char* out, size_t n,
                        size_t rank, const long* dims, const long* a_str,
                        const long* b_str, int op) {
    size_t o = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= n) return;
    size_t a_off = 0, b_off = 0, rem = o;
    for (size_t d = rank; d-- > 0;) {
        size_t dim = (size_t)dims[d];
        size_t coord = rem % dim;
        rem /= dim;
        a_off += coord * (size_t)a_str[d];
        b_off += coord * (size_t)b_str[d];
    }
    float av = a[a_off], bv = b[b_off];
    unsigned char r;
    switch (op) {
        case 0: r = av == bv; break;
        case 1: r = av != bv; break;
        case 2: r = av <= bv; break;
        case 3: r = av >= bv; break;
        case 4: r = av <  bv; break;
        case 5: r = av >  bv; break;
        default: r = 0;
    }
    out[o] = r;
}

// U32 variant of the strided compare (the attention mask is built as u32, so
// `mask.eq(0)` compares u32). Same layout contract; integer comparison.
__global__ void cmp_u32(const unsigned int* a, const unsigned int* b, unsigned char* out,
                        size_t n, size_t rank, const long* dims, const long* a_str,
                        const long* b_str, int op) {
    size_t o = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= n) return;
    size_t a_off = 0, b_off = 0, rem = o;
    for (size_t d = rank; d-- > 0;) {
        size_t dim = (size_t)dims[d];
        size_t coord = rem % dim;
        rem /= dim;
        a_off += coord * (size_t)a_str[d];
        b_off += coord * (size_t)b_str[d];
    }
    unsigned int av = a[a_off], bv = b[b_off];
    unsigned char r;
    switch (op) {
        case 0: r = av == bv; break;
        case 1: r = av != bv; break;
        case 2: r = av <= bv; break;
        case 3: r = av >= bv; break;
        case 4: r = av <  bv; break;
        case 5: r = av >  bv; break;
        default: r = 0;
    }
    out[o] = r;
}

// Strided where_cond (Phase 45f): any of cond/t/f may be broadcast/non-contiguous
// (e.g. `neg_inf.broadcast_as(...)` as the `t` operand). Each thread reconstructs
// output coords and forms each operand's offset from its own strides. out contiguous.
__global__ void where_cond_strided_u8_f32(const unsigned char* cond, const float* t,
    const float* f, float* out, size_t n, size_t rank, const long* dims,
    const long* c_str, const long* t_str, const long* f_str) {
    size_t o = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= n) return;
    size_t c_off = 0, t_off = 0, f_off = 0, rem = o;
    for (size_t d = rank; d-- > 0;) {
        size_t dim = (size_t)dims[d];
        size_t coord = rem % dim;
        rem /= dim;
        c_off += coord * (size_t)c_str[d];
        t_off += coord * (size_t)t_str[d];
        f_off += coord * (size_t)f_str[d];
    }
    out[o] = cond[c_off] ? t[t_off] : f[f_off];
}
__global__ void where_cond_strided_u32_f32(const unsigned int* cond, const float* t,
    const float* f, float* out, size_t n, size_t rank, const long* dims,
    const long* c_str, const long* t_str, const long* f_str) {
    size_t o = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= n) return;
    size_t c_off = 0, t_off = 0, f_off = 0, rem = o;
    for (size_t d = rank; d-- > 0;) {
        size_t dim = (size_t)dims[d];
        size_t coord = rem % dim;
        rem /= dim;
        c_off += coord * (size_t)c_str[d];
        t_off += coord * (size_t)t_str[d];
        f_off += coord * (size_t)f_str[d];
    }
    out[o] = cond[c_off] ? t[t_off] : f[f_off];
}

// ---- strided copy (Phase 45d): powers .contiguous() ------------------------
// dst[o] = src[strided_index(o)]; dims/strides passed by value (Phase 45g — was a
// per-call device `info` upload, ~96/token from KV-cache cat + contiguous). src/dst
// already advanced to their start offsets.
__global__ void ucopy_f32(const float* src, float* dst, size_t n, size_t rank,
                          AkhI64x8 dims, AkhI64x8 strides) {
    size_t o = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= n) return;
    size_t s = 0;
    size_t rem = o;
    for (size_t d = rank; d-- > 0;) {
        size_t dim = (size_t)dims.v[d];
        size_t coord = rem % dim;
        rem /= dim;
        s += coord * (size_t)strides.v[d];
    }
    dst[o] = src[s];
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
                   struct AkhI64x8 dims, struct AkhI64x8 strides, struct AkhI64x8 reduce_mask,
                   size_t reduce_numel, int op) {
    if (out_numel == 0) return 0;
    int t = 256;
    size_t blocks = (out_numel + t - 1) / t;
    hipLaunchKernelGGL(reduce_f32, dim3(blocks), dim3(t), 0, 0, x, y, out_numel,
                       num_dims, dims, strides, reduce_mask, reduce_numel, op);
    if (hipGetLastError() != hipSuccess) return 1;
    return hipDeviceSynchronize() == hipSuccess ? 0 : 2;
}

// argmin/argmax reduce -> u32 index. Same metadata as akh_reduce_f32; op 0 argmin, 1 argmax.
int akh_reduce_arg_f32(const float* x, unsigned int* y, size_t out_numel, int num_dims,
                       struct AkhI64x8 dims, struct AkhI64x8 strides, struct AkhI64x8 reduce_mask,
                       size_t reduce_numel, int op) {
    if (out_numel == 0) return 0;
    int t = 256;
    size_t blocks = (out_numel + t - 1) / t;
    hipLaunchKernelGGL(reduce_arg_f32, dim3(blocks), dim3(t), 0, 0, x, y, out_numel,
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

// strided broadcasting binary; dims/a_str/b_str are device long arrays of length
// rank, a/b already advanced to start offsets, out contiguous f32. op 0..3.
int akh_binary_strided_f32(const float* a, const float* b, float* c, size_t n,
                           size_t rank, struct AkhI64x8 dims, struct AkhI64x8 a_str,
                           struct AkhI64x8 b_str, int op) {
    if (n == 0) return 0;
    int t = 256;
    size_t blocks = (n + t - 1) / t;
    hipLaunchKernelGGL(binary_strided_f32, dim3(blocks), dim3(t), 0, 0,
                       a, b, c, n, rank, dims, a_str, b_str, op);
    if (hipGetLastError() != hipSuccess) return 1;
    return hipDeviceSynchronize() == hipSuccess ? 0 : 2;
}

// strided unary; dims/x_str passed by value, x advanced to its start offset, out
// contiguous f32. Same op codes as akh_unary_f32.
int akh_unary_strided_f32(const float* x, float* y, size_t n, size_t rank,
                          struct AkhI64x8 dims, struct AkhI64x8 x_str, int op) {
    if (n == 0) return 0;
    int t = 256;
    size_t blocks = (n + t - 1) / t;
    hipLaunchKernelGGL(unary_strided_f32, dim3(blocks), dim3(t), 0, 0,
                       x, y, n, rank, dims, x_str, op);
    if (hipGetLastError() != hipSuccess) return 1;
    return hipDeviceSynchronize() == hipSuccess ? 0 : 2;
}

// strided elementwise compare -> u8; dims/a_str/b_str are device long arrays of
// length rank, a/b already advanced to their start offsets, out contiguous u8.
int akh_cmp_f32(const float* a, const float* b, unsigned char* out, size_t n,
                size_t rank, const long* dims, const long* a_str, const long* b_str, int op) {
    if (n == 0) return 0;
    int t = 256;
    size_t blocks = (n + t - 1) / t;
    hipLaunchKernelGGL(cmp_f32, dim3(blocks), dim3(t), 0, 0, a, b, out, n, rank, dims, a_str, b_str, op);
    if (hipGetLastError() != hipSuccess) return 1;
    return hipDeviceSynchronize() == hipSuccess ? 0 : 2;
}

int akh_cmp_u32(const unsigned int* a, const unsigned int* b, unsigned char* out, size_t n,
                size_t rank, const long* dims, const long* a_str, const long* b_str, int op) {
    if (n == 0) return 0;
    int t = 256;
    size_t blocks = (n + t - 1) / t;
    hipLaunchKernelGGL(cmp_u32, dim3(blocks), dim3(t), 0, 0, a, b, out, n, rank, dims, a_str, b_str, op);
    if (hipGetLastError() != hipSuccess) return 1;
    return hipDeviceSynchronize() == hipSuccess ? 0 : 2;
}

// strided where_cond (cond/t/f own strides; dims/c_str/t_str/f_str device long
// arrays of length rank; cond/t/f advanced to start offsets; out contiguous f32).
int akh_where_cond_strided_u8_f32(const unsigned char* cond, const float* t, const float* f,
    float* out, size_t n, size_t rank, const long* dims, const long* c_str,
    const long* t_str, const long* f_str) {
    if (n == 0) return 0;
    int tpb = 256;
    size_t blocks = (n + tpb - 1) / tpb;
    hipLaunchKernelGGL(where_cond_strided_u8_f32, dim3(blocks), dim3(tpb), 0, 0,
                       cond, t, f, out, n, rank, dims, c_str, t_str, f_str);
    if (hipGetLastError() != hipSuccess) return 1;
    return hipDeviceSynchronize() == hipSuccess ? 0 : 2;
}
int akh_where_cond_strided_u32_f32(const unsigned int* cond, const float* t, const float* f,
    float* out, size_t n, size_t rank, const long* dims, const long* c_str,
    const long* t_str, const long* f_str) {
    if (n == 0) return 0;
    int tpb = 256;
    size_t blocks = (n + tpb - 1) / tpb;
    hipLaunchKernelGGL(where_cond_strided_u32_f32, dim3(blocks), dim3(tpb), 0, 0,
                       cond, t, f, out, n, rank, dims, c_str, t_str, f_str);
    if (hipGetLastError() != hipSuccess) return 1;
    return hipDeviceSynchronize() == hipSuccess ? 0 : 2;
}

// strided gather-copy: src/dst already advanced to start offsets; dims/strides by value.
int akh_ucopy_f32(const float* src, float* dst, size_t n, size_t rank,
                  struct AkhI64x8 dims, struct AkhI64x8 strides) {
    if (n == 0) return 0;
    int t = 256;
    size_t blocks = (n + t - 1) / t;
    hipLaunchKernelGGL(ucopy_f32, dim3(blocks), dim3(t), 0, 0, src, dst, n, rank, dims, strides);
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
