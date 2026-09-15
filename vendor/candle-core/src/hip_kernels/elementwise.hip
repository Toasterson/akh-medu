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

static hipStream_t g_akh_stream = 0; // Phase 46c: launch stream (0=default; set during graph capture)

// One value row per block. All state writes target a fresh allocation: snapshots
// keep their old storage. Each lane retains at most two decayed state elements.
__global__ void gated_delta_step_f32(
    const float* state, const float* q, const float* k, const float* v,
    const float* decay, const float* beta, float* output, float* next,
    size_t value_width, size_t key_width) {
    const size_t row = blockIdx.x;
    const size_t head = row / value_width;
    const unsigned lane = threadIdx.x;
    __shared__ float sums[128];
    float values[2] = {0.f, 0.f};
    float prediction = 0.f;
    for (unsigned part = 0; part < 2; ++part) {
        const size_t col = lane + 128 * part;
        if (col < key_width) {
            values[part] = state[row * key_width + col] * decay[head];
            prediction += values[part] * k[head * key_width + col];
        }
    }
    sums[lane] = prediction;
    __syncthreads();
    for (unsigned stride = 64; stride; stride >>= 1) {
        if (lane < stride) sums[lane] += sums[lane + stride];
        __syncthreads();
    }
    const float update = beta[head] * (v[row] - sums[0]);
    // Every lane must consume prediction before sums is reused below.
    __syncthreads();
    float result = 0.f;
    for (unsigned part = 0; part < 2; ++part) {
        const size_t col = lane + 128 * part;
        if (col < key_width) {
            const float updated = values[part] + update * k[head * key_width + col];
            next[row * key_width + col] = updated;
            result += updated * q[head * key_width + col];
        }
    }
    sums[lane] = result;
    __syncthreads();
    for (unsigned stride = 64; stride; stride >>= 1) {
        if (lane < stride) sums[lane] += sums[lane + stride];
        __syncthreads();
    }
    if (lane == 0) output[row] = sums[0] / sqrtf((float)key_width);
}

extern "C" int akh_gated_delta_step_f32(
    const float* state, const float* q, const float* k, const float* v,
    const float* decay, const float* beta, float* output, float* next,
    size_t rows, size_t value_width, size_t key_width) {
    if (!rows || !value_width || !key_width || key_width > 256 || rows > UINT32_MAX)
        return 1;
    // This opt-in inference kernel deliberately does not support graph streams.
    if (g_akh_stream != nullptr) return 1;
    hipLaunchKernelGGL(gated_delta_step_f32, dim3(rows), dim3(128), 0, 0,
        state, q, k, v, decay, beta, output, next, value_width, key_width);
    return hipGetLastError() == hipSuccess ? 0 : 1;
}

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

// Fused RMSNorm (Phase 45h): out[row,i] = x[row,i] / sqrt(mean_i(x^2) + eps) * w[i].
// One pass per row replaces the ~7 kernels of the rms_norm_slow decomposition
// (sqr, sum, /hidden, +eps, sqrt, broadcast_div, broadcast_mul) — the single
// biggest source of decode kernel launches (~2000/token). One block per row.
#define AKH_RMSNORM_BLOCK 256
__global__ void rms_norm_f32(const float* __restrict__ x, const float* __restrict__ w,
                             float* __restrict__ out, unsigned long n_rows,
                             unsigned long hidden, float eps) {
    unsigned long row = blockIdx.x;
    if (row >= n_rows) return;
    int tid = threadIdx.x;
    const float* xr = x + row * hidden;
    float* outr = out + row * hidden;
    float ss = 0.0f;
    for (unsigned long i = tid; i < hidden; i += AKH_RMSNORM_BLOCK) {
        float v = xr[i];
        ss += v * v;
    }
    __shared__ float sdata[AKH_RMSNORM_BLOCK];
    sdata[tid] = ss;
    __syncthreads();
    for (int s = AKH_RMSNORM_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    // Matches candle's rms_norm_slow: x / sqrt(mean(x^2) + eps), then * weight.
    float inv = 1.0f / sqrtf(sdata[0] / (float)hidden + eps);
    for (unsigned long i = tid; i < hidden; i += AKH_RMSNORM_BLOCK) {
        outr[i] = xr[i] * inv * w[i];
    }
}

// Flash-attention decode (Phase 46b): one warp per (batch, query-head). The single
// decode query streams the static KV prefix [0..len) with ONLINE softmax — running
// max + sum + output accumulator — so there is no materialised `[b,nh,1,len]` scores
// tensor and no separate softmax (collapses the ~9-kernel attention swarm → 1). GQA:
// kv head = head / (n_head/n_kv_head). No mask (decode passes none — matches the model).
//   q:   [B, n_head,   1,      head_dim]  contiguous
//   k,v: [B, n_kv_head, kv_ctx, head_dim] contiguous static buffers; only [0..len) read
//   out: [B, n_head,   1,      head_dim]
// head_dim must be a multiple of the warp size (32 on gfx1100); dims-per-lane ≤ MAXDPL.
#define AKH_FA_WARP 32
#define AKH_FA_MAXDPL 16
// Flash-DECODING (split-K): one warp per (batch, head, split) streams its KV chunk
// [j0,j1) with online softmax and writes the RAW (un-normalised) accumulator + running
// max `m` + sum `l` to scratch. A second kernel combines the splits. This adds the
// parallelism the naive one-warp-per-head version lacked (only B·n_head warps → the GPU
// idled at long context); splitting the KV gives B·n_head·n_split warps.
__global__ void flash_attn_partial_kernel(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    float* __restrict__ partial_acc, float* __restrict__ partial_ml,
    int B, int n_head, int n_kv_head, int head_dim,
    unsigned long kv_ctx, unsigned long len, float scale, int n_split) {
    int lane = threadIdx.x & (AKH_FA_WARP - 1);
    int gwarp = blockIdx.x * (blockDim.x / AKH_FA_WARP) + threadIdx.x / AKH_FA_WARP;
    if (gwarp >= B * n_head * n_split) return;
    int split = gwarp % n_split;
    int bh = gwarp / n_split;
    int b = bh / n_head, head = bh % n_head;
    int kvh = head / (n_head / n_kv_head);
    int dpl = head_dim / AKH_FA_WARP;
    unsigned long per = (len + n_split - 1) / n_split;
    unsigned long j0 = (unsigned long)split * per;
    unsigned long j1 = j0 + per;
    if (j1 > len) j1 = len;

    const float* qrow = q + (unsigned long)(b * n_head + head) * head_dim;
    const float* kbase = k + (unsigned long)(b * n_kv_head + kvh) * kv_ctx * head_dim;
    const float* vbase = v + (unsigned long)(b * n_kv_head + kvh) * kv_ctx * head_dim;
    float ql[AKH_FA_MAXDPL], acc[AKH_FA_MAXDPL];
    for (int i = 0; i < dpl; i++) { ql[i] = qrow[lane + AKH_FA_WARP * i]; acc[i] = 0.0f; }
    float m = -INFINITY, l = 0.0f;
    for (unsigned long j = j0; j < j1; j++) {
        const float* kj = kbase + j * head_dim;
        float partial = 0.0f;
        for (int i = 0; i < dpl; i++) partial += ql[i] * kj[lane + AKH_FA_WARP * i];
        for (int off = AKH_FA_WARP / 2; off > 0; off >>= 1) partial += __shfl_xor(partial, off);
        float score = partial * scale;
        float m_new = fmaxf(m, score);
        float corr = expf(m - m_new), p = expf(score - m_new);
        l = l * corr + p;
        const float* vj = vbase + j * head_dim;
        for (int i = 0; i < dpl; i++) acc[i] = acc[i] * corr + p * vj[lane + AKH_FA_WARP * i];
        m = m_new;
    }
    unsigned long po = (unsigned long)(bh * n_split + split) * head_dim;
    for (int i = 0; i < dpl; i++) partial_acc[po + lane + AKH_FA_WARP * i] = acc[i];
    if (lane == 0) {
        partial_ml[(bh * n_split + split) * 2 + 0] = m;
        partial_ml[(bh * n_split + split) * 2 + 1] = l;
    }
}

// Combine the per-split partials for each (batch, head): global max over splits, then
// `out = Σ_s exp(m_s-gm)·acc_s / Σ_s exp(m_s-gm)·l_s`. One warp per (batch, head).
__global__ void flash_attn_combine_kernel(
    const float* __restrict__ partial_acc, const float* __restrict__ partial_ml,
    float* __restrict__ out, int B, int n_head, int head_dim, int n_split) {
    int lane = threadIdx.x & (AKH_FA_WARP - 1);
    int bh = blockIdx.x * (blockDim.x / AKH_FA_WARP) + threadIdx.x / AKH_FA_WARP;
    if (bh >= B * n_head) return;
    int dpl = head_dim / AKH_FA_WARP;
    float gm = -INFINITY;
    for (int s = 0; s < n_split; s++) gm = fmaxf(gm, partial_ml[(bh * n_split + s) * 2]);
    float gl = 0.0f;
    float acc[AKH_FA_MAXDPL];
    for (int i = 0; i < dpl; i++) acc[i] = 0.0f;
    for (int s = 0; s < n_split; s++) {
        float ms = partial_ml[(bh * n_split + s) * 2];
        float ls = partial_ml[(bh * n_split + s) * 2 + 1];
        float w = expf(ms - gm);
        gl += ls * w;
        unsigned long po = (unsigned long)(bh * n_split + s) * head_dim;
        for (int i = 0; i < dpl; i++) acc[i] += w * partial_acc[po + lane + AKH_FA_WARP * i];
    }
    float inv = (gl > 0.0f) ? (1.0f / gl) : 0.0f;
    for (int i = 0; i < dpl; i++) out[(unsigned long)bh * head_dim + lane + AKH_FA_WARP * i] = acc[i] * inv;
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

// One block cooperatively reduces a contiguous vocabulary row. The generic
// reduction above stays the fallback for non-last dimensions and short rows.
// Preserve its strict comparison semantics: the first NaN sticks, later NaNs
// are ignored, and equal values (including signed zero/infinity) choose the
// lowest index regardless of the tree reduction order.
template<int THREADS>
__global__ void reduce_arg_rows_f32(const float* x, unsigned int* y,
                                    size_t width, int op) {
    const size_t row = blockIdx.x;
    const float* values = x + row * width;
    const unsigned int tid = threadIdx.x;
    if (isnan(values[0])) {
        if (tid == 0) y[row] = 0;
        return;
    }
    const unsigned int invalid = UINT32_MAX;
    float best = 0.0f;
    unsigned int best_idx = invalid;
    for (size_t i = tid; i < width; i += THREADS) {
        const float value = values[i];
        if (!isnan(value) && (best_idx == invalid ||
            (op == 1 ? value > best : value < best))) {
            best = value;
            best_idx = (unsigned int)i;
        }
    }
    __shared__ float candidates[THREADS];
    __shared__ unsigned int indices[THREADS];
    candidates[tid] = best;
    indices[tid] = best_idx;
    __syncthreads();
    for (unsigned int stride = THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const unsigned int other_idx = indices[tid + stride];
            const float other = candidates[tid + stride];
            const bool better = indices[tid] == invalid ||
                (op == 1 ? other > candidates[tid] : other < candidates[tid]) ||
                (other == candidates[tid] && other_idx < indices[tid]);
            if (other_idx != invalid && better) {
                candidates[tid] = other;
                indices[tid] = other_idx;
            }
        }
        __syncthreads();
    }
    if (tid == 0) y[row] = indices[0];
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

// 16-bit (f16) variant: identical strided gather, 2-byte elements (no arithmetic, so a
// raw u16 copy is dtype-correct). Powers `.contiguous()` on f16 (Phase 46 f16 KV cache).
__global__ void ucopy_f16(const unsigned short* src, unsigned short* dst, size_t n,
                          size_t rank, AkhI64x8 dims, AkhI64x8 strides) {
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

// Phase 46d-2: scatter the new token's K (or V) into a static KV cache at a DEVICE-side
// column index. The write offset is read from `d_pos` in global memory (so it can be
// updated between HIP-graph replays via a 4-byte H2D) instead of being baked into the
// launch as a host offset — the key to capturing a per-token decode forward ONCE per block
// and replaying it as the position advances. dtype-agnostic 2-byte (f16) raw copy.
//   src   : [n_kv, head_dim]            the new token's K/V for this layer (contiguous)
//   kbuf  : [n_kv, kv_ctx, head_dim]    the static cache (contiguous; batch = 1 on decode)
//   d_pos : device pointer to one uint32 column index in [0, kv_ctx)
__global__ void kv_scatter_f16_kernel(const unsigned short* __restrict__ src,
                                      unsigned short* __restrict__ kbuf,
                                      const unsigned int* __restrict__ d_pos,
                                      size_t n_kv, size_t head_dim, size_t kv_ctx) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = n_kv * head_dim;
    if (i >= total) return;
    size_t pos = (size_t)(*d_pos);
    size_t h = i / head_dim;
    size_t d = i - h * head_dim;
    kbuf[(h * kv_ctx + pos) * head_dim + d] = src[i];
}

extern "C" {

// Phase 46c: set the launch stream for this kernel library (null = default stream).
void akh_set_stream(void* s) { g_akh_stream = (hipStream_t)s; }

int akh_affine_f32(const float* x, float* y, size_t n, float mul, float add) {
    if (n == 0) return 0;
    int t = 256;
    size_t blocks = (n + t - 1) / t;
    // `stream` is null (the default stream) in normal execution; a real stream during
    // HIP-graph capture (Phase 46c) so the launch is recorded into the graph.
    hipLaunchKernelGGL(affine_f32, dim3(blocks), dim3(t), 0, g_akh_stream, x, y, n, mul, add);
    // No per-kernel sync (Phase 45h): kernels are ordered on the (null) stream; the
    // final D2H readback (to_cpu) synchronizes. Only launch errors checked here.
    return hipGetLastError() == hipSuccess ? 0 : 1;
}

// Fused RMSNorm; one block (AKH_RMSNORM_BLOCK threads) per row. No sync (45h).
int akh_rms_norm_f32(const float* x, const float* w, float* out,
                     size_t n_rows, size_t hidden, float eps) {
    if (n_rows == 0) return 0;
    hipLaunchKernelGGL(rms_norm_f32, dim3((unsigned)n_rows), dim3(AKH_RMSNORM_BLOCK), 0, g_akh_stream,
                       x, w, out, (unsigned long)n_rows, (unsigned long)hidden, eps);
    return hipGetLastError() == hipSuccess ? 0 : 1;
}

// Flash-decoding (split-K): partial kernel (B·n_head·n_split warps) then combine
// (B·n_head warps). `partial_acc`/`partial_ml` are caller-provided scratch sized
// B·n_head·n_split·head_dim and B·n_head·n_split·2 floats. head_dim must be a multiple
// of 32 and head_dim/32 <= AKH_FA_MAXDPL. No sync (ordered on the null stream).
#define AKH_FA_WARPS 8
int akh_flash_attn_decode_f32(const float* q, const float* k, const float* v, float* out,
                              float* partial_acc, float* partial_ml,
                              int B, int n_head, int n_kv_head, int head_dim,
                              size_t kv_ctx, size_t len, float scale, int n_split) {
    if (B * n_head == 0 || len == 0 || n_split <= 0) return 0;
    if (head_dim % AKH_FA_WARP != 0 || head_dim / AKH_FA_WARP > AKH_FA_MAXDPL) return 2;
    int pw = B * n_head * n_split;
    unsigned pgrid = (unsigned)((pw + AKH_FA_WARPS - 1) / AKH_FA_WARPS);
    hipLaunchKernelGGL(flash_attn_partial_kernel, dim3(pgrid), dim3(AKH_FA_WARP * AKH_FA_WARPS),
                       0, 0, q, k, v, partial_acc, partial_ml, B, n_head, n_kv_head, head_dim,
                       (unsigned long)kv_ctx, (unsigned long)len, scale, n_split);
    int cw = B * n_head;
    unsigned cgrid = (unsigned)((cw + AKH_FA_WARPS - 1) / AKH_FA_WARPS);
    hipLaunchKernelGGL(flash_attn_combine_kernel, dim3(cgrid), dim3(AKH_FA_WARP * AKH_FA_WARPS),
                       0, 0, partial_acc, partial_ml, out, B, n_head, head_dim, n_split);
    return hipGetLastError() == hipSuccess ? 0 : 1;
}

int akh_unary_f32(const float* x, float* y, size_t n, int op) {
    if (n == 0) return 0;
    int t = 256;
    size_t blocks = (n + t - 1) / t;
    hipLaunchKernelGGL(unary_f32, dim3(blocks), dim3(t), 0, g_akh_stream, x, y, n, op);
    // No per-kernel sync (Phase 45h): kernels are ordered on the null stream; the
    // final D2H readback (to_cpu) synchronizes. Only launch errors checked here.
    return hipGetLastError() == hipSuccess ? 0 : 1;
}

int akh_binary_f32(const float* a, const float* b, float* c, size_t n, int op) {
    if (n == 0) return 0;
    int t = 256;
    size_t blocks = (n + t - 1) / t;
    hipLaunchKernelGGL(binary_f32, dim3(blocks), dim3(t), 0, g_akh_stream, a, b, c, n, op);
    // No per-kernel sync (Phase 45h): kernels are ordered on the null stream; the
    // final D2H readback (to_cpu) synchronizes. Only launch errors checked here.
    return hipGetLastError() == hipSuccess ? 0 : 1;
}

// reduce: dims/strides/reduce_mask are DEVICE pointers (caller uploads + frees).
int akh_reduce_f32(const float* x, float* y, size_t out_numel, int num_dims,
                   struct AkhI64x8 dims, struct AkhI64x8 strides, struct AkhI64x8 reduce_mask,
                   size_t reduce_numel, int op) {
    if (out_numel == 0) return 0;
    int t = 256;
    size_t blocks = (out_numel + t - 1) / t;
    hipLaunchKernelGGL(reduce_f32, dim3(blocks), dim3(t), 0, g_akh_stream, x, y, out_numel,
                       num_dims, dims, strides, reduce_mask, reduce_numel, op);
    // No per-kernel sync (Phase 45h): kernels are ordered on the null stream; the
    // final D2H readback (to_cpu) synchronizes. Only launch errors checked here.
    return hipGetLastError() == hipSuccess ? 0 : 1;
}

// argmin/argmax reduce -> u32 index. Same metadata as akh_reduce_f32; op 0 argmin, 1 argmax.
int akh_reduce_arg_f32(const float* x, unsigned int* y, size_t out_numel, int num_dims,
                       struct AkhI64x8 dims, struct AkhI64x8 strides, struct AkhI64x8 reduce_mask,
                       size_t reduce_numel, int op) {
    if (out_numel == 0) return 0;
    bool contiguous_rows = num_dims > 0 && num_dims <= AKH_MAXR && reduce_numel >= 256;
    size_t expected_stride = 1;
    for (int d = num_dims - 1; contiguous_rows && d >= 0; --d) {
        contiguous_rows = strides.v[d] == (long)expected_stride &&
            reduce_mask.v[d] == (d == num_dims - 1 ? 1 : 0);
        expected_stride *= (size_t)dims.v[d];
    }
    if (contiguous_rows && (size_t)dims.v[num_dims - 1] == reduce_numel) {
        hipLaunchKernelGGL((reduce_arg_rows_f32<256>), dim3(out_numel), dim3(256), 0,
                          g_akh_stream, x, y, reduce_numel, op);
        return hipGetLastError() == hipSuccess ? 0 : 1;
    }
    int t = 256;
    size_t blocks = (out_numel + t - 1) / t;
    hipLaunchKernelGGL(reduce_arg_f32, dim3(blocks), dim3(t), 0, g_akh_stream, x, y, out_numel,
                       num_dims, dims, strides, reduce_mask, reduce_numel, op);
    // No per-kernel sync (Phase 45h): kernels are ordered on the null stream; the
    // final D2H readback (to_cpu) synchronizes. Only launch errors checked here.
    return hipGetLastError() == hipSuccess ? 0 : 1;
}

int akh_index_select_u32_f32(
    const float* src, const unsigned int* ids, float* out,
    size_t left, size_t dim_size, size_t n_idx, size_t right) {
    size_t total = left * n_idx * right;
    if (total == 0) return 0;
    int t = 256;
    size_t blocks = (total + t - 1) / t;
    hipLaunchKernelGGL(index_select_u32_f32, dim3(blocks), dim3(t), 0, g_akh_stream,
                       src, ids, out, left, dim_size, n_idx, right);
    // No per-kernel sync (Phase 45h): kernels are ordered on the null stream; the
    // final D2H readback (to_cpu) synchronizes. Only launch errors checked here.
    return hipGetLastError() == hipSuccess ? 0 : 1;
}
int akh_index_select_i64_f32(
    const float* src, const long long* ids, float* out,
    size_t left, size_t dim_size, size_t n_idx, size_t right) {
    size_t total = left * n_idx * right;
    if (total == 0) return 0;
    int t = 256;
    size_t blocks = (total + t - 1) / t;
    hipLaunchKernelGGL(index_select_i64_f32, dim3(blocks), dim3(t), 0, g_akh_stream,
                       src, ids, out, left, dim_size, n_idx, right);
    // No per-kernel sync (Phase 45h): kernels are ordered on the null stream; the
    // final D2H readback (to_cpu) synchronizes. Only launch errors checked here.
    return hipGetLastError() == hipSuccess ? 0 : 1;
}

int akh_where_cond_u8_f32(const unsigned char* cond, const float* t, const float* f,
                          float* out, size_t n) {
    if (n == 0) return 0;
    int tpb = 256;
    size_t blocks = (n + tpb - 1) / tpb;
    hipLaunchKernelGGL(where_cond_u8_f32, dim3(blocks), dim3(tpb), 0, g_akh_stream, cond, t, f, out, n);
    // No per-kernel sync (Phase 45h): kernels are ordered on the null stream; the
    // final D2H readback (to_cpu) synchronizes. Only launch errors checked here.
    return hipGetLastError() == hipSuccess ? 0 : 1;
}
int akh_where_cond_u32_f32(const unsigned int* cond, const float* t, const float* f,
                           float* out, size_t n) {
    if (n == 0) return 0;
    int tpb = 256;
    size_t blocks = (n + tpb - 1) / tpb;
    hipLaunchKernelGGL(where_cond_u32_f32, dim3(blocks), dim3(tpb), 0, g_akh_stream, cond, t, f, out, n);
    // No per-kernel sync (Phase 45h): kernels are ordered on the null stream; the
    // final D2H readback (to_cpu) synchronizes. Only launch errors checked here.
    return hipGetLastError() == hipSuccess ? 0 : 1;
}

// strided broadcasting binary; dims/a_str/b_str are device long arrays of length
// rank, a/b already advanced to start offsets, out contiguous f32. op 0..3.
int akh_binary_strided_f32(const float* a, const float* b, float* c, size_t n,
                           size_t rank, struct AkhI64x8 dims, struct AkhI64x8 a_str,
                           struct AkhI64x8 b_str, int op) {
    if (n == 0) return 0;
    int t = 256;
    size_t blocks = (n + t - 1) / t;
    hipLaunchKernelGGL(binary_strided_f32, dim3(blocks), dim3(t), 0, g_akh_stream,
                       a, b, c, n, rank, dims, a_str, b_str, op);
    // No per-kernel sync (Phase 45h): kernels are ordered on the null stream; the
    // final D2H readback (to_cpu) synchronizes. Only launch errors checked here.
    return hipGetLastError() == hipSuccess ? 0 : 1;
}

// strided unary; dims/x_str passed by value, x advanced to its start offset, out
// contiguous f32. Same op codes as akh_unary_f32.
int akh_unary_strided_f32(const float* x, float* y, size_t n, size_t rank,
                          struct AkhI64x8 dims, struct AkhI64x8 x_str, int op) {
    if (n == 0) return 0;
    int t = 256;
    size_t blocks = (n + t - 1) / t;
    hipLaunchKernelGGL(unary_strided_f32, dim3(blocks), dim3(t), 0, g_akh_stream,
                       x, y, n, rank, dims, x_str, op);
    // No per-kernel sync (Phase 45h): kernels are ordered on the null stream; the
    // final D2H readback (to_cpu) synchronizes. Only launch errors checked here.
    return hipGetLastError() == hipSuccess ? 0 : 1;
}

// strided elementwise compare -> u8; dims/a_str/b_str are device long arrays of
// length rank, a/b already advanced to their start offsets, out contiguous u8.
int akh_cmp_f32(const float* a, const float* b, unsigned char* out, size_t n,
                size_t rank, const long* dims, const long* a_str, const long* b_str, int op) {
    if (n == 0) return 0;
    int t = 256;
    size_t blocks = (n + t - 1) / t;
    hipLaunchKernelGGL(cmp_f32, dim3(blocks), dim3(t), 0, g_akh_stream, a, b, out, n, rank, dims, a_str, b_str, op);
    // No per-kernel sync (Phase 45h): kernels are ordered on the null stream; the
    // final D2H readback (to_cpu) synchronizes. Only launch errors checked here.
    return hipGetLastError() == hipSuccess ? 0 : 1;
}

int akh_cmp_u32(const unsigned int* a, const unsigned int* b, unsigned char* out, size_t n,
                size_t rank, const long* dims, const long* a_str, const long* b_str, int op) {
    if (n == 0) return 0;
    int t = 256;
    size_t blocks = (n + t - 1) / t;
    hipLaunchKernelGGL(cmp_u32, dim3(blocks), dim3(t), 0, g_akh_stream, a, b, out, n, rank, dims, a_str, b_str, op);
    // No per-kernel sync (Phase 45h): kernels are ordered on the null stream; the
    // final D2H readback (to_cpu) synchronizes. Only launch errors checked here.
    return hipGetLastError() == hipSuccess ? 0 : 1;
}

// strided where_cond (cond/t/f own strides; dims/c_str/t_str/f_str device long
// arrays of length rank; cond/t/f advanced to start offsets; out contiguous f32).
int akh_where_cond_strided_u8_f32(const unsigned char* cond, const float* t, const float* f,
    float* out, size_t n, size_t rank, const long* dims, const long* c_str,
    const long* t_str, const long* f_str) {
    if (n == 0) return 0;
    int tpb = 256;
    size_t blocks = (n + tpb - 1) / tpb;
    hipLaunchKernelGGL(where_cond_strided_u8_f32, dim3(blocks), dim3(tpb), 0, g_akh_stream,
                       cond, t, f, out, n, rank, dims, c_str, t_str, f_str);
    // No per-kernel sync (Phase 45h): kernels are ordered on the null stream; the
    // final D2H readback (to_cpu) synchronizes. Only launch errors checked here.
    return hipGetLastError() == hipSuccess ? 0 : 1;
}
int akh_where_cond_strided_u32_f32(const unsigned int* cond, const float* t, const float* f,
    float* out, size_t n, size_t rank, const long* dims, const long* c_str,
    const long* t_str, const long* f_str) {
    if (n == 0) return 0;
    int tpb = 256;
    size_t blocks = (n + tpb - 1) / tpb;
    hipLaunchKernelGGL(where_cond_strided_u32_f32, dim3(blocks), dim3(tpb), 0, g_akh_stream,
                       cond, t, f, out, n, rank, dims, c_str, t_str, f_str);
    // No per-kernel sync (Phase 45h): kernels are ordered on the null stream; the
    // final D2H readback (to_cpu) synchronizes. Only launch errors checked here.
    return hipGetLastError() == hipSuccess ? 0 : 1;
}

// strided gather-copy: src/dst already advanced to start offsets; dims/strides by value.
int akh_ucopy_f32(const float* src, float* dst, size_t n, size_t rank,
                  struct AkhI64x8 dims, struct AkhI64x8 strides) {
    if (n == 0) return 0;
    int t = 256;
    size_t blocks = (n + t - 1) / t;
    hipLaunchKernelGGL(ucopy_f32, dim3(blocks), dim3(t), 0, g_akh_stream, src, dst, n, rank, dims, strides);
    // No per-kernel sync (Phase 45h): kernels are ordered on the null stream; the
    // final D2H readback (to_cpu) synchronizes. Only launch errors checked here.
    return hipGetLastError() == hipSuccess ? 0 : 1;
}

int akh_ucopy_f16(const unsigned short* src, unsigned short* dst, size_t n, size_t rank,
                  struct AkhI64x8 dims, struct AkhI64x8 strides) {
    if (n == 0) return 0;
    int t = 256;
    size_t blocks = (n + t - 1) / t;
    hipLaunchKernelGGL(ucopy_f16, dim3(blocks), dim3(t), 0, g_akh_stream, src, dst, n, rank, dims, strides);
    return hipGetLastError() == hipSuccess ? 0 : 1;
}

// Phase 46d-2: device-indexed KV scatter (see kv_scatter_f16_kernel). One thread per
// (head, head_dim) element; the column index is read from `d_pos` on the device.
int akh_kv_scatter_f16(const unsigned short* src, unsigned short* kbuf,
                       const unsigned int* d_pos, size_t n_kv, size_t head_dim, size_t kv_ctx) {
    size_t total = n_kv * head_dim;
    if (total == 0) return 0;
    int t = 256;
    size_t blocks = (total + t - 1) / t;
    hipLaunchKernelGGL(kv_scatter_f16_kernel, dim3(blocks), dim3(t), 0, g_akh_stream,
                       src, kbuf, d_pos, n_kv, head_dim, kv_ctx);
    return hipGetLastError() == hipSuccess ? 0 : 1;
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
        /* Phase 46c: launch on g_akh_stream, NOT stream 0. Hardcoding the null   \
           stream made the f16<->f32 cast (the KV cache's per-layer read cast)    \
           run UN-captured during HIP-graph capture — missing from the replay, so \
           the attention read stale K/V (~0.05/layer, compounding to a wrong      \
           full-forward replay). Every other launcher already uses g_akh_stream. */\
        hipLaunchKernelGGL((cast_kernel<SRC, DST>), dim3(blocks), dim3(t), 0,  \
                           g_akh_stream, (xptr), (yptr), (n));                 \
        return hipGetLastError() == hipSuccess ? 0 : 1;                        \
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
    hipLaunchKernelGGL(cast_f16_bf16_kernel, dim3(blocks), dim3(t), 0, g_akh_stream, x, y, n);
    // No per-kernel sync (Phase 45h): kernels are ordered on the null stream; the
    // final D2H readback (to_cpu) synchronizes. Only launch errors checked here.
    return hipGetLastError() == hipSuccess ? 0 : 1;
}
int akh_cast_bf16_f16(const hip_bfloat16* x, __half* y, size_t n) {
    if (n == 0) return 0;
    int t = 256;
    size_t blocks = (n + t - 1) / t;
    hipLaunchKernelGGL(cast_bf16_f16_kernel, dim3(blocks), dim3(t), 0, g_akh_stream, x, y, n);
    // No per-kernel sync (Phase 45h): kernels are ordered on the null stream; the
    // final D2H readback (to_cpu) synchronizes. Only launch errors checked here.
    return hipGetLastError() == hipSuccess ? 0 : 1;
}
int akh_cast_f32_u32(const float* x, uint32_t* y, size_t n) { AKH_CAST_LAUNCH(float, uint32_t, x, y, n); }
int akh_cast_u32_f32(const uint32_t* x, float* y, size_t n) { AKH_CAST_LAUNCH(uint32_t, float, x, y, n); }
int akh_cast_f32_i64(const float* x, int64_t* y, size_t n) { AKH_CAST_LAUNCH(float, int64_t, x, y, n); }
int akh_cast_i64_f32(const int64_t* x, float* y, size_t n) { AKH_CAST_LAUNCH(int64_t, float, x, y, n); }

} // extern "C"
