// HIP Q4_K dequantization kernel for the vendored candle-core HIP backend
// (Phase 45e / ADR 073).
//
// Dequantizes GGUF/ggml **Q4_K** superblocks directly into f32, mirroring candle's CPU
// `BlockQ4K::to_float` (`quantized/k_quants.rs`) BIT-FOR-BIT, so the GPU consumes the GGUF
// weights *as shipped* with no dequant→requant precision loss. The result is written
// **row-major** `out[row*in_dim + col]` — identical element ordering to the CPU
// `dequantize`, which lets `QCudaStorage::dequantize` compare HIP↔CPU directly and lets the
// quantized `fwd` reuse the rocBLAS f32 matmul path unchanged.
//
// One Q4_K block holds 256 weights in 144 bytes (candle `#[repr(C)] BlockQ4K`):
//   [0:2]   d     (f16 LE)
//   [2:4]   dmin  (f16 LE)
//   [4:16]  scales[12]   (12 packed 6-bit (scale,min) pairs for 8 sub-blocks)
//   [16:144] qs[128]     (256 packed 4-bit weights, 2 nibbles/byte)
//
// Built to `libcandle_hip_quant.so` by build.rs (amdclang++ --offload-arch=gfx1100) and
// dlopen'd at runtime by `quantized/hip.rs`.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define AKH_Q4K_BLOCK_BYTES 144
#define AKH_Q4K_QK 256

// Mirror of candle's `get_scale_min_k4` (quantized/utils.rs): unpack the 6-bit scale & min
// for sub-block j in [0,8) from the 12 packed scale bytes.
__device__ __forceinline__ void akh_q4k_scale_min(int j, const unsigned char* s,
                                                  unsigned char* d, unsigned char* m) {
    if (j < 4) {
        *d = s[j] & 63;
        *m = s[j + 4] & 63;
    } else {
        *d = (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4);
        *m = (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
    }
}

// Row-major Q4_K dequant: `rows` rows, each `in_dim` weights (in_dim % 256 == 0). Block stride
// per row is `in_dim/256` superblocks. Writes `out[row*in_dim + col]` to match candle CPU
// `BlockQ4K::to_float` exactly.
//
// candle CPU per 256-element block, for j in {0,64,128,192} (iter = j/64):
//   q = qs[iter*32 .. iter*32+32]; (sc,m)=k4(2*iter) -> d1,m1 ; (sc,m)=k4(2*iter+1) -> d2,m2
//   first 32 outputs (positions j..j+32):    y = d1*(q & 0xF) - m1
//   next 32 outputs  (positions j+32..j+64): y = d2*(q >> 4) - m2
// so for position p in the block: iter=p/64, within=p%64; low nibble & sub-block (2*iter)
// when within<32, high nibble & sub-block (2*iter+1) when within>=32.
__global__ void akh_dequant_q4k_kernel(const unsigned char* __restrict__ blocks,
                                       float* __restrict__ out,
                                       unsigned long rows, unsigned long in_dim) {
    unsigned long idx = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long total = rows * in_dim;
    if (idx >= total) return;
    unsigned long row = idx / in_dim; // output channel
    unsigned long col = idx % in_dim; // input index
    unsigned long blocks_per_row = in_dim / AKH_Q4K_QK;
    unsigned long sb = col / AKH_Q4K_QK; // super-block within the row
    int p = (int)(col % AKH_Q4K_QK);     // position within the super-block

    const unsigned char* b =
        blocks + (row * blocks_per_row + sb) * AKH_Q4K_BLOCK_BYTES;
    // f16 d / dmin (little-endian, native on the GPU).
    float d = __half2float(*reinterpret_cast<const __half*>(b + 0));
    float dmin = __half2float(*reinterpret_cast<const __half*>(b + 2));
    const unsigned char* scales = b + 4;
    const unsigned char* qs = b + 16;

    int iter = p / 64;                    // 0..3  (candle's j = 64*iter)
    int within = p % 64;                  // 0..63
    int qoff = iter * 32 + (within & 31); // qs byte index (q = &qs[iter*32]; l = within%32)
    int is = 2 * iter + (within < 32 ? 0 : 1);
    unsigned char sc, mn;
    akh_q4k_scale_min(is, scales, &sc, &mn);
    int nib = (within < 32) ? (qs[qoff] & 0xF) : (qs[qoff] >> 4);
    float val = d * (float)sc * (float)nib - dmin * (float)mn;

    out[row * in_dim + col] = val;
}

// ---- Q6_K (Phase 45f) ------------------------------------------------------
// One Q6_K block holds 256 weights in 210 bytes (candle `#[repr(C)] BlockQ6K`):
//   [0:128]   ql[128]     (low 4 bits, 2 weights/byte)
//   [128:192] qh[64]      (high 2 bits, 4 weights/byte)
//   [192:208] scales[16]  (i8, signed)
//   [208:210] d           (f16 LE)
// Mirrors candle CPU `BlockQ6K::to_float` bit-for-bit. Per 256-block, candle walks
// idx in {0,1} (n=128*idx) and l in 0..32 emitting 4 quarters at l, l+32, l+64, l+96:
//   q1 = (ql[l]&0xF    | (qh[l]&3)<<4) - 32   -> sc[is]
//   q2 = (ql[l+32]&0xF | ((qh[l]>>2)&3)<<4)-32-> sc[is+2]
//   q3 = (ql[l]>>4     | ((qh[l]>>4)&3)<<4)-32-> sc[is+4]
//   q4 = (ql[l+32]>>4  | ((qh[l]>>6)&3)<<4)-32-> sc[is+6]
// where is=l/16, ql/qh/sc bases offset by 64*idx/32*idx/8*idx. Inverting for an
// arbitrary position p in [0,256): idx=p/128, within=p%128, quarter=within/32,
// l=within%32, is=l/16, sc index = is + 2*quarter.
#define AKH_Q6K_BLOCK_BYTES 210
#define AKH_Q6K_QK 256

__global__ void akh_dequant_q6k_kernel(const unsigned char* __restrict__ blocks,
                                       float* __restrict__ out,
                                       unsigned long rows, unsigned long in_dim) {
    unsigned long gidx = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long total = rows * in_dim;
    if (gidx >= total) return;
    unsigned long row = gidx / in_dim;
    unsigned long col = gidx % in_dim;
    unsigned long blocks_per_row = in_dim / AKH_Q6K_QK;
    unsigned long sb = col / AKH_Q6K_QK;
    int p = (int)(col % AKH_Q6K_QK);

    const unsigned char* b =
        blocks + (row * blocks_per_row + sb) * AKH_Q6K_BLOCK_BYTES;
    const unsigned char* ql = b + 0;
    const unsigned char* qh = b + 128;
    const signed char* sc = reinterpret_cast<const signed char*>(b + 192);
    float d = __half2float(*reinterpret_cast<const __half*>(b + 208));

    int idx = p / 128;          // 0 or 1
    int within = p % 128;       // 0..127
    int quarter = within / 32;  // 0..3
    int l = within % 32;        // 0..31
    int is = l / 16;            // 0 or 1

    const unsigned char* qlb = ql + 64 * idx;
    const unsigned char* qhb = qh + 32 * idx;
    const signed char* scb = sc + 8 * idx;
    unsigned char qhbyte = qhb[l];
    int qval;
    switch (quarter) {
        case 0: qval = (qlb[l] & 0xF)      | ((qhbyte & 3) << 4); break;
        case 1: qval = (qlb[l + 32] & 0xF) | (((qhbyte >> 2) & 3) << 4); break;
        case 2: qval = (qlb[l] >> 4)       | (((qhbyte >> 4) & 3) << 4); break;
        default: qval = (qlb[l + 32] >> 4) | (((qhbyte >> 6) & 3) << 4); break;
    }
    int q = qval - 32;
    int sc_index = is + 2 * quarter;
    out[row * in_dim + col] = d * (float)scb[sc_index] * (float)q;
}

// ---- fused quantized GEMV (Phase 45g) --------------------------------------
// Decode is a matrix-VECTOR product (m=1): out[row] = Σ_k dequant(W[row,k]) * x[k].
// Dequantizing inline (never materializing the f32 weight) streams only the
// quantized weights (7.3 GB) instead of writing+reading a 4× f32 temp (~48 GB)
// per token. One thread per output row (simple, correct; warp-cooperative is a
// later optimization). `W` is row-major Q4_K/Q6_K, `x` the f32 activation [in_dim].
__global__ void akh_gemv_q4k_f32_kernel(const unsigned char* __restrict__ W,
                                 const float* __restrict__ x, float* __restrict__ out,
                                 unsigned long n_rows, unsigned long in_dim) {
    unsigned long row = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    unsigned long bpr = in_dim / AKH_Q4K_QK;
    const unsigned char* base = W + row * bpr * AKH_Q4K_BLOCK_BYTES;
    float acc = 0.0f;
    for (unsigned long sb = 0; sb < bpr; ++sb) {
        const unsigned char* b = base + sb * AKH_Q4K_BLOCK_BYTES;
        float d = __half2float(*reinterpret_cast<const __half*>(b + 0));
        float dmin = __half2float(*reinterpret_cast<const __half*>(b + 2));
        const unsigned char* scales = b + 4;
        const unsigned char* qs = b + 16;
        const float* xb = x + sb * AKH_Q4K_QK;
        for (int p = 0; p < AKH_Q4K_QK; ++p) {
            int iter = p / 64;
            int within = p % 64;
            int qoff = iter * 32 + (within & 31);
            int is = 2 * iter + (within < 32 ? 0 : 1);
            unsigned char sc, mn;
            akh_q4k_scale_min(is, scales, &sc, &mn);
            int nib = (within < 32) ? (qs[qoff] & 0xF) : (qs[qoff] >> 4);
            float w = d * (float)sc * (float)nib - dmin * (float)mn;
            acc += w * xb[p];
        }
    }
    out[row] = acc;
}

__global__ void akh_gemv_q6k_f32_kernel(const unsigned char* __restrict__ W,
                                 const float* __restrict__ x, float* __restrict__ out,
                                 unsigned long n_rows, unsigned long in_dim) {
    unsigned long row = (unsigned long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    unsigned long bpr = in_dim / AKH_Q6K_QK;
    const unsigned char* base = W + row * bpr * AKH_Q6K_BLOCK_BYTES;
    float acc = 0.0f;
    for (unsigned long sb = 0; sb < bpr; ++sb) {
        const unsigned char* b = base + sb * AKH_Q6K_BLOCK_BYTES;
        const unsigned char* ql = b + 0;
        const unsigned char* qh = b + 128;
        const signed char* sc = reinterpret_cast<const signed char*>(b + 192);
        float d = __half2float(*reinterpret_cast<const __half*>(b + 208));
        const float* xb = x + sb * AKH_Q6K_QK;
        for (int p = 0; p < AKH_Q6K_QK; ++p) {
            int idx = p / 128;
            int within = p % 128;
            int quarter = within / 32;
            int l = within % 32;
            int isub = l / 16;
            const unsigned char* qlb = ql + 64 * idx;
            const unsigned char* qhb = qh + 32 * idx;
            const signed char* scb = sc + 8 * idx;
            unsigned char qhbyte = qhb[l];
            int qval;
            switch (quarter) {
                case 0: qval = (qlb[l] & 0xF)      | ((qhbyte & 3) << 4); break;
                case 1: qval = (qlb[l + 32] & 0xF) | (((qhbyte >> 2) & 3) << 4); break;
                case 2: qval = (qlb[l] >> 4)       | (((qhbyte >> 4) & 3) << 4); break;
                default: qval = (qlb[l + 32] >> 4) | (((qhbyte >> 6) & 3) << 4); break;
            }
            float w = d * (float)scb[isub + 2 * quarter] * (float)(qval - 32);
            acc += w * xb[p];
        }
    }
    out[row] = acc;
}

extern "C" {

// Fused Q4_K/Q6_K GEMV: out[n_rows] = W[n_rows,in_dim] · x[in_dim], no f32 temp.
int akh_gemv_q4k_f32(const unsigned char* W, const float* x, float* out,
                     size_t n_rows, size_t in_dim) {
    if (n_rows == 0) return 0;
    int threads = 256;
    unsigned long nblocks = ((unsigned long)n_rows + threads - 1) / threads;
    hipLaunchKernelGGL(akh_gemv_q4k_f32_kernel, dim3((unsigned)nblocks), dim3(threads), 0, 0,
                       W, x, out, (unsigned long)n_rows, (unsigned long)in_dim);
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) return (int)e;
    return (int)hipDeviceSynchronize();
}

int akh_gemv_q6k_f32(const unsigned char* W, const float* x, float* out,
                     size_t n_rows, size_t in_dim) {
    if (n_rows == 0) return 0;
    int threads = 256;
    unsigned long nblocks = ((unsigned long)n_rows + threads - 1) / threads;
    hipLaunchKernelGGL(akh_gemv_q6k_f32_kernel, dim3((unsigned)nblocks), dim3(threads), 0, 0,
                       W, x, out, (unsigned long)n_rows, (unsigned long)in_dim);
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) return (int)e;
    return (int)hipDeviceSynchronize();
}

// Launch the row-major Q6_K dequant kernel; `out` must hold rows*in_dim floats.
int akh_dequant_q6k(const unsigned char* blocks, float* out, size_t rows, size_t in_dim) {
    unsigned long total = (unsigned long)rows * (unsigned long)in_dim;
    int threads = 256;
    unsigned long nblocks = (total + threads - 1) / threads;
    hipLaunchKernelGGL(akh_dequant_q6k_kernel, dim3((unsigned)nblocks), dim3(threads), 0, 0,
                       blocks, out, (unsigned long)rows, (unsigned long)in_dim);
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) return (int)e;
    return (int)hipDeviceSynchronize();
}

// Launch the row-major Q4_K dequant kernel on the default stream and synchronize.
// `blocks` is the raw GGUF Q4_K bytes for `rows` rows of `in_dim` weights (in_dim % 256 == 0);
// `out` must hold rows*in_dim floats. Returns a hipError_t code (0 == success).
int akh_dequant_q4k(const unsigned char* blocks, float* out, size_t rows, size_t in_dim) {
    unsigned long total = (unsigned long)rows * (unsigned long)in_dim;
    int threads = 256;
    unsigned long nblocks = (total + threads - 1) / threads;
    hipLaunchKernelGGL(akh_dequant_q4k_kernel, dim3((unsigned)nblocks), dim3(threads), 0, 0,
                       blocks, out, (unsigned long)rows, (unsigned long)in_dim);
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) return (int)e;
    return (int)hipDeviceSynchronize();
}

} // extern "C"
