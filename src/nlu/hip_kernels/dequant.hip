// HIP dequantization kernel for the akh-medu AMD GPU backend (ADR 053).
//
// Stores linear-layer weights as 4-bit group-quantized data resident on the GPU (compact
// enough for 12B/27B in 24 GB) and dequantizes chunks on demand into an f32 scratch buffer
// that rocBLAS then consumes. This is the "dequantize → BLAS" path: the kernel here is only
// the elementwise dequant; the GEMM is rocBLAS sgemm.
//
// Quant format (akh-medu's own, deliberately simple & robust):
//   * symmetric 4-bit, group size G along the `in` dimension, one f32 scale per group.
//   * a weight row `o` of `in_dim` values → in_dim/2 packed bytes (2 nibbles/byte) and
//     in_dim/G f32 scales.
//   * nibble n in [0,15] encodes q = n - 8 in [-8,7]; value = q * scale[group].
//
// Built to a shared library by build.rs with amdclang++ --offload-arch=gfx1100 and dlopen'd
// at runtime (src/nlu/hip.rs).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Dequantize `rows` weight rows (each `in_dim` wide) of W[out,in] into `out`, written
// **transposed** as [in_dim, rows] (row-major, ld = rows). That makes `out` exactly the
// rocBLAS B-operand `Wᵀ[in, rows]` for y = x · Wᵀ, so no host-side transpose is needed.
__global__ void akh_dequant_q4_kernel(const unsigned char* __restrict__ q,
                                      const float* __restrict__ scales,
                                      float* __restrict__ out,
                                      int rows, int in_dim, int group) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)rows * in_dim;
    if (idx >= total) return;
    int row = (int)(idx / in_dim); // weight row (output channel, within chunk)
    int col = (int)(idx % in_dim); // input index
    int groups_per_row = in_dim / group;
    // packed nibble: byte holds elements (2b) low, (2b+1) high, in the weight's own layout.
    long elem = (long)row * in_dim + col;
    unsigned char byte = q[elem >> 1];
    int nib = (col & 1) ? (byte >> 4) : (byte & 0x0F);
    float scale = scales[(long)row * groups_per_row + (col / group)];
    // transposed write: out[in_idx, out_chan] with ld = rows
    out[(long)col * rows + row] = (float)(nib - 8) * scale;
}

// ── Native GGUF Q4_K dequant ────────────────────────────────────────────────────────────────
//
// Dequantize weights stored in llama.cpp/ggml **Q4_K** superblock format directly, exactly as
// candle's CPU `BlockQ4K::to_float` does — so the GPU uses the GGUF weights *as shipped* with no
// dequant→requant precision loss (vs akh-medu's own symmetric int4). One Q4_K block holds 256
// weights in 144 bytes: f16 d, f16 dmin, 12 packed 6-bit (scale,min) bytes, then 128 nibble bytes.
//
// Layout of one block (bytes): [0:2] d (f16 LE) [2:4] dmin (f16 LE) [4:16] scales[12] [16:144] qs[128].
#define AKH_Q4K_BLOCK_BYTES 144
#define AKH_Q4K_QK 256

// Mirror of candle's `get_scale_min_k4` (quantized/utils.rs): unpack the 6-bit scale & min for
// sub-block j in [0,8) from the 12 packed bytes.
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

// Dequantize `rows` Q4_K rows (each `in_dim` weights, `in_dim % 256 == 0`) into `out`, written
// **transposed** as [in_dim, rows] (ld = rows) — the rocBLAS B-operand Wᵀ[in, rows], same as the
// symmetric kernel above. `blocks` points at this chunk's first row's blocks; row stride is
// `in_dim/256` blocks.
__global__ void akh_dequant_q4k_kernel(const unsigned char* __restrict__ blocks,
                                       float* __restrict__ out, int rows, int in_dim) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)rows * in_dim;
    if (idx >= total) return;
    int row = (int)(idx / in_dim); // output channel within chunk
    int col = (int)(idx % in_dim); // input index
    int blocks_per_row = in_dim / AKH_Q4K_QK;
    int sb = col / AKH_Q4K_QK;     // super-block within the row
    int p = col % AKH_Q4K_QK;      // position within the super-block

    const unsigned char* b =
        blocks + ((long)row * blocks_per_row + sb) * AKH_Q4K_BLOCK_BYTES;
    // f16 d / dmin (little-endian, native on the GPU).
    float d = __half2float(*reinterpret_cast<const __half*>(b + 0));
    float dmin = __half2float(*reinterpret_cast<const __half*>(b + 2));
    const unsigned char* scales = b + 4;
    const unsigned char* qs = b + 16;

    int iter = p / 64;          // 0..3  (candle's j = 64*iter)
    int within = p % 64;        // 0..63
    int qoff = iter * 32 + (within & 31); // qs byte index (q = &qs[iter*32]; l = within%32)
    int is = 2 * iter + (within < 32 ? 0 : 1);
    unsigned char sc, mn;
    akh_q4k_scale_min(is, scales, &sc, &mn);
    int nib = (within < 32) ? (qs[qoff] & 0xF) : (qs[qoff] >> 4);
    float val = d * (float)sc * (float)nib - dmin * (float)mn;

    out[(long)col * rows + row] = val;
}

// SwiGLU activation, elementwise on the GPU: out[i] = silu(g[i]) * u[i], where
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x)). This is the only non-GEMM step of the MLP;
// running it here lets the hidden state stay resident on the GPU through gate → act → down
// instead of round-tripping to the host for the multiply (ADR 053, on-device activations).
__global__ void akh_silu_mul_kernel(const float* __restrict__ g,
                                    const float* __restrict__ u,
                                    float* __restrict__ out, long n) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float x = g[idx];
    float s = x / (1.0f + expf(-x));
    out[idx] = s * u[idx];
}

extern "C" {

// Launch the dequant kernel on the default stream and synchronize. Returns a hipError_t code
// (0 == success). Pointers are device pointers; `out` must hold rows*in_dim floats.
int akh_dequant_q4(const unsigned char* q, const float* scales, float* out,
                   int rows, int in_dim, int group) {
    long total = (long)rows * in_dim;
    int threads = 256;
    long blocks = (total + threads - 1) / threads;
    hipLaunchKernelGGL(akh_dequant_q4_kernel, dim3((unsigned)blocks), dim3(threads), 0, 0,
                       q, scales, out, rows, in_dim, group);
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) return (int)e;
    return (int)hipDeviceSynchronize();
}

// Launch the native Q4_K dequant kernel and synchronize. `blocks` is the raw GGUF Q4_K bytes
// for `rows` rows of `in_dim` weights (in_dim % 256 == 0); `out` holds rows*in_dim floats.
int akh_dequant_q4k(const unsigned char* blocks, float* out, int rows, int in_dim) {
    long total = (long)rows * in_dim;
    int threads = 256;
    long nblocks = (total + threads - 1) / threads;
    hipLaunchKernelGGL(akh_dequant_q4k_kernel, dim3((unsigned)nblocks), dim3(threads), 0, 0,
                       blocks, out, rows, in_dim);
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) return (int)e;
    return (int)hipDeviceSynchronize();
}

// out[i] = silu(g[i]) * u[i] for i in [0, n). All pointers are device pointers of length n.
// Launches on the default stream (serialized after the preceding gate/up sgemm) and syncs.
int akh_silu_mul(const float* g, const float* u, float* out, long n) {
    int threads = 256;
    long blocks = (n + threads - 1) / threads;
    hipLaunchKernelGGL(akh_silu_mul_kernel, dim3((unsigned)blocks), dim3(threads), 0, 0,
                       g, u, out, n);
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) return (int)e;
    return (int)hipDeviceSynchronize();
}

} // extern "C"
