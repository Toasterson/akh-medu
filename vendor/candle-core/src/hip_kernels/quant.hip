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

extern "C" {

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
