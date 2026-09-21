# The vision projector runs on CPU, HIP and Vulkan

Source `8027f15b` plus this change. Local RX 7900 XTX, exact pinned Ptah
projector, identical deterministic input on every device. No shared service
touched, no production daemon involved.

## Result

64x64 input, a 4x4 patch grid merging to 2x2, so four rows at the language
model's 5120 width. Every device produced finite, non-degenerate embeddings.

| Device | Load, s | Forward, s | Mean | Std dev | L2 norm |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU | 1.39 | 1.976 | -0.0088424 | 1.105577 | 158.2224 |
| HIP | 3.23 | 1.779 | -0.0088424 | 1.105577 | 158.2223 |
| Vulkan | 0.75 | 0.098 | -0.0088424 | 1.105575 | 158.2220 |

Against CPU as reference, the maximum absolute difference across the sampled
output values was **8.1e-06 on HIP** and **6.6e-06 on Vulkan**, with L2-norm
relative differences of 1.4e-07 and 2.1e-06. That is f32 accumulation-order
noise on values of order one, which is what cross-device agreement should look
like for a tower that is matmul-heavy and has no reduction-order guarantees.

The Vulkan forward time is an order of magnitude below HIP's here. Do not read
that as a general throughput claim: this is one 16-patch image, the timing
includes per-op dispatch overhead that a larger input would amortize, and no
warmup or repetition was done. It is recorded because it is the measurement
that was taken, not because it generalizes.

## Two portability defects fixed to get here

Neither device could run the tower as first written, and both failures were in
the vendored backends rather than in the projector.

**`conv2d` is unimplemented on HIP and Vulkan.** HIP reports it as pending
(Phase 45d/45e); Vulkan simply lacks it. Rather than write two convolution
kernels, the patch embedding is now an unfold plus a matmul: the kernel and
stride are both the patch size, so the operations are mathematically identical,
and matmul is the one primitive both backends implement well. The image is
reshaped to `(patches, channels*patch*patch)` and multiplied by the kernel
reshaped to `(width, channels*patch*patch)`. This also removes any dependency on
the convolution path the ledger records as opt-in on Vulkan.

**`softmax_last_dim` and unary `gelu` have no device implementations.** Both are
now dispatched per device, following what `quantized_qwen35` already does for
the language model's attention: the composed `candle_nn::ops::softmax` on HIP
and Vulkan, and for Vulkan a GELU built from primitives using
`tanh(z) = 2*sigmoid(2z) - 1`, so the tanh-approximate GELU becomes
`x * sigmoid(2u)` and needs only exp, reciprocal and arithmetic.

## Still not established

Cross-device agreement is not correctness. All three devices could agree on the
same wrong answer, because three modelling choices remain unverified against
llama.cpp: summing both temporal patch-embedding weights for a still frame,
align-corners bilinear position interpolation, and that `use_gelu` means the
tanh approximation. Confirming those needs a reference output for a known image,
which this does not have.

Also untested: any input large enough to exercise position interpolation away
from the learned 48x48 grid — 64x64 gives a 4x4 grid, so interpolation ran but
at an extreme downscale — and any integration with the language model, which
still lacks the mrope sections image tokens require.

Artifacts: [device runs](../../../experiments/native-vision/projector-20260919/devices.jsonl).
