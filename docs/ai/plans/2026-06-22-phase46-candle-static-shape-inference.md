# Phase 46 — Candle static-shape inference + in-process training engine

> Date: 2026-06-22
> Status: In progress
> ADR: [074](../decisions/074-candle-static-shape-inference-and-training.md)
> Builds on: [Phase 45](2026-06-21-phase45-full-candle-hip-backend.md) (resident
>            `Device::Hip`, GEMV win → 11.5 tok/s, event profiler, HIP-graph spike)
> Goal: close the measured **4.3× gap to Ollama** (11.5 → ≥30 tok/s near-term, roofline
>       131) by making the decode forward **static-shape + allocation-free**, then
>       capturing it as a HIP graph — the same architecture the **in-process training
>       engine** (Phase 27) needs.
> Hardware: AMD RX 7900 XTX (`gfx1100`), ROCm 6.3.4. Model: `gemma3-12b.gguf` Q4_K.

## Why (one line)

The gap to Ollama is **dispatch + dynamic shapes, not compute** (GEMV already ≈ 46% of
roofline). A static-shape, allocation-free forward unlocks HIP-graph replay (kills the
per-token dispatch storm) **and** is the prerequisite for a differentiable in-process
backward/LoRA pass. See ADR 074.

## Discipline (carried from Phase 45)

- **Measure-or-revert.** Every step is gated on **unprofiled tok/s** (the ground truth)
  + bit-exact parity vs CPU. Profiler percentages are diagnostic only (whole-op windows
  over-count host-gap). Revert anything that does not move tok/s.
- **One reviewable commit per green gate.** `git -c commit.gpgsign=false`, push via
  `git push https-origin HEAD` (ssh remote key is broken on this box).
- **CPU path byte-identical** — new work is GPU-only `cuda_fwd`/device routes.

## Sequenced phases (forced by dependency: 46a → 46b → 46c → 46d)

### 46a — Static KV cache ✅ (`0cae70d`)
Pre-allocate per-layer K/V `[b, n_kv_head, kv_ctx, head_dim]` once; each token
`slice_set`s the new position in place (one `copy2d` via the existing `hipMemcpy2D`)
and attention reads the `[0..pos+seq]` prefix as a `narrow` view. No `cat`, no realloc.
- **Implementation:** turned out **simpler than feared** — Candle's `Tensor::slice_set`
  (in-place, `&self`, routes to backend `copy2d`) + `narrow` (view) handle it entirely;
  no below-Candle device-buffer management needed. `index_pos==0` overwrites from offset
  0 (stale tail never read). CPU keeps the dynamic `cat`, byte-identical. `kv_ctx` =
  env `AKH_KV_CTX`, default 4096 (≈3.2 GB for the 12B).
- **Result:** output **byte-identical** to the cat path (same greedy tokens). gemma3
  11.5→11.9, gemma4 9.8→10.73 tok/s (short ctx); +3.2 GB VRAM (22.3 GB total). Standalone
  gain modest (cat was cheap) — the value is the **stable-buffer foundation** for 46b/46c.

### 46b — Flash-attention decode kernel
One HIP kernel per (batch, head): stream the static KV, accumulate q·kᵀ with online
softmax (running max + sum), apply scale + causal/length mask, and produce `·v` — no
intermediate `[b,nh,1,seq_kv]` scores tensor, no separate softmax swarm. GQA via the
KV-head mapping; f32 accumulation.
- **Gate:** bit-exact-ish vs the current `q.matmul(kᵀ)`→mask→softmax→`matmul(v)` (≤1e-3,
  online-softmax reordering); measure tok/s (collapses ~9 attention kernels → 1).

### 46c — Allocation-free forward + HIP-graph capture *(the dispatch lever)*
Make the per-token forward issue **zero `hipMalloc`**: pre-warm the caching pool so all
scratch is a cache hit (fixed shapes from 46a/46b make sizes stable), reusing the same
device addresses each token. Then capture the forward on a non-blocking stream and
replay with one `hipGraphLaunch`/token. Route the kernels + rocBLAS + memcpy onto the
capture stream (currently the null stream — uncapturable). Re-capture only on context
bucket change; feed `seq_kv` as a device-side length so the graph stays valid across
tokens.
- **Foundation:** stream/graph bindings + proven capture/replay already committed
  (Phase 45 `a969090`); the blocker (`hipMalloc` aborts capture) is removed by the
  pre-warm + fixed shapes here.
- **Gate:** capture succeeds (no malloc-abort); replay produces identical logits; tok/s
  — this is where most of the 4.3× should land.

### 46d — In-process training path *(the product goal, Phase 27 bridge)*
With a static, allocation-free forward: add a backward / LoRA-adapter pass over the
quantized base for live KG→LLM adaptation. Hidden states + gradients reachable in-proc
for the Neural→VSA bridge. (Scoped in detail in the Phase 27 plan once 46a–46c land.)

## Out of scope (this phase)
- 27B / multi-GPU; bf16/f16 KV-cache compression (PolarQuant, Phase 26); the f16/bf16
  matmul paths beyond what decode needs.
- Beating Ollama on raw serving for speed-only akhs — those stay on Ollama (ADR 070).

## Tracking
- Anima WI: Akh-Medu project `019d9844-ce74-7993-a9f0-c953b8f04a4e` (create a Phase 46
  work item; promote to agent-ready once 46a lands).
- Seshat: ingest this plan + ADR 074 so the corpus stays searchable.

## 46c status (2026-06-23) — HIP-graph capture VALIDATED, byte-exact

The full gemma3 decode forward now captures into a `hipGraph` and replays in one
`hipGraphLaunch`, **bit-identical to eager** (fixed-position probe `AKH_BENCH_GRAPH=1`:
parity max|Δlogits|=0, deterministic). Speedup: **1.33× @ P=22** (replay 41.6 vs eager
31.3 tok/s), **1.26× @ P=362** (27.6 vs 22.0, 3 GB pool). Confirms the dispatch-idle
hypothesis with correct output.

Three capture-enabling fixes (commit `d80d632`), found by bisecting capture failures
through fast candle-core op_tests (single → batched → chain → attn_block):
1. **The correctness bug**: the f16↔f32 cast (`AKH_CAST_LAUNCH`) launched on stream `0`
   instead of `g_akh_stream`. The KV cache's per-layer read-cast ran un-captured on the
   null stream → missing from replay → stale K/V → wrong output (compounded to max|Δ|=43.5).
   Byte-identical for normal execution (g_akh_stream is null there).
2. `event_time`'s `hipEventRecord` (null stream) invalidated Global-mode capture → skip
   while a launch stream is set.
3. rocBLAS automatic-mode per-call workspace hipMalloc aborted capture → fixed 64 MB
   `rocblas_set_workspace`.

API: `candle_core::cuda::HipGraph` + `device_synchronize` + `run_on_private_stream`;
gemma3 `embed`/`hidden_states_from_embed`/`forward_from_embed` split.

**Reframing for 46d (measured):** at P=362 the default 512 MB pool misses 241
hipMalloc/token (growing attention transients overflow the cap) → eager only 12.4 tok/s; a
3 GB pool is 0-miss → 22.0 tok/s (**1.77×**). So the growing-shape malloc-thrash, not the
0.5% seen at P=22, is the dominant large-P cost. The 46d static-buffer rewrite (pre-alloc
attention scratch at MAX size + len-extent views) eliminates BOTH the thrash AND enables
capture at all P → combined **~2× at P=362** (12.4 → 27.6). Remaining 46d work: the moving
KV-write (device-pointer write offset + attention len read from device, updated outside the
captured region — not per-token recapture); the flash-attn stash is groundwork.

## 46d status (2026-06-23) — pool is the lever; adaptive default shipped (+44%/+35%)

The 46d static-shape work was driven by measurement, and the measurement **corrected the
46c reframing above**. The "241 miss/token = 1.77× from a 3 GB pool" was a *fixed-position*
artifact: re-running the SAME P repeatedly, the 512 MB cap can't hold one forward's working
set so it evicts + re-mallocs even same-size blocks. In **real generation** every position
is a new size, so a bigger pool can't pre-cache a size you never re-see — block-padding (the
planned 46d-1) is what would make consecutive tokens share a size.

So 46d-1 (block-pad the decode KV read to a `kv_block` multiple, mask the tail to -inf,
bit-identical) was implemented and **measured against the trivial alternative — and reverted**
(measure-or-revert):

| config (gemma3-12b, long prompt, P≈200-330) | pool | tok/s | h2d |
|---|---|---|---|
| baseline (no padding)        | 512 MB | 7.25  | 690    |
| **pool-only (no padding)**   | **3 GB** | **10.42** | 690 |
| block-pad BLK=256            | 512 MB | 6.21  | 49 585 |
| block-pad BLK=256            | 3 GB   | 9.00  | 49 585 |

Block-padding **net-loses**: its per-token tail mask adds ~49k host-sync crossings, and even
zero-h2d it would only remove the ~240 *cheap* growing-buffer misses/token (~2-3% at a 3 GB
pool). The real bottleneck is the **pool cap**, and a pool sweep confirms a ~3 GB plateau:

  512 MB 7.25 | 2 GB 9.57 | 3 GB 10.39 | 4/6 GB 10.35/10.37 (13.50 GB peak / 24 GB card)

**Shipped (commit `e3a0e60`):** the HIP caching allocator default is now **adaptive with a
floor** — always recycle up to the 512 MB cap, then grow while ≥ 2.5 GB VRAM stays free
(live `hipMemGetInfo`). Strictly ≥ the old fixed-512 behavior: a roomy single-12B GPU
auto-reaches the plateau, multi-akh / op-offload / 27B pressure falls back to the floor (no
OOM). Parity-exact (byte-identical sample), graph-compatible (46c probe: 0 malloc-miss,
1.29× replay, max|Δ|=0).

  gemma3-12b 7.25 → 10.43 tok/s (+44%), 13.50 GB peak
  gemma4-12b 7.13 →  9.60 tok/s (+35%), 16.34 GB peak
  escape hatch `AKH_HIP_POOL_MB=512` restores 7.24 exactly

**The honest 46d conclusion:** the big, free, safe decode win was the pool default, not the
static-shape rewrite. The decode is still only ~8% of the bandwidth roofline → deeply
**dispatch-bound**, so the only remaining lever is the **HIP graph** (46c's 1.26-1.33×).

### 46d-2 (open): real per-token graph decode — large, genuinely uncertain
A real per-token decode graph needs stable shapes (block-padding, now cheap to land *inside*
a graph since the mask goes in a fixed buffer updated outside the captured region) PLUS the
moving KV-write handled by **device-pointer indirection** (a custom KV-scatter kernel + a
masked attention that read the position/length from a device variable, updated between
replays), with a **per-block graph cache** (~16 graphs over a 4 K context) so the captured
shape stays constant within a block. Ceiling ≈ 1.3× on top of the pool win (10.4 → ~13
tok/s); cost = a multi-session custom-kernel effort with real integration risk. The flash-attn
stash is groundwork. **Deferred pending go-ahead** — banking the pool win is a clean stop.

## 46d-2 progress (2026-06-23) — real per-token graph decode: foundations DONE

User chose "full 46d-2 now". The two genuinely-uncertain mechanisms are built, validated,
and shipped; the rest is integration. Three position-dependent reads must follow the
advancing decode position from ONE device variable (`d_pos`): the KV-write column, the rope
cos/sin row, and the causal mask. Per-block graph cache (~16 graphs / 4 K ctx) keeps the
captured shape constant within a block.

- **46d-2a working-set pinning (`c0709e4`).** THE make-or-break. A captured graph's
  intermediate buffers sit in the pool free-list after capture; the decode loop's own
  embed/argmax/sample allocations between replays would draw those addresses and the next
  replay would stomp live data. Fix: capture-scoped malloc tracking (`Pool.capture_track`)
  + `HipGraph` pins the touched-then-freed working set out of the free-list for its lifetime
  (the live result buffer, owned by its Tensor, is skipped); `Drop` unpins. Intra-capture
  reuse preserved (only the small unique set is pinned, not 48×). Test
  `hip_graph_pinning_survives_interleaved_allocs` — VERIFIED meaningful: fails ("victim block
  stomped") with pinning disabled, byte-exact with it on.
- **46d-2b device-pos KV-scatter (`edc2605`).** `kv_scatter_f16_kernel` writes the new K/V to
  cache column `*d_pos` (read on the device) so one captured graph follows the position;
  `KvScatter` InplaceOp2 + `kv_scatter_at` + `DevicePos`. Tests: byte-identical to `slice_set`
  at every position; capture at pos 2 + advance d_pos to 6 + replay → writes col 6 (device-pos
  indirection works inside a graph).
- **46d-2c prep (`7f80e76`).** `u32_device_ptr` extracts the raw device ptr from an
  in-place-updated u32 candle tensor so ONE `d_pos` tensor serves both the candle-op readers
  (rope `index_select`, mask) and the raw-ptr scatter. Tested.

### Remaining (next session — integration, hard unknowns resolved)
- **46d-2c forward:** thread a `GraphDecodeCtx { d_pos, positions(arange), read_len }` through
  the gemma3 decode forward (hip-gated): rope via `cos/sin.index_select(d_pos)`; KV via
  `kv_scatter_at` + block-padded fixed `read_len` narrow; additive causal mask built device-side
  from `positions < d_pos+1` (0/-inf, `broadcast_add`, bit-exact to the unpadded softmax). Gate:
  fixed-position A/B — graph-ready forward (run eagerly) byte-exact vs the normal decode forward.
- **46d-2d loop:** per-block graph cache; per token (outside capture) update `d_pos` (in-place
  slice_set), copy the new embedding into a fixed `embed_buf`, replay, D2H logits → sample;
  recapture at block boundaries. Gate: byte-exact full generation vs eager + measure tok/s
  (target ~1.3× over the 10.4 pool-tuned baseline; revert + record if it doesn't beat eager).
