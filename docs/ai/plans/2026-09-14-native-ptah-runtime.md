# Native Ptah runtime implementation ledger

User-authorized autonomous scope: update akh-medu, replace Ptah's external
Ollama/llama-server inference with resident Candle inside akhrtd, support HIP and
Vulkan, integrate with the homelab session daemon, and retire obsolete Ptah
deployment paths. Ptah downtime is acceptable. Preserve exact model identity,
model artifacts, unrelated providers, session history, and rollback evidence.

## Latest local checkpoint

Updated 2026-09-19. The entries below supersede the 2026-09-14 checkpoint notes;
the chronological evidence further down is retained as written.

- Committed HEAD `bb2c6a2a`. Nothing pushed. Local commits only.
- The native **tap -> projection -> retrieval -> prompt loop** named as the next
  feature in the 2026-09-15 handoff is implemented and validated through five
  private daemons on both devices (`fc1c5fe8`), including recall after
  session/core restart, publication, eviction and session isolation. See
  [recall loop](../research/2026-09-15-native-neural-recall-loop.md),
  [core transport](../research/2026-09-15-real-core-recall-transport.md) and
  [projector validation](../research/2026-09-15-span-projector-validation.md).
- The implementation was consolidated into a source checkpoint (`e291ab44`) that
  builds all five daemons from a clean worktree, with 48 Rust and 8 Python tests
  passing (`8f6f9d24`). Its review found one P2 issue, an unbounded Vulkan
  pipeline cache, fixed with LRU eviction in `9347805b`.
- Full-model resident-idle acceptance passed on both devices (`bb2c6a2a`):
  0% busy, about 10 W, fans stopped, model still owned. See
  [resident acceptance](../research/2026-09-18-resident-gpu-acceptance.md).
- **First keyed throughput measurement** taken 2026-09-19. Prefix-reused decode
  is 16.78 steps/s on HIP and 11.92 on Vulkan; full-prefill decode is 8.95 and
  11.13. Time to first token with a reused prefix is 0.92 s HIP, 1.43 s Vulkan.
  The Vulkan shader cache now reports counters: 99.44% hit rate, saturated at its
  2,048-entry limit with every miss evicting. See
  [keyed measurement](../research/2026-09-19-keyed-session-measurement.md).
- **The variance is explained** (2026-09-19). Pooling seven runs of the same
  benchmark: turns below 200 positions agree to 1%, and the three severe
  outliers all come from **one run** - the only one invoked through
  `resident-probe.py`, which runs continuation checks and a generation pair
  before the benchmark and samples sysfs throughout. Keyed decode is not noisy;
  that run started from a different resident state. The retracted comparison
  set a recorder run against a direct run and blamed the flag. **Rule: compare
  runs with identical preambles.** Repetition alone would not have caught it -
  repeating the recorder run reproduces the wrong number. See
  [measurement variance](../research/2026-09-19-keyed-measurement-variance.md).
- **Performance: claim retracted, question narrowed** (2026-09-19). An earlier
  entry blamed the non-monotonic keyed HIP decode on the Q8 activation GEMV.
  Four interleaved runs show that was wrong: at 191, 256 and 440 positions the
  flag makes no difference at all, because keyed completions already run inside
  `with_full_precision_activations`. The apparent effect was run-to-run
  variance. A real, reproducible 1.6x difference remains at 510 positions only,
  unexplained. The morning's wild profile did not reproduce either, so keyed
  decode at large context is **unstable between processes** and single-run
  comparisons of it cannot support causal claims. Per-step timing still shows no
  outlier steps, so whatever varies shifts the whole distribution. Unkeyed
  results stand: prefill unaffected, unkeyed decode 15-25% faster with the
  quantized GEMV. See
  [decode anomaly](../research/2026-09-19-hip-decode-anomaly.md).
- **Superseded.** An earlier entry here described keyed HIP decode as
  reproducibly non-monotonic in context position, with specific rates at 256,
  322, 440 and 510 positions. Those figures came from single runs. Four
  interleaved runs later showed the profile does **not** reproduce between
  processes, so neither the shape nor its attribution should be relied on. The
  retraction and the controlled data are the entry above; this note remains
  only so the earlier numbers are not rediscovered and believed.
- **Long residency soak passed** (2026-09-19): 160 keyed requests over 14m42s
  with the model owned. Decode flat, RSS flat at 0.600 GB, device VRAM flat at
  24.35-24.49 GB, resident idle 0% busy at 10.72 W afterwards. The prefill
  "drift" seen in a shorter run is a one-time ~1.1 s warm-up that then holds
  flat. See [long soak](../research/2026-09-19-long-residency-soak.md).
- **Five-daemon canary rerun passed** on both devices (2026-09-19) with all five
  daemons rebuilt from `028804f1`: recall after session/core restart, span
  publication, history eviction, provider continuity and session isolation. No
  processes left behind. See [canary rerun](../research/2026-09-19-canary-rerun.md).
- **Context assembly moved to the core** (2026-09-19), inert so far. The
  `/v1/context` contract now carries `messages`, `budget` and `assembled`, all
  skipped when unset so an unused request serializes byte-identically and
  neither side receives a field it would reject under `deny_unknown_fields`.
  `core_service::assemble_context` selects through
  `SafetyEnforced(SelectingManager)` when messages are supplied, returning the
  chosen messages with a dropped count and per-section tokens. Nothing sets
  them yet, so behaviour and the canary are unchanged.
- **The last join is blocked on a topology decision** (2026-09-19). The durable
  history is the `Conversation` in **akhrtd's own Store**, accumulated across
  turns as the provider-session continuation; the session service never holds
  it, `RuntimeJobRequest` has no message array, and akhrtd has no core client.
  So no party currently holds both the history and the policy. Either give
  akhrtd a core client, or move the conversation into the session record; the
  first widens a trust boundary, the second relocates the state the keyed
  cache's exact-prefix reuse depends on. Needs the user. See
  [materialization](2026-09-19-context-manager-materialization.md).
- **Context selection path built** (2026-09-19): `SelectingManager` selects over
  a message array, preserving tool traffic and recall markers, failing closed
  with a miette diagnostic when the pinned remainder plus the current turn
  overflow. Additive: `EpisodeState` gained an optional message view and the
  four shipped strategies are untouched. Not yet selected by configuration or
  wired into the native path.
- **ContextManager placement decided** (2026-09-19, design only, no code): the
  strategy stays in akhomed and becomes authoritative for native turns; akhrtd
  keeps `native_window` as a defensive capacity check and no eviction policy of
  its own. Rejecting the alternative keeps one cognitive core. First slice and
  acceptance gate in
  [materialization scoping](2026-09-19-context-manager-materialization.md).
- **Vision is feasible against the exact model** (2026-09-19): the pinned Ptah
  manifest carries a `qwen3vl_merger` projector layer and the 931 MB blob is
  already local, so no artifact is fetched and model identity is preserved.
  Projector resolution by manifest layer is implemented and tested against the
  real blob; the vision tower, merger, deepstack routing and image positions
  remain. `quantized_qwen35.rs` has no multimodal support today, but this
  artifact needs no deepstack routing (all 27 flags false, no deepstack
  tensors), so that step drops out. Dynamic-resolution preprocessing is
  resolved against llama.cpp mtmd and implemented with six CPU tests. The mrope
  sections the model declares are now read, validated against the rotary width
  and mapped per pair, with tests pinning that they collapse to plain 1-D RoPE
  for text - which is why the bit-exact gates were right to pass. The tower
  forward is implemented and runs the real projector end to end on CPU,
  producing finite embeddings at the language model's 5120 width, and **runs on
  CPU, HIP and Vulkan** agreeing within 8.1e-06 absolute. Getting it onto the
  devices required replacing `conv2d`, which neither backend implements, with an
  equivalent unfold plus matmul, and dispatching softmax and GELU per device.
  Still not numerically validated, and not validatable on this machine: no
  external consumer of this GGUF is installed, and a second in-house
  implementation would share the same comprehension errors rather than test
  them. See
  [device runs](../research/2026-09-19-vision-projector-devices.md) and
  [vision scope](2026-09-19-native-vision-scope.md).
- **Long-context memory quantified** (2026-09-19): only 16 of 64 target blocks
  hold a growing KV cache (`full_attention_interval = 4`); the other 48 are
  constant-memory recurrent. At F32 that is 128 KiB per token, so this 24 GiB
  card holds roughly 40,000 tokens against an advertised 262,144 — a 6x gap.
  The runtime now reports this itself at load (`kv_bytes_per_token`), returning
  131,072 bytes and projecting 34.36 GB at full context, verified on device.
  The PolarQuant helpers remain called from nowhere, and their stated reason for
  deferral (forking candle-transformers) is obsolete now the model is vendored.
  Quality is unmeasured and a lossy KV path needs a different acceptance gate
  than the bit-exact one. See
  [memory model](../research/2026-09-19-long-context-memory-model.md).
- **TurboQuant implemented and measured** (2026-09-19): arXiv:2504.19874
  Algorithm 1, validated against the paper's published centroids and distortion
  rates. On this model's real KV it reaches mean cosine 0.9954 at 4 bits and
  0.9829 at 3, treating keys and values identically, and **at 2 bits already
  beats PolarQuant at 3**. At 4 bits that is 7.8x compression, putting the
  advertised 262,144-token context within reach of this card for the first time.
  A gated lossy KV path is now wired into the model and **passes the five-daemon
  canary at 4, 3 and 2 bits**, recalling the correct number after eviction and
  restart. The fixture does not discriminate between those widths, so 4 bits is
  the conservative default. **Real paged storage is now wired into
- **Release bundle done** (2026-09-19). A manifest records what a build is:
  each binary's digest, the kernel libraries that binary actually resolves -
  read back from the path baked into it, because the tree holds six identically
  named candidates - and the pinned model, tokenizer and projector digests. It
  refuses a dirty tree. A companion assembles a relocatable bundle with `lib/`
  beside the executables and records ROCm 6.3.4-76 and kernel 6.8.0-47-generic
  from files rather than from commands on PATH, which were absent and would
  have recorded a misleading `null`. **The relocated bundle passes the
  five-daemon canary.**

  Not proven: that the kernels loaded from the bundle's `lib/` rather than the
  compile-time path, since both exist here. That needs a tree-free environment.
  See [release manifest](../research/2026-09-19-release-manifest.md).

- **KV compression works end to end on HIP** (2026-09-19). Pages hold one byte
  per code on the device and decode there via gather, norm scale and inverse
  rotation, proven equivalent to the host decode within 1e-4. A six-turn
  benchmark that **never finished** with host-side decode now completes, at
  about a third of decode throughput (11.2 against 16.5 steps/s on the stable
  early turns). The canary passes with `recent=4`, which forces paging every
  turn, still recalling the right number after eviction and restart.

  Two corrections this produced. The earlier canary passes at 4, 3 and 2 bits
  **never exercised paging** - a 1,024-token window ages nothing out of a
  16-position exact window - and the first real paging workload failed at once
  on an append invariant that assumed the state holds every position. And the
  Vulkan half is **not blocked after all**: its allocator gives every dtype four
  bytes per element, but three 8-bit codes can be packed per `u32` word and
  unpacked with `affine`, `floor` and binary ops the backend already has. Three
  rather than four, because unpacking goes through f32, exact only to 2^24. A 3x
  saving with no new kernel. Unimplemented and unmeasured: the extra elementwise
  passes could cost more than they save.

- **KV compression now works on Vulkan too, and the packing is nearly free**
  (2026-09-19). The packed layout above is implemented, selected automatically on
  Vulkan, and measured on a live cache: **2.94x** smaller KV (1,648,128 bytes
  against 4,849,664, 16 layers, 592 paged positions) for about 5% of decode.
  Warm decode is 13.38 steps/s uncompressed, 8.33 with the byte layout, 7.98
  packed - so nearly all of the ~40% cost is the page decode both layouts share,
  not the unpack. HIP is unaffected and still reports an honest 3.94x, a `u8`
  really being one byte there.

  **Greedy agreement is not evidence of quality here.** Output was byte-identical
  across all three Vulkan arms, but the same 4-bit policy on HIP changes it -
  30 tokens uncompressed against 23 paged, from two devices that agree with each
  other uncompressed. Measuring only Vulkan would have let me report token
  identity as a property of 4-bit paging when it is luck at one prompt. Whether
  4-bit paging is acceptable is **unsettled**: it needs the 0.9999 cosine gate run
  against paged reconstruction on both devices.

- **Compressed KV fails the 0.9999 cosine gate at every bit-width, including 8**
  (2026-09-19). Fidelity is now measured on the live path, at the moment
  positions are absorbed, running the same device decode attention reads
  (`AKH_QWEN35_KV_PAGE_FIDELITY=1`). Worst cosine over 16 layers: **0.99788** at
  8 bits, 0.98540 at 4, 0.96627 at 3, 0.91491 at 2. Mean at 8 bits is 0.99994 and
  grazes the gate, but the gate is on the worst vector. This had never been
  measured - the 4/3/2-bit canary is behavioural, and the quality report
  simulates a host round trip rather than reading the live path.

  That the 0.9999 gate is a **parity** gate, which a lossy compressor cannot meet
  by construction, is a real argument - and it is an argument that compressed KV
  needs its **own acceptance criterion**, not that it passes this one. That
  criterion does not exist and is a decision to take deliberately. **Compressed
  KV stays off by default and is not proposed for enabling.**

  **Retracted:** I guessed the HIP/Vulkan token divergence came from the page
  decode's device arithmetic. The two devices reconstruct identically to 8
  decimal places at every width (largest gap 6.1e-08), so that is wrong. At 4
  bits only, identical reconstructions give different text on the two devices -
  a knife-edge near a token boundary, not a quality difference. Greedy agreement
  there reported "nothing changed" on one device and "something changed" on the
  other from the same codes: not merely a weak signal but a misleading one.

- **Long-context gate: the runtime advertised 262,144 tokens and can serve
  48,672** (2026-09-19). The only admission check was against the GGUF's declared
  window, which is what the artifact supports, not what the card holds. At 128 KiB
  of KV per token the declared window needs 32 GiB on a 23.98 GiB card that
  already holds 16.8 GB of weights, so a request could pass every check and then
  fail on an allocation part way through prefill.

  `memory_bound_context_limit()` now reports what fits: **48,672** tokens on HIP
  (`hipMemGetInfo`) and **40,226** on Vulkan (`amdgpu` sysfs, since wgpu has no
  portable free-memory query). The two differ by ~1.1 GB because sysfs is
  card-wide and includes the compositor; that is the more conservative figure and
  the right bias for a gate. A device that can report neither stands the gate
  down rather than inventing a limit. Verified in both directions - normal
  requests admit at unchanged decode throughput, oversized ones exit 1 with zero
  generation before anything is allocated. The miette diagnostic names the limit,
  the per-token cost, the adjustable reserve and every remedy, including
  compressed KV **with its 0.9999 caveat attached** so nothing recommends it on
  the strength of a help string.

  **My first verification of this was hollow**: the rejection case looked like a
  pass because my grep excluded the generation lines, while generation had in
  fact completed with exit 0. The gate was on `generate_inner` and `prefill`,
  neither of which serves completions. Absence of expected output is not evidence
  of rejection. All four admission points are now covered.

- **The soak falsified that gate: KV is 23% of what a token costs**
  (2026-09-19). The gate above **admitted requests that then failed on
  allocation** - the exact failure it was built to prevent. A 20,833-token prompt
  was admitted and died on `hipMalloc(235929600)`; so was a ~41,000-token one.
  48k and 64k were refused correctly, and every earlier test used 15-token
  prompts, which pass under both the wrong and the right limit.

  A 10,417-token request consumes **5.44 GiB**, of which KV is 1.30 GiB:
  **547.5 KiB per token against 128 KiB of KV, a factor of 4.28**. Attention
  scores scale with chunk x positions, so the overhead grows with context and no
  fixed reserve can absorb it. The gate now sizes on total cost, with an
  **empirical single-measurement** multiplier of 4.5 that admits the largest
  request measured to work and refuses the smallest measured to fail;
  `AKH_CANDLE_CONTEXT_COST_MULTIPLIER` overrides it for another model or card.
  Verified both ways: 20,833 now refused with nothing allocated, 10,417 still
  admitted and completes.

  Serviceable context, as the measurement got honest: declared **262,144** ->
  KV-only gate **48,672** -> measured **10,816**. Four percent of the advertised
  window.

  **Unresolved.** The gate refuses a repeat of a request that just succeeded:
  after the 10,417-token run free memory reads 2.69 GiB and the limit collapses
  to 913, because `hipMemGetInfo` excludes the allocator's reusable cache.
  `release_cached_memory` is restricted to a serialized unload boundary, so
  calling it here would trade a wrong refusal for a correctness risk. Needs a
  pool-aware free figure or a supported reclaim point.

  **Performance, unbidden:** prefill of 10,417 tokens takes 282-298 s, about 35
  tokens/s against 0.86 s for 15 tokens. Long context is unusable on time well
  before memory becomes the binding constraint. Item 7, not addressed. See
  [long-context gate](../research/2026-09-19-long-context-gate.md).

- **Item 7: half of prefill is the repeated KV expansion, and the ceiling is 2x**
  (2026-09-19). Prefill chunks were hard-coded to 128, so a 10,417-token prompt
  runs as 82 forwards, each re-expanding the whole accumulated K and V through
  `repeat_kv` in all 16 attention layers. Doubling the chunk predicted 2x if that
  dominated; it gave **1.32x** (271.4 s -> 205.3 s, 38.4 -> 50.7 tok/s), which by
  Amdahl puts the chunk-count-proportional share at **49%**. The rest is
  per-token weight-bound matmul that no chunking change touches.

  So eliminating the expansion entirely - broadcasting kv-heads inside the
  attention matmul instead of materialising a copy - is worth about **1.95x**,
  roughly 75 tok/s, taking a 10k prompt from ~271 s to ~140 s. Worth doing, and
  **not enough to make long context interactive**. It is a numerical-path change
  and would need the 0.9999 cosine gate against the current implementation, not
  greedy agreement. Not attempted.

  Chunks of 512 and 1024 both fail on allocation: peak prefill memory scales with
  `chunk x positions`, the same term that makes a token cost 547.5 KiB against
  128 KiB of KV. **The default stays 128** - the 1.32x was measured at one
  context length, and raising it on one data point would repeat a mistake already
  made twice today. `AKH_QWEN35_PREFILL_CHUNK` makes it selectable for
  measurement. See [prefill cost](../research/2026-09-19-prefill-cost.md).

- **Item 5: the vision projector agrees across CPU, HIP and Vulkan**
  (2026-09-19). Vision was recorded as blocked on an external consumer of this
  GGUF. That is true of **semantic** validation and was too broad: cross-device
  agreement uses the project's own 0.9999 cosine gate and needs nothing external.
  All three backends agree at **cosine 1.0000000000** over the full 20,480-element
  projection, worst relative L2 4.05e-06. That covers the substitutions the tower
  makes because the device backends lack the ops: patch embedding as
  unfold+matmul instead of `conv2d`, and dispatched `softmax_last_dim` and `gelu`.

  **Summary statistics are not the test.** Mean, standard deviation and L2 norm
  are invariant under permutation and sign flip, so the probe now dumps the full
  projection and the gate compares elementwise. Confirmed the gate can fail: a
  dump with two elements swapped - identical on all three summary statistics -
  scores 0.9998978914 and is rejected.

  Two fixes this needed: the probe **did not build under HIP** (`new_vulkan` was
  not cfg-gated), and it had no way to emit more than a summary.
  `scripts/session-platform/compare-vision-dumps.py` makes the gate repeatable.

  **Still open:** semantic validation - nothing here shows this projector computes
  what Qwen3-VL's does, only that the backends agree with each other; a misread
  resize policy or mrope split would be wrong identically everywhere and pass.
  Position-triple threading for image tokens is also untouched. See
  [vision device gate](../research/2026-09-19-vision-device-gate.md).

- **Item 4: the seam had a silent fallback; the topology fork is confirmed real**
  (2026-09-19). `context_budgeted` existed in exactly one place in the tree -
  `native_window.rs` and its own tests. **The core never emitted it.** So once
  the topology lands and messages flow, the core would assemble, and the runtime
  would read `core_budgeted == false` and run its own selector over a list the
  core had already decided: two context policies disagreeing, silently, with the
  turn still succeeding on different context than the core chose.

  `assemble_context` now marks every message it returns, and strips any inbound
  marker first so a flag echoed back in history cannot describe an assembly that
  did not happen. The constant moved to `akh-session-types` as
  `CONTEXT_BUDGETED` and both crates use it - as two string literals, a change
  on one side would not fail to compile, it would quietly restore the double
  eviction. Four tests cover it on the core side, six on the runtime side.

  **The fork is not resolvable by reading the code.** `RuntimeJobRequest` carries
  `input`, `instructions` and `tools` and has **no message array**: akhsessd
  holds turns, not a rendered conversation, while the message array is built in
  `internal.rs` in the agent loop, which has no core client. No party holds both
  the conversation and a line to the core. Either the agent loop gains a core
  client or the conversation moves into the session record - **a user decision**.
  Everything above is required under both. See
  [context seam marker](../research/2026-09-19-context-seam-marker.md).

  Three corrections this produced. **The byte layout never ran on Vulkan at
  all**: `to_dtype` had no `U8 -> U32` arm, so the decode aborted the first time
  a position aged out. Compressed KV there was failing closed, not saving
  nominally, and the same never-pages blind spot hid it. Both casts are now in
  the vendored Vulkan backend with a device test. **And the accounting was wrong
  twice, both times flattering**: it charged one byte per code unconditionally,
  so the byte layout claimed 3.94x on the one backend where it is a net loss;
  the first fix branched on layout but kept the same assumption one level in.
  Element width is a property of the device, not the dtype's name. Corrected,
  the byte layout reports **0.996x** - it stores more than the f32 it replaces.
  The ratio is now read from the live path via `kv_page_residency()`, which
  previously existed only in a unit test. Measured at one short prompt and 592
  positions; the decode cost grows with context and is unmeasured there. See
  [Vulkan packed KV](../research/2026-09-19-vulkan-packed-kv.md).

  `AttentionState`** — bit-packed pages per (batch, head) slot, a shared
  quantizer, the recent window exact — and passes the canary at 4 and 2 bits
  with 21 model tests still green. **Host-side decode is now measured as not
  viable**: six keyed turns to 381 positions finished in about 3 minutes with
  paging off and did not finish in 30 minutes with it on, the GPU idling at 10%
  while the host converts. The canary passes only because its 1,024-token window
  ages almost nothing out. On-device decode is therefore a precondition for
  enabling this, not an optimization - and it needs a **backend kernel**, not
  just composition, and the two backends need different work. HIP stores `u8`
  at one byte per element and needs only a `u8 -> u32` cast kernel. **Vulkan
  allocates four bytes per element for every dtype**, so `u8` saves nothing
  there before any cast happens; it needs packed small-integer storage and an
  unpacking gather. Compressed resident KV is therefore achievable on HIP and
  not on Vulkan without backend storage work. See
  [lossy KV gate](../research/2026-09-19-lossy-kv-canary.md).
  See [TurboQuant](../research/2026-09-19-turboquant-kv.md).
- **KV quantization quality measured, and it fails** (2026-09-19): PolarQuant
  round-trips this model's real KV at mean cosine 0.904 for keys and 0.708 for
  values, worst case -0.371, against a project gate of 0.9999. The helpers must
  not be wired in. An outlier-aware quantizer with per-channel keys and
  per-token values is the real work, and the new
  `CandleBackend::kv_quantization_report` is its acceptance harness. See
  [KV quality](../research/2026-09-19-kv-quantization-quality.md).
- Prefix reuse is dropped whenever a turn ends at its output cap
  (`CandleFinishReason::Length`). Documented behaviour, now measured: a third of
  otherwise-consecutive turns paid a full prefill at a 48-token cap.
- Scope remains local-only: no Archibald/Forgejo work or shared-service changes.
  No production native daemon is installed. The old Ptah inference boot unit is
  stopped and disabled; artifacts are retained.
- Remaining, in order: long residency soak (starting from the HIP memory
  finding), rebuild and rerun five-daemon acceptance, ContextManager
  commit/materialization, exact vision path, live KV compression and
  long-context gates, performance, release bundle, then canary/cutover/cleanup.
  The last needs shared services and is explicitly blocked pending user
  direction. See [goal brief](2026-09-19-native-ptah-runtime-goal.md).

## Baseline and evidence

- Development checkout `/home/toasty/ws/agi/akh-medu` fast-forwarded from
  `2d73c068` to `e7e70bfe`; deployed `/home/toasty/src/akh-medu` is a separate,
  older non-git tree. Do not confuse its binaries with new builds.
- Exact target GGUF SHA-256:
  `3445102e9cde5d562508642c100a2f5ac3368a5a3f748442811d7a95daee3bec`.
  Qwen3.8-27B Q4_K_M, GGUF `qwen35`, 64 target blocks plus MTP block.
- RX 7900 XTX, gfx1100, 24 GB, kernel 6.8.0-47, local ROCm 6.3.4;
  Ollama also bundles ROCm 7.2. No driver upgrade/reboot performed.
- HIP external runners reproducibly remain at approximately 100% GPU and
  118–120 W while idle. Two HIP stream creations without model/compute also
  reproduced persistent high power; process exit clears it. A default-stream
  Candle matrix multiply stayed at 0% and 8–11 W with its process alive.
  This is NOT yet a full-model idle regression pass.
- Exact-model existing Q8/Q4_K kernel parity test passed: minimum hidden cosine
  0.999960874, logit cosine 0.999991194, same greedy tokens, 1.160x median
  three-step kernel-path speedup. NOT an end-to-end runtime speed claim.
- Gemma3-12B Candle benchmark approximately 11.9 decode tokens/s. External HIP
  and Vulkan short-prompt benchmarks approximately 50–59 and 53–55 tokens/s;
  long prompt HIP approximately 49–51 versus Vulkan 34 tokens/s. Prompts and
  generation paths differ from Candle harness; these are exploratory, not
  acceptance comparisons. External tests disabled flash attention, not tuned
  maxima. Exact Qwen external-runner loading attempts stalled and were stopped;
  no exact-model HIP/Vulkan throughput comparison obtained yet.
- Upstream matching report: [ROCm issue 2625](https://github.com/ROCm/ROCm/issues/2625).
  Current [Candle Vulkan issue](https://github.com/huggingface/candle/issues/1810)
  is not proof of a usable quantized Vulkan backend in this fork.

## Ordered implementation and acceptance gates

1. **Source update and native dependency removal.** Updated source; actual
   llama-cpp-2 dependency and Rust binding usage removed. NLU/extraction carrier
   interfaces retained with Candle compatibility. Focused translator (30) and
   extraction (10) tests passed. Historical oracle documentation and HTTP
   adapters for unrelated providers are not deletion targets.
2. **Native akhrtd tracer.** Typed explicit model/device configuration;
   resident, serialized compute; common durable model/tool loop; exact resume,
   tool brokering, cancellation and incomplete-output failures. First build
   passes, integration tests in progress. Text is buffered for safe tool parsing;
   no claim of token streaming yet. No silent unsupported-device CPU fallback.
3. **Full-model HIP idle gate.** New `candle_session_probe` performs two resident
   calls, checks greedy repeatability and samples GPU power while keeping model
   alive. Run with exact tokenizer/model and record output. Default-stream
   policy must hold through model load, decode, cancellation and subsequent
   requests. Graph/private streams need their own power gate.
4. **Exact model fidelity and performance.** Verify tokenizer token IDs/merges,
   special tokens/chat template, reference logits/text, prompt/decode timing,
   memory at matched context and quantization. Optimize only with parity gates.
5. **Vulkan.** Real quantized GPU kernels and tensor coverage, not feature names
   or CPU fallback. Isolated Q4_K probe first; full Qwen hybrid forward and
   runtime adapter remain required. Compare identical workloads to HIP.
6. **Session ownership/compression/vision.** Follow ADR 098: model weights shared,
   position-dependent state session-owned; hidden taps, existing virtual context,
   exact projector, quantized live KV pages (not unused compression helpers).
   Quantization quality and long-context memory measured before activation.
7. **Homelab canary.** Connect native runtime to canonical session daemon with
   scoped credentials and explicit Ptah binding; test message/tool/resume/cancel
   through actual session ingress. Other akh-platform runtime providers remain.
8. **Cutover and cleanup.** Remove obsolete Ptah startup units/config references
   only after canary. Keep model files and rollback artifacts, verify no orphan
   inference processes and quiet idle after service startup. Report recoverable
   removals. Do not declare complete until all required gates actually pass.

## Operational status at start of implementation

`ptah-inference.service` stopped but enabled; old system/user akhomed stopped and
disabled. Neither is yet the native canary. Homelab context
`admin@talos.wegmueller.it`, canonical namespace `akh-platform`, session deployment
`sessions`; old `akh-medu/akh-medu-seshd` also exists and requires ownership audit
before cleanup. No cluster mutations made during initial discovery.

## Verified progress (2026-09-14)

- Latest user direction: leave Archibald alone and focus on local runtime work.
  Shared cutover/restarts and further remote SSH work are paused regardless of
  the earlier maintenance approval. Other sessions own Forgejo; do not intervene.
- Local follow-up isolated the fusion accuracy failure: repeated reference
  hidden/logit vectors are bit-identical; all 33 positions pass without Q8, but
  Q8+fusion reaches hidden cosine 0.996354179 at step 29 (logits minimum
  0.999386782). Greedy IDs match throughout. The combined optimization remains
  rejected under the original threshold; ordinary HIP remains unchanged.
- Combined HIP+Vulkan release probe built successfully. Updated Vulkan matched
  counting workload: load 30.491 s; 36 prompt/128 output tokens, exact same text
  as HIP/original runner; decode 3.298/3.674 steps/s, first-token 3.412/2.984 s.
  The 10-second resident idle sample after 10-second settling FAILED utilization
  (5.7% versus 5% cap), while power 26.8 W passed its baseline 10 W + 20 W cap.
  A 30-second settle/60-sample repeat is required before attributing this narrow
  miss to a persistent idle defect. Do not call this run a passing idle gate.
- The longer Vulkan repeat PASSED with the same model resident: 30-second settle,
  60 samples, mean 0.7833% GPU / 12.6833 W from a 10 W baseline. Both 128-token
  counting outputs remained identical; decode 3.472/3.641 steps/s. This rules out
  sustained high idle load during that longer observation, not every possible
  future driver/workload failure. The original short-window miss stays recorded.
- Opt-in blockwise Vulkan single-token GEMV (`AKH_VULKAN_GEMV_BLOCKS=1`)
  passed the public hardware suite and 24 additional Q4_K/Q6_K comparisons
  against an f64 CPU oracle, including odd block counts, contiguous inputs and
  real GPU stride-2/offset-1 views (maximum absolute error 0.00001422).
  Actual-weight median microbenchmarks, reference -> candidate: Q4_K FFN up
  780.7 -> 681.9 us, Q6_K FFN down 990.4 -> 680.4 us, vocabulary output
  6074.5 -> 3267.1 us. These sequential measurements are preliminary, not
  counterbalanced whole-model throughput evidence. The candidate remains off
  by default; full-model numerical, throughput and resident-idle validation are
  still required. The current root session probe predates this candidate and
  must be rebuilt before testing it.
- Autonomous local continuation: the combined HIP/Vulkan probe was rebuilt
  with blockwise GEMV and passed the same 36-prompt/128-output counting fixture
  twice. Decode was 3.0565/3.2860 steps/s, TTFT 3.4644/3.0538 s, load 32.7722 s.
  A concurrent CPU test build makes this a provisional speed comparison; no
  whole-model speedup is claimed. Resident idle passed after 30-second settling
  and 60 samples: 0.7167% / 12.4 W, baseline 10 W. A new exclusive-artifact test
  compares all hidden/logit vectors and greedy IDs over 33 forced positions,
  with the unchanged 0.9999 cosine gate.
- Submission-overhead investigation has a public-tensor ordering/latency probe
  at `experiments/native-vulkan/src/bin/submission.rs`: incomplete batches,
  copies, explicit fences, and two callers sharing a device. Immediate-route
  baseline passed correctness and measured 10.737 ms for 256 dependent tiny
  operations. A bounded command-buffer batching candidate is guarded by
  `AKH_VULKAN_BATCH_SUBMIT=1`, default off; numerical kernels are unchanged.
  Candidate validation and timing are pending; it is not deployment-approved.
- Block GEMV full-model parity PASSED (122.17 s), all 33 positions: hidden/logit
  cosine rounded to 1.000000000 and every greedy token matched. Bounded
  submission full-model parity separately PASSED (151.16 s), with independently
  constructed devices/models and block GEMV disabled: the same 33-position
  numerical/token gate passed throughout. These compare Vulkan routes, not
  Vulkan versus HIP or the original runner's full vectors.
- Submission microbenchmarks in immediate/batched/batched/immediate order:
  9.921 / 6.560 / 6.446 / 11.177 ms for 256 dependent operations. Both modes
  passed copy/fence/tail/concurrent-caller ordering. Batched submission plus
  block GEMV also passed the public tensor and 24 packed-weight oracle checks.
  Production Qwen state/checkpoint tests: 9 passed; completion/backend: 12
  passed; legacy-translator compatibility: 30 passed. No acceptance thresholds
  or production defaults were changed.
- Added a lightweight test library in `experiments/native-vulkan` that imports
  the actual production Qwen35 source by path (no model implementation fork).
  Its first dependency build is pending. Future model-only test edits should
  avoid rebuilding the complete akh-medu application.
- No-compilation full-model submission comparison, same binary and 36/128
  counting fixture, block GEMV disabled: immediate decode 3.5415/3.7685 steps/s,
  TTFT 3.0943/2.9328 s, load 28.7634 s; batched decode 4.4872/4.2579 steps/s,
  TTFT 3.0432/2.3636 s, load 30.0597 s. Exact counting output matches in all
  four passes. This supports a real improvement but remains far below the
  30.731 original-runner reference. Batched resident-idle observation is pending.
- Lightweight test target built successfully; its nine production-source CPU
  model tests passed (0.03 s). The next weighted-RMS normalization slice has a
  public-interface numerical/latency probe prepared, with an intentionally
  unsupported candidate stub awaiting the first red run; it is not wired into
  model execution. No normalization optimization has been implemented yet.
- Batched submission resident idle PASSED: 0.8667% / 13.8167 W over 60 samples
  after 30-second settling. Its initial 31.6667 W baseline still reflected the
  immediately preceding benchmark; the observed final power also clears the
  earlier settled 10 W + 20 W cap. The probe now has an explicit pre-baseline
  settling interval (default 10 seconds) to avoid this carry-over in future runs.
- Weighted Vulkan RMS normalization progressed red -> green: the public probe
  first failed with the unsupported candidate, then passed an independent f64
  oracle after adding the GPU kernel. Coverage includes widths 1/127/128/255/
  256/257/5120/17408, strided/nonzero-offset views, 65537-row 2D dispatch tails,
  input immutability, and invalid shape/dtype/device/epsilon rejection. Maximum
  observed absolute error was 0.00000028. Short microbenchmark timings varied
  markedly with GPU warm-up (reference about 335-784 us; candidate about 109-296
  us), so full-model throughput remains the deciding performance gate.
  The model route is guarded by `AKH_VULKAN_QWEN35_FUSED_RMS=1`, default off;
  exact-model parity and throughput/idle tests are pending, not accepted yet.
- Weighted RMS full-model parity PASSED (156.87 s), batching enabled on both
  routes and block GEMV disabled: all 33 hidden/logit/greedy positions passed.
  Full native completion probe then produced the exact expected 128-token text
  twice at 5.1237/5.2734 decode steps/s, TTFT 3.1134/2.3408 s, load 29.3330 s.
  Settled resident idle PASSED: baseline 11.6667 W, final 0.75% / 13.0667 W
  over 60 samples after 30-second settling. A same-binary normalization-off
  reference repeat is running to check the throughput comparison.
- A fast core-model benchmark example is being added to the lightweight target.
  It imports production model source and pins both model/tokenizer SHA-256,
  asserts the complete counting response and reports readback-fenced decode
  timing. It does not replace the native completion/daemon or idle-power gates;
  its first build and execution are still pending.
- Same-binary normalization-off reference repeat completed at 4.1956/4.1368
  decode steps/s (TTFT 3.1636/2.3576 s), confirming the earlier normalization-on
  5.1237/5.2734 improvement without concurrent compilation. Batching and fused
  weighted RMS are now the Vulkan defaults in source; `AKH_VULKAN_BATCH_SUBMIT=0`
  and `AKH_VULKAN_QWEN35_FUSED_RMS=0` retain the reference paths. Block GEMV and
  HIP recurrence fusion remain default-off. Full-model numerical, real native
  counting output and resident-idle gates passed before this promotion; overall
  performance parity and shared deployment remain explicitly incomplete.
- Promoted-default public tensor, submission and RMS suites passed. The new
  production-source core benchmark verified both artifact hashes, asserted the
  entire expected counting response, and measured 5.1297/5.4118 steps/s, close
  to the native completion probe. It is a faster iteration tool, not a daemon
  or resident-idle acceptance substitute.
- Allocation hypothesis measured before implementation: 128 allocation/drop
  operations took 0.103/0.101/0.105 ms host time for widths 4/5120/786432;
  dependent arithmetic took 5.244/5.228/6.400 ms including the final fence
  (host enqueue 2.363/2.385/2.819 ms). Retained-alias, zero-initialization and
  invalid-index recovery tests passed. A buffer pool is not justified by these
  CPU allocation costs and has not been implemented. Local perf sampling is
  available without changing global kernel settings; warmed decode profiling
  is the next diagnostic step.
- Warmed local decode perf capture completed (15 s, 1418 samples, zero lost
  samples): `/tmp/ptah-vulkan-perf.Hlwh7I/decode.data`. Inclusive CPU attribution
  was about 80.77% under dispatch, 40.46% under queue submission and 25.42%
  under encoder finishing (overlapping call-tree percentages, not GPU time).
  Resource-tracker resizing/dropping was prominent. No global tracing settings
  were changed and only the owned benchmark process was sampled.
- Added opt-in shared command encoders (`AKH_VULKAN_BATCH_ENCODER=1`), keeping
  kernels and current default routes unchanged. Copies, readback and explicit
  synchronization participate in the same ordering; incomplete batches flush.
  Encoder validation failure poisons only that device instance so earlier
  uncomputed tensors cannot masquerade as valid zeros. The forced-failure /
  fresh-device-recovery hardware test passed (0.33 s).
  Counterbalanced 256-operation timings: reference 6.484/6.756 ms versus shared
  encoder 5.053/4.806 ms. Public tensor/RMS, alias/zero/error-recovery and
  concurrent-caller tests passed. Initial full-model vector-equality parity
  passed all 33 positions (148.01 s); the gate has subsequently been tightened
  to compare `to_bits()` explicitly, including signed zeros, and needs its
  strict rerun before any default promotion. Core throughput/idle are pending.
- The strict shared-encoder rerun PASSED (150.83 s): all 33 complete hidden and
  logit vectors have identical `to_bits()` values, including signed zeros;
  greedy IDs are unchanged. Preliminary core decode was 5.5403/6.1542 steps/s
  with the complete counting response preserved. Resident idle remains pending.
  The fast core benchmark now optionally samples a settled baseline and keeps
  the model alive for a 30-second settle plus 60-sample idle window using the
  same read-only sampler as the native completion probe. This does not replace
  the actual runtime/HTTP integration gate. Combined HIP/Vulkan akhrtd is being
  rebuilt for a private four-daemon Vulkan canary; no shared service is touched.
- Independent full-model Vulkan-versus-HIP-f32 comparison PASSED (141.06 s),
  33 forced positions with shared Vulkan encoders enabled. Minimum printed
  hidden cosine 0.999999999; logit cosine rounded to 1.000000000 throughout;
  every greedy token matched. Values are not bit-identical across backends,
  as expected for different arithmetic implementations. HIP Q8 and recurrence
  fusion were explicitly off for this independent f32 reference comparison.
  Combined HIP/Vulkan akhrtd release build passed (5m34s), staged locally only.
- Shared-encoder core throughput/idle gate PASSED without concurrent builds:
  5.4845/6.1274 decode steps/s, model TTFT 2.8160/2.2571 s, exact full counting
  response, load 29.0964 s. With model retained, settled 60-sample idle was
  0.4333% / 12.15 W from an 11 W baseline. The real four-daemon Vulkan canary
  is now running with the encoder explicitly enabled; default remains off.
  The next convolution slice has a public f64/history/snapshot/error test and
  an unsupported candidate stub only. It is not connected to model execution.
- The real four-daemon Vulkan canary PASSED: authentication, native model write,
  approval before effects, session restart with approval pending, native read,
  durable tool receipts, exact provider continuation, runtime restart/artifact
  reload, and recovery of the remembered word. All temporary daemons exited and
  the harness removed its disposable fixture/state directory. No shared service
  or user session data was changed. Shared encoders are now enabled by default
  in source; `AKH_VULKAN_BATCH_ENCODER=0` retains the previous route. The staged
  daemon binary ran this same route with the flag explicitly enabled; no live
  production daemon was installed or started.
- Vulkan causal convolution progressed from an unsupported red stub to passing
  public f64-oracle/history tests. The width-one shader needed an explicit
  padded-history binding reference; this was caught and corrected before model
  use. Tests cover strided input/kernel/history, exact chunk/step history,
  immutable snapshots, widths 1/4/7, 128-token prefill, 4,194,305-element 2D
  dispatch, and invalid shapes/dtypes/devices. Maximum absolute error was
  0.000000147. Counterbalanced isolated medians: reference 788.275/734.527 us,
  fused 129.325/93.010 us. This is not full-model throughput. The model hook is
  opt-in via `AKH_VULKAN_QWEN35_FUSED_CONV=1`; exact-model parity and resident
  throughput/idle checks are pending. All accepted defaults remain unchanged.
- Convolution full-model numerical parity PASSED (150.46 s): all 33 positions
  had hidden/logit cosine rounding to 1.000000000 and matching greedy IDs.
  Unlike scheduling-only batching, values are not bit-identical. The unchanged
  0.9999 gate passed; nine CPU model/state tests passed again. The production-
  source core throughput/resident-idle run is now in progress with convolution
  opt-in and accepted encoder/RMS defaults. No default promotion yet.
- A subsequent test scaffold retains a `ComputePass` slot and checks that a
  future pass-batching route keeps it open until copy/fence, preserving snapshots.
  There is no pass-batching implementation or enabled flag yet; the slot is
  always empty. The intended red hardware run awaits the convolution idle test
  so GPU tests stay serial and throughput has no concurrent compilation.
- Convolution core counting and resident idle passed: exact text twice,
  5.6317/6.2180 steps/s, TTFT 2.8284/2.2506 s, load 28.9414 s. Final 60-sample
  idle was 0.4333% / 12.2 W. The pre-load 25.333 W baseline retained some prior
  workload heat; the final power also passes the earlier settled 11 W + 20 W
  bound. The small throughput difference from 5.4845/6.1274 is not convincing
  evidence of a material whole-model gain despite the large isolated speedup.
  Convolution stays default-off; no acceptance threshold was relaxed.
- Compute-pass batching progressed red -> green for retained-pass lifecycle and
  the existing immediate encoder-failure/recovery tests. It is opt-in only via
  `AKH_VULKAN_BATCH_PASS=1`, requires shared encoders, and preserves each shader
  and dispatch order. Copies and fences end the pass inside validation scopes;
  failures poison the device, including cleanup errors. The incompatible
  pass-on/encoder-off configuration is explicitly rejected.
  Public tensor/packed-format, concurrent caller, RMS (both routes), convolution,
  alias/zero/error-recovery tests passed. Counterbalanced 256-operation medians
  were 4.772/4.922 ms reference versus 3.578/2.434 ms candidate. A separate
  deferred-invalid-dispatch test and strict bitwise full-model test are prepared;
  their builds are pending. No model throughput, idle or default approval yet.
- All three retained-pass private hardware tests passed (0.28 s), including
  invalid dispatch detected only at pass end. Both immediate and deferred
  failures prevent unfinished output reads, and fresh devices recover. The
  strict 33-position full-model bitwise comparison is running with other
  experimental arithmetic routes off. Combined akhrtd is rebuilding locally
  so subsequent canaries can actually exercise the new flag; no install/restart.
- Retained compute passes passed strict full-model parity (162.16 s): all 33
  complete hidden/logit vectors were bit-identical and greedy IDs unchanged.
  Convolution and block GEMV were off for that isolated comparison. The next
  numerical run combines retained passes with the existing block-GEMV candidate;
  its earlier kernel speedups have not yet established a model-level benefit.
  Throughput waits for the local runtime build to finish; GPU tests remain serial.
- Block GEMV with retained passes also passed all 33 full-model positions
  (152.62 s): cosine rounded to 1.000000000 and all greedy IDs matched; values
  were not bit-identical. The combined runtime rebuild finished successfully
  (5m10s). The staged binary now understands both new opt-in flags; it remains
  uninstalled. No-compilation core throughput/idle comparisons begin next.
- Source-inspection follow-up (not yet a reproduced runtime failure): HIP's
  process-global adaptive free list also receives dropped quantized weights.
  `QCudaStorage::drop -> Hip::free` may retain those allocations while headroom
  exceeds its margin; cache misses do not trim the free list. Model replacement
  or HIP-to-Vulkan switching therefore needs an explicit lifecycle/memory test
  before claiming safe hot switching. The cross-backend numerical test ran
  Vulkan first, HIP last to avoid assuming HIP releases its pool on device drop.
- Retained-pass core counting/idle gate PASSED without concurrent builds:
  6.0900/7.4063 steps/s, model TTFT 2.6843/2.1377 s, load 29.2788 s, exact text.
  Resident idle was 0.3833% / 12.0667 W from a 13.3333 W baseline. Block GEMV
  and convolution were off. The next matched run enables block GEMV on this
  same retained-pass route; the earlier combined numerical gate already passed.
  Pass batching remains opt-in pending the real-daemon canary.
- Retained passes plus block GEMV produced the exact counting response at
  11.0613/13.1479 steps/s, TTFT 2.6657/2.1209 s, load 28.4151 s, with no
  concurrent compilation. This is a substantial gain over 6.0900/7.4063 for
  passes alone, still below native HIP/original runner. Resident idle is pending;
  a real-daemon combined-route canary remains required before default promotion.
- A HIP cache-release API stub and ignored small-allocation lifecycle test are
  prepared (not wired into runtime). The test will record memory before/after
  dropping a temporary, require genuine headroom recovery, preserve a live
  tensor across fresh allocations, require idempotence, and reject capture
  tracking. Its red hardware run waits for the current exclusive GPU tests.
- Combined retained-pass/block-GEMV resident idle PASSED: 0.2667% / 11.75 W
  over 60 samples after settling, from a 10.6667 W baseline. The matching
  isolated four-daemon canary is now running with these flags explicitly on;
  only after that pass may the combination become the source default.
- The retained-pass/block-GEMV four-daemon canary PASSED all authentication,
  native write/read, approval/receipt and session/runtime-restart checks. Its
  disposable state and all four child daemons were cleaned up. Both optimizations
  are now default-on in source, with `AKH_VULKAN_BATCH_PASS=0` and
  `AKH_VULKAN_GEMV_BLOCKS=0` reference escapes. Encoder-off alone implicitly
  disables passes; explicitly requesting pass-on with encoder-off still errors.
  The staged daemon tested this exact route with both flags on, but predates
  these default promotions. No production or shared-service cutover occurred.
- HIP cache lifecycle gate passed after its unsupported red run. Dropping the
  temporary left free VRAM unchanged at 25,149,046,784 bytes; explicit release
  returned 335,544,320 bytes (320 MiB), raising free VRAM to 25,484,591,104 bytes.
  Live tensor contents survived fresh allocations, repeat release returned zero,
  and capture tracking was rejected (0.50 s, fixed 512 MiB test pool). The API
  synchronizes before taking only cached—not live/pinned—allocations. Ambiguous
  driver-free failures are surfaced and those pointers are never recycled.
  `CandleBackend::unload` now consumes the backend and releases this unused HIP
  cache. Native residency replacement calls it under the existing global compute
  lock; ordinary turns and per-call limit/name changes do not release the cache.
  Combined runtime rebuild and a real device-switch integration gate are next.
- Promoted-default Vulkan public tensor/packed-format tests passed again. Both
  encoder-off-alone and pass-off reference escapes preserve ordering; explicitly
  requesting pass-on/encoder-off is rejected. Timings from these regression runs
  are not acceptance measurements because a CPU runtime build was active.
  The isolated HTTP harness now has `--switch-device`: it creates a second
  private profile/session, round-trips devices without restarting akhrtd, verifies
  the original provider continuation, then still exercises the normal runtime
  restart recovery. Syntax/help checks passed; the hardware roundtrip is pending.
- Combined runtime build with cache-release wiring and promoted Vulkan defaults
  passed (4m31s). The isolated HIP -> Vulkan -> HIP roundtrip is running, with
  Vulkan experiment overrides removed so it exercises the actual defaults.
  HIP Q8 is on and rejected recurrence fusion is explicitly off. No live service
  install or shared restart occurred.
- Deployment packaging prerequisite found in source: HIP kernel loaders still
  use compile-time absolute `OUT_DIR` paths for `libcandle_hip_kernels.so`,
  `libcandle_hip_quant.so` and the older `libakh_hip_kernels.so` adapter. A copied
  binary alone is not a self-contained HIP deployment. Before installation or
  build-tree cleanup, add and test a relocatable kernel bundle/loader policy;
  do not assume those development paths survive deployment.
- The default-options HIP -> Vulkan -> HIP HTTP roundtrip passed: authentication,
  real model write with approval before effects, pending-approval session restart,
  read/receipts, exact original provider continuation, and runtime restart with
  artifact reload. All four temporary daemons exited successfully. This validates
  the cache-release integration, not throughput or canonical homelab deployment.
- Relocatable HIP kernel lookup added to both Candle loaders and the older
  offload adapter. Priority is explicit absolute `AKH_HIP_KERNEL_DIR`, executable
  sibling `lib/`, then development `OUT_DIR`. Explicit missing/relative overrides
  fail closed. Four CPU resolver tests passed after an unsupported red run.
  A copied core-test executable loaded both copied Candle libraries from its
  sibling directory (confirmed by openat tracing); Q4_K dequant/matmul parity
  and elementwise broadcast passed. A deliberately missing explicit directory
  failed as expected despite valid adjacent/development libraries. Full runtime
  rebuild/bundle integration remains in progress at this checkpoint.
- Combined relocatable-loader runtime build passed (4m37s). Successful Cargo JSON
  identifies the matching Candle and akh-medu output directories; the copied
  bundle is `/tmp/ptah-native-bundle.bKJjz5`, with `build.jsonl` and `SHA256SUMS`.
  Three helper service binaries are included only for isolated testing. The HIP
  tool/restart canary is running from that bundle with an explicit absolute kernel
  directory. `/proc` mappings confirm the actual daemon loaded both Candle HIP
  libraries from the copied bundle, not the development output paths. Five
  relocated quantization tests and the elementwise test passed separately.
  The example unit/config now agree on an immutable `canary-v1` bundle path;
  these remain templates, not installed service files.
- Relocated runtime canary PASSED from `/tmp/ptah-native-bundle.bKJjz5`: all
  authentication, real write, approval-before-effects, pending-approval session
  restart, read/receipt, exact continuation and runtime restart/artifact reload
  checks succeeded. All four temporary daemons exited; temporary credentials and
  conversation workspace were removed by the harness. Bundle/checksums remain
  local test evidence. No canonical homelab mutation, production install, remote
  restart or additional legacy deletion occurred. The product's tracked goal
  remains `usageLimited`, not complete; this local continuation does not satisfy
  the outstanding state/compression/vision/performance/cutover gates.
  Post-cleanup five samples over ten seconds were each 0% GPU, 11 W, 0 RPM;
  `ptah-inference.service` remained inactive and disabled. This is a cleanup
  observation, separate from the earlier resident-model idle acceptance tests.

### Next local state-ownership slice

Source inspection confirms `generate_completion` always starts a fresh prefill.
The new `InferenceSession` separates shared weights from mixer state, but its
opaque checkpoint is still groundwork, not an installed conversation cache.
Keep durable rendered
history authoritative. Prefix reuse must compare exact token IDs under the same
artifact/template identity, track the number of tokens actually forwarded (the
last sampled token may not have been forwarded), and retain full recurrent plus
attention state—not KV alone. Cancellation or failed append must discard or
restore the partial state. An isolated alternate conversation must never inherit
another conversation's position-dependent state. Validate fresh-vs-resumed logits,
interleaving, divergent prefixes, reload/eviction and cancellation before wiring
this into native runtime calls. Bound resident state before raising physical
context beyond the current 2048-token canary cap. Persistent continuation and
token accounting are not implemented yet; the structural ownership slice has
passed CPU, exact HIP/Vulkan and combined runtime build verification.

### Earlier continuation history (local-only scope above takes precedence)

- User explicitly approved all three shared service restarts (sessions, cognitive
  core, sandbox manager) after tests pass. This supersedes the earlier pending
  deployment decision. Authorization remains scoped to the Ptah cutover.

- `cargo test --lib --no-default-features --features candle-backend
  nlu::candle_backend::tests -- --nocapture`: 13 passed. Explicit device failure,
  context output reservation, and existing EOS family tests included.
- Exact tokenizer built from GGUF, SHA-256
  `e271c182ffa63eebd6d0f60a688fb90d3e87a1141c99c4d87a0d4508badfefea`.
  See [tokenizer evidence](../research/2026-09-14-ptah-native-tokenizer.md).
- `target/release/examples/candle_session_probe` with exact GGUF/tokenizer,
  `--device hip --gpu-sysfs /sys/bus/pci/devices/0000:03:00.0 --tokens 32`
  and ChatML one-sentence hello prompt with closed `<think>` prefix: loaded in
  40.580 s; repeated calls both `Hello!`, 2 tokens, EOS/Stop, prompt 19 tokens,
  1.180/1.009 s including prefill. Ten resident idle samples after 10 s settling:
  mean 1.5% GPU, 14.3 W, baseline 10 W, PASS (<=5%, <=baseline+20 W).
  This is a full exact-model short-turn idle pass, not long-context or graph proof.
- Isolated native Vulkan Q4_K/Q6_K probe passed eight shape/format cases on RADV
  GFX1100. See [Vulkan evidence](../research/2026-09-14-native-vulkan.md).
  Full-model Vulkan acceptance remains pending; explicit Vulkan selection exists,
  but the session-service preflight still rejects it until verified.
- Native session smoke tests are running; no homelab cutover yet. Existing
  `akh-medu/akh-medu-seshd` is a separate knowledge service, not automatically an
  obsolete session daemon. Do not delete it based on its name.
- Legacy boot autostart disabled with `systemctl disable ptah-inference.service`:
  only the multi-user.target wants symlink was removed. Unit, binaries and model
  files preserved; reversible with `systemctl enable ptah-inference.service`.
  Service remains stopped. Old akhomed remains stopped/disabled.
- Exact-model native durable continuation test passed (129.14 s including
  hash verification and cold model load): `cargo test --release -p
  akh-session-services --no-default-features --features hip --lib
  native_model_runs_and_resumes_the_durable_loop -- --ignored --nocapture` with
  the five explicit `AKH_NATIVE_*` artifact/device environment variables.
- Parallel HIP row argreduce hardware parity passed. Matched isolated vocabulary
  argmax median 48.956737 ms before, 0.471861 ms after (15 iterations, resident
  248320-float row plus four-byte readback). This is a sampling-stage result,
  not whole-model speedup. Native build must be rebuilt and remeasured.
- First-class Candle Vulkan tensor/storage integration now compiles alongside
  HIP; public tensor parity plus large dispatch-tail tests passed. Full-model
  Vulkan probe is building. Native session Vulkan preflight remains disabled
  until model-level verification succeeds.
- Canonical homelab configuration is Flux-managed and shared daemons have no
  live reload. User approval requested asynchronously for shared-service restarts;
  no such restarts or cluster changes have been performed. Local development
  and isolated verification continue while that deployment decision is pending.
- The expanded exact-model durable test also passed its third, tool-using turn:
  the model emitted a real file-read call, the broker callback ran once, its
  receipt was persisted, and the final answer included the returned contents.
  Started-generation cancellation and capacity recovery also passed. Combined
  serial artifact tests: 2 passed in 258.61 s, including model reloading when the
  configured token budget changed. Cancellation quarantined the interrupted
  conversation, explicit reconciliation retired it, and a fresh request succeeded.
- First full-model Vulkan attempt failed in GGUF's shared GGML tensor loader,
  before generation. That loader is now fixed and actual GGUF roundtrip fixtures
  pass for F32/F16/BF16/Q4_K/Q6_K; rebuilding for the exact-model retry.
- The original installed Ptah llama-server (not Ollama's generic runner) loaded
  the exact model successfully in 13.275 s with `--load-mode dio`, context 2048,
  flash attention on, Q4_0 K/V, all GPU layers, one slot, and speculation disabled.
  Identical 19-token hello prompt returned `Hello!`; a 36-token counting prompt
  produced 128 tokens at a first-run reported 30.815 decode tokens/s. Matched
  warm repetitions and native timings are pending; this is not the prior
  100K-context DFlash production configuration. This temporary reference process
  does not re-enable its service. After generation, sampled idle was 99%/129 W.
- Exact-model Vulkan retry passed: load 34.305 s; both 19-token hello calls
  returned `Hello!`/Stop with two text tokens. First-token times were 3.610/2.941 s;
  subsequent two sampled positions (including EOS) took 1.168/0.610 s. Resident
  idle averaged 2.9%/18.5 W. Baseline 55.7 W was contaminated by the preceding
  reference runner's cooldown, so repeat from a settled baseline before relying
  on the relative-power assertion. Short-turn numerical correctness is established,
  not throughput parity or a whole-model logit cosine gate. Session-service
  Vulkan feature is now available for explicit opt-in tests, not selected for
  the deployment canary. Native Vulkan performance work continues.
- Bounded Qwen35 prefill now uses 128-query chunks and one cached rectangular
  mask. CPU dense/chunk parity and cancellation/checkpoint tests passed; GPU
  chunk parity remains pending. Native generation checks cancellation between
  chunks. Live KV remains f32 and growing; this is not live KV compression.
- Exact original-runner warm counting trials gave 29.558/31.059/30.731 decode
  tokens/s (median 30.731), with 36 prompt tokens and 128 sampled tokens. The
  temporary reference process was stopped cleanly. Executable tokenizer parity
  also passed all 15 multilingual/Unicode/tool probes with full ID-vector equality.
- Matched native HIP counting probe (36 prompt/128 output, same exact output as
  original runner): primitive recurrence 19.777/19.710 decode steps/s, first-token
  1.029/0.831 s, resident idle 3.2%/19.0 W from a settled 10.33 W baseline.
  Opt-in fused recurrence produced 20.231/20.425 steps/s, first-token 0.942/0.771 s,
  resident idle 3.3%/20.2 W from 10.67 W baseline. Compiler processes were active;
  these are preliminary comparisons, not counterbalanced final acceptance.
  Both idle gates pass; throughput remains materially below the 30.731 reference.
- Fused HIP recurrence hardware microgate passed (3.05 s): scalar CPU and HIP
  primitive references, 128 recurrent steps, odd widths, exact 48x128x128 state,
  immutable snapshots and invalid input checks. Isolated exact-shape medians
  0.216792 ms primitive / 0.096959 ms fused. Full-model 33-position cosine/token
  test is prepared but not yet run; the fusion remains opt-in.
- Vulkan cooperative contiguous-row sum and four-token packed-prefill reuse
  passed public hardware tests, including signed/squared/offset/strided sums
  and Q4_K/Q6_K prefill batch tails 1/2/3/4/5/19. Full-model rerun is pending.
- Resident weight identity now excludes per-call limits/logical model name;
  changing them no longer reloads identical pinned weights. Context bounds are
  applied on every cache hit. Full durable conversation pins remain unchanged.
  Pure native tests: 14 passed, 2 artifact tests ignored.
- New `scripts/session-platform/candle-native-smoke.py` runs private loopback
  sessions/runtime/sandbox/executor daemons with real pinned native artifacts,
  approval-before-write, durable receipts, and session/runtime restarts. No shared
  service is affected by this harness. The initial HIP full HTTP run PASSED:
  authentication, real model write, approval before effects, session restart with
  pending approval, real read and durable receipts, exact provider continuation,
  runtime restart and artifact reload. All four temporary daemons stopped and
  the disposable fixture/state directory was removed by the harness.
- Deployment maintenance approval is granted, but Archibald SSH verification
  now reports a changed host key. Presented ED25519 fingerprint:
  `SHA256:Kz+VNqfgg87yGwpzaPuCrtqFOp0BobZALyQFargG04E`.
  The existing trusted entry is at `/home/toasty/.ssh/known_hosts:167`. No entry
  was removed or bypassed; user verification through a trusted route requested.
  GitHub main and the staged talos clone still both point to `0247a99ad0150d7a0d2ee68dedfadd21d17000ad`.
  All four generator tests passed again. No cluster mutations or pushes yet.
- Astra workers reached a usage limit; their local changes are retained. The
  persistent goal reports `usageLimited`, not complete. Remaining major gates:
  near-reference throughput, full-model Vulkan numerical/performance parity,
  live KV compression/long-context support, canonical canary/cutover and scoped
  legacy cleanup. Passing a short reply or isolated stack does not close them.
- Cooperative Vulkan sum follow-up (release microbenchmark): a 5120-wide row
  took 105.0 microseconds, 19 rows 105.3 microseconds, versus the earlier
  approximately 1054/828 microseconds serial baseline. Shape/stride correctness
  passed; this is an isolated normalization-stage result, not model throughput.
- Fused recurrence full-model parity FAILED (105.80 s). At forced decode step 3,
  hidden cosine was 0.999899431, below the unchanged 0.9999 gate; logit cosine
  0.999980445, greedy token 369. Earlier steps printed matching greedy IDs;
  execution stopped at the first failing assertion, so later positions are not
  validated. Do not enable `AKH_HIP_QWEN35_FUSED_DELTA=1` in a deployment. Scalar
  micro-parity and matching counting text are insufficient to override this gate.
  The ordinary HIP route remains the verified HTTP/idle canary path.
- Final checkpoint rebuild passed: `cargo build --release -p akh-session-services
  --no-default-features --features hip --bin akhrtd` (4m55s). Binary remains in
  `target/release`, not installed as the live service. No test daemons or reference
  runner remain; sampled host idle after cleanup was 0% GPU / 11 W. All changes
  remain uncommitted locally; no GitOps push, credential provisioning, shared
  restarts or canonical canary has occurred. Await trusted Archibald fingerprint
  verification and resumption of the usage-limited Astra work.
- User authorized the stated Archibald fingerprint. Fresh ED25519 scan matched
  it exactly. Backed up known_hosts to
  `/home/toasty/.ssh/known_hosts.archibald-20260914-verified.bak`, then replaced
  only the four old entries matching `archibald.solstice-ci.org` with the approved
  ED25519 key (hashed hostname retained). Strict host verification now succeeds.
  SSH authentication still fails: no connected authentication agent, configured
  id_ecdsa is absent, and normal configured identities did not authenticate. No
  remote command or restart was performed. Do not bypass authentication or ask
  for private-key/passphrase contents; operator must provide a usable SSH identity.
