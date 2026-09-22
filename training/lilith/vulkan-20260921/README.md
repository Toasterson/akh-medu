# Lilith Vulkan development checks — 2026-09-21

Gemma 4 now completes native Vulkan inference on hephaistos's RX 7900 XTX
(RADV gfx1100). This is backend/integration evidence, not a model quality score
or native illumos validation.

## Failures and repairs

The initial model load succeeded but inference failed with `no Vulkan
implementation for rms-norm` (`vulkan.log`). Three small tests reproduced missing
RMSNorm, rotary embedding and softmax paths (`op-tests-before.log`). Gemma's
helpers recognized HIP/CUDA but sent Vulkan tensors to unsupported custom ops.
Explicit Vulkan routing repaired all three (`op-tests-after.log`).

The next full forward failed at `Vulkan unary gelu unsupported`
(`fixed/vulkan.log`). The Vulkan backend now implements Candle's tanh-approximate
GELU in its existing device elementwise shader. All four operation tests pass
against CPU with absolute error below 2e-5; GELU includes strided layout coverage
(`op-tests-all-after.log`).

Gemma's legacy rocBLAS offload is now restricted to CPU-backed models. Native
Vulkan and HIP already own their model weights and must not duplicate them through
that older path. See `failure-ledger.json` for evidence-linked diagnoses. These
failures required runtime repairs, not training data or benchmark-specific tuning.

## Full-model diagnostic

`complete/vulkan.log` completes load, hidden-state capture and two repeated
greedy generations. Both return `Hello there!` (plus the model's empty thought
channel markers), matching HIP (`fixed/hip.log`). Across 3,840 hidden-state
components, HIP/Vulkan cosine is **0.9999707181**, above the predeclared 0.9999
gate. Relative L2 is 0.007698; this is numerical agreement, not bitwise equality.

A second Vulkan process with `AKH_GPU_MATMUL=1` and `AKH_GPU_QUANT=1` also passes
(`complete/vulkan-legacy-flags.log`). Its hidden dump is bitwise identical to the
flags-off Vulkan dump, proving the legacy flag regression is covered. All
pairwise comparisons are in `complete/parity.txt`.

Short decode samples varied from 4.3–7.9 steps/sec in the first Vulkan process
to 16.8–26.3 in the second; HIP was 29.2–29.9. These are seven-step samples, with
shader warmup and concurrent Rust compilation, not controlled throughput results.
Do not infer a stable speed ratio or broad backend parity from this one prompt.

Source and probe hashes are in `complete/provenance.json`; model/tokenizer
identities remain pinned by the startup inventory and project runtime config.
Earlier service hashes describe earlier builds, not the final GELU repair.

## Reproduce the operation checks

```sh
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json \
cargo test --release --locked --lib --features hip,vulkan gemma4_vulkan_ \
  -- --ignored --test-threads=1
```

Use the commands recorded in `complete/*.json` to repeat full-model diagnostics.
Run models sequentially. The isolated four-service review workflow is documented
in [the stack guide](../../../examples/lilith/README.md).

## Stack configuration regression

The first review run stopped before inference: workspace acquisition returned
`executor identity/generation mismatch`. The generated executor service identity
did not match `binding.executor`. A red/green configuration regression test now
covers that invariant (`workspace-config-before.log`, `workspace-config-after.log`;
six preparation tests passed at this stage). The executor keeps the endpoint identity
`akhexecd`; isolation remains in its private credentials, ports and state root.

The corrected stack is `/home/toasty/.local/share/akh-medu/lilith-dev-vulkan-20260921`.
The previous `lilith-dev-20260921` root is preserved, stopped, with the unsuccessful
queued session for inspection. Do not start both roots on their shared port range.
The smoke harness starts/stops only the explicitly selected project's processes.

The second run acquired a workspace but exposed the analogous runtime mismatch:
`binding.runtime=akhrtd` versus `service_id=lilith-akhrtd`. All generated service
identities now match their binding keys; a second red/green regression covers
both Lilith and Ptah runtime bindings (`runtime-config-before.log`,
`runtime-config-after.log`; seven preparation tests pass). The stalled turn was
cancelled before the next run. These configuration defects explain why health
checks alone were insufficient.

## Passing shared-runtime workflow

`review-run3/result.json` records a passing end-to-end development fixture:

- Lilith/Gemma on Vulkan wrote `clamp_u8.py` through a successful brokered
  `write_file` receipt. The independent verifier passed 2,051 integer cases.
- The runtime unloaded Gemma, loaded the pinned local Ptah derivative on HIP,
  and Ptah read both the artifact and verification evidence from a separate
  workspace. Successful `read_file` receipts are retained.
- The runtime unloaded Ptah (returning 21,651,507,468 bytes from the HIP cache),
  reloaded Gemma on Vulkan, and Lilith returned `cedar` from her original session.
  The provider-session binding and runtime PID stayed the same.
- The smoke harness stopped all four of its services. No persistent services
  or VMs were installed or left running.

Measured turn times were 94.9 seconds for creation, 110.1 seconds for the cold
Ptah ready turn, 42.2 seconds for review, and 78.0 seconds for return to Lilith.
These include artifact hashing, cold loading and prompt processing. Batch review
work where useful; this run does not establish interactive switching latency.

The handoff is orchestrated by the fixture, not yet by autonomous Lilith
specialist delegation. This proves a usable local model/tool/session foundation.
VSA lesson transfer, broad ADAM/Kaine capability coverage, Qwen quality parity,
and native illumos validation remain future work.

Ptah's review correctly found no integer-contract defect, but incorrectly grouped
floats with inputs that raise TypeError. The follow-up observation reproduces
float behavior in `review-run3/review-quality-observation.json`. Keep this as a
review-quality finding; passing orchestration is not proof of reviewer accuracy.
