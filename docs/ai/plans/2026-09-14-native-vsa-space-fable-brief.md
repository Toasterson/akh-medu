# Compact neural-span indexing: architectural decision

Review this source-grounded next-feature question in at most 600 words. Tools,
code execution, remote operations and repository access are disabled. Advice is
not retrieval-quality evidence.

Same-pass Qwen tap primitive now passes CPU and exact HIP/Vulkan gates: selected
absolute span, at most five post-block residual means, all pooling on GPU. Five
target indices are 5/19/33/47/61, each width 5120. No tap state escapes on
cancellation; cached spans explicitly return capture misses. Prompt and 33
subsequent hidden/logit steps are bit-identical to untapped runs. Runtime
continuation and quiet resident idle pass. Next needs compact projection and
real query-to-memory retrieval, then accepted-turn archival.

Important existing-source facts:
- NeuralVsaBridge is a dense learned linear projection + binarization to
  HyperVec (default Dimension(10000)). It initializes randomly but its own docs
  explicitly say concept identification is NOT useful until training aligns it
  with grounded symbolic ItemMemory vectors. At width5120 it is ~205MB.
- Therefore a new random structured projection cannot automatically query the
  existing symbolic index as if the coordinate system matched. Neural-span
  retrieval among spans encoded identically is a different proposition.
- Existing polar_quant module has CPU fwht_normalized and seeded xorshift sign
  flips. It is not a context projector or proof of retrieval quality. No new
  projection or learned weights exist yet.
- akhsessd prepares symbolic context once before durable runtime dispatch via
  the core's /v1/context. Native query signals arrive later, inside akhrtd.
- There is an existing durable completion exchange: akhsessd sends completed
  turns to /v1/turns/accepted; the core fingerprints operation IDs, honors
  transcript-capture policy, journals and persists. It currently carries no
  vectors and does not invoke ContextManager::commit. Protocol structs deny
  unknown fields; capability/version handling is necessary for any new fields.
- No second Engine may be created in akhrtd. Shared core changes may be developed
  and tested in isolation, but no production/shared restarts are in current scope.

Recommend a small vertical slice for a dedicated neural-span index versus a
trained low-rank adapter to existing symbolic vectors. Include identity rules
(model/tokenizer/projector/pooling/layers, tenant/session scope), minimal quality
gate, and how to avoid a third unrelated memory system. Is a structured random
projection a reasonable *evaluated baseline* only for same-space span retrieval?
What must be recorded before we can honestly call its results VSA hits?

Prefer a concrete minimal seam using the existing core's durable accepted-turn
path and bounded read retrieval; do not assume it is already wired. Distinguish
new code that is actually necessary now from optional future generalization.
