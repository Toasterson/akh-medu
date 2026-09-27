# Independent provenance audit

Read-only CPU audit performed after both Vulkan probes. No GPU work, export,
probe rerun, receipt modification, or protocol change was performed.

The source adapter SHA-256 matches the passed KLPO experiment receipt. The
exporter enforces the pinned training parent, candidate file hash, supported
PEFT configuration, exact last-query A/B tensor set, rank-2 geometry, finite
floating values, and scale `alpha / rank = 1`. The exported F32 tensors reload
exactly. The exporter does not merge or replace deployed weights.

The Vulkan probe independently hashes the actual GGUF, tokenizer, exported
adapter, and manifest before loading. It checks zero-B identity, a nonzero
learned-logit effect, clearing/rollback, and reloading the adapter. Both external
probe process receipts record exit code zero. Their base and adapted
full-vocabulary F32 hashes and their 32-token greedy sequences match exactly.

Every hash frozen in `probe-protocol.json` matched the inspected artifact:

- probe binary;
- probe driver;
- Rust example source;
- Gemma Vulkan runtime source;
- exported manifest; and
- source experiment receipt.

This supports mechanical export, application, rollback, reload, and repeatable
fresh-process results for the pinned prompt and artifacts. It does not establish
quality transfer from the BF16 parent to the quantized GGUF, throughput impact,
automatic daemon activation, deployment readiness, or promotion eligibility.

The process receipts do not record PIDs or output-file hashes. Separate
subprocess invocations, durable exit receipts, the frozen driver, and matching
probe results support the fresh-process claim, but those omissions remain a
minor provenance limitation.
