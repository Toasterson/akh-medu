# Gemma 4 adapter readiness — 2026-09-21

**Deployed weights unchanged; curriculum training and deployment remain blocked.** The full exact-parent synthetic optimizer/save-reload smoke now passes. The initial section records a CPU-only baseline audit; the later isolated mechanics preparation supersedes its missing-package/download observations. [preflight.json](preflight.json) records the actual interpreter, model content hash, corpus integrity, and source hashes. Exit code 2 means unmet readiness gates; malformed input exits 1. The four preflight tests and four legacy trainer safety tests pass.

## What is available

The local 7,121,860,000-byte Gemma GGUF is the existing Vulkan inference artifact. The inspected Python interpreter has no discoverable torch, transformers, peft, safetensors, or datasets packages. This is an interpreter-scoped check, not a claim that no other environment exists. The previous apprenticeship candidate export has one example, is integrity-pinned, and explicitly disallows training. Its excluded bounded-range family stays excluded. At baseline-audit time no packages or weights were installed and no GPU workload ran; the subsequent authorized isolated setup and controls below changed that state without changing deployed services/configuration.

The repository's quantized Gemma loader constructs `ModelWeights::from_gguf`; no adapter application was found in that deployed path. `LoraRegistry` loads bytes and metadata, which does not prove tensor application or changed logits. Both a native adapter path and a merged/requantized model route therefore need explicit round-trip certification.

The subsequently expanded reference curriculum contains **four candidate examples and four correction examples**, with 143,121 executable reference checks. Its three development holdout families remain separate. [expanded-corpus.json](expanded-corpus.json) pins the corpus files and counts independently of the earlier preflight format. These are assistant-authored references, not observed specialist teaching or diagnosed model failures; the expanded data remains insufficient for a certified quality training run.

## Legacy trainer defect corrected

The llama.cpp fallback previously copied opaque `.bin` bytes into a `.safetensors` filename. It now fails before launching training, and the conversion helper refuses to overwrite any destination. There is no format-compatible conversion implementation to preserve. MLX's actual training function remains available.

The quick evaluation previously returned a numeric zero without model inference. Unknown accuracy now remains `None`; an unevaluated trained checkpoint gets `candidate-meta.json` with deployment disabled, rather than registry-loadable `meta.json`. Before replacing the deployed checkpoint bytes, previous registry metadata is renamed to `previous-meta-*.json` and the containing directory is fsynced. A crash can leave the old checkpoint uncertified, but cannot certify new weights using old metadata. A fault-injection test reproduced the original crash hole and now passes; its failing run is preserved in `checkpoint-boundary-before.log`. This assumes the existing single training writer, not concurrent trainer/promotion processes. Training completion is still reported, explicitly without an evaluation claim. A real inference evaluator must produce measured accuracy before registry metadata is written. These changes do not certify adapter tensor compatibility.

## Feasible next experiment

1. Use the hash-pinned official parent candidate identified below, verify acquired weight bytes against its recorded SHA256, and prove conversion lineage or explicitly certify a new derived deployment artifact. Tokenizer/template/major geometry are already linked; the conversion source revision/recipe remains unproved.
2. Create an isolated, version-pinned ROCm/PyTorch/Transformers/PEFT environment, then test the GPU numerics independently. The historical HIP inference bug does not establish that PyTorch training fails, and successful Vulkan inference does not establish that Vulkan training works.
3. Run one tiny synthetic forward/backward/optimizer/save/reload probe on the exact architecture and selected adapter targets. Record finite gradients, changed adapter tensors, unchanged frozen tensors, identical pre-save/post-reload logits within declared tolerance, and optimizer restoration. This is mechanics evidence, not quality evidence.
4. Prove either actual adapter application in Akh or a merged/requantized export round trip in the deployed Vulkan backend. Pin all artifacts and compare fixed-prompt logits/output before any real curriculum run. Measure memory first: fitting inference weights does not prove optimizer/activation capacity.
5. Only after those pass, train admitted diagnosed weaknesses with retention replay and independent task-family holdouts. The current tiny export and three-arm all-pass transfer result do not justify a quality-improvement claim.

The primary documentation describes Gemma 4 and LoRA routes, but does **not** certify this specific local combination. Google lists approximate 12B BF16 inference memory at 26.7 GB and warns that training adds substantial overhead; the machine's 24 GiB GPU makes unquantized training feasibility especially uncertain. Quantized training/offload support and conversion fidelity require measurement. [Google model overview](https://ai.google.dev/gemma/docs/core), [Gemma tuning](https://ai.google.dev/gemma/docs/tune).

Transformers exposes Gemma 4 model classes; PEFT documents adapter configuration and initialization. These support considering an HF-based experiment, not promising that every Gemma 4 module or AMD quantization backend is compatible. [Transformers Gemma 4](https://huggingface.co/docs/transformers/model_doc/gemma4), [PEFT LoRA](https://huggingface.co/docs/peft/main/package_reference/lora).

## Reproduce the audit

```bash
python3 scripts/session-platform/lilith_adapter_preflight.py \
  --manifest training/lilith/apprenticeship-20260921/candidate-manifest.json \
  --corpus training/lilith/apprenticeship-20260921/candidate-train.jsonl \
  --hash-model
# Expected exit: 2 (blocked, with evidence), not success.
python3 -m unittest discover -s scripts/session-platform -p 'lilith_adapter*_test.py'
python3 -m unittest discover -s scripts -p 'test_train_lora_safety.py'
```

No speculative training command is supplied as runnable: the parent, environment lock, quantization implementation, module mapping, and export path remain unresolved. The bounded next action is the pinned-environment mechanics probe above, not a production training run.

## Lineage follow-up: concrete parent candidate identified

Metadata-only inspection and public Hugging Face APIs resolved considerably more provenance without downloading weights. [lineage.json](lineage.json) records assertions and comparisons; raw API snapshots and small configs accompany it.

- The local GGUF's SHA256 **exactly matches** `gemma-4-12b-it-Q4_K_M.gguf` in [Unsloth revision 3f09de26549e6d7ea54f1b83755149f840fcd333](https://huggingface.co/unsloth/gemma-4-12b-it-GGUF/tree/3f09de26549e6d7ea54f1b83755149f840fcd333). The present latest GGUF has different bytes, so using a floating `main` reference would change this artifact.
- The GGUF explicitly names Google's base repository. Candidate [Google revision 5926caa4ec0cac5cbfadaf4077420520de1d5205](https://huggingface.co/google/gemma-4-12B-it/tree/5926caa4ec0cac5cbfadaf4077420520de1d5205) contains `model.safetensors`, **23,919,549,408 bytes**, SHA256 `5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d`. This revision corrects the initial release's context configuration to 262144; weights are unchanged between inspected initial/current revisions.
- Local tokenizer bytes match the official parent tokenizer SHA256 exactly. The GGUF's embedded chat template exactly matches that revision's `chat_template.jinja` SHA256. Layers (48), hidden width (3840), FFN width (15360), attention heads (16), context (262144), sliding window (1024), and final logit softcap (30) all match the parent configuration.
- Public API reports `gated=false`, `private=false`, `disabled=false`; metadata/config access succeeded without credentials. No credential inspection occurred. The later `parent-download.json` records the completed hash-verified download to a separate private cache.

This resolves **parent candidate identification and local acquisition**, not quantizer source-commit proof or successful training. The exact conversion recipe/commit is absent from the inspected GGUF. Next work can target a specific, hash-pinned 23.9 GB parent instead of searching by model name; training memory, numerical stability, adapter mapping, and export/Vulkan parity still need the bounded mechanics experiment.

## Isolated mechanics preparation (later follow-up)

The exact parent and seven necessary artifacts are now downloaded in a separate private project cache and hash-verified: [download receipt](parent-download.json). A truncated first connection stopped before installation, preserving a 10.95 GB partial file; validated HTTP Range resume completed it without restarting or replacing the deployed GGUF.

An isolated Python 3.10 environment now contains PyTorch 2.7.1+ROCm6.3, Transformers 5.10.1, PEFT 0.18.1 and pinned dependencies ([environment lock](mechanics-environment.lock)). The original default-interpreter preflight remains historical evidence, not the current isolated environment's inventory.

The exact architecture/adapter target check passes on meta tensors ([architecture check](architecture-check.json)). The independent gfx1100 BF16 forward/backward control passes against exact expected matrices ([ROCm control](rocm-control.json)). These checks alone do not prove full-model training. See [mechanics protocol](MECHANICS.md) and the actual run receipt for the full-model result; no inference-service model configuration was changed.

## Full exact-parent mechanics result: passed

[Run receipt](mechanics-run1/result.json) · [Current readiness](current-readiness.json) · [Executed source snapshot](mechanics-run1/probe-source.py).

The full BF16 parent fit on the local gfx1100 GPU without offload. One synthetic 16-token step trained 24,064 rank-2 parameters in the last attention query projection. Loss and gradients were finite; the initially zero B matrix changed, while A's first-step gradient was zero as expected for zero-initialized LoRA B. All frozen base tensor mutation versions stayed unchanged, and the optimizer contained only adapter parameters.

The scratch adapter produced a measurable forward effect (maximum logit difference **0.25** versus zero adapter). After destroying the in-memory adapter, safetensors reload restored its tensors exactly; optimizer groups/state also reloaded exactly. Evaluation-mode logits matched exactly after reload (maximum error **0.0**). Peak allocated GPU memory was **24,060,346,880 bytes (22.41 GiB)**; the whole probe took **183.0 seconds**, including consumed-artifact rehash and full model loading. The process exited successfully and released GPU allocations.

This closes a bounded **training mechanics** uncertainty. It does not establish improved coding, safe curriculum admission, longer-context/broader-target memory capacity, cold-process reload, or adapter application on the deployed quantized Vulkan runtime. Only the scratch adapter changed; deployed Gemma weights/configuration remain unchanged. The new isolated environment and verified private parent cache are retained for the next bounded experiment; no training service remains running.

Follow-up: the [bounded KLPO experiment](../rl-20260921/README.md) now validates actual SFT/RL updates and exact fresh-process reload of the saved KLPO scratch adapter. It establishes no coding-quality gain and leaves Vulkan export/application unvalidated. The earlier mechanics receipt above remains historical.
