# Isolated adapter mechanics experiment

This is a single synthetic optimizer step, not curriculum training or deployment. The root agent owns GPU scheduling; installation/download are CPU/I/O-only and must finish before a GPU handover.

Environment: `/home/toasty/.local/share/akh-medu/lilith-training/venv310`, using `/usr/bin/python3` (3.10.12). The generic `uv --python python3` initially selected Python 3.14, for which the selected historical PyTorch wheel does not exist; an unused isolated `venv` directory records that first attempt. No system Python packages were changed.

PyTorch's official historical instructions offer 2.7.1 with ROCm 6.3. AMD's 6.3.4 hardware matrix includes RX 7900 XTX and Ubuntu 22.04 HWE/kernel 6.8, but its production-tested framework entry is older. This exact newer upstream wheel/local system combination therefore requires measured compatibility; installing it is not certification. [PyTorch installation](https://pytorch.org/get-started/previous-versions/), [AMD Linux matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-6.3.4/docs/compatibility/native_linux/native_linux_compatibility.html).

The exact HF architecture is **Gemma4UnifiedForConditionalGeneration**, not the separate Gemma4 family class. Its Transformers 5.10.1 source declares the text layer `q_proj` as an ordinary linear module; the probe targets only `model.language_model.layers.47.self_attn.q_proj`. [Versioned implementation](https://github.com/huggingface/transformers/blob/v5.10.1/src/transformers/models/gemma4_unified/modeling_gemma4_unified.py), [Unified model documentation](https://huggingface.co/docs/transformers/v5.10.0/en/model_doc/gemma4_unified).

The probe uses rank 2, one 16-token synthetic sequence, one AdamW step, no KV cache, and nonreentrant gradient checkpointing. It checks an independent tiny BF16 forward/backward numeric control before loading the parent. Full-model checks require finite loss/gradients, changed adapter tensors, frozen-base mutation versions unchanged, optimizer exclusion of all frozen parameters, exact adapter and optimizer state reload, and matching logits in `eval` mode after destroying/reloading the in-memory adapter. It separately reports whether the adapter changed actual forward logits; low-precision rounding can make this undemonstrated even when tensor changes occurred. It never runs additional steps just to force that observation.

Base invariance evidence uses every frozen tensor's PyTorch mutation version and optimizer membership; it is **not** a full bytewise rehash of all resident GPU weights. Reload validates adapter/optimizer serialization in the same loaded parent, not a cold process restart or the deployed Vulkan backend.

Memory risk: the parent is 23.9 GB on disk, close to the GPU's 24 GiB capacity before runtime overhead. Loading uses local safetensors and low-CPU-memory device mapping. An OOM is a valid mechanics failure to record, not grounds to silently switch models or precision. The parent download uses a private separate cache and validates public LFS SHA256 and file size before rename; it cannot replace the deployed GGUF. Partial downloads are retained and Range-resumable.

After explicit GPU handover from the coordinating agent:

```bash
/home/toasty/.local/share/akh-medu/lilith-training/venv310/bin/python \
  scripts/session-platform/lilith_adapter_mechanics.py \
  --parent /home/toasty/.local/share/akh-medu/lilith-training/parent-5926caa4 \
  --download-receipt training/lilith/adapter-readiness-20260921/parent-download.json \
  --output training/lilith/adapter-readiness-20260921/mechanics-run1
```

The output directory must be new, preserving all prior evidence. `result.json` reports the last phase even on Python-level failure; abrupt process termination may leave a running phase requiring operator interpretation. No success is claimed until an actual result exists.

## Observed result

`mechanics-run1/result.json` passed on the actual full parent in 183.0 seconds. All-GPU peak allocation was 22.41 GiB; no hybrid fallback was needed. Exact adapter/optimizer reload and exact evaluation logits were observed, and the adapter changed logits by up to 0.25. This is a scratch one-step mechanics result, with no coding-quality or deployed-Vulkan compatibility claim.
