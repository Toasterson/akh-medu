# Tiny LLMs, Huge Context — Frontier Attention/Context Managers — akh-medu

- **Date:** 2026-07-05
- **Status:** Imported research (external). Verbatim substance of a Gemini research thread, reorganized. Claims are Gemini's with the citations it supplied — **not** independently verified here. Companion to the [context-management research notes](2026-06-18-context-management-research.md) and the [Project Grug-Brain report](2026-07-05-project-grug-brain.md).
- **Source:** Gemini share thread "Tiny LLMs, Huge Context Research" — https://gemini.google.com/share/d/1Wzgx0oTwtqAH6Fhcxg01shbeuCM-_Hts (commissioned by Till). Imported into the corpus per the docs-maintenance flow.
- **Scope:** How do you give a sub-billion-parameter model a huge context window without blowing the memory budget the model exists to respect? Survey of "attention managers" and memory abstractions.
- **Relevance to akh-medu:** Feeds Phase 44 (context assembly runtime), Phase 26 (LLM-VSA / Candle inference), and the BabyLLM / Project Grug-Brain direction — these are the three inference-time context managers we'd hand-implement in Rust/Burn for a from-scratch model.

---

## 0. The paradox

Tiny Language Models (TLMs) target edge hardware — tens to a few hundred million parameters, minimal memory and compute (Lamaakal et al., 2025). But standard transformer attention scales **quadratically**: compute and memory are O(n²) in sequence length n (Fountas et al., 2025). Huge contexts demand huge Key-Value (KV) caches, which max out exactly the hardware a tiny model was meant to fit. This is the "memory wall."

The frontier response is not brute-force width. It is teaching the model **how to look at context** rather than forcing it to look at everything at once — "attention managers" and memory abstractions.

## 1. The four techniques

### 1.1 Asymmetric attention allocation — DuoAttention (Xiao et al., 2024)
Not all attention heads are equal. DuoAttention dynamically separates heads by function:
- **Retrieval heads** keep a *full* KV cache (they do the long-range recall).
- **Streaming heads** compress their cache to a lightweight, constant-length rolling buffer.

Result: large memory/latency cuts with no loss of the model's ability to pull facts from long context. **Requires a lightweight identification training phase** to learn which heads are which (see §2).

### 1.2 Human-inspired episodic memory — EM-LLM (Fountas et al., 2025, ICLR)
Give the model a cognitive memory abstraction instead of a flat attention window. EM-LLM dynamically groups the KV cache into blocks keyed on cognitive **"surprise"** (Bayesian surprise). Context is treated as *episodic memory* rather than fully-active attention, so the model retrieves across **millions of tokens** without crushing the hardware. **Pure inference-time trick — zero fine-tuning** (see §2).

### 1.3 Positional-encoding extrapolation — E2-LLM (Liu et al., 2024, ACL Findings)
Training on long sequences costs exorbitant GPU memory. E2-LLM instead manipulates **Rotary Position Embeddings (RoPE)** through dual augmentation: train on a short window (e.g., 4k tokens) and extrapolate at inference up to **~120k tokens**. The trick is dynamically scaling the RoPE base frequency (usually 10000) so a long prompt "looks like" the short context the model trained on. **Pure inference-time trick.**

### 1.4 Extreme edge compression — EmbBERT-Q (Bravin et al., 2025)
For truly constrained IoT hardware, EmbBERT optimizes the attention layer and feed-forward blocks to operate under a **~2 MB memory budget**, proving functional attention survives at the absolute micro-scale.

> **Thesis:** the future of tiny-LLM-huge-context is not an infinitely wide window; it is making the model's **memory recall infinitely smart**.

## 2. Which techniques need training vs. are pure inference hacks

This distinction (drawn out in the companion Grug-Brain thread) matters for our build order:

| Technique | When applied | Training needed? |
|---|---|---|
| **EM-LLM episodic memory** | Inference | None — add a surprise-based eviction rule to the forward pass; the model doesn't know its cache is being pruned. |
| **RoPE extrapolation** | Inference | None — scale the RoPE frequency at runtime by prompt length. |
| **DuoAttention** | Training + inference | Yes — a short "identification" pass with trainable per-head gate values pushes some heads to 1 (read whole doc) and others to 0 (recent only); then hardcode the retrieval heads. |

So the pure inference hacks (EM-LLM, RoPE scaling) can be layered onto any already-trained tiny model; only DuoAttention needs a dedicated training stage.

## References (as supplied by the source thread)

- Bravin, R., Pavan, M., Shalby, H. H. Y., Pittorino, F., & Roveri, M. (2025). *EmbBERT-Q: Breaking Memory Barriers in Embedded NLP.* arXiv. https://doi.org/10.48550/arxiv.2502.10001 — cited by 3.
- Fountas, Z., Benfeghoul, M. A., Oomerjee, A., Christopoulou, F., Lampouras, G., Bou-Ammar, H., & Wang, J. (2025). *Human-inspired Episodic Memory for Infinite Context LLMs.* ICLR 2025. https://doi.org/10.48550/arxiv.2407.09450 — cited by 95.
- Lamaakal, I., Maleh, Y., El Makkaoui, K., Ouahbi, I., Pławiak, P., Alfarraj, O., Almousa, M., & Abd El-Latif, A. A. (2025). *Tiny Language Models for Automation and Control.* Sensors, 25(5), 1318. https://doi.org/10.3390/s25051318 — cited by 79.
- Liu, J., et al. (2024). *E2-LLM: Efficient and Extreme Length Extension of Large Language Models.* ACL 2024 Findings, 4243–4253. https://doi.org/10.18653/v1/2024.findings-acl.252 — cited by 29.
- Xiao, G., Tang, J., Zuo, J., Guo, J., Yang, S., Tang, H., Fu, Y., & Han, S. (2024). *DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads.* arXiv. https://doi.org/10.48550/arxiv.2410.10819 — cited by 251.

## akh-medu notes (not from the source)

- These three managers are exactly what Project Grug-Brain proposes to hand-code in Burn's forward pass. See [2026-07-05-project-grug-brain.md](2026-07-05-project-grug-brain.md) §2.
- Our Phase 44 VirtualContextManager (MemGPT-style core/recall/archival) is the *message-level* analogue of EM-LLM; EM-LLM operates one level lower, at the KV cache — the "full control / in-process Candle" tier called out in [context-management-research.md](2026-06-18-context-management-research.md) §0.
- Verify before relying: citation counts and the 120k/2MB figures are the source thread's; confirm against the arXiv/ACL originals before they enter any ADR.
