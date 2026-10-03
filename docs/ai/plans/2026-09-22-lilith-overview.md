# Project Lilith — overview

Current CoS/eight-voice execution direction: [October 3 transcript-acceleration plan](2026-10-03-lilith-train-now-reconciliation.md). The dated prototype evidence below remains a historical record. Development training uses the harvested corpus and verified corrections; it does not wait for fourteen fresh recorder days.

Status as of 22 September 2026: local prototype with an operational illumos test
VM, actual training comparisons, and verified Vulkan adapter mechanics. Release
quality and Qwen parity remain unproven.

## Purpose

Lilith is the human interface and coordinator for other Akhs, built on the Gemma 4
track. She should inherit relevant ADAM/Kaine capabilities, handle preliminary
coding, and learn from verified peer lessons through VSA. She executes end-to-end
within assigned scope under existing approval rules. illumos is the main target OS.

## Current state

| Area | Measured state |
|---|---|
| Local runtime | Dedicated stack; Gemma Vulkan works. Earlier warm session probes measured roughly 23–26 tokens/sec. |
| Coordination | 50 coordination tests cover two specialists, repository snapshots, independent execution, corrections, cancellation and stale evidence. Full model-backed delegation → patch → illumos test → Ptah review → truthful Lilith report completed. Backend processes are isolated; the final report took 452 seconds. |
| Peer learning | Verified lesson admission, provenance, revocation, persistent reuse and VSA retrieval exist. No demonstrated VSA quality advantage yet. |
| Training | Actual v2 baseline/SFT/KLPO each pass 4/8: 2/4 train, 2/4 development-family tasks. Useful quality gain was not measured. All three arms retain 4/4 earlier tasks; KLPO cold reload is exact. |
| Current managed run | `train2` intentionally stopped for today at the user’s request. No completed episode, optimizer update or adapter. Last saved follow-up response count: 14,464 tokens. GPU released; Runpod/Astra handoff prepared at [Runpod handoff](2026-09-22-lilith-runpod-handoff.md). |
| Runtime training mechanics | An `akhrtd`-owned worker exercised the inference tool broker. Fixed an asynchronous CPU-cache race and unbounded sliding-cache allocation. Source-pinned controls pass: boundary logits/losses agree exactly, maximum gradient difference 2.9e-11, sampler/replay drift 4.6e-7. The 32,768-token memory check passes. This is a mechanics result, not a new trained adapter; Tecton’s Kimi K3 review is complete; BF16 cross-shape error calibration remains open. |
| Vulkan adapter | Exact export; two fresh Vulkan processes pass zero/effect/reload/rollback controls. Full-vocabulary logits match across processes. No deployed weight replacement or promotion. |
| Reference comparison | Same eight-task deployment pack: Gemma 4/8; pinned Ptah Qwen3.8-27B derivative 6/8. Target not met; no upstream Qwen or release-parity claim. |
| MiMo fallback | Pinned official MiMo V2.6 Distill Qwen 9B BF16 checkpoint scored 6/8 with its exact template, thinking enabled and an 8,192-token budget. Six grammar-admitted answers pass the frozen cases in the OpenIndiana guest; one reproduces the interpreter failure. This is not cap/backend comparable to the 512-token quantized runs and is not a promotion result. |
| Transcripts | MCP export admitted 559 supervised-imitation records from 12 complete historical sessions (359 train, 200 validation); zero rows currently carry external-verifier reward evidence. A frozen 8-train/8-validation Gemma subset exists, with no completed-training or quality-gain claim. |
| illumos | OpenIndiana Hipster 2026.04 VM with SSH, Python 3.9, GCC 14.3 and Rust/Cargo 1.96. Native toolchain and bounded executor checks pass. Full native Akh runtime acceptance remains open. |

## What the failures mean

The v2 reward accepts one enclosing code fence and separates presentation from
semantic checks. Actual invalid-input defects include division by zero for an
invalid alignment and accepting a zero-width field. Grammar violations remain
separate: the alignment solution passes 374 finite cases after an unscored,
syntax-only membership rewrite. Official scores stay unchanged. One HF response
was truncated; the deployment comparison completed all sixteen turns.

A long final-report prompt exposed a Vulkan failure/latency boundary. One mixed
backend episode triggered a GPU reset, and fresh Vulkan-only replays exceeded
180 seconds. A shorter evidence capsule completed with the normal output budget; the complete
new episode also finished, but its 735-token report took 452 seconds.
Backend process isolation and bounded report prompts are workflow mitigations;
the underlying long-prompt behavior remains an open runtime issue.

## Next steps

1. Diagnose and reduce Vulkan prefill/report latency using frozen short/long
   prompts and explicit process/context controls. Keep the candidate experimental
   unless useful transfer appears.
2. Build a new curriculum around validation before arithmetic, explicit boundary
   semantics, and concise complete patches. Keep diagnosed tasks in development
   and author new independent evaluation families.
3. Expand native work to real illumos repository builds and subsystem tasks,
   including SMF, IPS, ZFS, error paths and resource lifetime. Validate full Akh
   installation, delegation, cancellation, persistence and recovery on illumos.
4. Run broader pinned reference comparisons and ADAM/Kaine behavioral retention.
   Require per-family evidence and sufficient repeated observations before any
   match-or-beat claim. VSA benefit and adapter quality transfer remain gates.

## Evidence

- [Five-step run](2026-09-22-lilith-next-five.md)
- [Full project plan](2026-09-21-project-lilith.md)
- [Training evidence](../../../training/lilith/rl-20260922/)
- [Runtime training cache controls](../../../training/lilith/runtime-training-engine-20260922/README.md)
- [Vulkan adapter results](../../../training/lilith/vulkan-adapter-20260922/README.md)
- [Reference comparison](../../../training/lilith/comparison-20260922/README.md)
- [MiMo fallback evaluation](../../../training/lilith/mimo-fallback-20260922/README.md)
- [Transcript admission](../../../training/lilith/transcripts-20260922/README.md)
- [Native VM and commands](../../../training/lilith/native-20260922/README.md)
- [Coordination evidence](../../../training/lilith/coordination-20260922/README.md)
