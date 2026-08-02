# Phase 44n — Context-Rot Curve Bench (per-model MECW + positional sensitivity)

- **Status:** Planned
- **Date:** 2026-08-02
- **Anima:** project `019d9844-ce74-7993-a9f0-c953b8f04a4e` (Akh-Medu), WI `019fbf62-03e5-74e1-a1a4-e178708b3410` (ready); Seshat slug `phase44n-context-rot-bench`
- **Lineage:** [ADR 065](../decisions/065-context-assembly-runtime.md) (context assembly runtime), [ADR 070](../decisions/070-selectable-chat-backing.md) (per-akh backing), [44j report "Deferred"](../2026-06-21-phase44j-strategy-ab.md) (context-rot incidence explicitly deferred there), [ADR 064](../decisions/064-context-runtime-lessons.md)
- **Related research:** `docs/ai/research/2026-06-18-context-management-research.md` (cites the same context-rot literature)

## Motivation

Phase 44's strategies (CompactingManager et al.) budget context against the **advertised** window (`ChatBackend::capabilities().context_window`, from Ollama `/api/show` or Candle GGUF metadata). The context-rot literature (Liu et al. TACL 2024 lost-in-the-middle; Chroma 2025 context-rot report) shows the *effective* window is task- and model-dependent and often far smaller, with a U-shaped positional accuracy curve. For the tiny-LLM track (Phase 50/51 candidate models) this gap is largest and matters most.

The 44j A/B harness compares assembly strategies deterministically but says nothing about what the backing model can actually *use*. This bench closes that gap: it measures, **per backing model**, two regression-trackable numbers on our own domain distribution:

1. **`positional_sensitivity(L)`** — how much needle-retrieval accuracy drops in the middle of the context relative to the edges, per context length `L`. `0.0` = no middle blindness, `1.0` = middle fully blind while edges hold.
2. **`effective_length`** — the largest swept `L` at which mean accuracy across all depths stays ≥ threshold (default 0.8). This is the practical MECW: the number CompactingManager should budget against, instead of the advertised window.

These become (a) CI-trackable regression signals when a backing model/version changes, and (b) inputs to per-akh context budget sizing (44n.3).

## Design

### Template

Follow the `bench_thinking` pattern, **not** the cargo-test-only `bench_phase44` pattern, because this bench makes live LLM calls:

- Logic: `src/bench_phase44/rot.rs` (new submodule; keeps the Phase-44 bench code together, and 44n.2 reuses the existing scenario/assembly plumbing in `mod.rs`).
- Entry point: `examples/bench_context_rot.rs` (`cargo run --release --example bench_context_rot`), feature-gating the Candle leg behind `candle-backend` like `examples/candle_bench.rs`.
- Output: runs CSV + summary CSV under `target/bench-results/context-rot/`, summary table to stdout, findings pasted into a dated report `docs/ai/<date>-phase44n-rot-curves.md`.
- Determinism where possible: `StdRng::seed_from_u64` for all haystack/needle generation; temperature 0 (and sampler seed where the backend supports it) for probes. LLM output is inherently the non-deterministic part; repeats (below) handle that.

### Probe seam

Neither `ChatBackend::chat_with_tools` nor the agentic loop is needed — probes are plain prompt→text. Define a small internal trait in `rot.rs`:

```rust
pub trait RotProbe {
    fn label(&self) -> &str;                       // model id shown in reports
    fn context_window(&self) -> Option<usize>;     // advertised, caps the sweep
    fn complete(&mut self, system: &str, user: &str, max_tokens: u32)
        -> miette::Result<ProbeResponse>;          // ProbeResponse { text, prompt_tokens: Option<usize> }
}
```

Impls:
- `OllamaProbe` wrapping `OllamaClient` (`src/agent/ollama.rs`). **Must set `num_ctx`** in options to `L + headroom` per probe — Ollama's default `num_ctx` silently truncates long prompts, which would fake perfect edge recall (the truncated prompt keeps the end). If the existing `generate()` doesn't expose options, add a `generate_with_options` variant rather than changing the existing signature. Capture `prompt_eval_count` from the response as `prompt_tokens`.
- `CandleProbe` wrapping `CandleBackend::generate` (`src/nlu/candle_backend.rs`), `#[cfg(feature = "candle-backend")]`; window from GGUF metadata via the existing capabilities path.

Errors follow the house pattern: a `RotBenchError` enum with `#[derive(Debug, Error, Diagnostic)]`, per-variant `#[diagnostic(code(...), help(...))]`, transparent chaining of `OllamaError`/NLU errors.

### Haystack construction

- **Needle:** one synthetic fact per probe carrying a **seeded nonce answer** (e.g. `"The access code for vault {name} is {6-char alphanumeric}"`). Nonces make grading a reliable normalized-substring check and rule out parametric-knowledge contamination — this fixes the weak `last-word-in-answer` grader from the blog sketch.
- **Filler, two modes** (the gap between them is itself a finding — distractor similarity dominates MECW):
  - `distractor` — same-domain text: verbalized triples from a seeded synthetic KG, reusing the Phase 41 bench generators (`src/thinking/bench.rs` task families) and/or the grammar aggregation verbalizer. This approximates what an akh's assembled context actually looks like.
  - `neutral` — unrelated generic prose (seeded selection from a small built-in corpus).
- **Sizing:** target lengths are in tokens, sized with `HeuristicTokenCounter` (`CandleTokenCounter` when the feature is on), then **corrected post-hoc**: every run row records `prompt_tokens` reported by the backend, and the analysis uses actual tokens for the L axis. A cell whose `prompt_tokens` falls far below target (> 20% short) is marked `invalid` (truncation detector) rather than scored.
- **Prompt shape:** system prompt = "Answer only from the provided text."; user = haystack + `\n\nQuestion: <question>`. Question always last (recency slot), matching how our agentic prompts end.

### Grid

Defaults (env-overridable, example-local knobs per the `candle_bench` precedent — not registered in `src/config.rs` since they're not daemon-facing):

| Knob | Env | Default |
|---|---|---|
| Lengths (tokens) | `AKH_ROT_LENGTHS` | `1024,2048,4096,8192,16384,32768`, capped at `min(advertised_window − 512, 32768)` |
| Depths | `AKH_ROT_DEPTHS` | `0.0,0.1,0.25,0.5,0.75,0.9,1.0` |
| Repeats (fresh seed each) | `AKH_ROT_REPEATS` | `3` |
| Filler modes | `AKH_ROT_FILLER` | `both` (`distractor`\|`neutral`\|`both`) |
| Backend | `AKH_ROT_BACKEND` | `ollama` (`ollama`\|`candle`\|`both`) |
| Model (Ollama) | `AKH_LLM_MODEL` | existing resolution |
| Quick mode | `AKH_BENCH_QUICK` | lengths `1024,4096,8192`, depths `0.0,0.5,1.0`, R=1, `distractor` only (9 probes) |

Full grid = 6 × 7 × 3 × 2 = **252 probes per model**. Mean prompt length ≈ 10.7k tokens → ~2.7M prompt tokens per model per full sweep. On local Ollama with a small model this is tens of minutes; acceptable for an on-demand bench, and quick mode exists for smoke. The runner prints the probe count and estimated token volume before starting, and streams progress per cell.

### Metrics

Per `(model, filler_mode)`:

- `hit_rate(L, d)` — mean over repeats of the nonce-containment grade (invalid cells excluded).
- `positional_sensitivity(L)` — with `edge = {0.0, 0.1, 0.9, 1.0}`, `mid = {0.25, 0.5, 0.75}`:
  `max(0, edge_rate − mid_rate) / max(edge_rate, ε)`.
- `effective_length` — largest `L` (ascending, stop at first failure, matching the monotone-cliff shape) with mean `hit_rate` across all depths ≥ `AKH_ROT_THRESHOLD` (default `0.8`; stricter than the blog's 0.5 because this number sizes production budgets).

Outputs: `runs.csv` (one row per probe: model, backend, filler, L_target, L_actual, depth, repeat, seed, hit, invalid, latency_ms), `summary.csv` (per model×filler: the two headline numbers + per-L sensitivity), stdout markdown table, raw responses to `responses/` for audit of surprising cells.

## Work items

### 44n.1 — Core rot bench (this spec's main deliverable)
`RotProbe` trait + Ollama/Candle impls, haystack builder (both filler modes), grid runner, truncation detector, metrics, CSV/stdout emission, `examples/bench_context_rot.rs`, quick mode. Unit tests for the deterministic parts (haystack sizing, nonce grading, metric math on synthetic grids) — no LLM in `cargo test`.

### 44n.2 — Assembly-aware mode (rescue rate)
Grid-ify the existing 44j `salient_present` check by depth × length: build `EpisodeState`s where the needle turn sits at controlled depth in a filler history, run each strategy's `assemble()` (LLM-free, extends `bench_phase44::mod.rs` plumbing), and record **assembly survival** per (strategy, L, d). Then an end-to-end leg: feed the assembled window to the probe backend and grade. Headline metric: **rescue rate** — hit-rate delta of each strategy vs `PassThroughManager` at lengths above the model's measured `effective_length`. This is the number that justifies (or indicts) CompactingManager per model.

### 44n.3 — Budget wiring (separate, small)
Optional `effective_context` override in `akh.toml` (`src/akh_config.rs`); `run_agentic_turn` sizes `TokenBudget` from `min(capabilities.context_window, effective_context)` when set. Manual per-akh setting fed by measured results; no auto-measurement in the daemon. Registered in the `EnvVarSpec`/config docs.

## Acceptance criteria

- `AKH_BENCH_QUICK=1 cargo run --release --example bench_context_rot` completes against a local Ollama model and prints a grid + both headline numbers.
- Full sweep produces `runs.csv`/`summary.csv`; re-running with the same seeds regenerates identical haystacks (byte-identical prompts).
- Truncation detector demonstrably fires when `num_ctx` is forced low (test knob).
- 44n.2: assembly-survival grid runs in `cargo test --lib` (no LLM), end-to-end leg only via the example.
- Findings written up as `docs/ai/<date>-phase44n-rot-curves.md` with at least two models measured (one Ollama cloud-class, one tiny local), cross-linked here and from CLAUDE.md's phase list.

## Non-goals / deferred

- No `akh bench` CLI verb (ADR 071 tree stays daemon-RPC-shaped; benches remain example binaries).
- No LLM-judge NL-quality scoring (still deferred from 44j; nonce grading is exact and sufficient here).
- No automatic in-daemon MECW measurement or dynamic budget adaptation — 44n.3 is a manual config override; revisit only if per-model numbers prove volatile across versions.
- Multi-needle / multi-hop rot curves (reasoning-over-context degradation, which saturates later than retrieval): interesting, but belongs with the Phase 41 thinking-bench families; noted as a possible 44n.4.
