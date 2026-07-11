# Phase 51 — GWT Workspace-Emergence Probes: Stage L + Stage S Results

**Date:** 2026-07-11
**Status:** MEASURED — decision recorded
**Spec:** [research note §7](research/2026-07-09-global-workspace-language-models.md) (Stage L logit lens, Stage S single-swap-read-by-many); [Phase 51 plan §2a item 7](plans/2026-07-05-phase51-grug-brain-baby-llm.md)
**Harness:** `crates/grug-train/examples/probe_lens.rs`, `probe_swap.rs`; new `TinyTransformer::{hidden_layers, lens_logits, hidden_from}` hooks (forward path untouched, `hidden_from` round-trip invariant tested); `checkpoint::load_model_only`.
**Checkpoints probed:** `runs/p1a2` (JSON, step 9999), `runs/p2-grug` (Grug format, step 19999), `runs/p3d-registry-d1` (Grug + D1 description signal, step 19999). All 13M, 6 layers, d_model 384. GPU (ROCm), n=128/axis (Stage L), 13 families × K=4 framings = 52 member-runs/layer (Stage S).

## Question

Does a verbalizable, broadcast-like workspace form spontaneously inside a 6-layer organ — or does the tripartite sensory→workspace→motor division have no room at this depth (§8 risk)? The answer decides whether the reasoning-phase architecture needs an *imposed* workspace (High Grug supervised intermediates + optionally a Goyal/Bengio bandwidth-limited slot) or can rely on emergence.

## Stage L — logit lens per layer

top1_gold = lens argmax equals gold token; crossover = smallest layer with top1_final ≥ 0.9.

| layer | p1a2 axis-B | p1a2 axis-A | p2-grug axis-B | p3d-d1 registry | p3d-d1 abstention |
|---|---|---|---|---|---|
| 0 | 0.050 | 0.047 | 0.051 | 0.088 | 0.000 |
| 1 | 0.438 | 0.421 | 0.397 | 0.425 | 0.281 |
| 2 | 0.588 | 0.576 | 0.494 | 0.601 | 0.234 |
| 3 | 0.670 | 0.642 | 0.708 | 0.703 | 0.617 |
| 4 | 0.820 | 0.766 | 0.754 | 0.794 | 0.781 |
| 5 | 0.900 | 0.804 | 0.870 | 0.894 | 0.898 |
| 6 | 0.978 | 0.855 | 0.968 | 0.940 | 0.992 |
| **crossover** | **5** | **6** | **6** | **5** | **5** |

**Finding: no mid-stack verbalization locus.** Every checkpoint, every axis: a smooth monotonic ramp with the crossover at layer 5–6 *of 6*. Nothing resembling J-space's intermediate-layer localization (middle ~40% of a 100-layer model). The output becomes nameable only at the very top of the stack. Sharpest case: the gap-token abstention decision is unreadable (0.28) until layer 3 and crosses 0.9 only at layer 5 — the novelty classification is computed late, not held mid-stack.

Caveats: (a) the logit lens is the `J_ℓ = I` special case — a tuned lens could reveal verbalizable-but-rotated mid-stack directions; this negative is a **lower bound**. (b) The `first_tok` column (1.000 from layer 1 on p1a2/p2-grug) is format-scaffolding-dominated (`{"`/`(`) and carries no signal except on the abstention axis, where completion = gap token.

## Stage S — single-swap-read-by-many (JSON-era proxy)

Mining: 13 real families per run, each K=4 *different tools* sharing one verbatim string argument value (corpus value bank: places + phrase nouns); B substitute drawn from observed values, equal-token-length contract. Patch = splice B-donor hidden rows into the A-sequence at the value's prompt positions at layer ℓ, every decode step, resume via `hidden_from`. flip = generated call's argument becomes B; read_by_many = ALL 4 framings of a family flip.

| layer | p1a2 flip | p1a2 read-by-many | p2-grug flip | p2-grug read-by-many |
|---|---|---|---|---|
| 0 | 0.885 | 0.615 | 0.846 | 0.462 |
| 1 | 0.885 | 0.615 | 0.827 | 0.462 |
| 2 | 0.885 | 0.615 | 0.827 | 0.462 |
| 3 | 0.865 | 0.538 | **0.077** | 0.000 |
| 4 | 0.788 | 0.308 | 0.019 | 0.000 |
| 5 | **0.135** | 0.000 | 0.019 | 0.000 |
| 6 | 0.000 (structural) | 0.000 | 0.000 (structural) | 0.000 |

control_A (unpatched generation emits A): p1a2 0.923, p2-grug 0.769 — the known value-copy-fidelity ceiling; flip rates read against it. Layer 6 zero is architectural (no attention left to propagate a prompt-position patch), not evidence.

**Finding: transport-early, decide-late — the source span dies mid-stack.** Patching the value's prompt positions works almost perfectly while the value is still being *read from* those positions (layers 0–3 JSON / 0–2 Grug), then falls off a cliff (JSON: 4→5; Grug: 2→3). Attention completes its cross-position copy by mid-stack; after that the source positions are never consulted again. One span-patch at layer 0–2 redirects **four different tools' generated calls** in 62%/46% of families — the patch is *general across contexts* — but this proxy runs each framing as a separate forward pass, so it demonstrates patch generality, **not** intra-forward write-once-read-many broadcast. The true France→China test (multiple readers in ONE forward) needs High Grug multi-question prompts — P3+ as the spec anticipated.

## Combined verdict and decision

The 6-layer organ is a **feed-forward pipeline, not a workspace**: information transported off the prompt early (S), decisions crystallizing only at top-of-stack (L), no persistent, revisitable middle register. This is the §8 prediction confirmed at our scale — the emergence probe is **negative at 6 layers**.

Consequences (research note §8, now empirically armed):
1. **Engineering beats hoping — confirmed.** The reasoning phase cannot rely on spontaneous workspace formation. High Grug + supervised intermediate states are the *imposition* of a workspace the depth cannot grow.
2. **Goyal/Bengio bandwidth-limited slot: promoted from back-pocket to live design candidate** for the P3-reasoning architecture (behind the existing `Attention<B>` seam), per the plan's own trigger ("if the emergence probe is negative at 8 layers, the fallback fires"). Whether to adopt it is P3-design grilling material, now with data.
3. **The VSA loop is the workspace (§8.3) — now the default architectural stance for P6.** A feed-forward 6-layer organ cannot iterate internally; the Phase-41 thinking engine supplies the recurrent broadcast cycle over the shared SymbolId register. The organ proposes; System-2 iterates.

§7.4 emergence-over-training metrics (crossover index per checkpoint) fall out of `probe_lens.json`; running the probe per training checkpoint is cheap and should ride along with the first P3-reasoning run to test whether intermediate supervision *moves the crossover down the stack* — that is the measurable signature of a successfully imposed workspace.

## Harness notes

- Stage S mining originally paired each shared value A with a single fixed B candidate; the equal-token-length contract then rejected every member on the real corpora (zero families). Fixed: search B candidates in ranked order (bounded 32 attempts). The Grug-format cliff at layer 3 vs JSON's at layer 5 is confounded by training duration (p2-grug 20k steps vs p1a2 10k) — do not read it as a format effect without a matched-steps rerun.
- Outputs: `runs/{p1a2,p2-grug,p3d-registry-d1}/probe_lens.json`, `runs/{p1a2,p2-grug}/probe_swap.json`.
