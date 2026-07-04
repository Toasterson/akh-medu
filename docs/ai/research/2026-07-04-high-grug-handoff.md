# High Grug — Handoff / Continuation Brief

- **Date:** 2026-07-04
- **Status:** Handoff. Research complete; **no implementation yet.** Two ready-to-start tasks below.
- **Purpose:** Let a fresh session (on archibald or anywhere) pick up the High Grug work cold, without this conversation's context.
- **Owner:** Till. Authored by Claude (Fable 5).

---

## 0. TL;DR — what to do next

The research is done, verified where it could be, and pushed to `develop`. Two tasks are teed up. **Pick one** (they're independent; A is the recommended first move because it needs no external credits and produces a go/no-go signal fast):

- **Task A — P0 spike (recommended first).** Prototype High Grug as **GBNF-constrained notation over the existing tokenizer** (no training, no minted tokens, no Firecrawl credits). Measure accuracy-per-token against English chain-of-thought and Chain-of-Draft on the existing benchmark harness. Ship a go/no-go against a pre-committed kill criterion. See §4A.
- **Task B — ADR skeleton + re-verification.** Draft the ADR for the designed reasoning language now, then re-verify the load-bearing claims when Firecrawl credits return **2026-07-07**. See §4B.

Everything you need to understand *why* is in the four research docs (§2). Everything you need to *act* is in §4–§7.

---

## 1. What High Grug is

A **designed, dense reasoning language the akh's model is trained to think in** — named "High Grug" by Till. Core bets:

- **Symbols get custom token IDs.** A curated inventory of hand-made symbols (akh-medu `SymbolId` → minted token IDs), streamable directly into context.
- **Dense per token, exact in meaning.** More meaning per token than English prose; the density is *designed and paid for*, not left to emerge.
- **Diglossia — two registers, deliberately.** *High Grug* is the H-variety the model reasons in (dense, formal, exact); a *low gloss* channel renders it for humans and safety monitors. This mirrors the Egyptian hieroglyphic-vs-demotic register split (see the Egyptian rationale doc). *medu-akh* was floated as an alternate name and held in reserve; **High Grug is the official name.**

The founding intuition (Till's, from the start of the symbol-language research): *"a compute-optimized language akin to Egyptian where rules and connections make up meaning."* High Grug is that intuition realized at the token level.

---

## 2. Where everything lives (canonical sources — read these first)

### On disk (all pushed to `origin/develop` 2026-07-04; commits `2fca5c8` scripts, `3803d38` docs)

| Doc | Path | What it gives you |
|---|---|---|
| **High Grug report** | `docs/ai/research/2026-07-04-grug-reasoning-language.md` | The main artifact. Phenomenon evidence, prior-art map, six design laws, vocabulary budget, token mechanics, P0–P4 path, risk table, ~130-source link list. **NOT 3-vote verified** — single-agent sourced (see §5). |
| Two-lineage survey | `docs/ai/research/2026-06-23-symbol-language-state-of-the-art.md` | The verified foundation. Writing-systems history + AI KR; the single survival law; four-axis scorecard; Paths 1–6. |
| Coverage-gap addendum | `docs/ai/research/2026-06-23-symbol-language-coverage-gap-addendum.md` | Schank CD / Sowa CG / UNL-AMR; the two surviving governance disciplines (standardize-against-a-logic + corpus-anchor). |
| Egyptian design rationale | `docs/ai/research/2026-06-23-egyptian-writing-design-rationale.md` | Why the determinative = a type-tag / classifier system; lexical vs referent classifier = static vs microtheory-relative type. |

### Seshat corpus (vector-searchable; append-only — never re-ingest a slug)

| Slug | Doc id | Chunks |
|---|---|---|
| `2026-07-04-grug-reasoning-language` | `e62c3d0a` | 22 |
| `2026-06-23-symbol-language-state-of-the-art` | `ce4f6bab` | 16 |
| `2026-06-23-symbol-language-coverage-gap-addendum` | `57afc43b` | 10 |
| `2026-06-23-egyptian-writing-design-rationale` | `af3c9877` | 13 |

> ⚠️ The Seshat copy of the High Grug report predates the rename and still carries the old title ("Sophisticated Grug"). This is fine — Seshat is append-only, no upsert. The correct name **High Grug** rides along with the next ingest (the ADR, or the re-verified report). Vector search finds it either way ("Grug" is in both titles).

### Memory (auto-loaded each session)

- `project_grug_reasoning_language.md` — the condensed record of everything below.
- `project_firecrawl_lineage_a_research.md` — how the research got here (the lineage-A dropout diagnosis + Firecrawl wiring).

### Reusable tooling

- `scripts/research-deep-firecrawl.workflow.js` — Firecrawl-backed deep-research workflow (scholarly-PDF biased SEARCH + FETCH; verify gate kept at top-25, 3-vote). Use this for the Task B re-verification.
- `scripts/research-firecrawl-fetch.sh` — no-MCP curl fallback for headless runs.

---

## 3. The knowledge, condensed

*(Full treatment + citations in the High Grug report. This is the load-bearing skeleton.)*

**Why now / why it's not crazy.** RL-trained reasoning models already drift into dense private dialects on their own (o3's "disclaim/watchers/marinade"; DeepSeek R1's language-mixing). That drift is **load-bearing** — forcing models to reason only in legible English costs ~36–53% accuracy (arXiv:2510.27338) — and it's **translatable** ("a dense language, not an alien one"). Labs pay a "monitorability tax" either way: let it drift and lose oversight, or tax it back to English and lose performance. **The unclaimed third option: design the dense language deliberately + train a faithful gloss channel.** ORION / "Mentalese" (arXiv:2511.22891) is the first *trained constructed reasoning language* (4–16× fewer tokens, 90–98% accuracy) — a category precedent, unreplicated.

**The novelty (what's unpublished).** Curated symbol inventory + embedding-row-only training + mixed text/symbol streams. Each piece is validated separately in the literature (ToolkenGPT for symbol-as-token with a frozen model; Token Assorted for VQ discrete codes + mixing; Self-RAG for reflection tokens; DSI for the forgetting pitfall) — but nobody has published the combination as one system.

**Six design laws.**
1. **Vocabulary is cheap for models.** The human learning cost that killed the historical logographic/conceptual-language projects (Wilkins, Blissymbolics) does *not* bind an LLM. The logographic path reopens.
2. **Density must be designed and paid for.** Emergent codes are anti-efficient (Chaabouni). You don't get density for free; you engineer it.
3. **Executability is the win condition.** The KR systems that won did so via an executor (PAL, Logic-LM, PoT). The ones that stayed inert lost (AMRCoT *hurt*). ⇒ **High Grug is the surface syntax of the akh-medu KG; the engine is its interpreter.** This is the single most important law.
4. **Determinative → classifier tokens.** Egyptian's 3,000-year-lived move: type a token by an attached classifier relation. High Grug carries classifier tokens — both *lexical* (static type) and *referent* (microtheory-relative type).
5. **Closed relation core + open concept lexicon ⇒ two-tier tokens.** (See below — this is the architecture.)
6. **Dual-channel bilingual (Grug ↔ gloss).** The Phase 33 verbalizer is the decompiler. Fidelity metric = "same induced belief" (from the Neuralese-translation literature): the gloss is faithful iff it induces in a reader the same beliefs the Grug induces in the model.

**Two-tier token architecture (law 5, expanded).**
- **Tier 1 — minted, discrete, closed core (~300–450 tokens).** NSM semantic primes (~65–100), well-known relations (~100–200), classifiers (~50–100), epistemic/control markers (~20–40), structural (~10). These get real minted token IDs.
- **Tier 2 — open entities via a projector.** An xRAG-style **VSA-hypervector → embedding projector** maps any open-world entity to a single soft token. **The Phase 26 Neural→VSA bridge *is* this projector.** This dodges the DSI++ catastrophic-forgetting trap you'd hit if you minted a discrete token per entity.

**Phased path.**
- **P0** — no-training prototype (Task A below).
- **P1** — mint tier-1 tokens; train **embedding rows only** (PEFT `trainable_token_indices`; Phase 27/35 infra).
- **P2** — tier-2 projector (Phase 26 bridge).
- **P3** — bilingual Grug↔gloss via RL (SLPO-style), with the same-induced-belief fidelity reward.
- **P4** — inter-akh High Grug over ACP. **This is the strategic payoff:** discrete symbols are the *only* dense channel that works across heterogeneous akhs — KV/activation channels are model-locked, and the fleet is heterogeneous (Tecton=GLM, Thoth=cloud, Gemma=candle).

**GGUF gotcha (bites at P1).** Minted tier-1 tokens must be declared `token_type = CONTROL` with `parse_special = true`, or the tokenizer splits them back into pieces. There's a known Unsloth Gemma-3 breakage precedent — budget time for it.

---

## 4. The two open tasks, in detail

### Task A — P0 spike (no training, no credits) — RECOMMENDED FIRST

**Goal:** Answer one question with data — *does a dense symbolic reasoning notation beat English CoT on accuracy-per-token, before we spend a single training dollar?*

**Approach:**
1. Define **High Grug v0** as a **notation over the existing BPE tokenizer** — no minted tokens. Use ASCII/Unicode digraphs for the tier-1 core (primes, a starter set of relations, a few classifiers, epistemic markers). The v0 vocabulary sketch is in the High Grug report §4; treat it as a starting point, not gospel.
2. Constrain generation with **GBNF grammar** using the existing **ADR 067** candle grammar-constrained decoding machinery. The grammar enforces well-formed High Grug so you're measuring the *language*, not the model's ability to remember syntax.
3. Run on the **existing benchmark harness** — `src/bench_phase44/` (deterministic A/B harness) and/or the Phase-41 thinking-engine benchmark. Reuse, don't rebuild.
4. **Comparators:** English CoT (accuracy ceiling) and Chain-of-Draft (compression comparator).
5. **Metric:** accuracy per reasoning-token (accuracy ÷ mean reasoning tokens), plus raw accuracy and raw token count.

**Pre-committed kill criterion (set the exact numbers at spike start; here's the honest default):**
> High Grug v0 must land on the **accuracy/token Pareto frontier** against the two baselines — i.e. either **≥ parity accuracy with English CoT at materially fewer tokens (target ≥25% reduction)**, or **higher accuracy at parity tokens**. **KILL / rethink** if, after reasonable grammar+prompt iteration, it is *Pareto-dominated* — lower accuracy than CoT while not beating Chain-of-Draft's compression. A notation that is merely "shorter but dumber to the same degree CoD already is" is not worth minting tokens for.

**Why this is safe to run autonomously:** no training, no minted tokens, no external services, no Firecrawl credits. Pure measurement on existing harnesses. Reversible.

**Deliverable:** a short results doc under `docs/ai/research/` (or a P0 findings note) with the numbers and the go/no-go call, ingested to Seshat.

### Task B — ADR skeleton + re-verification

**Goal:** Turn the research into a decision record, and close the verification gap.

1. **Draft the ADR.** Next ADR number is **081** (verify against `docs/ai/decisions/` — the tree went to 080 on the merge you just pulled). Follow the repo ADR conventions in `CLAUDE.md` (miette/thiserror context isn't needed for a doc; the *documentation-maintenance* section is). The ADR states: the designed-reasoning-language decision, the two-tier token architecture, the P0–P4 path, the executability-is-the-win-condition principle, and the heterogeneous-akh cross-model-channel rationale. Save to `docs/ai/decisions/081-high-grug-reasoning-language.md`; ingest to Seshat (new slug); reference the Anima work item if one is created.
2. **Re-verify the load-bearing claims (needs Firecrawl, ≥ 2026-07-07).** The High Grug report is single-agent sourced. Re-run the verify pass via `scripts/research-deep-firecrawl.workflow.js` scoped to these specific claims:
   - the ~36% vs ~53% legible-CoT accuracy-cost discrepancy (arXiv:2510.27338) — resolve which figure is right and under what condition;
   - CCoT (compressed chain-of-thought) numbers;
   - TokenSkip headline result;
   - Token Assorted codebook internals;
   - ORION/"Mentalese" format details + the 4–16× / 90–98% figures (arXiv:2511.22891);
   - the o1/o3 system-card CoT-drift quote.
   - Confirm the arXiv IDs in the 25xx.xxxxx range resolve (they're valid as of 2026, but a prior agent flagged them from cutoff confusion — verify, don't assume).

---

## 5. Constraints & gotchas (read before you touch anything)

- **Firecrawl credits are exhausted until 2026-07-07.** Do NOT call `mcp__firecrawl__*` tools before then. Task A needs none; Task B's re-verification is blocked until then.
- **Firecrawl API key** lives only at `~/.config/firecrawl/key` (mode 600). Never echo it, never paste it into chat or shell history. The MCP URL embeds the key — always redact `fc-…` patterns in any command output (`sed -E 's#fc-[A-Za-z0-9_-]+#fc-<redacted>#g'`). Firecrawl scrapes from *its* cloud, so every fetched URL leaves the machine (fine for public papers).
- **Seshat is append-only** — `seshat_ingest` errors on a duplicate slug; no upsert, no delete. New slug per doc. Never re-ingest an existing slug.
- **The High Grug report is NOT 3-vote verified** (§4B) — unlike the other three docs in the series. Flag this honestly in the ADR until re-verification lands.
- **`cargo test --features server --lib` risks ENOSPC** — the link step builds a huge binary that can fill the disk. Use `--features postgres` for lib tests; `cargo clean -p akh-medu` frees ~10 GB if you hit it.
- **Global CLAUDE.md** mandates lean-ctx tools (`ctx_read`/`ctx_shell`/`ctx_search`/`ctx_tree`) over native equivalents.

---

## 6. akh-medu integration points (so the design lands in real code)

| Piece of High Grug | Existing akh-medu machinery |
|---|---|
| GBNF-constrained P0 notation | **ADR 067** — candle grammar-constrained (GBNF) tool calling |
| Benchmark harness | `src/bench_phase44/` (deterministic A/B); Phase-41 thinking-engine benchmark |
| Tier-2 entity projector (VSA→embedding) | **Phase 26** — LLM-VSA Deep Integration / Neural→VSA bridge |
| Embedding-row-only training (P1) | **Phase 27** (Burn/LoRA) + **Phase 35** (T5 training infra) |
| Gloss decompiler (dual-channel) | **Phase 33** — Grounded Verbalization (T5 NLG) |
| Streaming symbols into context | **Phase 44** — Context Assembly Runtime (this is a context-manager feature) |
| Inter-akh channel (P4) | ACP; heterogeneous fleet (Tecton=GLM, Thoth=cloud, Gemma=candle) |
| The design critique it answers | `src/agent/agent.rs`, `src/agent/project.rs`, `src/agent/resource.rs` — predicates accrete bottom-up via `resolve_or_create_relation("agent:…")` with no principled primitive set. High Grug's tier-1 closed core is the governance layer that gap is missing. *(Verify these file:line refs against current code — they're from June.)* |

**The synthesized path from the survey** (for the ADR's "considered alternatives"): Path 2 (graph-primary spine) + Path 6 (grounded-predicate governance = standardize-against-a-logic + corpus-anchor + Harnad bottom-up grounding), with Path 3/γ (VSA bridge) as the tier-2 spike, and VSA-primary gated on a capacity benchmark. High Grug realizes all of this at the token level.

---

## 7. How to resume in a new session (first 15 minutes)

On archibald (or wherever), in a fresh Claude Code session:

1. **Get the code.** `git fetch origin develop && git checkout develop && git pull` (or rebase your worktree onto it). The docs and scripts are at `bb5c510..3803d38`.
2. **Read this brief**, then the High Grug report (`docs/ai/research/2026-07-04-grug-reasoning-language.md`).
3. **Pull context from Seshat** if you want the verified foundation without re-reading everything: `seshat_search` for `high grug` / `symbol language survival law` / `two-tier tokens`.
4. **Confirm the ADR number** before writing one: check `docs/ai/decisions/` — next is expected to be **081**.
5. **Check the date** — if it's ≥ 2026-07-07, Firecrawl credits are back and Task B's re-verification is unblocked; if earlier, Task A only.
6. **Pick Task A or Task B** (§4) and go. Follow the `CLAUDE.md` documentation-maintenance flow: new work → doc under `docs/ai/` → ingest to Seshat → reference the Anima work item.

**Open follow-ups not yet commissioned** (surface if relevant, don't auto-start): Egyptian administrative/state-formation + religious *medu-netjer* drivers pass (vs Baines/Assmann/Abydos U-j); UNL-vs-AMR survival mechanism; a VSA capacity benchmark (m/F budget at KG scale); e-graph↔VSA coupling (candidate answer: they couple *in the token stream* — which is exactly what High Grug is).
