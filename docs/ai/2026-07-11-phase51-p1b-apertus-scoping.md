# Phase 51 P1b — Apertus Corpus Scoping Memo

**Date:** 2026-07-11
**Status:** Scoping complete — GO-WITH-CHANGES
**Context:** [Phase 51 plan](plans/2026-07-05-phase51-grug-brain-baby-llm.md) P1b ("Apertus foundation, deferred — blocked on corpus-prep scoping: license/coverage/format/size unknowns"). This memo removes those unknowns. Research performed 2026-07-11 (web, primary sources).

## 1. What the "Apertus corpus" actually is

Apertus (8B/70B, released 2025-09-02 by EPFL/ETH Zürich/CSCS under the `swiss-ai` org) is **not a single downloadable corpus** — it is a documented recipe/pipeline over public Hugging Face datasets, executed in 5 curriculum stages over 15T tokens (~60% English, ~40% non-English, >1,000 languages). Sources: [Apertus-8B-2509 model card](https://huggingface.co/swiss-ai/Apertus-8B-2509), [ETH press release](https://ethz.ch/en/news-and-events/eth-news/news/2025/09/press-release-apertus-a-fully-open-transparent-multilingual-language-model.html), tech report [arXiv:2509.14233](https://arxiv.org/abs/2509.14233).

Named components by stage:
- **English web**: `HuggingFaceFW/fineweb-edu` (Score-2/Score-3), FineWeb-HQ (top-33% quality), `DCLM-Edu`
- **Multilingual foundation**: `HuggingFaceFW/fineweb-2` (1,811 languages), [`epfml/FineWeb2-HQ`](https://github.com/epfml/fineweb2-hq) (top-20 languages, top-10% quality-classifier subset)
- **Code/math**: StarCoderData, StarCoder-Edu, CommonPile/Stack-v2-Edu, FineMath, LLM360 MegaMath
- **Specialty**: Clean Wikipedia, EuroParl/ParaDocs, Institutional Books 1.0 (28.7B tokens public-domain books), and [`swiss-ai/apertus-pretrain-swiss`](https://huggingface.co/datasets/swiss-ai/apertus-pretrain-swiss/viewer) (Curia Vista, Entscheidsuche court decisions, Swiss FineWeb-2 top-10% slice, Romansh sources)

Reconstruction tooling is published, not a data mirror: [`swiss-ai/pretrain-data`](https://github.com/swiss-ai/pretrain-data) and [`swiss-ai/data-tooling`](https://github.com/swiss-ai/data-tooling) (datatrove fork).

## 2. License / compliance

- **Weights**: Apache-2.0 — freely trainable-on and redistributable.
- **FineWeb-Edu / FineWeb-2 / FineWeb2-HQ**: ODC-By v1.0 + CommonCrawl ToU. Permits download, train-on, and redistribution of derived models with attribution — FLOSS-compatible.
- **Apertus's extra compliance layer** (retroactive robots.txt re-application across 2013–2024 crawls, removing ~8% EN / ~4% multilingual; regex PII redaction; per-language top-5% toxicity removal; Goldfish anti-memorization objective; live removal-request channel) is Swiss-AI's own pipeline output. Pulling the upstream HF datasets directly gives FineWeb's upstream filtering but **not** this pass, unless `swiss-ai/data-tooling` is re-run. Acceptable at hobby/research scale.
- **Swiss FADP** (GDPR-aligned, opt-out-based): low exposure for research training on already-published ODC-By derivatives; do not redistribute raw scraped text — only trained model/tokenizer artifacts.

## 3. Size / format / recommended subset

- Full corpus: 15T tokens, no single artifact; `fineweb-2` alone ~20TB — 3+ orders of magnitude over budget, confirming the plan's framing.
- Format: Parquet everywhere.
- Ready-made tiers: `fineweb-edu` ships `sample-10BT` (9.67M rows, ~10B GPT-2 tokens, **28.5 GB**), `sample-100BT`, `sample-350BT`. `FineWeb2-HQ` is per-language top-10% quality; `deu_Latn` is 43.1M docs / 618GB — stream + subsample, never bulk-download.
- **Recommended subset (~7–19B tokens, ~35–75GB)**:
  1. `HuggingFaceFW/fineweb-edu` config `sample-10BT` — ~10B tokens, 28.5GB, English, ODC-By.
  2. `epfml/FineWeb2-HQ` config `deu_Latn`, **streamed + fixed-seed reservoir-sampled** to ~1–2B tokens (~4–6GB).
  3. Optional: `swiss-ai/apertus-pretrain-swiss` in full — 6.9B tokens, 38.5GB, Apache-2.0; the one genuinely downloadable Apertus-team artifact.

## 4. Pipeline (simplest viable)

Upstream dedup/quality-scoring already done (FineWeb MinHash + classifiers). Ours: (1) stream/download the three subsets via HF client or direct parquet HTTP; (2) reservoir-sample `deu_Latn` to token budget; (3) optional stricter PII/toxicity pass via `swiss-ai/data-tooling` (not required at this scale); (4) parquet→JSONL/plain-text with shard-level shuffling (`arrow-rs`/`polars` on the Rust side — no datatrove/Slurm machinery needed at 10B tokens); (5) feed shards to the Burn trainer.

## 5. Tokenizer implications

Apertus uses a Mistral-Nemo v3 byte-level BPE, 131,072 vocab (chosen to minimize tokenization-cost Gini across >1,000 languages) — far too large for 13M–100M params. Plan stands: train our own **8k–32k byte-level BPE** (byte-fallback keeps full UTF-8 totality despite small vocab) on the exact EN+DE subset, with a **deliberate EN:DE mixture ratio (~75:25)** so German subword statistics aren't drowned out (German compounding directly affects fertility), then **reserve a contiguous trailing ID block (~1–4k) for High Grug SymbolIds**. Note the P1a-2 finding (confirmed 2026-07-11 from the tokenizer artifact): the procedural corpus saturates BPE at word level (~2953 tokens; 2693 merges, longest tokens are whole words) — vocab headroom only ever comes from corpus diversity, i.e. from this P1b feed.

## 6. Alternatives

Apertus's data *is* a recipe over public sets, so the practical alternative is picking its own components directly (done above). Zero-Apertus-dependency fallback: `fineweb-edu` alone (same component, single artifact, English-only — trades away DE/multilingual). Extreme low end: `roneneldan/TinyStories` (476.6M tokens, 7.62GB) — but synthetic GPT-3.5/4 output, a philosophical mismatch for the sovereignty principle. AI2 Dolma/OLMo: equivalent openness, US/English-centric, also a recipe — no advantage over FineWeb-Edu directly.

## Recommendation: GO-WITH-CHANGES

Use Apertus's **named component datasets**, not a nonexistent monolithic corpus: `fineweb-edu:sample-10BT` + reservoir-sampled `FineWeb2-HQ:deu_Latn` (+ optionally `apertus-pretrain-swiss`), ~7–19B tokens / ~35–75GB, all ODC-By or Apache-2.0. Train our own 8–32k BPE over that subset with reserved High-Grug SymbolId block. P1b's "corpus-prep pipeline" blocker is hereby scoped: the pipeline is a weekend of streaming + resharding, not a research project. Remaining open inputs before executing P1b: High Grug v1 vocabulary design (reserved-block sizing) and the GWT probe outcome (whether pretrain must also carry High-Grug fluency text per GWT amendment #4).
