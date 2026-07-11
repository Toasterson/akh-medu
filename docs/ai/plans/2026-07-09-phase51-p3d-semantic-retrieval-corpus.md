# Phase 51 P3d — Semantic-Retrieval Corpus: remove the name-only-corpus asterisk

- **Status:** **COMPLETE — P3d-1 + P3d-2 + P3d-3 all built + measured** (2026-07-09, commits `9ad8c13`, `fda1923`, `1c4d6ee`). **P3d-3 verdict: the 0.998 abstention was NOT a name artifact** — D1 retrain (name absent from requests, description-keyed retriever, seed 42, only the signal changed) holds abstention recall at **0.996** (premise 0.992, false-abstain 0.020), and the value-grounding control is corpus-invariant (+0.160 vs +0.145). Run `runs/p3d-registry-d1/`. **Headline: recall@8 = 1.000 was *entirely* surface coupling to the name** — lexical-on-name collapses 1.000 → 0.002 the moment the name leaves the request (D1); a description-keyed retriever fully recovers at D1 (0.99–1.00) but every surface retriever floors at ~0.01–0.02 under true paraphrase (D2). **P3d-2: a stock learned embedder (all-MiniLM-L6-v2) only *partially* bridges D2 — recall 0.210 (~11–17× the surface floor, but far from ~1.0)** → off-the-shelf semantic ≫ surface but is necessary-not-sufficient; genuine paraphrase over a large registry needs a fine-tuned/domain retriever (Phase-28 VSA) or richer descriptions. See [scaling-results §16](../2026-07-05-phase51-p1-scaling-results.md#16).
- **Deciders:** Till, Claude (Opus 4.8).
- **Relates:** extends [ADR 089](../decisions/089-organ-kg-zeroshot-path.md) (organ, not weights) and the P3a results ([scaling-results §14–15](../2026-07-05-phase51-p1-scaling-results.md#14)). Consumes the Phase-28 semantic router / Phase-37 Seshat embedder (`all-MiniLM-L6-v2`) as the learned retriever.
- **Anima:** log under project `019d9844-ce74-7993-a9f0-c953b8f04a4e` ("Akh-Medu"); this file is the backup + Seshat paste source.

## 1. Why — every P3a headline still carries the same asterisk

The three ADR 089 mechanisms are built and measured, but *all* on one synthetic corpus with a decisive property: **the retrieval signal is the tool's own name, embedded verbatim in every request** (`verb_phrase_for("fetch_ledger") → "fetch the ledger"`), and **tool descriptions merely echo the name** (`"...tool \`{name}\` with N parameter(s)."`). Three results rest on that property:

| result | number | the asterisk |
|---|---|---|
| retrieval recall@8 (M1 premise) | **1.000** | trivial — the request literally contains the tool's verb+noun |
| abstention recall (M1 rescues M3) | **0.998** | "exact match present vs only near-misses" is a *surface* cue, not a semantic judgment |
| lexical vs HDC retriever (lever 2) | clean 1.000 / 0.980; typos 0.000 / 0.588 | proved both retrievers key on the verbatim name; genuine paraphrase **untestable here** |

The production retriever is a *learned semantic* router (VSA over `goal ↔ tool`); the production schema read is semantic. So the honest, load-bearing unknown is: **does the organ story survive when retrieval is a genuine semantic problem — paraphrased requests and descriptions whose meaning is not reducible to the name?** P3d builds the corpus that can answer that, and re-measures the stack without the name-token crutch.

This is the same move P1a-2 made (explode the catalog to kill memorization): change the *corpus* so it tests the hard thing, then re-measure.

## 2. The core idea — move the signal off the name, graded by difficulty

Introduce a controlled **gloss lexicon**: each `verb` and `noun` in the procedural vocab gets two synonym glosses (a *primary* and an *alt*) with **no surface-token overlap with the name**. Then:

- **Descriptions become semantic**: built from the *primary* gloss, containing no name tokens. `fetch_ledger` → *"Retrieve the accounting records for a given account."* (retrieve ≉ fetch, records ≉ ledger).
- **Requests become paraphrased**: the phrasing's `verb_phrase` slot is filled from a gloss instead of `verb_phrase_for(name)`.

Three difficulty tiers, selectable per corpus (a `SignalMode` enum), so the experiment is graded, not all-or-nothing:

| tier | request verb-phrase source | description source | what it tests | who can retrieve |
|---|---|---|---|---|
| **D0** (baseline) | name (`verb_phrase_for`) — *current* | name-echo — *current* | control: reproduce recall 1.0 | lexical-on-name |
| **D1** (signal→description) | **primary gloss** | **primary gloss** | can a retriever key on description↔request overlap instead of name? | lexical/HDC-on-**description** |
| **D2** (true paraphrase) | **alt gloss** | **primary gloss** | request & description share *no* surface tokens — only meaning | **learned embedding only** |

D0→D1 moves the signal from the name to the description (still surface, but now the retriever must read the *description*, as the real system does). D1→D2 removes surface overlap entirely (request says "pull up the account book", description says "retrieve the accounting records") — the genuine paraphrase gap we called untestable. **Value slots are unchanged in all tiers** — the concrete argument values (`Zurich`, `401`) still appear verbatim (§4).

## 3. What gets built

### 3a. Gloss lexicon (`crates/grug-data/src/semantic.rs`, new — CPU, pure Rust)
Two `const` tables parallel to `procedural::{VERBS,NOUNS}` (50 verbs, 51 nouns):

```rust
/// (verb, primary_gloss, alt_gloss) — glosses share no whole-word token with the verb.
const VERB_GLOSS: &[(&str, &str, &str)] = &[
    ("fetch",   "retrieve",       "pull up"),
    ("archive", "move to storage","put into cold storage"),
    ("delete",  "remove",         "purge"),
    // ... all 50
];
/// (noun, primary_gloss, alt_gloss)
const NOUN_GLOSS: &[(&str, &str, &str)] = &[
    ("ledger",  "accounting records", "account book"),
    ("beacon",  "signal transmitter",  "locator broadcast"),
    // ... all 51
];
```

Authoring cost: ~101 entries × 2 glosses, hand-written once. A test asserts **zero whole-word overlap** between each name token and its glosses (so D2 is genuinely surface-disjoint), and that every gloss is unique enough to keep tools distinguishable.

### 3b. Semantic descriptions + paraphrase phrasing (extend `procedural.rs`)
- `describe_tool(tool, SignalMode) -> String`: D0 keeps the current echo; D1/D2 build *"{Verb-gloss-primary} the {noun-gloss-primary} for a given {first-param-role}."*
- `verb_phrase_for_signal(name, SignalMode) -> String`: D0 = `verb_phrase_for`; D1 = primary gloss phrase; D2 = alt gloss phrase. `build_templates` takes a `SignalMode` and calls this instead of `verb_phrase_for`. The `SHAPES` machinery is untouched — only the verb-phrase string changes, so held-out-phrasing (axis B) and the "value verbatim" invariant are preserved for free.
- `build_catalog_signal(seed, n_tools, templates, SignalMode)`; `Dataset` gains a `SignalMode` knob threaded to `new_sized_*`.

### 3c. Retriever keyed on description, not name (extend `retriever.rs`)
- `Retriever::new_over(&[&str])` / `SemanticRetriever::new_over(&[&str])` — build over an arbitrary per-tool text (name **or** description), so the recall matrix can compare name-keyed vs description-keyed retrieval without duplicating types. (Current `new` = name-keyed; add a description-keyed constructor.)

### 3d. Learned-embedding retriever for D2 (`examples/` + Seshat)
D2 is unretrievable by any surface method by construction, so it needs a *learned* embedder. Reuse **Seshat's `all-MiniLM-L6-v2`** (already in the stack, ADR 047) rather than training anything: embed each tool description and each request, rank by cosine. Delivered as a measurement example (`examples/semantic_recall_minilm.rs`) that calls the Seshat embedding path (or an in-process `candle`/`ort` MiniLM if the offline path is cleaner). This is the one piece that reaches outside grug-data — deliberately, because it *is* the "real router" the caveat points at.

## 4. The control — value-grounding must not move

Paraphrase attacks **retrieval** and **tool selection**; it does **not** touch argument *values*, which still appear verbatim in the request (§2). So `ground_values` is a **control**: its lift (+0.145 whole-call at seed 42) should be ~unchanged across D0/D1/D2. If it moves, the corpus change leaked into the value slots — a bug, not a finding. State and check this explicitly.

## 5. Measurements

### P3d-1 — recall matrix (CPU, model-independent, `examples/recall_matrix.rs`)
For each tier × retriever, recall@8 over the registry (reuse `retrieval_recall_at_k` / `semantic_recall_at_k`):

| retriever \ tier | D0 | D1 | D2 |
|---|---|---|---|
| lexical-on-name | 1.000 (expect) | **↓ (expect ≈0)** | ≈0 |
| lexical-on-description | — | **↑ (expect high)** | **↓ (expect low — disjoint vocab)** |
| HDC-on-description | — | high | partial (shared morphology only) |
| MiniLM-on-description (P3d-2) | — | high | **the test: how much does learned semantics recover?** |

This single table quantifies *how much of recall@8=1.0 was surface coupling* and *what a learned router buys* — the two things lever 2 could only gesture at.

### P3d-2 — learned embedder recovers D2 (Seshat MiniLM)
Report D2 recall@8 for MiniLM-on-description vs lexical/HDC. Success = MiniLM materially > surface methods on D2 (confirms the caveat's premise: the real router is the thing that makes paraphrase retrievable).

### P3d-3 — retrain + re-measure abstention and the full stack (GPU, ~65 min each)
Retrain the 13M on the **D1** retrieval-scoped corpus (offered set = description-keyed retriever's top-8; label = f(offered set), as P3a) and re-run `run_retrieval_eval_fmt` + `eval_registry_repair`:
- **Abstention recall / false-abstention** — *the number most at risk.* If it holds (≈0.9+), M1-rescues-M3 is a structural property, not a surface artifact. If it craters toward P3c's 0.24, then a large part of the 0.998 *was* the exact-match cue — a genuine, publishable negative that corrects the record.
- **Full stack** (raw → M2 → grounded) — does the four-stage stack survive the harder offered set? Value-grounding lift is the §4 control.
- Optionally a **D2** retrain (offered set from MiniLM at corpus-build time — heavier, couples the generator to the embedder), only if D1 is encouraging.

## 6. Success criteria / what would falsify the thesis

- **Confirms ADR 089:** D1 recall stays high on *description*-keyed retrieval, abstention stays high (≈0.9+) → "retrieval-scoped offered sets make call/abstain crisp" is semantic, not surface; MiniLM recovers D2 recall → the learned router is the right production component.
- **Falsifies / qualifies:** abstention collapses under D1/D2 → the 0.998 was substantially a surface artifact; the honest correction is that M3 at 13M is the P3c ~0.24 abstainer once the cue is semantic, and M1's rescue is weaker than measured. **This is a real possible outcome and the reason to run P3d** — better a measured negative than a shipped asterisk.
- **Control:** `ground_values` lift within noise of §Finding 5 across tiers.

## 7. Scope, ordering, risk

1. **P3d-1 first** (CPU, ~1 day incl. lexicon authoring): the recall matrix alone converts the two biggest caveats into numbers. Low risk, high information.
2. **P3d-2** (Seshat MiniLM wiring): the learned-retriever number for D2. Medium — the only external dependency.
3. **P3d-3** (GPU retrain): the abstention falsification test. One 65-min run for D1; D2 optional.

Risks: (a) gloss authoring quality — a lazy gloss that shares a token re-introduces surface signal (guarded by the zero-overlap test); (b) MiniLM on short tool descriptions may be weak — mitigated by richer descriptions (3b) and reporting honestly; (c) the D2-retrain generator↔embedder coupling — deferred behind D1.

## 8. Buildable checklist

- [x] `semantic.rs`: `SignalMode`, `VERB_GLOSS`/`NOUN_GLOSS` tables + zero-name-overlap + content-disjoint tests, `glossed_phrase`, `describe_tool` (`9ad8c13`).
- [x] `procedural.rs`: `build_catalog_signal` threads `SignalMode` (Name byte-identical); retired `verb_phrase_for` for `glossed_phrase` (`9ad8c13`).
- [x] `catalog.rs` / `dataset.rs`: `procedural_signal` / `new_sized_signal` constructors (`9ad8c13`). *(Retrieval-stream threading for the D1 retrain is P3d-3.)*
- [x] `retriever.rs`: `Retriever`/`SemanticRetriever::new_over(text)` description-keyed constructors (`9ad8c13`).
- [x] `examples/recall_matrix.rs` (P3d-1) — the tier × retriever recall table. **Done, measured.**
- [x] `examples/semantic_recall_minilm.rs` (P3d-2, in `crates/seshat/`) + `grug_data::EmbedRetriever` — Seshat/MiniLM D2 recall. **Done: D2 minilm@desc = 0.210** (beats the ~0.01–0.02 surface floor ~11–17× but far from 1.0 — partial bridge). Model provisioned from `sentence-transformers/all-MiniLM-L6-v2`; runtime needs `ORT_DYLIB_PATH` → onnxruntime 1.23.x.
- [x] `train_registry` + `eval_registry_repair`: `--signal-mode` + mode-aware `registry_retriever()` (desc-keyed under D1/D2; name-keyed D0 byte-identical) (`1c4d6ee`). **D1 retrain done (`runs/p3d-registry-d1/`, 82 min, loss 0.052): abstain recall 0.996 / false 0.020 / premise 0.992; raw 0.520 → grounded 0.680 (+0.160 control ✓); modest call-quality tax (desc-similar confusables). The 0.998 confirmed structural, not surface.**
- [x] Docs: scaling-results §16 (`46...`), plan + memory + Seshat. *(ADR 089 M1-caveat update folded into §16; the abstention confirm-or-correct waits on P3d-3.)*
