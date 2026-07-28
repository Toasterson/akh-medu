# LLM Wiki — Adversarial Comparison against Akh-Medu's ADRs, Gap Analysis, and Literature Review

> **Date:** 2026-07-28 · **Status:** Research (no decision taken) · **Author:** research pass on branch `claude/llm-wiki-adversarial-gap-analysis`
> **Primary targets:** [ADR 081](../decisions/081-akh-memory-schema-typing.md) (memory & schema typing), [ADR 021](../decisions/021-akhipedia-architecture.md) (Akhipedia), [ADR 065](../decisions/065-context-assembly-runtime.md) (context assembly), [ADR 057](../decisions/057-code-understanding-kg.md) (projected read-model)
> **Anima project:** `019d9844-ce74-7993-a9f0-c953b8f04a4e` ("Akh-Medu")

---

## 0. Scope and method

ADR 081 and its plan cite **"Obsidian / Karpathy LLM-Wiki"** as one of two external grounding systems — with no URL, no quotation, and no dated retrieval. That citation is load-bearing: it supplies one entire axis of the design ("Akh is already ~90% Obsidian-shaped") and justifies the permissive-substrate half of the thesis. This document does what the ADR did not: retrieves the primary source, reads it verbatim, reconstructs what the pattern actually claims, and then argues *against* Akh's design from that source and from the 2026 literature.

Method:

1. **Primary source retrieved verbatim** — `gist.github.com/karpathy/442a6bf555914893e9891c11519de94f` (`llm-wiki.md`), fetched 2026-07-28 via the raw endpoint. All quotations below are from that file.
2. **Landscape** — the derivative ecosystem (WUPHF, Hermes bundled skill, LLM Wiki v2, Nowledge Mem, qmd) and the critical literature that reviews it.
3. **Peer/preprint literature** — ~40 papers surveyed across six framings (agentic memory, wiki generation, extraction-vs-verbatim, consolidation drift, structured-output reliability in small models, temporal conflict). ~20 read or abstract-verified.
4. **Akh-side verification in code**, not from ADR prose. Every claim below marked **[verified]** was checked against the tree at `06d400f`. This matters: two of the sharpest findings are things the ADRs assert that the code does not do.

**Bias disclosure.** The adversarial frame is deliberate — this document argues the case *against* Akh's current design. Section 6 states, separately and honestly, where Akh is ahead of the pattern. Neither section is a verdict; Section 8 is.

---

## 1. What the LLM Wiki pattern actually says

Karpathy's `llm-wiki.md` is an **idea file** — explicitly "designed to be copy pasted to your own LLM Agent", and "intentionally abstract. It describes the idea, not a specific implementation."

### 1.1 The thesis

> "Instead of just retrieving from raw documents at query time, the LLM **incrementally builds and maintains a persistent wiki**… The knowledge is compiled once and then *kept current*, not re-derived on every query."

The framing is **compilation, not retrieval**. RAG re-derives synthesis per query; the wiki is "a persistent, compounding artifact" where "the cross-references are already there. The contradictions have already been flagged."

Division of labour is explicit: *"You never (or rarely) write the wiki yourself — the LLM writes and maintains all of it. You're in charge of sourcing, exploration, and asking the right questions."* And the slogan: **"Obsidian is the IDE; the LLM is the programmer; the wiki is the codebase."**

The closing rationale names the actual bet:

> "The tedious part of maintaining a knowledge base is not the reading or the thinking — it's the bookkeeping… Humans abandon wikis because the maintenance burden grows faster than the value. LLMs don't get bored… The wiki stays maintained because the cost of maintenance is near zero."

It is positioned as a solution to Vannevar Bush's Memex, whose unsolved part was *"who does the maintenance."*

### 1.2 The three layers

| Layer | Content | Ownership |
|---|---|---|
| **Raw sources** | Articles, papers, images, data files. **Immutable** — "the LLM reads from them but never modifies them. This is your source of truth." | Human curates |
| **The wiki** | LLM-generated markdown: summaries, entity pages, concept pages, comparisons, overview, synthesis. "The LLM owns this layer entirely." | LLM writes, human reads |
| **The schema** | `CLAUDE.md` / `AGENTS.md` — how the wiki is structured, conventions, workflows. **"This is the key configuration file — it's what makes the LLM a disciplined wiki maintainer rather than a generic chatbot."** Co-evolved by human + LLM. | Joint |

### 1.3 The three operations

- **Ingest** — read source → discuss takeaways → write summary page → update index → **update relevant entity and concept pages across the wiki** → append to log. "A single source might touch 10-15 wiki pages."
- **Query** — search, read, synthesize with citations. Critically: *"**good answers can be filed back into the wiki as new pages**… your explorations compound in the knowledge base just like ingested sources do."*
- **Lint** — periodic health check: *"contradictions between pages, stale claims that newer sources have superseded, orphan pages with no inbound links, important concepts mentioned but lacking their own page, missing cross-references, data gaps."*

### 1.4 The two navigation files

- **`index.md`** — content-oriented catalog, read *first* on every query, then drill in. Karpathy claims this "works surprisingly well at moderate scale (~100 sources, ~hundreds of pages) and **avoids the need for embedding-based RAG infrastructure**."
- **`log.md`** — chronological, append-only, grep-parseable by convention (`## [2026-04-02] ingest | Title`).

Search is *optional and deferred*: "at small scale the index file is enough, but as the wiki grows you want proper search" — pointing at `qmd` (local hybrid BM25/vector + LLM rerank).

---

## 2. Correcting ADR 081's characterization of its own grounding

This is the first adversarial finding, and it undermines an axis of the design rather than a detail.

ADR 081 and the plan describe the reference system as:

> **"Obsidian / Karpathy LLM-Wiki"** — an **emergent, loosely-typed** markdown + wikilink store the **LLM authors**, where typed links are an overlay synced to frontmatter.

And tabulate it against Orbit on a *schema rigidity* axis as "Open, emergent; typed links are an overlay synced to frontmatter."

Four problems:

**(a) It conflates two different systems under one slash.** "Obsidian" (a note app with a Properties/Dataview plugin ecosystem where typed links *are* a frontmatter overlay) and "Karpathy LLM-Wiki" (a pattern) are not the same reference. In the primary source Obsidian is **only a rendering surface** — "Obsidian is the IDE" — plus five optional "Tips and tricks" (Web Clipper, attachment paths, graph view, Marp, Dataview). Dataview is listed as one optional plugin, *conditional* on the LLM adding frontmatter: "**If** your LLM adds YAML frontmatter…". The typed-links-as-frontmatter-overlay claim describes Obsidian-the-app, not the pattern the ADR names.

**(b) The pattern is not schema-free.** Karpathy's third architectural layer *is* the schema, and he calls it **"the key configuration file."** Its absence is what makes the LLM "a generic chatbot" rather than a disciplined maintainer. ADR 081 places the pattern at the "open, emergent" pole of a *schema rigidity* axis — but the source's own architecture puts a schema document at the centre. The real contrast with Orbit is not *schema vs. no schema*; it is **prose-schema co-evolved with the LLM and enforced by instruction-following** vs. **machine-enumerated schema enforced by a type checker**. That is a materially different axis, and framing it correctly weakens the ADR's inference that Akh "is already ~90% Obsidian-shaped and merely lacks Orbit's typed-core discipline."

**(c) The three operations were dropped.** The ADR imports the substrate (permissive notes, open frontier, never block a write) and discards Ingest/Query/Lint, `index.md`, and `log.md` — which is where the pattern's actual value lives. See §5, gaps **G1–G3**.

**(d) No dated citation.** The gist has been revised since publication and its comment thread is active; the derivative ecosystem (LLM Wiki v2, WUPHF, Hermes, Nowledge Mem, Link, Kherad) post-dates it. An undated, un-URL'd citation for a load-bearing design input is not reproducible. **Recommended fix regardless of any other outcome in this document:** amend ADR 081's References with the gist URL, a retrieval date, and the corrected characterization above.

> **Caveat, stated plainly.** Correcting the characterization does not by itself invalidate ADR 081's decisions. Most of the twelve locked decisions are argued from Akh's own requirements (R1–R6) and from Orbit, not from Karpathy. The damage is confined to the "~90% Obsidian-shaped, just add typed core" framing and to the operations that framing caused the design to omit.

---

## 3. The 2026 landscape

The pattern spread quickly — three independent LLM-wiki projects hit the Hacker News front page within 24 hours of each other, per the [WUPHF thread](https://news.ycombinator.com/item?id=47899844) — and produced a visible cluster of implementations plus one academic operationalization:

| System | Shape | Relevance to Akh |
|---|---|---|
| **LLM-Wiki** (Tencent WeChat, [arXiv 2605.25480](https://arxiv.org/abs/2605.25480)) | The pattern turned into a measured system: compile → linked pages → agent-native traversal tools → **Error Book** self-correction | The strongest evidence *for* wiki-shaped memory, and the sharpest measurement of what Akh is missing |
| **Memory as Metabolism** ([arXiv 2604.12034](https://arxiv.org/abs/2604.12034)) | Normative governance profile *over* the pattern: TRIAGE / CONTEXTUALIZE / DECAY / CONSOLIDATE / AUDIT + memory gravity + minority-hypothesis retention | Names the retention/governance layer Akh also lacks |
| **LLM Wiki v2** | Ebbinghaus decay, multi-tier consolidation (working/episodic/semantic/procedural), audit trails | Practitioner prior art for decay policy |
| **WUPHF** (nex-crm, Show HN 260pts) | Markdown+git truth, bleve BM25 + SQLite index, per-agent private notebooks + shared team wiki, **broken-link detection rendered in red** | Multi-agent shared/private split ≈ Akh's R1; ships dangling-link lint |
| **Hermes bundled skill** (Nous Research) | `SCHEMA.md` template with typed frontmatter (`type`, `confidence`, `contested`, `contradictions[]`), ≥2 outbound links/page, provenance markers `^[raw/…]` | Closest existing frontmatter vocabulary to ADR 081's Note |
| **qmd** | Local hybrid BM25/vector + LLM rerank over markdown, CLI + MCP | The pattern's reference search backend |

Two things stand out for Akh:

1. **Hermes' `SCHEMA.md` already carries `contested: true` and `contradictions: [page-slug]` in frontmatter.** Akh has a far richer conflict apparatus (Dempster–Shafer + source reliability, `src/schema/conflict.rs` **[verified]**) but no surfaced per-note contested marker. The cheap thing is missing while the expensive thing is built.
2. **The critical community consensus is that the pattern's weak point is maintenance discipline, not storage.** The recurring HN objection — *"How do you keep LLMs from writing too much? … It looked good until you read the actual data"* — is the same failure the literature measures in §4.3.

The best available critical synthesis is zby's *Designing a Memory System for LLM-Based Agents* (Commonplace), which frames memory as a **context-engineering** problem and separates **representational form** from **behavioral authority**. Its verdict on the wiki class is precise and applies to Akh:

> "A wiki or rules engine can preserve or enforce knowledge, but runtime activation, artifact contracts, behavioral closure, rule selection, relaxation, and exception handling remain outside the substrate."

Its two framing claims worth adopting: *"Raw accumulation does not create usable memory"* and *"Knowledge storage does not imply contextual activation."* Both bear directly on ADR 081's read-path decision (§5, **G4**).

---

## 4. The scientific evidence, and what it does to Akh's design

### 4.1 Evidence *for* compiled, linked, traversable knowledge

**LLM-Wiki (arXiv 2605.25480, Tencent)** is the load-bearing positive result. It compiles documents into linked Wiki pages with bidirectional links, exposes `wiki_search` / `wiki_read`, and self-corrects via an Error Book. Against seven baselines on the same corpus with GLM-5.1 throughout:

| Method | HotpotQA F1 | MuSiQue F1 | 2WikiMHQA F1 |
|---|---|---|---|
| Vanilla RAG (Dense) | 0.764 | 0.611 | 0.815 |
| GraphRAG | 0.771 | 0.582 | 0.720 |
| HippoRAG 2 | 0.805 | 0.624 | 0.831 |
| LightRAG | 0.819 | 0.659 | 0.847 |
| **LLM-Wiki** | **0.839** | **0.739** | **0.911** |

Three findings that matter here:

- **Gains scale with reasoning depth.** On 2WikiMHQA the margin over LightRAG widens from 5.7 F1 (2-hop) to 8.3 F1 (4-hop), reaching 0.983 F1 at 4 hops. This is the same shape as Akh's own measured thinking-engine scaling law ([ADR 085](../decisions/085-thinking-engine-findings.md)) and suggests the two compose.
- **Traversal is not free with the structure — it must be *used*.** Ablating progressive traversal (single search, read top pages, answer, no link-following) costs **11.7–13.8 F1**. Compiled structure that nothing walks is worth roughly nothing.
- **The Error Book is worth 3.4–4.0 F1**, via two-layer repair: deterministic code auto-fix for structural errors (dangling links, index inconsistencies) after every batch, plus periodic LLM repair for semantic errors (unsupported facts, cross-page contradictions), with accumulated constraints injected into later compilation prompts.

And it is **not** slower: 14.9–27.1 s/query vs. LightRAG's 39.7–51.3 s, because pre-compiled links mean reading 2.5–3.9 pages/query instead of a fixed 5 passages.

Its stated limitations are Akh's too: one-time compilation cost, and *"as the Wiki grows to tens of thousands of pages, directory indices may become unwieldy"* — which is the falsification boundary on Karpathy's `index.md` claim.

### 4.2 Evidence *against* extraction that replaces its source — the sharpest attack on ADR 081

**"Verbatim Chunks Beat Extracted Artifacts" ([arXiv 2601.00821](https://arxiv.org/abs/2601.00821))** runs a controlled ablation: one fixed retrieval→rerank→reasoning pipeline, same backbone, retriever, reranker, and judge; **only the stored representation changes**.

| Stored representation | LoCoMo | LongMemEval-S |
|---|---|---|
| Verbatim conversation chunks | **43.9%** | **67.4%** |
| LLM-extracted typed artifacts | 28.0% | 45.4% |
| Naive RAG (k=10) | 27.6% | — |
| **chunks ∪ artifacts (union)** | **42.5%** | — |

- **−15.9 pts on LoCoMo, −22.0 on LongMemEval-S**, uniform across every answerable question type. The extracted pipeline **never beats naive RAG overall**.
- Five confound controls fail to rescue it: graph disabled (−0.8), zero chunk overlap (42.8), budget-matched at k=60 (recovers only 1.2pp), dense-only retrieval (gap still 14.3), session-level rather than per-turn extraction (gap 14.7). A 9-config GraphRAG sweep tops out at 13.0%.
- **Mechanism: lossy distillation.** "Extraction commits to relevance before the question exists." **69.0% of diagnosable failures are write-time losses** — facts the extractor never deliberately wrote down. Failure taxonomy: extraction gaps 78.8%, temporal blindness 14.9%, reasoning-chain breaks 6.4%.
- A stronger extractor does not help (gpt-4o-mini → gpt-4o: 29.7% vs 30.3%, p≈1). Nor does cost: $12.5 per 1,000 correct answers for chunks vs $14.9 for artifacts.
- **Where extraction wins:** a synthetic probe where every fact is pre-shaped into exactly one artifact (81.0% vs 55.5%) — *"the premise holds exactly where the world has been arranged to satisfy it."* And abstention (15.0% vs 6.5%), by returning nothing assertable.

The paper's own reconciliation is the constructive part, and it is the recommendation for Akh: **"structure that annotates verbatim text preserves a lossless path from question to source; structure that replaces it severs that path."** Retrieval accuracy tracks *distance from source*. Corroborating: *Storage Is Not Memory* ([arXiv 2605.04897](https://arxiv.org/abs/2605.04897)) — "extraction at ingestion is the wrong primitive" — reaches 93.0% on LoCoMo over verbatim-preserved events; MemMachine ([arXiv 2604.04853](https://arxiv.org/abs/2604.04853)) is explicitly "ground-truth-preserving" and critiques the MemGPT lineage for "compounding error over time."

> **How much of this lands on Akh.** Partially deflected, and the deflection is real: ADR 081 decision 2 keeps a **permissive truth store** and makes the typed KG a *projection*, never a replacement. That is architecturally a `chunks ∪ artifacts` union store — the configuration the paper finds *matches* verbatim (42.5 vs 43.9). Akh did not make the mistake the paper falsifies.
>
> **What still lands is the read path, not the write path.** Decision 11 makes core = "active-note **frontmatter only** + schema digest", recall = "recent episodic + recently-touched **frontmatter**", and puts bodies exclusively in archival behind an explicit `archival_search` call. The union exists on disk but the *default* context assembly is served almost entirely from the extracted side — precisely the losing configuration, with the verbatim side reachable only if the model thinks to ask. This is a one-line-of-reasoning fix, not a redesign; see **G4**.

### 4.3 Evidence against continuous LLM-driven consolidation

**"Useful Memories Become Faulty When Continuously Updated by LLMs" ([arXiv 2605.12978](https://arxiv.org/abs/2605.12978), UIUC/Tsinghua)** is the most important paper in this review for Akh's Librarian.

- **Memory utility is non-monotonic**: rises early, then falls *below the no-memory baseline*. On WebShop, AWM goes 0.64 @ 8 examples → 0.20 @ 128, where no-memory sits at 0.20.
- **The cleanest result:** GPT-5.4 solves a 19-problem ARC-AGI slice at **100%** with no memory. Streaming those same problems through consolidation **with ground-truth solutions available at every step** drops it to **54%**. The inputs are useful by construction, so the failure is attributable to the consolidation step itself.
- **Streaming loses 17–38 pts vs. one-shot whole-pool consolidation.** "Early abstractions can anchor later rewrites and mistakes are reused as context for future consolidation, so small errors compound into progressively distorted memories."
- **Grouping by task family beats pooling heterogeneous material** into a single abstraction step.
- **An episodic-only control — just retaining raw trajectories — is competitive with every consolidator tested**, and in ARC-AGI Stream agents given Retain/Delete/Consolidate actions *preserve episodes by default* and **double** the accuracy of forced-consolidation counterparts.
- Prescription: *"raw episodes should be treated as first-class evidence… Abstraction should be selective, delayed, and grounded in recoverable trajectories."* And architecturally: *"keep the episodic and abstraction-forming roles distinct rather than collapsing both into a single rewrite loop."*

> **How much of this lands on Akh.** Akh is unusually well-defended here and this deserves saying: the Librarian is a **daily batch** pass (decision 10), not update-after-every-interaction — which is the design CLIN/AWM/Dynamic Cheatsheet/ACE get punished for. `EpisodicEntry` is retained as the temporal spine, so raw episodes survive. Diff-based supersede via TMS (decision 12) preserves history rather than overwriting. Three of the paper's four prescriptions are already ADR 081 decisions.
>
> **What lands:** (i) "one-shot whole-pool beats streaming" argues for **periodic full re-projection**, which the incremental diff design does not do — errors from projection *N* condition projection *N+1* indefinitely; (ii) "grouping by family helps" argues for **per-microtheory / per-type batching** rather than one undifferentiated daily pass; (iii) there is **no consolidation gate** — nothing decides *whether* to consolidate a given note this cycle. See **G5**.

Convergent evidence: *Library Drift* ([arXiv 2605.19576](https://arxiv.org/abs/2605.19576)) — unbounded self-evolving skill libraries degrade retrieval; **LLM-authored skills deliver +0.0pp while human-curated deliver +16.2pp (SkillsBench)**. *On the Limits of Self-Improving* ([arXiv 2601.05280](https://arxiv.org/abs/2601.05280)) proves that when the exogenous grounding fraction α→0, the system undergoes entropy decay and variance amplification. *The Mirror Loop* ([arXiv 2510.21861](https://arxiv.org/abs/2510.21861)): ungrounded recursive self-critique yields reformulation, not progress; a single verification step restores it. *Knowledge Collapse* ([arXiv 2509.04796](https://arxiv.org/abs/2509.04796)): factual accuracy degrades while surface fluency persists — "confidently wrong."

### 4.4 Evidence against the R2 empirical bet

ADR 081's own risk register says it plainly: *"The whole tiered-authoring premise rests on the **R2 empirical bet** — must be probed first."* The status line records that the S0 spike shipped **structural seams with deterministic fakes** instead of the live small-model probe. So the bet is still unpaid — and the literature that has accumulated since is not encouraging:

- **The Format Tax ([arXiv 2604.03616](https://arxiv.org/abs/2604.03616)):** structured-output requirements substantially degrade reasoning across open-weight models, and *"format-requesting instructions alone cause most of the accuracy loss, before any decoder constraint is applied."* Prescription: decouple reasoning from formatting.
- **The Constraint Tax ([arXiv 2605.26128](https://arxiv.org/abs/2605.26128)):** the assumption that hard constraints improve reliability without changing the answer *"is unsafe for small models"* (sub-3B focus).
- **When Correct Isn't Usable ([arXiv 2605.02363](https://arxiv.org/abs/2605.02363)):** across three 7–9B models, joint task-correctness *and* schema-validity diverges sharply from task accuracy alone.
- **Tool Suppression ([arXiv 2606.25605](https://arxiv.org/abs/2606.25605)):** when tool-calling and JSON-Schema constraints are enabled *simultaneously*, multiple open-weight models **stop calling tools** while maintaining schema compliance. This is a direct hazard for ADR 081's grammar-constrained task tool combined with the agentic loop, and for [ADR 067](../decisions/067-candle-conversational-boundary.md)'s grammar-constrained Candle tool calling.
- Counterweight, in fairness: [arXiv 2510.03847](https://arxiv.org/abs/2510.03847) surveys SLMs (1–12B) and argues they are "sufficient and often superior for agentic workloads where the objective is schema- and API-constrained accuracy," when paired with guided decoding (XGrammar/Outlines). So the bet is *contested*, not refuted — which is exactly why it needs measuring rather than assuming.

ADR 081's mitigation for R2 — anchor on schema.org because "small models already emit it," converting *invent* into *recall* — is a genuinely good instinct and is not what the literature attacks. What it attacks is the **structured-emission channel**, independent of vocabulary familiarity. Those are separable and should be measured separately.

### 4.5 Evidence on staleness and temporal conflict

- **Temporal Validity in Retrieval Memory ([arXiv 2606.26511](https://arxiv.org/abs/2606.26511)):** cosine similarity distinguishes a *contradicted* fact from a *duplicated* one at **AUROC 0.59 — near chance**. Contradictions are often more embedding-similar to the original than rephrased duplicates are. Embedding retrieval structurally cannot do supersession.
- **Don't Ask the LLM to Track Freshness ([arXiv 2606.01435](https://arxiv.org/abs/2606.01435)):** on MemoryAgentBench FactConsolidation, *every* published system underperforms — HippoRAG-v2 54%, BM25 48%, Mem0 18%, **Zep/Graphiti (a temporal KG) 7%**; multi-hop near-unsolved. Argues for a deterministic recipe over LLM judgment.
- **STALE ([arXiv 2605.06527](https://arxiv.org/abs/2605.06527)):** *Implicit Conflict* — a later observation invalidates an earlier memory without explicit negation — is the underexplored failure mode. Akh's diff-supersede handles *explicit* re-assertion of the same triple identity; implicit invalidation is out of its reach.
- **TOKI ([arXiv 2606.06240](https://arxiv.org/abs/2606.06240)):** types the four production resolution heuristics (last-writer-wins, evidence-weighted merge, await-confirmation, per-rule policy) as bitemporal operators with declared isolation levels. Akh uses evidence-weighted merge (Dempster–Shafer + source reliability) — the *right* family per this taxonomy — but declares no isolation level and admits no write-time anomaly contract.
- **Presentation, Not Mechanism ([arXiv 2607.16019](https://arxiv.org/abs/2607.16019)):** a render confound — evaluations of deprecation-aware memory often change mechanism *and* prompt presentation together. A methodological warning for any Akh A/B on this axis.

**[verified]** `grep` for `valid_from|valid_until|valid_time|bitemporal` across `src/` returns only `src/email/compose.rs`. Akh triples carry provenance and TMS retraction state but **no temporal validity interval**.

---

## 5. Gap analysis

Severity: **P0** blocks a stated requirement or contradicts measured evidence · **P1** materially limits value · **P2** hygiene.
"State" is verified against the tree at `06d400f` unless marked *(ADR claim)*.

---

### G1 — No lint/audit pass over the note store · **P0**

**Pattern requires:** periodic health check for contradictions between pages, stale claims superseded by newer sources, **orphan pages with no inbound links**, concepts mentioned but lacking a page, missing cross-references, data gaps.

**Akh state [verified]:** `grep -rn "orphan|lint|contradict|stale|backlink" src/wiki/` → **no matches**. Sleep (`src/agent/sleep.rs`) prunes *orphaned KG edges* with an age threshold — a different operation on a different substrate; it does not read notes, does not check inbound links to note pages, and does not detect dangling wikilinks. `src/schema/conflict.rs` resolves conflicts *at assertion time* for a given subject-predicate; it is not a sweep. `CompositeVerifier`'s 9l contradiction check operates inside the thinking loop over hypotheses, not over the note/wiki store.

**Partial prior art, credited:** [ADR 078](../decisions/078-episodic-self-refinement.md) (*"the akh cleans its own memory"*) covers **one** of Karpathy's six lint checks — "data gaps that could be filled." `proposals_from_episodic_gaps` scans recent episodes, flags those with fewer than 5 *grounded* facts plus a known `tag:source:`, and raises a `GapDetection` goal to re-read the source. That is the right *shape* — lint-as-goal, reusing the OODA loop rather than bolting on a sweep — and G1's remedy should follow it rather than invent a parallel mechanism. But it is scoped to **episode thinness**, i.e. coverage; it checks no contradiction, no orphan, no dangling link, no staleness. Its own status section records it as `Prototype`, env-gated behind `AKH_ENRICH_MEMORY`, with **the execution half not implemented** and untuned heuristics (threshold 5, 50-episode window). So the gap is real, and ADR 078 is the template for closing it.

**Evidence:** Error Book ablation = **−3.4 to −4.0 F1** (2605.25480). AUDIT is the primary defense against Kuhnian ossification (2604.12034). WUPHF ships dangling-link detection as table stakes. Without lint, §4.3's compounding-error dynamic has no corrective channel — and 2601.05280 proves that a self-referential loop with vanishing exogenous grounding degenerates.

**Remedy:** a two-layer maintenance pass mirroring LLM-Wiki's Error Book. Layer 1 deterministic, no LLM, runs after every Librarian batch: dangling `links-to` targets (already projected as triples by `src/schema/project.rs` **[verified]**, so this is a graph query), orphan notes with zero inbound `links-to`, frontmatter keys violating installed `onto:` shapes, notes whose `is-a` type is unresolved. Layer 2 LLM-driven, periodic, batched: cross-note contradiction candidates surfaced by the existing DS machinery, stale-claim candidates, concepts referenced by ≥N notes with no page. Persist findings as an Error Book microtheory with lifecycle status so constraints accumulate and can be injected into later projection prompts. **Layer 1 is cheap and should not wait on Layer 2.**

---

### G2 — Query results never file back into memory · **P0**

**Pattern requires:** *"good answers can be filed back into the wiki as new pages… your explorations compound in the knowledge base just like ingested sources do."*

**Akh state [verified]:** the grounded chat path and `run_agentic_turn` produce answers; nothing writes a synthesized answer back as a Note. [ADR 091](../decisions/091-transcript-episodic-capture.md) captures transcripts as *episodic* entries — the temporal spine, correctly — but an episode is not a concept page. [ADR 078](../decisions/078-episodic-self-refinement.md) refines from episodes, again not into note-level synthesis. `MemoryWriteTool` exists but writes triples, not notes; and ADR 081 decision 10 deliberately removed model-decided capture ("no `take_note` tool exists, capture is structural").

**Why this is P0, not P1:** it breaks the pattern's central claim. Compounding is the *entire* differentiator over RAG. Without the query→wiki feedback edge, Akh compiles from sources but never from its own reasoning, and every expensive deliberation (the thinking engine's 18.7·d+20 steps at depth *d*, per ADR 085) evaporates into transcript. This is arguably the single highest value-per-effort gap in this document.

**Tension to resolve honestly:** ADR 081 decision 10 rejected `take_note` because small models judge noteworthiness unreliably — a sound argument that §4.4's evidence supports. But "the model decides what to note mid-stream" and "the daily Librarian decides which completed deliberations earned a page" are different decisions made by different models at different times. The latter is a batch judgment by gpt-oss 120B over completed, outcome-labelled work, which is exactly the regime §4.3 says abstraction should live in ("selective, delayed, and grounded in recoverable trajectories"). Route answer-filing through the Librarian, not through a small-model tool.

---

### G3 — No index or log surface · **P1**

**Pattern requires:** `index.md` read first on every query then drilled into; `log.md` append-only, greppable.

**Akh state:** ADR 081 decision 11 puts a `get_graph_schema` digest in core — that is a *type* digest, not a *content* catalog. Akh has a WAL and audit ledger, but no agent-legible chronological narrative of "what was ingested/queried/consolidated when."

**Evidence:** Karpathy claims index-first navigation obviates embedding RAG at ~100 sources / hundreds of pages; LLM-Wiki uses `_index.md` directory reads as a first-class traversal route ("browse-first") and reports its own scaling ceiling at tens of thousands of pages. Akh has HNSW + Heka and does not need index-as-substitute-for-search — but it does need index-as-**map**, the thing that tells the model what *exists* before it searches. That is the function `archival_search` cannot serve: you cannot search for a page whose existence you do not suspect.

**Remedy:** a generated, always-cheap note catalog (title + one-line + type + updated) projectable from the KG, tiered into core when small and into recall when large; and a rolling log projection over `EpisodicEntry` with a stable grep-parseable prefix.

---

### G4 — Default read path is served from the extracted side · **P0**

**Pattern requires (implicitly):** the wiki page *is* the prose. Raw sources stay immutable and reachable.

**Akh state (ADR claim, decision 11):** core = identity + prefs + active goals + active-note **frontmatter only** + schema digest; recall = recent episodic + recently-touched **frontmatter**; **bodies live only in archival**, reachable via `archival_search`. Rationale: "bodies are never resident → the footprint is horizon-independent → there is nothing to compact" — a genuinely elegant answer to R4.

**Evidence:** this is the exact configuration 2601.00821 measures at **−15.9 / −22.0 pts**. The union store exists on disk, so Akh escapes the worst case; but a union store only pays if the verbatim side is *retrieved*, and here it is gated behind an extra tool call the model must think to make. The paper's failure taxonomy predicts the symptom precisely: implicit attributes, emotional/affective states, casual temporal references ("last Tuesday"), and negations/constraints ("does NOT want X") are what frontmatter and typed triples systematically fail to carry. ADR 065's own lesson — retrieval accuracy tracks distance from source — is the same claim from the other direction.

**Remedy — and note this preserves R4:** make recall return **verbatim spans from note bodies** (a bounded window per recalled note), with frontmatter and typed triples as *annotations on* those spans rather than as substitutes. Bodies still do not reside wholesale; only the selected span does, so the footprint stays horizon-independent. Concretely, this is a change to `MemoryRecallTool` + the `CompactingManager`/`PassiveManager` recall stage, not to the storage design. **Then measure it** — the 44j A/B harness (`src/bench_phase44/`) already exists and can run frontmatter-only vs. span-annotated recall as a two-arm comparison.

---

### G5 — Consolidation is ungated, unbatched by family, and never re-baselined · **P1**

**Akh state (ADR claim, decisions 10 + 12):** one daily Librarian pass; incremental diff-supersede against the prior projection.

**Evidence:** 2605.12978 — streaming loses 17–38 pts vs. whole-pool; family-grouped beats pooled; forced consolidation halves accuracy vs. agent-chosen retain/consolidate. Incremental diffing means the projection is a chain: projection *N* is derived under the conditioning of projection *N−1*, forever. There is no point at which accumulated projection error is discharged.

**Remedy:** (i) **periodic full re-projection** from the permissive truth store — cheap because truth is retained, and it is the paper's measured-best regime; (ii) **group the daily pass by microtheory / entity type** rather than one undifferentiated sweep; (iii) a **consolidation gate** — a note is projected this cycle only if it changed materially or its type shape changed, with `Retain` as the default action; (iv) log per-cycle projection deltas so drift is *observable*, which is the precondition for the A/B in G4.

---

### G6 — R2 empirical bet still unmeasured · **P0 (process)**

**Akh state:** ADR 081 status line: *"Still open: the S0 **empirical** bet (live small-model schema.org emission — structural seams shipped with deterministic fakes instead)."* Nine slices landed on top of an unvalidated premise. The plan itself ordered this first: *"★ R2 is an empirical bet — de-risk first."*

**Evidence:** §4.4 — Format Tax, Constraint Tax, structured-output reliability gap, and especially **Tool Suppression**, which threatens the grammar-constrained task tool and the agentic tool-calling loop *jointly*.

**Remedy:** run the probe that was specified and deferred, and split it into the three independently-failing channels the literature distinguishes:
1. **Vocabulary recall** — does gemma4-12b emit *correct schema.org type names* for known entities? (This is the part ADR 081's mitigation actually defends, and it may well pass.)
2. **Format compliance under load** — does frontmatter emission survive when the model is also reasoning? Measure joint correctness, not format validity alone (2605.02363's methodology).
3. **Tool-calling under constraint** — does the structured task tool suppress *other* tool calls when grammar constraints are active (2606.25605)? Test with constraints on and off.

`examples/agentic_flow.rs` + `AKH_AGENT_DEBUG=1` is the harness the plan already named. This is a day of work that retires the largest single risk in ADR 081.

---

### G7 — Akhipedia is orphaned, and duplicates the Note carrier · **P1**

**Akh state [verified]:** `src/wiki/` is 2,896 lines implementing `ArticleStore`, typed blocks, comrak/minijinja rendering, Tantivy search, ActivityPub federation, and **four `impl Tool`** (`WriteArticleTool`, `CiteArticleTool`, `RespondToArticleTool`, `SearchWikiTool`). `grep` for those four names outside `src/wiki/tools.rs` returns **exactly one hit** — the `pub use` re-export in `src/wiki/mod.rs`. `Agent::register_builtin_tools` registers 30+ tools (`src/agent/agent.rs:316–433`); `src/bin/akhomed.rs` registers five more. **None is a wiki tool.** The akh cannot write, cite, respond to, or search its own encyclopedia.

Meanwhile ADR 081 built a *second* note substrate — `src/library/note.rs` with `Note`, `Frontmatter`, `Wikilink`, `extract_wikilinks` **[verified]** — projected to the KG by `src/schema/project.rs`, which emits `links-to` triples from wikilinks. Two overlapping designs; no reconciliation ADR. CLAUDE.md lists Phase 22 as "verified complete in the 2026-07-05 roadmap audit," which is true of the *module* and false of the *capability*.

Also: ADR 021's link model is `cites` + `responds_to` only — **no wikilinks, no backlinks, no index, no log**. It is an article store with federation, not a wiki in the pattern's sense. The Note carrier is closer to the pattern than Akhipedia is.

**Remedy:** decide and record which substrate is the note store. Recommended: **Note (ADR 081) is the memory substrate; Akhipedia becomes the *publication and federation surface* over it** — projected/rendered from Notes rather than separately authored. That keeps ADR 021's genuinely differentiated assets (typed blocks, static HTML rendering, ActivityPub federation, opinion/debate threads) and drops the duplicate authoring path. Either way, register the tools or delete them; unregistered `impl Tool` blocks are a maintenance liability that reads as a shipped feature.

---

### G8 — No traversal interface over the note graph · **P1**

**Akh state:** `QueryGraphTool`, `ArchivalSearchTool`, `MemoryRecallTool`, `SimilaritySearchTool` **[verified]** — all *lookup*. Wikilinks are projected as `links-to` triples, so the edges exist and SPARQL can walk them, but there is no `note_read(path) → body + outbound links as affordances` and no browse-first `read(_index)` route.

**Evidence:** 2605.25480's traversal ablation = **−11.7 to −13.8 F1**, and gains concentrate on high-hop compositional queries. The structure is only worth what walks it.

**Remedy:** two tools mirroring the measured design — `note_search` (prioritize titles/aliases/tags/type before body) and `note_read` (batch-read notes or the index, returning outbound links inline as traversal affordances). Then wire retrieval *into* the thinking loop rather than beside it: ADR 052's `ThinkingLoop` currently deliberates over a fixed focus; retrieval-as-reasoning means the loop can decide to fetch. That composition — Akh's measured depth scaling × wiki traversal's depth scaling — is the most interesting research bet in this document.

---

### G9 — No retention, decay, or entrenchment policy · **P1**

**Akh state:** VSA sleep merges near-duplicates and prunes orphaned *edges* **[verified]**; TMS retracts on supersede; DS weights conflicting sources. There is no salience decay on notes, no load-bearing protection, no stress-test of entrenched beliefs, no minority-hypothesis retention. Compounding this: ADR 081 decision 5 makes type names **globally unique, immutable, and additive** — deliberately, and it *does* solve R5 structurally. But it also means a promoted core type can never be retired, and promotion is a consensus ratchet that turns only one way.

**Evidence:** 2604.12034's five operations and two mechanisms name this layer precisely — **memory gravity** (dependency centrality protects load-bearing entries from a pruner that would otherwise delete foundations while keeping last week's popular noise), **AUDIT-by-suspension** (temporarily suspend a high-gravity entry, rerun the queries that used it, measure whether performance actually degrades — with the ShortGPT analogy that structural centrality does not predict functional necessity), and **minority-hypothesis retention** with multi-cycle buffer pressure so accumulated contrary evidence gets a structural path past a centrality-protected incumbent. The paper is honest that AUDIT sensitivity is its open problem. *Oblivion* ([arXiv 2604.00131](https://arxiv.org/abs/2604.00131)) offers decay-driven accessibility reduction rather than deletion — a better fit for Akh, which should not lose provenance.

**Remedy:** compute gravity as inbound `links-to` + `derivation` centrality (both already in the graph — this is a query, not new storage). Add a slow AUDIT cycle over the highest-gravity notes. For minority retention, Akh has an unusual advantage: DS belief intervals *already* represent "supported but not dominant" natively, so minority hypotheses need not be a separate mechanism — they need a **read path that surfaces them**, which is a recall-ranking change. Decay accessibility, never provenance.

---

### G10 — Human-gated promotion is the load-bearing part, and the consensus channel does not exist · **P2**

**Akh state:** ADR 081 status line: *"cross-akh promotion consensus signal (no cross-akh telemetry channel yet)."* So of the two registry inflows, operator pack-install works and Librarian bottom-up proposal has no consensus input.

**Evidence:** *Library Drift* — **LLM-authored skills +0.0pp vs. human-curated +16.2pp** on SkillsBench. On this evidence the mandatory human confirm (decision 8) is not bureaucratic overhead; it is the mechanism doing the work, and ADR 081 should say so explicitly rather than treating it as a governance formality. The verification-burden literature (2512.11661, 2606.01969) then applies: the confirm step must be *cheap and well-presented* or it becomes a rubber stamp, at which point automation bias makes it worse than nothing.

**Remedy:** treat human confirm as the primary quality gate, and invest in its ergonomics — show the operator the frontier-term usage counts, the aliasing candidates, the coverage delta, and the notes that would be reclassified, not just a type name. Defer the cross-akh consensus signal until there is a telemetry channel; do not let its absence block promotion, since the human gate is the part that measurably works.

---

### G11 — No temporal validity; implicit conflict undetectable · **P1**

**Akh state [verified]:** no `valid_from`/`valid_until`/bitemporal fields on triples. Supersession is diff-based on `TripleIdentity`, so it fires when the *same* triple identity is re-asserted with a different object.

**Evidence:** §4.5. STALE's *Implicit Conflict* — a later observation invalidating an earlier memory without explicit negation — is invisible to identity-diff. Embedding similarity cannot substitute (AUROC 0.59). Zep/Graphiti, a purpose-built temporal KG, scores 7% on FactConsolidation, so this is genuinely hard and not a matter of Akh being behind.

**Remedy:** minimally, a validity interval on assertions plus a deterministic recency operator (2606.01435's argument: do not ask the LLM to track freshness). Then declare Akh's isolation level in TOKI's terms — Akh already uses the right operator family (evidence-weighted merge); it just has not stated its write-time anomaly contract. Implicit-conflict detection belongs in G1's Layer-2 lint, not on the write path.

---

### G12 — No memory evaluation harness · **P1**

**Akh state:** `src/bench_phase44/` A/B harness exists for context strategies with a [report](../2026-06-21-phase44j-strategy-ab.md); `src/thinking/bench.rs` measures thinking scaling. Neither measures *memory quality*. [ADR 086](../decisions/086-feature-test-vm-harness-benchmarks.md) / the [Phase 50 plan](../plans/2026-07-05-phase50-feature-test-benchmarks.md) already scope LoCoMo and LongMemEval as public legs — good, and this gap is about wiring, not about a missing decision.

**Evidence:** every claim in §4 is a number on a public benchmark. Akh currently cannot say whether its memory is better or worse than naive RAG on any of them, which means G4, G5, and G9 are all currently un-adjudicable.

**Remedy:** wire the 44j harness to LoCoMo + LongMemEval-S as a third arm alongside the existing strategy comparison, and heed 2607.16019's render-confound warning — change *mechanism* while holding *presentation* fixed, or the result is uninterpretable. Add a multi-hop leg (MuSiQue / 2WikiMHQA) to test G8's traversal claim directly, since that is where the wiki-shaped gains concentrate.

---

## 6. Where Akh is ahead of the LLM Wiki pattern

An adversarial reading owes an honest inverse. On several axes the whole 2026 wiki cluster — Karpathy's pattern included — is *behind* Akh, and these should not be traded away while closing the gaps above.

| Capability | LLM Wiki pattern | Akh |
|---|---|---|
| **Provenance** | Absent from the gist; Hermes adds an optional `^[source]` markdown convention | Every triple and block carries `DerivationKind` + provenance chain **[verified]** |
| **Conflict** | "noting where new data contradicts old claims" — prose flagging; Hermes adds a `contested: true` boolean | Dempster–Shafer belief intervals weighted by source reliability (Phases 17 + 18), `src/schema/conflict.rs` **[verified]** |
| **Retraction** | Overwrite the file; git history is the only record | TMS with `was_retracted`, history preserved, diff-scoped supersede **[verified]** |
| **Contexts** | One flat namespace | Microtheories with `ctx:specializes` import stacks — schema packs *are* typed microtheories |
| **Schema rot** | Unaddressed — the recurring community complaint | Structurally solved: globally-unique immutable additive type names make conflict impossible by construction |
| **Multi-agent** | Single-user git repo (WUPHF adds a team layer) | ActivityPub federation + workspace membership + shared/own microtheory scoping |
| **Consolidation cadence** | Human-triggered lint | Daily batch Librarian + VSA sleep — the regime 2605.12978 measures as *safer* than the per-interaction updates that CLIN/AWM/ACE use |
| **Governance** | Absent (which is exactly why 2604.12034 was written) | Human-gated registry with compiled, checksummed, versioned packs |

Two of these deserve emphasis because the literature independently converged on them:

- **ADR 081 decision 5's immutable-additive type system is a genuinely better answer to schema rot than anything in the wiki cluster.** The HN thread's central complaint — "how do you keep LLMs from writing too much… it looked good until you read the actual data" — is R5, and Akh solved it structurally rather than by policy. That is a real contribution, not a defensive posture.
- **ADR 081 decision 2's permissive-truth-plus-projection is exactly what 2601.00821 recommends** ("structure that annotates verbatim text… preserves a lossless path from question to source"). Akh reached the right architecture *before* the paper measured it. The gap in G4 is that the read path does not exploit the architecture the write path earned.

---

## 7. Consolidated gap table

| # | Gap | Sev | Evidence | Effort |
|---|---|---|---|---|
| G1 | No lint/audit pass over notes | **P0** | Error Book −3.4/−4.0 F1; AUDIT vs. ossification | M (Layer 1 = S) |
| G2 | Query answers never file back | **P0** | Breaks the pattern's compounding claim | M |
| G4 | Read path served from extracted side | **P0** | −15.9 / −22.0 pts (2601.00821) | S–M |
| G6 | R2 empirical bet unmeasured | **P0** | Format/Constraint Tax; Tool Suppression | S |
| G3 | No index/log surface | P1 | Index-first navigation; browse-first traversal | S |
| G5 | Ungated, unbatched, never re-baselined consolidation | P1 | Streaming −17/−38 pts (2605.12978) | M |
| G7 | Akhipedia orphaned + duplicate note carrier | P1 | Verified: zero tool registrations | S (decision) / M (merge) |
| G8 | No traversal interface | P1 | Traversal ablation −11.7/−13.8 F1 | M |
| G9 | No retention/decay/entrenchment policy | P1 | Memory as Metabolism; Library Drift | M–L |
| G11 | No temporal validity; implicit conflict blind | P1 | STALE; AUROC 0.59; Zep 7% | M |
| G12 | No memory evaluation harness | P1 | All §4 claims are benchmark numbers | M |
| G10 | Consensus channel absent; human gate under-invested | P2 | LLM-authored skills +0.0pp | S |

---

## 8. Recommendation

Ordered by evidence strength × cost, not by gap number.

**Do first — cheap, and each retires a specific risk:**

1. **G6 — run the R2 probe**, split into vocabulary / format-under-load / tool-suppression channels. Nine slices are standing on it, and it is roughly a day.
2. **G1 Layer 1 — deterministic lint.** Dangling `links-to`, orphan notes, shape violations, unresolved types. All four are graph queries over triples `src/schema/project.rs` already emits. No LLM, no new storage.
3. **G7 — decide the substrate question** and record it in a short ADR. Note = memory, Akhipedia = publication/federation surface over Notes. Then register or delete the four orphaned tools.
4. **G12 — wire LoCoMo + LongMemEval-S into the existing 44j harness.** Everything below is currently un-adjudicable without it, and Phase 50 already scopes these legs.

**Then — the two changes most likely to move measured quality:**

5. **G4 — span-annotated recall.** Make recall return bounded verbatim spans from note bodies, annotated by frontmatter and typed triples, instead of frontmatter alone. R4 is preserved because only the selected span resides. Run it as a two-arm A/B on the harness from (4). This is the single most evidence-backed change in the document.
6. **G2 — Librarian-mediated answer filing.** Route completed deliberations through the daily Librarian for note synthesis. Respects decision 10's rejection of small-model `take_note` while restoring the compounding edge that distinguishes the pattern from RAG.

**Then — structural, and worth a plan of their own:**

7. **G8 + G5** together: traversal tools (`note_search` / `note_read` with links as affordances) wired *into* the ThinkingLoop, plus periodic full re-projection and family-grouped batching. The composition of Akh's measured depth-scaling with wiki traversal's depth-scaling is the genuinely novel bet here, and it needs the multi-hop benchmark leg from (4) to be legible.
8. **G9 + G11**: gravity, AUDIT-by-suspension, decay-not-deletion, temporal validity intervals. Largest and least certain — 2604.12034 is a vision paper with no implementation results, and 2606.01435 shows the whole field is at 7–54% on temporal conflict. Sequence after (4) exists so the work can be measured rather than asserted.

**Do regardless of all of the above:** amend ADR 081's References with the gist URL, a retrieval date, and the corrected characterization from §2. An undated, un-URL'd citation for a load-bearing design input is a reproducibility defect independent of whether the design is right.

---

## 9. Cross-check against open Anima work items

Added 2026-07-28 after the gap analysis was written, so the mapping is a genuine cross-check rather than a design input. Project `019d9844-ce74-7993-a9f0-c953b8f04a4e`; **49 open items** (33 parked, 13 draft, 1 working, 2 testing, 0 ready) out of 261 total.

**Headline: Phase 42 (Heka Long-Term Memory) is a parked plan that already covers roughly half of this gap analysis — but it was designed 2026-04-27, three months before the evidence in §4, and two of its acceptance criteria are pointed the wrong way.**

### 9.1 Heka's actual state [verified]

`crates/heka/src/` = `config.rs`, `entity/` (5 entities), `error.rs`, `migration/`, plus a 91-line `bin/hekad.rs`. There is no `eval/`, no `retrieve/`, no `mcp/`, no `extract/`. That is **42a only**; 42b–42h are unstarted. The 42g description's audit note claims heka contains "drawer storage + the `contradict` MCP tool" — grep for `contradict` across `crates/` hits only `crates/grug-data/src/reason/corpus.rs`, so even that note is optimistic.

### 9.2 Mapping

| Gap | Open WI | State | Verdict |
|---|---|---|---|
| **G12** eval harness | **42b — LongMemEval + LoCoMo harness with honest reporter** (high, parked) | unstarted | **Already correct. Unpark unchanged.** It is recommendation §8(4) verbatim, and it even pre-solves a gaming vector this review did not raise: a `min(5, conv_len/4)` auto-cap on top-k "to refuse the MemPalace `top_k=50`-on-30-items exploit by construction," with a loud warning on explicit override. Dependency root for 42c→42d→42e→42g. |
| **G4** read path | **42d — Hybrid retrieval (pgvec + VSA + KG + recency), RRF fusion** (high, parked) | unstarted | **Amend before building — see 9.3.** |
| **G8** traversal | 42d's graph channel (SPARQL traversal query-symbols → drawers) | unstarted | **Partial only.** 42d does *fused single-shot ranking*; arXiv:2605.25480 ablates exactly that against agent-planned traversal and finds −11.7/−13.8 F1. No open WI proposes `note_read` with links-as-affordances. **Genuinely uncovered.** |
| **G1** lint | **42e — `heka_contradict`** (Phase 18 ACH over two facts) + **`heka_explain`** (provenance chain) | unstarted | Complements, does not replace. `heka_contradict` is per-pair on demand; G1 Layer 2 needs a *sweep*. `heka_explain` is the right surface for Layer 1's output. |
| **G1** lint tooling | **MCP: provenance_filter — triples by derivation source/kind, paginated** (high, parked) | unstarted | Exactly the query surface G1 Layer 1 needs. Fold into the lint WI rather than shipping standalone. |
| **G7** Akhipedia | **42g — Akhipedia auto-extraction from drawers** (medium, parked; previously a *false* "done", reset 2026-06-02, Forgejo #230) | hollow scaffold | **Re-scope before reviving.** 42g gives Akhipedia a *write* path but not tool registration, so the akh still could not query it. Worse, it predates ADR 081's Note carrier and would deepen the duplicate-substrate problem. Resolve G7's substrate question first. |
| **G3** index/catalog | **MCP: CRUD for agent goals + expose entity schemas/types** (high, parked) | unstarted | Partial — exposes *schemas*, not a content catalog. The index-as-map function remains uncovered. |
| **G11** supersession | **Add kg_delete / triple-delete to MCP tools** (high, parked) | unstarted | **Scope hazard.** Deletion ≠ supersession. A hard-delete verb would destroy the provenance advantage §6 credits Akh with. Scope it as *retract-via-TMS* (`was_retracted`, history preserved), and pair it with validity intervals per G11 and decay-not-deletion per G9. |
| **G2** answers file back | **Anima integration — workspace daemon pulling session transcripts from Seshat** (high, parked); ADR 091 (done) | unstarted | Transcript → *episodic* only. Nothing turns a good answer into a concept page. **Genuinely uncovered.** |
| **G6** R2 bet | — | — | **No WI exists.** Largest uncovered P0. |
| **G5** consolidation gating | — | — | **No WI exists.** |
| **G9** retention / gravity / audit | — | — | **No WI exists.** |

### 9.3 The one direct contradiction — amend 42d before it is built

42d's `budget.rs` deliverable specifies, verbatim:

> "token-budget assembler. **Prefers Phase 22a article-block summaries over verbatim where available; falls back to verbatim.**"

That is precisely backwards under arXiv:2601.00821. Preferring distilled artifacts over the source text they replace is the configuration measured at **−15.9 pts (LoCoMo) / −22.0 (LongMemEval-S)**, and the paper's union-store control shows the *combination* only pays when verbatim is actually retrieved. 42d as written would hard-code the losing preference into the assembler and then measure it with 42b.

Two amendments, both cheap because they land before implementation:

1. **Invert the default, or better, parameterize it.** Make summary-vs-verbatim preference an A/B knob on the 42b harness rather than a fixed policy. This is the same experiment G4 recommends, and 42b+42d together are the natural place to run it — no separate harness needed.
2. **Add an end-to-end accuracy target beside R@5.** 42d's acceptance is "≥95% R@5 (within striking distance of MemPalace 96.6%)". R@5 is a *retrieval* metric; every number in §4 is end-to-end QA accuracy. A summary-preferring assembler can hit high R@5 and still lose on answers, because the retrieved unit no longer contains the surface detail the answer needs. R@5 alone cannot detect the failure this review predicts.

### 9.4 Strong convergence — ADR 094 is Karpathy's third layer

**ADR 094 — operating-guidance documents: a composable, testable context-injection layer for tool doctrine** (high, draft, filed 2026-07-27) independently reinvents the schema layer of §1.2. Karpathy: the schema doc "is what makes the LLM a disciplined wiki maintainer rather than a generic chatbot." ADR 094: an akh "has tool schemas but is never told *when* to reach for a tool, under what preferences, or with what terminology. Schemas do not teach doctrine."

Same diagnosis, arrived at from tool-use rather than knowledge-maintenance. And ADR 094 has the property Karpathy's prose schema lacks: it is **testable**, wired to the same `src/bench_phase44/` harness that G4 and G12 want.

**Consequence for this gap analysis:** the *conventions* half of G1/G2/G3 — when to lint, what counts as noteworthy enough to file back, how to name and link a page — should ship as `doctrine/` documents under ADR 094, not as new code. Only the *mechanisms* need code (Layer-1 deterministic lint queries, the Librarian answer-filing hook, the catalog projection). That collapses a meaningful fraction of the P0 work into a mechanism already designed and awaiting six human decisions.

Worth noting for whoever resolves ADR 094's open question 1 (section placement): wiki-maintenance doctrine is *not* tool-proximate the way "use glob not find" is, so the leading-block placement is likely right for it.

### 9.5 Independent field corroboration of G4

**"Sessions report the wrong work item: teach the akh to orient on the brief, not recall from memory"** (high, parked, 2026-07-26) documents two production sessions that narrated work against the wrong WI. Its stated direction:

> "anything you believe about this task that you did not read from the brief or the worktree in this session is a **memory, not a fact** — re-read the brief before acting on it."

That is the behavioural statement of §4.2's finding — prefer the source over the recalled distillate — arrived at from a production incident rather than a benchmark. The WI is careful to note that KG leakage is *not* the demonstrated mechanism for these two cases (both ran on the `claude-code` executor, which has no KG), so it is evidence about orientation-from-context generally rather than about Akh's KG specifically. Still, it is the same failure shape, and it strengthens the G4 case from an independent direction.

### 9.6 Cross-cutting blockers

- **"17 work items frozen since 2026-07-23: agent_diff set blocks dispatch"** (critical, `testing`) — nothing dispatches until this clears. Every recommendation below is gated on it.
- **"Sessions on one akh still serialise on the per-workspace Agent mutex"** (critical, draft) — parallel ACP sessions impossible; constrains how fast any of this can be worked.
- **"Adversarial review — Heka Long-Term Memory (42a–42h) gap analysis"** (medium, parked) — this document is a substantial fraction of that WI's deliverable. Recommend attaching it to the parent plan's thread rather than running the review from scratch.
- Phase 42's `updated_at` is 2026-07-27 (a bulk touch); `created_at` is 2026-04-27. The design predates every paper in §4.

### 9.7 What to do with the backlog

1. **Unpark 42b unchanged.** It is G12, it is the dependency root, and its anti-gaming design is better than what §8(4) proposed.
2. **Amend 42d's two acceptance criteria** (9.3) while it is still parked — the cheapest correction in this document.
3. **Fold wiki-maintenance conventions into ADR 094's doctrine corpus** instead of filing separate WIs for them; resolve ADR 094's six open questions.
4. **File three new WIs** for what nothing covers: **G6** (the R2 probe, three channels), **G5** (consolidation gate + periodic full re-projection + family batching), **G8** (`note_search` / `note_read` traversal wired into the ThinkingLoop).
5. **Re-scope before reviving:** 42g against ADR 081's Note carrier; `kg_delete` as TMS-retract rather than hard delete.
6. **Attach this document** to the Heka adversarial-review WI (`019f85b9-c9a9`).

---

## 10. References

### Primary
- Karpathy, A. *LLM Wiki* (`llm-wiki.md`). GitHub Gist `442a6bf555914893e9891c11519de94f`. Retrieved verbatim 2026-07-28.

### Landscape
- Ming, H., Li, F., Wu, X., Que, W. (Tencent WeChat). *Retrieval as Reasoning: Self-Evolving Agent-Native Retrieval via LLM-Wiki*. [arXiv:2605.25480](https://arxiv.org/abs/2605.25480)
- Miteski, S. *Memory as Metabolism: A Design for Companion Knowledge Systems*. [arXiv:2604.12034](https://arxiv.org/abs/2604.12034)
- zby. *Designing a Memory System for LLM-Based Agents*; *Agent Memory Systems* review collection. `zby.github.io/commonplace/`
- WUPHF (nex-crm) — [HN 47899844](https://news.ycombinator.com/item?id=47899844); Hermes bundled `research-llm-wiki` skill (Nous Research); `qmd` (tobi)

### Extraction vs. fidelity
- An, T. *Verbatim Chunks Beat Extracted Artifacts: A Controlled Ablation of Memory Representations for Long LLM Conversations*. [arXiv:2601.00821](https://arxiv.org/abs/2601.00821)
- *Storage Is Not Memory: A Retrieval-Centered Architecture for Agent Recall*. [arXiv:2605.04897](https://arxiv.org/abs/2605.04897)
- *MemMachine: A Ground-Truth-Preserving Memory System*. [arXiv:2604.04853](https://arxiv.org/abs/2604.04853)
- *Infini Memory: Maintainable Topic Documents for Long-Term LLM Agent Memory*. [arXiv:2606.10677](https://arxiv.org/abs/2606.10677)

### Consolidation drift
- Zhang, D., Lin, Y., Wu, Z., Sun, Y., Li, B., Li, D., Peng, H. *Useful Memories Become Faulty When Continuously Updated by LLMs*. [arXiv:2605.12978](https://arxiv.org/abs/2605.12978)
- *Library Drift: Diagnosing and Fixing a Silent Failure Mode in Self-Evolving LLM Skill Libraries*. [arXiv:2605.19576](https://arxiv.org/abs/2605.19576)
- *On the Limits of Self-Improving in LLMs: The Singularity Is Not Near Without Symbolic Model Synthesis*. [arXiv:2601.05280](https://arxiv.org/abs/2601.05280)
- *The Mirror Loop: Recursive Non-Convergence in Generative Reasoning Systems*. [arXiv:2510.21861](https://arxiv.org/abs/2510.21861)
- *Knowledge Collapse in LLMs*. [arXiv:2509.04796](https://arxiv.org/abs/2509.04796)
- *LLM as a Broken Telephone: Iterative Generation Distorts Information*. [arXiv:2502.20258](https://arxiv.org/abs/2502.20258)

### Structured output in small models (R2)
- *The Format Tax*. [arXiv:2604.03616](https://arxiv.org/abs/2604.03616)
- *The Constraint Tax: Measuring Validity-Correctness Tradeoffs in Structured Outputs for SLMs*. [arXiv:2605.26128](https://arxiv.org/abs/2605.26128)
- *When Correct Isn't Usable: Improving Structured Output Reliability in SLMs*. [arXiv:2605.02363](https://arxiv.org/abs/2605.02363)
- *Constraint Tax in Open-Weight LLMs: Tool Calling Suppression Under Structured Output Constraints*. [arXiv:2606.25605](https://arxiv.org/abs/2606.25605)
- *JSONSchemaBench*. [arXiv:2501.10868](https://arxiv.org/abs/2501.10868)
- *Small Language Models for Agentic Systems: A Survey* (counterweight). [arXiv:2510.03847](https://arxiv.org/abs/2510.03847)

### Temporal conflict and staleness
- *Temporal Validity in Retrieval Memory*. [arXiv:2606.26511](https://arxiv.org/abs/2606.26511)
- *Don't Ask the LLM to Track Freshness*. [arXiv:2606.01435](https://arxiv.org/abs/2606.01435)
- *STALE: Can LLM Agents Know When Their Memories Are No Longer Valid?* [arXiv:2605.06527](https://arxiv.org/abs/2605.06527)
- *TOKI: A Bitemporal Operator Algebra for Contradiction Resolution*. [arXiv:2606.06240](https://arxiv.org/abs/2606.06240)
- *Presentation, Not Mechanism: A Render Confound in Deprecation-Aware Memory Evaluation*. [arXiv:2607.16019](https://arxiv.org/abs/2607.16019)

### Benchmarks and agentic memory background
- Maharana et al. *Evaluating Very Long-Term Conversational Memory of LLM Agents* (LoCoMo). [arXiv:2402.17753](https://arxiv.org/abs/2402.17753)
- Wu et al. *LongMemEval*. [arXiv:2410.10813](https://arxiv.org/abs/2410.10813)
- Xu et al. *A-MEM: Agentic Memory for LLM Agents*. [arXiv:2502.12110](https://arxiv.org/abs/2502.12110)
- Shao et al. *Assisting in Writing Wikipedia-like Articles From Scratch* (STORM). [arXiv:2402.14207](https://arxiv.org/abs/2402.14207)
- *Oblivion: Self-Adaptive Agentic Memory Control through Decay-Driven Activation*. [arXiv:2604.00131](https://arxiv.org/abs/2604.00131)

### Human oversight
- *From Verification Burden to Trusted Collaboration*. [arXiv:2512.11661](https://arxiv.org/abs/2512.11661)
- *Trust-Calibrated Code Review*. [arXiv:2606.01969](https://arxiv.org/abs/2606.01969)

### Akh-Medu internal
- [ADR 081](../decisions/081-akh-memory-schema-typing.md) · [plan](../plans/2026-06-30-akh-memory-schema.md) · [work breakdown](../plans/2026-06-30-akh-memory-schema-issues.md)
- [ADR 021](../decisions/021-akhipedia-architecture.md) · [ADR 057](../decisions/057-code-understanding-kg.md) · [ADR 065](../decisions/065-context-assembly-runtime.md) · [ADR 052](../decisions/052-unified-thinking-loop.md) · [ADR 085](../decisions/085-thinking-engine-findings.md) · [ADR 086](../decisions/086-feature-test-vm-harness-benchmarks.md) · [ADR 091](../decisions/091-transcript-episodic-capture.md)
- Code verified at `06d400f`: `src/wiki/`, `src/library/note.rs`, `src/schema/{project,conflict,coverage}.rs`, `src/agent/agent.rs`, `src/agent/sleep.rs`, `src/bin/akhomed.rs`, `src/bench_phase44/`
