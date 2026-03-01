# ADR-002: Code Generation Research — Neuro-Symbolic Patterns, VSA Code Representation, and Library Learning

- **Date**: 2026-02-17
- **Status**: Accepted
- **Context**: Deep research for Phase 10 optimization, specifically how to integrate VSA, KG, and e-graph capabilities for pattern-based code generation

## Research Question

Can we learn code patterns from examples (blog posts, tutorials, library code) and use VSA/KG/e-graph to generate new code following those patterns — without LLM dependency?

## Key Findings

### 1. DreamCoder/LILO Wake-Sleep Library Learning (Most Directly Applicable)

**DreamCoder** (PLDI 2021) iteratively builds a domain-specific programming language through three phases:
- **Wake**: Synthesize programs for tasks using current library + neural guide
- **Abstraction Sleep**: Analyze solved programs via e-graph matching, extract common sub-components as new library functions (lambda abstractions)
- **Dreaming Sleep**: Generate synthetic tasks to bootstrap the recognition model

**LILO** (ICLR 2024) extends this with:
- **Stitch compression**: Branch-and-bound search for optimal lambda abstractions across a corpus. 3-4 orders of magnitude faster than DreamCoder's compression. Maximizes `occurrences * length`.
- **AutoDoc**: Human-readable names and docs for discovered abstractions — critical for reuse

**Mapping to akh-medu**: Wake = OODA cycle, Abstraction = consolidation phase with e-graph, Library = KG entities with provenance, Fast retrieval = VSA encoding of library signatures in HNSW.

### 2. babble: E-Graph Anti-Unification for Library Learning (POPL 2023)

Uses e-graphs to represent programs modulo an equational theory, then applies *anti-unification* on e-classes to discover common abstractions. Anti-unification finds "the most concrete pattern that matches two given terms." Achieves better compression orders of magnitude faster than prior approaches.

**Mapping**: The existing egg integration handles the e-graph. Anti-unification over e-classes discovers reusable code patterns. KG stores discovered patterns with provenance. VSA encodes pattern signatures for retrieval.

### 3. VSA Code Representation (Non-ML code2vec)

**code2vec** (POPL 2019) decomposes code into AST path-contexts: `(start_token, AST_path, end_token)`. The critical insight for akh-medu: this maps directly to VSA role-filler bindings without any neural network:

```
path_context = bind(start_token_vec, bind(path_vec, end_token_vec))
code_vec = bundle(path_context_1, path_context_2, ..., path_context_N)
```

With 10k-bit binary vectors:
- AST node types as atomic symbols in item memory
- AST paths encoded via `encode_sequence()` with permutation shifts
- Path-contexts as role-filler bindings via `encode_role_filler()`
- Full code vector as majority-vote bundle of all path-contexts
- Searchable via existing HNSW index with Hamming distance

**Capacity**: ~70-100 path-contexts can be reliably bundled. Recursive binding ~5-7 levels deep. HNSW accuracy >95% for databases up to millions of entries.

### 4. Frequent AST Subtree Mining (FREQT/FREQTALS)

Mine recurring structural patterns by treating code as ASTs and applying frequent subtree mining:
1. Parse source code into ASTs
2. Enumerate candidate subtrees via rightmost path extension
3. Filter by minimum support threshold
4. Apply maximality constraint (only keep patterns not subsumed by larger ones)

**Token Sugar** (2025) applied this specifically for code patterns: mined 799 pattern-shorthand pairs, achieving 15.1% token reduction with identical functional accuracy.

**Mapping**: Mined patterns become KG entities. Pattern structural signatures encode as VSA vectors via `encode_sequence()`. E-graph captures pattern equivalences (syntactically different but semantically identical patterns).

### 5. API Usage Pattern Mining (MAPO)

Discovers typical API calling sequences from code corpora:
1. Collect code using target APIs from repositories
2. Cluster by API method usage similarity
3. Extract frequent call subsequences per cluster
4. Partial orders capture must-precede vs. unordered relationships

**Mapping**: API methods as KG entities, usage patterns as ordered step triples, sequences encoded via VSA `encode_sequence()` with permutation. Partial order constraints become e-graph rewrite rules.

### 6. Program Analogies via VSA (A:B :: C:?)

VSA natively supports analogical reasoning through bind/unbind algebra:
```
transform = bind(pattern_A, pattern_B)    // extract what changed
pattern_D = bind(transform, pattern_C)    // apply change to new context
result = HNSW_search(pattern_D)           // clean up noisy result
```

This enables: "do for type X what we did for type Y" — encode both patterns, compute the transformation vector, apply to the new context, search item memory for the nearest known result.

**Resonator networks** extend this to decomposing compound vectors into multiple unknown factors simultaneously — useful for decomposing a complex code pattern into its constituent components.

### 7. Graph-Level Code Classification (GraphHD/VS-Graph)

Encode entire program graphs (AST + control flow + data flow) as single hypervectors for classification:
- **GraphHD**: PageRank-based node signatures, edge encoding via binding, graph via bundling
- **VS-Graph** (2024): Spike diffusion + associative message passing, GNN-level accuracy at 250x faster training

**Mapping**: Classify code structures ("this looks like a builder pattern") without neural networks. The KG provides the graph, VSA encodes it, HNSW finds the nearest class prototype.

### 8. E-Graph Innovations

- **egglog** (PLDI 2023): Combines equality saturation with Datalog for relational analysis on e-graphs. Could unify SPARQL queries and e-graph operations.
- **babble** (POPL 2023): Anti-unification on e-classes for library learning (see #2 above).
- **Guided Equality Saturation** (POPL 2024): Priority-directed saturation guided by external signals — the agent's utility scoring could guide which rewrites to prioritize.
- **Rewrite Rule Inference** (OOPSLA 2021): Automatically discover rewrite rules by enumerating terms and finding equalities — the agent could learn optimization rules from code examples.

### 9. CEGIS Pattern (Counter-Example Guided Inductive Synthesis)

Alternates between synthesis and verification:
1. Synthesize candidate from DSL/templates
2. Verify against specification (tests, types, compiler)
3. If verification fails, use counterexample to refine

**Mapping**: The OODA loop already implements this pattern — Observe task, Orient/Decide to synthesize, Act to verify via `shell_exec(cargo check)`, reflect on failures to guide next attempt.

## Decision

Enhance Phase 10 with three research-informed additions:

### 10f — VSA Code Pattern Encoding (New)
Non-ML code2vec analog using AST path-context encoding in 10k-bit binary vectors. Enables similarity-based pattern retrieval via HNSW without any neural network training.

### 10g — Pattern Learning from Examples (New)
Learn-from-blog-post pipeline: ingest code examples -> mine frequent AST patterns -> encode as VSA vectors -> store as KG entities -> retrieve via analogy for new generation tasks. Combines FREQT mining, VSA encoding, and KG storage.

### 10h — Library Learning Cycle (New)
DreamCoder/LILO-inspired wake-sleep cycle using e-graph compression (babble/Stitch-style anti-unification) to discover reusable abstractions from generated code. Abstractions become named KG entities with VSA signatures for fast retrieval.

## The Blog-Post-to-Code Pipeline

The user's specific request — "train a pattern from a blog post and build code like that pattern" — maps to:

```
Blog Post ──→ Library Ingest ──→ Code Examples (text chunks)
                                        │
                                        ▼
                                  syn Parse ──→ AST
                                        │
                                        ▼
                              FREQT Mining ──→ Frequent AST Patterns
                                        │
                                        ▼
                        VSA Encoding ──→ 10k-bit Pattern Vectors
                                        │
                                        ▼
                     KG Storage ──→ Pattern Entities with Triples
                                        │
                                        ▼
                  New Task ──→ VSA Analogy (A:B::C:?) ──→ Code Gen
                                        │
                                        ▼
                              E-graph ──→ Optimize/Simplify
                                        │
                                        ▼
                         RustCodeGrammar ──→ Linearize to Source
                                        │
                                        ▼
                           cargo check ──→ Verify ──→ Refine if needed
```

## Sources

### Neuro-Symbolic Program Synthesis
- [DreamCoder (PLDI 2021)](https://dl.acm.org/doi/10.1145/3453483.3454080)
- [LILO (ICLR 2024)](https://arxiv.org/abs/2310.19791)
- [Stitch Compression](https://dl.acm.org/doi/10.1145/3571234)
- [babble: E-Graph Anti-Unification (POPL 2023)](https://arxiv.org/abs/2212.04596)
- [AbstractBeam (2024)](https://arxiv.org/abs/2405.17514)
- [Scallop: Differentiable Datalog (PLDI 2023)](https://dl.acm.org/doi/10.1145/3591280)
- [Proof of Thought (NeurIPS 2024 Workshop)](https://arxiv.org/abs/2409.17270)

### E-Graph and Equality Saturation
- [egg (POPL 2021)](https://arxiv.org/abs/2004.03082)
- [egglog (PLDI 2023)](https://github.com/egraphs-good/egglog)
- [Guided Equality Saturation (POPL 2024)](https://steuwer.info/files/publications/2024/POPL-Guided-Equality-Saturation.pdf)
- [Rewrite Rule Inference (OOPSLA 2021)](https://arxiv.org/pdf/2108.10436)
- [ASPEN: LLM-Guided E-Graph Rewriting (2025)](https://www.csl.cornell.edu/~zhiruz/pdfs/aspen-mlcad2025.pdf)
- [DialEgg: MLIR + egglog (CGO 2025)](https://dl.acm.org/doi/10.1145/3696443.3708957)

### VSA for Code and Structured Data
- [VSA Survey Part I (ACM Computing Surveys)](https://dl.acm.org/doi/10.1145/3538531)
- [VSA Survey Part II (ACM Computing Surveys)](https://dl.acm.org/doi/10.1145/3558000)
- [GraphHD (2022)](https://arxiv.org/abs/2205.07826)
- [VS-Graph (2024)](https://arxiv.org/html/2512.03394)
- [Resonator Networks (MIT Press)](https://direct.mit.edu/neco/article/32/12/2332/95653)
- [HDC as Computing Framework (2024)](https://link.springer.com/article/10.1186/s40537-024-01010-8)

### Code Pattern Mining
- [Token Sugar: AST Pattern Mining (2025)](https://arxiv.org/abs/2512.08266)
- [FREQTALS: Frequent AST Subtree Mining](https://link.springer.com/chapter/10.1007/978-3-030-33778-0_35)
- [MAPO: API Usage Pattern Mining](https://link.springer.com/chapter/10.1007/978-3-642-03013-0_15)
- [code2vec (POPL 2019)](https://dl.acm.org/doi/10.1145/3290353)
- [GraphGen4Code: Code Knowledge Graphs](https://wala.github.io/graph4code/)

### Code Generation
- [KG-Based Repository-Level Code Generation (ICSE 2025)](https://dl.acm.org/doi/10.1145/3691620.3695054)
- [Neurosymbolic SE Workshop (ICSE 2025)](https://arxiv.org/abs/2505.02275)
- [Code Property Graphs](https://arxiv.org/html/2404.14719v1)

## Consequences

- Phase 10 expands from 5 to 8 sub-phases (10a-10h)
- New VSA encoding functions needed: `encode_ast_path_context()`, `encode_ast_tree()`, `encode_code_graph()`
- Pattern mining requires a new tool or extension of `code_ingest`
- Library learning cycle integrates with agent consolidation (Phase 8f reflection)
- The blog-post pipeline connects library ingest (existing) -> code_ingest (existing) -> new pattern mining -> existing VSA/KG storage -> new analogy-based generation
- E-graph anti-unification (babble-style) is a new primitive for the reason module
- Estimated additional scope for 10f-10h: ~1,500-2,200 lines
- Total Phase 10 scope revised: ~3,800-5,500 lines across 8 sub-phases
