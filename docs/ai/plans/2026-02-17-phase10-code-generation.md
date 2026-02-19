# Phase 10 — Generative Functions (Rust Code Generation)

- **Date**: 2026-02-17
- **Updated**: 2026-02-19 (Wave 3 complete: 10g pattern mining + 10h library learning)
- **Status**: Complete (all 8 sub-phases done)
- **Depends on**: Phase 9 (partially — 9b predicate hierarchy and 9f reasoner dispatch are most useful but not blocking)
- **Research**: `docs/ai/decisions/002-code-generation-research.md`

## Goal

Enable akh-medu to generate valid, idiomatic Rust code from its knowledge graph. The agent should be able to: query code structure from the KG, plan a code artifact, generate it through the grammar system, write it to disk, and validate it with the Rust toolchain — all autonomously through the OODA loop.

Beyond basic generation, the system should **learn code patterns from examples** (blog posts, tutorials, library code) using VSA encoding, store them in the KG, and reuse them via analogy-based retrieval — without LLM dependency.

## Existing Building Blocks

The system is ~80% ready. Key foundations already in place:

| Component | Location | What It Does |
|-----------|----------|-------------|
| `AbsTree::CodeSignature` | `grammar/abs.rs` | AST node for fn/struct/enum/trait with params, return type, traits, doc |
| `AbsTree::CodeModule` | `grammar/abs.rs` | AST node for module with children, role, importance, doc summary |
| `AbsTree::DataFlow` | `grammar/abs.rs` | AST node for directed data flow chains |
| `ConcreteGrammar` trait | `grammar/concrete.rs` | `linearize(AbsTree) → String` and `parse(String) → AbsTree` |
| `GrammarRegistry` | `grammar/mod.rs` | Runtime grammar selection with `register()` |
| `code_ingest` tool | `agent/tools/code_ingest.rs` | Parse Rust with `syn`, extract items as KG triples (`code:*` predicates) |
| `doc_gen` tool | `agent/tools/doc_gen.rs` | Query KG for code structure, render docs |
| `file_io` tool | `agent/tools/file_io.rs` | Write files to scratch dir |
| `shell_exec` tool | `agent/tools/shell_exec.rs` | Run `cargo check`, `rustfmt`, `clippy` |
| `synthesize_abs.rs` | `agent/tools/` | Convert KG facts → `AbsTree::Document` with code sections |
| `SkillManager` | `skills/` | Package capabilities as loadable skillpacks |
| Plan/OODA loop | `agent/plan.rs`, `agent/ooda.rs` | Multi-step goal execution with backtracking |
| VSA encoding | `vsa/encode.rs` | `encode_sequence()`, `encode_role_filler()`, `encode_token()` |
| Item Memory | `vsa/item_memory.rs` | HNSW nearest-neighbor search with Hamming distance |
| Grounding | `vsa/grounding.rs` | Aligns VSA vectors with KG structure via iterative binding |
| Library ingest | `library/` | PDF/EPUB/HTML → chunks with semantic enrichment |

## What's Missing

**Core generation (10a–10e)**:
1. **`RustCodeGrammar`** — a `ConcreteGrammar` impl that linearizes `AbsTree` code nodes into valid Rust source
2. **`code_gen` tool** — agent tool that orchestrates: query KG → build AbsTree → linearize → format → validate
3. **Code-aware planning** — plan generation recognizes code goals and produces code_gen steps
4. **Iterative refinement** — compile error → parse error → fix → retry loop
5. **Template library** — common Rust patterns (error types, trait impls, builders) as reusable AbsTree templates

**Pattern learning (10f–10h)** — identified by deep research:
6. **VSA code pattern encoding** — non-ML code2vec analog using AST path-context bindings in 10k-bit vectors
7. **Pattern mining from examples** — learn recurring AST patterns from blog posts / tutorials / library code
8. **Library learning cycle** — DreamCoder/LILO-inspired wake-sleep abstraction discovery via e-graph anti-unification

---

## Phase 10a — RustCodeGrammar

**Problem**: The grammar system can represent code structure (AbsTree) but has no linearizer that produces valid Rust source.

**Design**:
- Implement `ConcreteGrammar` for Rust code generation
- `linearize()` handles all code-related `AbsTree` variants:
  - `CodeSignature { kind: "fn" }` → function definition with signature, doc comment, `todo!()` body
  - `CodeSignature { kind: "struct" }` → struct definition with fields, derives, doc
  - `CodeSignature { kind: "enum" }` → enum with variants
  - `CodeSignature { kind: "trait" }` → trait definition with method signatures
  - `CodeSignature { kind: "impl" }` → impl block with methods
  - `CodeModule` → `mod` block with nested items
  - `DataFlow` → pipeline comment block or iterator chain
- `parse()` delegates to `syn` to round-trip: Rust source → `AbsTree`
- Register as `"rust-gen"` archetype in `GrammarRegistry`

**Changes**:
- [x] New file: `src/grammar/rust_gen.rs` (~980 lines, 14 tests)
- [x] `src/grammar/mod.rs` — register `RustCodeGrammar` as built-in archetype
- [x] Handle: derives, visibility (`pub`), indentation and formatting (pre-rustfmt)
- [x] Parse via `syn` for round-trip: Rust source → `AbsTree`
- [ ] Handle: generics, lifetimes, `where` clauses (deferred to 10e templates)
- [ ] Handle: `use` statements from `code:depends-on` triples (deferred to 10e templates)

**Estimated scope**: ~500–700 lines → ~980 lines

---

## Phase 10b — Code Generation Tool

**Problem**: No agent tool bridges "I know the structure" to "here's the code."

**Design**:
- New `CodeGenTool` implementing the `Tool` trait
- Input params:
  - `target` — SymbolId or name of the entity to generate code for
  - `scope` — `function | struct | enum | trait | module | file`
  - `output` — `string` (return code) or `file` (write to path)
- Execution pipeline:
  1. Query KG for `code:*` triples about target (reuse `doc_gen` query logic)
  2. Build `AbsTree` from KG facts (reuse `synthesize_abs` logic)
  3. Linearize through `RustCodeGrammar`
  4. Optionally run `rustfmt` via `shell_exec`
  5. Return code string in `ToolOutput`
- Provenance: `DerivationKind::CodeGenerated` with source symbols

**Changes**:
- [x] New file: `src/agent/tools/code_gen.rs` (~560 lines, 10 tests)
- [x] `src/agent/agent.rs` — register `CodeGenTool` in `register_builtin_tools()`
- [x] `src/agent/tool_semantics.rs` — add `code_gen` semantic profile
- [x] `src/provenance.rs` — `DerivationKind::CodeGenerated` (tag 37)
- [x] KG → AbsTree pipeline: auto-detect scope, build from code:* triples, collect source symbols
- [x] Optional `rustfmt` formatting, refinement analysis (`analyze_compiler_errors`)

**Estimated scope**: ~400–600 lines → ~560 lines

---

## Phase 10c — Code-Aware Planning

**Problem**: The plan generator doesn't know how to decompose code generation goals into steps.

**Design**:
- Detect code generation goals via VSA semantic matching + keyword heuristics ("generate", "implement", "write", "create function/struct/module")
- Code generation plan template:
  1. `kg_query` — gather existing code structure facts about the target
  2. `code_ingest` — if reference code exists, parse it first
  3. `kg_mutate` — define the target entity and its relations (params, return type, dependencies)
  4. `code_gen` — generate code from KG structure
  5. `file_io` — write to target path
  6. `shell_exec` — `cargo check` / `cargo clippy` / `cargo test`
  7. (on failure) → backtrack to step 3, adjust KG structure, retry
- Strategy alternation: "scaffold-first" (generate skeleton, fill in) vs "bottom-up" (generate helpers first, compose)

**Changes**:
- [x] `src/agent/plan.rs` — new `generate_code_plan()` function (~200 lines)
- [x] `src/agent/plan.rs` — `is_code_goal()` detection in `generate_plan()` with delegation
- [x] `src/agent/plan.rs` — two code strategies: scaffold-first vs bottom-up (alternated on backtrack)
- [x] `src/agent/plan.rs` — `extract_code_target()`, `detect_code_scope()`, `scope_to_type()` helpers

**Estimated scope**: ~300–500 lines → ~250 lines added

---

## Phase 10d — Iterative Refinement Loop

**Problem**: Generated code may not compile. The agent needs to parse errors and fix them.

**Design**:
- After `shell_exec(cargo check)`, parse compiler output for errors
- Error classification:
  - **Type mismatch** → adjust `code:return-type` or `code:param-type` in KG, regenerate
  - **Missing import** → add `code:depends-on` triple, regenerate `use` statements
  - **Syntax error** → likely grammar bug, log diagnostic, attempt fix
  - **Borrow checker** → adjust ownership annotations in KG, regenerate
  - **Missing trait impl** → generate the impl via `code_gen`
- Retry budget: max N attempts (default 3) before failing the goal
- Each attempt records provenance: `DerivationKind::CodeRefinement { attempt, error }`
- Compiler diagnostics stored in WM as `ToolResult` for episodic learning
- Implements a **CEGIS-like loop** (Counter-Example Guided Inductive Synthesis): each compiler error is a counterexample that guides the next synthesis attempt

**Changes**:
- [x] New file: `src/agent/tools/compile_feedback.rs` (~350 lines, 8 tests) — parse `cargo check` JSON output
- [x] `src/agent/tools/code_gen.rs` — `analyze_compiler_errors()` for refinement guidance
- [x] `src/provenance.rs` — `DerivationKind::CodeRefinement` (tag 38)
- [x] `src/agent/tool_semantics.rs` — `compile_feedback` semantic profile
- [x] `src/agent/agent.rs` — register `CompileFeedbackTool` in `register_builtin_tools()`
- [x] Code plans include `compile_feedback` validation step

**Estimated scope**: ~500–700 lines → ~350 lines + integration

---

## Phase 10e — Template Library

**Problem**: Generating code from scratch for every pattern is slow. Common Rust patterns should be reusable.

**Design**:
- `CodeTemplate` — parameterized `AbsTree` fragment with holes (placeholder symbols)
- Built-in templates:
  - **Error type**: `#[derive(Debug, Error, Diagnostic)]` enum with variants (from CLAUDE.md patterns)
  - **Trait impl**: `impl Trait for Type` with method stubs
  - **Builder pattern**: `TypeBuilder` with fluent setters + `build()`
  - **Iterator**: `impl Iterator for Type` with `Item` and `next()`
  - **From/Into**: conversion impls between types
  - **Test module**: `#[cfg(test)] mod tests` with `#[test]` functions
  - **CLI subcommand**: clap subcommand with args
- Templates stored as named entities in KG with `template:*` predicates
- Agent can select template via VSA similarity to goal description
- Templates compose: a "module" template can include "error type" + "struct" + "impl" templates

**Key types**:
```rust
CodeTemplate { name: String, params: Vec<TemplateParam>, body: AbsTree }
TemplateParam { name: String, kind: ParamKind, default: Option<String> }
ParamKind: TypeName | FieldList | TraitName | ModuleName
```

**Changes**:
- [x] New file: `src/grammar/templates.rs` (~400 lines, 14 tests) — 7 built-in templates
- [x] `src/grammar/mod.rs` — added `pub mod templates;`
- [x] `src/agent/tools/code_gen.rs` — template path: `template` + `template_params` params, 5 new tests
- [x] Built-in templates: error-type (thiserror+miette), trait-impl, builder, from-impl, test-module, iterator, new-constructor
- [x] Comma-respecting parser handles angle brackets (for generic types in method signatures)

**Estimated scope**: ~600–800 lines → ~400 lines + 80 lines integration

---

## Phase 10f — VSA Code Pattern Encoding

**Problem**: Code similarity search requires encoding code structure as fixed-size vectors. Neural approaches (code2vec, CodeBERT) require training. VSA can do this natively.

**Inspired by**: code2vec (AST path-contexts), GraphHD/VS-Graph (graph classification), VSA Survey

**Design**:

A **non-ML code2vec analog** using the existing 10k-bit binary VSA:

1. **AST node types as atomic symbols**: Each node type (`FnDecl`, `StructDef`, `IfExpr`, `MatchArm`, etc.) gets a `SymbolId` and thus a random base vector in item memory
2. **AST path encoding**: A path between two AST leaves (e.g., `[FnDecl, up, ImplBlock, down, TypeRef]`) encodes via `encode_sequence()` using permutation shifts
3. **Path-context encoding**: The triplet `(start_token, path, end_token)` encodes as:
   ```
   path_context = bind(start_token_vec, bind(path_vec, end_token_vec))
   ```
   This is exactly `encode_role_filler()` — no new primitives needed
4. **Full code vector**: Bundle all path-context vectors via majority vote:
   ```
   code_vec = bundle(pc_1, pc_2, ..., pc_N)
   ```
5. **Multi-granularity encoding** (optional): Combine token-level, AST-level, call-graph, and type-signature encodings with positional permutation:
   ```
   composite = bundle(rho^0(token_vec), rho^1(ast_vec), rho^2(call_graph_vec), rho^3(type_sig_vec))
   ```

**Capacity with 10k-bit vectors**: ~70-100 path-contexts per bundle. Recursive binding ~5-7 levels deep. HNSW search >95% recall at millions of entries.

**Key types**:
```rust
AstPathContext { start: SymbolId, path: Vec<SymbolId>, end: SymbolId }
CodePatternVec { symbol: SymbolId, vector: HyperVec, granularity: PatternGranularity }
PatternGranularity: Token | Ast | CallGraph | TypeSignature | Composite
```

**Changes**:
- [x] New file: `src/vsa/code_encode.rs` (~430 lines, 13 tests) — full VSA code pattern encoding
- [x] `src/vsa/mod.rs` — added `pub mod code_encode;`
- [x] `encode_path_context()` — bind(start, bind(path, end)) triplet encoding
- [x] `encode_code_vector()` — bundle multiple path-contexts into structural fingerprint
- [x] `encode_token_level()` — bag-of-words token encoding
- [x] `encode_type_signature()` — ordered param types + return type encoding
- [x] `encode_call_graph()` — ordered function call sequence encoding
- [x] `encode_composite()` — multi-granularity layer fusion with positional permutation
- [x] `AstNodeTypes` — 23 well-known AST node type labels
- [x] `extract_function_contexts()`, `extract_struct_contexts()`, `extract_enum_contexts()`, `extract_impl_contexts()` — KG→path-context helpers
- [x] Test: similar functions have higher similarity than dissimilar ones
- [ ] `src/agent/tools/code_ingest.rs` — extend to produce VSA encodings alongside KG triples (deferred to 10g integration)
- [ ] `src/vsa/item_memory.rs` — code pattern index (deferred to 10g integration)

**Estimated scope**: ~400–600 lines → ~430 lines

---

## Phase 10g — Pattern Learning from Examples

**Problem**: The user wants to "train a pattern from a blog post and build code like that pattern." This requires mining recurring code patterns from examples and making them retrievable.

**Inspired by**: FREQT/FREQTALS (AST subtree mining), Token Sugar (pattern mining for efficiency), MAPO (API usage patterns), program analogies via VSA

**Design**:

The **blog-post-to-pattern pipeline**:

1. **Ingest**: Blog post → library ingest (existing) → text chunks with code blocks extracted
2. **Parse**: Code blocks → `syn` parse → ASTs (reuse `code_ingest`)
3. **Mine**: Frequent AST subtree mining across all code blocks from the source
   - Enumerate candidate subtrees via rightmost path extension (FREQT algorithm)
   - Filter by minimum support threshold (appears in >= N examples)
   - Apply maximality constraint (only keep patterns not subsumed by larger ones)
4. **Encode**: Each mined pattern → VSA vector via `encode_ast_tree()` (from 10f)
5. **Store**: Pattern entities in KG with triples:
   - `(pattern_42, pattern:source, blog_post_entity)`
   - `(pattern_42, pattern:frequency, support_count)`
   - `(pattern_42, pattern:ast_structure, serialized_ast_fragment)`
   - `(pattern_42, pattern:category, "error-handling")` — inferred from context
6. **Retrieve via analogy**: Given a new task, compute the analogy vector:
   ```
   // "Do for HashSet what this pattern does for Vec"
   transform = bind(pattern_vec_for_vec, vec_concept_vec)
   target = bind(transform, hashset_concept_vec)
   matches = HNSW_search(target)
   ```

**API usage patterns** (secondary):
- Extract ordered API call sequences from code blocks
- Encode as VSA sequences via `encode_sequence()` with permutation
- Store as ordered step triples in KG
- Retrieve by partial sequence matching

**Key types**:
```rust
MinedPattern { id: SymbolId, ast_fragment: syn::Item, support: u32, source: SymbolId }
PatternMiner { min_support: u32, max_depth: u32, patterns: Vec<MinedPattern> }
PatternQuery { description: String, analogy_base: Option<SymbolId>, target_context: Option<SymbolId> }
```

**Changes**:
- [x] New file: `src/agent/tools/pattern_mine.rs` (~600 lines, 17 tests) — `PatternMineTool` implementing `Tool` trait
- [x] Code block extraction: markdown fenced blocks + HTML `<pre><code>` elements via scraper
- [x] `SimplifiedAst` structural skeleton with `ast_fingerprint()` for frequency-based pattern discovery
- [x] `PatternPredicates` + `mt:patterns` microtheory + `DerivationKind::SchemaDiscovered` provenance
- [x] `extract_simplified_contexts()` in pattern_mine.rs — VSA encoding of simplified AST shapes
- [x] Analogy search via VSA algebra: `bind(bind(pattern, source), target)`
- [x] Mine + search modes with language filtering, min_support threshold
- [x] `src/agent/tools/mod.rs` — added module + re-export
- [x] `src/agent/agent.rs` — registered `PatternMineTool`
- [x] `src/agent/tool_semantics.rs` — added `pattern_mine` semantic profile

**Estimated scope**: ~600–800 lines → ~600 lines

---

## Phase 10h — Library Learning Cycle

**Problem**: Over time, generated code accumulates common sub-patterns that should be extracted as reusable abstractions. Manual template creation (10e) doesn't scale.

**Inspired by**: DreamCoder/LILO (wake-sleep library learning), babble (e-graph anti-unification), Stitch (compression), AbstractBeam

**Design**:

A **wake-sleep cycle** for code abstraction discovery:

1. **Wake phase** (= normal OODA cycle): Agent generates code for tasks, writing results to KG
2. **Abstraction phase** (= enhanced consolidation): Periodically analyze all recently generated code:
   a. Load generated code ASTs from KG (`code:generated-by` provenance)
   b. Build e-graph containing all generated expressions
   c. Apply equality saturation with existing rewrite rules
   d. Run **anti-unification** on e-classes: find the most specific generalization shared by two or more expressions
   e. Score candidate abstractions by `occurrences * size` (Stitch metric)
   f. Extract top-K abstractions as new `CodeTemplate` entries
   g. Store in KG with `template:discovered-from` provenance and `DerivationKind::LibraryLearning`
3. **Naming phase**: Use the KG context (what goal was being solved, what domain) to assign meaningful names to discovered abstractions
4. **Integration phase**: New abstractions become available to `code_gen` tool for future tasks. VSA encoding enables similarity-based retrieval.

**Anti-unification on e-classes** (babble-style):
- Given e-class {A, B, C} where A, B, C are equivalent expressions:
- Anti-unify A and B: find pattern P with holes such that P[x:=a1] = A and P[x:=b1] = B
- P becomes a candidate abstraction (lambda with parameters)
- Working on e-classes (not raw ASTs) means discovering abstractions modulo the equational theory — more robust than syntactic matching

**Key types**:
```rust
DiscoveredAbstraction {
    id: SymbolId,
    pattern: AbsTree,       // the abstraction with holes
    params: Vec<String>,    // hole names
    occurrences: u32,
    compression: f64,       // bits saved by using this abstraction
    sources: Vec<SymbolId>  // provenance: which generated code it came from
}
LibraryLearner {
    min_occurrences: u32,    // minimum frequency to extract
    min_compression: f64,    // minimum bits saved
    max_abstractions: usize  // per cycle
}
```

**Changes**:
- [x] New file: `src/reason/anti_unify.rs` (~420 lines, 8 tests) — anti-unification on `SimplifiedAst` trees, `GeneralizedAst`/`AstSlot`/`AntiUnifyVar`, scoring, `DiscoveredAbstraction`
- [x] New file: `src/agent/library_learn.rs` (~310 lines, 5 tests) — `LibraryLearner` wake-sleep orchestration, KG code collection, AST reconstruction, template storage
- [x] `src/reason/mod.rs` — added `pub mod anti_unify;`
- [x] `src/agent/agent.rs` — `run_library_learning()` method, periodic trigger (4x less frequent than reflection)
- [x] `src/agent/mod.rs` — added `pub mod library_learn;` + re-exports
- [x] `src/grammar/templates.rs` — `TemplateGenerator::Learned` variant, `CodeTemplate::from_abstraction()`, `generate_learned()`
- [x] `src/provenance.rs` — `DerivationKind::LibraryLearning` (tag 39)

**Depends on**: 10a (grammar), 10b (code_gen tool), 10e (templates), 10f (VSA encoding)

**Estimated scope**: ~500–800 lines

---

## Implementation Order

```
Wave 1: Core Generation
  10a (RustCodeGrammar) ──→ 10b (CodeGenTool) ──→ 10c (Code-Aware Planning)
                                  │
                                  └──→ 10d (Iterative Refinement)

Wave 2: Pattern Infrastructure
  10e (Templates) — can start after 10a, enhances 10b
  10f (VSA Code Encoding) — can start after 10a, independent of 10b-10d

Wave 3: Learning
  10g (Pattern Mining) — depends on 10f (VSA encoding) + library ingest
  10h (Library Learning) — depends on 10a, 10b, 10e, 10f
```

**Wave 1** delivers working code generation.
**Wave 2** adds pattern infrastructure (manual templates + VSA encoding).
**Wave 3** closes the learning loop (mine patterns from examples + discover abstractions automatically).

10a is the core prerequisite — everything else builds on the grammar.
10b makes it available to the agent.
10c makes the agent plan code work autonomously.
10d closes the compiler feedback loop (CEGIS pattern).
10e provides manual templates for common patterns.
10f enables similarity-based code retrieval without ML.
10g delivers the "learn from blog post" capability.
10h makes the system self-improving over time.

## Total Estimated Scope

~3,800–5,500 lines across 8 sub-phases.

| Phase | Lines | Key Deliverable |
|-------|-------|-----------------|
| 10a | 500–700 | RustCodeGrammar linearizer |
| 10b | 400–600 | CodeGenTool for agent |
| 10c | 300–500 | Code-aware planning |
| 10d | 500–700 | Compiler feedback loop |
| 10e | 600–800 | Parameterized templates |
| 10f | 400–600 | VSA code pattern encoding |
| 10g | 600–800 | Pattern mining from examples |
| 10h | 500–800 | Library learning cycle |

## Key Research Insights Applied

1. **Non-ML code2vec** (10f): AST path-contexts encoded as VSA role-filler bindings give structural code similarity without any neural network training. HNSW search over 10k-bit vectors at >95% recall.

2. **CEGIS pattern** (10d): The OODA loop naturally implements counter-example guided synthesis — each compiler error is a counterexample guiding the next attempt.

3. **DreamCoder/LILO architecture** (10h): Wake phase = OODA cycle generating code. Abstraction sleep = consolidation discovering reusable patterns. The e-graph (egg) already provides the compression substrate.

4. **babble anti-unification** (10h): Finding shared patterns over e-classes (not raw ASTs) discovers abstractions that are robust to syntactic variation — two different implementations of the same algorithm get unified.

5. **Program analogies** (10g): VSA's algebraic properties (bind is its own inverse) enable "do for X what we did for Y" reasoning natively. The transformation vector captures what changed between two patterns and applies it to a new context.

6. **Graph-level classification** (10f): VS-Graph showed GNN-level accuracy at 250x faster training using HDC. Code structure classification ("this is a builder pattern") works via prototype matching in HNSW — no gradient-based training needed.

See `docs/ai/decisions/002-code-generation-research.md` for full research details and source references.
