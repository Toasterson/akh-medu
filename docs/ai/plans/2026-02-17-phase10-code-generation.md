# Phase 10 — Generative Functions (Rust Code Generation)

- **Date**: 2026-02-17
- **Status**: Planned
- **Depends on**: Phase 9 (partially — 9b predicate hierarchy and 9f reasoner dispatch are most useful but not blocking)

## Goal

Enable akh-medu to generate valid, idiomatic Rust code from its knowledge graph. The agent should be able to: query code structure from the KG, plan a code artifact, generate it through the grammar system, write it to disk, and validate it with the Rust toolchain — all autonomously through the OODA loop.

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

## What's Missing

1. **`RustCodeGrammar`** — a `ConcreteGrammar` impl that linearizes `AbsTree` code nodes into valid Rust source
2. **`code_gen` tool** — agent tool that orchestrates: query KG → build AbsTree → linearize → format → validate
3. **Code-aware planning** — plan generation recognizes code goals and produces code_gen steps
4. **Iterative refinement** — compile error → parse error → fix → retry loop
5. **Template library** — common Rust patterns (error types, trait impls, builders) as reusable AbsTree templates

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
- [ ] New file: `src/grammar/rust_gen.rs`
- [ ] `src/grammar/mod.rs` — register `RustCodeGrammar` as built-in archetype
- [ ] Handle: derives, visibility (`pub`), generics, lifetimes, `where` clauses
- [ ] Handle: `use` statements from `code:depends-on` triples
- [ ] Indentation and formatting (pre-rustfmt)

**Estimated scope**: ~500–700 lines

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
- [ ] New file: `src/agent/tools/code_gen.rs`
- [ ] `src/agent/mod.rs` — register `CodeGenTool` in `register_builtin_tools()`
- [ ] `src/agent/ooda.rs` — add `code_gen` to tool scoring (keyword detection for code goals)
- [ ] `src/provenance.rs` — `DerivationKind::CodeGenerated`

**Estimated scope**: ~400–600 lines

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
- [ ] `src/agent/plan.rs` — new `generate_code_plan()` function
- [ ] `src/agent/plan.rs` — detect code goals in `generate_plan()` and delegate
- [ ] `src/agent/plan.rs` — code-specific backtracking: parse compiler errors, adjust KG

**Estimated scope**: ~300–500 lines

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

**Changes**:
- [ ] New file: `src/agent/tools/compile_feedback.rs` — parse `cargo check` JSON output
- [ ] `src/agent/tools/code_gen.rs` — `refine()` method that takes errors + existing code
- [ ] `src/agent/ooda.rs` — refinement loop integration (detect compilation failure, trigger refinement)
- [ ] `src/agent/plan.rs` — code plans include conditional refinement steps

**Estimated scope**: ~500–700 lines

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
- [ ] New file: `src/grammar/templates.rs`
- [ ] `src/grammar/rust_gen.rs` — template instantiation during linearization
- [ ] `src/agent/tools/code_gen.rs` — template selection and parameterization
- [ ] Built-in templates for the patterns listed above

**Estimated scope**: ~600–800 lines

---

## Implementation Order

```
10a (RustCodeGrammar) ──→ 10b (CodeGenTool) ──→ 10c (Code-Aware Planning)
                                │
                                └──→ 10d (Iterative Refinement)
10e (Templates) — can start after 10a, enhances 10b
```

10a is the core prerequisite — everything else builds on the grammar.
10b makes it available to the agent.
10c makes the agent plan code work autonomously.
10d closes the feedback loop.
10e is a quality/velocity multiplier.

## Total Estimated Scope

~2,300–3,300 lines across 5 sub-phases.
