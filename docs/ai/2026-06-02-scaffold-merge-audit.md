# Scaffold-Merge Audit — 2026-06-02

## What this is

The `anima-triage` bot opens a **scaffold PR** per work item whose only content is a
`.anima/wip/<wi>.md` triage marker ("implementation worker picks it up from here"). An
auto-merge workflow merged **~35 of these scaffold PRs into `develop`/`main` carrying only the
marker — zero code**. "Hollow merge" does **not** by itself mean the feature is missing: for many,
the real implementation landed via a *different* branch/PR. This audit classifies every hollow
scaffold-merge reachable from `main` and records, per PR, whether the described feature actually
exists in the code.

**Method:** `git log --merges` on `main` → keep merges whose diff vs first parent is only `.anima/`
(0 `.rs/.toml/.yml/.json/.sql/.sh/Dockerfile` files) → per-PR grep of `src/`+`crates/` for the
described feature. Verdicts: **IMPLEMENTED** (real code exists, usually via a sibling PR),
**PARTIAL**, **PHANTOM** (described work has no code anywhere), **TRIVIAL** (no-op WI).

## Headline

- **35 hollow scaffold-merges** found in `main` history (almost all pre-date the 2026-06-02
  integration pass; merged by the auto-workflow, not by hand).
- Of the substantive ones, **6 are genuine phantoms** (claimed/merged but never built):
  **#65, #67, #63, #122, #123, #152** — plus my own erroneous closures **#162, #163** (reopened).
- Everything else is backed by real code (landed via sibling PRs) or is a partial.
- **Reopen note:** #162/#163 were reopened. **#65/#67/#63/#122/#123/#152 could not be reopened** —
  Forgejo refuses to reopen a PR it marked *merged* (they were auto-merged). Explanatory comments
  were posted on #65/#67. The genuinely-missing work should be tracked as **new issues / reactivated
  Anima WIs**, not via these merged scaffolds.

## Class A — hollow merge, feature implemented elsewhere (legitimately done)

| PR# | Feature | Evidence |
|-----|---------|----------|
| #28 | Phase 19 Epistemic Logic | `src/agent/epistemic.rs` exists |
| #121 | Phase 40 E-graph persistence | egraph-persist code present |
| #214 | OTLP exporter | opentelemetry deps/code present |
| #126 | Phase 26f hidden-state fidelity | `extract_hidden_state` exists — **but a STUB** (returns logits, not true hidden states); see gemma-vsa gap doc |

## Class C — precise per-PR verdicts (26 PRs)

| PR# | Title (short) | Verdict | Evidence |
|-----|---------------|---------|----------|
| #219 | Zero triples in tecton workspace | IMPLEMENTED | siblings #220–223 (persona/self-model/cluster/goal); SPARQL primary store `engine.rs:257` |
| #205 | Rebuild tecton: zero triples | IMPLEMENTED | #220-223 + auto seed-packs (`akhomed.rs:151`) + `data/seeds/tecton/seed.toml` |
| #208 | Seed pack never applied | IMPLEMENTED | auto-apply on init `workspace.rs:178`/`akhomed.rs:150`; sibling #212; `apply_seed` MCP tool |
| #207 | Provenance explosion 18:1 | IMPLEMENTED | `src/agent/compaction.rs` + daemon `run_kg_compaction` (30min); sibling #210 |
| #211 | Toasty zero outgoing triples | PARTIAL | seed gives Toasty `created-by`+`is-a person`; rich profile (location/email/interests) NOT seeded |
| #157 | Define Ennead member identities | IMPLEMENTED | `data/seeds/ennead/seed.toml` (7 members, archetype/role/domain/traits); sibling #213 |
| #152 | Define fluent infrastructure (home state) | **PHANTOM** | event-calculus infra exists generally, but no seeded home-state fluents/events anywhere |
| #146 | Entity hallucination (~108 triples each) | IMPLEMENTED | rule-engine guards `is_invalid_type_hierarchy_participant`/`is_noise_entity`/self-ref/min_conf; siblings #149/#151 |
| #145 | Cluster topology as flat strings | IMPLEMENTED | `src/cluster/mod.rs::ingest_topology` → structured K8sNode/IP/state triples |
| #144 | test | TRIVIAL | WI body literally "test" — no-op |
| #141 | Cluster topology flat text → K8sNode | IMPLEMENTED | same as #145 (`ingest_topology`/`extract_node_triples`) |
| #140 | Zero contacts (no people model) | IMPLEMENTED | `src/agent/contact.rs` (ContactManager, contact predicates) + tecton seed |
| #137 | Self-referential triple explosion | IMPLEMENTED | `engine.rs:493` rejects subject==object; `purge_self_referential_triples`; sibling #149 |
| #135 | Goal atomicity (single entity) | IMPLEMENTED | `goal.rs:318 create_goal` one entity + property triples; sibling #143 |
| #132 | ritual_of_awakening preview-only/leaks | IMPLEMENTED | `ritual_of_awakening` persists psyche directly (`mcp/mod.rs:91`,:2671) |
| #129 | Role-typed workspaces + seed packs | IMPLEMENTED | seed-packs at workspace create (`workspace.rs:178`); role packs bundled (`seeds/mod.rs:154`) |
| #127 | Phase 41 Pattern B (OpenRouter re-encode) | IMPLEMENTED | `src/bench_phase41g/pattern_b.rs` `PatternBClient` → openrouter.ai |
| #123 | Phase 42h cross-tool bridge daemon | **PHANTOM** | `crates/heka/README` documents heka-bridge :7000, but no bin target / router / chat-completions code |
| #122 | Phase 42g Akhipedia auto-extraction from drawers | **PHANTOM** | README documents it; no drawer→article extraction code (Phase 22 `src/wiki/` exists, drawer pipeline does not) |
| #120 | RWO PVC rollover deadlock → Recreate | IMPLEMENTED | `charts/akh-medu/templates/akhomed-deployment.yaml` `strategy: Recreate` |
| #84 | redb single-writer lock blocks MCP | IMPLEMENTED | sibling #225 serialized writes + daemon `try_lock` (8 sites) |
| #158 | Add write timeout to engine ops | PARTIAL | contention mitigated (try_lock, write serialization, 30s psyche self-heal) but no configurable engine-level write timeout on add/remove_triple |
| #153 | status tool ignores workspace JSON param | IMPLEMENTED | `mcp/mod.rs:55 deserialize_workspace` on all workspace tools; sibling #204 |
| #103 | MCP infer_preview (dry-run infer_rules) | PARTIAL | `RuleEngineConfig.auto_commit` primitive exists, but no `infer_preview` MCP tool / dry-run JSON surfaced |
| #64 | Replace continuous_learning with Seshat RAG ingester | PARTIAL | old ingest deactivated (`idle.rs:117`); Seshat-mediated RAG replacement NOT built |
| #78 | `.anima-exec/` gitignore | IMPLEMENTED | `.gitignore:5` |

## Class B — phantom (hollow merge AND no implementation)

| PR# | Feature | Status |
|-----|---------|--------|
| #65 | Phase 38 — Relocatable Session-Based Akh (Postgres-truth / 3-tier sleep / AGE) | not in `src/`; ADR 049 `Proposed`. Engine still redb+oxigraph+HNSW. |
| #67 | Custom CNPG image bake with AGE + pgvector | not in repo; chart uses *stock* pgvector image; "homelab-side dependency" belongs in infra repo |
| #63 | Bootstrap models via OCI instead of PVC | not done; models still via RWO PVC + `akh setup models` HF download |
| #162 | Gemma 4 OODA efficiency 7.5× worse than DeepSeek | hollow; **reopened** |
| #163 | Entity:Relation ratio 4.95:1 deficiency | hollow; **reopened** |
| #152, #122, #123 | (see Class C) | phantom |

## Tally

- **Class A (impl elsewhere): 4** (#28, #121, #214; #126 partial-stub)
- **Class C: 26** → 16 IMPLEMENTED, 5 PARTIAL (#211, #158, #103, #64, +#126 stub), 4 PHANTOM
  (#152, #122, #123 [+ #144 trivial])
- **Class B (phantom features): 6** (#65, #67, #63, #152, #122, #123) + 2 reopened (#162, #163)

**Net "claimed-but-never-built" work to re-track as fresh issues:** Phase 38 / Postgres-truth+AGE
(#65), CNPG+AGE image (#67, infra repo), OCI model bootstrap (#63), Akhipedia drawer
auto-extraction (#122), heka cross-tool bridge daemon (#123), home-state fluents (#152), the
Gemma-OODA-efficiency + entity:relation investigations (#162/#163), and the Seshat-RAG ingester
(#64). Everything else attributed to these scaffolds is genuinely present in the code.
