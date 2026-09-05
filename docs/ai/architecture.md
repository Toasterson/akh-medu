# Akh-medu Architecture

> Last updated: 2026-08-30 (overview corrected: the LLM is first-class — the agentic loop runs on per-akh selectable chat backing (Ollama local/cloud, in-process Candle with grammar-constrained tool calling, OpenAI-compatible endpoints; ADRs 070/095) through the Phase 44 context-assembly runtime. The symbolic core remains CPU-native and LLM-optional.)

> Previous update: 2026-08-26 (**ADR 098 — exact Qwen3.8 Candle inference stack on RDNA3** ([decision](decisions/098-candle-qwen35-rdna3-inference-stack.md), [plan](plans/2026-08-24-candle-qwen38-rdna3-inference.md))). `ModelFamily::Qwen35` owns the custom Qwen3.8 architecture and its ChatML behavior. Its reference Candle adapter validates the complete target+MTP GGUF schema before allocation and implements the exact 3:1 gated-delta/full-attention text forward with explicit KV, recurrent, convolution, checkpoint, and rewind state, including Qwen35 GGUF's tiled V-head conversion contract. Hermetic non-zero logits/state tests, the installed Ptah artifact schema, a bounded-memory exact-model multi-token CPU prefill, and a full HIP stepped decode all match the llama.cpp oracle in greedy token and log-probability. Slice 2a now privately quantizes each 32-value decode activation group to Q8 and consumes packed Q4_K weights with the gfx11 `dp4a` path; independent operation parity plus three accumulated exact-model steps preserve every greedy token with minimum hidden/logit cosine `0.999960874`/`0.999991194`, while twelve counterbalanced warm runs show a `1.162x` median win. That promotes it to the default while `AKH_HIP_Q8_GEMV=0` preserves the f32 reference. WMMA short-batch/prefill, gated-delta scan, and flash attention remain the next measured private operations. A bounded C1/C2/C4/C8 scheduler is recorded after the owned `InferenceSession`, with target-only and realistic-makespan gates before optional MTP. Exact vision, VSA signals, TurboQuant, and DFlash2 remain later parity-gated slices; llama.cpp remains the production oracle until the canary passes.

> Operational durability update, 2026-08-26: PostgreSQL no longer B-tree-indexes arbitrary `akhomed_symbols.label` text. A guarded store migration replaces that unbounded index with a fixed-size hash index for exact label equality; the primary key already covers `akh_id`. This prevents imported long knowledge labels from aborting otherwise valid periodic snapshots, while non-Postgres backends retain their portable index.

> Wake-replay durability update, 2026-08-26 ([ADR 049](decisions/049-relocatable-akh.md)): a captured graph can outrun its asynchronously flushed WAL while retaining the earlier durable watermark. Ordered replay now recognizes only the provably safe overlap shape—stale retract metadata followed by the first same-edge mutation being an exact assertion of the edge already present in the snapshot—and skips that intermediate retract. Every other metadata mismatch remains atomic and fail-closed; PostgreSQL history is never edited to recover.

> Previous update: 2026-08-23 (**ADR 097 — evidence-gated completion + VSA pre-action context** ([decision](decisions/097-evidence-gated-completion-and-vsa-context.md), [plan](plans/2026-08-23-evidence-gated-completion-controller.md))). An explicit, fail-closed `<akh:control>` directive gives any turn ordered successful-tool post-conditions; later evidence cannot arrive early, and re-running an earlier stage invalidates stale downstream evidence. Required tools survive catalog retrieval scoping. Bounded VSA spreading activation from explicit/exact/namespaced seeds is injected with provenance as advisory reasoning context, while domain verifiers remain authoritative. Ptah selects the existing MemGPT-style `virtual` context strategy independently.

> Previous update: 2026-08-11 (**OpenAI-compatible chat backing + per-akh context strategy** ([ADR 095](decisions/095-openai-compatible-chat-backing.md)): ADR 070's `ChatBacking` gains a third peer, `backend = "openai"` (`src/agent/openai_backend.rs`) — any host serving the OpenAI `/chat/completions` contract (vLLM, SGLang, llama.cpp server, OpenRouter, DeepSeek, Hetzner's free inference experiment), configured by a new `[openai_endpoint]` table (`base_url` incl. version segment, `env:VAR` `api_key`, **operator-stated `context_window`** since the OpenAI contract has no `/api/show` equivalent). Non-streaming + SSE streaming, full tool calling. The loop is unchanged: because `agentic_chat` builds conversations in *Ollama* shape (no call `id`, object-valued `arguments`, no `tool_call_id`), translation to strict OpenAI shape lives at the backend boundary (`to_openai_messages`, **deterministic** `call_{round}_{idx}` ids so replays are byte-identical), with the inverse on responses. Misconfiguration (`backend = "openai"` without an endpoint or pinned model) is **fatal for that workspace**, not a fallback — unlike the Candle branch, there is no same-akh endpoint to fall back to, so a silent substitution would answer from a different model than configured and invalidate any measurement. Separately, `context_strategy` becomes a per-akh scalar (precedence: field → `AKH_CONTEXT_STRATEGY` → `compacting`, unrecognized values now warn instead of silently downgrading): one daemon hosts several akhs, so a process-global env var could only move all of them together, leaving no control group. First use: the Mac mini's third implementer lane `deepseek` (DeepSeek-V4-Flash-0731 over Hetzner) running the Phase 44i MemGPT `virtual` strategy against `tecton` + `imhotep` on `compacting` — a live A/B on one task queue. Example config: `examples/three-akhs/workspaces/deepseek.toml`, now covered by a test that loads every shipped example and asserts no unknown keys.) — Prior: 2026-07-11 (**ADR 081 Akh Memory & Schema-Typing IMPLEMENTED — all work items** ([ADR 081](decisions/081-akh-memory-schema-typing.md), fleet session, branch `claude/typed-kg-storage-analysis-59ec37`): **P1** `library::note` — `Note` = `DocumentRecord` + typed `Frontmatter` (mirrors `schema::source::ValueType`) + `[[wikilinks]]` + stable `NoteId`; frontmatter-less Note byte-identical to Document. **Slice 1** `schema::project` — mechanical (no-LLM) projection: `type:` frontmatter → `is-a` + declared-property triples, unknown keys → `note:frontier/{key}` markers (writes never blocked), wikilinks → `links-to`, `typed_view` read-back; idempotent via the KG's (s,p,o) dedup. **Slice 2** — the PG registry half: `akhomed_schema_registry` table + `PgSchemaRegistry`, `akh schema admit --yes|list|install` over `_akh/schema/*`, wake-time lazy convergence (`snapshot::wake` step 7); activation rides `workspaces_active_microtheories`. **Slice 3** `schema::shape`+`digest` — is-a-chain effective property sets (cycle-safe, cross-pack), shape validation (violations demote to frontier + report, never block), `get_graph_schema`-style `schema::digest` + `_akh/schema/digest`. **Slice 4** `agent::librarian` — LLM extraction seam (`FakeLibrarianModel` deterministic + `OllamaLibrarianModel`, `AKH_LIBRARIAN_MODEL` default gpt-oss:120b), own `librarian_watermark` column, typed-vs-frontier validation, `DerivationKind::LibrarianExtraction`, daemon daily tick gated `AKH_LIBRARIAN=1`. **Slice 5** `agent::promotion` — mechanical candidates (≥3 distinct subjects + consistent value shape) → stored `schema:proposal/*` entities (never auto-applied) → `akh schema proposals|promote <id> --yes|reject`; promotion mints a one-property mini-pack via is-a subtyping through the normal admit path; aliasing = `onto:alias-of`. **Slice 6** `agent::tasks` + `structured_task_*` tools — typed Task (lifecycle-encoded completion), KG-authoritative via `task:` predicates, complete → past episode + recurrence spawn, task entity doubles as episodic future-spine node, Obsidian-Tasks emoji rendering boundary-only (no parser). **Slice 7** — context tiering: schema digest + active-note frontmatter in core, recent-note frontmatter in recall, bodies ONLY via `archival_search` (pre-Heka substring fallback over `NoteCatalog`) + new `query_graph` tool, both gated to the `virtual` strategy; never-resident-bodies invariant tested. **Slice 8** `schema::conflict` + `reproject_note` — per-note projection manifests as `(note-manifest:{id}, projection:asserted, {TripleIdentity})` triples in `mt:schema/projection-manifest`; diff by TripleIdentity: retract-changed via TMS (`was_retracted`-discoverable), `DerivationKind::NoteReprojection` audit; multi-akh conflict = DS belief intervals (Phase-17 masses × Phase-18 reliability), never last-writer-wins. **Slice 9** `schema::coverage` — grandfathering as pure classification (`TypedCore|Frontier|SchemaInfra`, no data rewrite), `akh schema coverage` + `_akh/schema/coverage`, top-N frontier predicates feed promotion. Full lib suite 2974 green (postgres); live-PG integration green (registry rerunnable, harvest, WAL, ADR-091 acceptance). Open: S0 empirical small-model bet (seams shipped with deterministic fakes), cross-akh promotion consensus signal, Heka vector archival, alias-aware coverage reclassification.) — Prior: 2026-07-08 (**ADR 091 — transcript→episodic capture designed (ADR-081 write-side first slice)**: a live audit found the akh memory loop broken at its first hop — conversations are recorded only into an ephemeral, never-serialized 10-turn ring buffer, so `akhomed_episodes` stays empty and the wired-but-starved downstream (recall `agent.rs:1711`, sleep/consolidation `daemon.rs:331,381`, introspection) never sees a transcript. A LongMemEval A/B made it concrete (a genuine `PassThroughManager` "off" arm was added to `akhomed.rs`'s `AKH_CONTEXT_STRATEGY` match — previously any unknown value silently fell through to `CompactingManager`, so the daemon had no true off mode): **0/6 harness-off vs 4/6 harness-on, but the 4/6 was ephemeral within-session working memory, not durable episodic**. Locked design (grilling 2026-07-08, [ADR 091](decisions/091-transcript-episodic-capture.md) / [plan](plans/2026-07-08-transcript-episodic-capture.md)): capture each completed turn at `run_agentic_turn` turn-end as a raw JSONL-style record in a **new `akhomed_mutation_log` `kind="turn"` row** (durable immediately, hypertable, wake-replay **skips** it — refines ADR 081 P1's "daily log = Document"); the daily-log is the projection `WHERE kind='turn'`; a **new harvest phase in `run_sleep_cycle`** reads turns by a per-akh `akh_state` watermark and mints lightweight `episode:turn:{seq}` indices (`learned`-linked to keyword-resolved symbols + `body_ref`=seq), reconciling onto the existing `memory.rs` `EpisodicEntry`; recall pages the body from the WAL on demand; the gpt-oss Librarian (ADR 081 slice 4) later supersedes the crude keyword distillation. Default-on + per-workspace opt-out; keep raw indefinitely (compressed), no prune this slice. Acceptance = cross-session recall across a **daemon restart** (inject A → sleep/harvest → restart → recall B), passthrough control = 0. Design only — not yet built. Also this session: benchmarked the harness-value question on single-turn illumos-e2e (Track 1 — KG grounding gave no reliable lift; prose-in-KG-labels *hurt* 8/15, atomic-KG 11/15 was n=1 variance, n=3 confirm ~50/50).) — Prior: 2026-07-06 (**Phase 50 W5 — first real public-benchmark leaderboard rows** (`benchmarks/`, ADR 086 §3/§4/§5): `bench-tool record-run` (`benchmarks/harness-core/src/bin/bench_tool.rs`) is a new schema-validated seam any harness (Rust or Python) can target — a JSON array of `RawTaskRow`s in, a provenance-stamped `RunRecord` out via `harness_core::store::write_run` — generalizing the W7 results pipeline beyond the `record-synthetic` demo path. `benchmarks/emit_run_record.py` converts `frontiercode`'s and `run_memory_bench.py`'s own result formats into that shape and calls the CLI. Both Python harnesses gained `--workspace`/`--host`/`--port` (previously hardcoded to `tecton`/`8201`) so they can drive *any* per-model benchmark akh; `run_memory_bench.py` also gained a hard `--max-requests` budget cap (ADR 086 §5: exceeding it aborts remaining items as `budget_exceeded`, never silently drops or loops) and per-item exception isolation (a transient ACP timeout aborts that item, not the whole process — a real crash this session exposed the gap). First real 3×3 matrix recorded: `frontiercode` (2 SWE-bench-Verified sympy tasks, full env+test grading) × `locomo` (1 conversation, all ~19 sessions, 5 questions — not the cheap oracle-capped mode) × `longmemeval` (oracle, 6 sampled items) against `deepseek-v4-pro` / `glm-5.2` / `qwen3.5:397b` over ollama.com cloud, driven through a dedicated benchmark `akhomed` instance (standalone Postgres via `scripts/pg-standalone.sh`, port 8301, separate `XDG_CONFIG_HOME` — the user's production `default`-workspace akhomed on 8201 was never touched). Result: frontiercode 0/2 for all three models (expected per the harness's own README — one real infra artifact disclosed: an accidental `kill -9` during live debugging hit a FAIL_TO_PASS pytest subprocess mid-run for glm-5.2/sympy-17630, recorded honestly as `rc=-9` in that task's notes rather than silently re-run); locomo (full-context, not oracle) 0/1 for all three (real long-context recall struggle, "no mention" hallucination pattern); longmemeval (oracle) 3/6, 3/6, 4/6 for deepseek/glm-5.2/qwen respectively. Terminal-Bench leg and W6 fleet-orchestration remain unbuilt (no adapter exists yet). `benchmarks/LEADERBOARD.md` + `benchmarks/site/index.html` regenerated (12 (harness,model) pairs total incl. the prior synthetic demo rows).) — Prior: 2026-07-05 (**Phase 48c `RemoteBackend` built** (`src/backend/remote.rs`, feature `remote-executor`, Phase 50/W2a): russh-over-SSH executor implementing the full `Backend` trait — SSH exec-channel for `exec`, shell-command file verbs (`cat`/`base64`/`ls`/`test`/`wc`) over the same channel on the remote FS (not SFTP), plain-provisioned-key auth with a `HostKeyPolicy` (KnownHosts/Pinned/InsecureAcceptAny; 48d OIDC→cert deferred per ADR 086 §1), no client-side path sandbox (OS is the sole boundary), new `Confinement::RemoteUnverified` posture; sync `Backend` trait bridges async russh via an owned multi-thread tokio runtime + `block_on` (trait signature unchanged); tested against an in-process russh **server** fixture over a `tokio::io::duplex` pipe (real SSH wire, 10 tests). — **Phase 40 measured + decided — e-graph persistence: Option (0), persist nothing** ([ADR 084](decisions/084-egraph-persistence-decision.md), [report](2026-07-05-phase40-egraph-resaturation.md)): production-limit saturation is 0.2–254 ms median across 10²–10⁵ facts (bounded at 5 s by `Runner::default()` itself), the Phase-38 `wake()` has no e-graph step, and HNSW rebuild — not the e-graph — is the real wake-path cost (~14 min at 100 k items; ADR 049's local-disk cache already covers it). New permanent measurement crate `crates/akh-egraph-bench/`; fixed latent `calendar_rules()` construction panic (unbound `?conflicts` in cal-conflict, hidden since Phase 13f).) — Prior: 2026-06-30 ( — **Phase 33a grammar aggregation rules built** — `grammar/aggregate.rs` folds shared-subject triples into fluent combined sentences before linearization (`AbsTree::Aggregated` + `PredicateGroup`): subject merge → apposition, predicate enumeration → Oxford-comma object lists, discourse ordering (is-a → properties → relations), pronoun-substituted continuation sentences, confidence-band-safe merging; wired into `build_discourse_response` (third person), `synthesize_abs::build_entity_section`, the communicative `triples_to_sentences` (34b style transform applies AFTER aggregation), and the `ground_query` fallback; knob `AKH_GRAMMAR_AGGREGATE` (default on); plan 33a, ADR 043. — **Phase 48a — executor `Backend` seam built** (`src/backend/`, [ADR 082](decisions/082-privilege-separated-executor-backend.md), [plan](plans/2026-07-04-phase48-privilege-separated-executor.md)): `Backend: Filesystem` + `exec`/`handshake`, `InProcessBackend` sole impl, `file_io`/`edit`/`glob`/`grep`/`shell_exec` rewired through `Arc<dyn Backend>` with zero behavior change; 48b LocalDrop / 48c russh Remote pending. — **Phase 29 capability exposure (29a–29e, 29h, 29i) shipped over ACP** — microtheories, skillpacks, psyche/archetypes, argumentation, explain/provenance, evidence assess, source trust as `_akh/*` groups + `akh acp` CLI verbs; see "Phase 29 capability exposure" under Interaction. Prior: — **ADR 081 prefactors built** — stable triple-identity hash (`graph::identity::TripleIdentity`, blake3 over canonical labels + compartment, cross-akh/session stable, minted in `add_triple` + persisted in the WAL record, queryable via `Engine::triple_by_identity`) and the schema-pack compiler (`src/schema/`: portable JSON source → validated → lowered to label-based native triples → bincode blob + `RegistryRow` checksum; per-akh `install_pack` bulk-insert-as-microtheory with the immutable-global-type-name guard; `akh schema compile|show`; seed pack `data/schema-packs/schema-org-core.json`). LLM Librarian, Obsidian/Orbit sides, and the Postgres registry table remain unbuilt. — **Phase 41 Thinking Engine built** — `src/thinking/`: unified `ThinkingLoop` + `ThoughtState` + budget + goal-shape scheduler + structural `CompositeVerifier` + VOC-adaptive budget + scaling-law bench; 41d Postgres persistence deferred to a `ThoughtPersistence` seam. See ADR 052 (design), [ADR 085](decisions/085-thinking-engine-findings.md) (findings). — **Phase 34 communicative intelligence built** — response strategy selector, psyche style transform, conversation trajectory, shadow filter; ADR 044. Prior update 2026-06-30: **Akh Memory & Schema-Typing subsystem designed** ([plan](plans/2026-06-30-akh-memory-schema.md), [ADR 081](decisions/081-akh-memory-schema-typing.md); *not yet built*) — *Obsidian write-side + Orbit read-side + an LLM Librarian projector + a human-gated schema-pack registry*: permissive note/transcript truth + a typed read-model projection that crystallizes a **globally-immutable, additive, `is-a`-subtyped** core at runtime; installable human-confirmed **microtheory packs** (dedicated registry table of native-triple byte blobs, compiled once from GitHub-curated Turtle/JSON-LD); tiered authoring (small local model append-only + type-guessing markers; **gpt-oss 120B daily Librarian** projector, complementing VSA sleep); feeds Passive/`VirtualContextManager` with bodies-in-archival so the resident footprint is horizon-independent (<100K, no compaction); reuses microtheories, `onto:` constraints, TMS supersede, Dempster-Shafer + source reliability, Phase-32 elicitation, Phase-44 context managers; work breakdown [`docs/ai/plans/2026-06-30-akh-memory-schema-issues.md`](plans/2026-06-30-akh-memory-schema-issues.md); de-risk first: the R2 small-model schema.org/task-tool bet. — **Release & packaging pipeline** (ADR 079): tag-triggered `.forgejo/workflows/release.yml` builds `akh`/`akhomed`/`seshd` container images for Linux + a native illumos **IPS pkg(5)** `.p5p` (SMF manifests under `contrib/smf/`, install helper `packaging/install.sh`, recipe under `packaging/illumos/`, runbook `docs/ops/release.md`); illumos builds are ORT-free (`akhomed --features server`, `seshd` synthesis-only) since onnxruntime has no illumos build; the Linux tarball reuses the image binaries so the memory-limited runner never recompiles the workspace. — **Phase 44 COMPLETE (44a–44j) — working-context assembly runtime** (`src/agent/context/`, `src/bench_phase44/`, ADR 065): a swappable `ContextManager` seam wired into `run_agentic_turn` with a `TokenCounter` abstraction (heuristic `len/4` + feature-gated Candle tokenizer) and a byte-identical `PassThroughManager` default (zero behaviour change), a `BudgetAllocator` (atomic sections all-or-nothing) + per-turn token telemetry, and provenance-derived `Retention` + a `SafetyEnforced` decorator that refuses to evict `PinnedSafety` spans (AgentDecision / ShadowVeto / dangerous-tool provenance) — guarding *eviction* as ADR-060 guards *execution*. Three live, per-akh-selectable strategies (`AKH_CONTEXT_STRATEGY`, default `compacting`): `CompactingManager` (seed→KG→recalled→history + budgeted tools; 44e recency/graph-proximity discourse; 44d opt-in KG-context delta), `PassiveManager` (44h, lean + background consolidation), `VirtualContextManager` (44i, MemGPT core/recall/archival tiers + `core_memory_append`/`core_memory_replace`/`archival_search` self-edit tools over KG-backed core memory). 44g recall = VSA-expanded + recency/confidence blend + TMS supersede filter (`was_retracted`). 44j deterministic A/B harness + dated report. 44k (Candle KV) / 44l (Heka retrieval swap) are `context::policy` trait stubs, recorded-deferred (gated on ADR 037 / Phase 42). Also: standalone CloudNativePG test harness (`scripts/pg-standalone.sh`) validating the whole Phase-38 workflow E2E, advisory-locked migrations (`run_migrations_locked`) for multi-host, and a 3-akh starter (`examples/three-akhs/`). Candle↔Ollama reconciliation + Phase 38e/f/g. **Selectable chat/agentic backing** — each akh picks Ollama (default; local/cloud, native tool calling) or Candle (in-process GGUF, *full* tool calling via grammar-constrained decoding `JsonConstraint`/`tool_call.gbnf`; device-limited), `ChatBackend` trait, ADR 070; forward-ported the candle boundary + ROCm/HIP AMD-GPU backend + Gemma-4 audio + hidden-states forks, ADRs 067/068/069. **Normalized** akhomed-store: JSON list-columns → edge/child tables (goal hierarchy/deps, method steps, provenance inputs, seed packs), ADR 072. **Phase 38e/f/g**: `SleepLevel` + HNSW local-disk wake cache; workspace-as-Postgres-row membership (`akh workspace join/leave/list-members`) + `WorkspaceJoin/Leave` protocol; relocation (`akh akh sleep/relocate`) with `last_seen` + Postgres advisory-lock split-brain guard + ops runbook + randomized recovery harness; the `akh` CLI command tree ADR 071. Deferred: 38d AGE (`projection refresh` stub), live `pg_notify` convergence + daemon auto-idle. Prior — Phase 38c: relational snapshot + wake/replay (Postgres truth, ADR 049); 38a/b `akhomed-store` + WAL; Pure-ACP **P8 COMPLETE**, ADR 061.)

## Overview

Akh-medu is a neuro-symbolic AI engine with a first-class agentic LLM loop. The symbolic core (VSA + KG + e-graphs) is CPU-native and can reason without any LLM; each akh additionally selects an LLM chat/agentic backing that drives the ACP harness. It hybridizes:

- **Vector Symbolic Architecture (VSA)** — 10,000-bit binary hypervectors for distributed representation
- **Knowledge Graphs** — dual-indexed (petgraph + oxigraph/SPARQL) for structured symbolic reasoning
- **E-graph Reasoning** — equality saturation via `egg` for symbolic rewriting
- **Autonomous Agent** — OODA-loop agent with 23+ tools, working/episodic memory, planning, reflection
- **Agentic LLM Loop** — `run_agentic_turn` over pure ACP with per-akh selectable chat backing (Ollama local/cloud, in-process Candle GGUF with grammar-constrained tool calling, OpenAI-compatible endpoints — ADRs 070/095/067) and swappable Phase 44 context-assembly strategies (`compacting`/`passive`/`virtual`, ADR 065)
- **Code Generation** — KG-to-Rust pipeline: code_gen tool, RustCodeGrammar, compiler feedback loop, parameterized templates, VSA code pattern encoding, pattern mining from examples, library learning cycle
- **Multilingual Grammar** — GF-inspired abstract/concrete syntax split for 5 languages
- **Four-Tier NLU** — rule parser → micro-ML NER → small LLM translator → VSA parse ranker (planned, Phase 14j-14m)
- **Content Library** — document ingestion (PDF, EPUB, HTML) with chunking and semantic enrichment (deprecated in favor of Seshat's Archive)
- **Seshat's Archive** (Phase 37) — shared corpus library backed by pgvector/PostgreSQL, dedicated MCP service (`crates/seshat/`), RAG retrieval + LLM microtheory synthesis (Candle/T5), multi-akh sharing
- **Tiered Storage** — hot (DashMap) → warm (mmap) → cold (redb) for scalability

## Module Map

```
src/
├── agent/              49 modules — OODA loop, tools (code_gen, code_ingest, compile_feedback, pattern_mine), memory, goals, drives, goal_generation, HTN decomposition, priority reasoning (argumentation), projects (microtheory-backed), planning, psyche, library learning, watch (GDA expectation monitoring), metacognition (Nelson-Narens monitoring/control, ZPD, AGM belief revision), resource awareness (VOC, CBR effort estimation), chunking (procedural learning), channel abstraction (CommChannel trait, ChannelRegistry, OperatorChannel), conversation (grounded dialogue, ConversationState, GroundedResponse), dialogue (KG-backed DialogueManager, DialoguePredicates, per-channel microtheory state), constraint_check (pre-communication constraint pipeline), interlocutor (social KG, InterlocutorRegistry, theory-of-mind microtheories, VSA interest vectors), oxifed (ActivityPub federation via AMQP bridge, feature-gated), explain (provenance-to-prose pipeline, DerivationNode trees, 5 query types), multi_agent (capability tokens, AgentProtocolMessage, TokenRegistry, trust bootstrap), pim (GTD + Eisenhower + PARA task overlay, petgraph dependency DAG, VSA priority encoding), causal (action schemas with preconditions/effects, pattern matching with variable binding, state transitions, outcome verification, VSA causal encoding), event_calculus (Phase 15b: temporal fluent reasoning, Event Calculus axioms, state projection, action sequence simulation, fluent history tracking), counterfactual (Phase 15c: Pearl Level 3 counterfactual reasoning, prediction tracking, causal model refinement), state_value (Phase 16a: TD-learned value function, compact state encoding, multi-factor rewards), mcts (Phase 16b: Monte Carlo Tree Search planner, UCT selection, R-MCTS reflection warm-start), plan_monitor (Phase 16c: plan execution monitoring, deviation detection, TD reward cycle, MCTS→Plan conversion), evidence (Phase 17: Dempster-Shafer mass functions, belief intervals, Dempster's rule combination, conflict detection, evidence fusion pipeline)
├── email/              9 modules — email channel (feature-gated): EmailConnector trait (JMAP/IMAP/Mock), MIME parsing (mail-parser), JWZ threading (RFC 5256), email composition (lettre), EmailChannel implementing CommChannel, EmailPredicates (14 well-known relations), OnlineHD spam classifier (VSA + Bayesian + deterministic rules), email triage & priority (sender reputation, four-feature importance scoring, VSA prototypes, HEY-style screening), structured extraction (regex + grammar hybrid, multi-language temporal/action NER, compartment-scoped KG persistence)
├── bootstrap/          10 modules — purpose/identity parser, Ritual of Awakening, domain expansion, curriculum ingestion, prerequisite analysis, ZPD competence assessment, resource discovery, bootstrap orchestrator, recipe, knowledge elicitation (Phase 32: KnowledgeElicitor + ElicitValidator, multi-pass T5 elicitation with self-consistency voting and 6-layer validation)
├── autonomous/          6 modules — background learning, confidence fusion, grounding
├── argumentation/       1 module  — pro/con argumentation (Phase 9e): meta-rules, verdicts, evidence chains
├── compartment/         5 modules — knowledge isolation, Jungian psyche, microtheories (Phase 9a, per-repo code scoping), CWA/circumscription (Phase 9m)
├── dispatch/            1 module  — competitive reasoner dispatch (Phase 9f): Reasoner trait, bid-based registry, 7 built-in reasoners
├── grammar/            21 modules — GF-inspired parsing/generation, entity resolution, Rust code gen (Phase 10a), templates (Phase 10e), aggregation rules (Phase 33a)
├── graph/              11 modules — KG (petgraph), SPARQL (oxigraph), analytics, predicate hierarchy (Phase 9b), defeasible reasoning (Phase 9d), arity constraints (Phase 9j), contradiction detection (Phase 9l), argumentation truth (Phase 9i), NARTs (Phase 9o), identity (ADR 081 P2: stable content-addressed TripleIdentity — blake3 over labels + compartment, cross-akh stable; engine-side index + `triple_by_identity`)
├── infer/               3 modules — spreading activation (with Phase 9 hierarchy + temporal context), backward chaining, superposition
├── library/            12 modules — document parsing, chunking, three-phase concept extraction (regex + standalone + NLU); seshat/ submodule for MCP client to Seshat service
crates/seshat/          — Seshat's Archive service: pgvector schema (SeaORM), MCP server (rmcp), ONNX embedding worker, optional Candle LLM synthesis, seshat-ingest CLI
crates/akh-egraph-bench/ — Phase 40 wake-path measurement harness (e-graph re-saturation / HNSW rebuild / fork arms; subprocess-per-cell; JSON + markdown reports) — permanent measurement infra per ADR 084
├── reason/              3 modules — e-graph language (AkhLang), rewrite rules, second-order quantification (Phase 9n), anti-unification (Phase 10h)
├── simd/                5 modules — runtime SIMD kernel dispatch (AVX2 / generic)
├── schema/              3 modules — ADR 081 schema packs: portable JSON source parser/validator (schema.org subset + Obsidian value types + is-a subtyping + onto: shapes), confirm-time compiler → native-triple bincode blob (`CompiledSchemaPack`, registry-row seam `SchemaRegistry`), per-akh install as microtheory with the global-immutable-type-name guard; `akh schema compile/show` CLI; seed pack `data/schema-packs/schema-org-core.json`
├── skills/              1 module  — skillpack lifecycle (Cold/Warm/Hot)
├── store/               3 modules — tiered storage (hot/warm/cold)
├── thinking/           10 modules — Phase 41 Thinking Engine (ADR 052): unified ThinkingLoop over existing primitives, ThoughtState (HyperVec focus + DS hypotheses + faithful trace), ThinkingBudget, schedulers (round-robin/weighted/goal-shape), CompositeVerifier (contradiction + evidence + source reliability + convergence), VocBudgetController (Phase 11g), ThoughtPersistence seam (41d Postgres deferred), 41f scaling-law bench harness
├── tui/                 6 modules — ratatui terminal UI, ACP remote backend
├── vsa/                 5 modules — HyperVec, VsaOps, encoding, item memory (HNSW), code pattern encoding (Phase 10f)
├── chat.rs                        — unified ChatProcessor (NLU-first AbsTree dispatch: dialogue acts, facts, queries, goals; DialogueManager integration)
├── acp/                           — ACP transport (agent-client-protocol): serve_unix/serve_tcp, control-plane (ServerControl), `_akh/*` data-plane, `session/prompt` → ChatProcessor bridge (ADR 059/060/061)
├── engine.rs                      — facade composing all subsystems (Phase9Config, 9 stored registries, wired add_triple/remove_triple pipeline)
├── error.rs                       — miette + thiserror rich diagnostics
├── rule_macro.rs                  — rule macro predicates (Phase 9g): RuleMacro trait, registry, genls/relationAllExists/relationExistsAll
├── temporal.rs                    — temporal projection (Phase 9k): TemporalProfile, decay computation, registry
├── provenance.rs                  — persistent explanation ledger (redb, multi-index, 60 derivation kinds)
├── skolem.rs                      — Skolem functions (Phase 9h): existential witnesses, grounding, auto-ground
├── tms.rs                         — truth maintenance system (Phase 9c): support sets, retraction cascades
├── symbol.rs                      — SymbolId (NonZeroU64), SymbolKind, allocator
├── pipeline.rs                    — composable stage pipelines
├── upstream_client.rs             — OUTBOUND MCP client to Anima/Seshat (ureq + JSON-RPC/SSE, no rmcp; currently dormant — see ADR-061)
├── service.rs                     — macOS launchd service management (plist generation, install/uninstall/start/stop/status)
└── main.rs                        — CLI (clap): control plane (init · workspace · daemon · config · acp) + chat
```

> Retired in P8 (ADR-061): `src/mcp/` (inbound MCP server), `src/api_types.rs`
> (REST DTOs), `data/web/` (HTMX/Alpine web UI), and the WebSocket TUI backend.

## Core Data Model

### Symbols
- **SymbolId**: `NonZeroU64` (niche-optimized for `Option` packing)
- **SymbolKind**: `Entity`, `Relation`, `Composite`, `Glyph(codepoint)`
- **AtomicSymbolAllocator**: thread-safe monotonic ID generator

### Triples
- `(subject: SymbolId, predicate: SymbolId, object: SymbolId)` with confidence, timestamp, provenance_id, compartment_id
- Stored in both petgraph (in-memory graph ops) and oxigraph (SPARQL queries)
- Each triple carries provenance linking back to how it was derived

### Hypervectors (VSA)
- 10,000-bit binary vectors, SIMD-accelerated (AVX2 with generic fallback)
- Operations: bind (XOR), unbind, bundle (majority vote), permute, similarity (Hamming)
- Item Memory: HNSW approximate nearest-neighbor search

## Reasoning Systems

| System | Strategy | Use Case |
|--------|----------|----------|
| Spreading Activation | Seeds → expand frontier via graph edges + VSA recovery | Forward inference, "what follows from X?" |
| Backward Chaining | Goal ← find supporting evidence recursively | Why-questions, evidence chains |
| Superposition | Parallel competing hypotheses, constructive/destructive interference | Multi-path exploration |
| E-graph Rewriting | `AkhLang` + `egg` equality saturation | Symbolic simplification, equivalence |
| Confidence Fusion | Noisy-OR and consensus across multi-source evidence | Combining evidence |
| Thinking Loop (Phase 41) | Budgeted rotation of all primitives over a `ThoughtState`, structural verifier halting | "Think harder" dial: `engine.deliberate(goal, budget)` |

### Phase 41 — The Thinking Engine (`src/thinking/`, ADR 052)

A unified deliberation loop that composes the existing primitives — spreading
activation (`Engine::infer`, one frontier ring per step; `infer_analogy` for
analogy-shaped goals), egg e-graph saturation, MCTS (16a), argumentation (9e),
counterfactual fork (15c), Dempster-Shafer fusion (17), and VSA superposition
merge — under a `ThinkingBudget` (steps / wall-clock / VOC credits) with a
faithful `ThoughtStep` trace. Goals are regular Entity symbols; optional
`thinking:*` relations (`seeks`, `relation`, `answer-type`, `analogy-a/b/c`,
`plan`) shape the query, mirroring the `agent:*` convention. The 41b scheduler
biases the rotation by goal shape (factual / analogy / planning / exploratory);
the 41c `CompositeVerifier` (Phase 9l contradictions + Phase 17 belief
intervals + Phase 18a source trust + focus-vector convergence) halts the loop
when the answer is structurally sufficient; the 41e `VocBudgetController`
(Phase 11g VOC) halts on diminishing returns or extends under high marginal
value. 41d persistence is a seam only (`ThoughtPersistence` trait + in-memory
impl; Postgres `akhomed.thinking_traces` deferred). Scaling-law results:
[ADR 085](decisions/085-thinking-engine-findings.md),
[2026-07-05 report](2026-07-05-phase41f-scaling-law.md).

### Phase 9 — Production Pipeline (Wired into Engine Lifecycle)

All Phase 9 systems are wired into the engine's `add_triple()` and `remove_triple()` paths, controlled by `Phase9Config` flags (all enabled by default):

**`add_triple()` pipeline** (pre/post hooks):
1. **Constraint check** — `ConstraintRegistry::check_triple()` rejects arity/type violations
2. **Contradiction detection** — `check_contradictions()` with configurable policy (Warn/Reject/Replace)
3. Core KG + SPARQL insertion
4. **Hierarchy invalidation** — `rel:generalizes`/`rel:inverse` triples mark hierarchy dirty for lazy rebuild
5. **Skolem auto-grounding** — `SkolemRegistry::auto_ground()` checks newly satisfiable existentials

**`remove_triple()` with TMS cascade**:
1. Core KG + SPARQL removal
2. **TMS retraction** — `TruthMaintenanceSystem::retract()` cascades to derived symbols
3. Cascaded triples removed from KG + SPARQL

**Inference integration** — `Engine::infer()` builds `InferPhase9Context`:
- Hierarchy-aware spreading activation (specialization + inverse predicates)
- Temporal decay of triple confidences based on `TemporalRegistry` profiles

**Persistence** — 7 registries serialized via bincode to `TieredStore::put_meta()`:
TMS, TemporalRegistry, ConstraintRegistry, FunctionalPredicates, DisjointnessConstraints, SkolemRegistry, NartRegistry. PredicateHierarchy rebuilt from KG on startup.

### Phase 9 — Cyc-Inspired HOL Enhancements (Module Status)

| System | Status | Description |
|--------|--------|-------------|
| **Microtheories** (9a) | Complete | `ist` operator, `genlMt` inheritance, lifting rules, 6 `ctx:` predicates, context-scoped queries, ancestry cache |
| **Predicate Hierarchy** (9b) | Complete | `genlPreds` subsumption, `genlInverse`, hierarchy-aware queries, transitive closure cache, 2 `rel:` predicates |
| **Truth Maintenance System** (9c) | Complete | Support sets with alternative justifications, BFS retraction cascade, re-evaluation, `remove_triple()` |
| **Defeasible Reasoning** (9d) | Complete | 5 override reasons (Monotonic, Specificity, Exception, Recency, Confidence), `DefeasiblePredicates` (4 well-known), BFS specificity, conflict resolution |
| **Pro/Con Argumentation** (9e) | Complete | 6 meta-rules (monotonicity, specificity, recency, depth, source quality, constructiveness), `Argument`/`ArgumentSet`/`Verdict` types, pro/con collection, weighted scoring, decisive rule detection |
| **Reasoner Dispatch** (9f) | Complete | `Reasoner` trait with bid-based dispatch, 7 built-in reasoners (spreading-activation, backward-chaining, superposition, egraph, transitive-closure, type-hierarchy, predicate-hierarchy), fallback on failure |
| **Rule Macro Predicates** (9g) | Complete | `RuleMacro` trait, `RuleMacroRegistry`, 3 built-in macros (Genls, RelationAllExists, RelationExistsAll), virtual expansion + query answering, 4 `macro:` predicates |
| **Arity & Type Constraints** (9j) | Complete | `ConstraintRegistry` with per-relation arity/arg-type declarations, `is-a` chain type checking (BFS), 3 `onto:` predicates, opt-in enforcement with diagnostic errors |
| **Temporal Projection** (9k) | Complete | `TemporalProfile` (Stable, Decaying, Ephemeral, Periodic), `TemporalRegistry` with default profiles, `apply_temporal_decay()`, filter-by-time, 1 `temporal:` predicate |
| **Contradiction Detection** (9l) | Complete | 4 contradiction kinds (functional, disjointness, temporal, intra-microtheory), `FunctionalPredicates`/`DisjointnessConstraints`, `check_contradictions()`, 2 `onto:` predicates |
| **Skolem Functions** (9h) | Complete | `SkolemSymbol`, `SkolemRegistry` with deduplication, create/ground/unground/auto_ground, check_grounding from KG, existential witness lifecycle |
| **Argumentation-Based Truth** (9i) | Complete | `ArgumentationCache` with verdict caching, `query_with_argumentation()`, cache invalidation by symbol or (subject, predicate), custom meta-rule queries |
| **Circumscription / CWA** (9m) | Complete | `ContextAssumptions` (CWA, UNA, circumscription), `AssumptionRegistry`, negation-as-failure queries, circumscribed instance enumeration, UNA entity identity, 3 `ctx:` predicates |
| **Second-Order Quantification** (9n) | Complete | `SecondOrderRule` with `RelationProperty` constraints, `SecondOrderRegistry` with 3 built-in rules (transitivity, symmetry, reflexivity), qualifying predicate enumeration, rule instantiation |
| **NARTs** (9o) | Complete | `NartDef` (function + args), `NartRegistry` with structural deduplication, structural unification with wildcards, function/arg lookup |

## Agent Architecture

OODA loop (synchronous, no async runtime):
1. **Observe** — scan KG for active goals, recall episodic memories
2. **Orient** — assess working memory, build context
3. **Decide** — utility-based tool scoring with recency penalty, novelty bonus, episodic hints
4. **Act** — execute selected tool, evaluate goal progress

Supporting infrastructure: working memory (ephemeral), episodic memory (consolidated), goal management with HTN decomposition (6+ built-in + learned methods, dependency DAGs, VSA-based method selection), multi-step planning with backtracking, periodic reflection, Jungian psyche model, autonomous goal generation (CLARION-inspired drives: curiosity, coherence, completeness, efficiency), metacognitive self-evaluation (Nelson-Narens monitoring/control, ZPD, competence tracking, AGM belief revision, e-graph goal reformulation), resource awareness (VOC-based goal switching, CBR effort estimation, dynamic stall thresholds, diminishing returns detection, opportunity cost recording), procedural learning (Soar-inspired chunking: trace extraction → generalization → method compilation, success/failure tracking, dormancy pruning).

## NLU Architecture (Phase 14j-14m)

Four-tier cascading pipeline for natural language understanding. Human language enters and exits at the boundary; all internal reasoning remains in VSA space.

```
Tier 1: Extended Rule Parser     — 0 MB,    <1ms,  handles ~70% of input
Tier 2: Micro-ML NER (ONNX)     — 130 MB,  ~5ms,  handles ~20% more (feature-gated: nlu-ml)
Tier 3: Small LLM (Qwen2.5-1.5B)— 1.1 GB,  ~800ms, handles remaining ~10% (feature-gated: nlu-llm)
Tier 4: VSA Parse Ranker         — 0 MB,    <1ms,  self-improving disambiguation
```

- **Tier 1** extends existing GF-inspired parser with negation, quantifiers, comparatives, conditionals, temporal, modals (all 5 languages)
- **Tier 2** uses DistilBERT multilingual NER via `ort` (ONNX Runtime) for entity boundary detection
- **Tier 3** uses Qwen2.5-1.5B-Instruct (Q4_K_M, Apache 2.0) via `llama-cpp-2` with GBNF-constrained decoding — can only output valid AbsTree JSON
- **Tier 4** encodes successful parses as VSA exemplars in HNSW ItemMemory; ranks ambiguous candidates by similarity to known-good parses

Total NLU memory: ~1.3 GB. Runs entirely on Mac Mini M2. No cloud API. FLOSS throughout.

### Input Processing (AbsTree Dispatch)

All user input (except slash commands) flows through `ChatProcessor::process_input()`, which runs the NLU pipeline and dispatches on the resulting `AbsTree` variant:

1. **Dialogue acts** (`Greeting`, `Farewell`, `Acknowledgment`, `FollowUpRequest`, `MetaQuery`, `GoalRequest`, `StructuralCommand`) route to the `DialogueManager`, which generates persona-aware responses and records state in a KG microtheory (`dialogue:<channel_id>`).
2. **Assertable structures** (triples, compounds, etc.) are ingested as facts via `TextIngestTool`.
3. **Query-like trees** are routed to the `ground_query` pipeline with discourse and synthesis.
4. **Freeform / unparseable input** is escalated to an autonomous goal.

The Tier 1 rule parser detects dialogue acts first via `try_dialogue_act()` using Lexicon word lists, before attempting structural or relational parses. The GBNF grammar also covers dialogue acts so Tier 3 (LLM) can produce them via constrained decoding.

### Communicative Intelligence (Phase 34, ADR 044)

On the grounded query path, `ChatProcessor` runs the communicative pipeline before constraint checking — one wiring point covers TUI, daemon WS, and headless CLI (ADR 024). Three symbolic stages, no LLM:

1. **Response strategy selection** (`agent/response_strategy.rs`, 34a) — an 18-strategy taxonomy (Knowledge / Epistemic / Conversational / Meta-cognitive) scored by a deterministic function over the `StrategyContext`: answer confidence, Dempster-Shafer conflict + belief interval (Phase 17 primitives), topic repetition, discourse mode, engagement, and user sentiment (Phase 23 affect) — modulated by the psyche's OCEAN profile and archetype weights. A conversational companion move (e.g. `DirectAnswer` + `Empathy`) can ride along.
2. **Content filter** (34c) — each strategy selects its own slice of the grounded triples (top-3 for direct answers, taxonomic-first for contextualization, pro/con partition for evidence weighing, derivation-tagged for reasoning traces).
3. **Psyche style transform** (`grammar/style_transform.rs`, 34b) — OCEAN → `CommunicationRegister` (formality, hedging, elaboration, warmth, metaphor), dominant archetype → opening/closing moves + characteristic phrases, all deterministic string rewrites.

The **shadow filter** (34e) sits between stages 1 and 2: shadow bias patterns soften assertive strategies (`Challenge` → `EvidenceWeighing`) and journal the encounter to the audit log. The **discourse trajectory** (`agent/trajectory.rs`, 34d) feeds selection: repeated topics switch strategy, argumentative mode favours evidence weighing, dropping engagement favours exploration prompts. Composition lives in `agent/communicative.rs` (34f); disable per-processor via `set_communicative(false)` or env `AKH_COMM_INTEL=0`. `concise` response detail bypasses the pipeline. Demo: `cargo run --example phase34_demo`.

**Deprecated**: `classify_intent()`, `classify_conversational()`, and `ConversationalKind` in `nlp.rs` are deprecated (ADR 025). They remain as a structural-command fast-path but will be removed once all callers migrate to AbsTree dispatch.

### Server-Side NLU

The akhomed WebSocket handler initialises an `NluPipeline` per session, restoring the VSA parse ranker from durable storage. All input goes through the NLU pipeline:

1. NLU parse succeeds → AbsTree variant dispatch (dialogue acts, facts, queries, goals)
2. NLU parse fails → input is escalated to a goal for the agent to investigate

This means `client-only` builds get the full NLU capability without any local ML models — the server does all the parsing. The available tiers depend on the server's feature flags:

| Server features | Available tiers |
|-----------------|-----------------|
| `server` | Tier 1 (rules) + Tier 4 (VSA ranker) |
| `server,nlu-ml` | + Tier 2 (ONNX NER) |
| `server,nlu-full` | + Tier 2 + Tier 3 (LLM translator) |

The ranker state is persisted per-workspace via `engine.store().put_meta(b"nlu_ranker_state", ...)` on session disconnect, shared with the TUI's ranker.

### NLU-Enriched Document Ingestion

The library document ingestion pipeline (`src/library/ingest.rs`) uses a three-phase concept extraction strategy in `run_concept_extraction()`:

- **Phase A (relational):** Regex-based triple extraction (`extract_triples` + `extract_richer_triples`) handles "X is-a Y", "X has Y", conjunction splitting, bare verbs, and adverb modulation.
- **Phase B (standalone):** Capitalized proper nouns, technical compounds, and repeated terms not captured by Phase A.
- **Phase C (NLU pipeline):** When an `NluPipeline` is available, each sentence is parsed through the 4-tier cascade. This captures constructs Phase A cannot: negation, conditionals, temporal scoping, modals, comparatives, quantifiers, and relative clauses. Parsed `AbsTree` structures are grounded against the symbol registry and committed via `engine.commit_abs_tree()`.

Phase C deduplicates against Phase A using shared `seen_triple_keys`. Parse failures are non-fatal — Phase A/B results stand.

The `ingest_document`, `ingest_file`, and `ingest_url` functions accept an optional `nlu: Option<&mut NluPipeline>` parameter. Call sites that lack NLU (bootstrap, inbox, agent tools, client-only) pass `None`. The akhomed `library_add_handler` initialises an NluPipeline from the workspace's persisted ranker state and persists it after ingestion.

ADR: `docs/ai/decisions/022-nlu-architecture.md`

## ACP Agentic Harness (Pure-ACP — sole transport; P0–P8 complete)

ACP — the **Agent Client Protocol** over a Unix socket and optional TCP — is
`akhomed`'s single inbound transport. `akhomed` is the ACP **agent**; the TUI,
CLI, and Anima are ACP **clients** (ADR-059). The legacy surfaces it once served
alongside ACP — a ~104-route REST API, WebSocket chat, a bundled HTMX/Alpine web
UI, and an inbound MCP server — were deleted in **P8** (ADR-061), and
`axum`/`tower-http`/`tokio-tungstenite`/`reqwest`/`rmcp` are no longer akh-medu
dependencies. Outbound MCP to Anima/Seshat stays (`src/upstream_client.rs`).

The 8 P8 increments: P8.1 ACP control-plane (`ServerControl`) · P8.2 ACP-native
discovery probe · P8.3 trim CLI to control-plane + chat · P8.4 delete
REST/WS/web-UI · P8.5 delete inbound MCP server · P8.6 drop axum/REST deps +
collapse the feature graph (`acp` is now a **default** feature; `server` =
`daemon + acp`; `client-only` retired) · P8.7 K8s/Helm → ACP-TCP (tcpSocket
probes, XDG data path, exposed TCP) · P8.8 bounded connect-retry hardening.

The former `client-only` build mode (a thin HTTP `akh` against a remote akhomed)
is retired with REST — `akh` talks to a daemon over ACP (`daemon`/`config`/`acp`
control commands), discovered via the PID record's ACP socket/TCP endpoints.

- **Transport** (`src/acp/`): `agent-client-protocol` 0.14; `session/prompt` →
  the LLM agentic loop (`run_agentic_turn`, else symbolic `ChatProcessor`) run in
  `spawn_blocking` (ADR-027). The turn now **streams**: each `AkhMessage` is
  emitted to a `ChannelSink` the instant it is produced and the prompt handler
  drains it concurrently into `session/update` notifications (answer →
  `AgentMessageChunk`, reasoning + tool steps → `AgentThoughtChunk`); the client
  consumes them incrementally via `read_update`. Token-level answer streaming
  rides Ollama's `/api/chat` `stream:true` (`OllamaClient::chat_stream`, content
  vs `thinking` deltas). A `_akh/<group>/<op>` data-plane carries read/admin ops
  (meta, workspace, symbols, triples, query, memory, infer, permission, model)
  over the same connection; cognitive ops are the agent LLM's *tools*, not
  client methods (the data-plane is deliberately small).
- **Phase 29 capability exposure** (plan
  [2026-03-26-phase29-capability-exposure](plans/2026-03-26-phase29-capability-exposure.md),
  [ADR 039](decisions/039-capability-exposure.md) — the plan's "MCP tools +
  REST dashboard" surface predates ADR 061's ACP-only transport and is realized
  as `_akh/*` groups + `akh acp` verbs): `microtheory/{list,create,inspect,
  activate,deactivate}` (29a, compartment lifecycle), `skills/{inspect,activate,
  deactivate}` (29b, Cold↔Hot skillpacks), `psyche` + `psyche/archetypes` (29c,
  read + clamped weights-only archetype tuning under the agent lock),
  `argue` (29d, pro/con verdicts from provenance chains), `explain` +
  `provenance` (29e, derivation trees + `ExplanationQuery` free-text routing),
  `evidence/assess` (29h, per-claim Dempster-Shafer) and `trust` (29i,
  Admiralty leaderboard). Mirrored 1:1 by `akh acp
  microtheory|skill|psyche|argue|explain|provenance|evidence|trust`. 29f
  (hypothesis explorer) and 29g (causal explorer) remain unexposed.
- **TUI** (`src/tui/`): pure-ACP ratatui client. Streams answer deltas into one
  growing message, folds thinking/tool steps into a collapsible block (Tab),
  renders answers as Markdown (`tui/markdown.rs`), and shows an animated spinner
  in the input border while a turn is in flight. `/model` (picker + type-ahead
  filter) and `/effort` switch the live model over `_akh/model/*`.
- **First-class akh** (`src/akh_config.rs`, ADR-053): `akh.toml` pins
  `{model + ollama endpoint + system prompt}` per workspace.
- **Filesystem sandbox** (`src/fs/`, ADR-060): a `Filesystem` trait with a
  `LocalFs` backend confining reads/writes to separate allow-listed roots with
  symlink-safe canonical containment (no-follow-symlinks walks, depth caps).
  All file-touching tools go through `Arc<dyn Filesystem>` — closing the old
  read-anywhere / parent-only-write holes.
- **Executor `Backend` seam** (`src/backend/`, ADR-082, Phase 48a): a unified
  `Backend: Filesystem` trait adding `exec(ExecRequest) -> ExecOutcome` +
  `handshake() -> Posture`, so the executor-facing tools
  (`file_io`/`edit`/`glob`/`grep`/`shell_exec`) bind files *and* process
  execution to one target. Sole 48a impl: `InProcessBackend` (wraps the
  `LocalFs` sandbox + the historic in-process `/bin/sh -c` exec, byte-identical;
  handshake honestly reports `Unconfined`, keeping `shell_exec` Critical).
  48b `LocalDrop` plugs in behind the same trait via the tools' `with_backend`
  constructors. **48c `RemoteBackend`** (`src/backend/remote.rs`, feature
  `remote-executor`, Phase 50/W2a) is **built**: a russh-over-SSH executor —
  `exec` runs the SSH exec-channel request capturing stdout/stderr/exit into
  the same `ExecOutcome`; the `Filesystem` verbs run as shell commands
  (`cat`/`base64`/`ls`/`test`/`wc`) over that same channel on the remote FS
  (not SFTP — ADR-082 §2 leaves this open, shell keeps it single-protocol/
  single-dependency); auth is **plain provisioned keys** (`HostKeyPolicy` =
  KnownHosts / Pinned / InsecureAcceptAny; 48d OIDC→cert deferred per ADR-086
  §1); no client-side path sandbox (the OS is the sole boundary, ADR-082 §8);
  handshake reports the new `Confinement::RemoteUnverified` (policy-equivalent
  to Unconfined until 48f's full self-report). The synchronous `Backend` trait
  bridges async russh by owning a dedicated multi-thread tokio runtime and
  `block_on`-ing each call (same sync-over-async shape as `main.rs`/`tui`);
  the trait signature is unchanged. Tested against an in-process russh **server**
  fixture over a `tokio::io::duplex` pipe (real SSH wire, not a mock).
- **Claude-Code tool parity** (ADR-060): `ToolInput` re-backed by
  `serde_json::Value` (string API preserved); new Edit / Glob / Grep /
  WebSearch / structured `task_list` tools, all sandbox-routed. Tool count 34.
- **Permission layer** (`src/agent/permission.rs`, ADR-060): a sync
  `PermissionPolicy` (Bypass default / Standard / AcceptEdits / Plan) gates
  side-effecting tools at `ToolRegistry::execute`; a supervising client sets the
  mode via `_akh/permission/mode` and resolves interactive `Ask` decisions over
  `session/request_permission` (bridged across the spawn_blocking↔async boundary).

## Storage Architecture

```
Hot  (DashMap)     — sub-microsecond, volatile
Warm (mmap)        — memory-mapped files, persistent, fast reads
Cold (redb)        — ACID transactions, durable, slower writes
Provenance (redb)  — multi-index ledger (derived/source/kind)
SPARQL (oxigraph)  — persistent RDF store for structured queries
```

### Postgres backing store (Phase 38, `crates/akhomed-store`)

Phase 38 makes an Akh **relocatable**: the truth of its state lives in Postgres
as **typed relational data** (no opaque blobs, no object store) while the daemon
keeps the stores above as an in-process cache. 38a–38c are implemented.

- New leaf crate `crates/akhomed-store/`: SeaORM entities + `sea-orm-migration`
  migrations (schema-builder, not raw SQL), a pool builder, and the
  `akhomed-migrate` CLI. The crate depends on nothing in `akh-medu`.
- **Backend-portable schema** (Postgres now; SQLite / libSQL-**Turso** droppable
  later): flat prefixed table names (`akhomed_*` / `workspaces_*`, **no** PG
  schemas — SQLite has none), list-valued fields as SeaORM `Json` (no native
  arrays), no DB-side defaults (UUIDs/timestamps generated in Rust). Postgres-only
  optimizations are conditional add-ons: the `mutation_log` **TimescaleDB
  hypertable** (probes `pg_available_extensions`; plain table where absent) and
  an **AGE** read-projection over `facts` (deferred, 38d). ADR 049.
- Tables: `akhomed_akh` (host-independent identity), `akhomed_facts` (KG triples,
  content-addressed PK; AGE projects over this), `akhomed_symbols` (registry rows
  — joins facts to labels), `akhomed_mutation_log` (the streaming WAL — 38b),
  `akhomed_akh_state` (one runtime row/akh: watermark, allocator, cycle_count,
  sleep phase, `psyche`/`extra` JSONB), the **cognitive tables** (`drives`,
  `agenda`, `curiosity_config`, `goals`, `failure_cases`, `effort_cases`,
  `learned_methods`, `competence_{tools,categories,calibration}`, `sleep_metrics`
  — inspectable/queryable), plus `provenance` / `episodes` and the `workspaces_*`
  membership tables (38f).
- Gated behind the `postgres` cargo feature (folded into `server`); the lean
  `akh` client never compiles SeaORM. Runtime-optional: with no `[postgres]`
  config section the daemon logs a warning and runs in legacy local-only mode.
- `[postgres]` config: `url`, `pool_size`, `connect_timeout_secs`,
  `migrate_on_start` (`DATABASE_URL` overrides `url`).

**Sub-phase 38b (implemented) — the mutation WAL.** Every successful
assert/retract streams to `akhomed_mutation_log` within seconds; wake (38c)
replays the tail. Bounded loss on crash, no loss while Postgres is down.

- `src/mutation.rs` (core, ungated): a tiny `MutationSink` trait + `MutationKind`
  taxonomy (assert/retract/microtheory_load/unload/compartment_create/destroy).
  `Engine` holds `OnceLock<Arc<dyn MutationSink>>` and calls it after a
  successful `add_triple` / `remove_triple`. The core carries **no** SeaORM
  dependency — the lean `akh` client is unaffected.
- `src/wal.rs` (postgres-gated): `MutationLog` implements `MutationSink`. A
  synchronous, byte-capped `WalBuffer` (overflow drops oldest non-`assert`
  first) feeds an async flusher that batches inserts every `flush_interval`.
  Failed flushes re-insert the batch at the front and retry (at-least-once;
  replay is idempotent). Stable per-workspace Akh/workspace UUIDs via UUIDv5.
- Daemon: `ServerState::wire_wal` installs the WAL per workspace on first load
  (ensuring the `akhomed_akh` row first); flushers drain on graceful shutdown.
  WAL tuning in `[postgres]`: `wal_flush_interval_secs` (2), `wal_batch_size`
  (64), `wal_max_buffer_mb` (16). WAL **retention/truncation** is deferred to a
  later SleepCycle self-consolidation phase (38c does **not** truncate).

**Sub-phase 38c (implemented) — relational snapshot + wake/replay.** The akh's
DB is mechanical; an LLM facts-synthesis layer (off-time WAL→facts + a Curator
Ennead-akh over Seshat) is deferred.

- `src/snapshot.rs` (postgres-gated): `capture_snapshot` (sync — materialises the
  live registry, KG, and cognitive subsystems into entity `Model`s; held under
  the agent's `std::sync::Mutex` only for the fast in-memory copy) + the async
  `persist_snapshot` (full-replaces the cognitive tables + upserts `akh_state`
  with `watermark = MAX(mutation_log.seq)`). `wake(data_dir, akh_id,
  workspace_id, db)` inverse-maps the rows back into a fresh host's redb meta
  (`sym_meta:` registry + the `agent:*` bincode blobs the unchanged
  `Agent::resume` reads), opens `Engine::new`, `restore_triples(facts)`, then
  replays the WAL tail (`seq > watermark`) **before** the sink is wired (no
  re-log). Goals are stored as full JSON in `goals.detail` (lossless across
  data-bearing status variants); `session_summary`/`audience`/
  `improvement_history`/`method_stats` ride in `akh_state.extra`. Managers
  (PIM/calendar/causal/contact/…) and working memory rebuild from the KG /
  defaults; a missing optional table degrades gracefully. The Phase 25c
  `person_memory` index rides in `extra` as a JSON-safe `PortablePersonMemory`
  projection (its `(contact, topic)` tuple key is flattened to a pair sequence).
- Daemon: a per-workspace **snapshot ticker** (`spawn_snapshot_ticker`, alongside
  `wire_wal`) snapshots every `snapshot_interval_secs` (default 300, `0`
  disables); `shutdown_wals` stops tickers, takes a **final snapshot** per loaded
  workspace, then drains the WAL. ACP control ops `_akh/akh/snapshot` /
  `_akh/akh/wake` (CLI `akh akh snapshot|wake <ws>`) drive it manually; wake
  routes a cold workspace through the normal lazy `get_agent` path so resume +
  model-manager + WAL all apply.

## Provenance

Every inference, agent decision, and knowledge derivation creates a `ProvenanceRecord`:
- Derived symbol, derivation kind (50 variants), confidence, depth, source symbols, metadata
- Full traceback from any result to its original sources
- Indices by derived symbol, source symbol, and kind for fast lookup

## Audit Ledger

Append-only activity log (`src/audit.rs`) recording all tool invocations, content ingestions, and agent-initiated log messages:
- `AuditLedger` backed by redb tables (`audit` + `audit_kind_idx`) sharing the same `Database` as provenance
- Sequential `AuditId(NonZeroU64)` keys, `AuditKind` discriminator (ToolInvocation, ContentIngestion, AgentLog)
- **No delete/update/truncate** — append-only invariant enforced by API surface
- Automatic tool invocation auditing via `ToolRegistry::execute()` wrapper (timing, params/output summaries)
- Content ingestion auditing emitted from `library/ingest.rs` after successful ingest
- `LogTool` (`audit_log`) lets the agent voluntarily write messages at info/warn/debug levels
- `tokio::sync::broadcast` channel pushes entries to WS clients in real-time
- `MessageSink` integration pushes `AkhMessage::AuditLog` to TUI live (togglable via `/audit`)
- REST endpoints: `GET /workspaces/{ws}/audit` (paginated, filterable by kind), `GET /workspaces/{ws}/audit/{id}`, `GET /workspaces/{ws}/goals`

## Development Phases

Phases 1–7: Engine foundation (VSA, KG, reasoning, storage, provenance, inference, pipeline, skills)
Phase 8a–8f: Agent evolution (wiring, goals, decision-making, persistence, external tools, planning)
Phase 9a–9o: Cyc-inspired HOL enhancements (15 sub-phases):
- **High**: 9a microtheories, 9b predicate hierarchy, 9c TMS, 9d defeasibility, 9e argumentation, 9f reasoner dispatch
- **Medium**: 9g rule macros, 9h skolem functions, 9i arg-based truth, 9j arity/types, 9k temporal projection, 9l contradiction detection
- **Lower**: 9m circumscription/CWA, 9n second-order quantification, 9o NARTs
Phase 10a–10h: Rust code generation (8 sub-phases):
- **Core (Wave 1 complete)**: 10a RustCodeGrammar, 10b code_gen tool, 10c code-aware planning, 10d iterative refinement
- **Pattern infrastructure (Wave 2 complete)**: 10e parameterized templates (7 built-in), 10f VSA code pattern encoding (path-contexts, multi-granularity)
- **Infra**: code_ingest per-repo microtheory scoping (mt:repo:<name> specializes mt:rust-code, ContextDomain::Code, clean re-ingestion)
- **Pattern learning**: 10g pattern mining from examples, 10h library learning cycle
Phase 11a–11h: Autonomous task system (8 sub-phases, all complete):
- **Core (Waves 1–3)**: 11a drive-based goal generation with impasse detection (CLARION/GDA/Soar), 11b HTN decomposition with dependency DAGs (6 built-in methods, VSA method selection, petgraph TaskTree), 11c value-based argumentation priority (Dung/VAF), 11d projects as microtheories with Soar/ACT-R memory, 11e GDA expectation monitoring with pattern-based KG watches, VSA semantic trigger matching, fluent-style state tracking
- **Meta (Wave 4)**: 11f metacognitive self-evaluation (Nelson-Narens monitoring/control, CompetenceModel, HNSW failure patterns, ZPD scoring, autoepistemic goal questioning, AGM belief revision with entrenchment cascade, e-graph goal reformulation)
- **Economic (Wave 5)**: 11g VOC resource reasoning (CBR effort estimation via HNSW EffortIndex, compute_voc, dynamic stall thresholds, diminishing returns detection, marginal-value goal ranking, opportunity cost recording, project budget tracking), 11h procedural learning/chunking (Soar-inspired trace extraction → generalization → method compilation, HNSW MethodIndex, success/failure tracking with retraction, HTN registry integration, dormancy pruning, compilation opportunity detection)
Phase 12a–12g: Interaction — communication protocols and social reasoning (7 sub-phases):
- **Core (12a–12d complete)**: 12a channel abstraction (CommChannel trait, ChannelRegistry with operator invariant, OCapN-inspired ChannelCapabilities per ChannelKind, OperatorChannel wrapping MessageSink with InboundHandle, TUI+headless wired transparently), 12b grounded operator dialogue (ConversationState with bounded turn history, ResponseDetail levels, GroundedResponse with provenance + confidence, ground_query pipeline, grounded-first query path in TUI+headless, SetDetail intent), 12c pre-communication constraint checking (6-check pipeline: consistency/confidence/rate/relevance/sensitivity/provenance, ConstraintChecker with CommunicationBudget, per-channel-kind emission decisions, SensitivityLevel, CheckOutcome, ConstraintCheckStatus evolution), 12d social KG with theory of mind (InterlocutorRegistry, per-interlocutor microtheories via Phase 9a, InterlocutorPredicates with 6 well-known relations, VSA interest bundling, Hamming-based similarity search, trust-level management with operator immutability, auto-registration in TUI+headless)
- **Federation (12e complete)**: 12e ActivityPub federation via oxifed (OxifedChannel implementing CommChannel with ChannelKind::Social, AMQP consumer/publisher background tasks via lapin, serde-compatible oxifed message types, activity↔InboundMessage bridge, constraint-checked outbound Notes, OxifedConfig, feature-gated under `oxifed`)
- **Transparency (12f complete)**: 12f transparent reasoning and explanations (ExplanationQuery with 5 query types: Why/How/WhatKnown/HowConfident/WhatChanged, DerivationNode tree built by recursive provenance walk, render_derivation_tree for indented hierarchy rendering, render_derivation_prose for concise output, derivation_kind_prose covering all 50 DerivationKind variants, explain_entity/explain_known/explain_confidence/explain_changes, ExplanationQuery::parse for NL recognition, Explain UserIntent variant, wired into TUI+headless chat)
- **Multi-Agent (12g complete)**: 12g multi-agent communication with OCapN-inspired capability tokens (CapabilityToken with scoped permissions, expiry, revocation; 6 CapabilityScope variants; TokenRegistry with pair indexing and validation; AgentProtocolMessage with 10 structured message types: Query/QueryResponse/Assert/ProposeGoal/Subscribe/Unsubscribe/GrantCapability/RevokeCapability/Ack/Error; InterlocutorKind Human/Agent on InterlocutorProfile; MessageContent::AgentMessage variant bypassing NLP; UserIntent::AgentProtocol; can_propose_goals capability flag; trust bootstrap via operator introduction)
Phase 13a–13i: Personal assistant (9 sub-phases):
- **Email (13d complete)**: 13a email channel (JMAP/IMAP + MIME + JWZ threading), 13b OnlineHD spam classification (VSA-native), 13c email triage & priority (sender reputation + HEY-style screening), 13d structured extraction (regex + grammar hybrid, multi-language temporal/action NER, compartment-scoped KG)
- **PIM (13e complete)**: 13e personal task & project management (GTD + Eisenhower + PARA, petgraph dependency DAG, VSA priority encoding, e-graph PIM rules, CLI commands)
- **Calendar (13f complete)**: 13f calendar & temporal reasoning (Allen interval algebra, iCalendar import, CalDAV sync, conflict detection, VSA temporal patterns)
- **Intelligence**: 13g preference learning & proactive assistance (HyperRec-style VSA profiles, serendipity engine), 13h structured output & operator dashboards (JSON-LD, briefings, notifications)
- **Delegation**: 13i delegated agent spawning (scoped knowledge, own identity, email composition pipeline)

### Phase 13a — Email Channel (JMAP/IMAP + MIME) ✓
- [x] `EmailError` miette diagnostic enum (7 variants: Connection, Authentication, Parse, Send, Threading, Config, Engine) with `EmailResult<T>` alias
- [x] `EmailConnector` trait: `fetch_new()`, `fetch_by_id()`, `send_email()`, `sync_state()` — with RawEmail, EmailConfig, EmailCredentials
- [x] `JmapConnector` — JMAP over ureq (sync HTTP), session discovery, Email/query + Email/get, delta sync via Email/changes
- [x] `ImapConnector` — sync IMAP via `imap` crate with `native-tls`, TLS connection, UID-based delta sync
- [x] `MockConnector` — in-memory queue for testing (`push_raw()`, `mock_send()`)
- [x] `ParsedEmail` struct (15 fields) with `parse_raw()` via `mail-parser` — multipart/alternative, multipart/mixed, nested MIME, 4KB text / 8KB HTML truncation
- [x] `extract_domain()` utility for email address domain extraction
- [x] JWZ threading (RFC 5256): `ThreadNode`, `ThreadTree`, `build_threads()` — 5-step algorithm with cycle protection, phantom parent nodes
- [x] `ComposedEmail` with `compose_reply()` (In-Reply-To, References chain, Re: prefix, quoted body), `compose_new()`, `to_mime()` via lettre
- [x] `EmailPredicates` — 14 well-known relation SymbolIds (message-id, from, to, cc, subject, date, thread-id, in-reply-to, has-attachment, content-type, body-text, list-id, dkim-pass, spf-pass)
- [x] `EmailInboundHandle` — cloneable `Arc<Mutex<VecDeque<InboundMessage>>>` with `push_email()` converting ParsedEmail → InboundMessage
- [x] `EmailChannel` implementing `CommChannel` — ChannelKind::Social, background std::thread polling, AtomicBool connected/shutdown, Drop cleanup
- [x] `DerivationKind::EmailIngested` (tag 48) and `DerivationKind::EmailThreaded` (tag 49) provenance variants
- [x] Feature-gated: `--features email` (adds `mail-parser`, `imap`, `native-tls`, `lettre`)
- [x] `AgentError::Email` transparent variant (cfg-gated)
- [x] 62 new unit tests across 6 modules

### Phase 13b — OnlineHD Spam & Relevance Classification ✓
- [x] `SpamDecision` enum: Spam, Ham, Uncertain — with Display, Serialize/Deserialize
- [x] `ClassificationResult` — decision + vsa_spam_similarity + vsa_ham_similarity + bayesian_score + confidence + rule_override + reasoning
- [x] `SpamRoleVectors` — 7 deterministic role HyperVecs (sender, domain, subject, body, has_attachments, has_list_id, time_bucket) via `encode_token(ops, "email-role:X")`
- [x] `TokenProbabilityTable` — per-token spam/ham counts, Robinson chi-square combination, MAX_TOKEN_TABLE_SIZE eviction
- [x] `SpamClassifier` — OnlineHD prototype vectors (spam/ham) + Bayesian supplement + whitelist/blacklist + persistence
- [x] `encode_email()` pipeline: 6-feature role-filler binding (domain, subject, body, attachments, list-id, time bucket) → bundle
- [x] `classify()` pipeline: deterministic rules → VSA similarity → Robinson chi-square → combined score (0.7 VSA + 0.3 Bayesian) → threshold
- [x] `train()` — OnlineHD adaptive update via majority-vote bundling + token table training
- [x] Whitelist/blacklist domain management with case-insensitive matching and dedup
- [x] `persist()`/`restore()` via bincode + `put_meta`/`get_meta` on engine's durable store
- [x] `record_classification_provenance()` — `DerivationKind::SpamClassification` (tag 50)
- [x] 24 new unit tests

### Phase 13c — Email Triage & Priority ✓
- [x] `EmailRoute` enum: Important, Feed, PaperTrail, ScreeningQueue, Spam — with Display, Serialize/Deserialize
- [x] `SenderRelationship` enum: Colleague, Friend, Service, Newsletter, Unknown — with Display, Serialize/Deserialize, weight()
- [x] `SenderStats` — per-sender reputation: address, message_count, reply_count, reply_rate (EMA), avg_reply_time_secs (EMA), relationship, routing, symbol_id
- [x] `TriageRoleVectors` — 8 deterministic role HyperVecs via `encode_token(ops, "triage-role:X")`
- [x] `ImportanceWeights` — configurable social/content/thread/label weights (default 0.35/0.25/0.20/0.20)
- [x] `TriagePredicates` — 7 well-known KG relations (sender: namespace)
- [x] `TriageEngine` — sender stats HashMap + OnlineHD important/low-priority prototypes + role vectors + weights
- [x] Four-feature importance scoring: social (reply_rate, frequency, recency, relationship), content (VSA prototype similarity), thread (in_reply_to, references depth), label (operator-assigned route)
- [x] HEY-style screening: `needs_screening()` → ScreeningQueue for first-time unrouted senders
- [x] `encode_email()` — 7-feature triage-oriented role-filler binding → bundle
- [x] `train_important()`/`train_low_priority()` — OnlineHD adaptive update via majority-vote bundling
- [x] `persist()`/`restore()` via bincode + `put_meta`/`get_meta`
- [x] `record_triage_provenance()` — `DerivationKind::EmailTriaged` (tag 51)
- [x] `sync_sender_to_kg()` — KG triple sync for SPARQL queryability
- [x] 26 new unit tests

### Phase 13d — Structured Email Extraction ✓
- [x] `ExtractedItemKind` enum: Date, RelativeDate, TrackingNumber, Url, PhoneNumber, ActionItem, EmailAddress — with Display, Serialize/Deserialize
- [x] `SourceField` enum: Subject, BodyText — with Display, Serialize/Deserialize
- [x] `ExtractedItem` — kind + raw_text + normalized + symbol_id + offset + confidence + source_field + language
- [x] `ExtractionResult` — items + counts + reasoning + detected_language
- [x] `ExtractionPredicates` — 8 well-known KG relations (extract: namespace)
- [x] `ExtractionScope` — account_compartment + correspondent_compartment for microtheory scoping
- [x] `ActionItemGoalSpec` — goal specification from action items (does NOT create goals)
- [x] 10 regex patterns (LazyLock): ISO dates, US/EU dates, written dates, UPS/FedEx/USPS tracking, URLs, phones, emails
- [x] FedEx false-positive mitigation: context keyword gating within 100-char window
- [x] `extract_temporal_via_grammar()` — multi-language relative date extraction (EN/RU/FR/ES/AR) + "in N days/weeks" patterns
- [x] `extract_actions_via_grammar()` — multi-language action item extraction (EN/RU/FR/ES/AR) with urgency boost
- [x] `extract_all()` — full pipeline: regex + grammar on subject + body, deduplication by (kind, normalized)
- [x] `ensure_extraction_scope()` — create account + correspondent microtheories via `engine.create_context()`
- [x] `store_extractions()` — compartment-scoped KG triples + carrier triples for tracking numbers
- [x] `record_extraction_provenance()` — `DerivationKind::EmailExtracted` (tag 52)
- [x] `action_items_to_goals()` — goal specs with multi-language urgency detection
- [x] Quick predicates: `has_action_items()`, `has_calendar_event()`, `has_shipment_info()`
- [x] ~26 new unit tests

### Phase 13e — Personal Task & Project Management ✓
- [x] `PimError` miette diagnostic enum (5 variants: TaskNotFound, InvalidTransition, CycleDetected, RecurrenceParse, Engine) with `PimResult<T>`
- [x] `GtdState` enum: Inbox, Next, Waiting, Someday, Reference, Done — with validated transitions via `can_transition_to()`
- [x] `EisenhowerQuadrant` enum: Do, Schedule, Delegate, Eliminate — with `classify(urgency, importance)` and `priority_bonus()`
- [x] `ParaCategory` enum: Project, Area, Resource, Archive — with as_label/from_label
- [x] `PimContext`, `EnergyLevel`, `Recurrence` enums — contexts (home/office/computer/phone/errands/anywhere), energy levels, RRULE-lite patterns (daily, weekly, monthly, yearly, every N days)
- [x] `PimMetadata` struct (14 fields) — per-goal overlay with GTD state, urgency/importance, quadrant, PARA, contexts, energy, time estimate, deadline, recurrence, next_due, last_completed
- [x] `PimPredicates` — 14 well-known KG relations in `pim:` namespace (gtd-state, context, energy, time-estimate, urgency, importance, para-category, deadline, quadrant, blocked-by, blocks, recurrence, next-due, last-done)
- [x] `PimRoleVectors` — 6 deterministic VSA role vectors for priority encoding via `encode_token(ops, "pim-role:X")`
- [x] `PimManager` — HashMap-based metadata + petgraph DiGraph DAG + predicates + role vectors; add_task, transition_gtd, update_eisenhower, set_para/context/energy/recurrence, available_tasks, add/remove_dependency, topological_order, critical_path, ready_tasks, process_recurring_completions, overdue_tasks, tasks_by_gtd_state/quadrant/para, encode_priority, find_similar_priority
- [x] `SerializableDag` — custom Serialize/Deserialize for petgraph DiGraph (nodes + edges vectors)
- [x] `GtdReviewResult` + `gtd_weekly_review()` — weekly review: stale_inbox, waiting_items, someday_candidates, overdue, stalled_projects, migration_candidates, adjustments, summary
- [x] `action_items_to_pim_tasks()` — bridge from Phase 13d ActionItemGoalSpec to PIM tasks
- [x] `persist()`/`restore()` via bincode + `put_meta`/`get_meta` with custom serde for DAG
- [x] `sync_to_kg()` — write PIM metadata as KG triples
- [x] `DerivationKind::PimTaskManaged` (tag 53) provenance variant
- [x] `pim_rules()` — 2 e-graph rewrite rules: `pim-unblock` (blocker done → blocked becomes next), `pim-deadline-chain` (earliest-start constraint propagation)
- [x] `Agent.pim_manager` field with init/resume/persist lifecycle, `pim_manager()` / `pim_manager_mut()` accessors
- [x] `reflect()` extended with optional `pim` and `projects` params, `gtd_review` field on `ReflectionResult`
- [x] `UserIntent::PimCommand` variant in NLP classifier, wired into TUI + headless chat
- [x] CLI: `Commands::Pim` with 9 subcommands (Inbox, Next, Review, Project, Add, Transition, Matrix, Deps, Overdue)
- [x] `AgentError::Pim` transparent variant
- [x] ~30 new unit tests

### Phase 13f — Calendar & Temporal Reasoning ✓
- [x] `CalendarError` miette diagnostic enum (5 variants: EventNotFound, Conflict, ParseError, SyncError, Engine) with `CalendarResult<T>`
- [x] `AllenRelation` enum: 13 variants (Before, After, Meets, MetBy, Overlaps, OverlappedBy, During, Contains, Starts, StartedBy, Finishes, FinishedBy, Equals) — pure `compute(s1,e1,s2,e2)`, `inverse()`, `is_overlapping()`, `as_label()`/`from_label()`, Display
- [x] `CalendarEvent` struct: symbol_id, summary, dtstart/dtend (u64), location, description, recurrence (reuses `pim::Recurrence`), ical_uid, confirmed — `duration_secs()`, `overlaps(other)`
- [x] `CalendarPredicates` — 13 Allen predicates (`time:before` .. `time:equals`) + 6 calendar metadata (`cal:dtstart`, `cal:dtend`, `cal:location`, `cal:summary`, `cal:conflicts-with`, `cal:requires-resource`) + `allen_predicate(relation)` mapper
- [x] `CalendarRoleVectors` — 4 deterministic VSA role vectors (day_of_week, time_of_day, activity_type, duration) via `encode_token(ops, "cal-role:*")`
- [x] `CalendarManager` — HashMap-based event store + optional predicates/role_vectors; add_event (create entity + KG sync + Allen relations + provenance), remove_event, get_event, events, events_in_range, today_events, week_events, detect_conflicts (sweep-line O(n log n)), compute_allen_relations, encode_temporal_pattern (VSA), sync_to_kg, persist/restore
- [x] `import_ical()` — RFC 5545 iCalendar parsing via `icalendar` crate, VEVENT extraction, chrono→u64 timestamp bridge, dedup by ical_uid (`#[cfg(feature = "calendar")]`)
- [x] `sync_caldav()` — HTTP GET via ureq with Basic auth, parse as iCal, dedup (`#[cfg(feature = "calendar")]`)
- [x] `DerivationKind::CalendarEventManaged` (tag 54) provenance variant
- [x] `calendar_rules()` — 2 e-graph rewrite rules: `before-trans` (transitivity of before), `cal-conflict` (overlapping + same resource → conflicts-with)
- [x] `Agent.calendar_manager` field with init/resume/persist lifecycle, `calendar_manager()` / `calendar_manager_mut()` accessors
- [x] `UserIntent::CalCommand` variant in NLP classifier, wired into TUI + headless chat
- [x] CLI: `Commands::Cal` with 6 subcommands (Today, Week, Conflicts, Add, Import, Sync)
- [x] `AgentError::Calendar` transparent variant
- [x] Feature: `calendar = ["icalendar", "chrono"]` — core module compiles without feature, only iCal import/CalDAV sync are feature-gated
- [x] 32 new unit tests (13 Allen relations + boundary cases, inverse symmetry, is_overlapping, label roundtrip, event duration/overlaps, manager CRUD, conflict detection, range queries, today/week, persist/restore, iCal import + dedup)

### Phase 25 — People & Contacts ✓

5 sub-phases building unified people-awareness:
- **25a — Contact Entity & Identity Resolution**: `ContactManager` with alias-based O(1) identity resolution, merge support, 13 KG predicates. `ContactError` (4 variants). Provenance: ContactResolved (72), ContactMerged (73).
- **25b — Relationship Graph**: `RelationshipGraph` with 10 `RelationshipKind` variants, strength reinforcement + exponential decay, VSA-based social circle detection. Provenance: RelationshipRecorded (74).
- **25c — Per-Person Conversation Memory**: `PersonMemoryIndex` mapping contacts to episodic memories and discussed topics. Recall by person, by person+topic, shared context between contacts.
- **25d — Communication Style Profiles**: `StyleManager` with rule-based formality/verbosity heuristics (no LLM), EMA tracking (alpha=0.3), VSA prototype similarity search. Provenance: StyleObserved (75).
- **25e — Calendar-Contact Integration**: `ContactCalendar` stateless utility — attendee management, co-attendance, meeting frequency, availability heatmap. Provenance: CalendarAttendeeLinked (76).
- 33 unit tests across all sub-phases.

Phase 14a–14i: Purpose-driven bootstrapping with identity (9 sub-phases):
- **Identity**: 14a purpose + identity parser (NL → PurposeModel + IdentityRef, character reference extraction), 14b identity resolution (Wikidata + DBpedia + Wikipedia cascade → 12 Jungian archetypes → OCEAN → Psyche construction, Ritual of Awakening: self-naming via cultural morphemes)
- **Domain**: 14c domain expansion (Wikidata + Wikipedia + ConceptNet, VSA boundary detection), 14d prerequisite discovery + ZPD classification (Vygotsky zones, curriculum generation)
- **Acquisition**: 14e resource discovery (Semantic Scholar + OpenAlex + Open Library, quality scoring), 14f iterative ingestion (curriculum-ordered, NELL-style multi-extractor cross-validation, personality-biased resource selection)
- **Assessment**: 14g competence assessment (Dreyfus model, competency questions, graph completeness, VSA structural analysis)
- **Orchestration**: 14h bootstrap orchestrator (meta-OODA loop, personality shapes exploration style, Dreyfus-adaptive exploration), 14i community recipe sharing (TOML purpose recipes with identity section, ActivityPub federation, skillpack export)
- **NLU Extension**: 14j extended rule parser (negation, quantifiers, comparatives, conditionals, temporal, modals — all 5 langs), 14k micro-ML NER (DistilBERT multilingual via ONNX, feature-gated), 14l small LLM translator (Qwen2.5-1.5B + GBNF constrained decoding, feature-gated), 14m VSA parse ranker (exemplar memory, self-improving disambiguation)

## Deployment Topology

### Binaries

| Binary | Purpose | Features |
|--------|---------|----------|
| `akh` | CLI client + TUI (ACP) | default (includes `acp`) |
| `akhomed` | ACP daemon (agent) | `server` (= `daemon` + `acp`) |
| `seshd` | Seshat's Archive MCP service | `embedding` (crate: `seshat`) |

### Daemon Mode (Primary)

`akhomed` is the primary deployment path. It serves ACP over a Unix socket (and optional TCP) with per-workspace agent daemons that execute 12 background tasks on independent tokio intervals:

- Equivalence learning, reflection, consolidation, schema discovery, rule inference, gap analysis
- Session persistence, idle OODA cycles, continuous learning (Semantic Scholar / OpenAlex / Open Library)
- Goal generation, sleep/consolidation cycles, trigger evaluation

### macOS Service Management (ADR-026)

`akh service` manages akhomed as a launchd user agent:

```
~/Library/LaunchAgents/dev.akh-medu.akhomed.plist
  ├── KeepAlive: true        (auto-restart on crash)
  ├── RunAtLoad: true        (start on login)
  ├── ExitTimeOut: 30        (graceful shutdown window)
  ├── ThrottleInterval: 10   (prevent restart loops)
  ├── Nice: 10               (low CPU priority)
  └── ProcessType: Background
```

Logs: `~/Library/Logs/akh-medu/akhomed.{stdout,stderr}.log`

### Monitoring

`DaemonStatus` (ACP `_akh/daemon/status`, workspace param) provides:
- Core: running, total_cycles, started_at, trigger_count
- Task timestamps: last_persist_at, last_learning_at, last_sleep_at, last_goal_gen_at
- KG stats: active_goals, kg_symbols, kg_triples

Scripts:
- `scripts/smoke-test.sh` — legacy REST endpoint check; **stale post-ADR-061** (akhomed has no HTTP surface — verify via an ACP client instead)
- `scripts/stability-monitor.sh` — CSV logging (RSS, KG size, cycles) for overnight monitoring

### Kubernetes Deployment (ADR-048)

Docker images (`docker/akhomed.Dockerfile`, `docker/seshd.Dockerfile`) produce multi-arch (`amd64`/`arm64`) containers with ONNX Runtime pre-installed. Helm chart at `charts/akh-medu/`:

```
charts/akh-medu/
├── Chart.yaml              # Bitnami PostgreSQL subchart dependency
├── values.yaml
└── templates/
    ├── akhomed-deployment.yaml   # Recreate strategy (single-writer KG)
    ├── akhomed-service.yaml
    ├── akhomed-pvc.yaml          # 30Gi data + 5Gi models
    ├── seshd-deployment.yaml     # With migrate init container
    ├── seshd-service.yaml
    ├── seshd-configmap.yaml      # seshat.toml (DB URL from subchart secret)
    ├── seshd-pvc.yaml            # 1Gi models
    └── seshd-ingress.yaml        # Optional (seshd HTTP only; akhomed is raw-TCP ACP, exposed via Service, no ingress)
```

PostgreSQL uses `pgvector/pgvector:pg16` image with `CREATE EXTENSION vector` in initdb.

## Planned: LLM-VSA Deep Integration (Phase 26, ADR-037)

Direction for bidirectional neural↔symbolic bridge:

```
Current:  Text ──NLU──▶ AbsTree ──▶ VSA/KG (deterministic encoding, LLM is opaque)
Target:   Text ──LLM──▶ Hidden states ──Neural encoder──▶ HyperVec ──▶ KG grounding
                              │                                          │
                              └────── Symbolic manipulation ◀────────────┘
```

Sub-phases:
- **26a**: ARM NEON SIMD kernels (2-4x speedup on M2)
- **26b**: Candle backend (pure Rust, hidden state access, Metal on M2)
- **26c**: PolarQuant KV cache compression (~5x, enables 7B models on M2)
- **26d**: Neural→VSA bridge (linear projection + binarization, self-supervised)
- **26e**: Pipeline integration (neural concept extraction as NLU enrichment)
- **26f**: Benchmarking

See [plan](plans/2026-03-26-phase26-llm-vsa-deep-integration.md) and [ADR 037](decisions/037-llm-vsa-deep-integration.md).

## Planned: Dual-Framework ML Stack (Phases 26-27)

- **Phase 26** (Candle): Inference + hidden state access + small encoder training
- **Phase 27** (Burn): LoRA fine-tuning, background training loops, hot-swap weights
- Both use safetensors format for weight exchange

## Planned: `n akh` Semantic Router (Phase 28, ADR-038)

Dynamic routing across all inference layers:

```
Query ──▶ n akh (VSA features + LinUCB bandit)
              │
              ├──▶ Rule parser      (free, <1ms)
              ├──▶ ONNX NER         (free, ~5ms)
              ├──▶ Candle LLM       (free, ~200ms)
              ├──▶ VSA direct       (free, <1ms, skip LLM)
              ├──▶ Cloud API        ($$$, ~1-3s)
              └──▶ Burn-trained     (free, ~200ms, Phase 27)
```

Self-optimizing: as Phase 27 training improves local models, bandit shifts
traffic from cloud → local automatically.

See [plan](plans/2026-03-26-phase28-semantic-router.md) and [ADR 038](decisions/038-semantic-router.md).

## Planned: Capability Exposure (Phase 29, ADR-039)

Expose hidden internal capabilities as user-facing operations (MCP + CLI + NLU + dashboard):

- **29a**: Microtheory management (create, browse, compose, query)
- **29b**: Skillpack lifecycle (activate, deactivate, share)
- **29c**: Psyche & Shadow dynamics (profile, shadow journal, archetype tuning)
- **29d**: Argumentation & debate ("argue for X", structured pro/con)
- **29e**: Transparent reasoning ("why do you think X?", derivation trees)
- **29f**: Hypothesis explorer (superposition visualization)
- **29g**: Causal explorer ("what caused X?", counterfactual timelines)
- **29h**: Evidence & belief dashboard (Dempster-Shafer intervals, conflict alerts)
- **29i**: Source trust dashboard (reliability scores, trust history)

See [plan](plans/2026-03-26-phase29-capability-exposure.md) and [ADR 039](decisions/039-capability-exposure.md).

## Planned: Procedural Skills (Phase 30, ADR-040)

Extend skills from "what is" (facts) to "how to do" (procedures):

- **30a**: Extended format — `actions.json` (causal schemas) + `plans.json` (HTN templates)
- **30b**: Activation wiring — skills register actions/plans with CausalManager + HTN + MCTS
- **30c**: LLM-assisted authoring — natural language → structured skill (3-tier user model)
- **30d**: Rule refinement loop — failure analysis → LLM suggests fixes → user approves
- **30e**: Quality metrics — success rates, failure modes, rule coverage, staleness

See [plan](plans/2026-03-26-phase30-procedural-skills.md) and [ADR 040](decisions/040-procedural-skills.md).

## Planned: Skill Intelligence (Phase 31, ADR-041)

Self-organizing skill ecosystem:

- **31a**: Dependency graph — `depends`, `conflicts`, `provides`/`requires` capability tags
- **31b**: Neural skill embeddings — per-skill prototype HyperVec for domain matching
- **31c**: Skill-aware routing — `n akh` lazy-loads skills on demand
- **31d**: Discovery from reasoning — agent detects latent skills during consolidation
- **31e**: Dynamic synthesis — agent creates skills to fill knowledge gaps
- **31f**: Sharing protocol — `.akhskill` archive export/import
- **31g**: Cross-skill inference — bridge rules for multi-domain reasoning
- **31h**: Curriculum — learning progressions for bootstrap

See [plan](plans/2026-03-26-phase31-skill-intelligence.md) and [ADR 041](decisions/041-skill-intelligence.md).
