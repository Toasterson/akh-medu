# ADR 123 — Per-model budgets, timeouts and relevance-scoped wakes

> Date: 2026-09-30
> Status: Proposed (branch `feat/per-model-budgets-timeouts`; takes effect when the Roman core is
> redeployed from `origin/main` and re-rendered)
> Relates: [ADR 113](113-full-akh-always.md) (full akh, per-akh LLM budgets, decision 6),
> [ADR 053](053-first-class-akh-config.md) (`akh.toml`), [ADR 065](065-context-assembly-runtime.md)
> (context runtime), [ADR 089](089-organ-kg-zeroshot-path.md) (retrieval-scoped tool catalog),
> [ADR 105](105-autonomous-run-governance.md) (reactor runs), [ADR 116](116-lilith-chief-of-staff-guided-rl.md)
> / [ADR 121](121-lilith-serving-on-hephaistos-llama-cpp.md) (Lilith), [ADR 122](122-model-family-prompt-guidance.md)
> Evidence: [2026-09-30 GLM / Kimi run evaluation](../research/2026-09-30-glm-kimi-run-evaluation.md), R-A and A-F2

## Context

The Romans (janus, minerva, vesta; GLM-5.3 on Ollama Cloud) wake about once a day, late. A wake
offers the whole tool catalog: 94 MCP tools on janus and minerva (61 of them Anima), about 150 on
vesta, plus about 47 native tools. The first request of a wake is 56,610 tokens, mostly catalog,
and a wake costs 280k–690k prompt tokens over 6–9 requests. The daily budget was 250,000 tokens per
akh, so "reactor model turn deferred: autonomy budget exhausted" fired 10 times between 09-25 and
09-30, and vesta saw the critical `node03-zpool-degraded` about 11 h after onset.

Every limit on the turn path was either a compiled constant, a process-wide env var (which moves
every akh on a host together), or a per-endpoint key with a model-blind default. GLM-5.3 at `max`
effort thinks for about 32 s before its first answer token; Kimi K3 for about 60 s; a local Gemma 4
12B has a 49k working window. One set of numbers cannot fit them. Timeouts and round caps have
bitten before (the 8-round cap that strangled recall; the 1800 s runtime deadline on Kimi K3
reviews, A-F2).

The operator's decision (2026-09-30): larger budgets **and** relevance-scoped tools per wake, with
budgets and timeouts configurable per model, with researched defaults per model and reasoning
effort. The accounts are subscriptions with usage limits, not pay-per-token.

## Inventory (before this ADR)

Line numbers are on this branch. "Env" = process-wide env var, "akh" = per-akh `akh.toml` key,
"const" = compiled in.

| Limit | Where | Default | Scope | Caps |
|---|---|---|---|---|
| Daily tokens / turns | `src/agent/autonomy.rs:1067`, `:1069`; enforced by `AutonomyGovernor::admit`, reactor `admit_model_turn` `src/agent/daemon.rs:2934`, deferral log `:2909` | 250,000 / 40 | akh `[autonomy.budget]`; Roman renderer hardcoded `BUDGET` (removed) | LLM-cost background work per UTC day |
| Agentic round cap | `src/agent/agentic_chat.rs:44`, window scaling `:450` | 12; 20 at ≥32k window; 30 at ≥200k | const + env `AKH_AGENT_MAX_ROUNDS` | model rounds per turn |
| Turn deadline | `src/agent/agentic_chat.rs:49` | 300 s | const + env `AKH_AGENT_TURN_TIMEOUT_SECS` | wall clock, checked between rounds |
| Dispatch lift | `src/agent/agentic_chat.rs:509` (`for_dispatch`) | no round/time cap | `is_autonomous_dispatch` (reactor `require_mutation`) | only explicit limits apply |
| Repeat / wedge | `src/agent/agentic_chat.rs:54`, `:60` | 3 identical calls / 60 s without a request | const + env | spin / wedge |
| Ollama HTTP | `src/agent/ollama.rs:99`, connect/read `:433` | 120 s read-idle, 20 s connect | env `AKH_OLLAMA_TIMEOUT_SECS` only | one request |
| Ollama output / window | not sent on `/api/chat` | server default | none | — |
| OpenAI HTTP / output / window | `src/akh_config.rs:1161`, `:1165`, `:1155`; total timeout `src/agent/openai_backend.rs:506` | 600 s total / server default / none (8,192 fallback) | akh `[openai_endpoint]` | one request |
| `reasoning_effort` | `src/akh_config.rs` (`reasoning_effort`), Ollama `think` | omitted | akh | effort |
| Context budget | `src/agent/context/types.rs:19`, `:22` | 8,192 window when unknown, 1,024 reserved output | const | assembly + in-turn tool results |
| Tool count | `AKH_AGENT_TOOL_RETRIEVAL` `src/agent/agentic_chat.rs:900`, `AKH_AGENT_MAX_TOOLS` `:912`, `ChatBackend::preferred_tool_count` `src/agent/chat_backend.rs:208` | whole catalog (Grug: 4) | env / backend | tools offered per request |
| MCP call | `src/agent/tools/mcp_config.rs:56`, `:59`; Roman `SERVER_TIMEOUTS` `deploy/roman-mini/full_akh.py:61` | 30 s (max 600); vesta-ops 300 | akh `[[mcp.servers]] timeout_secs` | one MCP request |
| ACP idle watchdog | `src/acp/mod.rs:599` | 300 s | env `AKH_TURN_IDLE_TIMEOUT_SECS` | ACP turn without activity |
| Reactor lease / retries / trip-wire | `src/agent/reactor.rs:295`, `:27` | 900 s lease, 3 retries, 10 no-progress runs | per schedule / const | one occurrence |
| Codex/Claude runtime | `src/agent/runtime.rs:73` | 600 s | akh `[agent_runtime]` | runtime turn |
| akh-platform runtimes | `crates/akh-session-types/src/lib.rs:1144`, `:1169`; `crates/akh-session-services/src/internal.rs:37` | 600 s; `max_iterations` 1..=256; output ≤ 65,536 | runtime profile (Talos) | internal loop (not this ADR) |
| Service delivery | `crates/akh-session-services/src/config.rs:26` | 30 s total, 5 s connect | endpoint config | akheventd → akhomed/akhsessd |

Nothing in akheventd or akhsessd caps a Roman wake's duration; akheventd only delivers the event
(30 s request timeout). The wake is bounded by the akhomed loop above and by the reactor lease.

## Research (per-model defaults)

Throughput on Ollama Cloud is not published or benchmarked; the tok/s and time-to-first-answer-token
(TTFAT) numbers below are Artificial Analysis figures for the vendor or median provider, used as a
proxy. Rows marked **E** are estimates; the derivation is in the note.

| Model | Context / max output | Speed (AA, max effort) | Effort | Price $/M in / cached / out (Ollama) | Sources |
|---|---|---|---|---|---|
| GLM-5.3 | 1M / 128K | 71 tok/s, TTFT 3.2 s, TTFAT 31.6 s | always reasons; `low`, `high`, `max` (default `max`) | 1.40 / 0.26 / 4.40 | [Z.ai GLM-5.3](https://docs.z.ai/guides/llm/glm-5.3), [AA](https://artificialanalysis.ai/models/comparisons/gemini-3-8-flash-vs-glm-5-3), [ollama.com/pricing](https://ollama.com/pricing) |
| GLM-5.3-flash | 1M / 128K **E** | 44 tok/s, TTFAT 48.5 s | as 5.3 | 0.15 / 0.03 / 0.50 | [AA](https://artificialanalysis.ai/models/comparisons/glm-5-3-flash-vs-glm-5-3), pricing |
| Kimi K3 | 1M (Ollama `/api/show` 1,048,576) / 131,072 default completion | 36 tok/s, TTFT 4.5 s, TTFAT 59.6 s | always reasons; `low`, `high`, `max` (default `max`); fixed sampling | 3.00 / 0.30 / 15.00 | [Kimi reasoning effort](https://platform.kimi.ai/docs/guide/use-reasoning-effort), [Kimi chat API](https://platform.kimi.ai/docs/api/chat), [AA](https://artificialanalysis.ai/models/comparisons/kimi-k3-vs-glm-5-3), pricing |
| Kimi K2.6 / K2.7-code | 256K / not stated | K2.7-code 75 tok/s | always reasons, no effort control | 0.95 / 0.16–0.19 / 4.00 | [Kimi models](https://platform.kimi.ai/docs/api/models-overview), [AA](https://artificialanalysis.ai/models/kimi-k2-7-code), pricing |
| DeepSeek V4-Pro | 1M / 384K | 80 tok/s, TTFAT 26.5 s | thinking on, default `high`; `low`/`high`/`max` | 1.32 / 0.044 / 3.96 (half off-peak) | [DeepSeek pricing](https://api-docs.deepseek.com/quick_start/pricing), [thinking mode](https://api-docs.deepseek.com/guides/thinking_mode), [AA](https://artificialanalysis.ai/models/comparisons/deepseek-v4-pro-vs-glm-5-3) |
| DeepSeek V4.1-Flash | 1M / 384K | 209 tok/s, TTFAT 10.5 s | as V4-Pro | 0.30 / 0.006 / 1.20 (half off-peak) | same, [AA](https://artificialanalysis.ai/models/comparisons/deepseek-v4-1-flash-vs-glm-5-3) |
| Gemma 4 12B (Lilith, llama.cpp Q8_0 on a 24 GB RX 7900 XTX) | 256K native; served 2 × 65,536 slots; akh window 49,152 | ≤ 75 tok/s (960 GB/s over 12.7 GB); about 50 tok/s measured for Q8_0 on a 24 GB RTX 3090 **E** | thinking on/off by template token; no effort levels | local | [HF model card](https://huggingface.co/google/gemma-4-12B), [club-3090 #412](https://github.com/noonghunna/club-3090/discussions/412), [ADR 121](121-lilith-serving-on-hephaistos-llama-cpp.md) |
| Qwen 3.8 | not on Ollama Cloud (2026-09-30); `qwen3.8:27b` local: 256K, 43 tok/s | | thinking on by default | local | [HF Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), [AA](https://artificialanalysis.ai/models/qwen3-8-27b), [Ollama cloud search](https://ollama.com/search?c=cloud) |

**Ollama Cloud metering** ([ollama.com/pricing](https://ollama.com/pricing), read 2026-09-30):
Free $0 (starter credit, 1 concurrent request), Pro $20/month with $60 of usage credit (3
concurrent), Max $100/month with $300 (10 concurrent), Team $500/month with $1,000 shared (10).
Usage is metered **in tokens at each model's price** (input, cached input, output); there are no
request counts and no separate premium multipliers — the effective multiplier is the price ratio
(Kimi K3 output costs 3.4× GLM-5.3's). Included credit resets monthly on the subscription day and
does not roll over; purchased credit is drawn after it; a 90 % email warning goes out. The old
session and weekly limits no longer apply on the new plans. Requests over the concurrency limit are
queued up to an undocumented bound, then rejected. No maximum request time is documented, so the
client bounds it. Errors: 429 rate limit, 502 model unreachable, mid-stream errors as an NDJSON
`error` line ([docs.ollama.com/api/errors](https://docs.ollama.com/api/errors)).

**Derivations (all E).** HTTP timeout: Ollama's is a read-idle gap, and a reasoning model streams
thinking tokens, so it only has to cover time to first token on a large prompt; 300 s for GLM-5.3
(about 10× a 32 s TTFAT), 600 s for Kimi K3. The OpenAI-compatible timeout is total, so Lilith's is
prefill of 49k tokens plus 8,192 output at about 40 tok/s ≈ 240 s, with 2.5× headroom = 600 s. Round
caps: Z.ai reports about 75K output tokens per coding task at `max` and 50K at `high`; at about 2.5K
per round that is 30 and 20 rounds; the Romans never hit 30 (evaluation, "no round-cap hits"), so 30
stays. Turn deadline: rounds × (TTFAT + output ÷ tok/s + tool time) with headroom, 1,800 s for
GLM-5.3 `max`, 3,000 s for Kimi K3 `max`. Output cap: high enough never to truncate a reasoning
answer in practice (GLM `max` 32,768 against a 128K ceiling). Working window: GLM and Kimi report 1M,
but a subscription pays for every prompt token, so the working window is capped at 262,144.

## Decision

1. **A typed per-model profile** (`src/agent/model_profile.rs`). Nine fields:
   `max_tokens_per_day`, `max_turns_per_day`, `max_tokens_per_wake`, `turn_deadline_secs`,
   `max_rounds`, `http_timeout_secs`, `max_output_tokens`, `context_window`,
   `max_tools_per_wake`. `ModelProfile` is the partial form (`deny_unknown_fields`); a
   `ResolvedModelProfile` holds each value as a `Setting { value, source }`, where
   `ProfileSource` is `AkhOverride(location) | Catalog { entry, effort } | Environment(var) |
   Builtin`. `ReasoningEffort` is an enum (`low | medium | high | max`).
2. **One catalog, two readers.** `data/model-profiles/catalog.toml` is compiled into akhomed
   (`ModelCatalog::builtin`) and read by the Roman renderer. Rows match on exact ids, else the
   longest prefix (case-insensitive, `:cloud` ignored); `[models.effort.<level>]` overrides the
   row for that effort. Validation refuses unknown keys, a wrong `schema_version`, an id or prefix
   in two rows, and zero limits.
3. **Resolution order, per field:** the akh's override > the catalog row for (model, effort) > the
   global default. The override is `[model_profile]` or, for back-compat, `[autonomy.budget]`
   (`max_tokens_per_day`, `max_turns_per_day`) and `[openai_endpoint]` (`timeout_secs`,
   `max_tokens`, `context_window`); setting the same limit in both places is refused
   (`akh::model_profile::duplicate_setting`). The global default is the process env var where one
   exists (`AKH_AGENT_MAX_ROUNDS`, `AKH_AGENT_TURN_TIMEOUT_SECS`, `AKH_OLLAMA_TIMEOUT_SECS`), else
   the compiled constant — so a model the catalog does not know behaves exactly as before, and an
   env var no longer beats a catalog row (none is set on the Roman core). `load_checked` resolves
   the profile, so a bad one refuses the akh at load with a miette diagnostic naming the options.
   An unknown `reasoning_effort` is refused for native backings and passed through for Codex/Claude
   runtimes. akhomed logs one `model profile resolved (ADR 123)` info line per akh with every value
   and its source, and re-resolves on an ACP `model/set` (turn limits apply from the next turn;
   the daily budget from the next daemon start).
4. **Where each value lands.** The daemon's governor gets the resolved daily budget
   (`AkhConfig::effective_autonomy`). `to_model_manager_config` sets the Ollama timeout and the new
   `num_predict` (`OllamaConfig::num_predict`, sent as `options.num_predict`); `to_openai_config`
   sets timeout, `max_tokens` and window. The agentic loop takes its round cap and deadline from
   the profile (`RoundBudget::from_profile`) and caps the model's reported window with the
   profile's (`working_context_window`: never above what the model reports). Catalog and built-in
   limits still do not cap autonomous dispatch; explicit ones (akh override, env var) do.
5. **Relevance-scoped wakes.** A reactor `llm_turn` with tools runs through
   `Agent::run_wake_turn`. `scope_wake_catalog` offers, first, the tools the wake pins — the new
   schedule field `required_tools`, then every registered tool the schedule's own prompt or exit
   condition names by its exact name (identifier-shaped tokens only; the untrusted event payload
   never pins) — then the top `max_tools_per_wake` of the rest by `organ::retrieve_scoped` (VSA,
   lexical fallback) against the full wake prompt. Pins come first because the compacting context
   manager trims tools from the end, and they never count against K. Operator env knobs
   (`AKH_AGENT_TOOL_RETRIEVAL`, `AKH_AGENT_MAX_TOOLS`) keep precedence for A/B runs. Interactive
   ACP turns keep the whole catalog.
6. **Per-wake token budget and lease.** A wake meters its own usage (`UsageScope`) and stops at
   `max_tokens_per_wake` with `TurnStop::TokenBudget` (code `token_budget`), then answers with a
   forced synthesis. The reactor settles that as done, not failed, since a retry would hit the same
   budget. A wake never runs past its schedule's `lease_seconds` (default 900 s), whatever the
   profile's deadline.
7. **Renderer.** `deploy/roman-mini/full_akh.py` drops the hardcoded `BUDGET` and writes no
   `[autonomy.budget]` (it would pin an override; a stale one is removed on re-render). It reads
   the catalog to comment each workspace's `[autonomy]` table with the resolved profile, fails the
   render closed if a Roman's or Lilith's model has no row, and renders `required_tools` per route
   kind (`ROUTE_REQUIRED_TOOLS`, filtered to the tools the role holds). The dojo teacher (sealed,
   no autonomy) gets the turn-limit comment only and may use a model without a row.

### Catalog (defaults at the model's default effort; all E)

| Row (matches) | tokens/day | turns/day | tokens/wake | deadline | rounds | HTTP | output | window | tools/wake |
|---|---|---|---|---|---|---|---|---|---|
| `glm-5.3` (`glm-5.3`, prefix `glm-5`) | 1,500,000 | 96 | 250,000 | 1,800 s | 30 | 300 s | 32,768 | 262,144 | 24 |
| ↳ effort `high` / `medium` | | | | 1,500 s | 28 | 240 s | 24,576 | | |
| ↳ effort `low` | | | 150,000 | 600 s | 16 | 180 s | 8,192 | | |
| `glm-5.2` | 1,500,000 | 96 | 250,000 | 1,800 s | 30 | 300 s | 32,768 | 262,144 | 24 |
| `glm-5.3-flash` | 6,000,000 | 192 | 400,000 | 2,400 s | 32 | 300 s | 24,576 | 262,144 | 24 |
| `kimi-k3` | 600,000 | 48 | 250,000 | 3,000 s | 40 | 600 s | 32,768 | 262,144 | 32 |
| ↳ effort `high` / `medium` | | | | 2,000 s | 30 | 420 s | 16,384 | | |
| ↳ effort `low` | | | 150,000 | 900 s | 16 | 240 s | 8,192 | | |
| `kimi-k2` (`kimi-k2.7-code`, `kimi-k2.6`, prefix `kimi-k2`) | 1,500,000 | 96 | 250,000 | 1,500 s | 30 | 300 s | 16,384 | 262,144 | 24 |
| `deepseek-v4-pro` (prefix `deepseek-v4`) | 1,500,000 | 96 | 250,000 | 1,500 s | 32 | 300 s | 32,768 | 262,144 | 24 |
| ↳ effort `max` | | | | 2,400 s | 40 | 420 s | 65,536 | | |
| `deepseek-v4.1-flash` (`deepseek-v4-flash`) | 5,000,000 | 192 | 400,000 | 900 s | 40 | 240 s | 32,768 | 262,144 | 24 |
| `gemma-4-12b` (`lilith`, `gemma4:12b`, prefix `gemma4`) | 3,000,000 | 96 | 150,000 | 1,200 s | 20 | 600 s | 8,192 | 49,152 | 16 |
| `qwen3.8` (prefix `qwen3`) | 3,000,000 | 96 | 250,000 | 1,800 s | 30 | 600 s | 16,384 | 131,072 | 20 |

Daily budgets follow price: at about the same price as GLM-5.3 a row gets 1.5M tokens; the cheap
flash rows get 3–4× more; Kimi K3 (2.1–3.4× the price) gets 600k. Local rows are bounded by GPU
time, not credit. Tool caps: 24 for frontier cloud models, 32 for Kimi K3 (strongest tool use,
largest context), 16–20 for small local models (ADR 089: small models degrade past about 14 tools).

## Roman before / after

Measured with this branch's harness (`dump_native_tool_catalog_for_adr_123`, run with `--ignored`)
for the native catalog (42 tools, 23,363 schema characters), the Anima MCP `tools/list` for the 61
granted Anima tools (48,179 characters, 790 per tool), and the Anima average for the other MCP
tools whose schemas were not reachable from this Mac (board, sokar, seshat, laya, vesta-ops,
grafana). Tokens are shown two ways: the repo's `len/4` heuristic, and calibrated to the observed
56,610-token first request, which works out to about 2.1 characters per token for this
schema-heavy prompt through GLM's template. Everything past the tool counts is an estimate.

| | janus / minerva | vesta |
|---|---|---|
| Daily budget | 250,000 → **1,500,000** tokens; 40 → **96** turns | same |
| Per-wake budget | none → **250,000** tokens, and at most the 900 s lease | same |
| Tools offered per wake | 141 / 138 (94 MCP + ~47 native) → **24 + 2–6 pinned** | 196 (151 MCP + ~45 native) → **24 + 2–6 pinned** |
| Tool schema per request | ≈100k chars ≈ 25k (len/4) / **≈48k (calibrated)** tokens → ≈19–22k chars ≈ **9–11k** tokens | ≈144k chars ≈ 36k / **≈69k** tokens → ≈ **9–11k** tokens |
| First request of a wake | 56.6k → ≈ **18–20k** tokens | ≈ 80k → ≈ **20k** tokens |
| Saving per wake (6–9 requests) | ≈ **220k–350k** tokens | ≈ **350k–530k** tokens |
| Tokens per wake | 280k–690k → ≈ **100k–350k** (conversation growth unchanged) | same range, from the high end |
| Wakes the daily budget allows | ≈ 0.4–0.9 → ≈ **4–15** (at least 6 even at the per-wake cap) | same |

Cost at the cap (E): 1.5M GLM-5.3 tokens a day, nearly all input, is at most $2.10 per akh per day
uncached ($1.40/M) and about $0.40 fully cached ($0.26/M). Three Romans at the cap cost at most
about $190 a month uncached. That fits the Max plan's $300 of credit with room for the Kimi K3
reviews on the same key; it does **not** fit Pro's $60 (see the operator decision below).

## Consequences

- Budgets and limits are data. Changing one is a catalog edit with its reason in this ADR, a main
  merge and a redeploy; an urgent per-akh change is a `[model_profile]` table, which the renderer
  preserves.
- Every number on the startup log names its source, so "where did 1800 s come from" has an answer.
- Wakes no longer see tools their route did not ask for and the prompt did not rank. A wake that
  needs a tool the scorer misses has to name it (route `required_tools` or the prompt); the
  `wake tool catalog relevance-scoped` info line lists pinned and unavailable tools per wake.
- A wake can now end at `token_budget`; the run report says so, and the next event wakes it again.
- The akh-level `[openai_endpoint] timeout_secs` default moved from a serde default (600) to the
  catalog; a TOML that set it keeps it as an override.

## Follow-ups (not in this change)

- **Operator: confirm the Ollama plan behind `ROMAN_OLLAMA_KEY`.** On Pro ($60/month), add
  `[model_profile] max_tokens_per_day = 400000` to each Roman (about $50/month for three at the
  uncached worst case) or move them to `glm-5.3-flash`. Skipping it on Pro risks the credit running
  out mid-month, after which wakes fail on the provider gate.
- akh-platform runtime profiles (Talos `resources.json`) are outside this catalog: Kimi K3 review
  runtimes still carry `timeout_secs: 1800` and no `reasoning_effort` (A-F2). The catalog's
  `kimi-k3` row (3,000 s at `max`, 2,000 s at `high`) is the recommendation for them.
- The context budget still reserves 1,024 output tokens (`DEFAULT_RESERVED_OUTPUT`), not the
  profile's `max_output_tokens`; changing it moves Lilith's prompt room and needs its own measure.
- Completion-contract pins (`pin_required_tools`) are still appended at the end of the catalog,
  where the compacting manager trims first; wake pins are not.
- Measure the scoped wakes on the live Romans (first-request tokens, wakes per day, missed-tool
  rate) after the redeploy, and tune `max_tools_per_wake`.
