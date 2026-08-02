# Example operating-guidance documents

Two sample `GuidanceDoc`s in the format ADR 094 defines — YAML frontmatter
(`id`, `version`, `applies_to`) + a markdown doctrine body. They are a **concrete
starter corpus** for the design: something real to render into
`ContextSection::Guidance` and to A/B against a model in the 44j harness
(`src/bench_phase44/`).

Both reference **real akh-medu tools** (verified against `src/agent/tools/`), so
they double as a smoke test for `applies_to` tool-matching.

| File | `id` | Activates when the akh has… |
| --- | --- | --- |
| `tools-filesystem.md` | `tools/filesystem` | `file_io`, `edit`, `glob`, `grep`, `shell_exec` |
| `tools-kg-memory.md` | `tools/kg-memory` | `kg_query`, `kg_mutate`, `memory_recall`, `memory_write`, `similarity_search` |

Each teaches the three things a schema alone doesn't convey — **when** to reach
for a tool, **preferences/rules**, and **terminology** — which is exactly the gap
this effort exists to close.

## How they'd be used (per ADR 094)

- Drop under the registry dir (`doctrine/` v1) or reference by `id` from an akh's
  `AkhConfig.guidance.docs`.
- The resolver filters by `applies_to` against the live tool catalog + pinned
  model, composes global → per-akh → per-tool, and hands the text to
  `EpisodeState::guidance` → rendered as the `Guidance` section (core tier, sibling
  to `System` — never overriding psyche).
- Bump `version` on every edit so an A/B run records which revision produced which
  behavior change.

These are **drafts for iteration**, not final doctrine — tune the wording against
real model behavior once the harness leg lands.
