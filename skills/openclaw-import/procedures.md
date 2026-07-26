# OpenClaw Workspace Self-Import Procedure

Import your *own* OpenClaw workspace into your knowledge graph. You are the
transformation engine: read the source files yourself, reason about them, and
write **typed** structured data via the writer tools. Never look for a raw-text
"ingest" tool — those are intentionally not available in this session.

Requires the `openclaw_import` switch to be on for this akh (it injects
`openclaw_discover` and read-only access to `~/.openclaw`). If `openclaw_discover`
is unavailable, tell the operator to set `openclaw_import = true` in the akh's
workspace config and restart the daemon.

## Steps

1. **Discover.** Call `openclaw_discover`. It returns your resolved workspace
   path and an inventory: which persona files exist, how many `memory/*.md`
   journals and `skills/*/SKILL.md` skills there are, and whether the SQLite /
   Lance stores are present, with ready-to-use globs. If `found` is false, stop
   and report which agent ids exist.

2. **Identity (once).** If persona files are present and you are not already
   awakened: read `IDENTITY.md` and `SOUL.md` with `file_io`. Distill the name,
   a description, the purpose/mission, and any traits/archetypes/domains, then
   report these to the operator. The operator awakens the akh by invoking the ACP
   `identity/awaken` extension (e.g., `akh acp <workspace> _akh/identity/awaken`
   with the distilled fields). Pass `force=true` only if explicitly asked to
   re-awaken an existing identity.

3. **Memory journals.** Use the `memory/*.md` glob from step 1. For each file:
   read it with `file_io`, then call `memory_write` with:
   - `summary`: one sentence capturing the entry,
   - `facts`: the durable facts you extracted, each `{subject, predicate, object}`,
   - `tags`: a few topical tags,
   - `source`: the file path,
   - `origin`: `"openclaw"`.
   Re-running is safe — episodes are idempotent on `source`.

4. **Skills.** Use the `skills/*/SKILL.md` glob. For each: read the file, parse
   the YAML frontmatter (`name`, `description`) and the markdown body, then call
   `skill_author` with `id` (the skill dir name), `name`, `description`,
   `procedures_md` (the body), and any `domains`.

5. **Persona long-form & operational notes.** For `USER.md`, `MEMORY.md`,
   `AGENTS.md`, `TOOLS.md` and topic notes under `memory/`: read them and capture
   the durable facts with `memory_write`/`kg_mutate` (e.g. the user's name and
   standing preferences, service topology). Do not store secrets.

6. **SQLite memory (if present).** Inspect the schema, then read rows read-only
   via `shell_exec`:
   - `sqlite3 -readonly -json ~/.openclaw/memory/<your-id>.sqlite '.schema'`
   - `sqlite3 -readonly -json ~/.openclaw/memory/<your-id>.sqlite 'SELECT ... LIMIT 200 OFFSET N'`
   Page through and route each memory row into `memory_write` (origin
   `"openclaw"`). Never read credential/secret tables.

7. **Lance knowledge library (optional).** If present, export it out-of-band with
   `shell_exec` using OpenClaw's `knowledge/scripts` exporter to JSONL, read the
   JSONL with `file_io`, and capture entries via `memory_write`/`kg_mutate`.

8. **Verify & report.** Use `memory_recall` and `kg_query` to confirm episodes
   and facts landed, and `list` your skills. Report counts: episodes written,
   skills authored, whether identity was awakened, and anything skipped.

## Safety

- Read-only on the OpenClaw side; never write back into `~/.openclaw`.
- Never read or store secrets: `.ssh`, `.gnupg`, `credentials`, `secrets`,
  `*.pem`, `id_rsa`, `.gitconfig`, `service-env`. `openclaw_discover` already
  filters these from its inventory.
- Prefer small, well-summarized episodes over dumping whole files verbatim.
