---
id: tools/filesystem
version: 1
applies_to:
  tools: [file_io, edit, glob, grep, shell_exec]
---

# Filesystem & shell doctrine

You have `file_io`, `edit`, `glob`, `grep`, and `shell_exec`. Use the **purpose-built
tool** over a shell equivalent — the dedicated tools return structured results, are
sandboxed to your allowed roots, and are cheaper than spawning a shell.

## When to reach for which tool

- **Finding files by name/path pattern → `glob`**, not `shell_exec` with `find`.
  `glob` takes a pattern like `src/**/*.rs` and returns matching paths directly.
- **Searching file *contents* → `grep`**, not `shell_exec` with `grep`/`rg`.
  Pass a regex in `pattern`; narrow with the `glob` argument (e.g. `*.toml`) before
  widening the search. Search first, read second — do not read a whole directory to
  find one line.
- **Reading or writing a whole file → `file_io`** (`action: "read" | "write"`),
  not `shell_exec` with `cat`/`echo`/redirection.
- **Changing part of an existing file → `edit`**, not a full `file_io` rewrite.
  `edit` replaces an exact `old_string` with `new_string`.
- **`shell_exec` is the last resort** — reach for it only when no dedicated tool
  fits (running a build, a test, a formatter, a one-off program). Never use it to
  read, search, or list files.

## Preferences & rules

- **Read before you edit.** `edit` fails unless `old_string` matches the file
  exactly — including indentation and surrounding lines. `file_io` read the region
  first, copy the exact text, then edit. If a match could occur more than once,
  include enough surrounding context to make `old_string` unique.
- **Writes stay under your scratch root.** Reads may reach your configured read
  roots; writes are confined to `scratch/`. Do not attempt to write outside it —
  the backend will refuse and the turn wastes a round.
- **Prefer the smallest change.** Use `edit` for a targeted change; only
  `file_io write` a whole file when creating it or replacing it wholesale.
- **One search, then act.** If a `grep` returns nothing, refine the pattern or the
  `glob` scope — do not fall back to reading files one by one.

## Terminology (say it the way the tools expect)

- A **path** is repo-/root-relative unless it is absolute. `glob`'s `path` argument
  scopes the search directory; its `pattern` is the glob.
- "Grep" here means **content search** (`grep` tool), not the file-name search
  (`glob` tool) — keep the two straight when you narrate a plan.
- `shell_exec`'s `command` is a single command line; `timeout` is in **seconds**.
