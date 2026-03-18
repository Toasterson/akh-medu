---
name: remember
description: Store facts, decisions, or patterns into the persistent knowledge graph so they survive across sessions. Use when the user says "remember this", or when you discover something worth persisting.
version: 0.1.0
---

# Remember

Encode knowledge into akh-medu's persistent knowledge graph.

## Process

1. Break the information into atomic subject-predicate-object triples
2. For each triple, call `mcp__akh-medu__assert_triple` with:
   - `subject`: the entity (e.g. "DatabaseModule")
   - `predicate`: the relationship (e.g. "has-issue")
   - `object`: the target (e.g. "N+1QueryBug")
   - `confidence`: how certain this is (0.0-1.0, default 0.9)
3. Confirm what was stored

## What to encode

- Architectural decisions and their rationale
- Bug patterns and root causes
- Dependency relationships between components
- User preferences and constraints
- Project milestones and state changes

## What NOT to encode

- Ephemeral task details (use tasks instead)
- Exact code snippets (the code is in git)
- Anything already in CLAUDE.md or git history
