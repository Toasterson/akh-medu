---
type: grilling
status: closed
resolved: 2026-09-19
---

## Question

Transcripts are full of credentials and private content, and this map moves them
from local files into a central store that machines search. What is the posture?

## Resolution

**Verbatim in, redact outward.**

- The private TimescaleDB store holds source records unmodified — the same trust
  level as the files already sitting on disk. Redacting at capture would defeat
  [the harvest unit decision](harvest-unit.md) outright.
- Credential-shaped content is **detected and flagged per line**, not mutated.
- Everything *leaving* the store passes through redaction, and flagged lines can
  be withheld entirely. **Amended 2026-09-19:** with the collapse to one store,
  that outward boundary is solely [the read MCP surface](read-mcp-surface.md) —
  there is no second service to leak into, and because we own delete and upsert a
  redaction miss is now *retractable* rather than permanent. That was a decisive
  input to [the revised read surface](read-surface.md).
- OpenCode's `account` and `credential` tables are **never read**. They hold live
  OAuth access and refresh tokens; there is no version of this feature that needs
  them.

Rejected: redact-at-capture (the store never holds a secret and could be freely
backed up, but fidelity is gone, false positives silently destroy content, and a
missed pattern is unrecoverable once the source file rotates); no-redaction with
a locked-down store (fully faithful and simplest, but drops the "machines can look
through it" half of the ask).

The detector's pattern set and its measured error rates are a separate open
ticket ([redaction detector](redaction-detector.md)). This ticket fixes the
posture, not the implementation.
