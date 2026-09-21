---
type: grilling
status: closed
resolved: 2026-09-19
---

## Question

The original ask floated "maybe a custom encoding on that?" — what is the
encoding actually for?

## Resolution

**A compact canonical render: one line per normalized event, stored as a derived
column.** Structure carried by cheap delimiters rather than JSON's braces, quotes
and per-field keys — the High-Grug shape of
[ADR 088](../../../ai/decisions/088-high-grug-boundary-format.md), which measured
encoded sequence length **halved** against JSON on an otherwise identical recipe.

The motivating cost is real: 68 MB of raw JSONL across ~77 sessions today, and an
agent reading it through an MCP tool pays that in context. Cross-tool events also
render identically regardless of origin, which is what makes one query surface
over three incompatible sources coherent.

It is **derived**, so it is cheap to get wrong and re-render as the format
evolves. Designing the render itself is a separate open ticket
([render format](render-format.md)).

Rejected: compression/dedup only (solves size, not the token cost of *reading*);
VSA hypervector per line (deepest engine integration, but presumes the akh is the
main consumer and inherits the known binary-VSA degeneracy); no encoding at all
(defensible — JSONB plus tsvector plus pgvector answers the literal ask — but
leaves the reading cost unaddressed).
