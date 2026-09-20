---
type: grilling
status: closed
closed: 2026-09-19
resolution: out of scope
---

## Question

What exactly is pushed to Seshat, and how do the two stores stay coherent?

## Closed — out of scope

[The read surface decision](read-surface.md) was revised on 2026-09-19 to collapse
to a single store. Nothing is pushed to Seshat, so this ticket has no subject.

Closed rather than resolved: ruling work beyond the destination is a scoping act,
not a step on the route, so it is recorded in the map's **Out of scope** section
and not in Decisions so far.

Its substance did not evaporate — it moved:

- Chunking, embedding and the model asset are now ours:
  [embedding pipeline](embedding-pipeline.md).
- The pointer-back problem is gone. A semantic hit and an exact line now live in
  the same database, so no `(source, session_id, seq)` handoff across services is
  needed, and no "which surface answers this" rule has to be taught to a model.
- Redaction at the outward boundary is now solely
  [the read MCP surface](read-mcp-surface.md)'s concern.

The research that closed it remains useful:
[the ingest contract](seshat-ingest-contract.md) is where the chunker's shape,
the embedder's model and dimensionality, and the `agent_session_transcript`
extraction format are documented — all of which are now being reused as a library
rather than called over the wire.
