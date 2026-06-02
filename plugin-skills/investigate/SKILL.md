---
name: investigate
description: Deep-dive investigation using the knowledge graph's autonomous OODA reasoning agent. Use when you need to trace causality, discover hidden patterns, or answer complex questions about what the system knows.
version: 0.1.0
---

# Investigate

Run an autonomous investigation through akh-medu's OODA reasoning loop.

## Process

1. Formulate the question clearly
2. Call `mcp__akh-medu__ask` with the question and appropriate `max_cycles` (default 10, use more for complex queries)
3. Review the narrative findings
4. If the answer is incomplete, follow up with targeted `mcp__akh-medu__sparql_query` or `mcp__akh-medu__search` calls
5. Synthesize findings for the user

## When to use

- Tracing why a decision was made
- Finding connections between concepts
- Answering "what does the system know about X?"
- Discovering gaps in the knowledge graph

## Arguments

Pass the investigation question as the argument: `/investigate why does the auth module depend on the cache layer?`
