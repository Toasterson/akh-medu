---
name: kg-query
description: Query the knowledge graph directly with SPARQL or keyword search. Use for precise, structured queries when you know what you're looking for.
version: 0.1.0
---

# Knowledge Graph Query

Direct access to akh-medu's knowledge graph via search and SPARQL.

## Usage

### Keyword search
If the argument is plain text, use `mcp__akh-medu__search` with it as the `term`.

### SPARQL query
If the argument starts with `SELECT`, `ASK`, `CONSTRUCT`, or `DESCRIBE`, pass it directly to `mcp__akh-medu__sparql_query`.

### No argument
If invoked without arguments, call `mcp__akh-medu__status` to show the current graph state (symbol count, triple count, loaded compartments).

## SPARQL patterns

Symbol IRIs follow the pattern `https://akh-medu.dev/sym/{id}`. Common queries:

```sparql
# All facts about an entity
SELECT ?p ?o WHERE { <https://akh-medu.dev/sym/ENTITY> ?p ?o }

# Find entities of a kind
SELECT ?s WHERE { ?s <https://akh-medu.dev/sym/is-a> <https://akh-medu.dev/sym/TYPE> }

# Transitive relationships
SELECT ?ancestor WHERE { <https://akh-medu.dev/sym/X> <https://akh-medu.dev/sym/is-a>+ ?ancestor }
```
