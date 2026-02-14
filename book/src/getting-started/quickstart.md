# Quick Start Tutorial

This tutorial walks through a complete session: creating an engine, adding
knowledge, querying it, running inference, and using the agent.

## 1. Initialize a Workspace

Start by creating a persistent workspace:

```bash
akh-medu init
```

All subsequent commands use the default workspace automatically. To use a
named workspace, add `-w my-project` to any command.

## 2. Bootstrap with Seed Packs

Seed packs load foundational knowledge. Three packs are bundled:

```bash
# See what's available
akh-medu seed list

# Apply the ontology (fundamental relations like is-a, has-part, causes)
akh-medu seed apply ontology

# Apply common-sense knowledge (animals, materials, spatial concepts)
akh-medu seed apply common-sense

# Check what's been applied
akh-medu seed status
```

Seeds are idempotent -- applying the same seed twice has no effect.

## 3. Ingest Your Own Knowledge

### From the grammar parser

The fastest way to add knowledge is natural language:

```bash
# Parse a statement and ingest it into the KG
akh-medu grammar parse "Dogs are mammals" --ingest
akh-medu grammar parse "Cats are mammals" --ingest
akh-medu grammar parse "Mammals have warm blood" --ingest
akh-medu grammar parse "The heart is part of the circulatory system" --ingest
```

### From a JSON file

For bulk ingestion, prepare a JSON file:

```json
[
  {"subject": "Earth", "predicate": "is-a", "object": "planet", "confidence": 1.0},
  {"subject": "Earth", "predicate": "orbits", "object": "Sun", "confidence": 1.0},
  {"subject": "Mars", "predicate": "is-a", "object": "planet", "confidence": 1.0},
  {"subject": "Mars", "predicate": "orbits", "object": "Sun", "confidence": 1.0}
]
```

```bash
akh-medu ingest --file planets.json
```

### From CSV

```bash
# Subject-Predicate-Object CSV
akh-medu ingest --file data.csv --format csv --csv-format spo

# Entity CSV (column headers become predicates)
akh-medu ingest --file entities.csv --format csv --csv-format entity
```

### From plain text

```bash
akh-medu ingest --file article.txt --format text --max-sentences 100
```

## 4. Query the Knowledge Graph

### List symbols

```bash
# List all known symbols
akh-medu symbols list

# Show details for a specific symbol
akh-medu symbols show Dog
```

### SPARQL queries

```bash
# Find all mammals
akh-medu sparql "SELECT ?s WHERE { ?s <https://akh-medu.dev/sym/is-a> <https://akh-medu.dev/sym/mammal> }"

# Or from a file
akh-medu sparql --file query.sparql
```

### Graph traversal

```bash
# BFS from Dog, 2 hops deep
akh-medu traverse --seeds Dog --max-depth 2

# Only follow is-a edges
akh-medu traverse --seeds Dog --predicates is-a --max-depth 3

# Output as JSON
akh-medu traverse --seeds Dog --max-depth 2 --format json
```

### Similarity search

```bash
# Find symbols similar to Dog via VSA
akh-medu search --symbol Dog --top-k 5
```

## 5. Run Inference

### Spreading activation

Discover related knowledge by spreading activation from seed symbols:

```bash
# What's related to Dog and Cat?
akh-medu query --seeds "Dog,Cat" --depth 2 --top-k 10
```

### Analogy

Compute "A is to B as C is to ?":

```bash
akh-medu analogy --a King --b Man --c Queen --top-k 5
```

### Role-filler recovery

Find the object of a (subject, predicate) pair via VSA:

```bash
akh-medu filler --subject Dog --predicate is-a --top-k 5
```

### Forward-chaining rules

Run e-graph rewrite rules to derive new facts:

```bash
akh-medu agent infer --max-iterations 10
```

## 6. Use the Agent

The autonomous agent uses an OODA loop (Observe-Orient-Decide-Act) with
utility-based tool selection.

### Single cycle

```bash
# Run one OODA cycle with a goal
akh-medu agent cycle --goal "Find what mammals eat"
```

### Multi-cycle run

```bash
# Run until the goal is satisfied or 20 cycles pass
akh-medu agent run --goals "Discover properties of planets" --max-cycles 20
```

### Interactive REPL

```bash
# Start the agent REPL
akh-medu agent repl
```

In the REPL, type goals in natural language. Commands:
- `p` or `plan` -- show the current plan
- `r` or `reflect` -- trigger reflection
- `q` or `quit` -- exit (session is auto-persisted)

### Resume a session

```bash
# Pick up where you left off
akh-medu agent resume --max-cycles 50
```

## 7. Use the TUI

The unified TUI provides an interactive chat interface:

```bash
akh-medu chat
```

TUI commands (prefix with `/`):
- `/help` -- show available commands
- `/grammar` -- switch grammar archetype (narrative, formal, terse)
- `/workspace` -- show workspace info
- `/goals` -- list active goals
- `/quit` -- exit

Type natural language to set goals or ask questions. The agent runs OODA
cycles automatically and synthesizes findings using the active grammar.

## 8. Export Data

```bash
# Export all symbols as JSON
akh-medu export symbols

# Export all triples
akh-medu export triples

# Export provenance chain for a symbol
akh-medu export provenance --symbol Dog
```

## 9. Graph Analytics

```bash
# Most connected symbols
akh-medu analytics degree --top-k 10

# PageRank importance
akh-medu analytics pagerank --top-k 10

# Strongly connected components
akh-medu analytics components

# Shortest path between two symbols
akh-medu analytics path --from Dog --to Cat
```

## Using the Rust API

All CLI operations are available programmatically:

```rust
use akh_medu::engine::{Engine, EngineConfig};
use akh_medu::symbol::SymbolKind;
use akh_medu::graph::Triple;
use akh_medu::infer::InferenceQuery;

// Create an in-memory engine
let engine = Engine::new(EngineConfig::default())?;

// Create symbols
let dog = engine.create_symbol(SymbolKind::Entity, "Dog")?;
let mammal = engine.create_symbol(SymbolKind::Entity, "mammal")?;
let is_a = engine.create_symbol(SymbolKind::Relation, "is-a")?;

// Add a triple
engine.add_triple(Triple::new(dog.id, is_a.id, mammal.id, 0.95))?;

// Query: what is Dog?
let triples = engine.triples_from(dog.id);
for t in &triples {
    println!("{} -> {} -> {}",
        engine.resolve_label(t.subject),
        engine.resolve_label(t.predicate),
        engine.resolve_label(t.object));
}

// Run inference
let query = InferenceQuery::default()
    .with_seeds(vec![dog.id])
    .with_max_depth(2)
    .with_min_confidence(0.2);

let result = engine.infer(&query)?;
for (sym, conf) in &result.activations {
    println!("  {} (confidence {:.2})", engine.resolve_label(*sym), conf);
}
```

## Next Steps

- Read [Symbols and Triples](../concepts/symbols-and-triples.md) for a deep
  dive into the data model.
- Read [Inference](../concepts/inference.md) for the three inference strategies.
- Read [OODA Loop](../agent/ooda-loop.md) for how the agent works.
- Read [Seed Packs](seed-packs.md) for creating custom knowledge bundles.
