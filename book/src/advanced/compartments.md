# Knowledge Compartments

The compartment system isolates knowledge by purpose. Instead of dumping all
triples into a single global graph, triples are tagged with a `compartment_id`
and stored in named graphs in Oxigraph. This makes knowledge portable,
removable, and scopable -- a skill's knowledge can be cleanly loaded and
unloaded without polluting the rest of the graph.

## Compartment Kinds

| Kind | Description | Lifecycle |
|------|-------------|-----------|
| **Core** | Always-active modules (psyche, personality). | Loaded at engine startup, never unloaded during normal operation. |
| **Skill** | Travels with skill packs. | Loaded/unloaded when skills activate/deactivate. |
| **Project** | Scoped to a specific project. | Loaded when the user switches project context. |

## Lifecycle States

Each compartment passes through three states:

```
Dormant  --load()-->  Loaded  --activate()-->  Active
   ^                     |                        |
   |                     |                        |
   +---- unload() -------+---- deactivate() ------+
```

- **Dormant**: Discovered on disk but not loaded. No triples in memory.
- **Loaded**: Triples are in the KG; queries can be scoped to this compartment.
- **Active**: Loaded AND actively influencing the OODA loop (e.g., psyche
  archetypes adjust tool scoring, shadow patterns gate actions).

## On-Disk Layout

Compartments live under `data/compartments/`. Each compartment is a directory
containing at minimum a `compartment.toml` manifest:

```
data/compartments/
  psyche/
    compartment.toml      # manifest (required)
    psyche.toml           # psyche-specific config (optional)
  personality/
    compartment.toml
  my-skill/
    compartment.toml
    triples.json          # knowledge triples to load (optional)
    rules.toml            # reasoning rules (optional)
```

## Manifest Format

`compartment.toml`:

```toml
id = "psyche"                    # Unique identifier (required)
name = "Jungian Psyche"          # Human-readable name (required)
kind = "Core"                    # "Core", "Skill", or "Project" (required)
description = "Agent psyche..."  # Purpose description
triples_file = "triples.json"   # Path to triples JSON (relative)
rules_file = "rules.toml"       # Path to rules file
grammar_ref = "narrative"        # Built-in grammar or custom TOML path
tags = ["core", "ethics"]        # Domain tags for search
```

## Triples JSON Format

When `triples_file` is specified, it should be a JSON array of triple objects:

```json
[
  {
    "subject": "Sun",
    "predicate": "has_type",
    "object": "Star",
    "confidence": 0.95
  },
  {
    "subject": "Earth",
    "predicate": "orbits",
    "object": "Sun",
    "confidence": 1.0
  }
]
```

Each triple is loaded with `compartment_id` set to the compartment's `id`.
Symbols are resolved or created automatically via the engine.

## Triple Tagging

Every `Triple` and `EdgeData` carries an optional `compartment_id: Option<String>`.
When a compartment loads triples, they are tagged with its ID. This enables:

- **Scoped queries**: `SparqlStore::query_in_graph(sparql, compartment_id)` injects
  a `FROM <graph_iri>` clause to restrict results.
- **Clean removal**: `SparqlStore::remove_graph(compartment_id)` drops all triples
  belonging to a compartment without touching others.
- **Provenance**: `DerivationKind::CompartmentLoaded { compartment_id, source_file }`
  tracks which compartment introduced a triple.

Named graph IRIs follow the pattern: `https://akh-medu.dev/compartment/{id}`.

## Usage

```rust
use akh_medu::compartment::{CompartmentManager, CompartmentKind};

let engine = Engine::new(config)?;

if let Some(cm) = engine.compartments() {
    // List discovered compartments.
    let cores = cm.compartments_by_kind(CompartmentKind::Core);

    // Load a compartment's triples into the KG.
    cm.load("my-skill", &engine)?;

    // Mark it as active (influences OODA loop).
    cm.activate("my-skill")?;

    // Query only this compartment's knowledge.
    if let Some(sparql) = engine.sparql() {
        let results = sparql.query_in_graph(
            "SELECT ?s ?p ?o WHERE { ?s ?p ?o }",
            Some("my-skill"),
        )?;
    }

    // Deactivate but keep triples loaded.
    cm.deactivate("my-skill")?;

    // Fully unload (remove triples from KG).
    cm.unload("my-skill", &engine)?;
}
```

## Creating a New Compartment

1. Create a directory: `data/compartments/astronomy/`

2. Add `compartment.toml`:
   ```toml
   id = "astronomy"
   name = "Astronomy Knowledge"
   kind = "Skill"
   description = "Star catalogs and celestial mechanics"
   triples_file = "triples.json"
   tags = ["science", "astronomy"]
   ```

3. Add `triples.json`:
   ```json
   [
     {"subject": "Sun", "predicate": "has_type", "object": "G-type_star", "confidence": 1.0},
     {"subject": "Earth", "predicate": "orbits", "object": "Sun", "confidence": 1.0},
     {"subject": "Mars", "predicate": "orbits", "object": "Sun", "confidence": 1.0}
   ]
   ```

4. The engine auto-discovers it on startup. Or trigger manually:
   ```rust
   if let Some(cm) = engine.compartments() {
       cm.discover()?;
       cm.load("astronomy", &engine)?;
       cm.activate("astronomy")?;
   }
   ```

## Error Handling

All compartment operations return `CompartmentResult<T>`. Errors include:

| Error | Code | When |
|-------|------|------|
| `NotFound` | `akh::compartment::not_found` | ID not in registry. Run `discover()` first. |
| `AlreadyLoaded` | `akh::compartment::already_loaded` | Tried to load a non-Dormant compartment. |
| `InvalidManifest` | `akh::compartment::invalid_manifest` | Malformed `compartment.toml` or `triples.json`. |
| `Io` | `akh::compartment::io` | Filesystem error. |
| `KindMismatch` | `akh::compartment::kind_mismatch` | Wrong compartment kind for the operation. |

## Provenance

Three provenance variants track compartment and psyche activity:

| Variant | Tag | Description |
|---------|-----|-------------|
| `CompartmentLoaded` | 15 | Records when triples are loaded from a compartment. |
| `ShadowVeto` | 16 | Records when a shadow pattern blocks an action. |
| `PsycheEvolution` | 17 | Records when the psyche auto-adjusts during reflection. |

## Design Rationale

Without isolation, a skill's triples are indistinguishable from the rest of
the KG once loaded. You can't unload a skill without knowing exactly which
triples it introduced. Compartment tagging solves this -- `compartment_id`
on every triple makes removal clean and queries scopable.
