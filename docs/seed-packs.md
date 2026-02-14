# Seed Packs

Seed packs are TOML-defined knowledge bundles that bootstrap workspaces with foundational triples.

## Bundled Packs

Three packs are compiled into the binary:

| Pack | Triples | Description |
|------|---------|-------------|
| `identity` | ~20 | Core identity: who akh-medu is, its capabilities and components |
| `ontology` | ~30 | Fundamental relations (is-a, has-part, causes, etc.) and category hierarchy |
| `common-sense` | ~40 | Basic world knowledge: animals, materials, spatial/temporal concepts |

## Usage

```bash
# List available seed packs
akh-medu seed list

# Apply a specific pack
akh-medu seed apply identity

# Check which seeds are applied to the current workspace
akh-medu seed status
```

Seeds are **idempotent**: applying the same seed twice has no effect. Applied seeds are tracked via the `akh:seed-applied` predicate in the knowledge graph.

## Creating a Custom Seed Pack

### Directory structure

```
my-seed/
  seed.toml     # manifest + inline triples
```

### seed.toml format

```toml
[seed]
id = "my-custom-pack"
name = "My Custom Knowledge"
version = "1.0.0"
description = "Domain-specific knowledge for my project"

[[triples]]
subject = "rust"
predicate = "is-a"
object = "programming language"
confidence = 0.95

[[triples]]
subject = "cargo"
predicate = "is-a"
object = "build tool"
confidence = 0.9

[[triples]]
subject = "cargo"
predicate = "has-part"
object = "dependency resolver"
confidence = 0.85
```

### Fields

| Field | Required | Description |
|-------|----------|-------------|
| `[seed].id` | yes | Unique identifier for the pack |
| `[seed].name` | yes | Human-readable name |
| `[seed].version` | yes | Semantic version string |
| `[seed].description` | yes | Brief description |
| `[[triples]].subject` | yes | Subject entity label |
| `[[triples]].predicate` | yes | Predicate relation label |
| `[[triples]].object` | yes | Object entity label |
| `[[triples]].confidence` | no | Confidence score 0.0-1.0 (default: 0.8) |

## Installing a Seed Pack

Copy the seed directory to the seeds location:

```bash
cp -r my-seed/ ~/.local/share/akh-medu/seeds/my-seed/
```

The pack will appear in `akh-medu seed list` on next invocation.

## Auto-Seeding on Workspace Creation

Workspace configs specify which seeds to apply on first initialization:

```toml
# ~/.config/akh-medu/workspaces/default.toml
name = "default"
seed_packs = ["identity", "ontology", "common-sense"]
```

When a workspace is created, the listed seeds are applied automatically.
