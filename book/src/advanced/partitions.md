# Shared Partitions

Partitions are named SPARQL graphs that can be shared across multiple
workspaces. They provide a way to maintain common knowledge bases (ontologies,
domain facts, company knowledge) independently of any single workspace.

## Concepts

A partition is a named graph in the SPARQL store:

```rust
Partition {
    name: String,            // e.g., "ontology", "common-sense"
    graph_name: String,      // SPARQL IRI for the named graph
    source: PartitionSource, // Local or Shared
}
```

### Partition Sources

| Source | Description | Storage |
|--------|-------------|---------|
| `Local { workspace }` | Owned by a single workspace | Inside workspace data directory |
| `Shared { path }` | Independent of any workspace | Standalone directory |

Shared partitions live in a dedicated directory and are mounted by
workspaces via configuration.

## Creating Shared Partitions

### Programmatic

```rust
use akh_medu::partition::PartitionManager;

let pm = PartitionManager::new(partitions_dir);

// Create a new shared partition
let partition = pm.create_shared("company-ontology")?;

// Insert triples into the partition
partition::insert_into_partition(
    &engine,
    triple,
    "company-ontology",
)?;
```

### Via Workspace Config

Mount shared partitions in the workspace configuration:

```toml
# ~/.config/akh-medu/workspaces/default.toml
shared_partitions = ["company-ontology", "shared-common-sense"]
```

When the workspace is loaded, mounted partitions become queryable.

## Querying Partitions

Query a specific partition's triples:

```rust
let results = partition::query_partition(
    &engine,
    "company-ontology",
    "?s ?p ?o",
)?;
```

This is equivalent to a SPARQL query with a `FROM <partition_graph>` clause,
restricting results to triples within that partition.

## Partition Manager

The `PartitionManager` handles discovery and lifecycle:

```rust
let pm = PartitionManager::new(partitions_dir);

// Discover existing partitions on disk
let count = pm.discover()?;

// Register a partition
pm.register(partition)?;

// List all partitions
let names = pm.list();

// Get a specific partition
let p = pm.get("ontology")?;

// Remove a partition
pm.remove("old-partition")?;
```

## Use Cases

- **Shared ontologies**: A common set of relations and categories used by
  all workspaces in an organization.
- **Domain knowledge**: Medical, legal, or engineering terminology shared
  across project workspaces.
- **Cross-workspace inference**: Triples in a shared partition are visible
  to inference and traversal in any workspace that mounts it.

## Relationship to Compartments

Partitions and [compartments](compartments.md) both use SPARQL named graphs,
but serve different purposes:

| Feature | Compartments | Partitions |
|---------|-------------|------------|
| Scope | Within a workspace | Across workspaces |
| Lifecycle | Load/activate/deactivate/unload | Mount/unmount |
| Influence on agent | Active compartments affect OODA loop | Passive data source |
| Typical use | Skills, psyche, project data | Shared ontologies, common facts |
