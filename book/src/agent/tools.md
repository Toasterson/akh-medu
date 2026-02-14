# Tools

The agent has 17 built-in tools organized into four categories: core
knowledge tools, external interaction tools, library tools, and advanced
tools.

## Tool Architecture

Each tool implements the `Tool` trait:

```rust
pub trait Tool: Send + Sync {
    fn signature(&self) -> ToolSignature;
    fn execute(&self, engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput>;
    fn manifest(&self) -> ToolManifest;
}
```

Tools produce a `ToolOutput` with:
- `success: bool` -- whether the operation succeeded
- `result: String` -- human-readable summary
- `symbols_involved: Vec<SymbolId>` -- entities touched (linked in working memory)

## Core Knowledge Tools

### kg_query

Query the knowledge graph by symbol, predicate, or direction.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `symbol` | yes | Symbol name or ID to query |
| `predicate` | no | Filter by predicate |
| `direction` | no | `outgoing` (default) or `incoming` |

### kg_mutate

Create new triples in the knowledge graph.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `subject` | yes | Subject entity label |
| `predicate` | yes | Relation label |
| `object` | yes | Object entity label |
| `confidence` | no | Confidence score (default: 0.8) |

### memory_recall

Fetch episodic memories relevant to the current context.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `query_symbols` | yes | Comma-separated symbol names to match against |
| `top_k` | no | Maximum results (default: 5) |

### reason

Simplify expressions via e-graph rewriting.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `expr` | yes | Expression to simplify |
| `verbose` | no | Show e-graph state |

### similarity_search

Find similar symbols via VSA hypervector similarity.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `symbol` | yes | Symbol name or ID |
| `top_k` | no | Maximum results (default: 5) |

## External Interaction Tools

### file_io

Read and write files, sandboxed to the workspace's scratch directory.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `operation` | yes | `read` or `write` |
| `path` | yes | File path (relative to scratch dir) |
| `content` | write only | Content to write |

Limits: 4 KB read truncation, 256 KB write limit.

### http_fetch

Synchronous HTTP GET via `ureq`.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `url` | yes | URL to fetch |
| `timeout_secs` | no | Request timeout (default: 30) |

Limit: 256 KB response truncation.

### shell_exec

Execute shell commands with poll-based timeout.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `command` | yes | Shell command to run |
| `timeout_secs` | no | Timeout (default: 30) |

Limits: 64 KB output, process killed on timeout.

### user_interact

Prompt the user for input via stdout/stdin.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `prompt` | yes | Question to display |
| `timeout_secs` | no | Input timeout |

## Library Tools

### content_ingest

Ingest a document (file or URL) into the [shared content library](../library/overview.md).
Parses HTML, PDF, EPUB, or plain text. Extracts triples and creates VSA
embeddings for semantic search.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `source` | yes | File path or URL to ingest |
| `title` | no | Override document title |
| `tags` | no | Comma-separated tags for categorization |

Danger level: **Cautious** (fetches/reads documents, writes triples + VSA embeddings).

### library_search

Search the shared content library for paragraphs matching a natural language
query via VSA semantic similarity. Returns the most relevant library paragraphs
with document context and similarity scores.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `query` | yes | Natural language search text |
| `top_k` | no | Number of results (default: 5) |
| `document` | no | Filter to a specific document slug |

Danger level: **Safe** (read-only VSA search, no side effects).

## Advanced Tools

### infer_rules

Run forward-chaining inference via e-graph rewrite rules.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `max_iterations` | no | Rule application iterations (default: 10) |
| `min_confidence` | no | Minimum confidence threshold (default: 0.5) |

### gap_analysis

Discover knowledge gaps by analyzing the graph structure.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `goal` | no | Focus analysis around a specific goal |
| `max_gaps` | no | Maximum gaps to report (default: 10) |

### csv_ingest

Ingest structured data from CSV files.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `path` | yes | Path to CSV file |
| `format` | no | `spo` (subject-predicate-object) or `entity` (header columns as predicates) |

### text_ingest

Extract facts from natural language text using the grammar parser.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `text` | yes | Text to parse, or `file:/path` to read from file |
| `max_sentences` | no | Maximum sentences to process |

### code_ingest

Parse Rust source code into knowledge graph entities.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `path` | yes | Path to Rust source file or directory |
| `recursive` | no | Recurse into subdirectories |
| `run_rules` | no | Apply inference rules after ingestion |
| `enrich` | no | Run semantic enrichment |

### docgen

Generate documentation from code knowledge in the graph.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `target` | yes | Symbol or path to document |
| `format` | no | `markdown`, `json`, or `both` |
| `polish` | no | Apply grammar polishing |

## Danger Metadata

Each tool carries a `ToolManifest` with danger metadata:

```rust
DangerLevel::Safe     // No external I/O (kg_query, reason, etc.)
DangerLevel::Caution  // Read-only external access (http_fetch, file_io read)
DangerLevel::Danger   // Write/exec capability (shell_exec, file_io write)
```

Capabilities tracked:
- `ReadKg`, `WriteKg` -- knowledge graph operations
- `VsaAccess` -- VSA similarity search
- `ReadFilesystem` -- filesystem reads
- `Network` -- HTTP requests
- `ShellExec` -- command execution
- `UserInteraction` -- stdin/stdout

The [psyche system](psyche.md) uses `shadow_triggers` from the manifest to
match veto patterns against tool usage.

## Tool Selection

During the Decide phase, each tool receives a utility score:

```
total_score = base_score - recency_penalty + novelty_bonus
            + episodic_bonus + pressure_bonus + archetype_bonus
```

The tool with the highest score is selected. See
[OODA Loop](ooda-loop.md) for details on the scoring factors.

## Custom Tools via Skills

Skill packs can register additional tools:
- **CLI tools**: JSON manifests describing shell commands.
- **WASM tools**: WebAssembly components (requires `wasm-tools` feature).

```bash
# List all registered tools (built-in + skill-provided)
akh-medu agent tools
```

## Listing Tools

```bash
akh-medu agent tools
```

This shows all registered tools with their names, descriptions, parameters,
and danger levels.
