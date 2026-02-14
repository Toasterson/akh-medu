# CLI Reference

## Global Options

| Option | Description | Default |
|--------|-------------|---------|
| `--data-dir <PATH>` | Override default XDG workspace path | XDG default |
| `-w, --workspace <NAME>` | Workspace name | `default` |
| `--dimension <DIM>` | Hypervector dimension | `10000` |
| `--language <LANG>` | Default parsing language | `auto` |

## Commands

### init

Initialize a new workspace with XDG directory structure.

```bash
akh-medu init
akh-medu -w my-project init
```

### workspace

Manage workspaces.

```bash
akh-medu workspace list
akh-medu workspace create <NAME>
akh-medu workspace delete <NAME>
akh-medu workspace info <NAME>
```

### seed

Manage seed packs.

```bash
akh-medu seed list              # List available packs
akh-medu seed apply <PACK>      # Apply a seed pack
akh-medu seed status            # Show applied seeds
```

### ingest

Load triples from files.

```bash
akh-medu ingest --file <PATH>
akh-medu ingest --file <PATH> --format csv --csv-format spo
akh-medu ingest --file <PATH> --format csv --csv-format entity
akh-medu ingest --file <PATH> --format text --max-sentences 100
```

| Option | Description | Default |
|--------|-------------|---------|
| `--file <PATH>` | Input file path | Required |
| `--format <FMT>` | `json`, `csv`, `text` | `json` |
| `--csv-format <FMT>` | `spo` or `entity` | `spo` |
| `--max-sentences <N>` | Max sentences for text format | unlimited |

### bootstrap

Load bundled skills, run grounding, and run inference.

```bash
akh-medu bootstrap
```

### query

Spreading-activation inference.

```bash
akh-medu query --seeds "Dog,Cat" --depth 2 --top-k 10
```

| Option | Description | Default |
|--------|-------------|---------|
| `--seeds <LIST>` | Comma-separated seed symbols | Required |
| `--depth <N>` | Expansion depth | `1` |
| `--top-k <N>` | Max results | `10` |

### traverse

BFS graph traversal.

```bash
akh-medu traverse --seeds Dog --max-depth 2
akh-medu traverse --seeds Dog --predicates is-a --format json
```

| Option | Description | Default |
|--------|-------------|---------|
| `--seeds <LIST>` | Starting symbols | Required |
| `--max-depth <N>` | Traversal depth | `2` |
| `--predicates <LIST>` | Filter by predicates | All |
| `--min-confidence <F>` | Minimum edge confidence | `0.0` |
| `--format <FMT>` | `text` or `json` | `text` |

### sparql

Run SPARQL queries.

```bash
akh-medu sparql "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10"
akh-medu sparql --file query.sparql
```

### reason

Simplify expressions via e-graph rewriting.

```bash
akh-medu reason --expr "unbind(bind(Dog, is-a), is-a)"
akh-medu reason --expr "..." --verbose
```

### search

Find similar symbols via VSA.

```bash
akh-medu search --symbol Dog --top-k 5
```

### analogy

A:B :: C:? analogical reasoning.

```bash
akh-medu analogy --a King --b Man --c Queen --top-k 5
```

### filler

Recover role-filler for (subject, predicate) pairs.

```bash
akh-medu filler --subject Dog --predicate is-a --top-k 5
```

### info

Show engine statistics.

```bash
akh-medu info
```

### symbols

List and inspect symbols.

```bash
akh-medu symbols list
akh-medu symbols show Dog
akh-medu symbols show 42
```

### export

Export engine data.

```bash
akh-medu export symbols
akh-medu export triples
akh-medu export provenance --symbol Dog
```

### skill

Manage skill packs.

```bash
akh-medu skill list
akh-medu skill load <NAME>
akh-medu skill unload <NAME>
akh-medu skill info <NAME>
akh-medu skill scaffold <NAME>     # Create a new skill template
```

### pipeline

Run processing pipelines.

```bash
akh-medu pipeline query --seeds "Dog"
akh-medu pipeline run --stages retrieve,infer,reason --infer-depth 3
```

### analytics

Graph analytics.

```bash
akh-medu analytics degree --top-k 10
akh-medu analytics pagerank --top-k 10
akh-medu analytics components
akh-medu analytics path --from Dog --to Cat
```

### render

Hieroglyphic notation rendering.

```bash
akh-medu render --entity Dog
akh-medu render --entity Dog --depth 3
akh-medu render --all
akh-medu render --legend
akh-medu render --no-color
```

### grammar

Bidirectional grammar system.

```bash
akh-medu grammar list
akh-medu grammar parse "Dogs are mammals"
akh-medu grammar parse "Dogs are mammals" --ingest
akh-medu grammar linearize --subject Dog --predicate is-a --object mammal
akh-medu grammar compare --subject Dog --predicate is-a --object mammal
akh-medu grammar load --file grammar.toml
akh-medu grammar render --entity Dog
```

### chat

Interactive TUI.

```bash
akh-medu chat
akh-medu chat --skill my-skill
akh-medu chat --headless         # No TUI, plain text
```

### preprocess

Pre-process text for the Eleutherios pipeline.

```bash
cat chunks.jsonl | akh-medu preprocess --format jsonl
cat chunks.json  | akh-medu preprocess --format json
cat chunks.jsonl | akh-medu preprocess --format jsonl --language ru
```

### equivalences

Cross-lingual equivalence mappings.

```bash
akh-medu equivalences list
akh-medu equivalences stats
akh-medu equivalences learn
akh-medu equivalences export > equivs.json
akh-medu equivalences import < equivs.json
```

### code-ingest

Ingest Rust source code.

```bash
akh-medu code-ingest --path src/
akh-medu code-ingest --path src/ --recursive --run-rules --enrich
akh-medu code-ingest --path src/main.rs --max-files 50
```

### enrich

Semantic enrichment on existing code knowledge.

```bash
akh-medu enrich
```

### docgen

Generate documentation from code.

```bash
akh-medu docgen --target Engine --format markdown --output docs/
akh-medu docgen --target Engine --format json --polish
```

## Agent Commands

All agent commands are subcommands of `akh-medu agent`.

### agent cycle

Run one OODA cycle.

```bash
akh-medu agent cycle --goal "Find mammals"
akh-medu agent cycle --goal "..." --priority 200
```

### agent run

Run until completion or max cycles.

```bash
akh-medu agent run --goals "Discover planets" --max-cycles 20
akh-medu agent run --goals "..." --fresh    # Ignore persisted session
```

### agent repl

Interactive agent REPL.

```bash
akh-medu agent repl
akh-medu agent repl --goals "Initial goal"
akh-medu agent repl --headless
```

REPL commands: `p`/`plan`, `r`/`reflect`, `q`/`quit`.

### agent resume

Resume a persisted session.

```bash
akh-medu agent resume
akh-medu agent resume --max-cycles 50
```

### agent chat

Agent chat mode.

```bash
akh-medu agent chat
akh-medu agent chat --max-cycles 10 --fresh --headless
```

### agent tools

List registered tools.

```bash
akh-medu agent tools
```

### agent consolidate

Trigger memory consolidation.

```bash
akh-medu agent consolidate
```

### agent recall

Recall episodic memories.

```bash
akh-medu agent recall --query "mammals" --top-k 5
```

### agent plan

Generate and display a goal plan.

```bash
akh-medu agent plan
```

### agent reflect

Trigger reflection.

```bash
akh-medu agent reflect
```

### agent infer

Run forward-chaining rules.

```bash
akh-medu agent infer --max-iterations 10 --min-confidence 0.5
```

### agent gaps

Analyze knowledge gaps.

```bash
akh-medu agent gaps --goal "Explore biology" --max-gaps 10
```

### agent schema

Discover schema patterns.

```bash
akh-medu agent schema
```
