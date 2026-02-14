# Grammars

The grammar framework is a bidirectional system for parsing natural language
into structured data (abstract syntax trees) and linearizing structured data
back into prose. It operates without any ML models -- all parsing and
generation is rule-based.

## Architecture

```
Prose Input --> Lexer --> Parser --> AbsTree --> ConcreteGrammar --> Styled Prose
   |             |          |          ^              ^
SymbolRegistry  VSA      bridge.rs    GrammarRegistry (formal/terse/
(exact match)   (fuzzy)              narrative/custom)
```

The system has two layers:

1. **Abstract layer**: Language-neutral `AbsTree` nodes representing
   entities, relations, triples, and sequences.
2. **Concrete layer**: Grammar-specific linearization rules that turn
   `AbsTree` nodes into styled prose.

## Abstract Syntax Trees

The `AbsTree` type represents parsed meaning:

| Variant | Description | Example |
|---------|-------------|---------|
| `Entity(String)` | A named thing | `Entity("Dog")` |
| `Relation(String)` | A relationship | `Relation("is-a")` |
| `Triple { subj, pred, obj }` | An RDF triple | `Triple(Entity("Dog"), Relation("is-a"), Entity("mammal"))` |
| `List(Vec<AbsTree>)` | Ordered collection | Multiple triples |
| `Sequence(Vec<AbsTree>)` | Narrative sequence | Story-like output |
| `Tag { tree, tag }` | Provenance/role tag | VSA role annotation |

## Built-in Grammar Archetypes

### Narrative

Flowing, story-like prose for interactive sessions:

```
The Dog has a relationship of type is-a with mammal.
Furthermore, the mammal possesses the property warm blood.
```

### Formal

Structured, academic-style output:

```
## Dog
- is-a: mammal (confidence: 0.95)
- has-part: tail (confidence: 0.85)
```

### Terse

Minimal output, facts only:

```
Dog is-a mammal [0.95]
mammal has warm blood [0.85]
```

## Using the Grammar System

### CLI

```bash
# Parse prose into abstract syntax
akh-medu grammar parse "Dogs are mammals"

# Parse and ingest into the knowledge graph
akh-medu grammar parse "Dogs are mammals" --ingest

# Linearize a triple back to prose
akh-medu grammar linearize --subject Dog --predicate is-a --object mammal

# Compare a triple against the KG
akh-medu grammar compare --subject Dog --predicate is-a --object mammal

# List available archetypes
akh-medu grammar list

# Load a custom grammar from TOML
akh-medu grammar load --file /path/to/grammar.toml

# Render an entity's KG neighborhood
akh-medu grammar render --entity Dog
```

### In the TUI

Switch grammar archetypes with the `/grammar` command:

```
/grammar narrative
/grammar formal
/grammar terse
```

The active grammar controls how the agent formats its responses.

## Custom Grammars

Implement the `ConcreteGrammar` trait:

```rust
pub trait ConcreteGrammar: Send + Sync {
    fn name(&self) -> &str;
    fn linearize(&self, tree: &AbsTree, ctx: &GrammarContext) -> GrammarResult<String>;
    fn parse(&self, input: &str, expected_cat: Option<&Cat>, ctx: &GrammarContext) -> GrammarResult<AbsTree>;
}
```

Or load a TOML-defined grammar at runtime:

```bash
akh-medu grammar load --file my-grammar.toml
```

Register custom grammars programmatically:

```rust
engine.grammar_registry().register(Box::new(MyGrammar));
engine.grammar_registry().set_default("my-grammar")?;
```

## Multilingual Support

The grammar system supports five languages:

| Language | Code | Relational Patterns | Void Words |
|----------|------|---------------------|------------|
| English | `en` | 21 | a, an, the |
| Russian | `ru` | 13 | (none) |
| Arabic | `ar` | 11 | al |
| French | `fr` | 16 | le, la, les, un, une, etc. |
| Spanish | `es` | 14 | el, la, los, las, un, una, etc. |

Language detection is automatic (script analysis + word frequency heuristics)
or can be forced:

```bash
# Auto-detect
akh-medu grammar parse "Собаки являются млекопитающими"

# Force Russian
akh-medu --language ru grammar parse "Собаки являются млекопитающими"
```

All languages map to the same 9 canonical predicates: `is-a`, `has-a`,
`contains`, `located-in`, `causes`, `part-of`, `composed-of`, `similar-to`,
`depends-on`.

## Lexer and Parser Pipeline

1. **Lexer**: Tokenizes input, strips void words, matches relational patterns.
   Unknown tokens are resolved via the symbol registry (exact match) or VSA
   item memory (fuzzy match, threshold >= 0.60).

2. **Parser**: Builds `AbsTree` from tokens. Detects intent (query, goal,
   statement) and categorizes grammatically.

3. **Bridge**: Converts between `AbsTree` and KG operations. `commit_abs_tree()`
   inserts parsed triples into the knowledge graph.

## Agent Integration

The agent's persona controls the default grammar via `grammar_preference`:

```toml
[persona]
name = "Scholar"
grammar_preference = "narrative"
```

When `Agent::synthesize_findings()` runs, it uses the persona's preferred
grammar to linearize results. See [Jungian Psyche](../agent/psyche.md) for
details.
