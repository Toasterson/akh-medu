# Introduction

**akh-medu** is a neuro-symbolic AI engine that combines hyperdimensional computing
(Vector Symbolic Architecture) with knowledge graphs and symbolic reasoning.
It runs entirely on the CPU with no LLM dependency, no GPU requirement, and
no external NLP models.

## What It Does

akh-medu stores, reasons about, and discovers knowledge. You feed it facts
(triples like "Dog is-a Mammal"), and it can:

- **Infer** new knowledge via spreading activation, backward chaining, and
  superposition reasoning
- **Reason** symbolically using e-graph rewrite rules (equality saturation)
- **Search** semantically using high-dimensional binary vectors (VSA)
- **Act** autonomously via an OODA-loop agent with 15 built-in tools
- **Parse** and generate natural language in 5 languages via a grammar framework
- **Serve** knowledge over REST and WebSocket APIs

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                      Engine API                          │
│  create_symbol · add_triple · infer · query · traverse   │
├──────────┬──────────┬──────────┬──────────┬──────────────┤
│   VSA    │Knowledge │Reasoning │Inference │   Agent      │
│  Ops     │  Graph   │  (egg)   │ Engine   │  OODA Loop   │
│ ─────────│──────────│──────────│──────────│──────────────│
│ HyperVec │petgraph  │ rewrite  │spreading │ 15 tools     │
│ SIMD     │oxigraph  │ rules    │backward  │ planning     │
│ ItemMem  │SPARQL    │ e-graphs │superpos. │ psyche       │
├──────────┴──────────┴──────────┴──────────┴──────────────┤
│                   Tiered Storage                         │
│  Hot (DashMap) · Warm (mmap) · Durable (redb)            │
└──────────────────────────────────────────────────────────┘
```

## Key Differentiators

- **CPU-only**: SIMD-accelerated hypervector operations (AVX2 where available,
  generic fallback everywhere else). No GPU, no CUDA, no model weights.
- **No LLM dependency**: All reasoning is algebraic (VSA bind/unbind, e-graph
  rewriting, graph traversal). Grammar-based NLP replaces transformer models.
- **Hyperdimensional computing**: 10,000-bit binary vectors encode symbols.
  Similarity is Hamming distance. Binding is XOR. Bundling is majority vote.
- **Full provenance**: Every derived fact has a traceable derivation chain
  recording exactly how it was produced.
- **Multilingual**: Grammar-based parsing and generation in English, Russian,
  Arabic, French, and Spanish with cross-language entity resolution.

## How to Read This Book

- **[Getting Started](getting-started/installation.md)** walks you through
  building from source and your first knowledge session.
- **[Concepts](concepts/architecture.md)** explains the core data model and
  reasoning strategies.
- **[Agent](agent/ooda-loop.md)** covers the autonomous OODA-loop agent.
- **[Advanced](advanced/compartments.md)** dives into compartments, workspaces,
  grammars, and shared partitions.
- **[Server](server/overview.md)** documents the REST and WebSocket APIs.
- **[Reference](reference/components.md)** has the component status matrix and
  full CLI command reference.

## Source Code

The source code is on GitHub: [Toasterson/akh-medu](https://github.com/Toasterson/akh-medu)

## License

akh-medu is licensed under the **GPLv3**. For integration into proprietary
applications, contact the author.
