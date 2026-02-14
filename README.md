# akh-medu

A neuro-symbolic AI engine made to run on the CPU. Made for people. Not for robber barons.

**[Documentation](https://akh-medu.dev)**

akh-medu combines hyperdimensional computing (Vector Symbolic Architecture)
with knowledge graphs and symbolic reasoning. It runs entirely on the CPU
with no LLM dependency, no GPU requirement, and no external NLP models.

- **Infer** new knowledge via spreading activation, backward chaining, and
  superposition reasoning
- **Reason** symbolically using e-graph rewrite rules (equality saturation)
- **Search** semantically using 10,000-bit binary hypervectors
- **Act** autonomously via an OODA-loop agent with 15 built-in tools
- **Parse** and generate natural language in 5 languages via a grammar framework
- **Serve** knowledge over REST and WebSocket APIs

## Quick Start

```bash
cargo build --release
akh-medu init
akh-medu seed apply ontology
akh-medu grammar parse "Dogs are mammals" --ingest
akh-medu query --seeds Dog --depth 2
```

See the [Quick Start Tutorial](https://akh-medu.dev/getting-started/quickstart.html)
for a full walkthrough.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                      Engine API                          │
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

## Documentation

The full documentation is hosted at **[akh-medu.dev](https://akh-medu.dev)** and covers:

- [Installation](https://akh-medu.dev/getting-started/installation.html) -- build from source, feature flags
- [Quick Start Tutorial](https://akh-medu.dev/getting-started/quickstart.html) -- first engine, symbols, queries, agent
- [Concepts](https://akh-medu.dev/concepts/architecture.html) -- VSA, knowledge graphs, inference, reasoning
- [Agent](https://akh-medu.dev/agent/ooda-loop.html) -- OODA loop, tools, planning, Jungian psyche
- [Server](https://akh-medu.dev/server/overview.html) -- REST and WebSocket APIs
- [CLI Reference](https://akh-medu.dev/reference/cli.html) -- all commands and options

To build the docs locally:

```bash
cd book && mdbook serve
```

## License

This repository is licensed under **GPLv3**. See the [LICENSE](LICENSE) file.

For integration into proprietary applications, contact the author.

## Contributing

To make it possible to milk as much money as possible from proprietary vendors
back into FLOSS work, every contributor must agree that we will sell very
expensive proprietary licenses.
