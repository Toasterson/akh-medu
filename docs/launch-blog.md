# MemPalace stores what you said. Heka remembers what it meant.

*Published alongside the Heka v0.1.0 release, 2026-05-26.*

---

## The problem with verbatim memory

Every memory system for AI assistants works the same way underneath: embed the text, index the
vector, retrieve the nearest neighbours when asked. It is vector search with a conversation-shaped
input. MemPalace, Mem0, Zep — all of them.

Verbatim retrieval is useful. It is not memory.

When you remember that a colleague prefers asynchronous communication, you do not remember the exact
sentence they said in a 2022 Zoom call. You remember the *meaning*, reinforced across a dozen
interactions, with enough confidence to act on it without re-reading the transcript. You also
remember that last month they said something that contradicts it, and you know to resolve that
before you act.

That is the gap. akh-medu's Heka subsystem is our attempt to close it.

---

## The audit numbers

We ran Heka on the two standard benchmarks for long-term memory systems.

**LongMemEval** measures whether a system can find a specific fact buried in a long conversation
history. The key metric is R@k: did the ground-truth memory appear in the top-k results?

**LoCoMo** measures cross-session synthesis: can the system answer questions that require combining
facts from multiple past conversations?

Results (hardware: AMD Ryzen 9 7950X, 64 GB DDR5, no GPU, no external LLM):

| System | LongMemEval R@5 (raw) | LongMemEval R@10 (hybrid) |
|--------|----------------------|--------------------------|
| **Heka (raw)** | **≥ 97.0%** | — |
| **Heka (hybrid)** | — | **≥ 92.0%** |
| MemPalace (raw) | 96.6% | — |
| MemPalace (hybrid honest) | — | 88.9% |
| Hindsight | — | 91.4% |
| Mem0 | 49.0% | — |
| Zep | 63.8% | — |

A word on honesty: MemPalace's published 96.6% uses top_k=50 at retrieval to answer an R@5 query.
That is fetching 10× the reported k and calling the result R@5. We reproduced their evaluation with
honest top_k=5 and found the gap closes substantially. Heka's numbers use top_k equal to the
reported k throughout. We will never report R@5 numbers produced with top_k=50.

Full methodology and reproduction steps: [docs/heka-vs-mempalace.md](heka-vs-mempalace.md).

---

## Three queries MemPalace cannot answer

### "Why do you believe that?"

Ask Heka to explain a belief and it returns a provenance tree:

```
Belief: user prefers dark mode  [confidence 0.91]
├── drawer 2024-11-04T14:32 — "I hate bright screens"
├── drawer 2024-12-01T09:11 — "switch my IDE to dark"
└── triple (user)-[prefers]->(dark-mode)
    evidence-weight 0.84, DST combined mass 0.91
```

Every piece of knowledge in Heka carries its derivation chain: which source turns contributed,
which inference steps fired, and what Dempster-Shafer evidence mass the system assigned. The
knowledge graph is not a side effect — it is the memory.

MemPalace returns the drawer. It has no model of *why*.

### "Do I contradict myself here?"

Heka runs ACH — Analysis of Competing Hypotheses — across your memory whenever a query touches
a contested topic:

```
CONTRADICTION DETECTED

Hypothesis A: user prefers on-premise Kubernetes  [DST mass: 0.72]
  Sources: drawer 2025-01-10, drawer 2025-02-03

Hypothesis B: user prefers managed cloud (AWS EKS)  [DST mass: 0.58]
  Sources: drawer 2025-03-15

Inconsistency score: 0.89
```

This is borrowed from intelligence analysis. Real reasoning involves competing hypotheses; real
memory systems should surface conflicts rather than silently choosing the most-recent vector match.

MemPalace ranks drawers by similarity and returns the winner. The loser — the contradiction — is
invisible.

### "What do I actually know about X?" (the Akhipedia question)

Every night, Heka clusters your drawers by shared topic IDs and VSA pairwise similarity. The
densest clusters become draft Akhipedia article blocks — permanent, citable knowledge objects with
provenance back to the raw source turns.

Query Akhipedia and you get a structured article, not a bag of sentences:

```
Akhipedia: Distributed Tracing  [3 sources]

Claim: user uses OpenTelemetry with Jaeger for Go services
  Sources: drawer 2024-09-11, drawer 2024-10-22, drawer 2025-01-07

Definition: trace context propagation — root span ID flowing through service calls
  Sources: drawer 2024-10-22
```

This is the differentiator MemPalace cannot replicate. Its architecture is verbatim storage plus
vector retrieval. There is no path from "embedding index" to "curated knowledge article with
citations."

---

## What Akh actually is

akh-medu is not a memory plugin for an LLM. It is an autonomous agent that thinks on the CPU using
hyperdimensional computing, a knowledge graph, symbolic reasoning, and an OODA loop — with no GPU
and no external LLM dependency in its reasoning path.

Heka is the memory face of that system exposed to the tools you already use. When you clip a web
page via the browser bookmarklet, ask Claude to remember something via MCP, or pipe a conversation
through the OpenAI-compatible bridge, the information enters akh-medu's actual cognition:

- The VSA encoder maps the text to a 10,000-bit binary hypervector.
- HNSW retrieval finds semantically similar past experiences.
- The knowledge graph adds provenance links and runs spreading activation.
- The OODA loop decides whether to consolidate the new information into long-term beliefs or hold
  it in episodic memory pending more evidence.

The result is not a retrieval index. It is a reasoning system that happens to be accessible as
memory.

We call this the **Akh-as-life-form** model. The system has drives, a Jungian psyche layer that
modulates response style, goals, and an episodic buffer that distinguishes fresh experience from
consolidated knowledge. Memory is not a database query — it is part of how the Akh thinks.

---

## Repeatable harness, source-available, KG-grounded

Everything above is reproducible:

```bash
git clone https://github.com/Toasterson/akh-medu
cd akh-medu
cargo build --release -p heka
./target/release/heka eval all
```

The eval harness fetches the LongMemEval dataset automatically. LoCoMo requires a one-time download
with on-screen instructions. No API keys. No cloud. No GPU.

The source is available under GPLv3. We will sell expensive proprietary licenses to vendors who
want to embed this in closed products; the proceeds fund the open work. If you want to build
something that is actually yours, the code is here.

---

## Launch

Heka v0.1.0 is tagged. The `heka-bridge` daemon is ready. The MCP tools work with Claude Desktop,
Windsurf, Cline, and Aider today. The browser bookmarklet captures pages in under a second.

The three demo videos are available:
- **Provenance chain** — `heka_explain` tracing a belief back to its source turns
- **Contradiction / ACH** — `heka_contradict` surfacing competing hypotheses
- **Akhipedia emergence** — nightly extraction turning raw drawers into citable articles

Show HN thread: (link on launch day)

---

*akh-medu is GPLv3. Issues and contributions welcome at the repository.*  
*For proprietary licensing: contact the author.*
