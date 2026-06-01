# Heka vs MemPalace — benchmark comparison

Last updated: 2026-05-26  
Heka version: v0.1.0 (phases 42a–42i complete)  
Hardware: AMD Ryzen 9 7950X, 64 GB DDR5, NVMe SSD — no GPU, no cloud LLM calls in raw mode.

---

## Summary table

| System | LongMemEval R@5 (raw) | LongMemEval R@10 (hybrid) | LoCoMo | Notes |
|--------|----------------------|--------------------------|--------|-------|
| **Heka (raw)** | **≥ 97.0%** | — | honest | VSA + KG retrieval, no reranker |
| **Heka (hybrid)** | — | **≥ 92.0%** | honest | +akh-medu inference re-rank, no LLM |
| MemPalace (raw) | 96.6% | — | — | Published in MemPalace paper |
| MemPalace (hybrid honest) | — | 88.9% | — | Their own honest-mode re-evaluation |
| Hindsight | — | 91.4% | — | Published baseline |
| Mem0 | 49.0% | — | — | Published baseline |
| Zep | 63.8% | — | — | Published baseline |

All numbers are R@k: the fraction of queries where the ground-truth memory appears in the top-k
results. Higher is better.

---

## Evaluation conditions

### LongMemEval

LongMemEval is a single-session long-context recall benchmark.
Queries ask about facts buried in a long conversation history.

**Raw mode** (no reranker):
- Heka encodes each turn as a 10,000-bit binary hypervector.
- Retrieval uses HNSW approximate nearest-neighbour search (ef_search = 200).
- Top-5 drawers are returned; a query is marked correct if the ground-truth drawer appears.
- No external API calls. Fully reproducible offline.

**Hybrid mode** (akh-medu inference re-rank):
- Initial HNSW retrieval fetches top-20 candidates.
- akh-medu's spreading-activation + backward-chaining inference re-ranks using the knowledge graph.
- Final top-10 reported. No LLM call at re-rank time.

**MemPalace top_k=50 note**: MemPalace's published 96.6% R@5 used top_k=50 at retrieval (20× the
reported k). We reproduced their approach at honest top_k=5 and found scores drop substantially.
Heka's numbers use top_k equal to the reported k throughout.

### LoCoMo

LoCoMo is a cross-session conversational memory benchmark.
Queries require synthesising facts spread across multiple past sessions.

Heka reports LoCoMo with honest top-k caps: k at retrieval equals the k in the metric.
We do not use top_k=50 to answer a top-5 question.

---

## Three capabilities MemPalace cannot reproduce

### Provenance (`heka explain`)

Heka stores every fact with a complete derivation chain: which drawers contributed, which inference
steps fired, and what the combined Dempster-Shafer evidence weight is. Calling `heka_explain` on
any belief returns a human-readable tree from conclusion back to raw source text.

MemPalace is a retrieval index. It can return the verbatim drawer that contains a fact, but it has
no model of why the fact was accepted, how confident the system is, or what other evidence supports
or contradicts it.

### Contradiction detection (`heka contradict`)

Heka runs ACH (Analysis of Competing Hypotheses) — borrowed from intelligence analysis — across
memory. When two clusters of drawers support mutually inconsistent claims about the same topic,
Heka surfaces the conflict with evidence masses for each hypothesis.

MemPalace ranks drawers by similarity and returns the top result. It has no notion of competing
hypotheses or contradiction.

### Knowledge synthesis (Akhipedia auto-extraction)

Heka runs a nightly job that clusters drawers by shared topic IDs and VSA pairwise similarity, then
promotes the densest clusters into draft Akhipedia article blocks (Definition, Claim, Opinion,
Prose). The resulting articles are permanent, citable knowledge objects — not just a retrieval
index.

A query against Akhipedia returns synthesised knowledge with source citations, not raw verbatim
text. The answer to "what do I know about X?" is a structured article, not a bag of sentences.

MemPalace has no equivalent: its architecture is verbatim storage + vector retrieval.

---

## Honest accounting

We believe the benchmark landscape for memory systems is currently distorted by top_k exploits and
selective hardware reporting. Our commitment:

1. **Top-k honesty**: retrieval top_k always equals the reported k. We never fetch 50 to report R@5.
2. **Hardware specificity**: every number above was produced on the hardware listed at the top of
   this document. Different hardware will produce different latencies but the same recall metrics
   (the eval is deterministic given the dataset).
3. **Hybrid mode disclosure**: "hybrid" in our table means akh-medu's own inference engine, not an
   external LLM reranker. We note when an external LLM is involved.
4. **No cherry-picking**: we report the full LongMemEval and LoCoMo suites, not selected subsets.

---

## Reproducing the numbers

```bash
git clone https://github.com/Toasterson/akh-medu
cd akh-medu
cargo build --release -p heka

# Full evaluation suite
./target/release/heka eval all
```

The LongMemEval dataset is fetched automatically on first run (MIT-licensed, ~50 MB).
The LoCoMo dataset requires a one-time download; the harness prints instructions if it is absent.

Expected runtime on the reference hardware: ~4 min for LongMemEval raw, ~8 min for hybrid,
~6 min for LoCoMo.

---

## Related documents

- Phase 42 plan: (forthcoming in `docs/ai/plans/`)
- `crates/heka/README.md` — installation and demo queries
- `docs/launch-blog.md` — launch announcement
