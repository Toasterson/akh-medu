# akh-medu + Eleutherios Integration Guide

## Overview

akh-medu serves as a **multilingual pre-processor** that sits between Eleutherios's
document chunking stage and its LLM extraction pipeline. Where Eleutherios's
7-dimensional extraction is English-strong but degrades for Russian, Arabic, and
historical mixed-language sources, akh-medu's GF-based grammar parser runs in
sub-millisecond time per chunk and produces **language-neutral** structured data:
entities, relations, and claims with confidence scores.

```
Documents
    |
    v
Eleutherios chunking (document -> text chunks)
    |
    v
akh-medu pre-processor (text -> entities + claims + AbsTrees)
    |
    v
Eleutherios LLM enrichment (Mistral Nemo 12B -> 7D extraction)
    |
    v
Neo4j / pgvector
```

The pre-processor gives Eleutherios **cleaner starting data** that reduces extraction
noise, particularly for multilingual corpora (technical manuals, diplomatic
correspondence, academic texts with mixed-language citations, etc.).

---

## Table of Contents

- [Quick Start](#quick-start)
  - [Build](#build)
  - [CLI Pipeline (JSONL)](#cli-pipeline-jsonl)
  - [CLI Pipeline (JSON batch)](#cli-pipeline-json-batch)
  - [HTTP Server](#http-server)
- [Running with Eleutherios Docker](#running-with-eleutherios-docker)
  - [Prerequisites](#prerequisites)
  - [Step 1: Start Ollama on All Interfaces](#step-1-start-ollama-on-all-interfaces)
  - [Step 2: Start Eleutherios Services](#step-2-start-eleutherios-services)
  - [Step 3: Build and Start akh-medu](#step-3-build-and-start-akh-medu)
  - [Step 4: Load Documents](#step-4-load-documents)
  - [Step 5: Run the Pipeline](#step-5-run-the-pipeline)
  - [Step 6: Pre-Process with akh-medu](#step-6-pre-process-with-akh-medu)
  - [Benchmark: Real-World Performance](#benchmark-real-world-performance)
  - [Memory Requirements](#memory-requirements)
- [Step-by-Step Tutorial](#step-by-step-tutorial)
  - [1. Prepare Your Corpus](#1-prepare-your-corpus)
  - [2. Pre-Process via CLI](#2-pre-process-via-cli)
  - [3. Pre-Process via HTTP](#3-pre-process-via-http)
  - [4. Interpret the Output](#4-interpret-the-output)
  - [5. Feed into Eleutherios](#5-feed-into-eleutherios)
- [What Meaning We Extract](#what-meaning-we-extract)
  - [Claim Types](#claim-types)
  - [Entity Types](#entity-types)
  - [Canonical Predicates](#canonical-predicates)
- [Available Languages](#available-languages)
  - [Language Detection](#language-detection)
  - [Mixed-Language Corpora](#mixed-language-corpora)
- [Cross-Language Entity Resolution](#cross-language-entity-resolution)
  - [Static Equivalence Table](#static-equivalence-table)
  - [Dynamic Equivalence Learning](#dynamic-equivalence-learning)
  - [Managing Equivalences via CLI](#managing-equivalences-via-cli)
  - [Managing Equivalences via HTTP](#managing-equivalences-via-http)
  - [Seeding Domain-Specific Equivalences](#seeding-domain-specific-equivalences)
- [VSA Similarity Algorithms](#vsa-similarity-algorithms)
  - [How Hypervectors Work](#how-hypervectors-work)
  - [Encoding Strategies](#encoding-strategies)
  - [Similarity Search](#similarity-search)
  - [Similarity Thresholds](#similarity-thresholds)
  - [How VSA Is Used in the Pipeline](#how-vsa-is-used-in-the-pipeline)
- [Extending with New Languages](#extending-with-new-languages)
  - [Step 1: Add the Language Variant](#step-1-add-the-language-variant)
  - [Step 2: Add the Language Lexicon](#step-2-add-the-language-lexicon)
  - [Step 3: Add Detection Support](#step-3-add-detection-support)
  - [Step 4: Add Equivalences](#step-4-add-equivalences)
  - [Step 5: Rebuild and Test](#step-5-rebuild-and-test)
  - [Checklist for New Languages](#checklist-for-new-languages)
- [CLI Reference](#cli-reference)
- [HTTP API Reference](#http-api-reference)
- [Integration Patterns](#integration-patterns)
  - [Python Integration](#python-integration)
  - [Eleutherios Mapping](#eleutherios-mapping)
- [Architecture Notes](#architecture-notes)
- [Relational Pattern Reference](#relational-pattern-reference)

---

## Quick Start

### Build

```bash
# Core binary (CLI pre-processor)
cargo build --release

# HTTP server (optional, for network integration)
cargo build --release --features server
```

### CLI Pipeline (JSONL)

Eleutherios pipes chunked text through stdin and reads structured output from stdout:

```bash
# Auto-detect language per chunk
cat chunks.jsonl | ./target/release/akh-medu preprocess --format jsonl > structured.jsonl

# Force a specific language
cat russian_chunks.jsonl | ./target/release/akh-medu preprocess --format jsonl --language ru > structured.jsonl
```

**Input format** (one JSON object per line):

```json
{"id": "doc1-p1", "text": "Protein folding is a fundamental process in molecular biology."}
{"id": "doc1-p2", "text": "Misfolded proteins are associated with neurodegenerative diseases."}
```

The `id` field is optional but recommended for traceability. The `language` field
is optional; omit it to auto-detect.

**Output format** (one JSON object per line):

```json
{
  "chunk_id": "doc1-p1",
  "source_language": "en",
  "detected_language_confidence": 0.80,
  "entities": [
    {
      "name": "protein folding",
      "entity_type": "CONCEPT",
      "canonical_name": "protein folding",
      "confidence": 0.83,
      "aliases": [],
      "source_language": "en"
    }
  ],
  "claims": [
    {
      "claim_text": "Protein folding is a fundamental process in molecular biology.",
      "claim_type": "FACTUAL",
      "confidence": 0.83,
      "subject": "protein folding",
      "predicate": "is-a",
      "object": "fundamental process in molecular biology",
      "source_language": "en"
    }
  ],
  "abs_trees": [...]
}
```

### CLI Pipeline (JSON batch)

For batch processing, use `--format json` with a JSON array on stdin:

```bash
echo '[
  {"id": "1", "text": "Gravity is a fundamental force of nature."},
  {"id": "2", "text": "Гравитация является фундаментальной силой природы."}
]' | ./target/release/akh-medu preprocess --format json
```

Returns:

```json
{
  "results": [...],
  "processing_time_ms": 0
}
```

### HTTP Server

For network-accessible integration (e.g., Eleutherios calling akh-medu over HTTP):

```bash
# Start server on port 8200
./target/release/akh-medu-server
```

**Endpoints summary:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Status, version, supported languages |
| `GET` | `/languages` | List languages with pattern counts |
| `POST` | `/preprocess` | Pre-process text chunks |
| `GET` | `/equivalences` | List all learned equivalences |
| `GET` | `/equivalences/stats` | Equivalence counts by source |
| `POST` | `/equivalences/learn` | Run learning strategies |
| `POST` | `/equivalences/import` | Import equivalences from JSON |

**POST /preprocess:**

```bash
curl -X POST http://localhost:8200/preprocess \
  -H 'Content-Type: application/json' \
  -d '{
    "chunks": [
      {"id": "1", "text": "The cell membrane contains phospholipids."},
      {"id": "2", "text": "Клеточная мембрана содержит фосфолипиды."},
      {"id": "3", "text": "La membrane cellulaire contient des phospholipides."}
    ]
  }'
```

---

## Running with Eleutherios Docker

This section covers running the full stack: Eleutherios Docker services, Ollama
for LLM inference, and akh-medu as the pre-processing layer.

### Prerequisites

| Component | Minimum Version | Purpose |
|-----------|----------------|---------|
| Docker | 24+ | Runs Neo4j, PostgreSQL, and the Eleutherios API |
| Docker Compose | v2 | Orchestrates the services |
| Ollama | 0.5+ | Hosts LLM models (Mistral Nemo 12B, nomic-embed-text) |
| Rust toolchain | 1.85+ | Builds akh-medu |
| RAM | 16 GB (32 GB recommended) | Mistral Nemo 12B alone needs ~8.4 GB |

### Step 1: Start Ollama on All Interfaces

Ollama must listen on `0.0.0.0` (not just `127.0.0.1`) so Docker containers
can reach it via `host.docker.internal`:

```bash
# Start Ollama listening on all interfaces
OLLAMA_HOST=0.0.0.0:11434 ollama serve &

# Pull required models
ollama pull nomic-embed-text      # Embeddings (274 MB)
ollama pull mistral-nemo:12b      # Extraction LLM (7.1 GB)

# Verify
curl -s http://localhost:11434/api/tags | python3 -c "
import sys, json
for m in json.load(sys.stdin).get('models', []):
    print(f'  {m[\"name\"]}')
"
```

**Common mistake**: If Ollama is started without `OLLAMA_HOST=0.0.0.0:11434`,
Docker containers will get "connection refused" when calling the LLM. You can
verify the listening address with `ss -tlnp | grep 11434` — it must show `*:11434`,
not `127.0.0.1:11434`.

### Step 2: Start Eleutherios Services

```bash
# Clone Eleutherios Docker
git clone https://github.com/Eleutherios-project/Eleutherios-docker.git
cd Eleutherios-docker

# Create data directories
mkdir -p data/inbox data/processed data/calibration_profiles

# Start services (Neo4j + PostgreSQL + API)
docker compose up -d

# First startup takes several minutes (demo data seeding).
# Watch progress:
docker compose logs -f api
```

The Eleutherios API (port 8001) runs a demo data import on first start
(`SEED_ON_FIRST_RUN=true`) with ~144K Cypher statements. This typically takes
5-15 minutes. Wait until the health check passes:

```bash
# Poll until healthy
until curl -sf http://localhost:8001/api/health/simple; do
    echo "Waiting for Eleutherios API..."
    sleep 10
done
echo "API is ready"
```

### Step 3: Build and Start akh-medu

```bash
cd /path/to/akh-medu

# Build CLI and HTTP server
cargo build --release
cargo build --release --features server

# Start the pre-processing server
./target/release/akh-medu-server &

# Verify
curl -s http://localhost:8200/health
```

### Step 4: Load Documents

Copy your PDF/EPUB corpus into the Eleutherios inbox:

```bash
# Copy files (do NOT symlink — Docker bind mounts can't follow host symlinks)
cp /path/to/your/corpus/*.pdf /path/to/Eleutherios-docker/data/inbox/

# Verify files are visible inside the container
curl -s http://localhost:8001/api/list-inbox-files | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f'{d[\"total_count\"]} files ({d[\"total_size_mb\"]:.0f} MB)')
for f in d['files'][:5]:
    print(f'  {f[\"filename\"]} ({f[\"size_mb\"]:.1f} MB)')
"
```

**Important**: Do not use symlinks into the Docker inbox directory. Docker bind
mounts expose the directory to the container, but symlinks pointing to paths
outside the mount will appear as broken links inside the container.

### Step 5: Run the Pipeline

```bash
# Start the Eleutherios load pipeline
JOB_ID=$(curl -s -X POST http://localhost:8001/api/load-pipeline \
  -H "Content-Type: application/json" \
  -d '{
    "type": "pdfs",
    "path": "/app/data/inbox",
    "selected_files": ["your-document.pdf"]
  }' | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")

echo "Job started: $JOB_ID"

# Monitor progress
watch -n 10 "curl -s http://localhost:8001/api/load-status/$JOB_ID | python3 -c \"
import sys,json
d=json.load(sys.stdin)
print(f'Status: {d.get(\"status\")} | Progress: {d.get(\"progress_percent\",0)}%')
s=d.get('stats',{})
print(f'Entities: {s.get(\"entities\",0)} | Claims: {s.get(\"claims\",0)}')
\""
```

### Step 6: Pre-Process with akh-medu

After Eleutherios completes Step 1 (chunking), retrieve the JSONL output and
run it through akh-medu for immediate structural extraction:

```bash
# Copy the JSONL chunks from the container
docker cp aegis-api:/tmp/aegis_imports/${JOB_ID}_jsonl/combined_chunks.jsonl /tmp/chunks.jsonl

# Pre-process through akh-medu (CLI)
cat /tmp/chunks.jsonl | ./target/release/akh-medu preprocess --format jsonl > /tmp/structured.jsonl

# Or via HTTP (batched, for production)
python3 -c "
import json, urllib.request

chunks = []
with open('/tmp/chunks.jsonl') as f:
    for line in f:
        data = json.loads(line)
        chunks.append({'id': data['metadata']['doc_id'], 'text': data['text']})

# Process in batches of 20
for i in range(0, len(chunks), 20):
    batch = chunks[i:i+20]
    payload = json.dumps({'chunks': batch}).encode()
    req = urllib.request.Request(
        'http://localhost:8200/preprocess',
        data=payload,
        headers={'Content-Type': 'application/json'}
    )
    resp = urllib.request.urlopen(req)
    result = json.loads(resp.read())
    for r in result['results']:
        print(json.dumps(r))
" > /tmp/structured.jsonl
```

### Benchmark: Real-World Performance

Tested with "Memories, Dreams, Reflections" by Carl Jung (1.8 MB PDF):

| Metric | akh-medu (grammar) | Eleutherios (Mistral Nemo 12B on CPU) |
|--------|-------------------|---------------------------------------|
| **Chunks processed** | 196 | 196 |
| **Processing time** | 0.8 seconds | Hours (CPU) / minutes (GPU) |
| **Throughput** | ~300 chunks/sec | ~0.3 chunks/sec (CPU) |
| **Entities extracted** | 2,943 | Requires LLM inference |
| **Claims extracted** | 1,474 | Requires LLM inference |
| **GPU required** | No | Strongly recommended |

akh-medu provides near-instant structural pre-extraction that complements the
deeper but slower LLM-based extraction. High-confidence akh-medu claims can be
ingested directly while the LLM pipeline runs.

### Memory Requirements

Running the full stack simultaneously:

| Component | Memory Usage |
|-----------|-------------|
| Neo4j | ~2-3 GB |
| PostgreSQL + pgvector | ~0.5 GB |
| Eleutherios API | ~1-2 GB |
| Ollama (Mistral Nemo 12B) | ~8.4 GB |
| akh-medu server | ~50 MB |
| **Total** | **~12-14 GB** |

If you hit "model requires more system memory" errors from Ollama, free memory
by dropping filesystem caches (`sync && echo 3 | sudo tee /proc/sys/vm/drop_caches`)
or by stopping unused services.

---

## Step-by-Step Tutorial

This section walks through a complete end-to-end workflow that you can replicate
with your own corpus.

### 1. Prepare Your Corpus

akh-medu expects text chunks as JSON objects. Each chunk has a `text` field and
an optional `id` and `language` field.

**Create a sample corpus file** (`chunks.jsonl`):

```bash
cat > /tmp/chunks.jsonl << 'EOF'
{"id": "intro-en-1", "text": "The mitochondria is a membrane-bound organelle found in eukaryotic cells."}
{"id": "intro-en-2", "text": "ATP synthesis depends on the electron transport chain."}
{"id": "intro-ru-1", "text": "Митохондрия является мембранным органоидом эукариотических клеток."}
{"id": "intro-fr-1", "text": "La mitochondrie est un organite présent dans les cellules eucaryotes."}
{"id": "intro-es-1", "text": "La mitocondria es un orgánulo presente en las células eucariotas."}
{"id": "intro-ar-1", "text": "الميتوكوندريا هي عضية موجودة في الخلايا حقيقية النواة."}
EOF
```

You can also convert existing documents. For PDF/EPUB corpora, use your preferred
text extraction tool first:

```bash
# Example: extract text from PDFs, then chunk
# (Use your own extraction tool — pdftotext, Apache Tika, etc.)
pdftotext /path/to/your/document.pdf - | \
  python3 -c "
import sys, json
text = sys.stdin.read()
# Simple paragraph-level chunking (replace with your own chunking strategy)
paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
for i, para in enumerate(paragraphs):
    print(json.dumps({'id': f'doc-p{i}', 'text': para}))
" > /tmp/chunks.jsonl
```

### 2. Pre-Process via CLI

```bash
# Build the binary
cargo build --release

# Run pre-processing with language auto-detection
cat /tmp/chunks.jsonl | ./target/release/akh-medu preprocess --format jsonl > /tmp/structured.jsonl

# Inspect the output
head -1 /tmp/structured.jsonl | python3 -m json.tool
```

**Expected output** (formatted for readability):

```json
{
  "chunk_id": "intro-en-1",
  "source_language": "en",
  "detected_language_confidence": 0.80,
  "entities": [
    {
      "name": "mitochondria",
      "entity_type": "CONCEPT",
      "canonical_name": "mitochondria",
      "confidence": 0.90,
      "aliases": [],
      "source_language": "en"
    },
    {
      "name": "membrane-bound organelle",
      "entity_type": "CONCEPT",
      "canonical_name": "membrane-bound organelle",
      "confidence": 0.90,
      "aliases": [],
      "source_language": "en"
    }
  ],
  "claims": [
    {
      "claim_text": "The mitochondria is a membrane-bound organelle found in eukaryotic cells.",
      "claim_type": "FACTUAL",
      "confidence": 0.90,
      "subject": "mitochondria",
      "predicate": "is-a",
      "object": "membrane-bound organelle",
      "source_language": "en"
    }
  ],
  "abs_trees": [...]
}
```

**Force a specific language** when auto-detection isn't reliable (short text,
mixed-script content, or unsupported language using English as fallback):

```bash
cat /tmp/german_chunks.jsonl | ./target/release/akh-medu preprocess --format jsonl --language en > /tmp/structured.jsonl
```

### 3. Pre-Process via HTTP

```bash
# Build and start the server
cargo build --release --features server
./target/release/akh-medu-server &

# Wait for startup
sleep 1

# Check health
curl -s http://localhost:8200/health | python3 -m json.tool

# Pre-process chunks
curl -s -X POST http://localhost:8200/preprocess \
  -H 'Content-Type: application/json' \
  -d '{
    "chunks": [
      {"id": "en-1", "text": "The cell membrane contains phospholipids."},
      {"id": "ru-1", "text": "Клеточная мембрана содержит фосфолипиды."}
    ]
  }' | python3 -m json.tool

# Check supported languages
curl -s http://localhost:8200/languages | python3 -m json.tool
```

### 4. Interpret the Output

Each output object maps directly to Eleutherios's data model:

| Output Field | Type | Purpose |
|-------------|------|---------|
| `chunk_id` | `string?` | Ties back to the input chunk |
| `source_language` | `string` | BCP 47 code (`en`, `ru`, `ar`, `fr`, `es`) |
| `detected_language_confidence` | `f32` | 0.0–1.0, how sure the detector is |
| `entities[].name` | `string` | Surface form of the entity |
| `entities[].entity_type` | `string` | `CONCEPT` or `PLACE` |
| `entities[].canonical_name` | `string` | Cross-language resolved name |
| `entities[].confidence` | `f32` | Extraction confidence |
| `entities[].aliases` | `string[]` | Known surface variants |
| `entities[].source_language` | `string` | Language the entity was found in |
| `claims[].claim_text` | `string` | Original sentence |
| `claims[].claim_type` | `string` | `FACTUAL`, `CAUSAL`, `SPATIAL`, etc. |
| `claims[].subject` | `string` | Subject entity label |
| `claims[].predicate` | `string` | Canonical predicate (`is-a`, `causes`, etc.) |
| `claims[].object` | `string` | Object entity label |
| `claims[].confidence` | `f32` | Extraction confidence |
| `abs_trees[]` | `AbsTree` | Raw abstract syntax trees (for advanced use) |

### 5. Feed into Eleutherios

Use the structured output to seed Eleutherios's extraction pipeline. The pre-extracted
entities and claims give the LLM a head start:

```python
import json

# Read akh-medu output
with open("/tmp/structured.jsonl") as f:
    for line in f:
        result = json.loads(line)

        # Use source_language to route to language-specific models
        lang = result["source_language"]

        # Pre-extracted entities → seed Neo4j / entity resolution
        for entity in result["entities"]:
            # entity["canonical_name"] is already cross-language resolved
            upsert_entity(entity["canonical_name"], entity["entity_type"])

        # Pre-extracted claims → skip LLM for high-confidence relations
        for claim in result["claims"]:
            if claim["confidence"] >= 0.85:
                # High confidence: ingest directly
                insert_triple(claim["subject"], claim["predicate"], claim["object"])
            else:
                # Lower confidence: send to LLM for validation
                queue_for_llm_validation(claim)
```

---

## What Meaning We Extract

### Claim Types

The pre-processor classifies every extracted relation into a **claim type** that
Eleutherios can use to route into the appropriate dimension of its extraction pipeline:

| Claim Type | Predicates | Example |
|------------|-----------|---------|
| **FACTUAL** | `is-a`, `has-a`, `contains`, `implements`, `defines` | "Gravity **is a** fundamental force" |
| **CAUSAL** | `causes` | "Deforestation **causes** soil erosion" |
| **SPATIAL** | `located-in` | "CERN **is located in** Geneva" |
| **RELATIONAL** | `similar-to` | "Graphene **is similar to** carbon nanotubes" |
| **STRUCTURAL** | `part-of`, `composed-of` | "The cortex **is part of** the brain" |
| **DEPENDENCY** | `depends-on` | "Photosynthesis **depends on** sunlight" |

### Entity Types

| Type | Inferred When |
|------|--------------|
| **CONCEPT** | Default for most entities |
| **PLACE** | Object of `located-in` predicate |

### Canonical Predicates

All languages map to the **same 9 canonical predicates**, ensuring Eleutherios
receives uniform relation labels regardless of source language:

| Canonical | Meaning |
|-----------|---------|
| `is-a` | Classification / type hierarchy |
| `has-a` | Possession / attribute |
| `contains` | Containment / composition |
| `located-in` | Spatial location |
| `causes` | Causation |
| `part-of` | Meronymy (part-whole) |
| `composed-of` | Material composition |
| `similar-to` | Similarity / analogy |
| `depends-on` | Dependency |

---

## Available Languages

| Language | Code | Detection | Patterns | Void Words | Notes |
|----------|------|-----------|----------|------------|-------|
| **English** | `en` | Latin script + word frequency | 21 | a, an, the | Default fallback |
| **Russian** | `ru` | Cyrillic script (>0.95 conf) | 13 | (none) | No articles in Russian |
| **Arabic** | `ar` | Arabic script (>0.95 conf) | 11 | ال | RTL handled correctly |
| **French** | `fr` | Latin + diacritics (é, ç) + markers | 16 | le, la, les, un, une, des, du, de, d', l' | Accent-insensitive matching |
| **Spanish** | `es` | Latin + diacritics (ñ) + markers + ¿¡ | 14 | el, la, los, las, un, una, unos, unas, del, de, al | Inverted punctuation stripped |
| **Auto** | `auto` | Script analysis + heuristics | (selected per chunk) | (selected per chunk) | Default mode |

### Language Detection

Detection uses a two-stage approach that requires **no external NLP models**:

**Stage 1 — Script analysis** (highest confidence):

| Script | Unicode Range | Detection | Confidence |
|--------|--------------|-----------|------------|
| Cyrillic | U+0400..U+052F | >50% of alphabetic chars | 0.70 + ratio×0.25 (max 0.95) |
| Arabic | U+0600..U+06FF, U+0750..U+077F, U+08A0..U+08FF | >50% of alphabetic chars | 0.70 + ratio×0.25 (max 0.95) |
| Latin | U+0041..U+024F | Proceeds to Stage 2 | — |

**Stage 2 — Latin disambiguation** (word frequency + diacritics):

Each Latin-script text is scored against word frequency markers:

| Language | Marker Words | Diacritical Boost |
|----------|-------------|-------------------|
| English | the, is, are, was, with, from, this, that, and, for, not, but, have, will, would, can, could, should, it, they, we, you, he, she (28) | — |
| French | le, la, les, des, est, dans, avec, une, sur, pour, pas, qui, que, sont, ont, fait, plus, mais, aussi, cette, nous, vous, ils, elles (24) | é, è, ê, ë, ç, à, ù, î, ô, œ → +2.0 |
| Spanish | el, los, las, está, tiene, por, para, pero, también, como, más, son, hay, ser, estar, muy, todo, puede, sobre, ese, esa, estos (26) | ñ, á, í, ó, ú, ü → +2.0; ¿ ¡ → +3.0 |

Winner is the language with the highest normalized score. Confidence ranges from 0.60–0.85.

### Mixed-Language Corpora

For documents that contain multiple languages (e.g., English text with Russian
quotations or Arabic transliterations), use **auto-detection per chunk** rather
than forcing a single language:

```bash
# Each chunk is detected independently — this is the default behavior
cat mixed_corpus.jsonl | ./target/release/akh-medu preprocess --format jsonl > structured.jsonl
```

You can also split mixed-language documents at the sentence level before chunking:

```python
# Pre-split mixed-language paragraphs into per-sentence chunks
import json

paragraph = "The experiment was conducted in Moscow. Результаты были опубликованы в журнале."

# Simple sentence-level splitting
sentences = [s.strip() for s in paragraph.replace('. ', '.\n').split('\n') if s.strip()]
for i, sentence in enumerate(sentences):
    print(json.dumps({"id": f"para1-s{i}", "text": sentence}))
```

akh-medu's internal `detect_per_sentence()` function handles this automatically when
processing through the grammar module.

---

## Cross-Language Entity Resolution

When the same entity appears in different languages, the resolver unifies them
under a canonical English label:

```
"Moscow"  (EN) -> canonical: "Moscow"
"Москва"  (RU) -> canonical: "Moscow", aliases: ["Москва"]
"Moscou"  (FR) -> canonical: "Moscow", aliases: ["Moscou"]
"موسكو"   (AR) -> canonical: "Moscow", aliases: ["موسكو"]
```

### Static Equivalence Table

The compiled-in static table covers ~120 entries across categories:

| Category | Examples |
|----------|---------|
| Countries | Russia/Россия/Russie, China/Китай/Chine, France/Франция/Francia |
| Cities | Moscow/Москва/Moscou, Beijing/Пекин/Pékin, Paris/Париж |
| Organizations | NATO/НАТО/OTAN, United Nations/ООН/ONU |
| Common terms | mammal/млекопитающее/mammifère, government/правительство/gouvernement |

### Dynamic Equivalence Learning

Beyond the static table, akh-medu can **discover new equivalences dynamically**
using three learning strategies. Discovered mappings persist across sessions via
the durable store (redb).

#### 4-Tier Resolution Order

When resolving an entity, the resolver checks in this order:

1. **Runtime aliases** — hot in-memory mappings added during the current session
2. **Learned equivalences** — persisted mappings discovered by learning strategies
3. **Static equivalence table** — ~120 hand-curated entries compiled into the binary
4. **Fallback** — return the surface form unchanged

Learned equivalences override the static table, allowing domain-specific corrections.

#### Strategy 1: KG Structural Fingerprints

Two entities in different languages that share identical relational patterns are
likely the same concept.

**How it works:**
1. For each unresolved entity `e`, collect its "relational fingerprint": the set
   of `(predicate, resolved_object)` tuples from the knowledge graph
2. For each already-resolved entity `c`, collect its fingerprint too
3. If `e` and `c` share >= 1 fingerprint tuple, propose `e -> c` with confidence
   based on the overlap ratio

**Example:** If the KG has `("собака", is-a, "млекопитающее")` and
`("Dog", is-a, "mammal")`, and "млекопитающее" already resolves to "mammal",
then "собака" structurally maps to "Dog".

#### Strategy 2: VSA Similarity

Hypervector encodings capture distributional similarity. For Latin-script
near-matches (programme/program, organisation/organization) and transliterated
names, VSA similarity catches what string matching misses. See
[VSA Similarity Algorithms](#vsa-similarity-algorithms) for a full explanation.

**How it works:**
1. For each unresolved entity, encode its label as a hypervector
2. Search item memory for the 5 nearest neighbors
3. For results above the similarity threshold (>= 0.65), check if the matched
   symbol resolves to a known canonical
4. If yes, propose the equivalence with confidence = similarity score

#### Strategy 3: Parallel Chunk Co-occurrence

When Eleutherios sends parallel translations of the same content, entities at
corresponding positions are likely equivalent.

**How it works:**
1. Group chunks by shared `chunk_id` prefix (e.g., `"doc1_en"` and `"doc1_ru"`
   share prefix `"doc1"`)
2. For each group with different languages, align entities by extraction order
3. For entities at the same index across languages, propose equivalence

**Chunk ID convention:** Use `{document_id}_{language_code}` format:
```json
{"id": "report-ch3_en", "text": "The experiment was conducted in Moscow.", "language": "en"}
{"id": "report-ch3_ru", "text": "Эксперимент был проведён в Москве.", "language": "ru"}
```

#### Strategy 4: Library Paragraph Context

When a shared content library has been populated (via `library add`), unresolved
entities can be matched against library paragraph embeddings.

**How it works:**
1. For each unresolved entity, encode its label as a hypervector
2. Search item memory for the 20 nearest neighbors, filtering to `para:*` symbols
3. For matching library paragraphs, walk KG triples to find connected entities
4. Skip structural labels (`para:`, `ch:`, `sec:`, numeric indices)
5. If a connected entity resolves to a known canonical, propose the equivalence
   with confidence `(similarity * 0.85).min(0.85)`

**Example:** If the library contains a paragraph about "Jungian psychology" that
connects to "Archetype", and an unresolved entity "архетип" has a similar
embedding, the strategy proposes `"архетип" → "Archetype"`.

#### Equivalence Sources

Each learned equivalence records how it was discovered:

| Source | Description |
|--------|-------------|
| `Static` | From the compiled-in equivalence table |
| `KgStructural` | Discovered by matching KG relational fingerprints |
| `VsaSimilarity` | Discovered by hypervector distributional similarity |
| `CoOccurrence` | Discovered from parallel chunk position correlation |
| `LibraryContext` | Discovered from shared library paragraph context |
| `Manual` | User-added via CLI or API import |

### Managing Equivalences via CLI

```bash
# List all learned equivalences
./target/release/akh-medu equivalences list

# Show counts by source (kg-structural, vsa-similarity, co-occurrence, manual)
./target/release/akh-medu equivalences stats

# Run all learning strategies on current engine state
./target/release/akh-medu equivalences learn

# Export to JSON for manual curation
./target/release/akh-medu equivalences export > /tmp/equivalences.json

# Import curated equivalences
./target/release/akh-medu equivalences import < /tmp/equivalences.json
```

### Managing Equivalences via HTTP

```bash
# List all learned equivalences
curl -s http://localhost:8200/equivalences | python3 -m json.tool

# Show statistics
curl -s http://localhost:8200/equivalences/stats | python3 -m json.tool
# => {"runtime_aliases": 0, "learned_total": 12, "kg_structural": 3, "vsa_similarity": 4, "co_occurrence": 2, "library_context": 0, "manual": 3}

# Trigger learning
curl -s -X POST http://localhost:8200/equivalences/learn | python3 -m json.tool
# => {"discovered": 5, "total_learned": 17}

# Import curated equivalences
curl -X POST http://localhost:8200/equivalences/import \
  -H 'Content-Type: application/json' \
  -d '[
    {"canonical": "mitochondria", "surface": "митохондрия", "source_language": "ru", "confidence": 0.95, "source": "Manual"},
    {"canonical": "cell membrane", "surface": "клеточная мембрана", "source_language": "ru", "confidence": 0.95, "source": "Manual"}
  ]'
# => {"imported": 2}
```

### Seeding Domain-Specific Equivalences

For specialized corpora (medical terminology, legal terms, engineering jargon),
seed the equivalence table with domain-specific terms before processing:

**1. Create an equivalence file** (`/tmp/domain-terms.json`):

```json
[
  {"canonical": "mitochondria", "surface": "митохондрия", "source_language": "ru", "confidence": 1.0, "source": "Manual"},
  {"canonical": "mitochondria", "surface": "mitochondrie", "source_language": "fr", "confidence": 1.0, "source": "Manual"},
  {"canonical": "mitochondria", "surface": "mitocondria", "source_language": "es", "confidence": 1.0, "source": "Manual"},
  {"canonical": "photosynthesis", "surface": "фотосинтез", "source_language": "ru", "confidence": 1.0, "source": "Manual"},
  {"canonical": "photosynthesis", "surface": "photosynthèse", "source_language": "fr", "confidence": 1.0, "source": "Manual"},
  {"canonical": "photosynthesis", "surface": "fotosíntesis", "source_language": "es", "confidence": 1.0, "source": "Manual"},
  {"canonical": "enzyme", "surface": "фермент", "source_language": "ru", "confidence": 1.0, "source": "Manual"},
  {"canonical": "enzyme", "surface": "إنزيم", "source_language": "ar", "confidence": 1.0, "source": "Manual"}
]
```

**2. Import before processing:**

```bash
# Via CLI
./target/release/akh-medu equivalences import < /tmp/domain-terms.json

# Via HTTP
curl -X POST http://localhost:8200/equivalences/import \
  -H 'Content-Type: application/json' \
  -d @/tmp/domain-terms.json
```

**3. Process your corpus — imported terms will be resolved:**

```bash
echo '{"text": "Митохондрия является органоидом клетки."}' | \
  ./target/release/akh-medu preprocess --format jsonl
# entities[0].canonical_name will be "mitochondria" instead of "митохондрия"
```

**4. Export/import workflow for iterative curation:**

```bash
# Process a batch, let learning discover new equivalences
cat /tmp/corpus.jsonl | ./target/release/akh-medu preprocess --format jsonl > /dev/null
./target/release/akh-medu equivalences learn

# Export for review
./target/release/akh-medu equivalences export > /tmp/review.json

# Edit review.json manually (fix mistakes, add missing terms)
# Then re-import
./target/release/akh-medu equivalences import < /tmp/review.json
```

---

## VSA Similarity Algorithms

akh-medu uses **Vector Symbolic Architecture (VSA)** — also known as Hyperdimensional
Computing — to encode symbols, detect similar entities, and support fuzzy matching
across the pipeline.

### How Hypervectors Work

A hypervector is a **high-dimensional binary vector** (default: 10,000 dimensions)
where each dimension is a single bit interpreted as +1 or -1 (bipolar encoding).

Key properties:

| Property | Description |
|----------|-------------|
| **Dimension** | 10,000 bits by default (`Dimension::DEFAULT`). 1,000 bits for tests (`Dimension::TEST`). |
| **Encoding** | Bipolar: each bit is +1 or -1. Stored as packed bytes. |
| **Random vectors** | Two random hypervectors have ~0.5 similarity (uncorrelated). |
| **Deterministic** | The same symbol ID always produces the same hypervector. |

### Encoding Strategies

akh-medu uses four encoding functions, each serving a different purpose:

#### 1. Symbol Encoding (`encode_symbol`)

Maps a `SymbolId` to a deterministic hypervector using seeded random generation.
The same ID always produces the same vector:

```
SymbolId(42) → deterministic random HyperVec (seeded with 42)
SymbolId(43) → different deterministic random HyperVec (seeded with 43)
```

This is the base encoding — every symbol in the system gets one.

#### 2. Token Encoding (`encode_token`)

Maps a text string to a hypervector by hashing it to a synthetic SymbolId:

```
"dog"  → hash("dog")  → synthetic SymbolId → deterministic HyperVec
"dogs" → hash("dogs") → different SymbolId  → different HyperVec
```

Token encoding is deterministic: the same word always produces the same vector.
However, "dog" and "dogs" produce **unrelated** vectors (no morphological awareness).

#### 3. Label Encoding (`encode_label`)

For multi-word labels, encodes each word separately and **bundles** them:

```
"big red dog" → bundle(encode_token("big"), encode_token("red"), encode_token("dog"))
```

The resulting vector is **similar to each component** (similarity > 0.55) but
identical to none. This captures set-like semantics: "big red dog" is similar
to anything containing "big", "red", or "dog".

Single-word labels fall through to `encode_token` directly.

#### 4. Role-Filler Encoding (`encode_role_filler`)

Encodes structured knowledge by **binding** a role vector with a filler vector:

```
bind(encode_symbol(color_id), encode_symbol(blue_id))
  → "the color is blue" as a single vector
```

Binding (XOR) produces a vector that is **dissimilar** to both inputs but can
be decoded: `unbind(bound, color_id) ≈ blue_id`.

#### 5. Sequence Encoding (`encode_sequence`)

Captures order information using **permutation**:

```
[A, B, C] → bundle(permute(A, 2), permute(B, 1), C)
```

Each element is shifted by its distance from the end, preserving positional
information. `[A, B, C]` produces a different vector than `[C, B, A]`.

### Similarity Search

Item Memory provides fast **approximate nearest-neighbor (ANN)** search using
the HNSW algorithm with Hamming distance:

```
Query: encode_token("programme")
Search: item_memory.search(&query_vec, k=5)
Results: [
  { symbol: "program",   similarity: 0.72 },  // near-match
  { symbol: "procedure", similarity: 0.53 },  // unrelated
  { symbol: "project",   similarity: 0.51 },  // unrelated
]
```

The search is **sub-linear** — HNSW provides O(log n) search time even with
millions of vectors.

**How ANN search works internally:**

1. The query vector is encoded as packed `u32` words for HNSW compatibility
2. HNSW navigates its layered graph using Hamming distance (bitwise XOR + popcount)
3. Raw Hamming distances are converted to similarity: `similarity = 1.0 - (hamming / total_bits)`
4. Results are sorted by descending similarity

### Similarity Thresholds

| Threshold | Used For | Meaning |
|-----------|----------|---------|
| ~0.50 | Random baseline | Two unrelated vectors |
| >= 0.60 | Fuzzy token resolution | Lexer resolves unknown tokens to known symbols |
| >= 0.65 | VSA equivalence learning | Dynamic equivalence strategy 2 threshold |
| >= 0.72 | High-confidence match | Near-certain the same entity (spelling variants) |
| 1.00 | Identity | Exact same vector |

### How VSA Is Used in the Pipeline

VSA operates at three points in the akh-medu pipeline:

**1. Lexer — Fuzzy Token Resolution**

When the lexer encounters an unknown word, it encodes it as a hypervector and
searches item memory for similar known symbols:

```
Input:  "programm" (misspelling)
Lookup: registry.lookup("programm") → None
Fuzzy:  item_memory.search(encode_token("programm"), k=3)
Match:  "program" with similarity 0.68 → Resolution::Fuzzy
```

The threshold is 0.60 (`DEFAULT_FUZZY_THRESHOLD`). Below this, the token stays
`Unresolved`.

**2. Dynamic Equivalence Learning — Strategy 2**

When `learn_equivalences()` is called, the VSA strategy encodes unresolved
entity labels and searches for similar resolved entities:

```
Unresolved: "organisation" (British spelling)
Search:     5 nearest neighbors in item memory
Match:      "organization" (American spelling), similarity 0.71
Result:     LearnedEquivalence { surface: "organisation", canonical: "organization", confidence: 0.71 }
```

The threshold is 0.65 (higher than lexer fuzzy matching for cross-lingual safety).

**3. Knowledge Graph — Similarity Queries**

The engine exposes similarity search for ad-hoc queries:

```bash
# Find symbols similar to a given symbol
# (Used internally by the agent module and available via Engine API)
engine.search_similar_to(symbol_id, top_k=10)
```

---

## Extending with New Languages

Adding a new language requires changes in **three files** and takes ~30 minutes.
Here is the complete procedure using **German** as an example.

### Step 1: Add the Language Variant

**File:** `src/grammar/lexer.rs`

Add the new variant to the `Language` enum:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum Language {
    English,
    Russian,
    Arabic,
    French,
    Spanish,
    German,       // <-- add here
    #[default]
    Auto,
}
```

Update the three methods on `Language`:

```rust
impl Language {
    pub fn bcp47(&self) -> &'static str {
        match self {
            // ...existing arms...
            Language::German => "de",
        }
    }

    pub fn from_code(code: &str) -> Option<Self> {
        match code.to_lowercase().as_str() {
            // ...existing arms...
            "de" | "german" => Some(Language::German),
            _ => None,
        }
    }
}

impl std::fmt::Display for Language {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            // ...existing arms...
            Language::German => write!(f, "German"),
        }
    }
}
```

### Step 2: Add the Language Lexicon

**File:** `src/grammar/lexer.rs`

A lexicon defines **five components** for each language:

| Component | Purpose | Example (German) |
|-----------|---------|-----------------|
| **Void words** | Semantically empty articles/determiners to strip | der, die, das, ein, eine |
| **Relational patterns** | Multi-word phrases that map to canonical predicates | "ist ein" → `is-a` |
| **Question words** | Trigger query parsing mode | was, wer, wo, wann |
| **Goal verbs** | Identify goal-setting input | finden, entdecken, erforschen |
| **Commands** | Special control patterns | help, status |

Add an arm to `Lexicon::for_language()` and create the lexicon constructor:

```rust
impl Lexicon {
    pub fn for_language(lang: Language) -> Self {
        match lang {
            // ...existing arms...
            Language::German => Self::default_german(),
        }
    }

    /// Build the German lexicon.
    pub fn default_german() -> Self {
        let void_words = vec![
            "der".into(), "die".into(), "das".into(),
            "ein".into(), "eine".into(), "eines".into(),
            "dem".into(), "den".into(), "des".into(),
        ];

        // Map German surface forms to canonical predicates.
        // IMPORTANT: Sort longest patterns first for greedy matching.
        let relational_patterns = vec![
            // 4+ word patterns first
            rel("befindet sich in", "located-in", 0.90),
            rel("ist ähnlich wie", "similar-to", 0.85),
            rel("ist zusammengesetzt aus", "composed-of", 0.85),
            rel("ist Teil von", "part-of", 0.90),
            rel("hängt ab von", "depends-on", 0.85),
            // 2-word patterns
            rel("ist ein", "is-a", 0.90),
            rel("ist eine", "is-a", 0.90),
            rel("hat ein", "has-a", 0.85),
            rel("hat eine", "has-a", 0.85),
            // 1-word patterns last
            rel("enthält", "contains", 0.85),
            rel("verursacht", "causes", 0.85),
            rel("ist", "is-a", 0.80),
            rel("hat", "has-a", 0.80),
        ];

        let question_words = vec![
            "was".into(), "wer".into(), "wo".into(), "wann".into(),
            "wie".into(), "warum".into(), "welcher".into(), "welche".into(),
        ];

        let goal_verbs = vec![
            "finden".into(), "entdecken".into(), "erforschen".into(),
            "analysieren".into(), "bestimmen".into(), "identifizieren".into(),
        ];

        // Commands stay English (CLI is English)
        let commands = vec![
            ("help".into(), CommandKind::Help),
            ("?".into(), CommandKind::Help),
            ("status".into(), CommandKind::ShowStatus),
        ];

        Self { void_words, relational_patterns, question_words, goal_verbs, commands }
    }
}
```

**Writing relational patterns — guidelines:**

- Patterns are matched **greedily, longest first**. Always sort multi-word patterns
  before shorter ones.
- Map every pattern to one of the 9 canonical predicates: `is-a`, `has-a`,
  `contains`, `located-in`, `causes`, `part-of`, `composed-of`, `similar-to`,
  `depends-on`.
- Assign confidence scores between 0.80 and 0.90:
  - 0.90 for unambiguous patterns ("ist ein" is always classification)
  - 0.85 for patterns with occasional false positives
  - 0.80 for very short/ambiguous patterns (bare "ist" could be copula or identity)
- Include accent-stripped variants for languages with diacritics (the existing French
  and Spanish lexicons do this).

### Step 3: Add Detection Support

**File:** `src/grammar/detect.rs`

This step makes `--language auto` work for the new language.

**For non-Latin scripts** (e.g., Chinese, Japanese, Korean, Hindi), add a
codepoint range check in the `detect_language()` function:

```rust
// In the character-counting loop:
match c {
    // CJK Unified Ideographs
    '\u{4E00}'..='\u{9FFF}' => { cjk += 1; }
    // Devanagari
    '\u{0900}'..='\u{097F}' => { devanagari += 1; }
    // ...
}

// After the loop, add a block like the Cyrillic/Arabic ones:
if cjk_ratio > 0.5 {
    return DetectionResult {
        language: Language::Chinese,
        confidence: (0.70 + cjk_ratio * 0.25).min(0.95),
    };
}
```

**For Latin-script languages** (like German), add word frequency markers to
`detect_latin_language()`:

```rust
const GERMAN_MARKERS: &[&str] = &[
    "der", "die", "das", "ist", "und", "ein", "eine", "nicht",
    "mit", "auf", "für", "von", "sich", "den", "dem", "auch",
    "werden", "haben", "sind", "wird", "kann", "nach", "über",
];
```

Then add a `german_score` accumulator alongside `english_score`, `french_score`,
`spanish_score`, and include German-specific diacritics (`ä`, `ö`, `ü`, `ß`):

```rust
let has_german_diacritics = lower.contains('ä')
    || lower.contains('ö')
    || lower.contains('ü')
    || lower.contains('ß');

if has_german_diacritics {
    german_score += 2.0;
}
```

Finally, include German in the winner selection:

```rust
let max_score = en_norm.max(fr_norm).max(es_norm).max(de_norm);
// ... pick the winner with the highest score
```

### Step 4: Add Equivalences

**File:** `src/grammar/equivalences.rs`

Add the new language's surface forms to existing entries:

```rust
Equivalence { canonical: "Germany", aliases: &["Allemagne", "Alemania", "Германия", "ألمانيا", "Deutschland"] },
//                                                                                              ^^^^^^^^^^ add
```

And add new entries for language-specific terms:

```rust
Equivalence { canonical: "psyche", aliases: &["Psyche", "психика", "نفس", "psyché", "psique"] },
```

You can also import equivalences at runtime instead of modifying the source:

```bash
echo '[
  {"canonical": "Germany", "surface": "Deutschland", "source_language": "de", "confidence": 1.0, "source": "Manual"}
]' | ./target/release/akh-medu equivalences import
```

### Step 5: Rebuild and Test

```bash
# Run tests
cargo test --lib

# Build release binary
cargo build --release

# Test with sample text (explicit language)
echo '{"text":"Der Archetyp ist ein universelles Muster."}' \
  | ./target/release/akh-medu preprocess --format jsonl --language de

# Test auto-detection (if Step 3 was implemented)
echo '{"text":"Die Zelle enthält Mitochondrien und andere Organellen."}' \
  | ./target/release/akh-medu preprocess --format jsonl
```

### Checklist for New Languages

- [ ] Add variant to `Language` enum in `lexer.rs`
- [ ] Update `bcp47()`, `from_code()`, `Display` on `Language`
- [ ] Add arm to `Lexicon::for_language()`
- [ ] Create `Lexicon::default_LANG()` with:
  - [ ] Void words (articles, determiners)
  - [ ] Relational patterns (sorted longest-first, mapping to canonical predicates)
  - [ ] Question words
  - [ ] Goal verbs
  - [ ] Commands (usually keep English)
- [ ] (Optional) Add detection markers in `detect.rs` for `Language::Auto` support
- [ ] (Optional) Add cross-lingual entries in `equivalences.rs`
- [ ] Run `cargo test --lib` (must pass, zero warnings)
- [ ] Test with sample text: `echo '{"text":"..."}' | akh-medu preprocess --format jsonl --language XX`

---

## CLI Reference

### Global Options

| Option | Description | Default |
|--------|-------------|---------|
| `--data-dir <PATH>` | Data directory for persistent storage | None (memory-only) |
| `--dimension <N>` | Hypervector dimension | 10000 |
| `--language <CODE>` | Default language for parsing | auto |

### Commands

#### `init`

Initialize a new akh-medu data directory:

```bash
./target/release/akh-medu --data-dir /tmp/akh-data init
```

#### `preprocess`

Pre-process text chunks from stdin:

```bash
# JSONL mode (streaming, one object per line)
cat chunks.jsonl | ./target/release/akh-medu preprocess --format jsonl

# JSON mode (batch, array on stdin)
cat chunks.json | ./target/release/akh-medu preprocess --format json

# With explicit language
cat chunks.jsonl | ./target/release/akh-medu preprocess --format jsonl --language ru
```

| Option | Description | Default |
|--------|-------------|---------|
| `--format <jsonl\|json>` | Input/output format | jsonl |
| `--language <CODE>` | Override language detection | auto |
| `--library-context` | Enrich entities with shared library context | off |

#### `ingest`

Ingest structured data into the knowledge graph:

```bash
# JSON triples
./target/release/akh-medu --data-dir /tmp/akh-data ingest --file /path/to/triples.json

# CSV (subject, predicate, object format)
./target/release/akh-medu --data-dir /tmp/akh-data ingest --file /path/to/data.csv --format csv --csv-format spo

# CSV (entity format: column headers are predicates)
./target/release/akh-medu --data-dir /tmp/akh-data ingest --file /path/to/data.csv --format csv --csv-format entity

# Plain text
./target/release/akh-medu --data-dir /tmp/akh-data ingest --file /path/to/text.txt --format text --max-sentences 100
```

#### `equivalences`

Manage cross-lingual entity equivalences:

```bash
./target/release/akh-medu equivalences list      # Show all learned equivalences
./target/release/akh-medu equivalences stats     # Counts by source
./target/release/akh-medu equivalences learn     # Run learning strategies
./target/release/akh-medu equivalences export    # Export to JSON (stdout)
./target/release/akh-medu equivalences import    # Import from JSON (stdin)
```

#### `grammar`

Grammar system commands:

```bash
# List available grammar archetypes
./target/release/akh-medu grammar list

# Parse prose to abstract syntax
./target/release/akh-medu grammar parse "Dogs are mammals"

# Parse and ingest into knowledge graph
./target/release/akh-medu --data-dir /tmp/akh-data grammar parse "Dogs are mammals" --ingest

# Linearize a triple (generate prose from structured data)
./target/release/akh-medu grammar linearize --subject Dog --predicate is-a --object mammal

# Compare a triple to knowledge graph
./target/release/akh-medu --data-dir /tmp/akh-data grammar compare --subject Dog --predicate is-a --object mammal

# Load a custom TOML grammar
./target/release/akh-medu grammar load --file /path/to/grammar.toml

# Render an entity's knowledge graph neighborhood
./target/release/akh-medu --data-dir /tmp/akh-data grammar render --entity Dog
```

---

## HTTP API Reference

**Server:** Listens on `0.0.0.0:8200`

**Build/Run:**
```bash
cargo build --release --features server
./target/release/akh-medu-server
```

**Environment:**
- `RUST_LOG` — Logging level (default: `info,egg=warn,hnsw_rs=warn`)

### `GET /health`

Health check.

**Response:**
```json
{
  "status": "ok",
  "version": "0.1.0",
  "supported_languages": ["en", "ru", "ar", "fr", "es", "auto"]
}
```

### `GET /languages`

List supported languages with pattern counts.

**Response:**
```json
{
  "languages": [
    {"code": "en", "name": "English", "pattern_count": 21},
    {"code": "ru", "name": "Russian", "pattern_count": 13},
    {"code": "ar", "name": "Arabic", "pattern_count": 11},
    {"code": "fr", "name": "French", "pattern_count": 16},
    {"code": "es", "name": "Spanish", "pattern_count": 14}
  ]
}
```

### `POST /preprocess`

Pre-process text chunks.

**Request:**
```json
{
  "chunks": [
    {"id": "optional-id", "text": "Text to process.", "language": "en"}
  ]
}
```

Fields: `id` and `language` are optional. Omit `language` for auto-detection.

**Response:**
```json
{
  "results": [
    {
      "chunk_id": "optional-id",
      "source_language": "en",
      "detected_language_confidence": 0.80,
      "entities": [...],
      "claims": [...],
      "abs_trees": [...]
    }
  ],
  "processing_time_ms": 0
}
```

### `GET /equivalences`

List all learned equivalences.

**Response:**
```json
[
  {
    "canonical": "Moscow",
    "surface": "москва",
    "source_language": "ru",
    "confidence": 0.95,
    "source": "Manual"
  }
]
```

### `GET /equivalences/stats`

Equivalence statistics by source.

**Response:**
```json
{
  "runtime_aliases": 0,
  "learned_total": 12,
  "kg_structural": 3,
  "vsa_similarity": 4,
  "co_occurrence": 2,
  "manual": 3
}
```

### `POST /equivalences/learn`

Trigger all learning strategies.

**Response:**
```json
{
  "discovered": 5,
  "total_learned": 17
}
```

### `POST /equivalences/import`

Bulk import equivalences.

**Request:** Array of `LearnedEquivalence` objects (see `GET /equivalences` for format).

**Response:**
```json
{
  "imported": 3
}
```

---

## Integration Patterns

### Python Integration

Three integration approaches, from simplest to most performant:

#### Subprocess (JSON batch)

Best for batch processing where latency per call isn't critical:

```python
import subprocess
import json
import os

def preprocess_chunks(chunks: list[dict]) -> list[dict]:
    """Send chunks through akh-medu for multilingual pre-processing."""
    input_json = json.dumps(chunks)
    result = subprocess.run(
        ["./target/release/akh-medu", "preprocess", "--format", "json"],
        input=input_json,
        capture_output=True,
        text=True,
        env={**os.environ, "RUST_LOG": "error"},
    )
    if result.returncode != 0:
        raise RuntimeError(f"akh-medu failed: {result.stderr}")
    response = json.loads(result.stdout)
    return response["results"]

# Usage
results = preprocess_chunks([
    {"id": "1", "text": "The cell membrane contains phospholipids."},
    {"id": "2", "text": "Клеточная мембрана содержит фосфолипиды."},
])
```

#### Subprocess (JSONL streaming)

Best for large corpora where you want to process chunks as they arrive:

```python
import subprocess
import json
import os

def preprocess_stream(chunks):
    """Stream chunks through akh-medu one at a time."""
    proc = subprocess.Popen(
        ["./target/release/akh-medu", "preprocess", "--format", "jsonl"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        env={**os.environ, "RUST_LOG": "error"},
    )
    for chunk in chunks:
        proc.stdin.write(json.dumps(chunk) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        if line:
            yield json.loads(line)
    proc.stdin.close()
    proc.wait()

# Usage
for result in preprocess_stream(chunks_iterator):
    process_result(result)
```

#### HTTP Client

Best for long-running services where the akh-medu server stays up:

```python
import requests

AKH_MEDU_URL = "http://localhost:8200"

def preprocess_http(chunks: list[dict]) -> list[dict]:
    """Call the akh-medu HTTP server."""
    resp = requests.post(
        f"{AKH_MEDU_URL}/preprocess",
        json={"chunks": chunks},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["results"]

def learn_equivalences() -> dict:
    """Trigger equivalence learning."""
    resp = requests.post(f"{AKH_MEDU_URL}/equivalences/learn", timeout=60)
    resp.raise_for_status()
    return resp.json()

def import_equivalences(equivs: list[dict]) -> int:
    """Import domain-specific equivalences."""
    resp = requests.post(
        f"{AKH_MEDU_URL}/equivalences/import",
        json=equivs,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["imported"]

# Usage
results = preprocess_http([
    {"id": "1", "text": "CERN is located in Geneva."},
    {"id": "2", "text": "Le CERN se trouve dans Genève."},
])
```

### Eleutherios Mapping

How akh-medu output maps to Eleutherios concepts:

| akh-medu Output | Eleutherios Use |
|----------------|-----------------|
| `source_language` | Route to language-specific extraction models |
| `entities[].canonical_name` | Seed entity for Neo4j lookup/creation |
| `entities[].entity_type` | Map to Eleutherios entity taxonomy |
| `claims[].predicate` | Pre-classified relation (skip LLM for simple facts) |
| `claims[].claim_type` | Route to the appropriate extraction dimension |
| `claims[].confidence` | Weight against LLM confidence for ensemble scoring |
| `abs_trees[]` | Raw parse trees for custom post-processing |

---

## Architecture Notes

### Performance

- Grammar parsing runs in **< 1ms per chunk** — 196-chunk batch in 0.8s (~300 chunks/sec)
- Batches of 20 chunks process in ~55ms (measured with HTTP endpoint)
- No external NLP dependencies, no model loading, no GPU required
- The HTTP server handles concurrent requests via `tokio` with a `RwLock<Engine>`
- Memory footprint: ~50MB for the engine with default 10,000-dimension hypervectors
- HNSW ANN search: O(log n) for similarity queries

### What the Pre-Processor Does NOT Do

- **Coreference resolution**: "He studied in Zurich" — "He" is not resolved
- **Complex clause parsing**: Subordinate clauses, relative clauses, passives
- **Morphological analysis**: No lemmatization (Russian "психики" stays inflected)
- **Named Entity Recognition**: Entity types are inferred from predicate context only

These are deliberate scope limits. Eleutherios's LLM pipeline handles them in its
enrichment stage. The pre-processor's job is to give it a clean head start with
the structural relations that grammar can catch deterministically.

### The `abs_trees` Field

The output includes full `AbsTree` abstract syntax trees for consumers that want
the raw parse. This is useful for:

- Debugging parse quality
- Custom post-processing beyond the entity/claim extraction
- Feeding back into akh-medu's knowledge graph via `Engine::commit_abs_tree()`

### Persistence

When started with `--data-dir`, the engine persists:
- Symbol registry (all known symbols and their metadata)
- Knowledge graph triples
- Learned equivalences (via `equiv:` prefix in redb)
- Agent session state (working memory, cycle count)

Data is stored in a 3-tier architecture:
- **Hot tier**: In-memory `DashMap` for fast concurrent access
- **Warm tier**: Memory-mapped files for large read-heavy data
- **Durable tier**: redb (ACID transactions) for data that must survive restarts

---

## Relational Pattern Reference

### English (21 patterns)

| Pattern | Canonical | Confidence | Example |
|---------|-----------|------------|---------|
| "is similar to" | `similar-to` | 0.85 | Graphene is similar to carbon nanotubes |
| "is located in" | `located-in` | 0.90 | CERN is located in Geneva |
| "is composed of" | `composed-of` | 0.85 | Water is composed of hydrogen and oxygen |
| "is part of" | `part-of` | 0.90 | The cortex is part of the brain |
| "is made of" | `composed-of` | 0.85 | Steel is made of iron and carbon |
| "depends on" | `depends-on` | 0.85 | Photosynthesis depends on sunlight |
| "belongs to" | `part-of` | 0.85 | This enzyme belongs to the kinase family |
| "is a" / "is an" | `is-a` | 0.90 | DNA is a nucleic acid |
| "are a" / "are an" | `is-a` | 0.85 | Mitochondria are a type of organelle |
| "has a" / "has an" | `has-a` | 0.85 | The cell has a nucleus |
| "have a" | `has-a` | 0.85 | Eukaryotes have a membrane-bound nucleus |
| "are" | `is-a` | 0.85 | Proteins are macromolecules |
| "has" / "have" | `has-a` | 0.85 | Enzymes have active sites |
| "contains" | `contains` | 0.85 | The nucleus contains chromosomes |
| "causes" | `causes` | 0.85 | Radiation causes DNA damage |
| "implements" | `implements` | 0.85 | (code domain) |
| "defines" | `defines` | 0.85 | (code domain) |

### Russian (13 patterns)

| Pattern | Canonical | Example |
|---------|-----------|---------|
| "является частью" | `part-of` | Кора является частью головного мозга |
| "находится в" | `located-in` | Институт находится в Женеве |
| "состоит из" | `composed-of` | Вода состоит из водорода и кислорода |
| "зависит от" | `depends-on` | Фотосинтез зависит от солнечного света |
| "похож на" | `similar-to` | Графен похож на углеродные нанотрубки |
| "содержит в себе" | `contains` | Ядро содержит в себе хромосомы |
| "является" | `is-a` | ДНК является нуклеиновой кислотой |
| "имеет" | `has-a` | Клетка имеет ядро |
| "содержит" | `contains` | Ядро содержит хромосомы |
| "вызывает" | `causes` | Радиация вызывает повреждение ДНК |
| "определяет" | `defines` | (определения) |
| "реализует" | `implements` | (код) |
| "это" | `is-a` | Митохондрия это органоид клетки |

### Arabic (11 patterns)

| Pattern | Canonical | Example |
|---------|-----------|---------|
| "يحتوي على" | `contains` | النواة يحتوي على الكروموسومات |
| "يقع في" | `located-in` | المعهد يقع في جنيف |
| "جزء من" | `part-of` | القشرة جزء من الدماغ |
| "يتكون من" | `composed-of` | الماء يتكون من الهيدروجين والأكسجين |
| "يعتمد على" | `depends-on` | التمثيل الضوئي يعتمد على ضوء الشمس |
| "هو" / "هي" | `is-a` | الحمض النووي هو حمض نووي |
| "لديه" / "لديها" | `has-a` | الخلية لديها نواة |
| "يسبب" | `causes` | الإشعاع يسبب تلف الحمض النووي |
| "يشبه" | `similar-to` | الجرافين يشبه أنابيب الكربون النانوية |

### French (16 patterns)

| Pattern | Canonical | Example |
|---------|-----------|---------|
| "est similaire à" / "est similaire a" | `similar-to` | Le graphène est similaire aux nanotubes de carbone |
| "se trouve dans" | `located-in` | Le CERN se trouve dans Genève |
| "est composé de" / "est compose de" | `composed-of` | L'eau est composée d'hydrogène et d'oxygène |
| "fait partie de" | `part-of` | Le cortex fait partie du cerveau |
| "dépend de" / "depend de" | `depends-on` | La photosynthèse dépend de la lumière du soleil |
| "est un" / "est une" | `is-a` | L'ADN est un acide nucléique |
| "a un" / "a une" | `has-a` | La cellule a un noyau |
| "contient" | `contains` | Le noyau contient des chromosomes |
| "cause" | `causes` | Le rayonnement cause des dommages à l'ADN |
| "définit" / "definit" | `defines` | (définitions) |

### Spanish (14 patterns)

| Pattern | Canonical | Example |
|---------|-----------|---------|
| "es similar a" | `similar-to` | El grafeno es similar a los nanotubos de carbono |
| "se encuentra en" | `located-in` | El CERN se encuentra en Ginebra |
| "está compuesto de" / "esta compuesto de" | `composed-of` | El agua está compuesta de hidrógeno y oxígeno |
| "es parte de" | `part-of` | El córtex es parte del cerebro |
| "depende de" | `depends-on` | La fotosíntesis depende de la luz solar |
| "es un" / "es una" | `is-a` | El ADN es un ácido nucleico |
| "tiene un" / "tiene una" | `has-a` | La célula tiene un núcleo |
| "contiene" | `contains` | El núcleo contiene cromosomas |
| "causa" | `causes` | La radiación causa daño al ADN |
| "tiene" | `has-a` | Los eucariotas tienen núcleo |
| "define" | `defines` | (definiciones) |
