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
noise, particularly for multilingual corpora like the Collected Works of Carl Jung
(44 volumes, primarily English with one German text and domain terminology drawn
from Latin, Greek, and Sanskrit).

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
{"id": "vol9-ch3-p42", "text": "The archetype is a universal pattern of the collective unconscious."}
{"id": "vol9-ch3-p43", "text": "The anima is part of the psyche and the shadow contains repressed elements."}
```

The `id` field is optional but recommended for traceability. The `language` field
is optional; omit it to auto-detect.

**Output format** (one JSON object per line):

```json
{
  "chunk_id": "vol9-ch3-p42",
  "source_language": "en",
  "detected_language_confidence": 0.80,
  "entities": [
    {
      "name": "archetype",
      "entity_type": "CONCEPT",
      "canonical_name": "archetype",
      "confidence": 0.83,
      "aliases": [],
      "source_language": "en"
    }
  ],
  "claims": [
    {
      "claim_text": "The archetype is a universal pattern of the collective unconscious.",
      "claim_type": "FACTUAL",
      "confidence": 0.83,
      "subject": "archetype",
      "predicate": "is-a",
      "object": "universal pattern of collective unconscious",
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
  {"id": "1", "text": "The archetype is a universal pattern."},
  {"id": "2", "text": "Архетип является универсальным паттерном."}
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

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Status, version, supported languages |
| `GET` | `/languages` | List languages with pattern counts |
| `POST` | `/preprocess` | Pre-process text chunks |

**POST /preprocess:**

```bash
curl -X POST http://localhost:8200/preprocess \
  -H 'Content-Type: application/json' \
  -d '{
    "chunks": [
      {"id": "1", "text": "The shadow contains repressed elements."},
      {"id": "2", "text": "Анима является частью психики."},
      {"id": "3", "text": "Le rêve est un phénomène naturel."}
    ]
  }'
```

---

## What Meaning We Extract

### Claim Types

The pre-processor classifies every extracted relation into a **claim type** that
Eleutherios can use to route into the appropriate dimension of its extraction pipeline:

| Claim Type | Predicates | Example |
|------------|-----------|---------|
| **FACTUAL** | `is-a`, `has-a`, `contains`, `implements`, `defines` | "The archetype **is a** universal pattern" |
| **CAUSAL** | `causes` | "Repression **causes** neurosis" |
| **SPATIAL** | `located-in` | "The institute **is located in** Zurich" |
| **RELATIONAL** | `similar-to` | "The anima **is similar to** the animus" |
| **STRUCTURAL** | `part-of`, `composed-of` | "The ego **is part of** the psyche" |
| **DEPENDENCY** | `depends-on` | "Individuation **depends on** integrating the shadow" |

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

### Cross-Language Entity Resolution

When the same entity appears in different languages, the resolver unifies them
under a canonical English label:

```
"Moscow"  (EN) -> canonical: "Moscow"
"Москва"  (RU) -> canonical: "Moscow", aliases: ["Москва"]
"Moscou"  (FR) -> canonical: "Moscow", aliases: ["Moscou"]
"موسكو"   (AR) -> canonical: "Moscow", aliases: ["موسكو"]
```

The static equivalence table covers ~120 entries: countries, capitals, major
organizations, and common domain terms (e.g., "mammal"/"млекопитающее"/"mammifère").

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

### Detection Strategy

1. **Script analysis** (highest confidence):
   - Cyrillic codepoints (U+0400..U+052F) > 50% of alphabetic chars -> Russian (conf 0.95)
   - Arabic codepoints (U+0600..U+06FF, U+0750..U+077F) > 50% -> Arabic (conf 0.95)

2. **Latin disambiguation** (word frequency + diacritics):
   - French markers: `le`, `la`, `les`, `des`, `est`, `dans` + accented chars `é`, `è`, `ç`
   - Spanish markers: `el`, `los`, `las`, `está`, `tiene` + `ñ`, inverted punctuation `¿`, `¡`
   - English markers: `the`, `is`, `are`, `was`, `with`, `from`
   - Winner selected by normalized score (conf 0.60-0.85)

---

## Applying to the Carl Jung Corpus

The Jung library at `/home/toasty/Documents/Library/Carl Jung/` contains:

- **44 files** (43 PDFs + 1 EPUB), ~1.29 GB total
- **20 Bollingen Series volumes** (Collected Works Vol. 1-20)
- **1 German text** (doctoral dissertation on Jung's association experiment)
- **42 English texts** (translations, letters, seminars, interviews)
- **2 scanned-image PDFs** without OCR (Red Book, Contributions to Analytical Psychology)

### Recommended Eleutherios Pipeline

```python
# In your Eleutherios chunking stage:
import subprocess
import json

def preprocess_chunks(chunks: list[dict]) -> list[dict]:
    """Send chunks through akh-medu for multilingual pre-processing."""
    input_json = json.dumps(chunks)
    result = subprocess.run(
        ["./akh-medu", "preprocess", "--format", "json"],
        input=input_json,
        capture_output=True,
        text=True,
        env={**os.environ, "RUST_LOG": "error"},
    )
    response = json.loads(result.stdout)
    return response["results"]

# Or for streaming JSONL processing:
def preprocess_stream(chunks):
    """Stream chunks through akh-medu one at a time."""
    proc = subprocess.Popen(
        ["./akh-medu", "preprocess", "--format", "jsonl"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        env={**os.environ, "RUST_LOG": "error"},
    )
    for chunk in chunks:
        proc.stdin.write(json.dumps(chunk) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        yield json.loads(line)
    proc.stdin.close()
    proc.wait()

# Or call the HTTP server:
import requests

def preprocess_http(chunks: list[dict]) -> list[dict]:
    """Call the akh-medu HTTP server."""
    resp = requests.post(
        "http://localhost:8200/preprocess",
        json={"chunks": chunks},
    )
    return resp.json()["results"]
```

### What Eleutherios Gets

For a chunk like *"The archetype is a universal pattern of the collective unconscious"*:

| Field | Value | How Eleutherios Uses It |
|-------|-------|------------------------|
| `source_language` | `"en"` | Route to language-specific extraction models |
| `entities[0].name` | `"archetype"` | Seed entity for Neo4j lookup / creation |
| `entities[0].entity_type` | `"CONCEPT"` | Map to Eleutherios entity taxonomy |
| `claims[0].predicate` | `"is-a"` | Pre-classified relation (skip LLM for simple facts) |
| `claims[0].claim_type` | `"FACTUAL"` | Route to factual extraction dimension |
| `claims[0].confidence` | `0.83` | Weight against LLM confidence |

For the German dissertation text, set the language explicitly since German is not
yet in the auto-detection lexicons:

```bash
cat german_chunks.jsonl | ./akh-medu preprocess --format jsonl --language en
```

(Use `en` as fallback — the grammar parser will still extract 3-token S-P-O
patterns from German since many academic German texts use Latin-rooted predicates.)

---

## Extending with New Languages

Adding a new language requires changes in **three files** and takes ~30 minutes.
Here is the complete procedure using **German** as an example.

### Step 1: Add the Language Variant

**File:** `src/grammar/lexer.rs`

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

**File:** `src/grammar/lexer.rs` — add a new method on `Lexicon`:

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
        // Longest patterns first for greedy matching.
        let relational_patterns = vec![
            rel("befindet sich in", "located-in", 0.90),
            rel("ist ähnlich wie", "similar-to", 0.85),
            rel("ist zusammengesetzt aus", "composed-of", 0.85),
            rel("ist Teil von", "part-of", 0.90),
            rel("hängt ab von", "depends-on", 0.85),
            rel("ist ein", "is-a", 0.90),
            rel("ist eine", "is-a", 0.90),
            rel("hat ein", "has-a", 0.85),
            rel("hat eine", "has-a", 0.85),
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

        let commands = vec![
            ("help".into(), CommandKind::Help),
            ("?".into(), CommandKind::Help),
            ("status".into(), CommandKind::ShowStatus),
        ];

        Self { void_words, relational_patterns, question_words, goal_verbs, commands }
    }
}
```

### Step 3: Add Detection Support (Optional)

**File:** `src/grammar/detect.rs`

For Latin-script languages like German, add word frequency markers to
`detect_latin_language()`:

```rust
const GERMAN_MARKERS: &[&str] = &[
    "der", "die", "das", "ist", "und", "ein", "eine", "nicht",
    "mit", "auf", "für", "von", "sich", "den", "dem", "auch",
    "werden", "haben", "sind", "wird", "kann", "nach", "über",
];
```

Then add a `german_score` accumulator alongside `english_score`, `french_score`,
`spanish_score`, and include German-specific diacritics (`ä`, `ö`, `ü`, `ß`).

For non-Latin scripts (e.g., Chinese, Japanese, Korean, Hindi), add a codepoint
range check in the `detect_language()` function:

```rust
// CJK Unified Ideographs
'\u{4E00}'..='\u{9FFF}' => { cjk += 1; }
// Devanagari
'\u{0900}'..='\u{097F}' => { devanagari += 1; }
```

### Step 4: Add Equivalences (Optional)

**File:** `src/grammar/equivalences.rs`

Add German surface forms to existing entries:

```rust
Equivalence { canonical: "Germany", aliases: &["Allemagne", "Alemania", "Германия", "ألمانيا", "Deutschland"] },
//                                                                                              ^^^^^^^^^^ add
```

And add new entries for German-specific terms:

```rust
Equivalence { canonical: "psyche", aliases: &["Psyche", "психика", "نفس", "psyché", "psique"] },
```

### Step 5: Rebuild and Test

```bash
cargo test --lib
cargo build --release

echo '{"text":"Der Archetyp ist ein universelles Muster."}' \
  | ./target/release/akh-medu preprocess --format jsonl --language de
```

### Checklist for New Languages

- [ ] Add variant to `Language` enum in `lexer.rs`
- [ ] Update `bcp47()`, `from_code()`, `Display` on `Language`
- [ ] Add arm to `Lexicon::for_language()`
- [ ] Create `Lexicon::default_LANG()` with relational patterns, void words, question words, goal verbs
- [ ] (Optional) Add detection markers in `detect.rs` for `Language::Auto` support
- [ ] (Optional) Add cross-lingual entries in `equivalences.rs`
- [ ] Run `cargo test --lib` (must pass, zero warnings)
- [ ] Test with sample text: `echo '{"text":"..."}' | akh-medu preprocess --format jsonl --language XX`

---

## Architecture Notes

### Performance

- Grammar parsing runs in **< 1ms per chunk** (measured 0ms for 5-chunk batch)
- No external NLP dependencies, no model loading, no GPU required
- The HTTP server handles requests synchronously (one engine instance, no async parsing needed)
- Memory footprint: ~50MB for the engine with default 10,000-dimension hypervectors

### What the Pre-Processor Does NOT Do

- **Coreference resolution**: "He studied in Zurich" -- "He" is not resolved to "Jung"
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

---

## Relational Pattern Reference

### English (21 patterns)

| Pattern | Canonical | Confidence | Example |
|---------|-----------|------------|---------|
| "is similar to" | `similar-to` | 0.85 | The anima is similar to the animus |
| "is located in" | `located-in` | 0.90 | Jung is located in Zurich |
| "is composed of" | `composed-of` | 0.85 | The psyche is composed of conscious and unconscious |
| "is part of" | `part-of` | 0.90 | The ego is part of the psyche |
| "is made of" | `composed-of` | 0.85 | The mandala is made of geometric patterns |
| "depends on" | `depends-on` | 0.85 | Individuation depends on shadow integration |
| "belongs to" | `part-of` | 0.85 | This archetype belongs to the collective unconscious |
| "is a" / "is an" | `is-a` | 0.90 | The shadow is a part of the unconscious |
| "are a" / "are an" | `is-a` | 0.85 | Dreams are a window to the unconscious |
| "has a" / "has an" | `has-a` | 0.85 | The psyche has a shadow component |
| "have a" | `has-a` | 0.85 | Archetypes have a universal quality |
| "are" | `is-a` | 0.85 | Dreams are manifestations of the unconscious |
| "has" / "have" | `has-a` | 0.85 | The ego has defense mechanisms |
| "contains" | `contains` | 0.85 | The unconscious contains archetypes |
| "causes" | `causes` | 0.85 | Repression causes neurosis |
| "implements" | `implements` | 0.85 | (code domain) |
| "defines" | `defines` | 0.85 | (code domain) |

### Russian (13 patterns)

| Pattern | Canonical | Example |
|---------|-----------|---------|
| "является частью" | `part-of` | Эго является частью психики |
| "находится в" | `located-in` | Институт находится в Цюрихе |
| "состоит из" | `composed-of` | Психика состоит из сознания и бессознательного |
| "зависит от" | `depends-on` | Индивидуация зависит от интеграции тени |
| "похож на" | `similar-to` | Анима похож на анимус |
| "содержит в себе" | `contains` | Бессознательное содержит в себе архетипы |
| "является" | `is-a` | Архетип является универсальным паттерном |
| "имеет" | `has-a` | Психика имеет теневой компонент |
| "содержит" | `contains` | Бессознательное содержит архетипы |
| "вызывает" | `causes` | Вытеснение вызывает невроз |
| "это" | `is-a` | Тень это часть бессознательного |

### Arabic (11 patterns)

| Pattern | Canonical | Example |
|---------|-----------|---------|
| "يحتوي على" | `contains` | اللاوعي يحتوي على النماذج الأولية |
| "يقع في" | `located-in` | المعهد يقع في زيورخ |
| "جزء من" | `part-of` | الأنا جزء من النفس |
| "يتكون من" | `composed-of` | النفس يتكون من الوعي واللاوعي |
| "يعتمد على" | `depends-on` | التفرد يعتمد على دمج الظل |
| "هو" / "هي" | `is-a` | النموذج الأولي هو نمط عالمي |
| "لديه" / "لديها" | `has-a` | النفس لديها مكون ظلي |
| "يسبب" | `causes` | الكبت يسبب العصاب |
| "يشبه" | `similar-to` | الأنيما يشبه الأنيموس |

### French (16 patterns)

| Pattern | Canonical | Example |
|---------|-----------|---------|
| "est similaire à" | `similar-to` | L'anima est similaire à l'animus |
| "se trouve dans" | `located-in` | L'institut se trouve dans Zurich |
| "est composé de" | `composed-of` | La psyché est composée de conscient et inconscient |
| "fait partie de" | `part-of` | L'ego fait partie de la psyché |
| "dépend de" | `depends-on` | L'individuation dépend de l'intégration de l'ombre |
| "est un" / "est une" | `is-a` | L'archétype est un modèle universel |
| "a un" / "a une" | `has-a` | La psyché a une composante d'ombre |
| "contient" | `contains` | L'inconscient contient des archétypes |
| "cause" | `causes` | Le refoulement cause la névrose |

### Spanish (14 patterns)

| Pattern | Canonical | Example |
|---------|-----------|---------|
| "es similar a" | `similar-to` | El anima es similar al animus |
| "se encuentra en" | `located-in` | El instituto se encuentra en Zúrich |
| "está compuesto de" | `composed-of` | La psique está compuesta de consciente e inconsciente |
| "es parte de" | `part-of` | El ego es parte de la psique |
| "depende de" | `depends-on` | La individuación depende de la integración de la sombra |
| "es un" / "es una" | `is-a` | El arquetipo es un patrón universal |
| "tiene un" / "tiene una" | `has-a` | La psique tiene un componente de sombra |
| "contiene" | `contains` | El inconsciente contiene arquetipos |
| "causa" | `causes` | La represión causa neurosis |
