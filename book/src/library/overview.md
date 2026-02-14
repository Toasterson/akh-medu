# Shared Content Library

The shared content library lets you ingest books, websites, papers, and
documents into the knowledge graph. Documents are parsed into structural
elements (chapters, sections, paragraphs), stored as KG symbols with
well-known `doc:*` predicates, and embedded via VSA for semantic search.

## How It Works

```
  File / URL
      │
      ▼
  ┌──────────┐     ┌───────────┐     ┌──────────┐     ┌───────────┐
  │  Parse    │ --> │  Chunk    │ --> │ Extract  │ --> │  Embed    │
  │ HTML/PDF/ │     │ normalize │     │ triples  │     │ VSA vecs  │
  │ EPUB/text │     │ 200-500w  │     │ NLP      │     │ per chunk │
  └──────────┘     └───────────┘     └──────────┘     └───────────┘
      │                                                     │
      ▼                                                     ▼
  ┌──────────┐                                       ┌───────────┐
  │ Catalog  │  catalog.json with document metadata  │ Item Mem  │
  └──────────┘                                       └───────────┘
```

1. **Parse** -- Format-specific parser (HTML, PDF, EPUB, plain text) extracts
   headings, paragraphs, and metadata.
2. **Chunk** -- Short paragraphs are merged and long ones split to produce
   chunks targeting 200-500 words for consistent NLP quality.
3. **Extract** -- NLP extraction discovers triples from each chunk's text.
4. **Embed** -- Each chunk is encoded as a VSA hypervector and stored in item
   memory for semantic search.

## Supported Formats

| Format | Detection | Parser |
|--------|-----------|--------|
| HTML | `.html`/`.htm` extension or URL source | `scraper` crate -- extracts `<h1>`-`<h6>`, `<p>`, `<meta>` |
| PDF | `.pdf` extension | `pdf-extract` crate -- page-level text extraction |
| EPUB | `.epub` extension | `epub` crate -- spine items become chapters |
| Plain text | fallback | Splits on double newlines for paragraphs |

## Document Structure in the KG

Each ingested document creates a hierarchy of symbols:

```
doc:{slug}                    # Document root
  ├── doc:has_chapter  → ch:{slug}:0
  │     └── doc:has_paragraph → para:{slug}:0
  │     └── doc:has_paragraph → para:{slug}:1
  │     └── doc:has_section   → sec:{slug}:0:1
  ├── doc:has_chapter  → ch:{slug}:1
  │     └── doc:has_paragraph → para:{slug}:2
  ...
```

### Well-Known Predicates

| Predicate | Description |
|-----------|-------------|
| `doc:has_chapter` | Document → chapter |
| `doc:has_section` | Chapter → section |
| `doc:has_paragraph` | Document/chapter → paragraph chunk |
| `doc:next_chunk` | Paragraph → next paragraph (reading order) |
| `doc:has_title` | Document → title string |
| `doc:has_author` | Document → author string |
| `doc:has_format` | Document → format (html, pdf, epub, text) |
| `doc:has_source` | Document → source path or URL |
| `doc:has_language` | Document → language code |
| `doc:has_description` | Document → description string |
| `doc:has_keyword` | Document → keyword |
| `doc:has_tag` | Document → user tag |
| `doc:chunk_text` | Paragraph → raw text content |
| `doc:chunk_index` | Paragraph → ordinal position |

## Catalog

The catalog is a persistent JSON index at `~/.local/share/akh-medu/library/catalog.json`
that tracks all ingested documents with their metadata. It stores the document
ID (slug), title, format, source path/URL, tags, and chunk count.

## CLI Commands

### library add

Add a document to the library from a file path or URL.

```bash
akh-medu library add paper.pdf
akh-medu library add https://example.com/article.html --title "My Article"
akh-medu library add book.epub --tags "physics,textbook"
akh-medu library add notes.txt --format text
```

| Option | Description |
|--------|-------------|
| `--title <TEXT>` | Override document title |
| `--tags <LIST>` | Comma-separated tags |
| `--format <FMT>` | Override format detection: `html`, `pdf`, `epub`, `text` |

### library list

List all documents in the library.

```bash
akh-medu library list
```

### library search

Search library content by text similarity.

```bash
akh-medu library search --query "quantum entanglement" --top-k 10
```

| Option | Description | Default |
|--------|-------------|---------|
| `--query <TEXT>` | Search text | Required |
| `--top-k <N>` | Maximum results | `5` |

### library info

Show detailed information about a document.

```bash
akh-medu library info quantum-mechanics-textbook
```

### library remove

Remove a document from the library.

```bash
akh-medu library remove quantum-mechanics-textbook
```

### library watch

Watch a directory for new files and auto-ingest them. Defaults to the
library inbox directory (`~/.local/share/akh-medu/library/inbox/`).

```bash
akh-medu library watch
akh-medu library watch --dir /path/to/papers/
```

## Agent Integration

The agent has two tools for working with library content:

- **`content_ingest`** -- Ingest a document (file or URL) into the library.
  The agent uses this when a goal involves importing or learning from external
  content. See [Tools](../agent/tools.md#content_ingest).

- **`library_search`** -- Search ingested library paragraphs by natural
  language query via VSA similarity. The agent uses this when a goal asks
  about previously ingested content (e.g., "What did that paper say about
  gravity?"). See [Tools](../agent/tools.md#library_search).

Both tools are scored by the OODA loop's [VSA-based tool selector](../agent/ooda-loop.md)
using synonym-expanded keyword profiles, so natural language goals reliably
activate them.

## Compartment Integration

Each ingested document is stored in its own knowledge compartment
(`library:{slug}`), which can be mounted by any workspace. This keeps
document knowledge isolated until explicitly shared. See
[Knowledge Compartments](../advanced/compartments.md) for details.
