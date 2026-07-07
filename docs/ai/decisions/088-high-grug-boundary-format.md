# ADR 088 — High-Grug v0: Retire JSON as the Tool-Calling Boundary Format

- **Status:** Accepted (built + measured). Number provisional — reconcile on merge.
- **Date:** 2026-07-07
- **Deciders:** Till, Fable 5
- **Plan:** [Phase 51 P2 — format-first corpus](../plans/2026-07-07-phase51-p2-format-first-corpus.md)
- **Evidence:** [P1 scaling results §10–§11](../2026-07-05-phase51-p1-scaling-results.md); `runs/p2-grug/`
- **Relates:** extends ADR 087 (BabyLLM); the boundary format is the first concrete piece of the "High Grug" language it calls for.

## Context

The P1a-4 arc tried to make a from-scratch 13M model do zero-shot JSON tool-calling and repeatedly failed by adding *model* machinery: more capacity (13M→30M, no help), a copy/pointer mechanism (gate collapsed, `p_gen`≈0.96), and copy-supervision (grad-clip-capped, zero-shot plateaued ~0.06). Constrained decoding (Step A) only ever produced *valid-but-wrong*.

The small-model literature names the cause: for a sub-3B model, format fidelity "consumes a material part of the generation problem" — JSON's braces, quotes, per-field keys, nesting and type punctuation compete with the task for scarce capacity ("the format tax"; *The Constraint Tax*, *When Correct Isn't Usable*, 2026). The field has already moved: Llama 3.2+ emit **pythonic** calls, Gemma uses **special-token** delimiters — structure carried by single tokens + a minimal payload, not hand-balanced punctuation. Our pipeline was *doubly* taxed: it serialized both the offered schema (prompt) and the emitted call (target) as JSON.

## Decision

**Retire JSON as the boundary representation. Adopt High-Grug v0**, a minimal flat format, for *both* the offered schema and the emitted call:

```text
schema (prompt):  tool <name> | <pname> <type> <req> | ...
call   (target):  call <name> | <argname> <value> | ...
```

- Structure is carried by the literal words `tool`/`call` and the ` | ` separator (each one token); an `<arg>` segment splits once on its first space into `(name, value)`, so values contain spaces without quoting. No `{}`, `""`, `:`, `,`, no nesting.
- **Decode-safe plain ASCII** — the delimiters are ordinary tokens, not tokenizer specials, so they survive `decode` and need **no tokenizer change**; the JSON↔Grug swap is pure.
- Implemented in `grug_data::format` (`TargetFormat {Json, Grug}`, `render_tools`/`render_call`/`parse_call`), threaded through `grug_train` via `*_fmt` variants; the JSON path is byte-identical (delegates), so all prior baselines stand.

High-Grug v0 is the seed of ADR 087's "High Grug designed as a language": these delimiters are the first designed structural symbols; later versions can promote them to SymbolId-bearing tokens.

## Consequences / evidence

Measured A/B on the identical `p1a2-ep8` recipe (base 13M, seed 42, 1000 tools, only the format varies; `runs/p2-grug/`):

- **Structural tax removed:** encoded sequence length **halved** (median 109 vs JSON's ~345); valid-format is trivial from step 0.
- **In-distribution accuracy is a wash** — Grug tracks *near, marginally below* JSON (arg ~0.63–0.66 vs ~0.78 at matched-ish steps); the freed capacity did not visibly lift accuracy on this synthetic task.
- **Zero-shot is unchanged** — axis-A schema mean **0.030** (max 0.055), identical to JSON's 0.031, flat across the whole run.

So the format is **not** the zero-shot cure — that is a fourth negative on the same axis (see ADR 089). But it is the right boundary representation to **lock in** for its efficiency and reliability: half the context per call, trivially-valid syntax (no wasted capacity on brace-balancing), a total/deterministic parser, and direct alignment with the High-Grug/SymbolId endgame. Go-forward training and eval use `TargetFormat::Grug` by default; JSON is retained only for reproducing historical baselines.

## Alternatives considered

- **Pythonic `f(a=1, b='x')`** (Llama 3.2): good, but still balances `()` and quotes strings; High-Grug v0's pipe form is flatter and quote-free, and closer to a symbol stream.
- **Special-token delimiters** (Gemma `<tool_call>`): more learnable per-token but `decode(skip_special_tokens)` strips them, breaking the scorer without a decode-path change; deferred to a later High-Grug version that parses from token ids.
- **Keep JSON + lean on constrained decoding:** rejected — Step A showed constrained JSON yields valid-but-wrong; it treats the symptom, not the tax.
