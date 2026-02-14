# Reasoning

akh-medu uses the `egg` e-graph library for symbolic reasoning via equality
saturation. E-graphs efficiently represent equivalence classes of expressions,
and rewrite rules transform them until a fixed point is reached.

## How E-Graphs Work

An e-graph (equality graph) stores expressions and their equivalences
simultaneously. When a rewrite rule fires, the e-graph doesn't destructively
replace -- it adds the new expression to the same equivalence class.

```
Before rule: bind(unbind(X, Y), Y) => X
E-graph: { bind(unbind(A, B), B) }

After rule:
E-graph: { bind(unbind(A, B), B), A }  <- both in the same e-class
```

This means multiple rules can fire without conflict, and the optimal result
is extracted after all rules have been applied.

## AkhLang

akh-medu defines `AkhLang`, a domain-specific language for the e-graph:

```rust
define_language! {
    pub enum AkhLang {
        // VSA operations
        "bind" = Bind([Id; 2]),
        "unbind" = Unbind([Id; 2]),
        "bundle" = Bundle([Id; 2]),
        "permute" = Permute([Id; 2]),
        // Knowledge operations
        "triple" = Triple([Id; 3]),
        "sim" = Similarity([Id; 2]),
        // Leaf nodes
        Symbol(Symbol),
    }
}
```

## Built-in Rewrite Rules

The engine ships with algebraic rules for VSA operations:

| Rule | Pattern | Result | Purpose |
|------|---------|--------|---------|
| Bind-unbind cancel | `unbind(bind(X, Y), Y)` | `X` | VSA algebra: unbinding reverses binding |
| Unbind-bind cancel | `bind(unbind(X, Y), Y)` | `X` | Symmetric cancellation |
| Bind commutative | `bind(X, Y)` | `bind(Y, X)` | XOR is commutative |
| Bundle commutative | `bundle(X, Y)` | `bundle(Y, X)` | Majority vote is commutative |
| Bind self-inverse | `bind(X, X)` | `identity` | XOR with self = zero |

Skills can contribute additional rules that are loaded dynamically.

## Using the Reasoner

### CLI

```bash
# Simplify an expression
akh-medu reason --expr "unbind(bind(Dog, is-a), is-a)"
# Output: Dog

# Verbose mode shows the e-graph state
akh-medu reason --expr "unbind(bind(Dog, is-a), is-a)" --verbose
```

### Rust API

```rust
use egg::{rewrite, Runner, Extractor, AstSize};
use akh_medu::reason::AkhLang;

let rules = akh_medu::reason::built_in_rules();

let expr = "unbind(bind(Dog, is-a), is-a)".parse()?;
let runner = Runner::default()
    .with_expr(&expr)
    .run(&rules);

let extractor = Extractor::new(&runner.egraph, AstSize);
let (cost, best) = extractor.find_best(runner.roots[0]);
println!("Simplified to: {} (cost {})", best, cost);
```

## Forward-Chaining Inference

The agent's `infer_rules` tool runs rewrite rules as forward-chaining
inference:

1. Existing triples are encoded as e-graph expressions.
2. Rewrite rules fire, potentially producing new triple expressions.
3. New triples are extracted and committed to the knowledge graph.

```bash
akh-medu agent infer --max-iterations 10 --min-confidence 0.5
```

## E-Graph Verification

During inference, VSA-recovered results can optionally be verified by the
e-graph. If `unbind(bind(A, B), B)` doesn't simplify to `A`, it suggests
the recovery was noisy and confidence is reduced by 10%.

This is controlled by `InferenceQuery::with_egraph_verification()`:

```rust
let query = InferenceQuery::default()
    .with_seeds(vec![dog_id])
    .with_egraph_verification();
```

## Skills and Custom Rules

Skill packs can contribute domain-specific rewrite rules:

```toml
# In a skill's rules.toml
[[rules]]
name = "transitive-is-a"
lhs = "(triple ?a is-a ?b) (triple ?b is-a ?c)"
rhs = "(triple ?a is-a ?c)"
```

When a skill is loaded, its rules are compiled into `egg` rewrites and added
to the engine's rule set. When the skill is unloaded, the rules are removed.

## Design Rationale

**Why e-graphs?** Traditional rule engines apply rules destructively --
once a rule fires, the original expression is gone. E-graphs keep all
intermediate forms, avoiding the phase-ordering problem where the order of
rule application matters. With equality saturation, all rules fire
simultaneously and the optimal result is extracted at the end.

**Why algebraic verification?** VSA recovery via `unbind(subject, predicate)`
is approximate -- the recovered vector is searched against item memory for
the nearest match. This can produce false positives, especially with high
symbol density. The e-graph provides a cheap mathematical sanity check:
if the algebraic identity doesn't hold, the recovery is suspect.
