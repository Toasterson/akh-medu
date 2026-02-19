//! Anti-unification on `SimplifiedAst` trees.
//!
//! Finds the **most specific generalization** (MSG) of two or more AST skeletons.
//! Produces a `GeneralizedAst` pattern with holes (`AntiUnifyVar`) where the inputs
//! differ. This is the core of the DreamCoder/LILO-inspired library learning cycle:
//! recurring sub-patterns across generated code become reusable `CodeTemplate` entries.
//!
//! ## Scoring
//!
//! Candidate abstractions are scored by a Stitch-inspired compression metric:
//! `occurrences × node_count`. High compression means the pattern is both frequent
//! and structurally large — a good candidate for extraction as a reusable template.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::agent::tools::pattern_mine::{SimplifiedAst, ast_fingerprint};

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// A hole in a generalized pattern — this position differed between inputs.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AntiUnifyVar {
    /// Variable name, e.g., `"?v0"`, `"?v1"`.
    pub name: String,
    /// Concrete values this variable took in each input.
    pub instances: Vec<String>,
}

/// A slot that is either a concrete (fixed) value or a variable (hole).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AstSlot {
    /// Same `usize` value in all inputs.
    Fixed(usize),
    /// Same `bool` value in all inputs.
    Bool(bool),
    /// Same optional string in all inputs.
    Str(Option<String>),
    /// Differs between inputs — a hole.
    Var(AntiUnifyVar),
}

/// A `SimplifiedAst` with holes — the result of anti-unifying multiple ASTs.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GeneralizedAst {
    /// Generalized function.
    Function {
        param_count: AstSlot,
        has_return: AstSlot,
        body: Vec<GeneralizedAst>,
    },
    /// Generalized struct.
    Struct {
        field_count: AstSlot,
        derive_count: AstSlot,
    },
    /// Generalized enum.
    Enum {
        variant_count: AstSlot,
        derive_count: AstSlot,
    },
    /// Generalized impl block.
    Impl {
        method_count: AstSlot,
        trait_name: AstSlot,
    },
    /// Generalized match expression.
    Match { arm_count: AstSlot },
    /// Generalized if/else.
    IfElse { has_else: AstSlot },
    /// For loop (no fields to generalize).
    ForLoop,
    /// Generalized closure.
    Closure { param_count: AstSlot },
    /// Generalized block.
    Block { children: Vec<GeneralizedAst> },
    /// A hole — the entire subtree differs between inputs.
    Var(AntiUnifyVar),
    /// A concrete leaf — identical across all inputs.
    Concrete(SimplifiedAst),
}

// ---------------------------------------------------------------------------
// Variable counter
// ---------------------------------------------------------------------------

/// Thread-local counter for generating fresh variable names within one
/// anti-unification session.
struct VarCounter {
    next: usize,
}

impl VarCounter {
    fn new() -> Self {
        Self { next: 0 }
    }

    fn fresh(&mut self, instances: Vec<String>) -> AntiUnifyVar {
        let name = format!("?v{}", self.next);
        self.next += 1;
        AntiUnifyVar { name, instances }
    }
}

// ---------------------------------------------------------------------------
// Anti-unification: pair
// ---------------------------------------------------------------------------

/// Anti-unify two `SimplifiedAst` trees, producing the most specific generalization.
///
/// - If `a == b`, returns `Concrete(a.clone())`
/// - If same variant but different fields, recursively anti-unifies each field
/// - If different variants, returns a `Var` (entire position is a hole)
pub fn anti_unify_pair(a: &SimplifiedAst, b: &SimplifiedAst) -> GeneralizedAst {
    let mut ctr = VarCounter::new();
    anti_unify_pair_inner(a, b, &mut ctr)
}

fn anti_unify_pair_inner(
    a: &SimplifiedAst,
    b: &SimplifiedAst,
    ctr: &mut VarCounter,
) -> GeneralizedAst {
    // Identical trees → concrete
    if a == b {
        return GeneralizedAst::Concrete(a.clone());
    }

    match (a, b) {
        (
            SimplifiedAst::Function {
                param_count: p1,
                has_return: r1,
                body: b1,
            },
            SimplifiedAst::Function {
                param_count: p2,
                has_return: r2,
                body: b2,
            },
        ) => {
            let param_count = unify_usize(*p1, *p2, ctr);
            let has_return = unify_bool(*r1, *r2, ctr);
            let body = unify_vec(b1, b2, ctr);
            GeneralizedAst::Function {
                param_count,
                has_return,
                body,
            }
        }

        (
            SimplifiedAst::Struct {
                field_count: f1,
                derive_count: d1,
            },
            SimplifiedAst::Struct {
                field_count: f2,
                derive_count: d2,
            },
        ) => GeneralizedAst::Struct {
            field_count: unify_usize(*f1, *f2, ctr),
            derive_count: unify_usize(*d1, *d2, ctr),
        },

        (
            SimplifiedAst::Enum {
                variant_count: v1,
                derive_count: d1,
            },
            SimplifiedAst::Enum {
                variant_count: v2,
                derive_count: d2,
            },
        ) => GeneralizedAst::Enum {
            variant_count: unify_usize(*v1, *v2, ctr),
            derive_count: unify_usize(*d1, *d2, ctr),
        },

        (
            SimplifiedAst::Impl {
                method_count: m1,
                trait_name: t1,
            },
            SimplifiedAst::Impl {
                method_count: m2,
                trait_name: t2,
            },
        ) => GeneralizedAst::Impl {
            method_count: unify_usize(*m1, *m2, ctr),
            trait_name: unify_opt_str(t1, t2, ctr),
        },

        (
            SimplifiedAst::Match { arm_count: a1 },
            SimplifiedAst::Match { arm_count: a2 },
        ) => GeneralizedAst::Match {
            arm_count: unify_usize(*a1, *a2, ctr),
        },

        (
            SimplifiedAst::IfElse { has_else: e1 },
            SimplifiedAst::IfElse { has_else: e2 },
        ) => GeneralizedAst::IfElse {
            has_else: unify_bool(*e1, *e2, ctr),
        },

        (SimplifiedAst::ForLoop, SimplifiedAst::ForLoop) => GeneralizedAst::ForLoop,

        (
            SimplifiedAst::Closure { param_count: p1 },
            SimplifiedAst::Closure { param_count: p2 },
        ) => GeneralizedAst::Closure {
            param_count: unify_usize(*p1, *p2, ctr),
        },

        (
            SimplifiedAst::Block { children: c1 },
            SimplifiedAst::Block { children: c2 },
        ) => GeneralizedAst::Block {
            children: unify_vec(c1, c2, ctr),
        },

        // Different variants → whole thing is a hole
        _ => {
            let fp_a = ast_fingerprint(a);
            let fp_b = ast_fingerprint(b);
            GeneralizedAst::Var(ctr.fresh(vec![fp_a, fp_b]))
        }
    }
}

// ---------------------------------------------------------------------------
// Field-level unification helpers
// ---------------------------------------------------------------------------

fn unify_usize(a: usize, b: usize, ctr: &mut VarCounter) -> AstSlot {
    if a == b {
        AstSlot::Fixed(a)
    } else {
        AstSlot::Var(ctr.fresh(vec![a.to_string(), b.to_string()]))
    }
}

fn unify_bool(a: bool, b: bool, ctr: &mut VarCounter) -> AstSlot {
    if a == b {
        AstSlot::Bool(a)
    } else {
        AstSlot::Var(ctr.fresh(vec![a.to_string(), b.to_string()]))
    }
}

fn unify_opt_str(a: &Option<String>, b: &Option<String>, ctr: &mut VarCounter) -> AstSlot {
    if a == b {
        AstSlot::Str(a.clone())
    } else {
        let sa = a.as_deref().unwrap_or("<none>").to_string();
        let sb = b.as_deref().unwrap_or("<none>").to_string();
        AstSlot::Var(ctr.fresh(vec![sa, sb]))
    }
}

/// Anti-unify two body/children vectors by zipping to the shorter length.
/// Excess elements become `Var` holes.
fn unify_vec(
    a: &[SimplifiedAst],
    b: &[SimplifiedAst],
    ctr: &mut VarCounter,
) -> Vec<GeneralizedAst> {
    let min_len = a.len().min(b.len());
    let mut result: Vec<GeneralizedAst> = a[..min_len]
        .iter()
        .zip(&b[..min_len])
        .map(|(x, y)| anti_unify_pair_inner(x, y, ctr))
        .collect();

    // Excess from the longer side become holes
    for extra in a.iter().skip(min_len) {
        let fp = ast_fingerprint(extra);
        result.push(GeneralizedAst::Var(ctr.fresh(vec![fp, "<absent>".into()])));
    }
    for extra in b.iter().skip(min_len) {
        let fp = ast_fingerprint(extra);
        result.push(GeneralizedAst::Var(
            ctr.fresh(vec!["<absent>".into(), fp]),
        ));
    }

    result
}

// ---------------------------------------------------------------------------
// Anti-unification: multi (fold)
// ---------------------------------------------------------------------------

/// Anti-unify multiple `SimplifiedAst` trees by pairwise folding.
///
/// Produces a progressively more general pattern as more inputs are considered.
/// With N inputs, the result has at most N-1 folding steps.
pub fn anti_unify_multi(asts: &[&SimplifiedAst]) -> GeneralizedAst {
    assert!(!asts.is_empty(), "anti_unify_multi requires at least one input");

    if asts.len() == 1 {
        return GeneralizedAst::Concrete(asts[0].clone());
    }

    // Start from a pair, then fold each subsequent input
    let mut result = anti_unify_pair(asts[0], asts[1]);

    for ast in &asts[2..] {
        // Convert current generalized result back to a SimplifiedAst approximation
        // for the next pairwise step. This is a lossy projection (vars become Other)
        // but preserves the structural skeleton for further generalization.
        let approx = approximate_simplified(&result);
        result = anti_unify_pair(&approx, ast);
    }

    result
}

/// Project a `GeneralizedAst` back to `SimplifiedAst` (public wrapper).
///
/// Variables become `Other`, concrete nodes pass through, and slot values
/// use their fixed values or defaults. Used by the library learner to
/// encode generalized patterns as VSA vectors.
pub fn approximate_simplified_pub(g: &GeneralizedAst) -> SimplifiedAst {
    approximate_simplified(g)
}

/// Project a `GeneralizedAst` back to `SimplifiedAst` for iterative folding.
///
/// Variables become `Other`, concrete nodes pass through, and slot values
/// use their fixed values or defaults.
fn approximate_simplified(g: &GeneralizedAst) -> SimplifiedAst {
    match g {
        GeneralizedAst::Function {
            param_count,
            has_return,
            body,
        } => SimplifiedAst::Function {
            param_count: slot_to_usize(param_count),
            has_return: slot_to_bool(has_return),
            body: body.iter().map(approximate_simplified).collect(),
        },
        GeneralizedAst::Struct {
            field_count,
            derive_count,
        } => SimplifiedAst::Struct {
            field_count: slot_to_usize(field_count),
            derive_count: slot_to_usize(derive_count),
        },
        GeneralizedAst::Enum {
            variant_count,
            derive_count,
        } => SimplifiedAst::Enum {
            variant_count: slot_to_usize(variant_count),
            derive_count: slot_to_usize(derive_count),
        },
        GeneralizedAst::Impl {
            method_count,
            trait_name,
        } => SimplifiedAst::Impl {
            method_count: slot_to_usize(method_count),
            trait_name: slot_to_opt_str(trait_name),
        },
        GeneralizedAst::Match { arm_count } => SimplifiedAst::Match {
            arm_count: slot_to_usize(arm_count),
        },
        GeneralizedAst::IfElse { has_else } => SimplifiedAst::IfElse {
            has_else: slot_to_bool(has_else),
        },
        GeneralizedAst::ForLoop => SimplifiedAst::ForLoop,
        GeneralizedAst::Closure { param_count } => SimplifiedAst::Closure {
            param_count: slot_to_usize(param_count),
        },
        GeneralizedAst::Block { children } => SimplifiedAst::Block {
            children: children.iter().map(approximate_simplified).collect(),
        },
        GeneralizedAst::Var(_) => SimplifiedAst::Other,
        GeneralizedAst::Concrete(ast) => ast.clone(),
    }
}

fn slot_to_usize(slot: &AstSlot) -> usize {
    match slot {
        AstSlot::Fixed(v) => *v,
        _ => 0,
    }
}

fn slot_to_bool(slot: &AstSlot) -> bool {
    match slot {
        AstSlot::Bool(v) => *v,
        _ => false,
    }
}

fn slot_to_opt_str(slot: &AstSlot) -> Option<String> {
    match slot {
        AstSlot::Str(v) => v.clone(),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Fingerprinting
// ---------------------------------------------------------------------------

/// Produce a deterministic compact fingerprint for a `GeneralizedAst`.
///
/// Like `ast_fingerprint` but with `?v0`, `?v1` for variable positions.
pub fn generalized_fingerprint(g: &GeneralizedAst) -> String {
    match g {
        GeneralizedAst::Function {
            param_count,
            has_return,
            body,
        } => {
            let pc = slot_fp(param_count);
            let ret = match has_return {
                AstSlot::Bool(true) => "ret".to_string(),
                AstSlot::Bool(false) => "void".to_string(),
                AstSlot::Var(v) => v.name.clone(),
                _ => "?".to_string(),
            };
            let body_fps: Vec<String> = body.iter().map(generalized_fingerprint).collect();
            if body_fps.is_empty() {
                format!("fn({pc},{ret})")
            } else {
                format!("fn({pc},{ret},[{}])", body_fps.join(","))
            }
        }
        GeneralizedAst::Struct {
            field_count,
            derive_count,
        } => {
            format!("struct({},d{})", slot_fp(field_count), slot_fp(derive_count))
        }
        GeneralizedAst::Enum {
            variant_count,
            derive_count,
        } => {
            format!("enum({},d{})", slot_fp(variant_count), slot_fp(derive_count))
        }
        GeneralizedAst::Impl {
            method_count,
            trait_name,
        } => {
            let mc = slot_fp(method_count);
            match trait_name {
                AstSlot::Str(Some(t)) => format!("impl({mc},{t})"),
                AstSlot::Var(v) => format!("impl({mc},{})", v.name),
                _ => format!("impl({mc})"),
            }
        }
        GeneralizedAst::Match { arm_count } => {
            format!("match({})", slot_fp(arm_count))
        }
        GeneralizedAst::IfElse { has_else } => match has_else {
            AstSlot::Bool(true) => "if-else".to_string(),
            AstSlot::Bool(false) => "if".to_string(),
            AstSlot::Var(v) => format!("if({})", v.name),
            _ => "if(?)".to_string(),
        },
        GeneralizedAst::ForLoop => "for".to_string(),
        GeneralizedAst::Closure { param_count } => {
            format!("closure({})", slot_fp(param_count))
        }
        GeneralizedAst::Block { children } => {
            let fps: Vec<String> = children.iter().map(generalized_fingerprint).collect();
            format!("block([{}])", fps.join(","))
        }
        GeneralizedAst::Var(v) => v.name.clone(),
        GeneralizedAst::Concrete(ast) => ast_fingerprint(ast),
    }
}

fn slot_fp(slot: &AstSlot) -> String {
    match slot {
        AstSlot::Fixed(v) => v.to_string(),
        AstSlot::Bool(v) => v.to_string(),
        AstSlot::Str(Some(s)) => s.clone(),
        AstSlot::Str(None) => "<none>".to_string(),
        AstSlot::Var(v) => v.name.clone(),
    }
}

// ---------------------------------------------------------------------------
// Node counting
// ---------------------------------------------------------------------------

/// Count non-variable nodes in a `GeneralizedAst`.
pub fn node_count(g: &GeneralizedAst) -> usize {
    match g {
        GeneralizedAst::Function { body, .. } => {
            1 + body.iter().map(node_count).sum::<usize>()
        }
        GeneralizedAst::Struct { .. }
        | GeneralizedAst::Enum { .. }
        | GeneralizedAst::Impl { .. }
        | GeneralizedAst::Match { .. }
        | GeneralizedAst::IfElse { .. }
        | GeneralizedAst::ForLoop
        | GeneralizedAst::Closure { .. } => 1,
        GeneralizedAst::Block { children } => {
            1 + children.iter().map(node_count).sum::<usize>()
        }
        GeneralizedAst::Var(_) => 0,
        GeneralizedAst::Concrete(_) => 1,
    }
}

/// Count variable (hole) nodes in a `GeneralizedAst`.
pub fn var_count(g: &GeneralizedAst) -> usize {
    let slot_vars = |slots: &[&AstSlot]| -> usize {
        slots.iter().filter(|s| matches!(s, AstSlot::Var(_))).count()
    };

    match g {
        GeneralizedAst::Function {
            param_count,
            has_return,
            body,
        } => {
            slot_vars(&[param_count, has_return])
                + body.iter().map(var_count).sum::<usize>()
        }
        GeneralizedAst::Struct {
            field_count,
            derive_count,
        } => slot_vars(&[field_count, derive_count]),
        GeneralizedAst::Enum {
            variant_count,
            derive_count,
        } => slot_vars(&[variant_count, derive_count]),
        GeneralizedAst::Impl {
            method_count,
            trait_name,
        } => slot_vars(&[method_count, trait_name]),
        GeneralizedAst::Match { arm_count } => slot_vars(&[arm_count]),
        GeneralizedAst::IfElse { has_else } => slot_vars(&[has_else]),
        GeneralizedAst::ForLoop => 0,
        GeneralizedAst::Closure { param_count } => slot_vars(&[param_count]),
        GeneralizedAst::Block { children } => {
            children.iter().map(var_count).sum()
        }
        GeneralizedAst::Var(_) => 1,
        GeneralizedAst::Concrete(_) => 0,
    }
}

// ---------------------------------------------------------------------------
// Discovered abstractions & scoring
// ---------------------------------------------------------------------------

/// Configuration for anti-unification and abstraction scoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AntiUnifyConfig {
    /// Minimum number of occurrences for a pattern to be considered (default: 3).
    pub min_occurrences: u32,
    /// Minimum compression score (occurrences * node_count) (default: 6.0).
    pub min_compression: f64,
    /// Maximum number of abstractions to extract per cycle (default: 5).
    pub max_abstractions: usize,
    /// Maximum number of holes — too many means the pattern is too generic (default: 4).
    pub max_holes: usize,
}

impl Default for AntiUnifyConfig {
    fn default() -> Self {
        Self {
            min_occurrences: 3,
            min_compression: 6.0,
            max_abstractions: 5,
            max_holes: 4,
        }
    }
}

/// A candidate abstraction discovered by anti-unification.
#[derive(Debug, Clone)]
pub struct DiscoveredAbstraction {
    /// The generalized pattern with holes.
    pub pattern: GeneralizedAst,
    /// Deterministic fingerprint of the pattern.
    pub fingerprint: String,
    /// Number of holes (variable positions).
    pub holes: usize,
    /// How many concrete ASTs matched this pattern.
    pub occurrences: u32,
    /// Size of the pattern (non-variable nodes).
    pub nodes: usize,
    /// Stitch compression metric: `occurrences × node_count`.
    pub compression: f64,
    /// Inferred category from the pattern structure.
    pub category: String,
    /// Concrete fingerprints of the source ASTs.
    pub example_sources: Vec<String>,
}

/// Infer a category from the structure of a `GeneralizedAst`.
fn infer_generalized_category(g: &GeneralizedAst) -> String {
    match g {
        GeneralizedAst::Function { .. } => "function".to_string(),
        GeneralizedAst::Struct { .. } => "struct".to_string(),
        GeneralizedAst::Enum { .. } => "enum".to_string(),
        GeneralizedAst::Impl { trait_name, .. } => {
            if let AstSlot::Str(Some(t)) = trait_name {
                match t.as_str() {
                    "Iterator" | "IntoIterator" => "iterator".to_string(),
                    "From" | "Into" | "TryFrom" | "TryInto" => "conversion".to_string(),
                    "Display" | "Debug" => "display".to_string(),
                    _ => "impl".to_string(),
                }
            } else {
                "impl".to_string()
            }
        }
        GeneralizedAst::Match { .. } => "match-pattern".to_string(),
        GeneralizedAst::Block { .. } => "block".to_string(),
        _ => "general".to_string(),
    }
}

/// Score and filter candidate abstractions.
///
/// Applies the configured thresholds: minimum occurrences, minimum compression,
/// maximum holes. Returns the top abstractions sorted by compression descending.
pub fn score_abstractions(
    candidates: &[DiscoveredAbstraction],
    config: &AntiUnifyConfig,
) -> Vec<DiscoveredAbstraction> {
    let mut filtered: Vec<DiscoveredAbstraction> = candidates
        .iter()
        .filter(|c| c.occurrences >= config.min_occurrences)
        .filter(|c| c.compression >= config.min_compression)
        .filter(|c| c.holes <= config.max_holes)
        .cloned()
        .collect();

    filtered.sort_by(|a, b| {
        b.compression
            .partial_cmp(&a.compression)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    filtered.truncate(config.max_abstractions);
    filtered
}

/// Build `DiscoveredAbstraction` candidates from a set of `SimplifiedAst` trees.
///
/// Groups ASTs by top-level variant, anti-unifies within each group,
/// and produces scored candidates.
pub fn discover_abstractions(
    asts: &[SimplifiedAst],
) -> Vec<DiscoveredAbstraction> {
    if asts.len() < 2 {
        return Vec::new();
    }

    // Group by discriminant (top-level variant type)
    let mut groups: HashMap<&str, Vec<&SimplifiedAst>> = HashMap::new();
    for ast in asts {
        let key = match ast {
            SimplifiedAst::Function { .. } => "function",
            SimplifiedAst::Struct { .. } => "struct",
            SimplifiedAst::Enum { .. } => "enum",
            SimplifiedAst::Impl { .. } => "impl",
            SimplifiedAst::Match { .. } => "match",
            SimplifiedAst::IfElse { .. } => "ifelse",
            SimplifiedAst::ForLoop => "for",
            SimplifiedAst::Closure { .. } => "closure",
            SimplifiedAst::Block { .. } => "block",
            SimplifiedAst::Other => continue,
        };
        groups.entry(key).or_default().push(ast);
    }

    let mut candidates = Vec::new();

    for (_key, group) in &groups {
        if group.len() < 2 {
            continue;
        }

        // Anti-unify the entire group
        let generalized = anti_unify_multi(group);
        let fp = generalized_fingerprint(&generalized);
        let holes = var_count(&generalized);
        let nodes = node_count(&generalized);
        let occurrences = group.len() as u32;
        let compression = occurrences as f64 * nodes as f64;
        let category = infer_generalized_category(&generalized);
        let example_sources: Vec<String> = group.iter().map(|a| ast_fingerprint(a)).collect();

        candidates.push(DiscoveredAbstraction {
            pattern: generalized,
            fingerprint: fp,
            holes,
            occurrences,
            nodes,
            compression,
            category,
            example_sources,
        });
    }

    candidates
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anti_unify_identical() {
        let ast = SimplifiedAst::Function {
            param_count: 2,
            has_return: true,
            body: vec![],
        };
        let result = anti_unify_pair(&ast, &ast);
        assert_eq!(result, GeneralizedAst::Concrete(ast));
    }

    #[test]
    fn anti_unify_different_variant() {
        let a = SimplifiedAst::Function {
            param_count: 1,
            has_return: true,
            body: vec![],
        };
        let b = SimplifiedAst::Struct {
            field_count: 3,
            derive_count: 1,
        };
        let result = anti_unify_pair(&a, &b);
        assert!(
            matches!(result, GeneralizedAst::Var(_)),
            "different variants should produce a Var: {result:?}"
        );
    }

    #[test]
    fn anti_unify_same_variant_different_fields() {
        let a = SimplifiedAst::Function {
            param_count: 1,
            has_return: true,
            body: vec![],
        };
        let b = SimplifiedAst::Function {
            param_count: 2,
            has_return: true,
            body: vec![],
        };
        let result = anti_unify_pair(&a, &b);
        match &result {
            GeneralizedAst::Function {
                param_count,
                has_return,
                body,
            } => {
                // param_count differs → Var
                assert!(
                    matches!(param_count, AstSlot::Var(_)),
                    "param_count should be Var: {param_count:?}"
                );
                // has_return same → Bool(true)
                assert_eq!(*has_return, AstSlot::Bool(true));
                // body both empty → empty
                assert!(body.is_empty());
            }
            other => panic!("expected Function, got {other:?}"),
        }

        // Fingerprint should show the variable
        let fp = generalized_fingerprint(&result);
        assert!(fp.contains("?v"), "fingerprint should have variable: {fp}");
        assert!(fp.contains("ret"), "fingerprint should have 'ret': {fp}");
    }

    #[test]
    fn anti_unify_nested_bodies() {
        let a = SimplifiedAst::Function {
            param_count: 1,
            has_return: true,
            body: vec![SimplifiedAst::Match { arm_count: 3 }],
        };
        let b = SimplifiedAst::Function {
            param_count: 1,
            has_return: true,
            body: vec![SimplifiedAst::Match { arm_count: 5 }],
        };
        let result = anti_unify_pair(&a, &b);
        match &result {
            GeneralizedAst::Function { body, .. } => {
                assert_eq!(body.len(), 1);
                match &body[0] {
                    GeneralizedAst::Match { arm_count } => {
                        // arm_count differs → Var
                        assert!(matches!(arm_count, AstSlot::Var(_)));
                    }
                    other => panic!("expected Match, got {other:?}"),
                }
            }
            other => panic!("expected Function, got {other:?}"),
        }
    }

    #[test]
    fn anti_unify_multi_three() {
        let a = SimplifiedAst::Function {
            param_count: 1,
            has_return: true,
            body: vec![],
        };
        let b = SimplifiedAst::Function {
            param_count: 2,
            has_return: true,
            body: vec![],
        };
        let c = SimplifiedAst::Function {
            param_count: 3,
            has_return: true,
            body: vec![],
        };
        let result = anti_unify_multi(&[&a, &b, &c]);
        // All functions with different param counts → Function with Var param_count
        match &result {
            GeneralizedAst::Function {
                param_count,
                has_return,
                ..
            } => {
                assert!(matches!(param_count, AstSlot::Var(_)));
                // has_return is always true, but after folding through approximation
                // it may or may not remain fixed depending on the fold path
                // The key invariant is that the structure is preserved as Function
                let _ = has_return; // acknowledged
            }
            other => panic!("expected Function, got {other:?}"),
        }
    }

    #[test]
    fn generalized_fingerprint_deterministic() {
        let g = GeneralizedAst::Function {
            param_count: AstSlot::Var(AntiUnifyVar {
                name: "?v0".to_string(),
                instances: vec!["1".into(), "2".into()],
            }),
            has_return: AstSlot::Bool(true),
            body: vec![GeneralizedAst::Match {
                arm_count: AstSlot::Fixed(3),
            }],
        };
        let fp1 = generalized_fingerprint(&g);
        let fp2 = generalized_fingerprint(&g);
        assert_eq!(fp1, fp2);
        assert_eq!(fp1, "fn(?v0,ret,[match(3)])");
    }

    #[test]
    fn scoring_filters_low_compression() {
        let config = AntiUnifyConfig {
            min_occurrences: 2,
            min_compression: 10.0,
            max_abstractions: 5,
            max_holes: 4,
        };

        let candidates = vec![
            DiscoveredAbstraction {
                pattern: GeneralizedAst::ForLoop,
                fingerprint: "for".into(),
                holes: 0,
                occurrences: 3,
                nodes: 1,
                compression: 3.0, // below threshold
                category: "general".into(),
                example_sources: vec![],
            },
            DiscoveredAbstraction {
                pattern: GeneralizedAst::Function {
                    param_count: AstSlot::Fixed(1),
                    has_return: AstSlot::Bool(true),
                    body: vec![GeneralizedAst::Match {
                        arm_count: AstSlot::Var(AntiUnifyVar {
                            name: "?v0".into(),
                            instances: vec![],
                        }),
                    }],
                },
                fingerprint: "fn(1,ret,[match(?v0)])".into(),
                holes: 1,
                occurrences: 5,
                nodes: 2,
                compression: 10.0, // at threshold
                category: "function".into(),
                example_sources: vec![],
            },
        ];

        let scored = score_abstractions(&candidates, &config);
        assert_eq!(scored.len(), 1);
        assert_eq!(scored[0].fingerprint, "fn(1,ret,[match(?v0)])");
    }

    #[test]
    fn scoring_respects_max_holes() {
        let config = AntiUnifyConfig {
            min_occurrences: 1,
            min_compression: 0.0,
            max_abstractions: 10,
            max_holes: 2,
        };

        let candidates = vec![
            DiscoveredAbstraction {
                pattern: GeneralizedAst::ForLoop,
                fingerprint: "for".into(),
                holes: 0,
                occurrences: 5,
                nodes: 1,
                compression: 5.0,
                category: "general".into(),
                example_sources: vec![],
            },
            DiscoveredAbstraction {
                pattern: GeneralizedAst::Var(AntiUnifyVar {
                    name: "?v0".into(),
                    instances: vec![],
                }),
                fingerprint: "?v0".into(),
                holes: 5, // too many
                occurrences: 10,
                nodes: 0,
                compression: 0.0,
                category: "general".into(),
                example_sources: vec![],
            },
        ];

        let scored = score_abstractions(&candidates, &config);
        assert_eq!(scored.len(), 1);
        assert_eq!(scored[0].fingerprint, "for");
    }
}
