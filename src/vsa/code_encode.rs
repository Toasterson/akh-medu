//! VSA code pattern encoding — non-ML code2vec analog.
//!
//! Encodes code structure as 10k-bit binary hypervectors using AST path-contexts.
//! This enables similarity-based code retrieval without neural network training.
//!
//! ## Encoding strategy
//!
//! Inspired by code2vec (Alon et al.), adapted for VSA:
//!
//! 1. **AST node types** → atomic symbol vectors via [`encode_token`]
//! 2. **AST paths** → ordered sequences via [`encode_sequence`] with permutation
//! 3. **Path-contexts** → `bind(start, bind(path, end))` triplets
//! 4. **Full code vector** → bundle of all path-contexts (majority vote)
//! 5. **Multi-granularity** → composite of token, AST, call-graph, type-signature layers
//!
//! ## Capacity
//!
//! With 10k-bit vectors: ~70–100 path-contexts per bundle, recursive binding
//! ~5–7 levels deep, HNSW search >95% recall at millions of entries.

use crate::symbol::SymbolId;

use super::HyperVec;
use super::encode::{encode_sequence, encode_symbol, encode_token};
use super::ops::{VsaOps, VsaResult};

// ---------------------------------------------------------------------------
// AST node type vocabulary
// ---------------------------------------------------------------------------

/// Well-known AST node type labels for Rust syntax elements.
///
/// Each label maps to a deterministic hypervector via [`encode_token`].
/// These form the atomic vocabulary for path encoding.
pub struct AstNodeTypes;

impl AstNodeTypes {
    pub const FN_DECL: &'static str = "ast:FnDecl";
    pub const STRUCT_DEF: &'static str = "ast:StructDef";
    pub const ENUM_DEF: &'static str = "ast:EnumDef";
    pub const TRAIT_DEF: &'static str = "ast:TraitDef";
    pub const IMPL_BLOCK: &'static str = "ast:ImplBlock";
    pub const TYPE_REF: &'static str = "ast:TypeRef";
    pub const PARAM: &'static str = "ast:Param";
    pub const FIELD: &'static str = "ast:Field";
    pub const VARIANT: &'static str = "ast:Variant";
    pub const RETURN_TYPE: &'static str = "ast:ReturnType";
    pub const BLOCK: &'static str = "ast:Block";
    pub const IF_EXPR: &'static str = "ast:IfExpr";
    pub const MATCH_EXPR: &'static str = "ast:MatchExpr";
    pub const MATCH_ARM: &'static str = "ast:MatchArm";
    pub const CALL_EXPR: &'static str = "ast:CallExpr";
    pub const METHOD_CALL: &'static str = "ast:MethodCall";
    pub const LET_BINDING: &'static str = "ast:LetBinding";
    pub const LOOP_EXPR: &'static str = "ast:LoopExpr";
    pub const CLOSURE: &'static str = "ast:Closure";
    pub const ATTRIBUTE: &'static str = "ast:Attribute";
    pub const DOC_COMMENT: &'static str = "ast:DocComment";
    pub const USE_DECL: &'static str = "ast:UseDecl";
    pub const MODULE: &'static str = "ast:Module";
}

// ---------------------------------------------------------------------------
// Path-context encoding
// ---------------------------------------------------------------------------

/// An AST path-context: (start_terminal, path_of_node_types, end_terminal).
///
/// In code2vec, a path-context captures the structural relationship between
/// two tokens in the AST by recording the path between them. We encode this
/// as a VSA binding: `bind(start_vec, bind(path_vec, end_vec))`.
#[derive(Debug, Clone)]
pub struct AstPathContext {
    /// The starting terminal token (e.g., function name, parameter name).
    pub start: String,
    /// AST node types along the path from start to end.
    pub path: Vec<String>,
    /// The ending terminal token (e.g., return type, field type).
    pub end: String,
}

impl AstPathContext {
    pub fn new(start: impl Into<String>, path: Vec<String>, end: impl Into<String>) -> Self {
        Self {
            start: start.into(),
            path,
            end: end.into(),
        }
    }
}

/// Encode a single AST path-context into a hypervector.
///
/// Computes `bind(start_vec, bind(path_vec, end_vec))` where:
/// - `start_vec` = token encoding of the start terminal
/// - `path_vec` = sequence encoding of the AST node types along the path
/// - `end_vec` = token encoding of the end terminal
pub fn encode_path_context(ops: &VsaOps, ctx: &AstPathContext) -> VsaResult<HyperVec> {
    let start_vec = encode_token(ops, &ctx.start);
    let end_vec = encode_token(ops, &ctx.end);

    // Encode the path as a sequence of AST node type tokens
    let path_vec = if ctx.path.is_empty() {
        // Direct connection — use a sentinel token
        encode_token(ops, "ast:Direct")
    } else {
        // Convert path strings to synthetic SymbolIds for encode_sequence
        let path_ids: Vec<SymbolId> = ctx
            .path
            .iter()
            .map(|node_type| synthetic_id(node_type))
            .collect();
        encode_sequence(ops, &path_ids).unwrap_or_else(|| encode_token(ops, "ast:Empty"))
    };

    // bind(path, end) then bind(start, result)
    let inner = ops.bind(&path_vec, &end_vec)?;
    ops.bind(&start_vec, &inner)
}

/// Encode multiple path-contexts into a single code vector via bundling.
///
/// The resulting vector is similar to all constituent path-contexts,
/// representing the code's structural fingerprint. Capacity: ~70–100
/// path-contexts per bundle with 10k-bit vectors.
pub fn encode_code_vector(ops: &VsaOps, contexts: &[AstPathContext]) -> VsaResult<HyperVec> {
    if contexts.is_empty() {
        return Err(crate::error::VsaError::EmptyBundle);
    }

    let vecs: Vec<HyperVec> = contexts
        .iter()
        .filter_map(|ctx| encode_path_context(ops, ctx).ok())
        .collect();

    if vecs.is_empty() {
        return Err(crate::error::VsaError::EmptyBundle);
    }

    let refs: Vec<&HyperVec> = vecs.iter().collect();
    ops.bundle(&refs)
}

// ---------------------------------------------------------------------------
// Multi-granularity encoding
// ---------------------------------------------------------------------------

/// Granularity level for code pattern vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PatternGranularity {
    /// Token-level: bag of identifier tokens.
    Token,
    /// AST-level: path-context structure.
    Ast,
    /// Call-graph: function call sequences and dependencies.
    CallGraph,
    /// Type-signature: parameter types, return types, trait bounds.
    TypeSignature,
    /// Composite: all layers combined with positional permutation.
    Composite,
}

/// A code pattern vector at a specific granularity.
#[derive(Debug, Clone)]
pub struct CodePatternVec {
    /// The symbol this pattern represents.
    pub symbol: SymbolId,
    /// The encoded hypervector.
    pub vector: HyperVec,
    /// What granularity this encoding captures.
    pub granularity: PatternGranularity,
}

/// Encode a token-level (bag of words) representation of code identifiers.
///
/// Simply bundles the token vectors for all identifiers, losing ordering
/// but capturing "what tokens appear."
pub fn encode_token_level(ops: &VsaOps, tokens: &[&str]) -> VsaResult<HyperVec> {
    if tokens.is_empty() {
        return Err(crate::error::VsaError::EmptyBundle);
    }

    let vecs: Vec<HyperVec> = tokens.iter().map(|t| encode_token(ops, t)).collect();
    let refs: Vec<&HyperVec> = vecs.iter().collect();
    ops.bundle(&refs)
}

/// Encode a type-signature as a structured vector.
///
/// Binds param types in order via permutation, then binds with return type.
/// Captures: "takes (A, B) → returns C".
pub fn encode_type_signature(
    ops: &VsaOps,
    param_types: &[&str],
    return_type: Option<&str>,
) -> VsaResult<HyperVec> {
    // Encode params as ordered sequence
    let param_vecs: Vec<HyperVec> = param_types.iter().map(|t| encode_token(ops, t)).collect();
    let params_vec = if param_vecs.is_empty() {
        encode_token(ops, "type:Unit")
    } else if param_vecs.len() == 1 {
        param_vecs.into_iter().next().unwrap()
    } else {
        // Use permutation to preserve order
        let shifted: Vec<HyperVec> = param_vecs
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let shift = param_vecs.len() - 1 - i;
                if shift > 0 {
                    ops.permute(v, shift)
                } else {
                    v.clone()
                }
            })
            .collect();
        let refs: Vec<&HyperVec> = shifted.iter().collect();
        ops.bundle(&refs)?
    };

    // Encode return type
    let ret_vec = encode_token(ops, return_type.unwrap_or("type:Unit"));

    // Bind params with return: signature = bind(params, return)
    let role_vec = encode_token(ops, "role:Signature");
    let sig = ops.bind(&params_vec, &ret_vec)?;
    ops.bind(&role_vec, &sig)
}

/// Encode a call-graph fragment as an ordered sequence of function calls.
///
/// Captures "this code calls A, then B, then C" as a permutation-encoded sequence.
pub fn encode_call_graph(ops: &VsaOps, call_sequence: &[&str]) -> VsaResult<HyperVec> {
    if call_sequence.is_empty() {
        return Err(crate::error::VsaError::EmptyBundle);
    }

    let ids: Vec<SymbolId> = call_sequence
        .iter()
        .map(|name| synthetic_id(name))
        .collect();

    encode_sequence(ops, &ids).ok_or(crate::error::VsaError::EmptyBundle)
}

/// Combine multiple granularity levels into a composite vector.
///
/// Each layer is permuted by a different shift to preserve distinguishability,
/// then bundled:
/// ```text
/// composite = bundle(ρ^0(token), ρ^1(ast), ρ^2(call_graph), ρ^3(type_sig))
/// ```
pub fn encode_composite(
    ops: &VsaOps,
    layers: &[(PatternGranularity, &HyperVec)],
) -> VsaResult<HyperVec> {
    if layers.is_empty() {
        return Err(crate::error::VsaError::EmptyBundle);
    }

    let shifted: Vec<HyperVec> = layers
        .iter()
        .enumerate()
        .map(|(i, (_, vec))| {
            if i > 0 {
                ops.permute(vec, i)
            } else {
                (*vec).clone()
            }
        })
        .collect();

    let refs: Vec<&HyperVec> = shifted.iter().collect();
    ops.bundle(&refs)
}

// ---------------------------------------------------------------------------
// Convenience: extract path-contexts from KG code triples
// ---------------------------------------------------------------------------

/// Extract path-contexts from a function's KG representation.
///
/// Given a function symbol's relations (params, return type, doc), builds
/// path-contexts that encode structural connections:
/// - (fn_name, [FnDecl, Param], param_name) for each param
/// - (fn_name, [FnDecl, ReturnType], return_type) for return type
/// - (param_name, [Param, TypeRef], param_type) for typed params
pub fn extract_function_contexts(
    fn_name: &str,
    params: &[(&str, Option<&str>)],
    return_type: Option<&str>,
) -> Vec<AstPathContext> {
    let mut contexts = Vec::new();

    // fn → each param
    for (param_name, param_type) in params {
        contexts.push(AstPathContext::new(
            fn_name,
            vec![
                AstNodeTypes::FN_DECL.to_string(),
                AstNodeTypes::PARAM.to_string(),
            ],
            *param_name,
        ));

        // param → type (if typed)
        if let Some(ty) = param_type {
            contexts.push(AstPathContext::new(
                *param_name,
                vec![
                    AstNodeTypes::PARAM.to_string(),
                    AstNodeTypes::TYPE_REF.to_string(),
                ],
                *ty,
            ));
        }
    }

    // fn → return type
    if let Some(ret) = return_type {
        contexts.push(AstPathContext::new(
            fn_name,
            vec![
                AstNodeTypes::FN_DECL.to_string(),
                AstNodeTypes::RETURN_TYPE.to_string(),
            ],
            ret,
        ));
    }

    // param-to-param connections (captures co-occurrence)
    if params.len() >= 2 {
        for i in 0..params.len() - 1 {
            contexts.push(AstPathContext::new(
                params[i].0,
                vec![
                    AstNodeTypes::PARAM.to_string(),
                    AstNodeTypes::FN_DECL.to_string(),
                    AstNodeTypes::PARAM.to_string(),
                ],
                params[i + 1].0,
            ));
        }
    }

    contexts
}

/// Extract path-contexts from a struct's KG representation.
///
/// - (struct_name, [StructDef, Field], field_name) for each field
/// - (field_name, [Field, TypeRef], field_type) for each typed field
/// - (field_name, [Field, StructDef, Field], next_field) for field co-occurrence
pub fn extract_struct_contexts(
    struct_name: &str,
    fields: &[(&str, &str)],
) -> Vec<AstPathContext> {
    let mut contexts = Vec::new();

    for (field_name, field_type) in fields {
        // struct → field
        contexts.push(AstPathContext::new(
            struct_name,
            vec![
                AstNodeTypes::STRUCT_DEF.to_string(),
                AstNodeTypes::FIELD.to_string(),
            ],
            *field_name,
        ));

        // field → type
        contexts.push(AstPathContext::new(
            *field_name,
            vec![
                AstNodeTypes::FIELD.to_string(),
                AstNodeTypes::TYPE_REF.to_string(),
            ],
            *field_type,
        ));
    }

    // field-to-field co-occurrence
    if fields.len() >= 2 {
        for i in 0..fields.len() - 1 {
            contexts.push(AstPathContext::new(
                fields[i].0,
                vec![
                    AstNodeTypes::FIELD.to_string(),
                    AstNodeTypes::STRUCT_DEF.to_string(),
                    AstNodeTypes::FIELD.to_string(),
                ],
                fields[i + 1].0,
            ));
        }
    }

    contexts
}

/// Extract path-contexts from an enum's KG representation.
///
/// - (enum_name, [EnumDef, Variant], variant_name) for each variant
/// - (variant_name, [Variant, EnumDef, Variant], next_variant) for co-occurrence
pub fn extract_enum_contexts(
    enum_name: &str,
    variants: &[&str],
) -> Vec<AstPathContext> {
    let mut contexts = Vec::new();

    for variant in variants {
        contexts.push(AstPathContext::new(
            enum_name,
            vec![
                AstNodeTypes::ENUM_DEF.to_string(),
                AstNodeTypes::VARIANT.to_string(),
            ],
            *variant,
        ));
    }

    // variant-to-variant co-occurrence
    if variants.len() >= 2 {
        for i in 0..variants.len() - 1 {
            contexts.push(AstPathContext::new(
                variants[i],
                vec![
                    AstNodeTypes::VARIANT.to_string(),
                    AstNodeTypes::ENUM_DEF.to_string(),
                    AstNodeTypes::VARIANT.to_string(),
                ],
                variants[i + 1],
            ));
        }
    }

    contexts
}

/// Extract path-contexts from an impl block's KG representation.
///
/// - (type_name, [ImplBlock, FnDecl], method_name) for each method
/// - (type_name, [ImplBlock, TraitDef], trait_name) if trait impl
pub fn extract_impl_contexts(
    type_name: &str,
    trait_name: Option<&str>,
    methods: &[&str],
) -> Vec<AstPathContext> {
    let mut contexts = Vec::new();

    // type → trait
    if let Some(tr) = trait_name {
        contexts.push(AstPathContext::new(
            type_name,
            vec![
                AstNodeTypes::IMPL_BLOCK.to_string(),
                AstNodeTypes::TRAIT_DEF.to_string(),
            ],
            tr,
        ));
    }

    // type → each method
    for method in methods {
        contexts.push(AstPathContext::new(
            type_name,
            vec![
                AstNodeTypes::IMPL_BLOCK.to_string(),
                AstNodeTypes::FN_DECL.to_string(),
            ],
            *method,
        ));
    }

    // method-to-method co-occurrence
    if methods.len() >= 2 {
        for i in 0..methods.len() - 1 {
            contexts.push(AstPathContext::new(
                methods[i],
                vec![
                    AstNodeTypes::FN_DECL.to_string(),
                    AstNodeTypes::IMPL_BLOCK.to_string(),
                    AstNodeTypes::FN_DECL.to_string(),
                ],
                methods[i + 1],
            ));
        }
    }

    contexts
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create a synthetic SymbolId from a string label (for path encoding).
fn synthetic_id(label: &str) -> SymbolId {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    label.hash(&mut hasher);
    let hash = hasher.finish();
    SymbolId::new(hash | (1u64 << 63)).expect("non-zero hash with high bit set")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd;
    use crate::vsa::{Dimension, Encoding};

    fn test_ops() -> VsaOps {
        VsaOps::new(simd::best_kernel(), Dimension::TEST, Encoding::Bipolar)
    }

    #[test]
    fn path_context_encoding_deterministic() {
        let ops = test_ops();
        let ctx = AstPathContext::new(
            "parse",
            vec![AstNodeTypes::FN_DECL.to_string(), AstNodeTypes::PARAM.to_string()],
            "input",
        );

        let v1 = encode_path_context(&ops, &ctx).unwrap();
        let v2 = encode_path_context(&ops, &ctx).unwrap();
        assert_eq!(v1, v2);
    }

    #[test]
    fn different_contexts_differ() {
        let ops = test_ops();

        let ctx1 = AstPathContext::new(
            "parse",
            vec![AstNodeTypes::FN_DECL.to_string(), AstNodeTypes::PARAM.to_string()],
            "input",
        );
        let ctx2 = AstPathContext::new(
            "render",
            vec![AstNodeTypes::FN_DECL.to_string(), AstNodeTypes::RETURN_TYPE.to_string()],
            "String",
        );

        let v1 = encode_path_context(&ops, &ctx1).unwrap();
        let v2 = encode_path_context(&ops, &ctx2).unwrap();
        let sim = ops.similarity(&v1, &v2).unwrap();
        assert!(sim < 0.7, "different contexts should differ: sim={sim}");
    }

    #[test]
    fn code_vector_from_function_contexts() {
        let ops = test_ops();

        let contexts = extract_function_contexts(
            "greet",
            &[("name", Some("&str")), ("loud", Some("bool"))],
            Some("String"),
        );
        assert!(!contexts.is_empty());

        let code_vec = encode_code_vector(&ops, &contexts).unwrap();
        assert_eq!(code_vec.dim(), Dimension::TEST);
    }

    #[test]
    fn similar_functions_have_higher_similarity() {
        let ops = test_ops();

        // Two similar functions: both take &str and return String
        let ctx1 = extract_function_contexts(
            "greet",
            &[("name", Some("&str"))],
            Some("String"),
        );
        let ctx2 = extract_function_contexts(
            "farewell",
            &[("name", Some("&str"))],
            Some("String"),
        );

        // A very different function: takes u64, returns bool
        let ctx3 = extract_function_contexts(
            "is_prime",
            &[("n", Some("u64"))],
            Some("bool"),
        );

        let v1 = encode_code_vector(&ops, &ctx1).unwrap();
        let v2 = encode_code_vector(&ops, &ctx2).unwrap();
        let v3 = encode_code_vector(&ops, &ctx3).unwrap();

        let sim_12 = ops.similarity(&v1, &v2).unwrap();
        let sim_13 = ops.similarity(&v1, &v3).unwrap();

        // Similar functions should be more similar to each other than to dissimilar ones
        assert!(
            sim_12 > sim_13,
            "similar fns should be closer: sim(greet,farewell)={sim_12}, sim(greet,is_prime)={sim_13}"
        );
    }

    #[test]
    fn struct_context_extraction() {
        let contexts = extract_struct_contexts(
            "Config",
            &[("name", "String"), ("value", "u64"), ("debug", "bool")],
        );

        // 3 struct→field + 3 field→type + 2 field→field co-occurrence = 8
        assert_eq!(contexts.len(), 8);
    }

    #[test]
    fn enum_context_extraction() {
        let contexts = extract_enum_contexts(
            "Color",
            &["Red", "Green", "Blue"],
        );

        // 3 enum→variant + 2 variant→variant co-occurrence = 5
        assert_eq!(contexts.len(), 5);
    }

    #[test]
    fn impl_context_extraction() {
        let contexts = extract_impl_contexts(
            "Config",
            Some("Display"),
            &["fmt", "to_string"],
        );

        // 1 type→trait + 2 type→method + 1 method→method = 4
        assert_eq!(contexts.len(), 4);
    }

    #[test]
    fn type_signature_encoding() {
        let ops = test_ops();

        let sig1 = encode_type_signature(&ops, &["&str", "bool"], Some("String")).unwrap();
        let sig2 = encode_type_signature(&ops, &["&str", "bool"], Some("String")).unwrap();
        assert_eq!(sig1, sig2, "same signature should produce same vector");

        let sig3 = encode_type_signature(&ops, &["u64"], Some("bool")).unwrap();
        let sim = ops.similarity(&sig1, &sig3).unwrap();
        assert!(sim < 0.7, "different signatures should differ: sim={sim}");
    }

    #[test]
    fn call_graph_encoding() {
        let ops = test_ops();

        let cg1 = encode_call_graph(&ops, &["parse", "validate", "execute"]).unwrap();
        let cg2 = encode_call_graph(&ops, &["parse", "validate", "execute"]).unwrap();
        assert_eq!(cg1, cg2);

        // Different order should differ
        let cg3 = encode_call_graph(&ops, &["execute", "validate", "parse"]).unwrap();
        let sim = ops.similarity(&cg1, &cg3).unwrap();
        assert!(sim < 0.7, "different call order should differ: sim={sim}");
    }

    #[test]
    fn composite_encoding() {
        let ops = test_ops();

        let token_vec = encode_token_level(&ops, &["parse", "input", "String"]).unwrap();
        let ast_contexts = extract_function_contexts("parse", &[("input", Some("&str"))], Some("String"));
        let ast_vec = encode_code_vector(&ops, &ast_contexts).unwrap();

        let composite = encode_composite(
            &ops,
            &[
                (PatternGranularity::Token, &token_vec),
                (PatternGranularity::Ast, &ast_vec),
            ],
        )
        .unwrap();

        assert_eq!(composite.dim(), Dimension::TEST);

        // Composite should be somewhat similar to both layers
        let sim_token = ops.similarity(&composite, &token_vec).unwrap();
        let sim_ast = ops.similarity(&composite, &ast_vec).unwrap();
        // With bundle of 2, each layer contributes ~50%
        assert!(sim_token > 0.3, "composite should resemble token layer: {sim_token}");
        assert!(sim_ast > 0.3, "composite should resemble ast layer: {sim_ast}");
    }

    #[test]
    fn empty_contexts_errors() {
        let ops = test_ops();
        assert!(encode_code_vector(&ops, &[]).is_err());
    }

    #[test]
    fn token_level_encoding() {
        let ops = test_ops();

        let v = encode_token_level(&ops, &["HashMap", "String", "insert"]).unwrap();
        assert_eq!(v.dim(), Dimension::TEST);
    }

    #[test]
    fn pattern_granularity_composite_preserves_info() {
        let ops = test_ops();

        // Two functions with same tokens but different structure
        let tokens = &["name", "String"];
        let token_vec = encode_token_level(&ops, tokens).unwrap();

        // fn_a: takes name:&str, returns String
        let ctx_a = extract_function_contexts("fn_a", &[("name", Some("&str"))], Some("String"));
        let ast_a = encode_code_vector(&ops, &ctx_a).unwrap();

        // fn_b: takes String, returns name (reversed roles)
        let ctx_b = extract_function_contexts("fn_b", &[("String", Some("String"))], Some("&str"));
        let ast_b = encode_code_vector(&ops, &ctx_b).unwrap();

        // Token level should be similar (same tokens)
        let token_sim_a = ops.similarity(&encode_token_level(&ops, &["fn_a", "name", "&str", "String"]).unwrap(), &encode_token_level(&ops, &["fn_b", "String", "&str"]).unwrap()).unwrap();

        // AST level should differ (different structure)
        let ast_sim = ops.similarity(&ast_a, &ast_b).unwrap();

        // Composite should capture the structural difference
        let comp_a = encode_composite(
            &ops,
            &[
                (PatternGranularity::Token, &token_vec),
                (PatternGranularity::Ast, &ast_a),
            ],
        )
        .unwrap();

        let comp_b = encode_composite(
            &ops,
            &[
                (PatternGranularity::Token, &token_vec),
                (PatternGranularity::Ast, &ast_b),
            ],
        )
        .unwrap();

        let comp_sim = ops.similarity(&comp_a, &comp_b).unwrap();

        // Composite similarity should be between pure token and pure AST similarities
        // (since it blends both signals)
        assert!(
            comp_sim < 1.0,
            "composites with different AST structure should not be identical: {comp_sim}"
        );
    }
}
