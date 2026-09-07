//! Greedy autoregressive generation from a trained [`TinyTransformer`].
//!
//! Runs on a plain (non-autodiff) backend -- callers pass the result of
//! `model.valid()` (see [`crate::train::to_eval_model`]) so generation does
//! not build a computation graph it will never call `.backward()` on.

use burn::tensor::backend::Backend;
use burn::tensor::{ElementConversion, Int, Tensor, TensorData};
use grug_data::constrain::{
    ConstrainedDecodeState, CopyConstraints, DecodeReport, GrammarState, RuntimeCallGrammar,
    ToolCallGrammar, build_decode_report,
};

use crate::pointer::PointerTransformer;
use crate::tokenizer::Vocab;
use crate::transformer::TinyTransformer;

/// Generation options.
#[derive(Debug, Clone, Copy)]
pub struct GenerateConfig {
    /// Hard stop on total generated tokens (beyond the prompt), even if
    /// `EOS` is never produced -- guards against runaway generation from an
    /// undertrained model.
    pub max_new_tokens: usize,
    /// Must match (or be <=) the model's configured `max_seq_len`; sequences
    /// longer than this are truncated from the left (oldest tokens dropped)
    /// before each forward pass.
    pub max_seq_len: usize,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 128,
            max_seq_len: 256,
        }
    }
}

/// Greedily generate a tool-call string from `prompt`, using the same
/// `BOS <prompt> SEP` framing [`crate::train::encode_example`] trains on.
///
/// Stops at `EOS`, at `max_new_tokens`, or (defensively) if the running
/// sequence would exceed `max_seq_len` even after left-truncation of prompt
/// tokens -- returns whatever was generated so far in every case. Decoding
/// goes through `tokenizer.decode`, which strips special tokens; if decoding
/// fails outright (e.g. [`crate::tokenizer::ByteTokenizer`] hitting a
/// mid-codepoint stop), the raw generated ids are decoded lossily as UTF-8
/// bytes instead of erroring, since the eval harness needs a string back
/// either way -- an invalid-JSON string still scores as "invalid json",
/// which is a real and useful eval outcome.
pub fn generate<B: Backend, V: Vocab>(
    model: &TinyTransformer<B>,
    tokenizer: &V,
    prompt: &str,
    config: GenerateConfig,
    device: &B::Device,
) -> String {
    let prompt_ids = tokenizer.encode(prompt).unwrap_or_default();
    generate_from_prompt_ids(model, tokenizer, &prompt_ids, config, device)
}

/// Greedily generate from an already-tokenized prompt body.
///
/// This is the lower-level seam used by knowledge-workflow inference, where a
/// long prompt is compacted by retaining both its tool-policy head and latest
/// conversation tail before generation. Callers provide neither BOS nor SEP;
/// this function adds the same framing as [`generate`].
pub fn generate_from_prompt_ids<B: Backend, V: Vocab>(
    model: &TinyTransformer<B>,
    tokenizer: &V,
    prompt_ids: &[u32],
    config: GenerateConfig,
    device: &B::Device,
) -> String {
    let mut ids = Vec::with_capacity(prompt_ids.len() + config.max_new_tokens + 2);
    ids.push(tokenizer.bos_id());
    ids.extend_from_slice(prompt_ids);
    ids.push(tokenizer.sep_id());

    let mut generated: Vec<u32> = Vec::new();

    for _ in 0..config.max_new_tokens {
        let mut window = ids.clone();
        if window.len() > config.max_seq_len {
            let drop = window.len() - config.max_seq_len;
            window.drain(0..drop);
        }

        let seq_len = window.len();
        let input_flat: Vec<i64> = window.iter().map(|&t| t as i64).collect();
        let input =
            Tensor::<B, 2, Int>::from_data(TensorData::new(input_flat, [1, seq_len]), device);

        let logits = model.forward(input); // [1, seq_len, vocab]
        let vocab_size = logits.dims()[2];
        let last_logits = logits
            .slice([0..1, seq_len - 1..seq_len, 0..vocab_size])
            .reshape([vocab_size]);

        let next_id = argmax(last_logits);

        if next_id == tokenizer.eos_id() {
            break;
        }
        ids.push(next_id);
        generated.push(next_id);
    }

    tokenizer.decode(&generated).unwrap_or_else(|_| {
        // Fallback for tokenizers (e.g. the byte tokenizer) whose `decode`
        // can fail on a truncated multi-byte codepoint -- lossily decode the
        // byte-range tokens directly rather than surfacing an error from a
        // generation call, per this function's documented contract.
        let bytes: Vec<u8> = generated
            .iter()
            .copied()
            .filter(|&t| t < 256)
            .map(|t| t as u8)
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    })
}

/// Greedy generation with a per-layer key/value cache.
///
/// When the framed prompt plus the requested completion fits in the declared
/// context, this performs one full prompt prefill and then one-token decoder
/// steps. If a generic caller has not reserved completion room, it falls back
/// to [`generate_from_prompt_ids`], whose left-truncation semantics remain the
/// compatibility path. Knowledge-runtime callers compact before entering this
/// function, so normal Akh serving and checkpoint exams use the cached path.
pub fn generate_from_prompt_ids_cached<B: Backend, V: Vocab>(
    model: &TinyTransformer<B>,
    tokenizer: &V,
    prompt_ids: &[u32],
    config: GenerateConfig,
    device: &B::Device,
) -> String {
    let mut framed = Vec::with_capacity(prompt_ids.len() + 2);
    framed.push(tokenizer.bos_id());
    framed.extend_from_slice(prompt_ids);
    framed.push(tokenizer.sep_id());
    if framed.len().saturating_add(config.max_new_tokens) > config.max_seq_len {
        return generate_from_prompt_ids(model, tokenizer, prompt_ids, config, device);
    }

    let input = Tensor::<B, 2, Int>::from_data(
        TensorData::new(
            framed.iter().map(|token| i64::from(*token)).collect(),
            [1, framed.len()],
        ),
        device,
    );
    let (mut logits, mut cache) = model.prefill_cached(input);
    let mut generated = Vec::with_capacity(config.max_new_tokens);
    for index in 0..config.max_new_tokens {
        let [_, seq_len, vocab_size] = logits.dims();
        let last_logits = logits
            .clone()
            .slice([0..1, seq_len - 1..seq_len, 0..vocab_size])
            .reshape([vocab_size]);
        let next_id = argmax(last_logits);
        if next_id == tokenizer.eos_id() {
            break;
        }
        generated.push(next_id);
        if index + 1 < config.max_new_tokens {
            let input = Tensor::<B, 2, Int>::from_data(
                TensorData::new(vec![i64::from(next_id)], [1, 1]),
                device,
            );
            (logits, cache) = model.decode_cached(input, cache);
        }
    }
    tokenizer.decode(&generated).unwrap_or_else(|_| {
        let bytes: Vec<u8> = generated
            .iter()
            .copied()
            .filter(|&token| token < 256)
            .map(|token| token as u8)
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    })
}

/// What governs value-copy-constrained decoding for one generation call:
/// the runtime grammar (`None` when no tools were offered) and the exact
/// context text it scans for copy candidates. Bundled into one type rather
/// than several more parameters on
/// [`generate_from_prompt_ids_cached_constrained`] because they are always
/// supplied (or omitted) together.
#[derive(Debug, Clone, Copy)]
pub struct ConstrainedDecode<'a> {
    pub grammar: Option<&'a RuntimeCallGrammar>,
    pub context: &'a str,
    /// Phase 51 / ADR 104 D2 mechanism 1 (identifier hard-copy lock) on/off
    /// switch. Default `true`.
    pub identifier_copy: bool,
    /// Phase 51 / ADR 104 D2 mechanism 2 (soft suffix-copy bias) margin, in
    /// nats. `None` disables the bias entirely. Default `Some(4.0)`.
    pub copy_bias_margin: Option<f32>,
}

impl<'a> ConstrainedDecode<'a> {
    /// The runtime's default value-copy configuration: both D2 mechanisms
    /// on, D1's schema-pattern/enum copy always active regardless (it needs
    /// no toggle -- it only ever fires when the schema itself declares a
    /// `pattern`/`enum`).
    pub fn new(grammar: Option<&'a RuntimeCallGrammar>, context: &'a str) -> Self {
        Self {
            grammar,
            context,
            identifier_copy: true,
            copy_bias_margin: Some(4.0),
        }
    }
}

/// KV-cached generation under the runtime's `<tool_call>` envelope grammar,
/// with byte-level value-copy constraints applied to identifier-shaped string
/// arguments (Phase 51 / ADR 104 D1).
///
/// Unlike [`generate_constrained`] (which constrains the isolated, untagged
/// procedural-catalog surface for the Step A/B research harness), this is the
/// grammar the deployed runtime and its evaluators actually decode under:
/// Akh-Medu's `<tool_call>\n{...}\n</tool_call>` envelope, or unconstrained
/// free text for an `Answer`/`Knowledge gap:` turn (see
/// [`grug_data::constrain::RuntimeCallState`] for why free text is never
/// blocked).
///
/// `decode.context` is the exact text the model can see (the rendered, and
/// possibly left-compacted, prompt) — it is scanned once up front for
/// value-copy candidates ([`grug_data::constrain::ValueConstraintTable::build`]),
/// not re-derived per step. `decode.grammar` is `None` when no tools were
/// offered (a pure-answer turn), in which case this degrades byte-for-byte to
/// [`generate_from_prompt_ids_cached`] and reports `constrained: false`.
///
/// Returns the decoded text plus a [`DecodeReport`] describing which
/// (tool, property) pairs were value-copy-constrained and whether the
/// emitted value actually landed on a context candidate.
pub fn generate_from_prompt_ids_cached_constrained<B: Backend, V: Vocab>(
    model: &TinyTransformer<B>,
    tokenizer: &V,
    prompt_ids: &[u32],
    decode: ConstrainedDecode<'_>,
    pieces: &[Option<Vec<u8>>],
    config: GenerateConfig,
    device: &B::Device,
) -> (String, DecodeReport) {
    let ConstrainedDecode {
        grammar,
        context,
        identifier_copy,
        copy_bias_margin,
    } = decode;
    let Some(grammar) = grammar else {
        let text = generate_from_prompt_ids_cached(model, tokenizer, prompt_ids, config, device);
        return (text, DecodeReport::default());
    };

    let mut framed = Vec::with_capacity(prompt_ids.len() + 2);
    framed.push(tokenizer.bos_id());
    framed.extend_from_slice(prompt_ids);
    framed.push(tokenizer.sep_id());
    if framed.len().saturating_add(config.max_new_tokens) > config.max_seq_len {
        // Compatibility fallback, matching `generate_from_prompt_ids_cached`'s
        // own overflow behavior -- normal serving always compacts the prompt
        // to fit, so this path should not be exercised in practice.
        let text = generate_from_prompt_ids(model, tokenizer, prompt_ids, config, device);
        return (text, DecodeReport::default());
    }

    let copy = CopyConstraints::build(grammar.inner(), context, identifier_copy, copy_bias_margin);
    let eos_id = tokenizer.eos_id();

    let input = Tensor::<B, 2, Int>::from_data(
        TensorData::new(
            framed.iter().map(|token| i64::from(*token)).collect(),
            [1, framed.len()],
        ),
        device,
    );
    let (mut logits, mut cache) = model.prefill_cached(input);
    let mut generated: Vec<u32> = Vec::with_capacity(config.max_new_tokens);
    let mut state = ConstrainedDecodeState::start(grammar);
    let mut identifier_copy_locks = 0u32;
    let mut identifier_copy_releases = 0u32;
    let mut copy_bias_overrides = 0u32;

    for index in 0..config.max_new_tokens {
        let [_, seq_len, vocab_size] = logits.dims();
        let last_logits = logits
            .clone()
            .slice([0..1, seq_len - 1..seq_len, 0..vocab_size])
            .reshape([vocab_size]);

        // Pass 1: the identifier lock (if engaged) is enforced strictly by
        // `state.advance` (Phase 51 / ADR 104 D2 mechanism 1 -- see
        // `ConstrainedDecodeState::step_copy`). If that leaves no legal
        // continuation at all, this is the documented deadlock case: force
        // the lock released for the rest of the value and retry the same
        // step (never emit an illegal byte, never hang).
        let (mut mask, mut any_allowed) =
            build_copy_mask(&state, grammar, &copy, pieces, vocab_size, eos_id);
        let mut active_state = state.clone();
        if !any_allowed && state.identifier_lock_engaged() {
            active_state = state.with_identifier_lock_released();
            identifier_copy_releases += 1;
            (mask, any_allowed) =
                build_copy_mask(&active_state, grammar, &copy, pieces, vocab_size, eos_id);
        }
        if !any_allowed {
            // Dead end -- should not happen (every reachable state has a
            // legal continuation), but stop rather than emit an illegal byte.
            break;
        }

        // Phase 51 / ADR 104 D2 mechanism 2: inside a string value, prefer a
        // near-argmax, already-legal token that continues a verbatim context
        // match over one that doesn't. Falls back to the exact pre-existing
        // tensor argmax path whenever not inside a string (the overwhelming
        // majority of steps: structural bytes, keys, the tool name), so that
        // path stays byte-for-byte unchanged.
        let next_id = match (copy_bias_margin, active_state.active_string_content()) {
            (Some(margin), Some(content)) => select_biased_next_id(
                &last_logits,
                &mask,
                pieces,
                &copy,
                content,
                margin,
                &mut copy_bias_overrides,
            )
            .unwrap_or_else(|| {
                let mask_tensor =
                    Tensor::<B, 1>::from_data(TensorData::new(mask.clone(), [vocab_size]), device);
                argmax(last_logits.clone() + mask_tensor)
            }),
            _ => {
                let mask_tensor =
                    Tensor::<B, 1>::from_data(TensorData::new(mask, [vocab_size]), device);
                argmax(last_logits + mask_tensor)
            }
        };

        if next_id == eos_id {
            break;
        }

        let Some(bytes) = pieces.get(next_id as usize).and_then(|p| p.as_ref()) else {
            break;
        };
        let was_locked = active_state.identifier_lock_engaged();
        match active_state.advance(grammar, &copy, bytes) {
            Some(next_state) => {
                if !was_locked && next_state.identifier_lock_engaged() {
                    identifier_copy_locks += 1;
                }
                state = next_state;
            }
            None => break, // was checked allowed above; defensive only
        }
        generated.push(next_id);

        if index + 1 < config.max_new_tokens {
            let input = Tensor::<B, 2, Int>::from_data(
                TensorData::new(vec![i64::from(next_id)], [1, 1]),
                device,
            );
            (logits, cache) = model.decode_cached(input, cache);
        }
    }

    let text = tokenizer.decode(&generated).unwrap_or_else(|_| {
        let bytes: Vec<u8> = generated
            .iter()
            .copied()
            .filter(|&token| token < 256)
            .map(|token| token as u8)
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    });
    let mut report = build_decode_report(grammar.inner(), &copy.table, &text);
    report.identifier_copy_locks = identifier_copy_locks;
    report.identifier_copy_releases = identifier_copy_releases;
    report.copy_bias_overrides = copy_bias_overrides;
    (text, report)
}

/// Build the additive legality mask for one decode step from `state`: `0.0`
/// for every vocabulary id whose piece bytes keep `state` on a path to an
/// accepted call (structurally, under the D1 schema-trie, and under the D2
/// identifier lock), plus `EOS` once a call is complete; `f32::NEG_INFINITY`
/// otherwise. Returns the mask plus whether it offers at least one legal
/// continuation.
fn build_copy_mask(
    state: &ConstrainedDecodeState,
    grammar: &RuntimeCallGrammar,
    copy: &CopyConstraints,
    pieces: &[Option<Vec<u8>>],
    vocab_size: usize,
    eos_id: u32,
) -> (Vec<f32>, bool) {
    let mut mask = vec![f32::NEG_INFINITY; vocab_size];
    let mut any_allowed = false;
    for (id, piece) in pieces.iter().enumerate().take(vocab_size) {
        let Some(bytes) = piece else { continue };
        if bytes.is_empty() {
            continue;
        }
        if state.advance(grammar, copy, bytes).is_some() {
            mask[id] = 0.0;
            any_allowed = true;
        }
    }
    if grammar.accepts_eos(&state.frontier) && (eos_id as usize) < vocab_size {
        mask[eos_id as usize] = 0.0;
        any_allowed = true;
    }
    (mask, any_allowed)
}

/// Pull `logits` to a host `Vec<f32>` (the same `.into_data().to_vec()`
/// idiom [`crate::vision`]'s decode loops already use) and run the D2
/// mechanism-2 soft bias, incrementing `overrides` when it picks something
/// other than the masked argmax. Returns `None` when the host extraction
/// fails (a non-`f32` backend float element type) or there is no context
/// suffix match at all, in which case the caller falls back to the exact
/// pre-existing tensor argmax path.
fn select_biased_next_id<B: Backend>(
    last_logits: &Tensor<B, 1>,
    mask: &[f32],
    pieces: &[Option<Vec<u8>>],
    copy: &CopyConstraints,
    content: &[u8],
    margin: f32,
    overrides: &mut u32,
) -> Option<u32> {
    let continuations = copy.suffix.continuations(content)?;
    let host: Vec<f32> = last_logits.clone().into_data().to_vec().ok()?;
    if host.len() != mask.len() {
        return None;
    }
    let (id, overridden) = select_next_token_with_copy_bias(&host, mask, pieces, &continuations, margin)?;
    if overridden {
        *overrides += 1;
    }
    Some(id)
}

/// Pure token-selection logic for the D2 soft suffix-copy bias (mechanism 2),
/// decoupled from any tensor backend so it can be exercised directly with
/// scripted logits in tests. `logits` and `mask` are parallel arrays over the
/// vocabulary (`mask[i]` is `0.0` for a legal continuation, exactly what
/// [`build_copy_mask`] produces, `f32::NEG_INFINITY` otherwise). Returns
/// `(chosen_id, overridden)`, where `overridden` is `true` iff the margin
/// bias picked something other than the masked argmax; `None` only when no
/// id is legal at all (should not happen -- the caller already checked
/// `any_allowed`).
///
/// The bias never considers an illegal id (`legal` gates every candidate),
/// so the structural grammar and any D1/D2 hard constraint always take
/// precedence over this ranking choice.
pub fn select_next_token_with_copy_bias(
    logits: &[f32],
    mask: &[f32],
    pieces: &[Option<Vec<u8>>],
    continuations: &std::collections::HashSet<u8>,
    margin: f32,
) -> Option<(u32, bool)> {
    let len = logits.len().min(mask.len());
    let legal = |id: usize| mask[id].is_finite();
    let score = |id: usize| logits[id] + mask[id];
    let starts_with_continuation = |id: usize| {
        pieces
            .get(id)
            .and_then(|piece| piece.as_ref())
            .and_then(|bytes| bytes.first())
            .is_some_and(|byte| continuations.contains(byte))
    };

    let mut best_id = None;
    let mut best_score = f32::NEG_INFINITY;
    for id in 0..len {
        if !legal(id) {
            continue;
        }
        let s = score(id);
        if best_id.is_none() || s > best_score {
            best_id = Some(id);
            best_score = s;
        }
    }
    let best_id = best_id?;

    if starts_with_continuation(best_id) {
        return Some((best_id as u32, false));
    }

    let mut override_id = None;
    let mut override_score = f32::NEG_INFINITY;
    for id in 0..len {
        if !legal(id) || id == best_id {
            continue;
        }
        let s = score(id);
        if s < best_score - margin {
            continue;
        }
        if starts_with_continuation(id) && s > override_score {
            override_id = Some(id);
            override_score = s;
        }
    }

    match override_id {
        Some(id) => Some((id as u32, true)),
        None => Some((best_id as u32, false)),
    }
}

/// Schema-constrained greedy generation (Phase 51 P1a-4 Step A).
///
/// Identical framing to [`generate`], but at every step the next-token choice
/// is restricted to tokens whose byte surface keeps the running output on a
/// path to a call accepted by `grammar` (a byte-level acceptor over the
/// offered tools' schemas — see [`grug_data::constrain`]). Disallowed tokens'
/// logits are masked to `-inf` before the argmax, so the model still picks its
/// highest-scoring *legal* continuation. `EOS` is offered only once a complete
/// call has been formed (`grammar.accepts_eos`), which prevents the model
/// stopping mid-call.
///
/// `pieces` is the tokenizer's per-id byte table ([`Vocab::token_pieces`]),
/// built once by the caller and shared across examples. Tokens with no
/// appendable bytes (the special tokens) are never offered as a continuation.
///
/// This constrains *structure and parameter names*, not values (see the module
/// grammar docs): it can force valid, schema-consistent keys but cannot supply
/// correct argument *values*, which must be copied from the request — exactly
/// the isolation Step A measures.
pub fn generate_constrained<B: Backend, V: Vocab>(
    model: &TinyTransformer<B>,
    tokenizer: &V,
    prompt: &str,
    grammar: &ToolCallGrammar,
    pieces: &[Option<Vec<u8>>],
    config: GenerateConfig,
    device: &B::Device,
) -> String {
    let eos_id = tokenizer.eos_id();

    let mut ids: Vec<u32> = Vec::new();
    ids.push(tokenizer.bos_id());
    ids.extend(tokenizer.encode(prompt).unwrap_or_default());
    ids.push(tokenizer.sep_id());

    let mut generated: Vec<u32> = Vec::new();
    let mut frontier: GrammarState = grammar.start();

    for _ in 0..config.max_new_tokens {
        let mut window = ids.clone();
        if window.len() > config.max_seq_len {
            let drop = window.len() - config.max_seq_len;
            window.drain(0..drop);
        }

        let seq_len = window.len();
        let input_flat: Vec<i64> = window.iter().map(|&t| t as i64).collect();
        let input =
            Tensor::<B, 2, Int>::from_data(TensorData::new(input_flat, [1, seq_len]), device);

        let logits = model.forward(input); // [1, seq_len, vocab]
        let vocab_size = logits.dims()[2];
        let last_logits = logits
            .slice([0..1, seq_len - 1..seq_len, 0..vocab_size])
            .reshape([vocab_size]);

        // Build an additive mask: 0.0 for tokens the grammar allows as the
        // next continuation, -inf otherwise. Precompute the per-byte
        // legality of the current frontier so most tokens reject on their
        // first byte without cloning grammar state.
        let mut mask = vec![f32::NEG_INFINITY; vocab_size];
        let mut any_allowed = false;
        for (id, piece) in pieces.iter().enumerate().take(vocab_size) {
            if let Some(bytes) = piece
                && !bytes.is_empty()
                && grammar.advance(&frontier, bytes).is_some()
            {
                mask[id] = 0.0;
                any_allowed = true;
            }
        }
        // Offer EOS only when a complete call has been formed.
        let eos_ok = grammar.accepts_eos(&frontier);
        if eos_ok && (eos_id as usize) < vocab_size {
            mask[eos_id as usize] = 0.0;
            any_allowed = true;
        }
        if !any_allowed {
            // Dead end (should not happen: every non-`End` state has a legal
            // continuation, and `End` offers EOS) — stop rather than emit an
            // out-of-grammar token.
            break;
        }

        let mask_tensor = Tensor::<B, 1>::from_data(TensorData::new(mask, [vocab_size]), device);
        let masked = last_logits + mask_tensor;
        let next_id = argmax(masked);

        if next_id == eos_id {
            break;
        }

        // Commit: advance the frontier by the chosen token's bytes. It was in
        // the allowed set, so `advance` is `Some`; fall back defensively.
        if let Some(bytes) = pieces.get(next_id as usize).and_then(|p| p.as_ref()) {
            match grammar.advance(&frontier, bytes) {
                Some(next_state) => frontier = next_state,
                None => break,
            }
        } else {
            break;
        }
        ids.push(next_id);
        generated.push(next_id);
    }

    tokenizer.decode(&generated).unwrap_or_else(|_| {
        let bytes: Vec<u8> = generated
            .iter()
            .copied()
            .filter(|&t| t < 256)
            .map(|t| t as u8)
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    })
}

/// Greedy generation from a [`PointerTransformer`] (Phase 51 P1a-4 Step B).
///
/// Identical framing to [`generate`], but the next token is the argmax of the
/// model's *mixed* distribution (`PointerTransformer::log_probs`) at the last
/// position — argmax over log-probs equals argmax over probs, so this needs no
/// special handling versus the plain LM path.
pub fn generate_ptr<B: Backend, V: Vocab>(
    model: &PointerTransformer<B>,
    tokenizer: &V,
    prompt: &str,
    config: GenerateConfig,
    device: &B::Device,
) -> String {
    let mut ids: Vec<u32> = Vec::new();
    ids.push(tokenizer.bos_id());
    ids.extend(tokenizer.encode(prompt).unwrap_or_default());
    ids.push(tokenizer.sep_id());

    let mut generated: Vec<u32> = Vec::new();

    for _ in 0..config.max_new_tokens {
        let mut window = ids.clone();
        if window.len() > config.max_seq_len {
            let drop = window.len() - config.max_seq_len;
            window.drain(0..drop);
        }

        let seq_len = window.len();
        let input_flat: Vec<i64> = window.iter().map(|&t| t as i64).collect();
        let input =
            Tensor::<B, 2, Int>::from_data(TensorData::new(input_flat, [1, seq_len]), device);

        let log_probs = model.log_probs(input); // [1, seq_len, vocab]
        let vocab_size = log_probs.dims()[2];
        let last = log_probs
            .slice([0..1, seq_len - 1..seq_len, 0..vocab_size])
            .reshape([vocab_size]);

        let next_id = argmax(last);
        if next_id == tokenizer.eos_id() {
            break;
        }
        ids.push(next_id);
        generated.push(next_id);
    }

    tokenizer.decode(&generated).unwrap_or_else(|_| {
        let bytes: Vec<u8> = generated
            .iter()
            .copied()
            .filter(|&t| t < 256)
            .map(|t| t as u8)
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    })
}

/// Schema-constrained greedy generation from a [`PointerTransformer`] (Phase 51
/// P1a-4 Step B): [`generate_ptr`] with the [`generate_constrained`] grammar
/// mask applied to the mixed distribution's log-probs. Lets the combined
/// effect of copy mechanism + schema constraint be measured.
pub fn generate_ptr_constrained<B: Backend, V: Vocab>(
    model: &PointerTransformer<B>,
    tokenizer: &V,
    prompt: &str,
    grammar: &ToolCallGrammar,
    pieces: &[Option<Vec<u8>>],
    config: GenerateConfig,
    device: &B::Device,
) -> String {
    let eos_id = tokenizer.eos_id();

    let mut ids: Vec<u32> = Vec::new();
    ids.push(tokenizer.bos_id());
    ids.extend(tokenizer.encode(prompt).unwrap_or_default());
    ids.push(tokenizer.sep_id());

    let mut generated: Vec<u32> = Vec::new();
    let mut frontier: GrammarState = grammar.start();

    for _ in 0..config.max_new_tokens {
        let mut window = ids.clone();
        if window.len() > config.max_seq_len {
            let drop = window.len() - config.max_seq_len;
            window.drain(0..drop);
        }

        let seq_len = window.len();
        let input_flat: Vec<i64> = window.iter().map(|&t| t as i64).collect();
        let input =
            Tensor::<B, 2, Int>::from_data(TensorData::new(input_flat, [1, seq_len]), device);

        let log_probs = model.log_probs(input); // [1, seq_len, vocab]
        let vocab_size = log_probs.dims()[2];
        let last = log_probs
            .slice([0..1, seq_len - 1..seq_len, 0..vocab_size])
            .reshape([vocab_size]);

        let mut mask = vec![f32::NEG_INFINITY; vocab_size];
        let mut any_allowed = false;
        for (id, piece) in pieces.iter().enumerate().take(vocab_size) {
            if let Some(bytes) = piece
                && !bytes.is_empty()
                && grammar.advance(&frontier, bytes).is_some()
            {
                mask[id] = 0.0;
                any_allowed = true;
            }
        }
        let eos_ok = grammar.accepts_eos(&frontier);
        if eos_ok && (eos_id as usize) < vocab_size {
            mask[eos_id as usize] = 0.0;
            any_allowed = true;
        }
        if !any_allowed {
            break;
        }

        let mask_tensor = Tensor::<B, 1>::from_data(TensorData::new(mask, [vocab_size]), device);
        let next_id = argmax(last + mask_tensor);

        if next_id == eos_id {
            break;
        }
        if let Some(bytes) = pieces.get(next_id as usize).and_then(|p| p.as_ref()) {
            match grammar.advance(&frontier, bytes) {
                Some(next_state) => frontier = next_state,
                None => break,
            }
        } else {
            break;
        }
        ids.push(next_id);
        generated.push(next_id);
    }

    tokenizer.decode(&generated).unwrap_or_else(|_| {
        let bytes: Vec<u8> = generated
            .iter()
            .copied()
            .filter(|&t| t < 256)
            .map(|t| t as u8)
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    })
}

/// Argmax over a 1-D tensor, returned as `u32` (token id).
///
/// Uses `into_scalar().elem()` (the same backend-agnostic conversion path
/// [`crate::train::train_step`] uses for the loss scalar) rather than a
/// hardcoded `as_slice::<i64>()`: Burn's `Int` element type is
/// backend-defined (`burn-ndarray`'s CPU backend uses `i64`, but CubeCL
/// backends like `Rocm`/`Wgpu` default their `Int` type to `i32`), so
/// assuming `i64` panics with a `TypeMismatch` on GPU backends. `.elem::<i64>()`
/// converts whatever the backend's actual element type is into `i64` first,
/// so this works for any backend's `Int` representation.
fn argmax<B: Backend>(logits: Tensor<B, 1>) -> u32 {
    let idx: Tensor<B, 1, Int> = logits.argmax(0);
    let value: i64 = idx.into_scalar().elem();
    value as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CpuBackend;
    use crate::tokenizer::ByteTokenizer;
    use crate::transformer::TinyTransformerConfig;

    #[test]
    fn generate_stops_at_max_new_tokens_on_untrained_model() {
        let device = Default::default();
        let model_config =
            TinyTransformerConfig::new(crate::tokenizer::VOCAB_SIZE, 16, 2, 1, 32, 64);
        let model = model_config.init::<CpuBackend>(&device);
        let tok = ByteTokenizer::new();

        let config = GenerateConfig {
            max_new_tokens: 5,
            max_seq_len: 64,
        };
        let out = generate(&model, &tok, "hello", config, &device);
        // No correctness claim on an untrained model -- just that it
        // terminates without panicking. Note the bound is on *tokens*
        // generated (<= max_new_tokens = 5), not on the resulting string's
        // byte length: a lossily-decoded replacement character (U+FFFD, an
        // expected outcome from an untrained model emitting arbitrary byte
        // tokens) is 3 UTF-8 bytes per single generated token.
        assert!(
            out.chars().count() <= 5 * 3,
            "expected at most 5 generated tokens' worth of (possibly replacement-char-inflated) output, got {out:?}"
        );
    }

    #[test]
    fn cached_generation_matches_the_compatibility_path() {
        let device = Default::default();
        let model_config =
            TinyTransformerConfig::new(crate::tokenizer::VOCAB_SIZE, 16, 2, 1, 32, 64);
        let model = model_config.init::<CpuBackend>(&device);
        let tokenizer = ByteTokenizer::new();
        let prompt_ids = tokenizer.encode("cache parity");
        let config = GenerateConfig {
            max_new_tokens: 8,
            max_seq_len: 64,
        };
        assert_eq!(
            generate_from_prompt_ids(&model, &tokenizer, &prompt_ids, config, &device),
            generate_from_prompt_ids_cached(&model, &tokenizer, &prompt_ids, config, &device)
        );
    }

    /// Phase 51 / ADR 104 D1 (f): with a grammar offered but no field
    /// value-copy-constrained (a tool with only a non-string parameter), the
    /// constrained cached decoder must match the plain cached decoder
    /// byte-for-byte on the same untrained model -- the runtime envelope's
    /// `Undecided`/`Free` states never mask a byte the model wouldn't already
    /// have produced (see `RuntimeCallState` docs), so there is nothing here
    /// for the copy constraint to change.
    #[test]
    fn constrained_cached_generation_matches_the_plain_cached_path_with_no_copy_fields() {
        // Seed the backend so the random model does not depend on how many
        // other tests consumed the global RNG before this one ran.
        <CpuBackend as burn::tensor::backend::Backend>::seed(&Default::default(), 7);
        let device = Default::default();
        let model_config =
            TinyTransformerConfig::new(crate::tokenizer::VOCAB_SIZE, 16, 2, 1, 32, 64);
        let model = model_config.init::<CpuBackend>(&device);
        let tokenizer = ByteTokenizer::new();
        let prompt_ids = tokenizer.encode("cache parity");
        let config = GenerateConfig {
            max_new_tokens: 8,
            max_seq_len: 64,
        };
        let pieces = tokenizer.token_pieces();
        let tools = vec![grug_data::ToolSchema::new("noop", "d", vec![])];
        let grammar = grug_data::constrain::RuntimeCallGrammar::new(&tools);

        let plain =
            generate_from_prompt_ids_cached(&model, &tokenizer, &prompt_ids, config, &device);
        let (constrained, report) = generate_from_prompt_ids_cached_constrained(
            &model,
            &tokenizer,
            &prompt_ids,
            ConstrainedDecode::new(Some(&grammar), "irrelevant context"),
            &pieces,
            config,
            &device,
        );
        assert_eq!(plain, constrained);
        assert!(report.constrained);
        assert!(report.copy_fields.is_empty());
        // No string parameters exist at all, so neither D2 mechanism ever
        // had anything to engage on.
        assert_eq!(report.identifier_copy_locks, 0);
        assert_eq!(report.identifier_copy_releases, 0);
        assert_eq!(report.copy_bias_overrides, 0);
    }

    /// With `grammar: None` the constrained entry point must degrade
    /// byte-for-byte to the plain cached path and report `constrained: false`
    /// -- this is the pure-answer/no-tools-offered turn.
    #[test]
    fn constrained_cached_generation_with_no_grammar_matches_the_plain_cached_path() {
        let device = Default::default();
        let model_config =
            TinyTransformerConfig::new(crate::tokenizer::VOCAB_SIZE, 16, 2, 1, 32, 64);
        let model = model_config.init::<CpuBackend>(&device);
        let tokenizer = ByteTokenizer::new();
        let prompt_ids = tokenizer.encode("no tools offered");
        let config = GenerateConfig {
            max_new_tokens: 8,
            max_seq_len: 64,
        };
        let pieces = tokenizer.token_pieces();

        let plain =
            generate_from_prompt_ids_cached(&model, &tokenizer, &prompt_ids, config, &device);
        let (constrained, report) = generate_from_prompt_ids_cached_constrained(
            &model,
            &tokenizer,
            &prompt_ids,
            ConstrainedDecode::new(None, "irrelevant context"),
            &pieces,
            config,
            &device,
        );
        assert_eq!(plain, constrained);
        assert!(!report.constrained);
    }

    // ── D2 mechanism 2: soft suffix-copy bias (pure selection logic) ───────

    fn byte_pieces(bytes: &[u8]) -> Vec<Option<Vec<u8>>> {
        bytes.iter().map(|&b| Some(vec![b])).collect()
    }

    /// (b) The measured failure this mechanism targets: the argmax choice
    /// does not continue a verbatim context match, but a near-argmax
    /// alternative (here, `'9'` continuing `TV-9..` toward `TV-926`) does,
    /// and is well within the default 4.0-nat margin -- it must be preferred.
    #[test]
    fn margin_bias_overrides_the_argmax_when_the_alternative_is_within_margin() {
        // id 0 = 'z' (argmax, does not continue the match), id 1 = '9'
        // (continues the match, close in logit), id 2 = 'x' (irrelevant).
        let pieces = byte_pieces(b"z9x");
        let mask = vec![0.0, 0.0, 0.0];
        let continuations = std::collections::HashSet::from(*b"9");
        let logits = vec![10.0, 7.0, 0.0]; // deficit for id1 is 3.0 <= 4.0
        let (id, overridden) =
            select_next_token_with_copy_bias(&logits, &mask, &pieces, &continuations, 4.0)
                .expect("some legal id must be selected");
        assert_eq!(id, 1, "the within-margin context-continuing token must win");
        assert!(overridden);
    }

    /// The negative case: the same alternative exists but its logit deficit
    /// exceeds the margin, so the plain masked argmax must stand.
    #[test]
    fn margin_bias_does_not_override_when_the_alternative_is_far_outside_margin() {
        let pieces = byte_pieces(b"z9x");
        let mask = vec![0.0, 0.0, 0.0];
        let continuations = std::collections::HashSet::from(*b"9");
        let logits = vec![10.0, 4.0, 0.0]; // deficit for id1 is 6.0 > 4.0
        let (id, overridden) =
            select_next_token_with_copy_bias(&logits, &mask, &pieces, &continuations, 4.0)
                .expect("some legal id must be selected");
        assert_eq!(id, 0, "a far-outside-margin alternative must not override argmax");
        assert!(!overridden);
    }

    /// When the argmax itself already continues the context match, there is
    /// nothing to override -- report it as such rather than a spurious
    /// override.
    #[test]
    fn margin_bias_is_a_no_op_when_the_argmax_already_continues_the_match() {
        let pieces = byte_pieces(b"9zx");
        let mask = vec![0.0, 0.0, 0.0];
        let continuations = std::collections::HashSet::from(*b"9");
        let logits = vec![10.0, 0.0, 0.0];
        let (id, overridden) =
            select_next_token_with_copy_bias(&logits, &mask, &pieces, &continuations, 4.0)
                .expect("some legal id must be selected");
        assert_eq!(id, 0);
        assert!(!overridden);
    }

    /// (e) Grammar/hard-constraint precedence: an id that continues the
    /// context match but is illegal under the mask (e.g. rejected by the
    /// structural grammar or a D1/D2 hard lock) must never be selected, no
    /// matter how attractive its logit -- the bias only ranks among ids the
    /// mask already allows.
    #[test]
    fn margin_bias_never_selects_an_id_the_mask_marks_illegal() {
        let pieces = byte_pieces(b"z9");
        let mask = vec![0.0, f32::NEG_INFINITY]; // id 1 ('9') is illegal
        let continuations = std::collections::HashSet::from(*b"9");
        let logits = vec![5.0, 100.0]; // id 1 would dominate if legality were ignored
        let (id, overridden) =
            select_next_token_with_copy_bias(&logits, &mask, &pieces, &continuations, 4.0)
                .expect("the one legal id must still be selected");
        assert_eq!(id, 0, "an illegal id must never be picked regardless of margin or logit");
        assert!(!overridden);
    }

    /// `None` (no legal id at all) only when the mask allows nothing.
    #[test]
    fn margin_bias_returns_none_when_nothing_is_legal() {
        let pieces = byte_pieces(b"z9");
        let mask = vec![f32::NEG_INFINITY, f32::NEG_INFINITY];
        let continuations = std::collections::HashSet::from(*b"9");
        let logits = vec![5.0, 100.0];
        assert_eq!(
            select_next_token_with_copy_bias(&logits, &mask, &pieces, &continuations, 4.0),
            None
        );
    }
}
