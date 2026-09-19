//! Shared ONNX-based sentence embedding using all-MiniLM-L6-v2.
//!
//! Produces 384-dimensional float32 vectors suitable for pgvector cosine
//! search. Extracted from `seshat::embedder` — see plan increment 4 ("akh-embed
//! extraction") and
//! `docs/wayfinder/transcript-harvest/tickets/embedding-pipeline.md` §3 — so
//! both `seshat` and `akh-transcripts` depend on one implementation instead of
//! carrying two copies of `ort` + `tokenizers`.

mod error;

pub use error::{EmbedError, EmbedResult};

use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Embedding dimension for all-MiniLM-L6-v2.
pub const EMBEDDING_DIM: usize = 384;

/// Maximum token length for the model.
const MAX_TOKENS: usize = 512;

/// Sentence embedder backed by an ONNX model.
pub struct SentenceEmbedder {
    session: ort::session::Session,
    tokenizer: tokenizers::Tokenizer,
}

impl SentenceEmbedder {
    /// Load the embedding model from the given directory.
    ///
    /// Expects `model.onnx` and `tokenizer.json` inside `model_dir`.
    pub fn load(model_dir: &Path) -> EmbedResult<Self> {
        let model_path = model_dir.join("model.onnx");
        let tokenizer_path = model_dir.join("tokenizer.json");

        if !model_path.exists() || !tokenizer_path.exists() {
            let model_name = model_dir
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("MODEL_NAME")
                .to_string();
            return Err(EmbedError::ModelFilesMissing {
                dir: model_dir.to_path_buf(),
                model_name,
            });
        }

        let session = ort::session::Session::builder()
            .map_err(|e| EmbedError::SessionBuild(format!("session builder: {e}")))?
            .commit_from_file(&model_path)
            .map_err(|e| EmbedError::SessionBuild(format!("loading model: {e}")))?;

        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| EmbedError::TokenizerLoad(format!("{e}")))?;

        Ok(Self { session, tokenizer })
    }

    /// Embed a single text string into a 384-dim vector.
    pub fn embed_text(&mut self, text: &str) -> EmbedResult<Vec<f32>> {
        let batch = self.embed_batch(&[text])?;
        Ok(batch.into_iter().next().unwrap_or_default())
    }

    /// Embed a batch of texts into 384-dim vectors.
    pub fn embed_batch(&mut self, texts: &[&str]) -> EmbedResult<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Tokenize all texts.
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| EmbedError::Tokenize(format!("{e}")))?;

        let batch_size = encodings.len();

        // Find max length (capped at MAX_TOKENS).
        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len().min(MAX_TOKENS))
            .max()
            .unwrap_or(0);

        // Build padded input tensors.
        let mut input_ids = vec![0i64; batch_size * max_len];
        let mut attention_mask = vec![0i64; batch_size * max_len];
        let mut token_type_ids = vec![0i64; batch_size * max_len];

        for (i, encoding) in encodings.iter().enumerate() {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            let type_ids = encoding.get_type_ids();
            let len = ids.len().min(max_len);

            for j in 0..len {
                input_ids[i * max_len + j] = ids[j] as i64;
                attention_mask[i * max_len + j] = mask[j] as i64;
                token_type_ids[i * max_len + j] = type_ids[j] as i64;
            }
        }

        let shape = [batch_size, max_len];

        // Run inference.
        let input_ids_tensor = ort::value::Value::from_array(ndarray_owned(&input_ids, &shape))
            .map_err(|e| EmbedError::Tensor(format!("input_ids: {e}")))?;
        let attention_mask_tensor =
            ort::value::Value::from_array(ndarray_owned(&attention_mask, &shape))
                .map_err(|e| EmbedError::Tensor(format!("attention_mask: {e}")))?;
        let token_type_ids_tensor =
            ort::value::Value::from_array(ndarray_owned(&token_type_ids, &shape))
                .map_err(|e| EmbedError::Tensor(format!("token_type_ids: {e}")))?;

        let inputs = ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
            "token_type_ids" => token_type_ids_tensor,
        ];

        let outputs = self
            .session
            .run(inputs)
            .map_err(|e| EmbedError::Inference(format!("running model: {e}")))?;

        // Extract embeddings from output.
        // The model outputs token embeddings of shape [batch, seq_len, hidden_dim].
        // We do mean pooling over the token dimension using the attention mask.
        // ort 2.0.0-rc returns (Shape, &[f32]) from try_extract_tensor.
        let (output_shape, output_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| EmbedError::Inference(format!("extracting output tensor: {e}")))?;

        let hidden_dim = if output_shape.len() == 3 {
            output_shape[2] as usize
        } else {
            EMBEDDING_DIM
        };

        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let mask_sum: f32 = (0..max_len)
                .map(|j| attention_mask[i * max_len + j] as f32)
                .sum();

            let mut embedding = vec![0.0f32; hidden_dim];
            if mask_sum > 0.0 {
                for j in 0..max_len {
                    let mask_val = attention_mask[i * max_len + j] as f32;
                    if mask_val > 0.0 {
                        // Index into flat array: [batch][seq][hidden] = i*max_len*hidden + j*hidden + k
                        let base = i * max_len * hidden_dim + j * hidden_dim;
                        for k in 0..hidden_dim {
                            embedding[k] += output_data[base + k] * mask_val;
                        }
                    }
                }
                for val in &mut embedding {
                    *val /= mask_sum;
                }
            }

            // L2 normalize.
            let norm: f32 = embedding.iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm > 0.0 {
                for val in &mut embedding {
                    *val /= norm;
                }
            }

            results.push(embedding);
        }

        Ok(results)
    }
}

/// Helper: create an owned ndarray from a flat slice and shape.
fn ndarray_owned(data: &[i64], shape: &[usize; 2]) -> ndarray::Array2<i64> {
    ndarray::Array2::from_shape_vec(*shape, data.to_vec())
        .expect("shape mismatch in tensor creation")
}

/// Resolve the model directory, downloading if necessary.
pub fn resolve_model_dir(base_dir: &Path, model_name: &str) -> EmbedResult<PathBuf> {
    let model_dir = base_dir.join(model_name);
    if model_dir.join("model.onnx").exists() && model_dir.join("tokenizer.json").exists() {
        return Ok(model_dir);
    }

    tracing::info!(
        model = model_name,
        dir = %model_dir.display(),
        "embedding model not found — download required"
    );

    // TODO: Implement auto-download from HuggingFace.
    // For now, return an error with instructions.
    Err(EmbedError::ModelFilesMissing {
        dir: model_dir,
        model_name: model_name.to_string(),
    })
}

/// Shared embedder handle for use across async tasks.
/// Wrapped in a Mutex because `Session::run` requires `&mut self`.
pub type SharedEmbedder = Arc<std::sync::Mutex<SentenceEmbedder>>;

#[cfg(test)]
mod tests {
    use super::*;

    /// Embeds one real string through the actual ONNX model and checks the
    /// output shape and L2 normalisation — the properties the rest of the
    /// pipeline (pgvector cosine search) relies on.
    ///
    /// Needs `model.onnx` + `tokenizer.json` on disk, so it is gated behind
    /// `AKH_MINILM_DIR` (a no-op skip, not a failure, when unset) rather than
    /// running unconditionally in CI.
    ///
    /// ```text
    /// AKH_MINILM_DIR=/path/to/all-MiniLM-L6-v2 cargo test -p akh-embed embed_produces_a_384_dim_l2_normalised_vector
    /// ```
    #[test]
    fn embed_produces_a_384_dim_l2_normalised_vector() {
        let Ok(dir) = std::env::var("AKH_MINILM_DIR") else {
            eprintln!(
                "skipping: set AKH_MINILM_DIR to a directory with model.onnx + tokenizer.json to run this test"
            );
            return;
        };
        let dir = PathBuf::from(dir);

        let mut embedder = SentenceEmbedder::load(&dir).expect("load real MiniLM model");
        let vector = embedder
            .embed_text("the quick brown fox jumps over the lazy dog")
            .expect("embed a real sentence");

        assert_eq!(vector.len(), EMBEDDING_DIM);
        let norm: f32 = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-3,
            "expected an L2-normalised vector, got norm {norm}"
        );
    }
}
