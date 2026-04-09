//! ONNX-based sentence embedding using all-MiniLM-L6-v2.
//!
//! Produces 384-dimensional float32 vectors suitable for pgvector cosine search.
//! Uses the same `ort` + `tokenizers` crates as akh-medu's NLU micro-ML tier.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::error::{SeshatError, SeshatResult};

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
    pub fn load(model_dir: &Path) -> SeshatResult<Self> {
        let model_path = model_dir.join("model.onnx");
        let tokenizer_path = model_dir.join("tokenizer.json");

        if !model_path.exists() || !tokenizer_path.exists() {
            return Err(SeshatError::Embedding(format!(
                "model files not found in {}. Expected model.onnx and tokenizer.json",
                model_dir.display()
            )));
        }

        let session = ort::session::Session::builder()
            .map_err(|e| SeshatError::Embedding(format!("ONNX session builder: {e}")))?
            .commit_from_file(&model_path)
            .map_err(|e| SeshatError::Embedding(format!("loading model: {e}")))?;

        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| SeshatError::Embedding(format!("loading tokenizer: {e}")))?;

        Ok(Self { session, tokenizer })
    }

    /// Embed a single text string into a 384-dim vector.
    pub fn embed_text(&mut self, text: &str) -> SeshatResult<Vec<f32>> {
        let batch = self.embed_batch(&[text])?;
        Ok(batch.into_iter().next().unwrap_or_default())
    }

    /// Embed a batch of texts into 384-dim vectors.
    pub fn embed_batch(&mut self, texts: &[&str]) -> SeshatResult<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Tokenize all texts.
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| SeshatError::Embedding(format!("tokenization: {e}")))?;

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
        let input_ids_tensor =
            ort::value::Value::from_array(ndarray_owned(&input_ids, &shape))
                .map_err(|e| SeshatError::Embedding(format!("tensor creation: {e}")))?;
        let attention_mask_tensor =
            ort::value::Value::from_array(ndarray_owned(&attention_mask, &shape))
                .map_err(|e| SeshatError::Embedding(format!("tensor creation: {e}")))?;
        let token_type_ids_tensor =
            ort::value::Value::from_array(ndarray_owned(&token_type_ids, &shape))
                .map_err(|e| SeshatError::Embedding(format!("tensor creation: {e}")))?;

        let inputs = ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
            "token_type_ids" => token_type_ids_tensor,
        ];

        let outputs = self
            .session
            .run(inputs)
            .map_err(|e| SeshatError::Embedding(format!("inference: {e}")))?;

        // Extract embeddings from output.
        // The model outputs token embeddings of shape [batch, seq_len, hidden_dim].
        // We do mean pooling over the token dimension using the attention mask.
        // ort 2.0.0-rc returns (Shape, &[f32]) from try_extract_tensor.
        let (output_shape, output_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| SeshatError::Embedding(format!("extracting output: {e}")))?;

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
    ndarray::Array2::from_shape_vec(*shape, data.to_vec()).expect("shape mismatch in tensor creation")
}

/// Resolve the model directory, downloading if necessary.
pub fn resolve_model_dir(base_dir: &Path, model_name: &str) -> SeshatResult<PathBuf> {
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
    Err(SeshatError::Embedding(format!(
        "model '{}' not found at {}. \
         Download from https://huggingface.co/sentence-transformers/{}/tree/main \
         and place model.onnx + tokenizer.json in that directory.",
        model_name,
        model_dir.display(),
        model_name,
    )))
}

/// Shared embedder handle for use across async tasks.
/// Wrapped in a Mutex because `Session::run` requires `&mut self`.
pub type SharedEmbedder = Arc<std::sync::Mutex<SentenceEmbedder>>;
