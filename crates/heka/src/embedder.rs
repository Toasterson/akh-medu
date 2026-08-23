//! ONNX-based sentence embedding using all-MiniLM-L6-v2.
//!
//! Ported from `crates/seshat/src/embedder.rs` (Phase 37). Produces
//! 384-dimensional float32 vectors suitable for pgvector cosine search.
//! Uses the same `ort` + `tokenizers` crates as seshat.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::error::{HekaError, HekaResult};

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
    pub fn load(model_dir: &Path) -> HekaResult<Self> {
        let model_path = model_dir.join("model.onnx");
        let tokenizer_path = model_dir.join("tokenizer.json");

        if !model_path.exists() || !tokenizer_path.exists() {
            return Err(HekaError::Embedding(format!(
                "model files not found in {}. Expected model.onnx and tokenizer.json",
                model_dir.display()
            )));
        }

        let session = ort::session::Session::builder()
            .map_err(|e| HekaError::Embedding(format!("ONNX session builder: {e}")))?
            .commit_from_file(&model_path)
            .map_err(|e| HekaError::Embedding(format!("loading model: {e}")))?;

        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| HekaError::Embedding(format!("loading tokenizer: {e}")))?;

        Ok(Self { session, tokenizer })
    }

    /// Embed a single text string into a 384-dim vector.
    pub fn embed_text(&mut self, text: &str) -> HekaResult<Vec<f32>> {
        let batch = self.embed_batch(&[text])?;
        Ok(batch.into_iter().next().unwrap_or_default())
    }

    /// Embed a batch of texts into 384-dim vectors.
    pub fn embed_batch(&mut self, texts: &[&str]) -> HekaResult<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Tokenize all texts.
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| HekaError::Embedding(format!("tokenization: {e}")))?;

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
            .map_err(|e| HekaError::Embedding(format!("tensor creation: {e}")))?;
        let attention_mask_tensor =
            ort::value::Value::from_array(ndarray_owned(&attention_mask, &shape))
                .map_err(|e| HekaError::Embedding(format!("tensor creation: {e}")))?;
        let token_type_ids_tensor =
            ort::value::Value::from_array(ndarray_owned(&token_type_ids, &shape))
                .map_err(|e| HekaError::Embedding(format!("tensor creation: {e}")))?;

        let inputs = ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
            "token_type_ids" => token_type_ids_tensor,
        ];

        let outputs = self
            .session
            .run(inputs)
            .map_err(|e| HekaError::Embedding(format!("inference: {e}")))?;

        // Extract embeddings from output.
        // The model outputs token embeddings of shape [batch, seq_len, hidden_dim].
        // We do mean pooling over the token dimension using the attention mask.
        let (output_shape, output_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| HekaError::Embedding(format!("extracting output: {e}")))?;

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
pub fn resolve_model_dir(base_dir: &Path, model_name: &str) -> HekaResult<PathBuf> {
    let model_dir = base_dir.join(model_name);
    if model_dir.join("model.onnx").exists() && model_dir.join("tokenizer.json").exists() {
        return Ok(model_dir);
    }

    tracing::info!(
        model = model_name,
        dir = %model_dir.display(),
        "embedding model not found — download required"
    );

    Err(HekaError::Embedding(format!(
        "model '{}' not found at {}. \
         Download from https://huggingface.co/sentence-transformers/{}/tree/main \
         and place model.onnx + tokenizer.json in that directory.",
        model_name,
        model_dir.display(),
        model_name,
    )))
}

/// Shared embedder handle for use across async tasks.
///
/// `Session::run` requires mutable access, while ONNX inference must execute on
/// a blocking thread. A timed-out `spawn_blocking` task cannot be cancelled, so
/// the semaphore deliberately remains owned by that task until it really exits.
/// Callers therefore backpressure behind at most one outstanding inference
/// instead of spawning an unbounded queue of stale retries.
#[derive(Clone)]
pub struct SharedEmbedder {
    model: Arc<Mutex<SentenceEmbedder>>,
    gate: InferenceGate,
}

impl SharedEmbedder {
    pub fn new(model: SentenceEmbedder) -> Self {
        Self {
            model: Arc::new(Mutex::new(model)),
            gate: InferenceGate::new(),
        }
    }

    pub(crate) async fn acquire_inference(
        &self,
        wait_timeout: Duration,
    ) -> HekaResult<InferencePermit> {
        self.gate.acquire(wait_timeout).await
    }

    pub(crate) async fn infer_with_permit<T, F>(
        &self,
        permit: InferencePermit,
        inference_timeout: Duration,
        inference: F,
    ) -> HekaResult<T>
    where
        T: Send + 'static,
        F: FnOnce(&mut SentenceEmbedder) -> HekaResult<T> + Send + 'static,
    {
        let model = Arc::clone(&self.model);
        run_blocking_inference(permit, inference_timeout, move || {
            let mut guard = model
                .lock()
                .map_err(|error| HekaError::Embedding(format!("embedder lock: {error}")))?;
            inference(&mut guard)
        })
        .await
    }

    pub(crate) async fn infer<T, F>(
        &self,
        inference_timeout: Duration,
        inference: F,
    ) -> HekaResult<T>
    where
        T: Send + 'static,
        F: FnOnce(&mut SentenceEmbedder) -> HekaResult<T> + Send + 'static,
    {
        let permit = self.acquire_inference(inference_timeout).await?;
        self.infer_with_permit(permit, inference_timeout, inference)
            .await
    }
}

#[derive(Clone)]
struct InferenceGate {
    slot: Arc<Semaphore>,
}

impl InferenceGate {
    fn new() -> Self {
        Self {
            slot: Arc::new(Semaphore::new(1)),
        }
    }

    async fn acquire(&self, wait_timeout: Duration) -> HekaResult<InferencePermit> {
        let permit = tokio::time::timeout(wait_timeout, Arc::clone(&self.slot).acquire_owned())
            .await
            .map_err(|_| {
                HekaError::Embedding(format!(
                    "previous embedding inference is still running after {} seconds",
                    wait_timeout.as_secs_f64()
                ))
            })?
            .map_err(|_| HekaError::Embedding("embedding inference gate closed".to_string()))?;
        Ok(InferencePermit { _permit: permit })
    }
}

pub(crate) struct InferencePermit {
    _permit: OwnedSemaphorePermit,
}

async fn run_blocking_inference<T, F>(
    permit: InferencePermit,
    inference_timeout: Duration,
    inference: F,
) -> HekaResult<T>
where
    T: Send + 'static,
    F: FnOnce() -> HekaResult<T> + Send + 'static,
{
    let task = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        inference()
    });
    tokio::time::timeout(inference_timeout, task)
        .await
        .map_err(|_| {
            HekaError::Embedding(format!(
                "embedding inference exceeded {} second limit",
                inference_timeout.as_secs_f64()
            ))
        })?
        .map_err(|error| HekaError::Embedding(format!("spawn_blocking: {error}")))?
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn timed_out_inference_keeps_single_flight_backpressure_until_it_exits() {
        let gate = InferenceGate::new();
        let first_permit = gate.acquire(Duration::from_secs(1)).await.unwrap();
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let started_count = Arc::new(AtomicUsize::new(0));
        let first_count = Arc::clone(&started_count);

        let first = run_blocking_inference(first_permit, Duration::from_millis(25), move || {
            first_count.fetch_add(1, Ordering::SeqCst);
            started_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            Ok(1usize)
        })
        .await;
        assert!(first.unwrap_err().to_string().contains("exceeded"));
        started_rx.recv_timeout(Duration::from_secs(1)).unwrap();

        // The timed-out blocking task still owns the sole permit. A retry is
        // rejected without starting a second blocking inference.
        for _ in 0..3 {
            let blocked = gate.acquire(Duration::from_millis(25)).await;
            assert!(blocked.err().unwrap().to_string().contains("still running"));
        }
        assert_eq!(started_count.load(Ordering::SeqCst), 1);

        release_tx.send(()).unwrap();
        let second_permit = gate.acquire(Duration::from_secs(1)).await.unwrap();
        let second_count = Arc::clone(&started_count);
        let result = run_blocking_inference(second_permit, Duration::from_secs(1), move || {
            second_count.fetch_add(1, Ordering::SeqCst);
            Ok(2usize)
        })
        .await
        .unwrap();
        assert_eq!(result, 2);
        assert_eq!(started_count.load(Ordering::SeqCst), 2);
    }
}
