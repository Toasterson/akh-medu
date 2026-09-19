//! Errors for the shared ONNX sentence embedder, with miette diagnostics.

use std::path::PathBuf;

use miette::Diagnostic;
use thiserror::Error;

/// Result alias for `akh-embed` operations.
pub type EmbedResult<T> = Result<T, EmbedError>;

/// Errors that can occur while loading or running the sentence embedder.
#[derive(Debug, Error, Diagnostic)]
pub enum EmbedError {
    /// `model.onnx` and/or `tokenizer.json` are missing from the resolved
    /// model directory — either at `SentenceEmbedder::load` time or because
    /// `resolve_model_dir` could not find them and has no auto-download.
    #[error("model files not found in {}. Expected model.onnx and tokenizer.json", dir.display())]
    #[diagnostic(
        code(akh_embed::model_files_missing),
        help(
            "Download from https://huggingface.co/sentence-transformers/{model_name}/tree/main and place model.onnx + tokenizer.json in {}.",
            dir.display()
        )
    )]
    ModelFilesMissing { dir: PathBuf, model_name: String },

    /// Building the ONNX runtime session failed, either constructing the
    /// session builder or committing `model.onnx` to it.
    #[error("building ONNX session: {0}")]
    #[diagnostic(
        code(akh_embed::session_build),
        help(
            "Check that the ONNX runtime shared library is installed and discoverable \
             (see ORT_DYLIB_PATH), and that model.onnx is a valid ONNX file for this ort version."
        )
    )]
    SessionBuild(String),

    /// Loading `tokenizer.json` failed.
    #[error("loading tokenizer: {0}")]
    #[diagnostic(
        code(akh_embed::tokenizer_load),
        help("Check that tokenizer.json is present, valid JSON, and matches the model.")
    )]
    TokenizerLoad(String),

    /// Tokenizing input text failed.
    #[error("tokenization error: {0}")]
    #[diagnostic(
        code(akh_embed::tokenize),
        help("Check that the input text is valid UTF-8 and that tokenizer.json matches the model.")
    )]
    Tokenize(String),

    /// Building an input tensor for inference failed.
    #[error("tensor creation error: {0}")]
    #[diagnostic(
        code(akh_embed::tensor),
        help(
            "This usually indicates a shape mismatch between the input_ids/attention_mask/\
             token_type_ids batches; please file an issue against akh-embed."
        )
    )]
    Tensor(String),

    /// Running the model or extracting its output tensor failed.
    #[error("inference error: {0}")]
    #[diagnostic(
        code(akh_embed::inference),
        help(
            "Check that model.onnx exports input_ids/attention_mask/token_type_ids inputs and a \
             token-embedding output shaped [batch, seq_len, hidden]; re-download the model if it \
             may be corrupted."
        )
    )]
    Inference(String),
}
