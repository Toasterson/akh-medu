//! Verified provider-free runtime for the shared BabyLLM text + OCR model.
//!
//! The training command writes a candidate pointer.  Offline evaluation uses
//! [`MultimodalModel::load_candidate`]; serving only uses
//! [`MultimodalModel::load_active`], which refuses an artifact without the
//! frozen multimodal certification.  Both paths reconstruct the exact compact
//! visual stem and shared text decoder recorded in the artifact.

use std::io::{BufReader, Cursor};
use std::path::{Component, Path, PathBuf};

use burn::module::Module;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::backend::Backend;
use grug_data::{
    KnowledgeSftRecord, ToolContractSnapshot, letterbox_rgb, read_png_normalized_rgb,
    render_knowledge_prompt_with_system,
};
use serde::Deserialize;
use serde_json::Value;
use thiserror::Error;

use crate::{
    BpeTokenizer, CompactConvVisionStemConfig, CompactVisionTransformer,
    CompactVisionTransformerConfig, CpuBackend, GenerateConfig, MultimodalModelArtifact,
    compact_knowledge_prompt_ids, fingerprint_artifact_file, generate_compact_vision_ocr,
    generate_from_prompt_ids_cached, rgb_batch_to_nchw,
};

const DEFAULT_MAXIMUM_IMAGE_PIXELS: u64 = 16_000_000;

#[derive(Debug, Error)]
pub enum MultimodalRuntimeError {
    #[error("failed to read {path}: {source}")]
    Read {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse {path}: {source}")]
    Parse {
        path: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("invalid multimodal BabyLLM artifact: {0}")]
    InvalidArtifact(String),
    #[error("unsafe artifact path {path:?}: paths must stay inside the bundle")]
    UnsafePath { path: String },
    #[error("artifact hash mismatch for {path}: expected {expected}, got {actual}")]
    HashMismatch {
        path: String,
        expected: String,
        actual: String,
    },
    #[error("unsupported prompt image MIME type {0:?}; native BabyLLM OCR accepts image/png")]
    UnsupportedImageMime(String),
    #[error(transparent)]
    Contract(#[from] grug_data::ToolContractError),
    #[error(transparent)]
    VisionData(#[from] grug_data::VisionTextError),
    #[error(transparent)]
    VisionModel(#[from] crate::VisionError),
    #[error(transparent)]
    Train(#[from] crate::GrugError),
}

#[derive(Debug, Deserialize)]
struct MultimodalCheckpointMetadata {
    schema_version: u32,
    architecture: String,
    update: usize,
    parameter_count: usize,
    complete: bool,
}

/// One decoded image accepted by the native visual path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativePromptImage {
    pub mime_type: String,
    pub bytes: Vec<u8>,
}

impl NativePromptImage {
    pub fn png(bytes: Vec<u8>) -> Self {
        Self {
            mime_type: "image/png".into(),
            bytes,
        }
    }
}

/// Fully verified shared text-and-vision decoder.
pub struct MultimodalModel<B: Backend> {
    artifact: MultimodalModelArtifact,
    contracts: ToolContractSnapshot,
    tokenizer: BpeTokenizer,
    model: CompactVisionTransformer<B>,
    device: B::Device,
    maximum_image_pixels: u64,
}

pub type CpuMultimodalModel = MultimodalModel<CpuBackend>;

#[cfg(any(feature = "gpu", feature = "cuda"))]
pub type GpuMultimodalModel = MultimodalModel<crate::GpuBackend>;

impl MultimodalModel<CpuBackend> {
    /// Load only an exam-certified `multimodal-active.json` pointer.
    pub fn load_active(bundle: impl AsRef<Path>) -> Result<Self, MultimodalRuntimeError> {
        Self::load_manifest_with_device(
            bundle.as_ref(),
            "multimodal-active.json",
            true,
            Default::default(),
        )
    }

    /// Load an unpromoted candidate strictly for offline evaluation.
    pub fn load_candidate(bundle: impl AsRef<Path>) -> Result<Self, MultimodalRuntimeError> {
        Self::load_manifest_with_device(
            bundle.as_ref(),
            "multimodal-candidate.json",
            false,
            Default::default(),
        )
    }
}

#[cfg(any(feature = "gpu", feature = "cuda"))]
impl MultimodalModel<crate::GpuBackend> {
    pub fn load_active_gpu(bundle: impl AsRef<Path>) -> Result<Self, MultimodalRuntimeError> {
        Self::load_manifest_with_device(
            bundle.as_ref(),
            "multimodal-active.json",
            true,
            crate::gpu_device(),
        )
    }

    pub fn load_candidate_gpu(bundle: impl AsRef<Path>) -> Result<Self, MultimodalRuntimeError> {
        Self::load_manifest_with_device(
            bundle.as_ref(),
            "multimodal-candidate.json",
            false,
            crate::gpu_device(),
        )
    }
}

impl<B: Backend> MultimodalModel<B> {
    fn load_manifest_with_device(
        bundle: &Path,
        manifest_name: &str,
        require_certification: bool,
        device: B::Device,
    ) -> Result<Self, MultimodalRuntimeError> {
        let manifest_path = resolve_bundle_child(bundle, manifest_name)?;
        let artifact: MultimodalModelArtifact = serde_json::from_reader(BufReader::new(
            std::fs::File::open(&manifest_path).map_err(|source| MultimodalRuntimeError::Read {
                path: manifest_path.display().to_string(),
                source,
            })?,
        ))
        .map_err(|source| MultimodalRuntimeError::Parse {
            path: manifest_path.display().to_string(),
            source,
        })?;
        artifact
            .verify()
            .map_err(MultimodalRuntimeError::InvalidArtifact)?;
        if require_certification && artifact.certification.is_none() {
            return Err(MultimodalRuntimeError::InvalidArtifact(
                "active multimodal model has no frozen OCR/text-retention certification".into(),
            ));
        }

        let tokenizer_path = resolve_bundle_child(bundle, &artifact.tokenizer_file)?;
        verify_file_hash(&tokenizer_path, &artifact.tokenizer_hash)?;
        let tokenizer = BpeTokenizer::load(&tokenizer_path)?;
        if tokenizer.vocab_size() != artifact.text_shape.vocab_size {
            return Err(MultimodalRuntimeError::InvalidArtifact(format!(
                "tokenizer vocabulary {} does not match artifact shape {}",
                tokenizer.vocab_size(),
                artifact.text_shape.vocab_size
            )));
        }

        let contracts_path = resolve_bundle_child(bundle, &artifact.contracts_file)?;
        let contracts: ToolContractSnapshot = serde_json::from_reader(BufReader::new(
            std::fs::File::open(&contracts_path).map_err(|source| {
                MultimodalRuntimeError::Read {
                    path: contracts_path.display().to_string(),
                    source,
                }
            })?,
        ))
        .map_err(|source| MultimodalRuntimeError::Parse {
            path: contracts_path.display().to_string(),
            source,
        })?;
        contracts.verify_integrity()?;
        if contracts.manifest_hash != artifact.contract_hash {
            return Err(MultimodalRuntimeError::InvalidArtifact(format!(
                "contracts file identifies {}, artifact declares {}",
                contracts.manifest_hash, artifact.contract_hash
            )));
        }

        let checkpoint_dir = resolve_bundle_child(bundle, &artifact.checkpoint_dir)?;
        let model_path = ensure_bundle_descendant(bundle, &checkpoint_dir.join("model.mpk"))?;
        verify_file_hash(&model_path, &artifact.checkpoint_model_hash)?;
        if let Some(expected) = &artifact.checkpoint_optimizer_hash {
            let optimizer_path =
                ensure_bundle_descendant(bundle, &checkpoint_dir.join("optim.mpk"))?;
            verify_file_hash(&optimizer_path, expected)?;
        }
        let metadata_path = ensure_bundle_descendant(bundle, &checkpoint_dir.join("meta.json"))?;
        let metadata: MultimodalCheckpointMetadata = serde_json::from_reader(BufReader::new(
            std::fs::File::open(&metadata_path).map_err(|source| MultimodalRuntimeError::Read {
                path: metadata_path.display().to_string(),
                source,
            })?,
        ))
        .map_err(|source| MultimodalRuntimeError::Parse {
            path: metadata_path.display().to_string(),
            source,
        })?;
        if metadata.schema_version != 1
            || metadata.architecture != "compact_native_ocr_shared_text_decoder_v1"
            || metadata.update != artifact.checkpoint_step
            || metadata.parameter_count != artifact.parameter_count
            || !metadata.complete
        {
            return Err(MultimodalRuntimeError::InvalidArtifact(
                "checkpoint metadata is incomplete or does not match the artifact".into(),
            ));
        }

        let config = CompactVisionTransformerConfig::new(
            artifact.text_shape.config(),
            CompactConvVisionStemConfig::new(
                artifact.vision_shape.input_channels,
                artifact.vision_shape.hidden_channels,
                artifact.text_shape.d_model,
                artifact.vision_shape.max_grid_rows,
                artifact.vision_shape.max_grid_columns,
            ),
        );
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        let model = config
            .init::<B>(&device)
            .load_file(checkpoint_dir.join("model"), &recorder, &device)
            .map_err(|error| {
                MultimodalRuntimeError::InvalidArtifact(format!(
                    "could not load multimodal checkpoint: {error}"
                ))
            })?;
        let actual_parameters = model.num_params();
        if actual_parameters != artifact.parameter_count {
            return Err(MultimodalRuntimeError::InvalidArtifact(format!(
                "loaded model has {actual_parameters} parameters, artifact says {}",
                artifact.parameter_count
            )));
        }

        Ok(Self {
            artifact,
            contracts,
            tokenizer,
            model,
            device,
            maximum_image_pixels: DEFAULT_MAXIMUM_IMAGE_PIXELS,
        })
    }

    pub fn artifact(&self) -> &MultimodalModelArtifact {
        &self.artifact
    }

    pub fn contracts(&self) -> &ToolContractSnapshot {
        &self.contracts
    }

    pub fn with_maximum_image_pixels(
        mut self,
        maximum: u64,
    ) -> Result<Self, MultimodalRuntimeError> {
        if maximum == 0 {
            return Err(MultimodalRuntimeError::InvalidArtifact(
                "maximum image pixels must be positive".into(),
            ));
        }
        self.maximum_image_pixels = maximum;
        Ok(self)
    }

    /// Provider-free OCR over a PNG payload. Common 8-bit screenshot forms are
    /// normalized to RGB; frozen corpus validation remains exact-RGB-only.
    pub fn generate_ocr_image(
        &self,
        image: &NativePromptImage,
        max_new_tokens: usize,
    ) -> Result<String, MultimodalRuntimeError> {
        if image.mime_type.trim().to_ascii_lowercase() != "image/png" {
            return Err(MultimodalRuntimeError::UnsupportedImageMime(
                image.mime_type.clone(),
            ));
        }
        let (width, height, rgb) = read_png_normalized_rgb(
            Cursor::new(image.bytes.as_slice()),
            self.maximum_image_pixels,
        )?;
        self.generate_ocr_rgb(&rgb, width, height, max_new_tokens)
    }

    pub fn generate_ocr_rgb(
        &self,
        rgb: &[u8],
        width: u32,
        height: u32,
        max_new_tokens: usize,
    ) -> Result<String, MultimodalRuntimeError> {
        let maximum = self
            .artifact
            .text_shape
            .max_seq_len
            .saturating_sub(self.artifact.vision_shape.visual_tokens() + 3)
            .max(1);
        if max_new_tokens == 0 || max_new_tokens > maximum {
            return Err(MultimodalRuntimeError::InvalidArtifact(format!(
                "OCR max_new_tokens must be in 1..={maximum}, got {max_new_tokens}"
            )));
        }
        let normalized = letterbox_rgb(
            rgb,
            width,
            height,
            self.artifact.vision_shape.canvas_width as u32,
            self.artifact.vision_shape.canvas_height as u32,
            [248, 248, 248],
        )?;
        let image = rgb_batch_to_nchw::<B>(
            std::slice::from_ref(&normalized),
            self.artifact.vision_shape.canvas_width,
            self.artifact.vision_shape.canvas_height,
            &self.device,
        )?;
        Ok(generate_compact_vision_ocr(
            &self.model,
            image,
            &self.tokenizer,
            max_new_tokens,
            self.artifact.text_shape.max_seq_len,
            &self.device,
        ))
    }

    /// Text/tool inference uses the exact same decoder weights as OCR.
    pub fn generate_turn_with_system(
        &self,
        tools: &[Value],
        conversation: &[Value],
        system_context: Option<&str>,
        max_new_tokens: usize,
    ) -> Result<(String, usize, usize), MultimodalRuntimeError> {
        let maximum = self.artifact.text_shape.max_seq_len.saturating_sub(3);
        if max_new_tokens == 0 || max_new_tokens > maximum {
            return Err(MultimodalRuntimeError::InvalidArtifact(format!(
                "max_new_tokens must be in 1..={maximum}, got {max_new_tokens}"
            )));
        }
        let prompt = render_knowledge_prompt_with_system(tools, conversation, system_context);
        let prompt_ids = self.tokenizer.encode(&prompt)?;
        let config = GenerateConfig {
            max_new_tokens,
            max_seq_len: self.artifact.text_shape.max_seq_len,
        };
        let compacted = compact_knowledge_prompt_ids(&prompt_ids, config);
        let output = generate_from_prompt_ids_cached(
            self.model.transformer(),
            &self.tokenizer,
            &compacted,
            config,
            &self.device,
        );
        let completion_tokens = self.tokenizer.encode(&output)?.len();
        Ok((output, compacted.len() + 2, completion_tokens))
    }

    pub fn generate_records(
        &self,
        records: &[KnowledgeSftRecord],
        max_new_tokens: usize,
    ) -> Result<Vec<String>, MultimodalRuntimeError> {
        records
            .iter()
            .map(|record| {
                self.generate_turn_with_system(
                    &record.tools,
                    &record.conversation,
                    record.system_context.as_deref(),
                    max_new_tokens,
                )
                .map(|(output, _, _)| output)
            })
            .collect()
    }
}

fn resolve_bundle_child(bundle: &Path, relative: &str) -> Result<PathBuf, MultimodalRuntimeError> {
    let relative_path = Path::new(relative);
    if relative_path.as_os_str().is_empty()
        || relative_path
            .components()
            .any(|component| !matches!(component, Component::Normal(_) | Component::CurDir))
    {
        return Err(MultimodalRuntimeError::UnsafePath {
            path: relative.into(),
        });
    }
    let canonical_bundle =
        bundle
            .canonicalize()
            .map_err(|source| MultimodalRuntimeError::Read {
                path: bundle.display().to_string(),
                source,
            })?;
    ensure_bundle_descendant(&canonical_bundle, &canonical_bundle.join(relative_path))
}

fn ensure_bundle_descendant(
    bundle: &Path,
    unresolved: &Path,
) -> Result<PathBuf, MultimodalRuntimeError> {
    let canonical_bundle =
        bundle
            .canonicalize()
            .map_err(|source| MultimodalRuntimeError::Read {
                path: bundle.display().to_string(),
                source,
            })?;
    let child = unresolved
        .canonicalize()
        .map_err(|source| MultimodalRuntimeError::Read {
            path: unresolved.display().to_string(),
            source,
        })?;
    if !child.starts_with(&canonical_bundle) {
        return Err(MultimodalRuntimeError::UnsafePath {
            path: unresolved.display().to_string(),
        });
    }
    Ok(child)
}

fn verify_file_hash(path: &Path, expected: &str) -> Result<(), MultimodalRuntimeError> {
    let actual =
        fingerprint_artifact_file(path).map_err(|source| MultimodalRuntimeError::Read {
            path: path.display().to_string(),
            source,
        })?;
    if actual != expected {
        return Err(MultimodalRuntimeError::HashMismatch {
            path: path.display().to_string(),
            expected: expected.into(),
            actual,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multimodal_artifact_paths_reject_parent_and_absolute_components() {
        let directory = tempfile::tempdir().unwrap();
        assert!(matches!(
            resolve_bundle_child(directory.path(), "../outside"),
            Err(MultimodalRuntimeError::UnsafePath { .. })
        ));
        assert!(matches!(
            resolve_bundle_child(directory.path(), "/tmp/outside"),
            Err(MultimodalRuntimeError::UnsafePath { .. })
        ));
    }

    #[test]
    fn native_prompt_image_records_png_mime() {
        assert_eq!(NativePromptImage::png(vec![1, 2]).mime_type, "image/png");
    }
}
