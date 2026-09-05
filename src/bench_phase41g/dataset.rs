//! Dataset loading from TOML on disk.

use std::path::Path;

use miette::Diagnostic;
use serde::Deserialize;
use thiserror::Error;

use super::BenchPrompt;

#[derive(Debug, Error, Diagnostic)]
pub enum DatasetError {
    #[error("cannot read dataset file '{path}': {source}")]
    #[diagnostic(
        code(bench::dataset::io),
        help("ensure data/bench/prompt_to_plan/dataset.toml exists in the workspace root")
    )]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("TOML parse error in dataset: {0}")]
    #[diagnostic(code(bench::dataset::toml))]
    Toml(#[from] toml::de::Error),
}

#[derive(Debug, Deserialize)]
struct RawDataset {
    prompts: Vec<BenchPrompt>,
}

/// Parsed benchmark dataset.
#[derive(Debug, Clone)]
pub struct Dataset {
    pub prompts: Vec<BenchPrompt>,
}

impl Dataset {
    pub fn len(&self) -> usize {
        self.prompts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.prompts.is_empty()
    }
}

/// Load the dataset from a TOML file at `path`.
pub fn load_dataset(path: &Path) -> Result<Dataset, DatasetError> {
    let text = std::fs::read_to_string(path).map_err(|e| DatasetError::Io {
        path: path.display().to_string(),
        source: e,
    })?;
    let raw: RawDataset = toml::from_str(&text)?;
    Ok(Dataset {
        prompts: raw.prompts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_bundled_dataset() {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("data/bench/prompt_to_plan/dataset.toml");
        let dataset = load_dataset(&path).expect("dataset should parse cleanly");
        assert!(!dataset.is_empty(), "dataset must have at least one prompt");
        for p in &dataset.prompts {
            assert!(!p.id.is_empty());
            assert!(!p.prompt.is_empty());
            assert!(
                !p.acceptance.is_empty(),
                "prompt '{}' has no predicates",
                p.id
            );
        }
    }
}
