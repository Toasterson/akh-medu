//! Locate a model's multimodal projector, including content-addressed stores.
//!
//! The Gemma sidecar convention names the projector after the model
//! (`<stem>.mmproj.gguf`). The exact Ptah model is an ollama blob whose file
//! name is its own digest, so the projector cannot be derived from it: the two
//! blobs are related only through the manifest that lists both layers. This
//! resolves either layout without guessing, and never rewrites model identity.
use std::path::{Path, PathBuf};

/// Media type of the projector layer in an ollama manifest.
const PROJECTOR_MEDIA_TYPE: &str = "application/vnd.ollama.image.projector";
/// Media type of the weights layer, used to match the manifest to this model.
const MODEL_MEDIA_TYPE: &str = "application/vnd.ollama.image.model";

/// The sidecar next to the model, when that convention applies.
pub fn sidecar(model_path: &Path) -> Option<PathBuf> {
    let path = model_path.with_extension("mmproj.gguf");
    path.exists().then_some(path)
}

/// Split an ollama blob path into its store root and `sha256:<hex>` digest.
///
/// Returns `None` for anything that is not `<root>/blobs/sha256-<hex>`, so an
/// ordinary model file never triggers a manifest search.
fn blob_identity(model_path: &Path) -> Option<(PathBuf, String)> {
    let name = model_path.file_name()?.to_str()?;
    let hex = name.strip_prefix("sha256-")?;
    if hex.len() != 64 || !hex.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    let blobs = model_path.parent()?;
    if blobs.file_name()? != "blobs" {
        return None;
    }
    Some((blobs.parent()?.to_path_buf(), format!("sha256:{hex}")))
}

/// Every manifest file under `<root>/manifests`, depth-first.
fn manifest_files(root: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut pending = vec![root.join("manifests")];
    while let Some(dir) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                pending.push(path);
            } else {
                found.push(path);
            }
        }
    }
    found
}

/// Digest of the projector layer in a manifest whose model layer is `model_digest`.
fn projector_digest(manifest: &Path, model_digest: &str) -> Option<String> {
    let text = std::fs::read_to_string(manifest).ok()?;
    let value: serde_json::Value = serde_json::from_str(&text).ok()?;
    let layers = value.get("layers")?.as_array()?;
    let describes_this_model = layers.iter().any(|layer| {
        layer.get("mediaType").and_then(|m| m.as_str()) == Some(MODEL_MEDIA_TYPE)
            && layer.get("digest").and_then(|d| d.as_str()) == Some(model_digest)
    });
    if !describes_this_model {
        return None;
    }
    layers
        .iter()
        .find(|layer| {
            layer.get("mediaType").and_then(|m| m.as_str()) == Some(PROJECTOR_MEDIA_TYPE)
        })?
        .get("digest")?
        .as_str()
        .map(str::to_owned)
}

/// The projector blob shipped alongside `model_path` in a content-addressed store.
///
/// A manifest that lists this model but no projector is not an error: a
/// text-only variant of the same weights exists (`dflash2-100k` ships a draft
/// layer instead), so the search continues rather than concluding absence.
pub fn companion_blob(model_path: &Path) -> Option<PathBuf> {
    let (root, model_digest) = blob_identity(model_path)?;
    for manifest in manifest_files(&root) {
        let Some(digest) = projector_digest(&manifest, &model_digest) else {
            continue;
        };
        let blob = root.join("blobs").join(digest.replace(':', "-"));
        if blob.exists() {
            return Some(blob);
        }
    }
    None
}

/// The projector for `model_path`, sidecar first, then the content-addressed store.
pub fn locate(model_path: &Path) -> Option<PathBuf> {
    sidecar(model_path).or_else(|| companion_blob(model_path))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store(root: &Path, digests: &[(&str, &str)], manifest: &str) {
        std::fs::create_dir_all(root.join("blobs")).unwrap();
        std::fs::create_dir_all(root.join("manifests/registry/library/model")).unwrap();
        for (name, body) in digests {
            std::fs::write(root.join("blobs").join(name), body).unwrap();
        }
        std::fs::write(
            root.join("manifests/registry/library/model/stable"),
            manifest,
        )
        .unwrap();
    }

    const MODEL: &str = "sha256-1111111111111111111111111111111111111111111111111111111111111111";
    const PROJ: &str = "sha256-2222222222222222222222222222222222222222222222222222222222222222";

    fn manifest(model: &str, extra: &str) -> String {
        format!(
            r#"{{"layers":[{{"mediaType":"application/vnd.ollama.image.model","digest":"{}"}}{}]}}"#,
            model.replace('-', ":"),
            extra
        )
    }

    #[test]
    fn finds_the_projector_listed_beside_this_model_in_a_manifest() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        store(
            root,
            &[(MODEL, "weights"), (PROJ, "projector")],
            &manifest(
                MODEL,
                &format!(
                    r#",{{"mediaType":"application/vnd.ollama.image.projector","digest":"{}"}}"#,
                    PROJ.replace('-', ":")
                ),
            ),
        );
        assert_eq!(
            locate(&root.join("blobs").join(MODEL)),
            Some(root.join("blobs").join(PROJ))
        );
    }

    #[test]
    fn a_text_only_manifest_for_the_same_weights_is_not_an_answer() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        // A draft layer, as the dflash2-100k variant ships: same model, no projector.
        store(
            root,
            &[(MODEL, "weights")],
            &manifest(
                MODEL,
                r#",{"mediaType":"application/vnd.ollama.image.draft","digest":"sha256:dead"}"#,
            ),
        );
        assert_eq!(locate(&root.join("blobs").join(MODEL)), None);
    }

    #[test]
    fn a_manifest_for_a_different_model_never_contributes_its_projector() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let other = "sha256-3333333333333333333333333333333333333333333333333333333333333333";
        store(
            root,
            &[(MODEL, "weights"), (other, "other"), (PROJ, "projector")],
            &manifest(
                other,
                &format!(
                    r#",{{"mediaType":"application/vnd.ollama.image.projector","digest":"{}"}}"#,
                    PROJ.replace('-', ":")
                ),
            ),
        );
        assert_eq!(locate(&root.join("blobs").join(MODEL)), None);
    }

    /// Artifact test: resolves the real pinned Ptah projector. Ignored by
    /// default because it depends on the local ollama store.
    #[test]
    #[ignore]
    fn resolves_the_pinned_ptah_projector() {
        let model = Path::new(
            "/usr/share/ollama/.ollama/models/blobs/\
             sha256-3445102e9cde5d562508642c100a2f5ac3368a5a3f748442811d7a95daee3bec",
        );
        let found = locate(model).expect("pinned Ptah projector");
        assert_eq!(
            found.file_name().unwrap(),
            "sha256-add205b7bfdb3f71f6da36b0a82aa20928dd829a920878c602628cdfbebc5288"
        );
        assert_eq!(found.metadata().unwrap().len(), 931_145_984);
    }

    #[test]
    fn the_sidecar_convention_still_wins_and_ordinary_paths_skip_the_search() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("gemma4-12b.gguf");
        std::fs::write(&model, "weights").unwrap();
        assert_eq!(locate(&model), None);
        let sidecar = dir.path().join("gemma4-12b.mmproj.gguf");
        std::fs::write(&sidecar, "projector").unwrap();
        assert_eq!(locate(&model), Some(sidecar));
        // Not a blob layout: no manifest search is attempted.
        assert_eq!(blob_identity(&model), None);
    }
}
