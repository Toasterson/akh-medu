//! Compartment manager: discover, load, unload, activate, deactivate compartments.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::RwLock;

use super::error::{CompartmentError, CompartmentResult};
use super::manifest::{CompartmentKind, CompartmentManifest, CompartmentState};
use super::psyche::Psyche;

/// A compartment that has been discovered and possibly loaded.
#[derive(Debug)]
struct LoadedCompartment {
    manifest: CompartmentManifest,
    state: CompartmentState,
    triple_count: usize,
    /// Named graph IRI for this compartment's triples.
    #[allow(dead_code)] // Will be used when SPARQL named graph routing is implemented
    graph_name: String,
}

/// Manages the lifecycle of knowledge compartments.
pub struct CompartmentManager {
    compartments_dir: PathBuf,
    compartments: RwLock<HashMap<String, LoadedCompartment>>,
    /// The loaded Jungian psyche (deserialized from psyche compartment).
    psyche: RwLock<Option<Psyche>>,
}

impl std::fmt::Debug for CompartmentManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let count = self.compartments.read().map(|c| c.len()).unwrap_or(0);
        f.debug_struct("CompartmentManager")
            .field("compartments_dir", &self.compartments_dir)
            .field("compartment_count", &count)
            .finish()
    }
}

/// Namespace for compartment named graphs in Oxigraph.
pub const COMPARTMENT_NS: &str = "https://akh-medu.dev/compartment/";

impl CompartmentManager {
    /// Create a new compartment manager scanning the given directory.
    pub fn new(compartments_dir: PathBuf) -> Self {
        Self {
            compartments_dir,
            compartments: RwLock::new(HashMap::new()),
            psyche: RwLock::new(None),
        }
    }

    /// Scan the compartments directory for `compartment.toml` files.
    ///
    /// Each subdirectory containing a `compartment.toml` is registered as Dormant.
    /// Returns the number of newly discovered compartments.
    pub fn discover(&self) -> CompartmentResult<usize> {
        if !self.compartments_dir.exists() {
            return Ok(0);
        }

        let entries =
            std::fs::read_dir(&self.compartments_dir).map_err(|e| CompartmentError::Io {
                id: "<discovery>".into(),
                source: e,
            })?;

        let mut count = 0;
        let mut compartments = self
            .compartments
            .write()
            .expect("compartments lock poisoned");

        for entry in entries {
            let entry = entry.map_err(|e| CompartmentError::Io {
                id: "<discovery>".into(),
                source: e,
            })?;
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }

            let manifest_path = path.join("compartment.toml");
            if !manifest_path.exists() {
                continue;
            }

            let content =
                std::fs::read_to_string(&manifest_path).map_err(|e| CompartmentError::Io {
                    id: path.display().to_string(),
                    source: e,
                })?;

            let manifest: CompartmentManifest =
                toml::from_str(&content).map_err(|e| CompartmentError::InvalidManifest {
                    path: manifest_path.display().to_string(),
                    message: e.to_string(),
                })?;

            let id = manifest.id.clone();
            if let std::collections::hash_map::Entry::Vacant(entry) = compartments.entry(id) {
                let graph_name = format!("{COMPARTMENT_NS}{}", entry.key());
                entry.insert(LoadedCompartment {
                    manifest,
                    state: CompartmentState::Dormant,
                    triple_count: 0,
                    graph_name,
                });
                count += 1;
            }
        }

        Ok(count)
    }

    /// Load a compartment's triples into the knowledge graph.
    ///
    /// For the "psyche" core compartment, also deserializes the `Psyche` struct
    /// from `psyche.toml` in the compartment directory.
    pub fn load(&self, id: &str, engine: &crate::engine::Engine) -> CompartmentResult<()> {
        let compartment_dir = self.compartments_dir.join(id);

        // Load psyche if this is the psyche compartment.
        if id == "psyche" {
            let psyche_path = compartment_dir.join("psyche.toml");
            let psyche = if psyche_path.exists() {
                let content =
                    std::fs::read_to_string(&psyche_path).map_err(|e| CompartmentError::Io {
                        id: id.into(),
                        source: e,
                    })?;
                toml::from_str::<Psyche>(&content).unwrap_or_default()
            } else {
                Psyche::default()
            };
            *self.psyche.write().expect("psyche lock poisoned") = Some(psyche);
        }

        let mut compartments = self
            .compartments
            .write()
            .expect("compartments lock poisoned");
        let compartment = compartments
            .get_mut(id)
            .ok_or_else(|| CompartmentError::NotFound { id: id.into() })?;

        if compartment.state != CompartmentState::Dormant {
            return Err(CompartmentError::AlreadyLoaded { id: id.into() });
        }

        // Load triples if specified.
        let mut triple_count = 0;
        if let Some(ref triples_file) = compartment.manifest.triples_file {
            let triples_path = compartment_dir.join(triples_file);
            if triples_path.exists() {
                let content =
                    std::fs::read_to_string(&triples_path).map_err(|e| CompartmentError::Io {
                        id: id.into(),
                        source: e,
                    })?;

                let raw: Vec<serde_json::Value> = serde_json::from_str(&content).map_err(|e| {
                    CompartmentError::InvalidManifest {
                        path: triples_path.display().to_string(),
                        message: format!("triples parse error: {e}"),
                    }
                })?;

                let compartment_id = id.to_string();
                for val in &raw {
                    let subject = val["subject"].as_str().unwrap_or("");
                    let predicate = val["predicate"].as_str().unwrap_or("");
                    let object = val["object"].as_str().unwrap_or("");
                    let confidence = val["confidence"].as_f64().unwrap_or(1.0) as f32;

                    if !subject.is_empty() && !predicate.is_empty() && !object.is_empty() {
                        let s = engine.resolve_or_create_entity(subject).map_err(|_| {
                            CompartmentError::InvalidManifest {
                                path: triples_path.display().to_string(),
                                message: format!("failed to resolve subject: {subject}"),
                            }
                        })?;
                        let p = engine.resolve_or_create_relation(predicate).map_err(|_| {
                            CompartmentError::InvalidManifest {
                                path: triples_path.display().to_string(),
                                message: format!("failed to resolve predicate: {predicate}"),
                            }
                        })?;
                        let o = engine.resolve_or_create_entity(object).map_err(|_| {
                            CompartmentError::InvalidManifest {
                                path: triples_path.display().to_string(),
                                message: format!("failed to resolve object: {object}"),
                            }
                        })?;

                        let triple = crate::graph::Triple::new(s, p, o)
                            .with_confidence(confidence)
                            .with_compartment(compartment_id.clone());
                        let _ = engine.add_triple(&triple);
                        triple_count += 1;
                    }
                }
            }
        }

        compartment.triple_count = triple_count;
        compartment.state = CompartmentState::Loaded;
        Ok(())
    }

    /// Unload a compartment's triples from the knowledge graph.
    pub fn unload(&self, id: &str, engine: &crate::engine::Engine) -> CompartmentResult<()> {
        let mut compartments = self
            .compartments
            .write()
            .expect("compartments lock poisoned");
        let compartment = compartments
            .get_mut(id)
            .ok_or_else(|| CompartmentError::NotFound { id: id.into() })?;

        if compartment.state == CompartmentState::Dormant {
            return Ok(());
        }

        // Remove triples from the named graph in SPARQL store.
        if let Some(sparql) = engine.sparql() {
            let _ = sparql.remove_graph(id);
        }

        if id == "psyche" {
            *self.psyche.write().expect("psyche lock poisoned") = None;
        }

        compartment.state = CompartmentState::Dormant;
        compartment.triple_count = 0;
        Ok(())
    }

    /// Mark a loaded compartment as Active (influencing OODA loop).
    pub fn activate(&self, id: &str) -> CompartmentResult<()> {
        let mut compartments = self
            .compartments
            .write()
            .expect("compartments lock poisoned");
        let compartment = compartments
            .get_mut(id)
            .ok_or_else(|| CompartmentError::NotFound { id: id.into() })?;

        if compartment.state == CompartmentState::Dormant {
            return Err(CompartmentError::NotFound { id: id.into() });
        }
        compartment.state = CompartmentState::Active;
        Ok(())
    }

    /// Mark an active compartment as merely Loaded (no longer influencing OODA).
    pub fn deactivate(&self, id: &str) -> CompartmentResult<()> {
        let mut compartments = self
            .compartments
            .write()
            .expect("compartments lock poisoned");
        let compartment = compartments
            .get_mut(id)
            .ok_or_else(|| CompartmentError::NotFound { id: id.into() })?;

        if compartment.state == CompartmentState::Active {
            compartment.state = CompartmentState::Loaded;
        }
        Ok(())
    }

    /// List all active compartment manifests.
    pub fn active_compartments(&self) -> Vec<CompartmentManifest> {
        let compartments = self
            .compartments
            .read()
            .expect("compartments lock poisoned");
        compartments
            .values()
            .filter(|c| c.state == CompartmentState::Active)
            .map(|c| c.manifest.clone())
            .collect()
    }

    /// Get the manifest for a specific compartment.
    pub fn get(&self, id: &str) -> Option<CompartmentManifest> {
        let compartments = self
            .compartments
            .read()
            .expect("compartments lock poisoned");
        compartments.get(id).map(|c| c.manifest.clone())
    }

    /// List compartments of a specific kind.
    pub fn compartments_by_kind(&self, kind: CompartmentKind) -> Vec<CompartmentManifest> {
        let compartments = self
            .compartments
            .read()
            .expect("compartments lock poisoned");
        compartments
            .values()
            .filter(|c| c.manifest.kind == kind)
            .map(|c| c.manifest.clone())
            .collect()
    }

    /// Get the loaded psyche (if the psyche compartment has been loaded).
    pub fn psyche(&self) -> Option<Psyche> {
        self.psyche.read().expect("psyche lock poisoned").clone()
    }

    /// Update the stored psyche (after evolution).
    ///
    /// Returns `PsycheImmutable` if the existing psyche is awakened.
    /// Use [`force_set_psyche`] for admin override.
    pub fn set_psyche(&self, psyche: Psyche) -> CompartmentResult<()> {
        let mut guard = self.psyche.write().expect("psyche lock poisoned");
        if let Some(ref existing) = *guard {
            if existing.is_awakened() {
                return Err(CompartmentError::PsycheImmutable);
            }
        }
        self.persist_psyche_to_disk(&psyche)?;
        *guard = Some(psyche);
        Ok(())
    }

    /// Unconditionally replace the stored psyche (admin override).
    pub fn force_set_psyche(&self, psyche: Psyche) {
        if let Err(e) = self.persist_psyche_to_disk(&psyche) {
            tracing::warn!(error = %e, "failed to persist psyche to disk");
        }
        *self.psyche.write().expect("psyche lock poisoned") = Some(psyche);
    }

    /// Persist the psyche to `compartments/psyche/psyche.toml` on disk.
    ///
    /// Creates the compartment directory and manifest if they don't exist,
    /// ensuring the psyche survives process restarts.
    fn persist_psyche_to_disk(&self, psyche: &Psyche) -> CompartmentResult<()> {
        let psyche_dir = self.compartments_dir.join("psyche");
        std::fs::create_dir_all(&psyche_dir).map_err(|e| CompartmentError::Io {
            id: "psyche".into(),
            source: e,
        })?;

        // Ensure compartment.toml manifest exists so discover() finds it.
        let manifest_path = psyche_dir.join("compartment.toml");
        if !manifest_path.exists() {
            let manifest = CompartmentManifest {
                id: "psyche".into(),
                name: "Psyche".into(),
                kind: CompartmentKind::Core,
                description: "Jungian psyche configuration — persona, shadow, archetypes."
                    .into(),
                triples_file: None,
                rules_file: None,
                grammar_ref: None,
                tags: vec!["identity".into(), "core".into()],
            };
            let toml_str = toml::to_string_pretty(&manifest).map_err(|e| {
                CompartmentError::InvalidManifest {
                    path: manifest_path.display().to_string(),
                    message: e.to_string(),
                }
            })?;
            std::fs::write(&manifest_path, toml_str).map_err(|e| CompartmentError::Io {
                id: "psyche".into(),
                source: e,
            })?;
        }

        // Write psyche.toml.
        let psyche_path = psyche_dir.join("psyche.toml");
        let toml_str =
            toml::to_string_pretty(psyche).map_err(|e| CompartmentError::InvalidManifest {
                path: psyche_path.display().to_string(),
                message: e.to_string(),
            })?;
        std::fs::write(&psyche_path, toml_str).map_err(|e| CompartmentError::Io {
            id: "psyche".into(),
            source: e,
        })?;

        tracing::info!("persisted psyche to {}", psyche_path.display());
        Ok(())
    }

    /// List the IDs of all discovered but not-yet-loaded (Dormant) compartments.
    pub fn dormant_ids(&self) -> Vec<String> {
        let compartments = self
            .compartments
            .read()
            .expect("compartments lock poisoned");
        compartments
            .iter()
            .filter(|(_, c)| c.state == CompartmentState::Dormant)
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// List all discovered compartments with their current state.
    pub fn all_compartments(&self) -> Vec<(CompartmentManifest, CompartmentState, usize)> {
        let compartments = self
            .compartments
            .read()
            .expect("compartments lock poisoned");
        compartments
            .values()
            .map(|c| (c.manifest.clone(), c.state, c.triple_count))
            .collect()
    }

    /// Get the compartments directory path.
    pub fn compartments_dir(&self) -> &std::path::Path {
        &self.compartments_dir
    }
}
