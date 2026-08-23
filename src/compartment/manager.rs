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

/// Whether the live edge for `triple` is already backed by the exact pack
/// provenance that would be written for it now.
///
/// Stable identity proves the live edge, while the ledger record proves the
/// producer and binds that edge's object and subject.  Checking all of them is
/// what makes it safe for a deferred compartment load to preserve volatile edge
/// metadata restored by cold wake instead of publishing a replacement record.
pub(super) fn has_exact_live_pack_provenance(
    engine: &crate::engine::Engine,
    triple: &crate::graph::Triple,
    compartment_id: &str,
    source_file: &str,
) -> bool {
    if triple.compartment_id.as_deref() != Some(compartment_id) {
        return false;
    }

    let identity = engine.triple_identity(triple);
    let Some(existing) = engine.triple_by_identity(&identity) else {
        return false;
    };
    let Some(provenance_id) = existing
        .provenance_id
        .and_then(crate::provenance::ProvenanceId::new)
    else {
        return false;
    };
    let Ok(record) = engine.get_provenance(provenance_id) else {
        return false;
    };

    record.derived_id == triple.object
        && record.sources.as_slice() == [triple.subject]
        && matches!(
            record.kind,
            crate::provenance::DerivationKind::CompartmentLoaded {
                compartment_id: ref restored_compartment,
                source_file: ref restored_source,
            } if restored_compartment == compartment_id && restored_source == source_file
        )
}

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
            .unwrap_or_else(|e| e.into_inner());

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
    /// from `psyche.toml` in the compartment directory and translates its fields
    /// (archetypes, shadow patterns, persona traits, individuation level) into
    /// KG triples so the reasoning pipeline can inspect identity state at runtime.
    pub fn load(&self, id: &str, engine: &crate::engine::Engine) -> CompartmentResult<()> {
        let compartment_dir = self.compartments_dir.join(id);
        let mut psyche_triple_count = 0usize;

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

            // Translate persona metadata into KG triples before storing the psyche.
            psyche_triple_count = psyche.inject_into_kg(engine, id).map_err(|e| {
                CompartmentError::InvalidManifest {
                    path: compartment_dir.display().to_string(),
                    message: format!("failed to inject psyche triples into KG: {e}"),
                }
            })?;

            *self.psyche.write().unwrap_or_else(|e| e.into_inner()) = Some(psyche);
        }

        let mut compartments = self
            .compartments
            .write()
            .unwrap_or_else(|e| e.into_inner());
        let compartment = compartments
            .get_mut(id)
            .ok_or_else(|| CompartmentError::NotFound { id: id.into() })?;

        if compartment.state != CompartmentState::Dormant {
            return Err(CompartmentError::AlreadyLoaded { id: id.into() });
        }

        // Load triples if specified; start the count from any psyche-injected triples.
        let mut triple_count = psyche_triple_count;
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

                        // Cold wake restores live facts (including their volatile
                        // provenance/timestamp metadata) before deferred
                        // compartment loading. Re-inserting an identical fact via
                        // `add_triples_with_provenance` would first mint a redundant
                        // ledger record, then overwrite that restored metadata on
                        // the idempotent graph edge. The stable identity includes
                        // the compartment. Only an edge whose live ledger record
                        // proves this exact pack/file is safe to preserve; an
                        // untracked or differently sourced collision still needs
                        // the normal tracked insertion path.
                        let already_tracked_by_this_pack = has_exact_live_pack_provenance(
                            engine,
                            &triple,
                            &compartment_id,
                            triples_file,
                        );
                        if !already_tracked_by_this_pack {
                            let mut triple = [triple];
                            let mut provenance = crate::provenance::ProvenanceRecord::new(
                                o,
                                crate::provenance::DerivationKind::CompartmentLoaded {
                                    compartment_id: compartment_id.clone(),
                                    source_file: triples_file.clone(),
                                },
                            )
                            .with_sources(vec![s])
                            .with_confidence(confidence);
                            engine
                                .add_triples_with_provenance(&mut triple, &mut provenance)
                                .map_err(|e| CompartmentError::InvalidManifest {
                                    path: triples_path.display().to_string(),
                                    message: format!("failed to add tracked triple: {e}"),
                                })?;
                        }
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
            .unwrap_or_else(|e| e.into_inner());
        let compartment = compartments
            .get_mut(id)
            .ok_or_else(|| CompartmentError::NotFound { id: id.into() })?;

        if compartment.state == CompartmentState::Dormant {
            return Ok(());
        }

        // Remove all KG triples tagged with this compartment (includes
        // psyche-injected triples) and clear its named SPARQL graph under the
        // Engine's canonical mutation/quiescence checkpoint.
        engine
            .remove_triples_in_compartment_checked(id)
            .map_err(|error| CompartmentError::InvalidManifest {
                path: id.to_string(),
                message: format!("failed to unload compartment: {error}"),
            })?;

        if id == "psyche" {
            *self.psyche.write().unwrap_or_else(|e| e.into_inner()) = None;
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
            .unwrap_or_else(|e| e.into_inner());
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
            .unwrap_or_else(|e| e.into_inner());
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
            .unwrap_or_else(|e| e.into_inner());
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
            .unwrap_or_else(|e| e.into_inner());
        compartments.get(id).map(|c| c.manifest.clone())
    }

    /// List compartments of a specific kind.
    pub fn compartments_by_kind(&self, kind: CompartmentKind) -> Vec<CompartmentManifest> {
        let compartments = self
            .compartments
            .read()
            .unwrap_or_else(|e| e.into_inner());
        compartments
            .values()
            .filter(|c| c.manifest.kind == kind)
            .map(|c| c.manifest.clone())
            .collect()
    }

    /// Get the loaded psyche (if the psyche compartment has been loaded).
    pub fn psyche(&self) -> Option<Psyche> {
        self.psyche
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Update the stored psyche (after evolution).
    ///
    /// Returns `PsycheImmutable` if the existing psyche is awakened.
    /// Use [`force_set_psyche`] for admin override.
    pub fn set_psyche(&self, psyche: Psyche) -> CompartmentResult<()> {
        let mut guard = self.psyche.write().unwrap_or_else(|e| e.into_inner());
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
        *self.psyche.write().unwrap_or_else(|e| e.into_inner()) = Some(psyche);
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
                description: "Jungian psyche configuration — persona, shadow, archetypes.".into(),
                triples_file: None,
                rules_file: None,
                grammar_ref: None,
                tags: vec!["identity".into(), "core".into()],
                parent: None,
                scope: super::manifest::CompartmentScope::Public,
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
            .unwrap_or_else(|e| e.into_inner());
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
            .unwrap_or_else(|e| e.into_inner());
        compartments
            .values()
            .map(|c| (c.manifest.clone(), c.state, c.triple_count))
            .collect()
    }

    /// Get the compartments directory path.
    pub fn compartments_dir(&self) -> &std::path::Path {
        &self.compartments_dir
    }

    /// List all compartments whose `parent` matches the given id.
    pub fn compartments_by_parent(&self, parent_id: &str) -> Vec<CompartmentManifest> {
        let compartments = self
            .compartments
            .read()
            .expect("compartments lock poisoned");
        compartments
            .values()
            .filter(|c| c.manifest.parent.as_deref() == Some(parent_id))
            .map(|c| c.manifest.clone())
            .collect()
    }

    /// Create a new compartment on disk and register it as Dormant.
    ///
    /// Writes `compartment.toml` into `<compartments_dir>/<id>/` and calls
    /// `discover()` to register it. Returns `AlreadyExists` if the directory
    /// already contains a manifest.
    pub fn create_compartment(&self, manifest: CompartmentManifest) -> CompartmentResult<()> {
        let dir = self.compartments_dir.join(&manifest.id);
        let manifest_path = dir.join("compartment.toml");

        if manifest_path.exists() {
            return Err(super::error::CompartmentError::AlreadyExists {
                id: manifest.id.clone(),
            });
        }

        std::fs::create_dir_all(&dir).map_err(|e| super::error::CompartmentError::Io {
            id: manifest.id.clone(),
            source: e,
        })?;

        let toml_str = toml::to_string_pretty(&manifest).map_err(|e| {
            super::error::CompartmentError::InvalidManifest {
                path: manifest_path.display().to_string(),
                message: e.to_string(),
            }
        })?;

        std::fs::write(&manifest_path, toml_str).map_err(|e| {
            super::error::CompartmentError::Io {
                id: manifest.id.clone(),
                source: e,
            }
        })?;

        // Register the new compartment without re-scanning the whole directory.
        let id = manifest.id.clone();
        let graph_name = format!("{COMPARTMENT_NS}{}", &id);
        self.compartments
            .write()
            .expect("compartments lock poisoned")
            .entry(id)
            .or_insert_with(|| LoadedCompartment {
                manifest,
                state: CompartmentState::Dormant,
                triple_count: 0,
                graph_name,
            });

        Ok(())
    }

    /// Write the Ennead role to the psyche compartment.
    ///
    /// Persists the role in two places for inter-agent discovery:
    /// 1. `compartments/psyche/ennead-role` — plain-text file for filesystem scanning.
    /// 2. `psyche.toml` — `ennead_role` field in the serialised `Psyche`.
    ///
    /// If the psyche compartment directory doesn't exist yet it is created along
    /// with a minimal `compartment.toml` manifest, mirroring what
    /// `persist_psyche_to_disk` does.
    ///
    /// Returns the previously-assigned role if one was already present, or `None`
    /// if this is a fresh assignment.
    pub fn write_ennead_role_file(&self, role: &str) -> CompartmentResult<Option<String>> {
        let psyche_dir = self.compartments_dir.join("psyche");
        std::fs::create_dir_all(&psyche_dir).map_err(|e| CompartmentError::Io {
            id: "psyche".into(),
            source: e,
        })?;

        // Ensure compartment.toml exists so discover() can find the compartment.
        let manifest_path = psyche_dir.join("compartment.toml");
        if !manifest_path.exists() {
            let manifest = CompartmentManifest {
                id: "psyche".into(),
                name: "Psyche".into(),
                kind: CompartmentKind::Core,
                description: "Jungian psyche configuration — persona, shadow, archetypes.".into(),
                triples_file: None,
                rules_file: None,
                grammar_ref: None,
                tags: vec!["identity".into(), "core".into()],
                parent: None,
                scope: super::manifest::CompartmentScope::Public,
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

        // Check whether an ennead-role file already exists.
        let role_file_path = psyche_dir.join("ennead-role");
        let existing_role = if role_file_path.exists() {
            std::fs::read_to_string(&role_file_path)
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
        } else {
            None
        };

        // Write the plain-text role file.
        std::fs::write(&role_file_path, role).map_err(|e| CompartmentError::Io {
            id: "psyche".into(),
            source: e,
        })?;

        // Update the in-memory psyche and re-persist psyche.toml.
        let mut guard = self.psyche.write().unwrap_or_else(|e| e.into_inner());
        let psyche = guard.get_or_insert_with(Psyche::default);
        psyche.ennead_role = Some(role.to_string());
        let updated = psyche.clone();
        drop(guard);
        self.persist_psyche_to_disk(&updated)?;

        tracing::info!(
            role,
            "wrote ennead-role file to {}",
            role_file_path.display()
        );
        Ok(existing_role)
    }

    /// Read the Ennead role from the psyche compartment's `ennead-role` file.
    ///
    /// Returns `None` if the file does not exist or is empty.
    pub fn read_ennead_role_file(&self) -> Option<String> {
        let role_path = self.compartments_dir.join("psyche").join("ennead-role");
        std::fs::read_to_string(role_path)
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compartment::manifest::{CompartmentKind, CompartmentScope};

    fn tmp_manager() -> (tempfile::TempDir, CompartmentManager) {
        let tmp = tempfile::TempDir::new().unwrap();
        let mgr = CompartmentManager::new(tmp.path().to_path_buf());
        (tmp, mgr)
    }

    fn workspace_manifest(id: &str) -> CompartmentManifest {
        CompartmentManifest {
            id: id.into(),
            name: format!("{id} Workspace"),
            kind: CompartmentKind::Workspace,
            description: "test workspace compartment".into(),
            triples_file: None,
            rules_file: None,
            grammar_ref: None,
            tags: vec!["test".into()],
            parent: None,
            scope: CompartmentScope::Private,
        }
    }

    fn child_manifest(id: &str, parent: &str) -> CompartmentManifest {
        CompartmentManifest {
            id: id.into(),
            name: format!("{id} child"),
            kind: CompartmentKind::Project,
            description: "child compartment".into(),
            triples_file: None,
            rules_file: None,
            grammar_ref: None,
            tags: vec![],
            parent: Some(parent.into()),
            scope: CompartmentScope::Protected,
        }
    }

    #[test]
    fn create_compartment_writes_toml_and_registers() {
        let (_tmp, mgr) = tmp_manager();
        let m = workspace_manifest("tecton");
        mgr.create_compartment(m).unwrap();

        // Should appear as Dormant.
        let all = mgr.all_compartments();
        assert_eq!(all.len(), 1);
        let (manifest, state, _tc) = &all[0];
        assert_eq!(manifest.id, "tecton");
        assert_eq!(manifest.kind, CompartmentKind::Workspace);
        assert_eq!(manifest.scope, CompartmentScope::Private);
        assert_eq!(*state, CompartmentState::Dormant);
    }

    #[test]
    fn create_compartment_duplicate_returns_error() {
        let (_tmp, mgr) = tmp_manager();
        mgr.create_compartment(workspace_manifest("alpha")).unwrap();
        let err = mgr.create_compartment(workspace_manifest("alpha")).unwrap_err();
        assert!(matches!(err, super::CompartmentError::AlreadyExists { .. }));
    }

    #[test]
    fn compartments_by_parent_filters_correctly() {
        let (_tmp, mgr) = tmp_manager();
        mgr.create_compartment(workspace_manifest("root")).unwrap();
        mgr.create_compartment(child_manifest("child-a", "root")).unwrap();
        mgr.create_compartment(child_manifest("child-b", "root")).unwrap();
        mgr.create_compartment(workspace_manifest("other")).unwrap();

        let children = mgr.compartments_by_parent("root");
        assert_eq!(children.len(), 2);
        let ids: Vec<&str> = children.iter().map(|m| m.id.as_str()).collect();
        assert!(ids.contains(&"child-a"));
        assert!(ids.contains(&"child-b"));

        let others = mgr.compartments_by_parent("other");
        assert!(others.is_empty());
    }

    #[test]
    fn discover_loads_workspace_kind_from_toml() {
        let (tmp, mgr) = tmp_manager();
        // Write a compartment.toml with Workspace kind.
        let dir = tmp.path().join("tecton");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("compartment.toml"),
            r#"id = "tecton"
name = "Tecton"
kind = "Workspace"
scope = "Private"
description = "workspace"
tags = []
"#,
        )
        .unwrap();

        let count = mgr.discover().unwrap();
        assert_eq!(count, 1);

        let m = mgr.get("tecton").unwrap();
        assert_eq!(m.kind, CompartmentKind::Workspace);
        assert_eq!(m.scope, CompartmentScope::Private);
    }

    #[test]
    fn unload_keeps_compartment_loaded_when_engine_is_quiesced() {
        use crate::engine::{Engine, EngineConfig};
        use crate::graph::Triple;
        use crate::vsa::Dimension;

        let (tmp, manager) = tmp_manager();
        manager
            .create_compartment(workspace_manifest("retained"))
            .unwrap();
        let engine = Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            data_dir: Some(tmp.path().join("engine-data")),
            ..Default::default()
        })
        .unwrap();
        manager.load("retained", &engine).unwrap();
        let subject = engine.resolve_or_create_entity("retained-subject").unwrap();
        let predicate = engine.resolve_or_create_relation("retained-predicate").unwrap();
        let object = engine.resolve_or_create_entity("retained-object").unwrap();
        engine
            .add_triple(
                &Triple::new(subject, predicate, object)
                    .with_compartment("retained".into()),
            )
            .unwrap();

        engine.quiesce_mutations();
        assert!(manager.unload("retained", &engine).is_err());
        let (_, state, _) = manager
            .all_compartments()
            .into_iter()
            .find(|(manifest, _, _)| manifest.id == "retained")
            .unwrap();
        assert_eq!(state, CompartmentState::Loaded);
        assert!(engine.has_triple(subject, predicate, object));
    }

    #[test]
    fn same_pack_reload_preserves_cold_woken_provenance_and_timestamp() {
        use crate::engine::{Engine, EngineConfig};
        use crate::vsa::Dimension;

        let (tmp, manager) = tmp_manager();
        let mut manifest = workspace_manifest("facts");
        manifest.triples_file = Some("triples.json".into());
        manager.create_compartment(manifest).unwrap();
        std::fs::write(
            tmp.path().join("facts").join("triples.json"),
            r#"[{"subject":"Sun","predicate":"emits","object":"Light","confidence":0.8}]"#,
        )
        .unwrap();

        let engine = Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            data_dir: Some(tmp.path().join("engine-data")),
            ..Default::default()
        })
        .unwrap();

        manager.load("facts", &engine).unwrap();
        let sun = engine.resolve_symbol("Sun").unwrap();
        let emits = engine.resolve_symbol("emits").unwrap();
        let light = engine.resolve_symbol("Light").unwrap();
        let moon = engine.resolve_or_create_entity("Moon").unwrap();
        let pack_fact = || {
            engine
                .triples_from(sun)
                .into_iter()
                .find(|triple| {
                    triple.predicate == emits
                        && triple.object == light
                        && triple.compartment_id.as_deref() == Some("facts")
                })
                .expect("pack fact should be live")
        };
        let assert_exact_pack_record = |raw_id: u64| {
            let record = engine
                .get_provenance(crate::provenance::ProvenanceId::new(raw_id).unwrap())
                .unwrap();
            assert_eq!(record.derived_id, light);
            assert_eq!(record.sources, vec![sun]);
            assert!(matches!(
                &record.kind,
                crate::provenance::DerivationKind::CompartmentLoaded {
                    compartment_id,
                    source_file,
                } if compartment_id == "facts" && source_file == "triples.json"
            ));
        };

        let first = pack_fact();
        let original_provenance = first
            .provenance_id
            .expect("persistent pack load should attach provenance");
        assert_exact_pack_record(original_provenance);
        let ledger_len = engine.all_provenance().unwrap().len();

        // Model snapshot restore with a recognizable preserved timestamp. A
        // fresh manager below represents deferred compartment startup after
        // that restore has already made the fact live.
        let restored_timestamp = 1_700_000_123;
        let mut restored = first;
        restored.timestamp = restored_timestamp;
        engine.add_triple(&restored).unwrap();
        assert_eq!(pack_fact().timestamp, restored_timestamp);

        let cold_woken_manager = CompartmentManager::new(tmp.path().to_path_buf());
        assert_eq!(cold_woken_manager.discover().unwrap(), 1);
        cold_woken_manager.load("facts", &engine).unwrap();

        let after_reload = pack_fact();
        assert_eq!(after_reload.provenance_id, Some(original_provenance));
        assert_eq!(after_reload.timestamp, restored_timestamp);
        assert_eq!(engine.all_provenance().unwrap().len(), ledger_len);
        assert_eq!(
            engine
                .triples_from(sun)
                .into_iter()
                .filter(|triple| {
                    triple.predicate == emits
                        && triple.object == light
                        && triple.compartment_id.as_deref() == Some("facts")
                })
                .count(),
            1
        );

        engine.remove_triple(sun, emits, light).unwrap();
        engine
            .add_triple(
                &crate::graph::Triple::new(sun, emits, light)
                    .with_confidence(0.8)
                    .with_compartment("facts".into()),
            )
            .unwrap();
        assert_eq!(pack_fact().provenance_id, None);

        let repair_manager = CompartmentManager::new(tmp.path().to_path_buf());
        assert_eq!(repair_manager.discover().unwrap(), 1);
        repair_manager.load("facts", &engine).unwrap();
        let repaired = pack_fact();
        let repaired_id = repaired
            .provenance_id
            .expect("an untracked collision still needs pack provenance");
        assert_ne!(repaired_id, original_provenance);
        assert_exact_pack_record(repaired_id);
        assert_eq!(engine.all_provenance().unwrap().len(), ledger_len + 1);

        // A provenance ID alone is not proof that the live edge came from this
        // exact pack. Exercise each ledger field independently so none can be
        // omitted from the preservation predicate without this regression
        // catching it.
        let wrong_records = [
            (
                "wrong subject source",
                light,
                vec![moon],
                crate::provenance::DerivationKind::CompartmentLoaded {
                    compartment_id: "facts".into(),
                    source_file: "triples.json".into(),
                },
            ),
            (
                "additional subject source",
                light,
                vec![sun, moon],
                crate::provenance::DerivationKind::CompartmentLoaded {
                    compartment_id: "facts".into(),
                    source_file: "triples.json".into(),
                },
            ),
            (
                "wrong derived object",
                moon,
                vec![sun],
                crate::provenance::DerivationKind::CompartmentLoaded {
                    compartment_id: "facts".into(),
                    source_file: "triples.json".into(),
                },
            ),
            (
                "wrong pack source file",
                light,
                vec![sun],
                crate::provenance::DerivationKind::CompartmentLoaded {
                    compartment_id: "facts".into(),
                    source_file: "other.json".into(),
                },
            ),
            (
                "wrong provenance compartment",
                light,
                vec![sun],
                crate::provenance::DerivationKind::CompartmentLoaded {
                    compartment_id: "other".into(),
                    source_file: "triples.json".into(),
                },
            ),
        ];

        for (case, derived_id, sources, kind) in wrong_records {
            let mut collision = [crate::graph::Triple::new(sun, emits, light)
                .with_confidence(0.8)
                .with_compartment("facts".into())];
            let mut wrong_record = crate::provenance::ProvenanceRecord::new(derived_id, kind)
                .with_sources(sources)
                .with_confidence(0.8);
            let wrong_id = engine
                .add_triples_with_provenance(&mut collision, &mut wrong_record)
                .unwrap()
                .expect("persistent collision should receive provenance")
                .get();
            assert_eq!(pack_fact().provenance_id, Some(wrong_id), "{case}");
            let ledger_before_repair = engine.all_provenance().unwrap().len();

            let repair_manager = CompartmentManager::new(tmp.path().to_path_buf());
            assert_eq!(repair_manager.discover().unwrap(), 1);
            repair_manager.load("facts", &engine).unwrap();

            let repaired_id = pack_fact()
                .provenance_id
                .expect("invalid pack provenance must be repaired");
            assert_ne!(repaired_id, wrong_id, "{case}");
            assert_exact_pack_record(repaired_id);
            assert_eq!(
                engine.all_provenance().unwrap().len(),
                ledger_before_repair + 1,
                "{case} must mint exactly one replacement record"
            );
        }
    }

    fn make_manager() -> (CompartmentManager, tempfile::TempDir) {
        let dir = tempfile::TempDir::new().unwrap();
        let manager = CompartmentManager::new(dir.path().to_path_buf());
        (manager, dir)
    }

    #[test]
    fn read_ennead_role_returns_none_when_file_absent() {
        let (manager, _dir) = make_manager();
        assert!(manager.read_ennead_role_file().is_none());
    }

    #[test]
    fn write_then_read_ennead_role_roundtrip() {
        let (manager, _dir) = make_manager();
        let prev = manager.write_ennead_role_file("architect").unwrap();
        assert!(prev.is_none(), "first write should return no prior role");
        assert_eq!(
            manager.read_ennead_role_file().as_deref(),
            Some("architect")
        );
    }

    #[test]
    fn write_ennead_role_creates_psyche_compartment_dir() {
        let (manager, dir) = make_manager();
        manager.write_ennead_role_file("investigator").unwrap();
        assert!(dir.path().join("psyche").join("ennead-role").exists());
        assert!(dir.path().join("psyche").join("compartment.toml").exists());
    }

    #[test]
    fn write_ennead_role_updates_in_memory_psyche() {
        let (manager, _dir) = make_manager();
        manager.write_ennead_role_file("executor").unwrap();
        let psyche = manager
            .psyche()
            .expect("psyche should be populated after write");
        assert_eq!(psyche.ennead_role.as_deref(), Some("executor"));
    }

    #[test]
    fn write_ennead_role_returns_existing_role_on_overwrite() {
        let (manager, _dir) = make_manager();
        manager.write_ennead_role_file("architect").unwrap();
        let prev = manager.write_ennead_role_file("investigator").unwrap();
        assert_eq!(
            prev.as_deref(),
            Some("architect"),
            "should return the prior role"
        );
    }
}
