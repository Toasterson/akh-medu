//! Shared knowledge partitions: cross-workspace named graphs.
//!
//! Partitions allow common knowledge (ontologies, common-sense facts) to be
//! shared across multiple workspaces. Each partition maps to a SPARQL named
//! graph and can be either local (owned by a workspace) or shared (stored
//! independently under `$XDG_DATA_HOME/akh-medu/partitions/`).

use std::collections::HashMap;
use std::path::PathBuf;

use miette::Diagnostic;
use thiserror::Error;

use crate::engine::Engine;
use crate::graph::Triple;

/// IRI namespace for partition named graphs.
const PARTITION_NS: &str = "https://akh-medu.dev/partition/";

/// Errors from partition operations.
#[derive(Debug, Error, Diagnostic)]
pub enum PartitionError {
    #[error("partition \"{name}\" not found")]
    #[diagnostic(
        code(akh::partition::not_found),
        help("Available partitions can be listed with the partition manager.")
    )]
    NotFound { name: String },

    #[error("partition \"{name}\" already exists")]
    #[diagnostic(
        code(akh::partition::already_exists),
        help("Use a different name or remove the existing partition.")
    )]
    AlreadyExists { name: String },

    #[error("partition storage error: {message}")]
    #[diagnostic(
        code(akh::partition::storage),
        help("Check file permissions and disk space.")
    )]
    Storage { message: String },

    #[error("partition query error: {message}")]
    #[diagnostic(
        code(akh::partition::query),
        help("Check the SPARQL query syntax.")
    )]
    Query { message: String },
}

pub type PartitionResult<T> = std::result::Result<T, PartitionError>;

/// Where a partition's data lives.
#[derive(Debug, Clone)]
pub enum PartitionSource {
    /// Partition owned by a specific workspace, stored in its KG.
    Local { workspace: String },
    /// Shared partition stored independently.
    Shared { path: PathBuf },
}

/// A named knowledge partition.
#[derive(Debug, Clone)]
pub struct Partition {
    /// Partition name (e.g., "ontology", "common-sense").
    pub name: String,
    /// SPARQL named graph IRI.
    pub graph_name: String,
    /// Where the data is stored.
    pub source: PartitionSource,
}

impl Partition {
    /// Create a new shared partition.
    pub fn shared(name: &str, base_dir: &std::path::Path) -> Self {
        Self {
            name: name.to_string(),
            graph_name: format!("{PARTITION_NS}{name}"),
            source: PartitionSource::Shared {
                path: base_dir.join(name),
            },
        }
    }

    /// Create a new local partition (owned by a workspace).
    pub fn local(name: &str, workspace: &str) -> Self {
        Self {
            name: name.to_string(),
            graph_name: format!("{PARTITION_NS}{name}"),
            source: PartitionSource::Local {
                workspace: workspace.to_string(),
            },
        }
    }
}

/// Manages shared partitions across workspaces.
pub struct PartitionManager {
    /// Registered partitions by name.
    partitions: HashMap<String, Partition>,
    /// Base directory for shared partitions.
    partitions_dir: PathBuf,
}

impl PartitionManager {
    /// Create a new partition manager.
    pub fn new(partitions_dir: PathBuf) -> Self {
        Self {
            partitions: HashMap::new(),
            partitions_dir,
        }
    }

    /// Discover existing shared partitions on disk.
    pub fn discover(&mut self) -> PartitionResult<usize> {
        if !self.partitions_dir.exists() {
            return Ok(0);
        }

        let entries = std::fs::read_dir(&self.partitions_dir).map_err(|e| {
            PartitionError::Storage {
                message: format!(
                    "failed to read partitions dir {}: {e}",
                    self.partitions_dir.display()
                ),
            }
        })?;

        let mut count = 0;
        for entry in entries.flatten() {
            if entry.path().is_dir() {
                if let Some(name) = entry.file_name().to_str() {
                    let partition = Partition::shared(name, &self.partitions_dir);
                    self.partitions.insert(name.to_string(), partition);
                    count += 1;
                }
            }
        }

        Ok(count)
    }

    /// Register a partition (does not create storage).
    pub fn register(&mut self, partition: Partition) -> PartitionResult<()> {
        if self.partitions.contains_key(&partition.name) {
            return Err(PartitionError::AlreadyExists {
                name: partition.name,
            });
        }
        self.partitions.insert(partition.name.clone(), partition);
        Ok(())
    }

    /// Create a new shared partition on disk.
    pub fn create_shared(&mut self, name: &str) -> PartitionResult<&Partition> {
        if self.partitions.contains_key(name) {
            return Err(PartitionError::AlreadyExists {
                name: name.to_string(),
            });
        }

        let dir = self.partitions_dir.join(name);
        std::fs::create_dir_all(&dir).map_err(|e| PartitionError::Storage {
            message: format!("failed to create partition dir {}: {e}", dir.display()),
        })?;

        let partition = Partition::shared(name, &self.partitions_dir);
        self.partitions.insert(name.to_string(), partition);
        Ok(&self.partitions[name])
    }

    /// Get a partition by name.
    pub fn get(&self, name: &str) -> PartitionResult<&Partition> {
        self.partitions
            .get(name)
            .ok_or_else(|| PartitionError::NotFound {
                name: name.to_string(),
            })
    }

    /// List all registered partition names.
    pub fn list(&self) -> Vec<&str> {
        self.partitions.keys().map(|s| s.as_str()).collect()
    }

    /// Remove a shared partition (deletes data on disk).
    pub fn remove(&mut self, name: &str) -> PartitionResult<()> {
        let partition = self.partitions.remove(name).ok_or_else(|| {
            PartitionError::NotFound {
                name: name.to_string(),
            }
        })?;

        if let PartitionSource::Shared { path } = &partition.source {
            if path.exists() {
                std::fs::remove_dir_all(path).map_err(|e| PartitionError::Storage {
                    message: format!("failed to remove partition dir: {e}"),
                })?;
            }
        }

        Ok(())
    }
}

/// Insert a triple into a specific partition's named graph.
///
/// Routes the triple to the SPARQL store under the partition's named graph IRI.
pub fn insert_into_partition(
    engine: &Engine,
    triple: &Triple,
    partition_name: &str,
) -> PartitionResult<()> {
    let sparql = engine.sparql().ok_or_else(|| PartitionError::Storage {
        message: "SPARQL store not available (engine has no data_dir)".to_string(),
    })?;
    sparql
        .insert_triple_in_graph(triple, Some(partition_name))
        .map_err(|e| PartitionError::Storage {
            message: format!("failed to insert into partition \"{partition_name}\": {e}"),
        })
}

/// Query a specific partition using SPARQL.
///
/// Wraps the query pattern in a GRAPH clause targeting the partition's named graph.
/// Returns rows of `(variable_name, value)` bindings.
pub fn query_partition(
    engine: &Engine,
    partition_name: &str,
    sparql_pattern: &str,
) -> PartitionResult<Vec<Vec<(String, String)>>> {
    let sparql = engine.sparql().ok_or_else(|| PartitionError::Query {
        message: "SPARQL store not available (engine has no data_dir)".to_string(),
    })?;

    let graph_iri = format!("{PARTITION_NS}{partition_name}");
    // Wrap the user query in a GRAPH clause if not already referencing one.
    let query = if sparql_pattern.contains("GRAPH") {
        sparql_pattern.to_string()
    } else {
        format!("SELECT * WHERE {{ GRAPH <{graph_iri}> {{ {sparql_pattern} }} }}")
    };

    sparql.query_select(&query).map_err(|e| PartitionError::Query {
        message: format!("partition query failed: {e}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partition_naming() {
        let p = Partition::shared("ontology", std::path::Path::new("/tmp"));
        assert_eq!(p.graph_name, "https://akh-medu.dev/partition/ontology");
        assert_eq!(p.name, "ontology");
    }

    #[test]
    fn partition_manager_register_and_list() {
        let mut pm = PartitionManager::new(PathBuf::from("/tmp/akh-test-partitions"));
        let p = Partition::local("test-part", "default");
        pm.register(p).unwrap();
        assert_eq!(pm.list().len(), 1);
        assert!(pm.get("test-part").is_ok());
        assert!(pm.get("nonexistent").is_err());
    }

    #[test]
    fn partition_manager_create_shared() {
        let dir = tempfile::tempdir().unwrap();
        let mut pm = PartitionManager::new(dir.path().to_path_buf());
        let p = pm.create_shared("my-partition").unwrap();
        assert_eq!(p.name, "my-partition");
        assert!(dir.path().join("my-partition").exists());

        // Duplicate creation fails.
        assert!(pm.create_shared("my-partition").is_err());

        // Remove.
        pm.remove("my-partition").unwrap();
        assert!(!dir.path().join("my-partition").exists());
    }

    #[test]
    fn partition_manager_discover() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("ontology")).unwrap();
        std::fs::create_dir(dir.path().join("common-sense")).unwrap();

        let mut pm = PartitionManager::new(dir.path().to_path_buf());
        let count = pm.discover().unwrap();
        assert_eq!(count, 2);
        assert!(pm.get("ontology").is_ok());
        assert!(pm.get("common-sense").is_ok());
    }
}
