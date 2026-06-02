//! Kubernetes cluster topology ingestion.
//!
//! Queries the cluster via `kubectl` and asserts structured triples into the
//! knowledge graph. Covers nodes, storage classes, and pod placements for the
//! known set of services (FluxCD, CloudNativePG, Envoy Gateway, Solstice,
//! Forgejo, Anima).

use std::process::Command;

use miette::Diagnostic;
use serde_json::Value;
use thiserror::Error;

use crate::engine::Engine;
use crate::error::AkhError;
use crate::provenance::{DerivationKind, ProvenanceRecord};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error, Diagnostic)]
pub enum ClusterError {
    #[error("kubectl invocation failed: {msg}")]
    #[diagnostic(
        code(akh::cluster::kubectl_failed),
        help(
            "Ensure `kubectl` is installed and KUBECONFIG / in-cluster service-account \
             credentials are available."
        )
    )]
    KubectlFailed { msg: String },

    #[error("kubectl returned non-zero exit status {code}: {stderr}")]
    #[diagnostic(
        code(akh::cluster::kubectl_exit),
        help(
            "Check cluster connectivity (`kubectl cluster-info`) and that the \
             ServiceAccount has the required RBAC permissions."
        )
    )]
    KubectlNonZero { code: i32, stderr: String },

    #[error("failed to parse kubectl JSON output: {msg}")]
    #[diagnostic(
        code(akh::cluster::parse_error),
        help("Run `kubectl get nodes -o json` manually to verify the response format.")
    )]
    ParseError { msg: String },

    #[error("engine error while asserting triples: {0}")]
    #[diagnostic(transparent)]
    Engine(#[from] Box<AkhError>),
}

impl From<AkhError> for ClusterError {
    fn from(e: AkhError) -> Self {
        Self::Engine(Box::new(e))
    }
}

pub type ClusterResult<T> = Result<T, ClusterError>;

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

/// Summary returned by [`ingest_topology`].
#[derive(Debug, Default)]
pub struct IngestReport {
    pub nodes_processed: usize,
    pub storage_classes_processed: usize,
    pub pods_processed: usize,
    pub triples_asserted: usize,
    pub errors: Vec<String>,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Query the Kubernetes cluster and assert topology triples into the KG.
///
/// Returns an [`IngestReport`] summarising what was ingested. On partial
/// failure the errors are collected in `report.errors` rather than aborting.
pub fn ingest_topology(engine: &Engine, dry_run: bool) -> ClusterResult<IngestReport> {
    let mut report = IngestReport::default();
    let mut triples: Vec<(String, String, String)> = Vec::new();

    // -- Nodes ---------------------------------------------------------------
    let nodes_json = kubectl_json(&["get", "nodes", "-o", "json"])?;
    for item in json_items(&nodes_json) {
        match extract_node_triples(item) {
            Ok(mut t) => {
                report.nodes_processed += 1;
                triples.append(&mut t);
            }
            Err(e) => report.errors.push(format!("node: {e}")),
        }
    }

    // -- Storage classes -----------------------------------------------------
    let sc_json = kubectl_json(&["get", "storageclasses", "-o", "json"])?;
    for item in json_items(&sc_json) {
        match extract_storageclass_triples(item) {
            Ok(mut t) => {
                report.storage_classes_processed += 1;
                triples.append(&mut t);
            }
            Err(e) => report.errors.push(format!("storageclass: {e}")),
        }
    }

    // -- Pods (all namespaces) -----------------------------------------------
    let pods_json =
        kubectl_json(&["get", "pods", "--all-namespaces", "-o", "json"])?;
    for item in json_items(&pods_json) {
        match extract_pod_triples(item) {
            Ok(mut t) => {
                report.pods_processed += 1;
                triples.append(&mut t);
            }
            Err(e) => report.errors.push(format!("pod: {e}")),
        }
    }

    if dry_run {
        for (s, p, o) in &triples {
            println!("  {s} --[{p}]--> {o}");
        }
        report.triples_asserted = triples.len();
        return Ok(report);
    }

    // -- Assert into KG ------------------------------------------------------
    let (asserted, errs) = engine.add_triples_batch(&triples, 1.0);
    for e in errs {
        report.errors.push(e);
    }
    report.triples_asserted = asserted;

    // -- Provenance ----------------------------------------------------------
    // Use the "k8s-cluster" entity as the anchor for the provenance record.
    if let Ok(cluster_entity) = engine.resolve_or_create_entity("k8s-cluster") {
        let mut prov = ProvenanceRecord::new(
            cluster_entity,
            DerivationKind::ClusterTopologyIngested {
                nodes: report.nodes_processed,
                storage_classes: report.storage_classes_processed,
                pods: report.pods_processed,
                triples_asserted: report.triples_asserted,
            },
        );
        if let Err(e) = engine.store_provenance(&mut prov) {
            report.errors.push(format!("provenance: {e}"));
        }
    }

    if let Err(e) = engine.persist() {
        report.errors.push(format!("persist: {e}"));
    }

    Ok(report)
}

/// Print a summary of cluster-topology triples already in the KG.
pub fn show_topology(engine: &Engine) {
    let all_triples = engine.export_triples();
    let cluster_triples: Vec<_> = all_triples
        .iter()
        .filter(|t| t.predicate_label.starts_with("cluster:"))
        .collect();

    if cluster_triples.is_empty() {
        println!("No cluster-topology triples found. Run `akh cluster ingest` first.");
        return;
    }

    println!("{:<40} {:<35} Object", "Subject", "Predicate");
    println!("{}", "-".repeat(110));
    for t in &cluster_triples {
        println!(
            "{:<40} {:<35} {}",
            t.subject_label, t.predicate_label, t.object_label
        );
    }
    println!("\nTotal: {} triple(s)", cluster_triples.len());
}

// ---------------------------------------------------------------------------
// kubectl helpers
// ---------------------------------------------------------------------------

fn kubectl_json(args: &[&str]) -> ClusterResult<Value> {
    let output = Command::new("kubectl")
        .args(args)
        .output()
        .map_err(|e| ClusterError::KubectlFailed {
            msg: format!("{e} (is kubectl in PATH?)"),
        })?;

    if !output.status.success() {
        let code = output.status.code().unwrap_or(-1);
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        return Err(ClusterError::KubectlNonZero { code, stderr });
    }

    serde_json::from_slice(&output.stdout).map_err(|e| ClusterError::ParseError {
        msg: e.to_string(),
    })
}

/// Extract the `.items[]` array from a kubectl List response.
fn json_items(list: &Value) -> Vec<&Value> {
    list.get("items")
        .and_then(Value::as_array)
        .map(|v| v.iter().collect())
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Node triple extraction
// ---------------------------------------------------------------------------

fn extract_node_triples(node: &Value) -> ClusterResult<Vec<(String, String, String)>> {
    let name = node_name(node)?;
    let mut triples = Vec::new();

    // Role
    let role = infer_node_role(node, &name);
    triples.push((name.clone(), "cluster:role".into(), role));

    // Internal IP
    if let Some(ip) = node_internal_ip(node) {
        triples.push((name.clone(), "cluster:ip".into(), ip));
    }

    // Ready status
    let status = node_ready_status(node);
    triples.push((name.clone(), "cluster:status".into(), status));

    // OS image
    if let Some(os) = node
        .pointer("/status/nodeInfo/osImage")
        .and_then(Value::as_str)
    {
        triples.push((name.clone(), "cluster:os-image".into(), os.to_string()));
    }

    // Architecture
    if let Some(arch) = node
        .pointer("/status/nodeInfo/architecture")
        .and_then(Value::as_str)
    {
        triples.push((name.clone(), "cluster:arch".into(), arch.to_string()));
    }

    // Kubelet version
    if let Some(ver) = node
        .pointer("/status/nodeInfo/kubeletVersion")
        .and_then(Value::as_str)
    {
        triples.push((name.clone(), "cluster:kubelet-version".into(), ver.to_string()));
    }

    Ok(triples)
}

fn node_name(node: &Value) -> ClusterResult<String> {
    node.pointer("/metadata/name")
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| ClusterError::ParseError {
            msg: "node missing metadata.name".into(),
        })
}

fn infer_node_role(node: &Value, name: &str) -> String {
    // K8s standard label wins
    if let Some(labels) = node.pointer("/metadata/labels").and_then(Value::as_object) {
        let is_control_plane =
            labels.contains_key("node-role.kubernetes.io/control-plane")
            || labels.contains_key("node-role.kubernetes.io/master");
        if is_control_plane {
            return "control-plane".into();
        }
        // Taints-based or custom role labels
        for key in labels.keys() {
            if key.starts_with("node-role.kubernetes.io/") {
                let role = key.trim_start_matches("node-role.kubernetes.io/");
                if !role.is_empty() {
                    return role.to_string();
                }
            }
        }
    }

    // Infer from hostname convention
    let n = name.to_lowercase();
    if n.contains("head") || n.contains("master") || n.contains("control") {
        "control-plane".into()
    } else if n.contains("wktiny") {
        "tiny-worker".into()
    } else if n.contains("wk") {
        "worker".into()
    } else if n.contains("stor") {
        "storage".into()
    } else {
        "worker".into()
    }
}

fn node_internal_ip(node: &Value) -> Option<String> {
    let addresses = node
        .pointer("/status/addresses")
        .and_then(Value::as_array)?;
    for addr in addresses {
        if addr.get("type").and_then(Value::as_str) == Some("InternalIP") {
            return addr.get("address").and_then(Value::as_str).map(str::to_string);
        }
    }
    None
}

fn node_ready_status(node: &Value) -> String {
    let conditions = match node
        .pointer("/status/conditions")
        .and_then(Value::as_array)
    {
        Some(c) => c,
        None => return "Unknown".into(),
    };
    for cond in conditions {
        if cond.get("type").and_then(Value::as_str) == Some("Ready") {
            return match cond.get("status").and_then(Value::as_str) {
                Some("True") => "Ready".into(),
                Some("False") => "NotReady".into(),
                _ => "Unknown".into(),
            };
        }
    }
    "Unknown".into()
}

// ---------------------------------------------------------------------------
// StorageClass triple extraction
// ---------------------------------------------------------------------------

fn extract_storageclass_triples(
    sc: &Value,
) -> ClusterResult<Vec<(String, String, String)>> {
    let name = sc
        .pointer("/metadata/name")
        .and_then(Value::as_str)
        .ok_or_else(|| ClusterError::ParseError {
            msg: "storageclass missing metadata.name".into(),
        })?
        .to_string();

    let mut triples = Vec::new();

    // Mark as a storage class entity
    triples.push((
        name.clone(),
        "cluster:is-storage-class".into(),
        "true".into(),
    ));

    // Provisioner
    if let Some(prov) = sc.get("provisioner").and_then(Value::as_str) {
        triples.push((name.clone(), "cluster:provisioner".into(), prov.to_string()));
    }

    // Reclaim policy
    if let Some(policy) = sc.get("reclaimPolicy").and_then(Value::as_str) {
        triples.push((name.clone(), "cluster:reclaim-policy".into(), policy.to_string()));
    }

    // Default annotation
    let annotations = sc.pointer("/metadata/annotations");
    let is_default = annotations
        .and_then(|a| a.get("storageclass.kubernetes.io/is-default-class"))
        .and_then(Value::as_str)
        == Some("true");
    if is_default {
        triples.push((
            name.clone(),
            "cluster:is-default-storage-class".into(),
            "true".into(),
        ));
    }

    Ok(triples)
}

// ---------------------------------------------------------------------------
// Pod triple extraction
// ---------------------------------------------------------------------------

/// Namespaces that contain services we care about.
const TRACKED_NAMESPACES: &[&str] = &[
    "flux-system",
    "cnpg-system",
    "cloudnative-pg",
    "postgres-operator",
    "envoy-gateway-system",
    "solstice",
    "forgejo",
    "anima",
    "kube-system",
];

fn extract_pod_triples(pod: &Value) -> ClusterResult<Vec<(String, String, String)>> {
    let ns = pod
        .pointer("/metadata/namespace")
        .and_then(Value::as_str)
        .unwrap_or("default");

    if !TRACKED_NAMESPACES.contains(&ns) {
        return Ok(vec![]);
    }

    let name = pod
        .pointer("/metadata/name")
        .and_then(Value::as_str)
        .ok_or_else(|| ClusterError::ParseError {
            msg: "pod missing metadata.name".into(),
        })?;

    // Use a stable label based on owner (deployment/statefulset) if available
    let pod_label = pod_stable_label(pod, ns, name);
    let mut triples = Vec::new();

    // Namespace
    triples.push((pod_label.clone(), "cluster:namespace".into(), ns.to_string()));

    // Node placement
    if let Some(node) = pod
        .pointer("/spec/nodeName")
        .and_then(Value::as_str)
        .filter(|n| !n.is_empty())
    {
        triples.push((pod_label.clone(), "cluster:runs-on".into(), node.to_string()));
    }

    // First container image
    if let Some(image) = pod
        .pointer("/spec/containers/0/image")
        .and_then(Value::as_str)
    {
        triples.push((pod_label.clone(), "cluster:image".into(), image.to_string()));
    }

    // Phase
    if let Some(phase) = pod.pointer("/status/phase").and_then(Value::as_str) {
        triples.push((pod_label.clone(), "cluster:phase".into(), phase.to_string()));
    }

    Ok(triples)
}

/// Build a stable entity label for a pod by preferring owner references.
///
/// Falls back to `<namespace>/<pod-name>` when no owner is found.
fn pod_stable_label(pod: &Value, ns: &str, pod_name: &str) -> String {
    if let Some(owners) = pod
        .pointer("/metadata/ownerReferences")
        .and_then(Value::as_array)
    {
        // Prefer Deployment/StatefulSet → use their name
        for owner in owners {
            let kind = owner.get("kind").and_then(Value::as_str).unwrap_or("");
            let oname = owner.get("name").and_then(Value::as_str).unwrap_or("");
            match kind {
                "ReplicaSet" => {
                    // Strip the hash suffix from ReplicaSet name to get Deployment name
                    let deployment_name = strip_replicaset_suffix(oname);
                    return format!("k8s-deploy:{ns}/{deployment_name}");
                }
                "StatefulSet" => return format!("k8s-statefulset:{ns}/{oname}"),
                "DaemonSet" => return format!("k8s-daemonset:{ns}/{oname}"),
                _ => {}
            }
        }
    }
    format!("k8s-pod:{ns}/{pod_name}")
}

/// Strip the random hash suffix from a ReplicaSet name to recover the Deployment name.
///
/// K8s ReplicaSet names are `<deployment>-<10-char-hash>`. Splitting on the
/// last '-' recovers the Deployment name regardless of how many dashes appear
/// in the deployment name itself.
///
/// e.g. `flux-controller-5f9bbd7b5a` → `flux-controller`
fn strip_replicaset_suffix(rs_name: &str) -> &str {
    rs_name.rfind('-').map(|pos| &rs_name[..pos]).unwrap_or(rs_name)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn strip_suffix_basic() {
        assert_eq!(strip_replicaset_suffix("flux-controller-5f9bbd7b5a"), "flux-controller");
    }

    #[test]
    fn strip_suffix_no_hash() {
        assert_eq!(strip_replicaset_suffix("simple"), "simple");
    }

    #[test]
    fn strip_suffix_multi_dash() {
        assert_eq!(strip_replicaset_suffix("my-long-deploy-name-abc123"), "my-long-deploy-name");
    }

    #[test]
    fn node_role_control_plane_label() {
        let node = json!({
            "metadata": {
                "name": "headnode",
                "labels": { "node-role.kubernetes.io/control-plane": "" }
            }
        });
        assert_eq!(infer_node_role(&node, "headnode"), "control-plane");
    }

    #[test]
    fn node_role_inferred_from_name() {
        let node = json!({ "metadata": { "name": "wk01", "labels": {} } });
        assert_eq!(infer_node_role(&node, "wk01"), "worker");

        let stor = json!({ "metadata": { "name": "stor01", "labels": {} } });
        assert_eq!(infer_node_role(&stor, "stor01"), "storage");

        let tiny = json!({ "metadata": { "name": "wktiny01", "labels": {} } });
        assert_eq!(infer_node_role(&tiny, "wktiny01"), "tiny-worker");
    }

    #[test]
    fn extract_node_triples_basic() {
        let node = json!({
            "metadata": {
                "name": "wk01",
                "labels": {}
            },
            "status": {
                "addresses": [
                    { "type": "InternalIP", "address": "10.0.0.1" }
                ],
                "conditions": [
                    { "type": "Ready", "status": "True" }
                ],
                "nodeInfo": {
                    "osImage": "Ubuntu 24.04",
                    "architecture": "amd64",
                    "kubeletVersion": "v1.32.0"
                }
            }
        });
        let triples = extract_node_triples(&node).unwrap();
        assert!(triples.iter().any(|(s, p, o)| s == "wk01" && p == "cluster:ip" && o == "10.0.0.1"));
        assert!(triples.iter().any(|(s, p, o)| s == "wk01" && p == "cluster:status" && o == "Ready"));
        assert!(triples.iter().any(|(s, p, o)| s == "wk01" && p == "cluster:arch" && o == "amd64"));
        assert!(triples.iter().any(|(s, p, o)| s == "wk01" && p == "cluster:role" && o == "worker"));
        assert!(triples.iter().any(|(s, p, o)| s == "wk01" && p == "cluster:kubelet-version" && o == "v1.32.0"));
    }

    #[test]
    fn extract_storageclass_triples_default() {
        let sc = json!({
            "metadata": {
                "name": "longhorn",
                "annotations": {
                    "storageclass.kubernetes.io/is-default-class": "true"
                }
            },
            "provisioner": "driver.longhorn.io",
            "reclaimPolicy": "Delete"
        });
        let triples = extract_storageclass_triples(&sc).unwrap();
        assert!(triples.iter().any(|(s, p, o)| s == "longhorn" && p == "cluster:provisioner" && o == "driver.longhorn.io"));
        assert!(triples.iter().any(|(s, p, o)| s == "longhorn" && p == "cluster:is-default-storage-class" && o == "true"));
        assert!(triples.iter().any(|(s, p, o)| s == "longhorn" && p == "cluster:reclaim-policy" && o == "Delete"));
    }

    #[test]
    fn extract_pod_triples_skips_untracked_namespace() {
        let pod = json!({
            "metadata": {
                "name": "mypod",
                "namespace": "ignored-ns"
            }
        });
        let triples = extract_pod_triples(&pod).unwrap();
        assert!(triples.is_empty());
    }

    #[test]
    fn extract_pod_triples_deployment_owner() {
        let pod = json!({
            "metadata": {
                "name": "anima-abc-def",
                "namespace": "anima",
                "ownerReferences": [
                    { "kind": "ReplicaSet", "name": "anima-server-7f9c8b6d4" }
                ]
            },
            "spec": {
                "nodeName": "wk01",
                "containers": [{ "image": "ghcr.io/anima/server:latest" }]
            },
            "status": { "phase": "Running" }
        });
        let triples = extract_pod_triples(&pod).unwrap();
        let label = "k8s-deploy:anima/anima-server";
        assert!(triples.iter().any(|(s, p, o)| s == label && p == "cluster:runs-on" && o == "wk01"));
        assert!(triples.iter().any(|(s, p, o)| s == label && p == "cluster:phase" && o == "Running"));
        assert!(triples.iter().any(|(s, p, o)| s == label && p == "cluster:namespace" && o == "anima"));
    }

    #[test]
    fn extract_pod_triples_statefulset_owner() {
        let pod = json!({
            "metadata": {
                "name": "pg-0",
                "namespace": "cnpg-system",
                "ownerReferences": [
                    { "kind": "StatefulSet", "name": "pg-cluster" }
                ]
            },
            "spec": {
                "nodeName": "stor01",
                "containers": [{ "image": "ghcr.io/cloudnative-pg/postgresql:16" }]
            },
            "status": { "phase": "Running" }
        });
        let triples = extract_pod_triples(&pod).unwrap();
        let label = "k8s-statefulset:cnpg-system/pg-cluster";
        assert!(triples.iter().any(|(s, p, o)| s == label && p == "cluster:runs-on" && o == "stor01"));
    }
}
