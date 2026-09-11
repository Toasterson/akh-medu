//! External scheduler client. The occurrence identity/time comes from the scheduler.
use akh_session_services::{config::Endpoint, http::segment, types::*};
use clap::Parser;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    io::Read,
    path::{Path, PathBuf},
};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    endpoint_config: PathBuf,
    /// Send to this akhsessd automation; omit to send through akheventd routing.
    #[arg(long)]
    automation: Option<String>,
    /// An immutable, complete EventEnvelope; reuse exactly this file on retries.
    #[arg(
        long,
        required_unless_present = "template",
        conflicts_with = "template"
    )]
    event: Option<PathBuf>,
    /// Static source/type/subject/correlation_id/data JSON, without occurrence identity/time.
    #[arg(long)]
    template: Option<PathBuf>,
    #[arg(long, requires = "template", conflicts_with = "kubernetes_job")]
    occurrence: Option<String>,
    #[arg(long, requires = "occurrence")]
    occurred_at_ms: Option<u64>,
    /// Resolve AKH_JOB_NAME/AKH_JOB_NAMESPACE with the Job API to survive Pod retries.
    #[arg(long,requires_all=["template","kubernetes_api_config"])]
    kubernetes_job: bool,
    #[arg(long)]
    kubernetes_api_config: Option<PathBuf>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Template {
    source: String,
    #[serde(rename = "type")]
    event_type: String,
    subject: Option<String>,
    correlation_id: Option<String>,
    data: Value,
}
fn read(path: &Path) -> Result<Vec<u8>> {
    let mut bytes = vec![];
    std::fs::File::open(path)
        .and_then(|f| f.take(1024 * 1024 + 1).read_to_end(&mut bytes))
        .map_err(|_| Fault::invalid("trigger input file cannot be read"))?;
    if bytes.len() > 1024 * 1024 {
        return Err(Fault::invalid("trigger input exceeds 1 MiB"));
    }
    Ok(bytes)
}
fn endpoint(path: &Path) -> Result<Endpoint> {
    let bytes = read(path)?;
    let endpoint: Endpoint = if path.extension().is_some_and(|e| e == "json") {
        serde_json::from_slice(&bytes).map_err(|_| Fault::invalid("invalid endpoint JSON"))?
    } else {
        toml::from_str(
            std::str::from_utf8(&bytes)
                .map_err(|_| Fault::invalid("endpoint config is not UTF-8"))?,
        )
        .map_err(|_| Fault::invalid("invalid endpoint TOML"))?
    };
    endpoint.validate()?;
    Ok(endpoint)
}
async fn response(mut response: reqwest::Response) -> Result<Value> {
    let status = response.status();
    let mut bytes = vec![];
    while let Some(chunk) = response.chunk().await.map_err(|_| {
        Fault::new(
            "transport",
            "trigger response interrupted; retry only the same occurrence",
        )
    })? {
        if bytes.len() + chunk.len() > 1024 * 1024 {
            return Err(Fault::invalid("trigger response exceeds 1 MiB"));
        }
        bytes.extend_from_slice(&chunk);
    }
    if !status.is_success() {
        return Err(Fault::new(
            "trigger_rejected",
            format!(
                "trigger request returned HTTP {}; preserve its occurrence on retry",
                status.as_u16()
            ),
        ));
    }
    serde_json::from_slice(&bytes).map_err(|_| Fault::invalid("invalid trigger response JSON"))
}
fn job_occurrence(job: &Value, name: &str, namespace: &str) -> Result<(String, u64)> {
    if job["metadata"]["name"] != name || job["metadata"]["namespace"] != namespace {
        return Err(Fault::conflict("Kubernetes returned a different Job"));
    }
    let uid = job["metadata"]["uid"]
        .as_str()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| Fault::invalid("Job has no UID"))?;
    let scheduled =
        job["metadata"]["annotations"]["batch.kubernetes.io/cronjob-scheduled-timestamp"].as_str();
    let at = scheduled
        .or_else(|| job["metadata"]["creationTimestamp"].as_str())
        .ok_or_else(|| Fault::invalid("Job has no scheduler/creation timestamp"))?;
    let at: u64 = chrono::DateTime::parse_from_rfc3339(at)
        .map_err(|_| Fault::invalid("invalid Job timestamp"))?
        .timestamp_millis()
        .try_into()
        .map_err(|_| Fault::invalid("Job timestamp precedes epoch"))?;
    let cron = job["metadata"]["ownerReferences"]
        .as_array()
        .and_then(|owners| {
            owners
                .iter()
                .find(|o| o["kind"] == "CronJob" && o["controller"] == true)
        })
        .and_then(|owner| owner["uid"].as_str());
    let id = match (cron, scheduled) {
        (Some(owner), Some(_)) => format!("cron:{namespace}/{owner}/{at}"),
        _ => format!("k8s:{namespace}/{uid}"),
    };
    Ok((id, at))
}
async fn run(args: Args) -> Result<()> {
    let target = endpoint(&args.endpoint_config)?;
    let event: EventEnvelope = if let Some(path) = args.event {
        serde_json::from_slice(&read(&path)?)
            .map_err(|_| Fault::invalid("invalid event envelope"))?
    } else {
        let template: Template = serde_json::from_slice(&read(
            args.template
                .as_ref()
                .ok_or_else(|| Fault::invalid("event or template is required"))?,
        )?)
        .map_err(|_| Fault::invalid("invalid event template"))?;
        let (id, occurred_at) = if args.kubernetes_job {
            let api = endpoint(args.kubernetes_api_config.as_ref().unwrap())?;
            let name = std::env::var("AKH_JOB_NAME")
                .map_err(|_| Fault::invalid("AKH_JOB_NAME is required"))?;
            let namespace = std::env::var("AKH_JOB_NAMESPACE")
                .map_err(|_| Fault::invalid("AKH_JOB_NAMESPACE is required"))?;
            let job = response(
                api.http_client()?
                    .get(api.route(&format!(
                        "/apis/batch/v1/namespaces/{}/jobs/{}",
                        segment(&namespace),
                        segment(&name)
                    )))
                    .bearer_auth(api.token()?)
                    .send()
                    .await
                    .map_err(|_| Fault::new("transport", "Job metadata could not be read"))?,
            )
            .await?;
            job_occurrence(&job, &name, &namespace)?
        } else {
            (args.occurrence.ok_or_else(||Fault::invalid("template needs an explicit occurrence ID or --kubernetes-job"))?,
             args.occurred_at_ms.ok_or_else(||Fault::invalid("template needs the scheduler's stable --occurred-at-ms; do not regenerate time on retries"))?)
        };
        EventEnvelope {
            schema_version: SCHEMA_VERSION,
            id,
            occurred_at,
            source: template.source,
            event_type: template.event_type,
            subject: template.subject,
            correlation_id: template.correlation_id,
            data: template.data,
        }
    };
    if event.schema_version != SCHEMA_VERSION {
        return Err(Fault::invalid("unsupported event schema"));
    }
    validate_key(&event.id)?;
    validate_key(&event.source)?;
    validate_key(&event.event_type)?;
    let (path, body) = match args.automation {
        Some(automation) => (
            format!("/v1/automations/{}/invoke", segment(&automation)),
            json!(InvokeAutomation {
                occurrence_id: event.id.clone(),
                event
            }),
        ),
        None => ("/v1/events".into(), json!(event)),
    };
    let reply = response(
        target
            .http_client()?
            .post(target.route(&path))
            .bearer_auth(target.token()?)
            .json(&body)
            .send()
            .await
            .map_err(|_| {
                Fault::new(
                    "transport",
                    "trigger delivery was not confirmed; retry only the same occurrence",
                )
            })?,
    )
    .await?;
    println!(
        "{}",
        serde_json::to_string(&reply)
            .map_err(|_| Fault::invalid("cannot encode admission receipt"))?
    );
    Ok(())
}
#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    run(Args::parse()).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pod_retries_use_the_jobs_immutable_identity_and_timestamp() {
        let mut job = json!({"metadata":{"name":"nightly-123","namespace":"jobs","uid":"immutable-job-uid","creationTimestamp":"2026-09-12T01:00:03Z","annotations":{"batch.kubernetes.io/cronjob-scheduled-timestamp":"2026-09-12T01:00:00Z"}},"status":{"failed":0}});
        let original = job_occurrence(&job, "nightly-123", "jobs").unwrap();
        job["status"]["failed"] = json!(2);
        assert_eq!(
            job_occurrence(&job, "nightly-123", "jobs").unwrap(),
            original
        );
        assert_eq!(original.0, "k8s:jobs/immutable-job-uid");
        assert_eq!(original.1, 1789174800000);
        job["metadata"]["ownerReferences"] =
            json!([{"kind":"CronJob","controller":true,"uid":"cron-owner"}]);
        let scheduled = job_occurrence(&job, "nightly-123", "jobs").unwrap();
        job["metadata"]["uid"] = json!("duplicate-job-uid");
        assert_eq!(
            job_occurrence(&job, "nightly-123", "jobs").unwrap(),
            scheduled
        );
        assert!(job_occurrence(&job, "other", "jobs").is_err());
    }
}
