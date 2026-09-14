use miette::Diagnostic;
use opendal::ErrorKind;
use sokar_client::StoreName;
pub type SokarResult<T> = Result<T, SokarError>;

#[derive(Debug, thiserror::Error, Diagnostic)]
pub enum SokarError {
    #[error("configuration error: {0}")]
    #[diagnostic(
        code(sokar::config),
        help("Check the TOML fields and env:/file: references; run sokard --check.")
    )]
    Config(String),
    #[error("store {store}: {op} failed ({kind:?})")]
    #[diagnostic(code(sokar::storage), help("{}", storage_help(*kind)))]
    Storage {
        store: StoreName,
        op: &'static str,
        kind: ErrorKind,
        #[source]
        source: opendal::Error,
    },
    #[error("database operation failed")]
    #[diagnostic(
        code(sokar::database),
        help("Check PostgreSQL connectivity and permissions; run sokard --migrate.")
    )]
    Database(#[from] sqlx::Error),
    #[error("database migration failed")]
    #[diagnostic(
        code(sokar::database),
        help("Check migration checksums and database CREATE privileges.")
    )]
    Migration(#[from] sqlx::migrate::MigrateError),
    #[error("bearer credential is unavailable, invalid or ambiguous")]
    #[diagnostic(
        code(sokar::authn),
        help("Supply a bearer token matching exactly one configured principal.")
    )]
    Authn,
    #[error("service principal is not authorized")]
    #[diagnostic(code(sokar::authz), help("Use a configured service principal."))]
    Authz,
    #[error("resource not found")]
    #[diagnostic(
        code(sokar::not_found),
        help("Check the resource UUID or scope/path lookup.")
    )]
    NotFound,
    #[error("conflict: {0}")]
    #[diagnostic(
        code(sokar::conflict),
        help("Use the existing collection or a new idempotency key for a different request.")
    )]
    Conflict(String),
    #[error("invalid request: {0}")]
    #[diagnostic(
        code(sokar::request),
        help("Check the request fields, headers and body.")
    )]
    Invalid(String),
    #[error("request body exceeds configured limit")]
    #[diagnostic(
        code(sokar::body_limit),
        help("Reduce the request size or raise limits.max_body_bytes.")
    )]
    BodyLimit,
    #[error("unsupported manifest schema {actual}; expected {expected}")]
    #[diagnostic(
        code(sokar::manifest_schema),
        help("Upgrade Sokar before serving a newer store or manifest schema.")
    )]
    ManifestSchema {
        actual: String,
        expected: &'static str,
    },
    #[error("range is unsatisfiable for {size} bytes")]
    #[diagnostic(
        code(sokar::delivery),
        help("Supply one valid bytes range within the artifact size.")
    )]
    Range { size: u64 },
    #[error("serialization error: {0}")]
    #[diagnostic(
        code(sokar::manifest),
        help("Check manifest JSON and required fields; preserve the bucket for repair.")
    )]
    Serde(#[from] serde_json::Error),
    #[error("I/O operation failed: {0}")]
    #[diagnostic(
        code(sokar::io),
        help("Check filesystem permissions and listener addresses.")
    )]
    Io(#[from] std::io::Error),
}
fn storage_help(kind: ErrorKind) -> &'static str {
    match kind {
        ErrorKind::PermissionDenied => {
            "Check store credentials and bucket policy; run sokard --check-stores."
        }
        ErrorKind::ConfigInvalid => {
            "Check the store service and option names/values; run sokard --check-stores."
        }
        ErrorKind::NotFound => {
            "Check the bucket, prefix and manifest/blob keys; reindex or restore missing objects."
        }
        ErrorKind::Unsupported => {
            "Select a store supporting read, write, copy, list, stat and delete."
        }
        ErrorKind::RateLimited => "Reduce concurrency or increase backend quotas.",
        _ => "Check backend health and connectivity; run sokard --check-stores.",
    }
}
impl SokarError {
    pub fn storage(store: &StoreName, op: &'static str, source: opendal::Error) -> Self {
        Self::Storage {
            store: store.clone(),
            op,
            kind: source.kind(),
            source,
        }
    }
    pub fn status(&self) -> axum::http::StatusCode {
        use axum::http::StatusCode as S;
        match self {
            Self::Authn => S::UNAUTHORIZED,
            Self::Authz => S::FORBIDDEN,
            Self::NotFound => S::NOT_FOUND,
            Self::Conflict(_) => S::CONFLICT,
            Self::Invalid(_) => S::BAD_REQUEST,
            Self::BodyLimit => S::PAYLOAD_TOO_LARGE,
            Self::Range { .. } => S::RANGE_NOT_SATISFIABLE,
            Self::Storage { .. } | Self::Database(_) => S::SERVICE_UNAVAILABLE,
            _ => S::INTERNAL_SERVER_ERROR,
        }
    }
    pub fn problem(&self, request_id: String) -> sokar_client::Problem {
        let status = self.status();
        let code = self
            .code()
            .expect("all errors have diagnostic codes")
            .to_string();
        sokar_client::Problem {
            type_url: format!("urn:{code}"),
            title: status.canonical_reason().unwrap_or("Error").into(),
            status: status.as_u16(),
            detail: self.to_string(),
            code,
            request_id,
        }
    }
}
impl From<sokar_client::Error> for SokarError {
    fn from(error: sokar_client::Error) -> Self {
        match error {
            sokar_client::Error::ManifestSchema { actual, expected } => {
                Self::ManifestSchema { actual, expected }
            }
            other => Self::Invalid(other.to_string()),
        }
    }
}
