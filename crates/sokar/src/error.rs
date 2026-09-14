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
    #[error("unknown store {0}")]
    #[diagnostic(
        code(sokar::config),
        help("Configure the named store; run sokard --check.")
    )]
    UnknownStore(StoreName),
    #[error("collection scope already exists: {0}")]
    #[diagnostic(
        code(sokar::scope_exists),
        help("Use the existing collection's store or choose a new scope.")
    )]
    ScopeExists(String),
    #[error("idempotency key reused with different content or metadata")]
    #[diagnostic(
        code(sokar::idempotency_mismatch),
        help("Retry the original request or use a new idempotency key.")
    )]
    IdempotencyMismatch,
    #[error("store mismatch: expected {expected}, got {actual}")]
    #[diagnostic(
        code(sokar::store_mismatch),
        help("Use the collection's pinned store; relocate explicitly.")
    )]
    StoreMismatch {
        expected: StoreName,
        actual: StoreName,
    },
    #[error(transparent)]
    #[diagnostic(transparent)]
    Client(#[from] sokar_client::Error),
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
    pub fn code(&self) -> &'static str {
        match self {
            Self::Config(_) | Self::UnknownStore(_) => "sokar::config",
            Self::ScopeExists(_) => "sokar::scope_exists",
            Self::IdempotencyMismatch => "sokar::idempotency_mismatch",
            Self::StoreMismatch { .. } => "sokar::store_mismatch",
            Self::Client(e) => e.diagnostic_code(),
            Self::Storage { .. } => "sokar::storage",
            Self::Database(_) | Self::Migration(_) => "sokar::database",
            Self::Authn => "sokar::authn",
            Self::Authz => "sokar::authz",
            Self::NotFound => "sokar::not_found",
            Self::Conflict(_) => "sokar::conflict",
            Self::Invalid(_) => "sokar::request",
            Self::BodyLimit => "sokar::body_limit",
            Self::ManifestSchema { .. } => "sokar::manifest_schema",
            Self::Range { .. } => "sokar::delivery",
            Self::Serde(_) => "sokar::manifest",
            Self::Io(_) => "sokar::io",
        }
    }

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
            Self::Conflict(_)
            | Self::ScopeExists(_)
            | Self::IdempotencyMismatch
            | Self::StoreMismatch { .. } => S::CONFLICT,
            Self::Client(sokar_client::Error::ManifestSchema { .. }) => S::INTERNAL_SERVER_ERROR,
            Self::Client(_) => S::BAD_REQUEST,
            Self::Invalid(_) => S::BAD_REQUEST,
            Self::BodyLimit => S::PAYLOAD_TOO_LARGE,
            Self::Range { .. } => S::RANGE_NOT_SATISFIABLE,
            Self::Storage { .. } | Self::Database(_) => S::SERVICE_UNAVAILABLE,
            _ => S::INTERNAL_SERVER_ERROR,
        }
    }
    pub fn problem(&self, request_id: String) -> sokar_client::Problem {
        let status = self.status();
        let code = self.code().to_string();
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn diagnostic_codes_match_problem_codes_for_every_variant() {
        let json_error = || serde_json::from_str::<serde_json::Value>("{").unwrap_err();
        let errors = vec![
            SokarError::Config("test".into()),
            SokarError::UnknownStore("missing".parse().unwrap()),
            SokarError::ScopeExists("scope".into()),
            SokarError::IdempotencyMismatch,
            SokarError::StoreMismatch {
                expected: "a".parse().unwrap(),
                actual: "b".parse().unwrap(),
            },
            SokarError::Client(sokar_client::Error::Invalid("test".into())),
            SokarError::Client(sokar_client::Error::ManifestSchema {
                actual: "sokar.artifact.v2".into(),
                expected: sokar_client::Artifact::SCHEMA,
            }),
            SokarError::Client(sokar_client::Error::Serde(json_error())),
            SokarError::Client(sokar_client::Error::Remote(Box::new(
                SokarError::NotFound.problem("test".into()),
            ))),
            SokarError::Client(sokar_client::Error::Transport(
                reqwest::Client::new().get("not a URL").build().unwrap_err(),
            )),
            SokarError::storage(
                &"test".parse().unwrap(),
                "read",
                opendal::Error::new(ErrorKind::NotFound, "missing"),
            ),
            SokarError::Database(sqlx::Error::RowNotFound),
            SokarError::Migration(sqlx::migrate::MigrateError::VersionMissing(1)),
            SokarError::Authn,
            SokarError::Authz,
            SokarError::NotFound,
            SokarError::Conflict("test".into()),
            SokarError::Invalid("test".into()),
            SokarError::BodyLimit,
            SokarError::ManifestSchema {
                actual: "sokar.artifact.v2".into(),
                expected: sokar_client::Artifact::SCHEMA,
            },
            SokarError::Range { size: 10 },
            SokarError::Serde(json_error()),
            SokarError::Io(std::io::Error::other("test")),
        ];
        for e in errors {
            assert_eq!(Diagnostic::code(&e).unwrap().to_string(), e.code(), "{e:?}");
            let problem = e.problem("request-123".into());
            assert_eq!(problem.code, e.code());
            assert_eq!(problem.request_id, "request-123");
        }
        let invalid = SokarError::from(sokar_client::Error::Invalid("test".into()));
        assert_eq!(invalid.status(), axum::http::StatusCode::BAD_REQUEST);
        assert!(matches!(invalid, SokarError::Client(_)));
        let schema = SokarError::from(sokar_client::Error::ManifestSchema {
            actual: "sokar.artifact.v2".into(),
            expected: sokar_client::Artifact::SCHEMA,
        });
        assert_eq!(
            schema.status(),
            axum::http::StatusCode::INTERNAL_SERVER_ERROR
        );
    }
}
