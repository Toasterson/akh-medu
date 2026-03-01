//! Email subsystem error types with rich miette diagnostics.

use miette::Diagnostic;
use thiserror::Error;

/// Errors specific to the email channel subsystem.
#[derive(Debug, Error, Diagnostic)]
pub enum EmailError {
    #[error("email connection failed: {message}")]
    #[diagnostic(
        code(akh::email::connection),
        help(
            "Check that the mail server is reachable and the host/port are correct. \
             For IMAP, ensure TLS is available on the specified port."
        )
    )]
    Connection { message: String },

    #[error("email authentication failed: {message}")]
    #[diagnostic(
        code(akh::email::auth),
        help(
            "Check your credentials. For app passwords, ensure they are still valid. \
             For OAuth2, the token may have expired — refresh it."
        )
    )]
    Authentication { message: String },

    #[error("email parsing failed: {message}")]
    #[diagnostic(
        code(akh::email::parse),
        help(
            "The MIME message could not be parsed. It may be malformed or use an \
             unsupported encoding. Check the raw message for RFC 5322 compliance."
        )
    )]
    Parse { message: String },

    #[error("email send failed: {message}")]
    #[diagnostic(
        code(akh::email::send),
        help(
            "SMTP delivery failed. Check the SMTP server configuration, credentials, \
             and that the recipient addresses are valid."
        )
    )]
    Send { message: String },

    #[error("email threading failed: {message}")]
    #[diagnostic(
        code(akh::email::threading),
        help(
            "The JWZ threading algorithm encountered an error. This may indicate \
             circular References headers or corrupted Message-ID data."
        )
    )]
    Threading { message: String },

    #[error("email configuration invalid: {message}")]
    #[diagnostic(
        code(akh::email::config),
        help(
            "Check the email configuration. Required fields: connection type, host, \
             port, and credentials. The poll_interval must be at least 10 seconds."
        )
    )]
    Config { message: String },

    #[error("{0}")]
    #[diagnostic(
        code(akh::email::engine),
        help("An engine-level error occurred during an email operation.")
    )]
    Engine(Box<crate::error::AkhError>),
}

impl From<crate::error::AkhError> for EmailError {
    fn from(e: crate::error::AkhError) -> Self {
        Self::Engine(Box::new(e))
    }
}

/// Convenience alias for email operations.
pub type EmailResult<T> = std::result::Result<T, EmailError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn connection_error_display() {
        let err = EmailError::Connection {
            message: "timeout after 30s".to_string(),
        };
        assert!(err.to_string().contains("timeout after 30s"));
    }

    #[test]
    fn auth_error_display() {
        let err = EmailError::Authentication {
            message: "invalid credentials".to_string(),
        };
        assert!(err.to_string().contains("invalid credentials"));
    }

    #[test]
    fn parse_error_display() {
        let err = EmailError::Parse {
            message: "missing Content-Type".to_string(),
        };
        assert!(err.to_string().contains("missing Content-Type"));
    }

    #[test]
    fn send_error_display() {
        let err = EmailError::Send {
            message: "relay denied".to_string(),
        };
        assert!(err.to_string().contains("relay denied"));
    }

    #[test]
    fn threading_error_display() {
        let err = EmailError::Threading {
            message: "circular reference".to_string(),
        };
        assert!(err.to_string().contains("circular reference"));
    }

    #[test]
    fn config_error_display() {
        let err = EmailError::Config {
            message: "missing host".to_string(),
        };
        assert!(err.to_string().contains("missing host"));
    }

    #[test]
    fn engine_error_wraps_boxed() {
        // Verify the Engine variant stores a boxed AkhError.
        let inner = crate::error::StoreError::Redb {
            message: "test engine error".to_string(),
        };
        let err = EmailError::Engine(Box::new(crate::error::AkhError::Store(inner)));
        assert!(err.to_string().contains("test engine error"));
    }

    #[test]
    fn result_alias_works() {
        let ok: EmailResult<u32> = Ok(42);
        assert_eq!(ok.unwrap(), 42);

        let err: EmailResult<u32> = Err(EmailError::Config {
            message: "bad".to_string(),
        });
        assert!(err.is_err());
    }
}
