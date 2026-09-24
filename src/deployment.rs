//! Deployed-mode policy (Phase 53 decision 6, ADR 113 decision 3).
//!
//! A deployed akhomed executes no workspace side effect in its own address
//! space. Concretely, when the process runs in [`DeploymentMode::Deployed`]:
//!
//! - the in-process executor (`[exec] kind = "in-process"`, also the implicit
//!   default when no `[exec]` is configured) is refused;
//! - in-process coding runtimes (`backend = "codex"` / `"claude"`) are refused,
//!   because in a deployment those workers are delegated through akhrtd;
//! - the legacy executors `local-drop` and `remote` are refused unless the host
//!   config opts in with `allow_legacy_exec = true`, and then admitted with a
//!   warning;
//! - `execd` (akhsbxd/akhexecd) and `none` are always admitted.
//!
//! [`DeploymentMode::Development`] is the default and admits everything, so
//! tests and local runs are unaffected. The mode comes from
//! `AkhomedConfig.deployment` or `AKH_DEPLOYMENT=deployed`. The environment can
//! only make a process stricter, never relax it.

use std::sync::OnceLock;

use miette::Diagnostic;
use thiserror::Error;

pub use akh_session_types::{DeploymentMode, UnknownDeploymentMode};

use crate::akh_config::{ChatBacking, ExecBackendConfig};

/// What would have run inside akhomed's address space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InProcessEffect {
    /// `[exec] kind = "in-process"`, explicitly or as the implicit default.
    Executor,
    /// `backend = "codex"`: the Codex app-server in-process bridge.
    CodexRuntime,
    /// `backend = "claude"`: the Claude Agent SDK in-process bridge.
    ClaudeRuntime,
}

impl std::fmt::Display for InProcessEffect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Executor => "the in-process executor ([exec] kind = \"in-process\")",
            Self::CodexRuntime => "the in-process Codex runtime (backend = \"codex\")",
            Self::ClaudeRuntime => "the in-process Claude runtime (backend = \"claude\")",
        })
    }
}

/// A configuration the deployment policy refuses.
#[derive(Debug, Error, Diagnostic)]
pub enum DeploymentPolicyError {
    /// Something would execute workspace side effects inside akhomed.
    #[error("{subject}: {effect} is forbidden in deployed mode")]
    #[diagnostic(
        code(akh::config::in_process_forbidden),
        help(
            "Deployed akhomed runs no side effect in its own address space (ADR 113). \
             Options: route tools through a sandbox with `[exec] kind = \"execd\"` \
             (akhsbxd + akhexecd); use `kind = \"none\"` for a cognition-only akh; \
             delegate Codex/Claude work to akhrtd instead of `backend = \"codex\"|\"claude\"`; \
             or, for local development only, drop `deployment = \"deployed\"` and \
             `AKH_DEPLOYMENT=deployed`."
        )
    )]
    InProcessForbidden {
        subject: String,
        effect: InProcessEffect,
    },

    /// A legacy host-local executor without the explicit opt-in.
    #[error(
        "{subject}: exec kind \"{kind}\" is a legacy executor; deployed mode admits it only \
         with allow_legacy_exec = true"
    )]
    #[diagnostic(
        code(akh::config::legacy_exec_forbidden),
        help(
            "`local-drop` and `remote` execute outside akhsbxd/akhexecd, so their effects \
             are not leased, receipted or fenced. Prefer `[exec] kind = \"execd\"`. If this \
             host must keep the legacy executor for now, set `allow_legacy_exec = true` at \
             the top of akhomed's config.toml (above any [table]) and accept the startup \
             warning."
        )
    )]
    LegacyExecForbidden { subject: String, kind: &'static str },

    /// `AKH_DEPLOYMENT` is set to something that is not a mode.
    #[error(transparent)]
    #[diagnostic(
        code(akh::config::deployment_mode),
        help("Set AKH_DEPLOYMENT to \"deployed\" or \"development\", or unset it.")
    )]
    UnknownMode(#[from] UnknownDeploymentMode),
}

impl DeploymentPolicyError {
    /// One line with the message, diagnostic code and help, for call sites
    /// that report errors as strings (akhomed's lazy agent load).
    pub fn summary(&self) -> String {
        let code = self
            .code()
            .map(|code| format!(" [{code}]"))
            .unwrap_or_default();
        let help = self
            .help()
            .map(|help| format!(" help: {help}"))
            .unwrap_or_default();
        format!("{self}{code}{help}")
    }
}

/// How the policy admitted a configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecAdmission {
    /// Nothing to report.
    Permitted,
    /// A legacy executor admitted through `allow_legacy_exec`; the caller
    /// should warn.
    LegacyOptIn { kind: &'static str },
}

/// The deployment policy for one process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ExecPolicy {
    pub mode: DeploymentMode,
    pub allow_legacy_exec: bool,
}

impl ExecPolicy {
    /// Admit the executor and chat backing one akh would run with.
    pub fn check(
        &self,
        subject: &str,
        exec: &ExecBackendConfig,
        backing: ChatBacking,
    ) -> Result<ExecAdmission, DeploymentPolicyError> {
        if !self.mode.is_deployed() {
            return Ok(ExecAdmission::Permitted);
        }
        let runtime = match backing {
            ChatBacking::Codex => Some(InProcessEffect::CodexRuntime),
            ChatBacking::Claude => Some(InProcessEffect::ClaudeRuntime),
            _ => None,
        };
        if let Some(effect) = runtime {
            return Err(DeploymentPolicyError::InProcessForbidden {
                subject: subject.to_owned(),
                effect,
            });
        }
        self.check_exec(subject, exec)
    }

    /// Admit an executor on its own (the host-level `[exec]` default).
    pub fn check_exec(
        &self,
        subject: &str,
        exec: &ExecBackendConfig,
    ) -> Result<ExecAdmission, DeploymentPolicyError> {
        if !self.mode.is_deployed() {
            return Ok(ExecAdmission::Permitted);
        }
        match exec {
            ExecBackendConfig::InProcess => Err(DeploymentPolicyError::InProcessForbidden {
                subject: subject.to_owned(),
                effect: InProcessEffect::Executor,
            }),
            ExecBackendConfig::LocalDrop { .. } | ExecBackendConfig::Remote { .. } => {
                if self.allow_legacy_exec {
                    Ok(ExecAdmission::LegacyOptIn {
                        kind: exec.kind_name(),
                    })
                } else {
                    Err(DeploymentPolicyError::LegacyExecForbidden {
                        subject: subject.to_owned(),
                        kind: exec.kind_name(),
                    })
                }
            }
            ExecBackendConfig::Execd { .. } | ExecBackendConfig::None => {
                Ok(ExecAdmission::Permitted)
            }
        }
    }

    /// [`Self::check`], logging a warning for a legacy opt-in.
    pub fn admit(
        &self,
        subject: &str,
        exec: &ExecBackendConfig,
        backing: ChatBacking,
    ) -> Result<(), DeploymentPolicyError> {
        warn_on_legacy(subject, self.check(subject, exec, backing)?);
        Ok(())
    }
}

/// Log the warning a legacy opt-in owes the operator.
pub fn warn_on_legacy(subject: &str, admission: ExecAdmission) {
    if let ExecAdmission::LegacyOptIn { kind } = admission {
        tracing::warn!(
            target: "akh::deployment",
            subject,
            kind,
            "deployed mode admits a legacy executor through allow_legacy_exec = true; its \
             effects bypass akhsbxd/akhexecd leases and receipts (move to kind = \"execd\")"
        );
    }
}

impl ExecPolicy {
    /// Combine this (startup) policy with a later `requested` one under
    /// only-stricter semantics: the stricter mode wins, and a legacy executor
    /// stays admitted only if **both** allow it. A config replaced after
    /// startup can therefore tighten the policy but never relax it.
    pub fn only_stricter(self, requested: ExecPolicy) -> ExecPolicy {
        ExecPolicy {
            mode: self.mode.max(requested.mode),
            allow_legacy_exec: self.allow_legacy_exec && requested.allow_legacy_exec,
        }
    }
}

static PROCESS_MODE: OnceLock<DeploymentMode> = OnceLock::new();
static PROCESS_POLICY: OnceLock<ExecPolicy> = OnceLock::new();

/// Record the mode akhomed resolved at startup. The first call wins; a later
/// call returns the mode already in force.
pub fn set_process_mode(mode: DeploymentMode) -> Result<(), DeploymentMode> {
    PROCESS_MODE.set(mode).map_err(|_| process_mode())
}

/// Record the whole policy akhomed resolved at startup (mode **and**
/// `allow_legacy_exec`). The first call wins; a later call returns the policy
/// already in force. Also records the mode for [`process_mode`].
pub fn set_process_policy(policy: ExecPolicy) -> Result<(), ExecPolicy> {
    let _ = set_process_mode(policy.mode);
    PROCESS_POLICY.set(policy).map_err(|_| process_policy())
}

/// The policy fixed at startup. Before [`set_process_policy`] ran, it is
/// [`process_mode`] with no legacy opt-in (fail closed).
pub fn process_policy() -> ExecPolicy {
    PROCESS_POLICY.get().copied().unwrap_or(ExecPolicy {
        mode: process_mode(),
        allow_legacy_exec: false,
    })
}

/// The process's deployment mode: what akhomed recorded at startup, otherwise
/// `AKH_DEPLOYMENT`. An unreadable `AKH_DEPLOYMENT` fails closed to
/// [`DeploymentMode::Deployed`].
pub fn process_mode() -> DeploymentMode {
    if let Some(mode) = PROCESS_MODE.get() {
        return *mode;
    }
    DeploymentMode::resolve_from_env(DeploymentMode::Development)
        .unwrap_or(DeploymentMode::Deployed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deployed(allow_legacy_exec: bool) -> ExecPolicy {
        ExecPolicy {
            mode: DeploymentMode::Deployed,
            allow_legacy_exec,
        }
    }

    fn local_drop() -> ExecBackendConfig {
        ExecBackendConfig::LocalDrop {
            user: "akh".into(),
            workdir: None,
            shell: None,
            path: None,
        }
    }

    fn execd() -> ExecBackendConfig {
        ExecBackendConfig::Execd {
            sandbox_url: "https://akhsbxd.lab".into(),
            sandbox_token_file: "/run/akh/token".into(),
            executor_url: None,
            executor_token_file: None,
            profile: "rust".into(),
            principal: "janus".into(),
            poll_wait_ms: 5_000,
        }
    }

    fn code(error: &DeploymentPolicyError) -> String {
        error.code().map(|c| c.to_string()).unwrap_or_default()
    }

    #[test]
    fn development_admits_everything() {
        let policy = ExecPolicy::default();
        for exec in [ExecBackendConfig::InProcess, local_drop(), execd()] {
            for backing in [ChatBacking::Ollama, ChatBacking::Codex, ChatBacking::Claude] {
                assert_eq!(
                    policy.check("janus", &exec, backing).unwrap(),
                    ExecAdmission::Permitted
                );
            }
        }
    }

    #[test]
    fn deployed_mode_rejects_in_process_executor_and_runtimes() {
        let policy = deployed(true);
        let error = policy
            .check("janus", &ExecBackendConfig::InProcess, ChatBacking::Ollama)
            .unwrap_err();
        assert_eq!(code(&error), "akh::config::in_process_forbidden");
        assert!(matches!(
            error,
            DeploymentPolicyError::InProcessForbidden {
                effect: InProcessEffect::Executor,
                ..
            }
        ));
        for (backing, effect) in [
            (ChatBacking::Codex, InProcessEffect::CodexRuntime),
            (ChatBacking::Claude, InProcessEffect::ClaudeRuntime),
        ] {
            // Even with a sandboxed executor, an in-process coding runtime is refused.
            let error = policy.check("minerva", &execd(), backing).unwrap_err();
            assert_eq!(code(&error), "akh::config::in_process_forbidden");
            assert!(matches!(
                error,
                DeploymentPolicyError::InProcessForbidden { effect: e, .. } if e == effect
            ));
        }
        let summary = policy
            .check("vesta", &ExecBackendConfig::InProcess, ChatBacking::Ollama)
            .unwrap_err()
            .summary();
        assert!(summary.starts_with("vesta: "), "{summary}");
        assert!(
            summary.contains("[akh::config::in_process_forbidden]"),
            "{summary}"
        );
        assert!(summary.contains("kind = \"execd\""), "{summary}");
    }

    #[test]
    fn a_later_policy_can_only_tighten_the_startup_policy() {
        let startup = ExecPolicy {
            mode: DeploymentMode::Deployed,
            allow_legacy_exec: false,
        };
        // A swapped config that relaxes both knobs changes nothing.
        let relaxed = startup.only_stricter(ExecPolicy {
            mode: DeploymentMode::Development,
            allow_legacy_exec: true,
        });
        assert_eq!(relaxed, startup);
        assert!(relaxed.check_exec("a", &local_drop()).is_err());
        // A startup opt-in can still be withdrawn by a later config.
        let opted_in = deployed(true);
        assert_eq!(opted_in.only_stricter(deployed(false)), deployed(false));
        // Development at startup, deployed later: the stricter mode wins.
        let dev = ExecPolicy::default();
        assert_eq!(
            dev.only_stricter(deployed(false)).mode,
            DeploymentMode::Deployed
        );
    }

    #[test]
    fn deployed_mode_admits_legacy_executors_only_with_the_opt_in() {
        let error = deployed(false)
            .check("janus", &local_drop(), ChatBacking::Ollama)
            .unwrap_err();
        assert_eq!(code(&error), "akh::config::legacy_exec_forbidden");
        assert_eq!(
            deployed(true)
                .check("janus", &local_drop(), ChatBacking::Ollama)
                .unwrap(),
            ExecAdmission::LegacyOptIn { kind: "local-drop" }
        );
        for exec in [execd(), ExecBackendConfig::None] {
            assert_eq!(
                deployed(false)
                    .check("janus", &exec, ChatBacking::Ollama)
                    .unwrap(),
                ExecAdmission::Permitted
            );
        }
    }
}
