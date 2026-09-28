//! Harness permission policy for tool execution.
//!
//! A [`PermissionPolicy`] gates side-effecting tools by [`PermissionMode`]
//! before [`crate::agent::tool::ToolRegistry::execute`] dispatches them. The
//! decision is **synchronous** and derived from the tool's [`ToolManifest`]
//! (its capability set + danger level): no client round-trip is needed to
//! *deny* or *allow*. When a tool needs interactive approval, the policy
//! returns [`PermissionDecision::Ask`]; the registry resolves that through an
//! optional [`PermissionAsker`] — the seam where the ACP client's
//! `session/request_permission` round-trip plugs in. With no asker attached,
//! `Ask` fails closed.
//!
//! The default mode is [`PermissionMode::Bypass`] (allow everything), so a
//! standalone autonomous akh is unaffected; a supervising client (over ACP)
//! tightens the mode for the session.

use crate::agent::tool_manifest::{Capability, DangerLevel, ToolManifest};

/// How strictly tool side effects are gated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PermissionMode {
    /// Every tool runs without gating. Default — preserves a standalone akh's
    /// autonomous behaviour.
    #[default]
    Bypass,
    /// Read/safe tools run; side-effecting tools (filesystem write, KG write,
    /// process exec, network) require approval.
    Standard,
    /// Filesystem edits are auto-approved; other side effects still require
    /// approval.
    AcceptEdits,
    /// Read-only planning: any side-effecting tool is denied outright.
    Plan,
}

/// The outcome of a permission check for one tool invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PermissionDecision {
    /// Run the tool.
    Allow,
    /// Run only if an approver grants it (resolved via [`PermissionAsker`]).
    Ask(String),
    /// Refuse outright.
    Deny(String),
}

/// A synchronous approver for [`PermissionDecision::Ask`] tools.
///
/// Implementations may block — e.g. bridge a blocking call out to an async ACP
/// `session/request_permission` round-trip and wait for the client's choice.
pub trait PermissionAsker: Send + Sync {
    /// Approve (`true`) or reject (`false`) a pending tool action.
    fn ask(&self, manifest: &ToolManifest, input_summary: &str) -> bool;
}

/// Opaque handle linking a [`ToolCallGuard::admit`] to its
/// [`ToolCallGuard::settle`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GuardTicket(pub u64);

/// A per-akh rule layer that runs after the [`PermissionPolicy`] and around
/// the tool call itself: it may refuse a call with a diagnostic, strip
/// guard-owned arguments from the input, and act on the outcome (for
/// Lilith's Chief-of-Staff profile: announce an operator-class action with
/// its receipts and rollback path — ADR 116).
pub trait ToolCallGuard: Send + Sync {
    /// Admit the call (possibly consuming guard-owned params from `input`) or
    /// refuse it with a rendered diagnostic.
    fn admit(
        &self,
        manifest: &ToolManifest,
        input: &mut crate::agent::tool::ToolInput,
    ) -> Result<GuardTicket, String>;

    /// Report how the admitted call ended. Called exactly once per admitted
    /// ticket, including when the tool panics (with [`CallOutcome::Aborted`]).
    /// An error means the guard's after-effect (e.g. the announcement)
    /// failed; the call itself already ran.
    fn settle(&self, ticket: GuardTicket, outcome: CallOutcome, detail: &str)
    -> Result<(), String>;

    /// Extra parameters the guard needs in `manifest`'s advertised schema
    /// (e.g. the Chief-of-Staff `cos_declaration`), so grammar-constrained
    /// backends can emit them. Default: none.
    fn advertised_params(
        &self,
        _manifest: &ToolManifest,
    ) -> Vec<crate::agent::tool_manifest::ToolParamSchema> {
        Vec::new()
    }
}

/// How a guarded tool call ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallOutcome {
    /// The tool returned a successful output.
    Succeeded,
    /// The tool returned a failed output or an error.
    Failed,
    /// The call unwound (panicked) before returning: its effect is unknown.
    Aborted,
}

/// Decides whether a tool may run under the active [`PermissionMode`].
#[derive(Debug, Clone, Copy, Default)]
pub struct PermissionPolicy {
    /// The active mode.
    pub mode: PermissionMode,
}

impl PermissionPolicy {
    /// Construct a policy for `mode`.
    pub fn new(mode: PermissionMode) -> Self {
        Self { mode }
    }

    /// Decide whether `manifest`'s tool may run.
    pub fn decide(&self, manifest: &ToolManifest) -> PermissionDecision {
        let caps = &manifest.danger.capabilities;
        let writes_fs = caps.contains(&Capability::WriteFilesystem);
        let side_effect = writes_fs
            || caps.contains(&Capability::WriteKg)
            || caps.contains(&Capability::ProcessExec)
            || caps.contains(&Capability::Network);

        match self.mode {
            PermissionMode::Bypass => PermissionDecision::Allow,
            PermissionMode::Plan => {
                if side_effect {
                    PermissionDecision::Deny(format!(
                        "plan mode is read-only; tool \"{}\" has side effects ({})",
                        manifest.name, manifest.danger.level
                    ))
                } else {
                    PermissionDecision::Allow
                }
            }
            PermissionMode::AcceptEdits => {
                // Filesystem edits up to Dangerous are auto-approved; anything
                // more severe, or non-edit side effects, still require approval.
                if writes_fs && manifest.danger.level <= DangerLevel::Dangerous {
                    PermissionDecision::Allow
                } else if side_effect {
                    PermissionDecision::Ask(format!(
                        "tool \"{}\" requires approval (acceptEdits covers edits only)",
                        manifest.name
                    ))
                } else {
                    PermissionDecision::Allow
                }
            }
            PermissionMode::Standard => {
                if side_effect {
                    PermissionDecision::Ask(format!(
                        "tool \"{}\" has side effects ({}) and requires approval",
                        manifest.name, manifest.danger.level
                    ))
                } else {
                    PermissionDecision::Allow
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::tool_manifest::{DangerInfo, ToolEffect, ToolManifest, ToolSource};
    use std::collections::HashSet;

    fn manifest(level: DangerLevel, caps: &[Capability]) -> ToolManifest {
        ToolManifest {
            name: "t".into(),
            description: "t".into(),
            parameters: vec![],
            danger: DangerInfo {
                effect: ToolEffect::Mutating,
                level,
                capabilities: caps.iter().copied().collect::<HashSet<_>>(),
                description: "t".into(),
                shadow_triggers: vec![],
            },
            source: ToolSource::Native,
        }
    }

    #[test]
    fn bypass_allows_everything() {
        let p = PermissionPolicy::new(PermissionMode::Bypass);
        let m = manifest(DangerLevel::Critical, &[Capability::WriteFilesystem]);
        assert_eq!(p.decide(&m), PermissionDecision::Allow);
    }

    #[test]
    fn plan_denies_side_effects_allows_reads() {
        let p = PermissionPolicy::new(PermissionMode::Plan);
        let write = manifest(DangerLevel::Dangerous, &[Capability::WriteFilesystem]);
        let read = manifest(DangerLevel::Cautious, &[Capability::ReadFilesystem]);
        assert!(matches!(p.decide(&write), PermissionDecision::Deny(_)));
        assert_eq!(p.decide(&read), PermissionDecision::Allow);
    }

    #[test]
    fn accept_edits_allows_writes_asks_for_exec() {
        let p = PermissionPolicy::new(PermissionMode::AcceptEdits);
        let edit = manifest(DangerLevel::Dangerous, &[Capability::WriteFilesystem]);
        let shell = manifest(DangerLevel::Dangerous, &[Capability::ProcessExec]);
        assert_eq!(p.decide(&edit), PermissionDecision::Allow);
        assert!(matches!(p.decide(&shell), PermissionDecision::Ask(_)));
    }

    #[test]
    fn standard_asks_for_side_effects_allows_reads() {
        let p = PermissionPolicy::new(PermissionMode::Standard);
        let net = manifest(DangerLevel::Cautious, &[Capability::Network]);
        let read = manifest(DangerLevel::Cautious, &[Capability::ReadFilesystem]);
        assert!(matches!(p.decide(&net), PermissionDecision::Ask(_)));
        assert_eq!(p.decide(&read), PermissionDecision::Allow);
    }
}
