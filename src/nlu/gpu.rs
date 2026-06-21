//! GPU acceleration detection for the Candle LLM backend.
//!
//! Candle 0.10 ships `Device::{Cpu, Cuda, Metal}` only — there is **no upstream AMD
//! backend**. On the target box (AMD RX 7900 XTX, `gfx1100`) the path to running quantized
//! Gemma on the GPU is ROCm/HIP (see [ADR 068](../../docs/ai/decisions/068-rocm-candle-gpu-backend.md)).
//!
//! This module is the always-compiled *detection* layer: it probes the host for the ROCm
//! compute stack and reports, with rich [`miette`] diagnostics, exactly what the user must
//! install when the GPU is present but the userspace is missing. The actual HIP device is a
//! follow-up fork gated behind the `rocm` feature; `select_device` consults [`detect_rocm`]
//! to decide and to log a clear, honest message.
//!
//! Probing is pure filesystem/dlopen inspection — no GPU work — so it is cheap and safe to
//! call at startup regardless of feature gates.

use miette::Diagnostic;
use std::path::Path;
use thiserror::Error;

/// Result of probing the host for a usable ROCm/HIP compute stack.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RocmStatus {
    /// Kernel KFD node **and** the HIP userspace runtime are present. The GPU can run
    /// compute; the Candle HIP backend (once built, `feature = "rocm"`) can target it.
    Ready {
        /// The HIP runtime library that was found (e.g. `libamdhip64.so`).
        runtime: String,
    },
    /// The kernel compute node (`/dev/kfd`) is present — the amdgpu/KFD driver exposes the
    /// GPU for compute — but the ROCm **userspace** (HIP runtime / rocBLAS) is not
    /// installed. This is the one actionable state: install the userspace, no reboot.
    KernelReadyUserspaceMissing,
    /// No ROCm compute stack at all (no `/dev/kfd`). Either no AMD GPU, or the amdgpu KFD
    /// node is unavailable. The CPU device is used.
    Unavailable {
        /// Human-readable reason for diagnostics/logging.
        reason: String,
    },
}

impl RocmStatus {
    /// True only when the GPU is ready for compute.
    pub fn is_ready(&self) -> bool {
        matches!(self, RocmStatus::Ready { .. })
    }
}

/// The KFD compute node exposed by the amdgpu kernel driver. Its presence means the kernel
/// side of ROCm compute is live (no DKMS/reboot needed).
const KFD_NODE: &str = "/dev/kfd";

/// Candidate locations for the HIP runtime shared library.
const HIP_RUNTIME_CANDIDATES: &[&str] = &[
    "/opt/rocm/lib/libamdhip64.so",
    "/opt/rocm/lib/libamdhip64.so.6",
    "/usr/lib/x86_64-linux-gnu/libamdhip64.so",
];

/// Probe the host for a usable ROCm/HIP compute stack.
///
/// Layered on observable facts, cheapest first:
/// 1. `/dev/kfd` present? → kernel compute node live.
/// 2. a HIP runtime `.so` present? → userspace installed.
///
/// Returns [`RocmStatus::KernelReadyUserspaceMissing`] for the actionable middle state so
/// callers can surface [`GpuDiagnostic::RocmUserspaceMissing`] to the user.
pub fn detect_rocm() -> RocmStatus {
    let kfd = Path::new(KFD_NODE).exists();
    let runtime = HIP_RUNTIME_CANDIDATES
        .iter()
        .find(|p| Path::new(p).exists())
        .copied();

    match (kfd, runtime) {
        (true, Some(rt)) => RocmStatus::Ready {
            runtime: rt.to_string(),
        },
        (true, None) => RocmStatus::KernelReadyUserspaceMissing,
        (false, _) => RocmStatus::Unavailable {
            reason: format!("{KFD_NODE} not present (no AMD KFD compute node)"),
        },
    }
}

/// GPU-related diagnostics surfaced to the user via miette.
#[derive(Debug, Error, Diagnostic)]
pub enum GpuDiagnostic {
    /// The kernel exposes the GPU for compute, but the ROCm userspace is missing. This is
    /// recoverable with a userspace-only install — no kernel driver, no reboot.
    #[error("AMD GPU detected for compute, but the ROCm userspace is not installed")]
    #[diagnostic(
        code(akh::gpu::rocm_userspace_missing),
        url("https://rocm.docs.amd.com/projects/install-on-linux/en/latest/"),
        help(
            "The kernel KFD node (/dev/kfd) is present, so the GPU is ready for compute and \
             no reboot is needed — only the ROCm userspace is missing. On Ubuntu 22.04 / \
             Zorin 17 (gfx1100 is officially supported):\n\n  \
             sudo mkdir -p /etc/apt/keyrings\n  \
             curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg >/dev/null\n  \
             echo 'deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.3.4 jammy main' | sudo tee /etc/apt/sources.list.d/rocm.list\n  \
             sudo apt-get update && sudo apt-get install -y rocminfo rocm-hip-runtime rocblas hipblas\n\n\
             Then verify with `rocminfo | grep gfx` (expect gfx1100). \
             See docs/ai/plans/2026-06-04-gpu-rocm-candle.md."
        )
    )]
    RocmUserspaceMissing,

    /// The ROCm userspace is present but akh-medu was built without the `rocm` feature, so
    /// the Candle HIP backend is not compiled in. Falls back to CPU.
    #[error("ROCm runtime found ({runtime}) but akh-medu was built without the `rocm` feature")]
    #[diagnostic(
        code(akh::gpu::rocm_feature_disabled),
        help(
            "Rebuild with the GPU backend enabled, e.g.\n  \
             cargo build --release --bin akh --features candle-backend,rocm\n\n\
             Until then the LLM runs on the CPU. (The `rocm` Candle backend is a fork — see \
             ADR 068.)"
        )
    )]
    RocmFeatureDisabled { runtime: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_returns_a_status() {
        // Pure host probe — must not panic and must classify into one of the three states.
        let status = detect_rocm();
        match status {
            RocmStatus::Ready { runtime } => assert!(!runtime.is_empty()),
            RocmStatus::KernelReadyUserspaceMissing => {}
            RocmStatus::Unavailable { reason } => assert!(!reason.is_empty()),
        }
    }

    #[test]
    fn is_ready_only_for_ready() {
        assert!(
            RocmStatus::Ready {
                runtime: "x".into()
            }
            .is_ready()
        );
        assert!(!RocmStatus::KernelReadyUserspaceMissing.is_ready());
        assert!(
            !RocmStatus::Unavailable {
                reason: "x".into()
            }
            .is_ready()
        );
    }

    #[test]
    fn userspace_missing_diagnostic_has_install_help() {
        // The actionable diagnostic must carry a help message with the install command.
        let d = GpuDiagnostic::RocmUserspaceMissing;
        let help = d.help().map(|h| h.to_string()).unwrap_or_default();
        assert!(help.contains("rocm-hip-runtime"), "help must name the package");
    }
}

/// Phase 45b (ADR 073): the vendored candle-core HIP backend fills the `cuda`
/// module slot under the `hip` feature, so a HIP device constructs via the standard
/// candle `Device::new_cuda` constructor and reports a `Cuda`-slot location. Real
/// on-device init + op kernels follow in Phase 45c/45d; this guards the wiring.
#[cfg(all(test, feature = "hip"))]
mod hip_backend_tests {
    #[test]
    fn hip_device_constructs_via_cuda_slot() {
        let dev = candle_core::Device::new_cuda(0)
            .expect("HIP device should construct (cuda slot filled by hip_backend)");
        assert!(
            matches!(dev.location(), candle_core::DeviceLocation::Cuda { .. }),
            "HIP device should report a Cuda-slot location"
        );
    }
}
