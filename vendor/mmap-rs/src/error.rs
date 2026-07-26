//! This module implements the error type used throughout this crate.

use crate::UnsafeMmapFlags;
use thiserror::Error;

/// The error type.
#[derive(Debug, Error)]
pub enum Error {
    /// The following set of unsafe flags must be set to call this function.
    #[error("{0:?} must be set")]
    UnsafeFlagNeeded(UnsafeMmapFlags),

    /// The size is invalid.
    #[error("invalid size")]
    InvalidSize,

    /// The offset is invalid.
    #[error("invalid offset")]
    InvalidOffset,

    /// The operation is invalid.
    #[error("invalid operation")]
    InvalidOperation,

    /// The memory maps must be adjacent.
    #[error("the memory maps must be adjacent")]
    MustBeAdjacent,

    /// The attributes of the memory maps do not match.
    #[error("the memory maps must share the same attributes")]
    AttributeMismatch,

    /// The backings of the memory maps do not match.
    #[error("the memory maps must share the same backing")]
    BackingMismatch,

    /// Represents [`std::io::Error`].
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// Represents [`std::num::ParseIntError`].
    #[error(transparent)]
    ParseInt(#[from] std::num::ParseIntError),

    /// Represents [`std::num::TryFromIntError`].
    #[error(transparent)]
    TryFromInt(#[from] std::num::TryFromIntError),

    /// Represents [`std::str::Utf8Error`].
    #[error(transparent)]
    Utf8(std::str::Utf8Error),

    #[cfg(unix)]
    /// Represents [`nix::Errno`].
    #[error(transparent)]
    Nix(#[from] nix::errno::Errno),

    // illumos carve-out: the `sysctl` crate (v0.5/0.6, pulled in unconditionally
    // for every `cfg(unix)` target by mmap-rs's own Cargo.toml) only implements a
    // `sys` backend for linux/android and macos/ios/tvos/visionos/freebsd — it
    // has no illumos module at all, so it fails to compile there (E0432/E0433:
    // `mod sys` is "configured out"). Nothing in this crate's illumos path
    // (os_impl::unix, which illumos falls through to) ever constructs a
    // `sysctl::SysctlError`, so the variant is dead code on illumos anyway;
    // narrowing the cfg drops the otherwise-unsatisfiable dependency on illumos
    // without touching behavior on any platform that actually uses it.
    #[cfg(all(unix, not(target_os = "illumos")))]
    /// Represents [`sysctl::SysctlError`].
    #[error(transparent)]
    Sysctl(#[from] sysctl::SysctlError),

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    /// The error code returned from the Mach API.
    #[error("Mach kernel result = {0}")]
    Mach(libc::c_int),

    #[cfg(target_os = "windows")]
    /// Represents [`windows::core::Error`].
    #[error(transparent)]
    Windows(#[from] windows::core::Error),
}
