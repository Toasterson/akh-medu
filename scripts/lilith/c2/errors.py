"""Diagnostic errors for C2 tooling: stable `code`, a `help` line naming the options, and the
exit code the CLI returns for that class of failure."""
from __future__ import annotations

from dojo.errors import DojoError

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_NFS_UNMOUNTED = 3
EXIT_ACP = 4


class C2Error(DojoError):
    code = "lilith::c2"
    exit_code = EXIT_FAILED


class NfsUnmountedError(C2Error):
    code = "lilith::c2::nfs_unmounted"
    exit_code = EXIT_NFS_UNMOUNTED
    help = ("mount the akh-medu-runs share (Finder -> Go -> Connect to Server, or the usual mount command), "
            "then re-run; nothing was written. --out DIR writes somewhere else for a dry look.")


class AcpUnavailableError(C2Error):
    code = "lilith::c2::acp_unavailable"
    exit_code = EXIT_ACP
    help = ("the core must be running with a build that has `_akh/transcripts/list` (akhomed from main "
            "after ADR 116 C2), and the operator key must be readable; check --endpoint and --token-file")


class TranscriptsUnavailableError(C2Error):
    code = "lilith::c2::transcripts_unavailable"
    exit_code = EXIT_ACP
    help = "the akh must be resident on the core (wake it) and the core must be an ADR-116-C2 build"
