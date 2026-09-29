"""Diagnostic errors for the CoS dojo.

Python counterpart of the repo's miette pattern: every error carries a stable
`code` and a `help` line that names the options the caller has.
"""
from __future__ import annotations


class DojoError(Exception):
    """Base error: `code` is stable and greppable, `help` says what to do."""

    code: str = "lilith::dojo"
    help: str = ""

    def __init__(self, message: str, *, help: str | None = None):
        super().__init__(message)
        if help is not None:
            self.help = help

    def render(self) -> str:
        text = f"[{self.code}] {self}"
        if self.help:
            text += f"\n  help: {self.help}"
        return text


class SnapshotFormatError(DojoError):
    code = "lilith::dojo::snapshot_format"
    help = (
        "the directory is not a `lilith/cos-snapshot@1` snapshot; point --root at the recorder's "
        "snapshot root (the directory holding `<ts>/manifest.json` and `objects/`)"
    )


class ObjectIntegrityError(DojoError):
    code = "lilith::dojo::object_integrity"
    help = (
        "an object's bytes no longer hash to its file name; the snapshot root was modified. "
        "Restore it from the NFS copy, or drop the snapshot from the window"
    )


class UnknownSnapshotError(DojoError):
    code = "lilith::dojo::unknown_snapshot"
    help = "list the available ids with `dojo_report.py --root <root>`"


class LedgerError(DojoError):
    code = "lilith::dojo::ledger"
    help = (
        "the dojo tenancy ledger is append-only and hash-chained; start a new ledger file for a new "
        "tenancy instead of editing an existing one"
    )


class FrozenFloorsError(DojoError):
    code = "lilith::dojo::floors_frozen"
    help = (
        "exam floors are frozen before any candidate is scored; do not edit exam-floors-v1.json. "
        "Write exam-floors-v2.json with its own pinned hash and record the change in the plan"
    )


class SplitLeakError(DojoError):
    code = "lilith::dojo::split_leak"
    help = (
        "a split assignment would let sealed-period evidence reach train/dev; shorten the label "
        "horizon, or move the episode to the sealed split"
    )


class ModelError(DojoError):
    """The model endpoint failed (transport, HTTP status, or an unreadable reply).

    The runner turns this into a failed episode with the reason recorded; it never
    aborts the batch."""

    code = "lilith::dojo::model"
    help = (
        "check the endpoint host, the key source and the model id; a 5xx or 429 is retried a few "
        "times already, so re-run the same command to retry only the failed episodes (resume keeps "
        "the completed ones)"
    )


class SecretSourceError(DojoError):
    code = "lilith::dojo::secret_source"
    help = (
        "name an environment variable with --key-env, or a file with --key-file (mode 0600: "
        "`chmod 600 <file>`); the key itself is never accepted on the command line"
    )


class BrokerError(DojoError):
    code = "lilith::dojo::broker"
    help = "the replay servers do not share one receipt log; build them with `replay_server.episode_servers`"


class RunProvenanceError(DojoError):
    code = "lilith::dojo::run_provenance"
    help = (
        "commit or stash the tracked changes so the run names a real commit, or pass --allow-dirty "
        "for a throwaway trial run (the summary then records `git_dirty: true`); resuming needs the "
        "same model, prompt, snapshots and seed, otherwise use a fresh --out directory"
    )


class CompareError(DojoError):
    code = "lilith::dojo::compare"
    help = "compare needs `baseline-summary.json` files written by `runner.py run`; check the paths"
