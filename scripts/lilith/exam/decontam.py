"""Decontamination hook: training material vs the manifest's frozen sentinels (P6.3 method).

Reuses `dojo.splits` (NFKC, per-character lowercase, alphanumeric words; exact and
13-word-shingle BLAKE3 hashes; baseline subtraction), and adds the manifest binding:
every sentinel file must hash to its pin and the baseline file to the decontamination
pin before the index is built, so the check runs against exactly the frozen cases.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

from dojo.snapshot import sha256_hex
from dojo.splits import DecontaminationReport, SealedIndex, decontaminate
from exam.errors import ManifestIntegrityError
from exam.manifest import ExamManifest


def sentinel_index(manifest: ExamManifest, sentinel_files: Mapping[str, Path],
                   baseline_file: Path | None = None) -> SealedIndex:
    """`sentinel_files`: family -> the frozen sentinel JSONL. Every manifest family is required."""
    manifest.require_scorable()
    missing = sorted(set(manifest.sentinels) - set(sentinel_files))
    if missing:
        raise ManifestIntegrityError(f"no sentinel file given for {missing}",
                                     help="pass every family's frozen sentinel file; decontamination against a "
                                          "subset would let the others leak")
    for family, path in sentinel_files.items():
        manifest.verify_sentinel_file(family, path)
    index = SealedIndex.from_jsonl(sentinel_files[f] for f in sorted(sentinel_files))
    pin = manifest.decontamination.sentinel_baseline_sha256
    if baseline_file is not None or pin is not None:
        if baseline_file is None or pin is None:
            raise ManifestIntegrityError("sentinel baseline given without a manifest pin (or pinned but not given)")
        actual = sha256_hex(Path(baseline_file).read_bytes())
        if actual != pin:
            raise ManifestIntegrityError(f"{baseline_file} hashes to {actual}; manifest pins {pin}")
        index.subtract_baseline(line for line in Path(baseline_file).read_text().splitlines() if line.strip())
    return index


def decontaminate_training(records: Iterable[tuple[str, Sequence[str]]], manifest: ExamManifest,
                           sentinel_files: Mapping[str, Path],
                           baseline_file: Path | None = None) -> tuple[list[str], DecontaminationReport]:
    """`records`: (record id, texts). Returns the kept ids and the report (dropped ids + reason)."""
    return decontaminate(records, sentinel_index(manifest, sentinel_files, baseline_file))
