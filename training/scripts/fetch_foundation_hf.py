#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#   "datasets==5.0.1",
# ]
# ///
"""Stream a pinned Hugging Face mixture into canonical FoundationRecords.

Run production acquisition with ``uv run`` so the exact client version above
is used.  The script deliberately does not download whole repositories: every
source is opened as an iterable dataset, projected to the required columns,
and bounded by explicit record, scan, and output-byte limits.

Completed sources are immutable, content-verified directories.  If a later
source is interrupted, rerunning the same command verifies and reuses the
completed sources.  A final ``acquisition.json`` is written only when every
source reached both of its deterministic train/evaluation quotas.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
FOUNDATION_RECORD_SCHEMA_VERSION = 1
GENERATOR_VERSION = "akh-hf-foundation-fetch-v1"
SELECTION_ALGORITHM = "stream-shuffle-buffer+stable-id-hash-split-v1"
SOURCE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
MIB = 1024 * 1024
GIB = 1024 * MIB


class AcquisitionError(RuntimeError):
    """A fail-closed acquisition or verification error."""


@dataclass(frozen=True)
class SourceSpec:
    name: str
    dataset: str
    config: str
    split: str
    revision: str
    license: str
    hub_license_tag: str
    language: str
    text_field: str
    id_field: str | None
    language_field: str | None
    accepted_language_values: tuple[str, ...]
    train_records: int
    evaluation_records: int
    evaluation_basis_points: int
    min_chars: int
    max_chars: int
    shuffle_buffer: int
    max_scan_records: int
    max_output_bytes: int
    dataset_card: str
    license_url: str

    @property
    def source_label(self) -> str:
        return f"hf:{self.dataset}/{self.config}"


@dataclass(frozen=True)
class AcquisitionConfig:
    run_id: str
    shuffle_seed: int
    minimum_free_bytes_after_run: int
    sources: tuple[SourceSpec, ...]


@dataclass(frozen=True)
class EffectiveSourceSpec:
    source: SourceSpec
    shuffle_seed: int
    train_records: int
    evaluation_records: int
    evaluation_basis_points: int
    max_scan_records: int
    max_output_bytes: int


@dataclass
class Counters:
    scanned: int = 0
    written_training: int = 0
    written_evaluation: int = 0
    dropped_short: int = 0
    dropped_long: int = 0
    dropped_invalid: int = 0
    dropped_language: int = 0
    dropped_duplicate_id: int = 0
    skipped_partition_full: int = 0


class HashedJsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._file = path.open("xb")
        self._hasher = hashlib.sha256()
        self.records = 0
        self.bytes = 0

    def write(self, record: dict[str, Any]) -> int:
        encoded = canonical_json(record) + b"\n"
        self._file.write(encoded)
        self._hasher.update(encoded)
        self.records += 1
        self.bytes += len(encoded)
        return len(encoded)

    def close(self) -> dict[str, Any]:
        self._file.flush()
        os.fsync(self._file.fileno())
        self._file.close()
        return {
            "file": self.path.name,
            "sha256": f"sha256:{self._hasher.hexdigest()}",
            "records": self.records,
            "bytes": self.bytes,
        }

    def abort(self) -> None:
        if not self._file.closed:
            self._file.close()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def hash_bytes(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(MIB):
            digest.update(chunk)
            size += len(chunk)
    return f"sha256:{digest.hexdigest()}", size


def stable_digest(parts: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = part.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def upstream_id(row: dict[str, Any], spec: SourceSpec, text: str) -> str:
    value = row.get(spec.id_field) if spec.id_field else None
    if isinstance(value, (str, int, float, bool)) and str(value).strip():
        return str(value)
    return f"text-sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def record_id(spec: SourceSpec, row_id: str, text: str) -> str:
    digest = stable_digest(
        [spec.dataset, spec.config, spec.split, spec.revision, row_id, text]
    )
    return f"{spec.name}-{digest}"


def evaluation_partition(
    spec: SourceSpec, row_id: str, basis_points: int
) -> bool:
    digest = stable_digest(
        [spec.dataset, spec.config, spec.split, spec.revision, row_id]
    )
    bucket = int(digest[:16], 16) % 10_000
    return bucket < basis_points


def make_record(
    spec: SourceSpec, row: dict[str, Any], text: str, row_id: str
) -> dict[str, Any]:
    return {
        "schema_version": FOUNDATION_RECORD_SCHEMA_VERSION,
        "id": record_id(spec, row_id, text),
        "source": spec.source_label,
        "source_revision": spec.revision,
        "license": spec.license,
        "language": spec.language,
        "text": text,
    }


def require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AcquisitionError(f"{label} must be a JSON object")
    return value


def require_exact_keys(
    value: dict[str, Any], required: set[str], optional: set[str], label: str
) -> None:
    missing = sorted(required - value.keys())
    unknown = sorted(value.keys() - required - optional)
    if missing or unknown:
        raise AcquisitionError(
            f"{label} keys are invalid: missing={missing}, unknown={unknown}"
        )


def positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise AcquisitionError(f"{label} must be a positive integer")
    return value


def nonblank(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AcquisitionError(f"{label} must be a non-blank string")
    return value


def optional_field(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return nonblank(value, label)


def load_config(path: Path) -> tuple[AcquisitionConfig, str, dict[str, Any]]:
    raw_bytes = path.read_bytes()
    try:
        raw = require_object(json.loads(raw_bytes), "configuration")
    except json.JSONDecodeError as error:
        raise AcquisitionError(f"{path} is not valid JSON: {error}") from error
    require_exact_keys(
        raw,
        {
            "schema_version",
            "run_id",
            "selection_algorithm",
            "minimum_free_bytes_after_run",
            "shuffle_seed",
            "sources",
        },
        {"description"},
        "configuration",
    )
    if raw["schema_version"] != SCHEMA_VERSION:
        raise AcquisitionError(
            f"configuration schema {raw['schema_version']!r} is unsupported"
        )
    if raw["selection_algorithm"] != SELECTION_ALGORITHM:
        raise AcquisitionError(
            "configuration selection_algorithm does not match this executable"
        )
    run_id = nonblank(raw["run_id"], "run_id")
    shuffle_seed = positive_int(raw["shuffle_seed"], "shuffle_seed")
    free_after = positive_int(
        raw["minimum_free_bytes_after_run"], "minimum_free_bytes_after_run"
    )
    if not isinstance(raw["sources"], list) or not raw["sources"]:
        raise AcquisitionError("sources must be a non-empty array")

    required = {
        "name",
        "dataset",
        "config",
        "split",
        "revision",
        "license",
        "hub_license_tag",
        "language",
        "text_field",
        "id_field",
        "language_field",
        "accepted_language_values",
        "train_records",
        "evaluation_records",
        "evaluation_basis_points",
        "min_chars",
        "max_chars",
        "shuffle_buffer",
        "max_scan_records",
        "max_output_bytes",
        "dataset_card",
        "license_url",
    }
    specs: list[SourceSpec] = []
    names: set[str] = set()
    for index, item in enumerate(raw["sources"]):
        label = f"sources[{index}]"
        item = require_object(item, label)
        require_exact_keys(item, required, set(), label)
        name = nonblank(item["name"], f"{label}.name")
        if not SOURCE_NAME_RE.fullmatch(name):
            raise AcquisitionError(f"{label}.name is not a safe source name: {name!r}")
        if name in names:
            raise AcquisitionError(f"duplicate source name {name!r}")
        names.add(name)
        dataset = nonblank(item["dataset"], f"{label}.dataset")
        if dataset.count("/") != 1:
            raise AcquisitionError(f"{label}.dataset must be OWNER/NAME")
        revision = nonblank(item["revision"], f"{label}.revision")
        if not REVISION_RE.fullmatch(revision):
            raise AcquisitionError(f"{label}.revision must be a full commit SHA")
        accepted = item["accepted_language_values"]
        if not isinstance(accepted, list) or not accepted or not all(
            isinstance(entry, str) and entry for entry in accepted
        ):
            raise AcquisitionError(
                f"{label}.accepted_language_values must be a non-empty string array"
            )
        language_field = optional_field(
            item["language_field"], f"{label}.language_field"
        )
        if language_field is None and accepted:
            raise AcquisitionError(
                f"{label} has accepted_language_values but no language_field"
            )
        train_records = positive_int(item["train_records"], f"{label}.train_records")
        eval_records = positive_int(
            item["evaluation_records"], f"{label}.evaluation_records"
        )
        eval_bps = positive_int(
            item["evaluation_basis_points"],
            f"{label}.evaluation_basis_points",
        )
        if eval_bps >= 5_000:
            raise AcquisitionError(
                f"{label}.evaluation_basis_points must be below 5000"
            )
        min_chars = positive_int(item["min_chars"], f"{label}.min_chars")
        max_chars = positive_int(item["max_chars"], f"{label}.max_chars")
        if min_chars > max_chars:
            raise AcquisitionError(f"{label}.min_chars exceeds max_chars")
        max_scan = positive_int(
            item["max_scan_records"], f"{label}.max_scan_records"
        )
        if max_scan < train_records + eval_records:
            raise AcquisitionError(f"{label}.max_scan_records cannot satisfy its quotas")
        specs.append(
            SourceSpec(
                name=name,
                dataset=dataset,
                config=nonblank(item["config"], f"{label}.config"),
                split=nonblank(item["split"], f"{label}.split"),
                revision=revision,
                license=nonblank(item["license"], f"{label}.license"),
                hub_license_tag=nonblank(
                    item["hub_license_tag"], f"{label}.hub_license_tag"
                ),
                language=nonblank(item["language"], f"{label}.language"),
                text_field=nonblank(item["text_field"], f"{label}.text_field"),
                id_field=optional_field(item["id_field"], f"{label}.id_field"),
                language_field=language_field,
                accepted_language_values=tuple(accepted),
                train_records=train_records,
                evaluation_records=eval_records,
                evaluation_basis_points=eval_bps,
                min_chars=min_chars,
                max_chars=max_chars,
                shuffle_buffer=positive_int(
                    item["shuffle_buffer"], f"{label}.shuffle_buffer"
                ),
                max_scan_records=max_scan,
                max_output_bytes=positive_int(
                    item["max_output_bytes"], f"{label}.max_output_bytes"
                ),
                dataset_card=nonblank(
                    item["dataset_card"], f"{label}.dataset_card"
                ),
                license_url=nonblank(
                    item["license_url"], f"{label}.license_url"
                ),
            )
        )
    config = AcquisitionConfig(run_id, shuffle_seed, free_after, tuple(specs))
    return config, hash_bytes(raw_bytes), raw


def effective_spec(
    spec: SourceSpec, smoke_records: int | None, shuffle_seed: int
) -> EffectiveSourceSpec:
    if smoke_records is None:
        return EffectiveSourceSpec(
            source=spec,
            shuffle_seed=shuffle_seed,
            train_records=spec.train_records,
            evaluation_records=spec.evaluation_records,
            evaluation_basis_points=spec.evaluation_basis_points,
            max_scan_records=spec.max_scan_records,
            max_output_bytes=spec.max_output_bytes,
        )
    train_records = smoke_records
    evaluation_records = max(1, smoke_records // 10)
    evaluation_basis_points = max(spec.evaluation_basis_points, 1_000)
    worst_record_bytes = spec.max_chars * 4 + 2_048
    return EffectiveSourceSpec(
        source=spec,
        shuffle_seed=shuffle_seed,
        train_records=train_records,
        evaluation_records=evaluation_records,
        evaluation_basis_points=evaluation_basis_points,
        max_scan_records=max(1_000, (train_records + evaluation_records) * 100),
        max_output_bytes=min(
            spec.max_output_bytes,
            (train_records + evaluation_records) * worst_record_bytes,
        ),
    )


def nearest_existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise AcquisitionError(f"cannot resolve an existing parent for {path}")
        candidate = parent
    return candidate


def disk_preflight(
    out: Path,
    config: AcquisitionConfig,
    effective: tuple[EffectiveSourceSpec, ...],
    smoke_records: int | None,
    enforce: bool,
) -> dict[str, int | bool]:
    usage = shutil.disk_usage(nearest_existing_parent(out))
    output_cap = sum(item.max_output_bytes for item in effective)
    reserve = GIB if smoke_records is not None else config.minimum_free_bytes_after_run
    required = reserve + output_cap
    result: dict[str, int | bool] = {
        "free_bytes": usage.free,
        "output_cap_bytes": output_cap,
        "reserved_after_run_bytes": reserve,
        "required_free_bytes": required,
        "ok": usage.free >= required,
    }
    if enforce and not result["ok"]:
        raise AcquisitionError(
            "disk preflight failed: "
            f"free={usage.free / GIB:.1f} GiB, required={required / GIB:.1f} GiB "
            f"(bounded output={output_cap / GIB:.1f} GiB, "
            f"post-run reserve={reserve / GIB:.1f} GiB); use a larger volume"
        )
    return result


def source_config_hash(item: EffectiveSourceSpec) -> str:
    return hash_bytes(canonical_json(asdict(item)))


def import_hf_clients() -> tuple[Any, Any, str]:
    try:
        import datasets
        from huggingface_hub import HfApi
    except ImportError as error:
        raise AcquisitionError(
            "Hugging Face dependencies are absent; run this script with `uv run`"
        ) from error
    return datasets.load_dataset, HfApi, datasets.__version__


def verify_hub_source(api: Any, spec: SourceSpec, token: str | None) -> str:
    info = api(token=token).dataset_info(spec.dataset, revision=spec.revision)
    resolved = str(info.sha)
    if resolved != spec.revision:
        raise AcquisitionError(
            f"{spec.name}: requested revision {spec.revision}, Hub resolved {resolved}"
        )
    if getattr(info, "private", False) or getattr(info, "gated", False):
        raise AcquisitionError(
            f"{spec.name}: source unexpectedly became private or gated"
        )
    tags = set(getattr(info, "tags", ()) or ())
    if spec.hub_license_tag not in tags:
        raise AcquisitionError(
            f"{spec.name}: expected Hub tag {spec.hub_license_tag!r}, got {sorted(tags)}"
        )
    return resolved


def stream_source(
    item: EffectiveSourceSpec,
    source_index: int,
    output_root: Path,
    load_dataset: Any,
    hf_api: Any,
    datasets_version: str,
    token: str | None,
) -> dict[str, Any]:
    spec = item.source
    final_dir = output_root / f"source-{source_index:02}-{spec.name}"
    expected_hash = source_config_hash(item)
    if final_dir.exists():
        return verify_completed_source(final_dir, expected_hash)

    temp_dir = output_root / f".{final_dir.name}.tmp-{os.getpid()}"
    if temp_dir.exists():
        raise AcquisitionError(f"stale temporary source directory: {temp_dir}")
    temp_dir.mkdir()
    training_writer = HashedJsonlWriter(temp_dir / "training.jsonl")
    evaluation_writer = HashedJsonlWriter(temp_dir / "evaluation.jsonl")
    counters = Counters()
    ids: set[str] = set()
    try:
        resolved_revision = verify_hub_source(hf_api, spec, token)
        print(
            f"{spec.name}: streaming {spec.dataset}/{spec.config}@"
            f"{resolved_revision[:12]} (train={item.train_records}, "
            f"eval={item.evaluation_records})",
            flush=True,
        )
        dataset = load_dataset(
            spec.dataset,
            name=spec.config,
            split=spec.split,
            revision=spec.revision,
            streaming=True,
            token=token,
        )
        required_columns = [spec.text_field]
        if spec.id_field:
            required_columns.append(spec.id_field)
        if spec.language_field:
            required_columns.append(spec.language_field)
        missing_columns = sorted(set(required_columns) - set(dataset.column_names))
        if missing_columns:
            raise AcquisitionError(
                f"{spec.name}: source is missing columns {missing_columns}; "
                f"available={dataset.column_names}"
            )
        dataset = dataset.select_columns(required_columns)
        dataset = dataset.shuffle(
            seed=item.shuffle_seed, buffer_size=spec.shuffle_buffer
        )

        for row in dataset:
            counters.scanned += 1
            if counters.scanned > item.max_scan_records:
                break
            if not isinstance(row, dict):
                counters.dropped_invalid += 1
                continue
            text = row.get(spec.text_field)
            if not isinstance(text, str) or not text.strip() or "\0" in text:
                counters.dropped_invalid += 1
                continue
            length = len(text)
            if length < spec.min_chars:
                counters.dropped_short += 1
                continue
            if length > spec.max_chars:
                counters.dropped_long += 1
                continue
            if spec.language_field and str(row.get(spec.language_field)) not in set(
                spec.accepted_language_values
            ):
                counters.dropped_language += 1
                continue

            row_id = upstream_id(row, spec, text)
            record = make_record(spec, row, text, row_id)
            if record["id"] in ids:
                counters.dropped_duplicate_id += 1
                continue
            ids.add(record["id"])
            is_eval = evaluation_partition(
                spec, row_id, item.evaluation_basis_points
            )
            if is_eval:
                if counters.written_evaluation >= item.evaluation_records:
                    counters.skipped_partition_full += 1
                    continue
                encoded_size = len(canonical_json(record)) + 1
                if training_writer.bytes + evaluation_writer.bytes + encoded_size > item.max_output_bytes:
                    raise AcquisitionError(
                        f"{spec.name}: output byte cap reached before quotas"
                    )
                evaluation_writer.write(record)
                counters.written_evaluation += 1
            else:
                if counters.written_training >= item.train_records:
                    counters.skipped_partition_full += 1
                    continue
                encoded_size = len(canonical_json(record)) + 1
                if training_writer.bytes + evaluation_writer.bytes + encoded_size > item.max_output_bytes:
                    raise AcquisitionError(
                        f"{spec.name}: output byte cap reached before quotas"
                    )
                training_writer.write(record)
                counters.written_training += 1

            if counters.scanned % 25_000 == 0:
                print(
                    f"{spec.name}: scanned={counters.scanned} "
                    f"train={counters.written_training}/{item.train_records} "
                    f"eval={counters.written_evaluation}/{item.evaluation_records}",
                    flush=True,
                )
            if (
                counters.written_training == item.train_records
                and counters.written_evaluation == item.evaluation_records
            ):
                break

        if counters.written_training != item.train_records or counters.written_evaluation != item.evaluation_records:
            raise AcquisitionError(
                f"{spec.name}: exhausted scan bound before quotas: {asdict(counters)}"
            )
        training = training_writer.close()
        evaluation = evaluation_writer.close()
        result = {
            "schema_version": SCHEMA_VERSION,
            "generator_version": GENERATOR_VERSION,
            "source_config_hash": expected_hash,
            "selection_algorithm": SELECTION_ALGORITHM,
            "datasets_version": datasets_version,
            "resolved_revision": resolved_revision,
            "source": asdict(spec),
            "effective": {
                "train_records": item.train_records,
                "evaluation_records": item.evaluation_records,
                "evaluation_basis_points": item.evaluation_basis_points,
                "shuffle_seed": item.shuffle_seed,
                "max_scan_records": item.max_scan_records,
                "max_output_bytes": item.max_output_bytes,
            },
            "counters": asdict(counters),
            "training": training,
            "evaluation": evaluation,
        }
        write_new_json(temp_dir / "source-result.json", result)
        os.replace(temp_dir, final_dir)
        print(
            f"{spec.name}: complete scanned={counters.scanned} "
            f"bytes={(training['bytes'] + evaluation['bytes']) / MIB:.1f} MiB",
            flush=True,
        )
        return result
    except BaseException:
        training_writer.abort()
        evaluation_writer.abort()
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def write_new_json(path: Path, value: Any) -> None:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def verify_file_entry(root: Path, entry: dict[str, Any]) -> None:
    filename = nonblank(entry.get("file"), "completed file name")
    if Path(filename).name != filename:
        raise AcquisitionError(f"unsafe completed-source file name {filename!r}")
    expected_hash = nonblank(entry.get("sha256"), "completed file hash")
    if not SHA256_RE.fullmatch(expected_hash):
        raise AcquisitionError(f"invalid completed-source hash {expected_hash!r}")
    expected_bytes = positive_int(entry.get("bytes"), "completed file bytes")
    positive_int(entry.get("records"), "completed file records")
    actual_hash, actual_bytes = hash_file(root / filename)
    if actual_hash != expected_hash or actual_bytes != expected_bytes:
        raise AcquisitionError(
            f"completed source file drift at {root / filename}: "
            f"expected {expected_hash}/{expected_bytes}, got {actual_hash}/{actual_bytes}"
        )


def verify_completed_source(source_dir: Path, expected_config_hash: str) -> dict[str, Any]:
    result_path = source_dir / "source-result.json"
    try:
        result = require_object(json.loads(result_path.read_bytes()), str(result_path))
    except (OSError, json.JSONDecodeError) as error:
        raise AcquisitionError(f"cannot verify completed source {source_dir}: {error}") from error
    if result.get("source_config_hash") != expected_config_hash:
        raise AcquisitionError(
            f"completed source {source_dir} belongs to a different configuration"
        )
    verify_file_entry(source_dir, require_object(result.get("training"), "training"))
    verify_file_entry(source_dir, require_object(result.get("evaluation"), "evaluation"))
    print(f"{source_dir.name}: verified completed source", flush=True)
    return result


def source_manifest_entry(index: int, spec: SourceSpec, result: dict[str, Any]) -> dict[str, Any]:
    directory = f"source-{index:02}-{spec.name}"
    return {
        "name": spec.name,
        "dataset": spec.dataset,
        "config": spec.config,
        "revision": spec.revision,
        "license": spec.license,
        "language": spec.language,
        "dataset_card": spec.dataset_card,
        "license_url": spec.license_url,
        "source_config_hash": result["source_config_hash"],
        "counters": result["counters"],
        "training": {
            **result["training"],
            "file": f"{directory}/{result['training']['file']}",
        },
        "evaluation": {
            **result["evaluation"],
            "file": f"{directory}/{result['evaluation']['file']}",
        },
    }


def verify_final_manifest(
    output_root: Path, expected_config_hash: str
) -> dict[str, Any]:
    path = output_root / "acquisition.json"
    result = require_object(json.loads(path.read_bytes()), str(path))
    if result.get("config_hash") != expected_config_hash:
        raise AcquisitionError(
            f"{path} belongs to a different configuration; outputs are immutable"
        )
    for source in result.get("sources", []):
        for split in ("training", "evaluation"):
            entry = require_object(source.get(split), f"{source.get('name')}.{split}")
            relative = Path(nonblank(entry.get("file"), "manifest file"))
            if relative.is_absolute() or ".." in relative.parts:
                raise AcquisitionError(f"unsafe acquisition path {relative}")
            actual_hash, actual_bytes = hash_file(output_root / relative)
            if actual_hash != entry.get("sha256") or actual_bytes != entry.get("bytes"):
                raise AcquisitionError(f"acquisition file drift: {relative}")
    return result


def acquire(args: argparse.Namespace) -> None:
    config, config_hash, _ = load_config(args.config)
    effective = tuple(
        effective_spec(source, args.smoke_records, config.shuffle_seed + index)
        for index, source in enumerate(config.sources)
    )
    disk = disk_preflight(
        args.out,
        config,
        effective,
        args.smoke_records,
        enforce=not args.dry_run,
    )
    print(
        f"configuration={config_hash} sources={len(config.sources)} "
        f"disk_ok={disk['ok']} free={disk['free_bytes'] / GIB:.1f} GiB "
        f"required={disk['required_free_bytes'] / GIB:.1f} GiB"
    )
    if args.dry_run:
        return

    if args.out.exists() and not args.out.is_dir():
        raise AcquisitionError(f"output path is not a directory: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "acquisition.json"
    if manifest_path.exists():
        manifest = verify_final_manifest(args.out, config_hash)
        print(
            f"verified complete acquisition {config.run_id}: "
            f"train={manifest['training_records']} eval={manifest['evaluation_records']}"
        )
        return

    lock_path = args.out / ".acquire.lock"
    try:
        lock_fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        owner = lock_path.read_text(encoding="utf-8", errors="replace").strip()
        raise AcquisitionError(
            f"acquisition lock already exists at {lock_path}: {owner}"
        ) from error
    with os.fdopen(lock_fd, "w", encoding="utf-8") as lock:
        lock.write(f"pid={os.getpid()} host={socket.gethostname()}\n")
        lock.flush()
        os.fsync(lock.fileno())

    try:
        def interrupted(signum: int, _frame: Any) -> None:
            raise AcquisitionError(f"interrupted by signal {signum}")

        signal.signal(signal.SIGINT, interrupted)
        signal.signal(signal.SIGTERM, interrupted)
        load_dataset, hf_api, datasets_version = import_hf_clients()
        token = os.environ.get("HF_TOKEN") or None
        results = [
            stream_source(
                item,
                index,
                args.out,
                load_dataset,
                hf_api,
                datasets_version,
                token,
            )
            for index, item in enumerate(effective)
        ]
        sources = [
            source_manifest_entry(index, item.source, result)
            for index, (item, result) in enumerate(zip(effective, results, strict=True))
        ]
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "generator_version": GENERATOR_VERSION,
            "run_id": config.run_id,
            "config_file": str(args.config),
            "config_hash": config_hash,
            "selection_algorithm": SELECTION_ALGORITHM,
            "datasets_version": datasets_version,
            "smoke_records": args.smoke_records,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "training_records": sum(
                source["training"]["records"] for source in sources
            ),
            "evaluation_records": sum(
                source["evaluation"]["records"] for source in sources
            ),
            "training_bytes": sum(source["training"]["bytes"] for source in sources),
            "evaluation_bytes": sum(
                source["evaluation"]["bytes"] for source in sources
            ),
            "sources": sources,
        }
        temporary_manifest = args.out / f".acquisition.json.tmp-{os.getpid()}"
        write_new_json(temporary_manifest, manifest)
        os.replace(temporary_manifest, manifest_path)
        print(
            f"acquisition complete: train={manifest['training_records']} "
            f"eval={manifest['evaluation_records']} -> {manifest_path}"
        )
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def self_test() -> None:
    source = SourceSpec(
        name="test",
        dataset="owner/dataset",
        config="default",
        split="train",
        revision="a" * 40,
        license="Apache-2.0",
        hub_license_tag="license:apache-2.0",
        language="en",
        text_field="text",
        id_field="id",
        language_field="language",
        accepted_language_values=("en",),
        train_records=10,
        evaluation_records=1,
        evaluation_basis_points=100,
        min_chars=3,
        max_chars=100,
        shuffle_buffer=10,
        max_scan_records=100,
        max_output_bytes=MIB,
        dataset_card="https://example.test/dataset",
        license_url="https://example.test/license",
    )
    row = {"id": "42", "text": "A small training document.", "language": "en"}
    one = make_record(source, row, row["text"], row["id"])
    two = make_record(source, row, row["text"], row["id"])
    assert one == two
    assert one["id"].startswith("test-") and len(one["id"]) == len("test-") + 64
    assert one["source_revision"] == "a" * 40
    assert evaluation_partition(source, "42", 10_000)
    assert not evaluation_partition(source, "42", 0)
    smoke = effective_spec(source, 2, 51_004)
    assert smoke.train_records == 2 and smoke.evaluation_records == 1
    assert smoke.max_output_bytes <= source.max_output_bytes
    assert canonical_json({"b": 1, "a": "é"}) == b'{"a":"\xc3\xa9","b":1}'
    print("fetch_foundation_hf self-test: PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream pinned Hugging Face datasets into FoundationRecord JSONL"
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--smoke-records",
        type=int,
        help="override each source with N train records and a tiny eval quota",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate configuration and report disk admission without network access",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return args
    if args.config is None or args.out is None:
        parser.error("--config and --out are required unless --self-test is used")
    if args.smoke_records is not None and args.smoke_records <= 0:
        parser.error("--smoke-records must be positive")
    return args


def main() -> int:
    args = parse_args()
    try:
        if args.self_test:
            self_test()
        else:
            acquire(args)
        return 0
    except (AcquisitionError, OSError, ValueError) as error:
        print(f"fetch_foundation_hf: FAIL: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
