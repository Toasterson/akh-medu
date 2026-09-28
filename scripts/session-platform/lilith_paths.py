#!/usr/bin/env python3
"""Operator-owned filesystem locations for the Lilith training stack.

Resolution order per key: explicit JSON paths file, then environment variable,
then the historical hephaistos default. The hephaistos defaults exist only as
fallbacks so that the frozen September runs remain reproducible on their
source machine; a Runpod pod supplies ``/workspace/lilith/paths.json`` (or the
``LILITH_*`` environment) instead.

Resolving paths never rewrites a frozen configuration: callers record the
resolved values and their provenance in a *new* config next to the run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Mapping

SCHEMA = 'lilith/paths@1'
CONFIG_ENV = 'LILITH_PATHS_CONFIG'
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

# key -> environment variable
ENVIRONMENT = {
    'parent_dir': 'LILITH_PARENT_DIR',
    'transcript_dataset': 'LILITH_TRANSCRIPT_DATASET',
    'templates_dir': 'LILITH_TEMPLATES_DIR',
    'venv_python': 'LILITH_VENV_PYTHON',
    'runtime_bin_dir': 'LILITH_RUNTIME_BIN_DIR',
    'reviewer_runtime_binary': 'LILITH_REVIEWER_RUNTIME_BINARY',
    'private_root': 'LILITH_PRIVATE_ROOT',
    'output_root': 'LILITH_OUTPUT_ROOT',
    'cache_offloading': 'LILITH_CACHE_OFFLOADING',
}
PATH_KEYS = tuple(key for key in ENVIRONMENT if key != 'cache_offloading')


class LilithPathsError(ValueError):
    """A path configuration problem, with a stable code and an actionable hint."""

    def __init__(self, code: str, message: str, help: str):
        super().__init__(f'[{code}] {message} (help: {help})')
        self.code = code
        self.help = help


def hephaistos_defaults(home: Path) -> dict[str, Any]:
    """The locations the September 2026 runs were frozen against."""
    share = home / '.local/share/akh-medu'
    return {
        'parent_dir': share / 'lilith-training/parent-5926caa4',
        'transcript_dataset': share / 'lilith-training/transcripts-20260922/sft-subset-v1.jsonl',
        'templates_dir': share / 'lilith-dev-vulkan-20260921',
        'venv_python': share / 'lilith-training/venv310/bin/python',
        'runtime_bin_dir': REPOSITORY_ROOT / 'target/release',
        'reviewer_runtime_binary': REPOSITORY_ROOT / 'target/release/akhrtd',
        'private_root': share,
        'output_root': REPOSITORY_ROOT / 'training/lilith',
        # ROCm/24 GB default. H100 configs set this explicitly to False.
        'cache_offloading': True,
    }


def parse_bool(value: Any, *, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ('1', 'true', 'yes', 'on'):
        return True
    if isinstance(value, str) and value.strip().lower() in ('0', 'false', 'no', 'off'):
        return False
    raise LilithPathsError('lilith::paths::not_boolean', f'{key} is not a boolean: {value!r}',
                           'use true/false (JSON) or 1/0, true/false in the environment')


@dataclass(frozen=True)
class LilithPaths:
    parent_dir: Path
    transcript_dataset: Path
    templates_dir: Path
    venv_python: Path
    runtime_bin_dir: Path
    reviewer_runtime_binary: Path
    private_root: Path
    output_root: Path
    cache_offloading: bool
    provenance: Mapping[str, str] = field(default_factory=dict)

    def runtime_binary(self, name: str) -> Path:
        return self.runtime_bin_dir / name

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {'schema': SCHEMA}
        for key in PATH_KEYS:
            record[key] = str(getattr(self, key))
        record['cache_offloading'] = self.cache_offloading
        record['provenance'] = dict(self.provenance)
        return record


def load_config_file(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text())
    except FileNotFoundError as error:
        raise LilithPathsError('lilith::paths::config_missing', f'paths config {path} does not exist',
                               f'create it or unset {CONFIG_ENV}') from error
    except json.JSONDecodeError as error:
        raise LilithPathsError('lilith::paths::config_invalid', f'paths config {path} is not JSON: {error}',
                               'fix the JSON syntax') from error
    if not isinstance(data, dict):
        raise LilithPathsError('lilith::paths::config_invalid', 'paths config must be a JSON object',
                               'write {"parent_dir": "/workspace/lilith/parent", ...}')
    schema = data.pop('schema', SCHEMA)
    if schema != SCHEMA:
        raise LilithPathsError('lilith::paths::schema', f'unsupported paths schema {schema!r}',
                               f'use "{SCHEMA}"')
    unknown = sorted(set(data) - set(ENVIRONMENT))
    if unknown:
        raise LilithPathsError('lilith::paths::unknown_key', f'unknown paths keys: {unknown}',
                               f'allowed keys: {sorted(ENVIRONMENT)}')
    return data


def resolve(config_path: Path | str | None = None, *, environ: Mapping[str, str] | None = None,
            home: Path | None = None) -> LilithPaths:
    """Resolve every location; relative paths are refused rather than guessed."""
    environ = os.environ if environ is None else environ
    home = Path.home() if home is None else Path(home)
    if config_path is None and environ.get(CONFIG_ENV):
        config_path = environ[CONFIG_ENV]
    configured = load_config_file(Path(config_path)) if config_path is not None else {}
    defaults = hephaistos_defaults(home)
    values: dict[str, Any] = {}
    provenance: dict[str, str] = {}
    for key, variable in ENVIRONMENT.items():
        if key in configured:
            value, source = configured[key], 'config:' + str(config_path)
        elif environ.get(variable):
            value, source = environ[variable], 'env:' + variable
        else:
            value, source = defaults[key], 'default:hephaistos'
        if key == 'cache_offloading':
            values[key] = parse_bool(value, key=key)
        else:
            if not isinstance(value, (str, Path)) or not str(value):
                raise LilithPathsError('lilith::paths::not_a_path', f'{key} must be a path string',
                                       f'set {key} in the paths config or {variable}')
            path = Path(value).expanduser()
            if not path.is_absolute():
                raise LilithPathsError('lilith::paths::relative', f'{key} is relative: {value}',
                                       'operator-owned locations must be absolute')
            values[key] = path
        provenance[key] = source
    return LilithPaths(provenance=provenance, **values)
