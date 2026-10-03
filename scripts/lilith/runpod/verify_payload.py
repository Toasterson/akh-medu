#!/usr/bin/env python3
"""Source provenance for the minimal Lilith pod payload (stdlib only).

The pod never receives the whole repository (Runpod rule). ``make_payload.sh``
builds a tarball of exactly ``PAYLOAD_FILES`` from a clean origin/main commit and
a ``source-manifest.json`` holding that commit and each file's sha256. On the pod,
``run_smoke.sh`` calls ``verify`` with the supervisor-supplied
``LILITH_EXPECTED_MAIN_SHA``; nothing relies on a local git ref.

    verify_payload.py list
    verify_payload.py build  --root DIR --main-sha SHA --output FILE
    verify_payload.py verify --root DIR --manifest FILE --expected-sha SHA --output FILE [--allow-unmerged]

``verify`` fails (exit 1) when the SHA is empty or malformed, the manifest is
missing or unreadable, its commit differs, its file set differs from
``PAYLOAD_FILES``, or any file is missing or differs. ``--allow-unmerged`` turns
those failures into a recorded ``unverified_allowed`` status (exit 0).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

SCHEMA = 'lilith/source-manifest@1'
PAYLOAD_FILES = (
    'scripts/lilith/history/train.py',
    'scripts/lilith/history/run_pod.py',
    'scripts/lilith/history/supervise.py',
    # Guided five-stage training program (docs/ai/plans/2026-10-03-lilith-train-now-reconciliation.md):
    # trainer + checkpoint protocol + pod entrypoint + candidate serving; the Mac host guard rides along
    # so its code SHA is on the same main commit as the trainer it supervises.
    'scripts/lilith/program/train_program.py',
    'scripts/lilith/program/checkpoint.py',
    'scripts/lilith/program/run_program_pod.py',
    'scripts/lilith/program/supervise_program.py',
    'scripts/lilith/program/serve_candidate.sh',
    'scripts/lilith/runpod/bootstrap.sh',
    'scripts/lilith/runpod/run_smoke.sh',
    'scripts/lilith/runpod/verify_payload.py',
    'scripts/lilith/runpod/lilith_lora_trainer.py',
    'scripts/lilith/runpod/reference_microrun.py',
    'scripts/lilith/runpod/requirements-cuda.txt',
    # CUDA 12.8-driver lane (LILITH_CUDA_LANE=cu128): hash-locked cu129 wheels + the file that regenerates them.
    'scripts/lilith/runpod/requirements-cuda128.in',
    'scripts/lilith/runpod/requirements-cuda128.txt',
    'scripts/session-platform/lilith_adapter_mechanics.py',
    'scripts/session-platform/lilith_training_engine.py',
    'scripts/session-platform/lilith_training_cache_control.py',
    'scripts/session-platform/lilith_cache_path_diagnostic.py',
    'scripts/session-platform/lilith_numerics.py',
    'scripts/session-platform/lilith_rl.py',
    'training/lilith/transcripts-20260922/sft-subset-v1-manifest.json',
    # Lilith C2 serving (Anima WI 01a0e270-314e): the serve entrypoint and the C4b
    # certification closure serve_adapter.py imports to accept an active adapter.
    'scripts/lilith/serve/serve_pod.sh',
    'scripts/lilith/serve/serve_adapter.py',
    'scripts/lilith/serve/log_redact.py',
    'scripts/lilith/serve/serve_tools.py',
    'scripts/lilith/exam/__init__.py',
    'scripts/lilith/exam/certify.py',
    'scripts/lilith/exam/errors.py',
    'scripts/lilith/exam/manifest.py',
    'scripts/lilith/dojo/__init__.py',
    'scripts/lilith/dojo/contracts.py',
    'scripts/lilith/dojo/episode.py',
    'scripts/lilith/dojo/errors.py',
    'scripts/lilith/dojo/exam-floors-v1.json',
    'scripts/lilith/dojo/families.py',
    'scripts/lilith/dojo/floors.py',
    'scripts/lilith/dojo/hindsight.py',
    'scripts/lilith/dojo/language.py',
    'scripts/lilith/dojo/replay_anima.py',
    'scripts/lilith/dojo/replay_memory.py',
    'scripts/lilith/dojo/replay_server.py',
    'scripts/lilith/dojo/replay_match.py',
    'scripts/lilith/dojo/reward.py',
    'scripts/lilith/dojo/snapshot.py',
)
_SHA = re.compile(r'^[0-9a-f]{40}$')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def build(root: Path, main_sha: str) -> dict[str, Any]:
    if not _SHA.match(main_sha):
        raise ValueError(f'main sha {main_sha!r} is not a 40-hex commit id')
    files = {}
    for name in PAYLOAD_FILES:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f'payload file {name} is missing under {root}')
        files[name] = sha256_file(path)
    return {'schema': SCHEMA, 'main_sha': main_sha, 'files': files,
            'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat()}


def verify(root: Path, manifest_path: Path, expected_sha: str, allow_unmerged: bool) -> dict[str, Any]:
    problems: list[str] = []
    manifest: dict[str, Any] = {}
    if not _SHA.match(expected_sha or ''):
        problems.append(f'LILITH_EXPECTED_MAIN_SHA {expected_sha!r} is empty or not a 40-hex commit id')
    if not manifest_path.is_file():
        problems.append(f'source manifest {manifest_path} is missing')
    else:
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            problems.append(f'source manifest unreadable: {error}')
    if manifest:
        if manifest.get('schema') != SCHEMA:
            problems.append(f'manifest schema {manifest.get("schema")!r} != {SCHEMA}')
        if manifest.get('main_sha') != expected_sha:
            problems.append(f'manifest commit {manifest.get("main_sha")!r} != expected {expected_sha!r}')
        files = manifest.get('files') if isinstance(manifest.get('files'), dict) else {}
        if set(files) != set(PAYLOAD_FILES):
            problems.append('manifest file set differs from PAYLOAD_FILES: missing '
                            f'{sorted(set(PAYLOAD_FILES) - set(files))}, extra {sorted(set(files) - set(PAYLOAD_FILES))}')
        for name in PAYLOAD_FILES:
            path = root / name
            if not path.is_file():
                problems.append(f'{name} is missing from the payload')
            elif files.get(name) != sha256_file(path):
                problems.append(f'{name} sha256 differs from the manifest')
    status = 'verified' if not problems else ('unverified_allowed' if allow_unmerged else 'failed')
    return {'schema': 'lilith/payload-provenance@1', 'status': status, 'expected_main_sha': expected_sha,
            'manifest': str(manifest_path),
            'manifest_sha256': sha256_file(manifest_path) if manifest_path.is_file() else None,
            'allow_unmerged': allow_unmerged, 'problems': problems,
            'files': {name: sha256_file(root / name) for name in PAYLOAD_FILES if (root / name).is_file()}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('list')
    build_parser = commands.add_parser('build')
    build_parser.add_argument('--root', type=Path, required=True)
    build_parser.add_argument('--main-sha', required=True)
    build_parser.add_argument('--output', type=Path, required=True)
    verify_parser = commands.add_parser('verify')
    verify_parser.add_argument('--root', type=Path, required=True)
    verify_parser.add_argument('--manifest', type=Path, required=True)
    verify_parser.add_argument('--expected-sha', default='')
    verify_parser.add_argument('--output', type=Path, required=True)
    verify_parser.add_argument('--allow-unmerged', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'list':
        print('\n'.join(PAYLOAD_FILES))
        return 0
    if args.command == 'build':
        manifest = build(args.root, args.main_sha)
        with args.output.open('x') as stream:
            stream.write(json.dumps(manifest, indent=2) + '\n')
        return 0
    record = verify(args.root, args.manifest, args.expected_sha, args.allow_unmerged)
    with args.output.open('x') as stream:
        stream.write(json.dumps(record, indent=2) + '\n')
    for problem in record['problems']:
        print(f'provenance: {problem}', file=sys.stderr)
    print(f'provenance: {record["status"]}', file=sys.stderr)
    return 1 if record['status'] == 'failed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
