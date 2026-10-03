#!/usr/bin/env python3
"""Pod entrypoint for the Lilith training program; never provisions, stops or deletes a pod.

Generalizes ``history/run_pod.py``: instead of a one-shot pilot it starts or resumes
``train_program.py`` for one resource window, detached (nohup + new session), with a pid file
and a unique marker, and offers the small control verbs the Mac supervisor drives over ssh.
Everything it writes lives under ``<root>/program/<run_id>/`` on the /workspace volume.

    run_program_pod.py start  --manifest M --approval A --main-sha SHA --max-wall-seconds N [--resume]
                              [--init-adapter DIR --init-adapter-sha256 H] [--onpolicy-wait-mode poll|exit] [-- <train_program args>]
    run_program_pod.py status --run-id ID                  one JSON line: pid/marker liveness, progress, gpu, handshake files
    run_program_pod.py stop   --run-id ID [--grace S]      SIGTERM (the trainer checkpoints and exits 75); waits for it
    run_program_pod.py onpolicy-ready --run-id ID --stage S --archive NAME --sha256 H --adapter-sha256 H
                                                           host handshake: validates the rsynced archive, writes onpolicy-ready.json

Liveness is decided by pid file + marker found in the process command line, never by ``pgrep -f``
(a pattern would match the probing ssh command; the 2026-09 incident cost 2.5 h of idle GPU).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Sequence

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / 'runpod'))

from checkpoint import ProgramError, read_json, sha256_file, write_json_atomic  # noqa: E402
import verify_payload  # noqa: E402

_RUN_ID = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{2,79}$')
_STAGE = re.compile(r'^(white|yellow|green|brown|black)$')
_SHA256 = re.compile(r'^[0-9a-f]{64}$')
READY_SCHEMA = 'lilith/program-onpolicy-ready@1'


def run_dir_for(root: Path, run_id: str) -> Path:
    if not _RUN_ID.match(run_id):
        raise ProgramError('lilith::program::run_id', f'run id {run_id!r} is not a safe identifier', 'use the manifest run_id')
    return Path(root) / 'program' / run_id


def process_matches(pid: int, marker: str, proc_root: Path = Path('/proc')) -> bool:
    """True when ``pid`` is alive and its command line carries this window's unique marker."""
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    cmdline = proc_root / str(pid) / 'cmdline'
    if cmdline.is_file():
        try:
            return marker.encode() in cmdline.read_bytes()
        except OSError:
            return False
    ps = subprocess.run(['ps', '-p', str(pid), '-o', 'command='], capture_output=True, text=True, timeout=10)
    return ps.returncode == 0 and marker in ps.stdout


def read_pidfile(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / 'program.pid'
    return read_json(path) if path.is_file() else None


def alive(run_dir: Path) -> bool:
    record = read_pidfile(run_dir)
    return bool(record) and process_matches(int(record['pid']), record['marker'])


def newest_write_age(run_dir: Path, now: float | None = None) -> float | None:
    """Seconds since the newest file write anywhere under the run directory (checkpoints included)."""
    newest = 0.0
    for current, _dirs, files in os.walk(run_dir):
        for name in files:
            try:
                newest = max(newest, os.stat(os.path.join(current, name)).st_mtime)
            except OSError:
                continue
    return None if not newest else (now if now is not None else time.time()) - newest


def gpu_utilization() -> int | None:
    if shutil.which('nvidia-smi') is None:
        return None
    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, timeout=20)
    try:
        return int(result.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


def stage_handshake(run_dir: Path) -> dict[str, dict[str, bool]]:
    out = {}
    stages = run_dir / 'stages'
    if stages.is_dir():
        for stage in sorted(p for p in stages.iterdir() if p.is_dir()):
            needed, ready = (stage / 'onpolicy-needed.json').is_file(), (stage / 'onpolicy-ready.json').is_file()
            if needed or ready:
                out[stage.name] = {'needed': needed, 'ready': ready}
    return out


def status(root: Path, run_id: str) -> dict[str, Any]:
    run_dir = run_dir_for(root, run_id)
    record = read_pidfile(run_dir)
    live = bool(record) and process_matches(int(record['pid']), record['marker'])
    marker = record['marker'] if record else None
    window_exit = read_json(run_dir / 'window-exit.json') if (run_dir / 'window-exit.json').is_file() else None
    if window_exit and marker and window_exit.get('marker') != marker:
        window_exit = None     # a previous window's verdict is not this window's
    handshake = stage_handshake(run_dir)
    return {'schema': 'lilith/program-pod-status@1', 'run_id': run_id, 'run_dir_exists': run_dir.is_dir(), 'pid': record['pid'] if record else None,
            'marker': marker, 'alive': live,
            'progress': read_json(run_dir / 'progress.json') if (run_dir / 'progress.json').is_file() else None,
            'window_exit': window_exit, 'program_complete': (run_dir / 'program-complete.json').is_file(),
            'failure': read_json(run_dir / 'failure.json') if (run_dir / 'failure.json').is_file() else None,
            'onpolicy': handshake,
            'onpolicy_pending': sorted(s for s, v in handshake.items() if v['needed'] and not v['ready']),
            'newest_write_age_seconds': newest_write_age(run_dir) if run_dir.is_dir() else None,
            'gpu_util': gpu_utilization(), 'disk_free_gb': round(shutil.disk_usage(root).free / 1e9, 1) if Path(root).is_dir() else None,
            'at': time.time()}


def window_number(run_dir: Path) -> int:
    path = run_dir / 'windows.jsonl'
    if not path.is_file():
        return 1
    return 1 + sum(1 for line in path.read_text().splitlines() if '"event": "start"' in line)


def start(args: argparse.Namespace, passthrough: Sequence[str]) -> dict[str, Any]:
    manifest = json.loads(Path(args.manifest).read_text())
    run_dir = run_dir_for(args.root, manifest['run_id'])
    if alive(run_dir):
        raise ProgramError('lilith::program::already_running', f'{run_dir.name} already has a live trainer',
                           'run `stop` first; two windows must never train the same run')
    payload = Path(args.payload_root) if args.payload_root else HERE.parents[3]
    provenance = verify_payload.verify(payload, payload / 'source-manifest.json', args.main_sha, False)
    if provenance['status'] != 'verified':
        raise ProgramError('lilith::program::payload', 'main-only payload provenance failed: ' + '; '.join(provenance['problems']),
                           'extract a payload built by make_payload.sh from origin/main and pass its main SHA')
    python = str(Path(args.root) / 'env/bin/python')
    bootstrap = subprocess.run(['bash', str(payload / 'scripts/lilith/runpod/bootstrap.sh'), '--check'],
                               capture_output=True, text=True, timeout=600,
                               env={**os.environ, 'LILITH_ROOT': str(args.root)})
    if bootstrap.returncode:
        detail = (bootstrap.stderr.strip().splitlines() or ['no stderr'])[-1]
        raise ProgramError('lilith::program::bootstrap', f'bootstrap --check failed: {detail}',
                           'fix the volume environment (bootstrap.sh) before spending GPU time')
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'logs').mkdir(exist_ok=True)
    window = window_number(run_dir)
    marker = f'{manifest["run_id"]}-w{window}-{secrets.token_hex(4)}'
    command = [python, str(payload / 'scripts/lilith/program/train_program.py'),
               '--manifest', str(args.manifest), '--parent', str(Path(args.root) / 'parent'), '--approval', str(args.approval),
               '--main-sha', args.main_sha, '--root', str(args.root), '--payload-root', str(payload),
               '--max-wall-seconds', str(args.max_wall_seconds), '--onpolicy-wait-mode', args.onpolicy_wait_mode,
               '--marker', marker]
    if args.resume:
        command.append('--resume')
    if args.init_adapter:
        command += ['--init-adapter', str(args.init_adapter), '--init-adapter-sha256', args.init_adapter_sha256]
    command += list(passthrough)
    if shutil.which('nohup'):
        command = ['nohup', *command]
    env = {**os.environ, 'LILITH_ROOT': str(args.root), 'HF_HUB_OFFLINE': '1', 'PYTHONNOUSERSITE': '1',
           'PATH': str(Path(args.root) / 'env/bin') + os.pathsep + os.environ.get('PATH', ''), 'LILITH_PROGRAM_MARKER': marker}
    log = run_dir / 'logs' / f'train-{marker}.log'
    with log.open('ab') as stream:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, env=env,
                                   start_new_session=True)
    write_json_atomic(run_dir / 'program.pid', {'pid': process.pid, 'marker': marker, 'window': window, 'started_at': time.time(),
                                                'log': str(log), 'resume': bool(args.resume), 'main_sha': args.main_sha})
    time.sleep(args.startup_check_seconds)
    exited = process.poll()
    return {'schema': 'lilith/program-pod-start@1', 'run_id': manifest['run_id'], 'pid': process.pid, 'marker': marker, 'window': window,
            'log': str(log), 'exited_immediately': exited is not None, 'exit_code': exited}


def stop(root: Path, run_id: str, grace: float, kill_after_grace: bool) -> dict[str, Any]:
    run_dir = run_dir_for(root, run_id)
    record = read_pidfile(run_dir)
    if not record or not process_matches(int(record['pid']), record['marker']):
        return {'run_id': run_id, 'signalled': False, 'alive': False}
    os.kill(int(record['pid']), signal.SIGTERM)       # the trainer checkpoints at the next step boundary and exits 75
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline and process_matches(int(record['pid']), record['marker']):
        time.sleep(2)
    still = process_matches(int(record['pid']), record['marker'])
    killed = False
    if still and kill_after_grace:
        os.kill(int(record['pid']), signal.SIGKILL)
        killed = True
        time.sleep(2)
        still = process_matches(int(record['pid']), record['marker'])
    return {'run_id': run_id, 'signalled': True, 'alive': still, 'killed': killed}


def onpolicy_ready(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = run_dir_for(args.root, args.run_id)
    if not _STAGE.match(args.stage) or not _SHA256.match(args.sha256) or not _SHA256.match(args.adapter_sha256):
        raise ProgramError('lilith::program::ready_args', 'stage or sha256 malformed', 'stage is a belt name; hashes are 64 hex chars')
    stage_dir = run_dir / 'stages' / args.stage
    needed_path = stage_dir / 'onpolicy-needed.json'
    if not needed_path.is_file():
        raise ProgramError('lilith::program::ready_without_need', f'{args.stage} has no onpolicy-needed.json',
                           'the trainer has not asked for an on-policy round for this stage')
    needed = read_json(needed_path)
    expected_name = needed['archive_name']
    if args.archive != expected_name:
        raise ProgramError('lilith::program::ready_name', f'archive must be named {expected_name}', 'the trainer fixes the name')
    if args.adapter_sha256 != needed['adapter_sha256']:
        raise ProgramError('lilith::program::ready_adapter', 'the archive was generated by a different adapter than the saved stage adapter',
                           'serve stages/<stage>/adapter and regenerate')
    archive = stage_dir / expected_name
    if not archive.is_file() or sha256_file(archive) != args.sha256:
        raise ProgramError('lilith::program::ready_hash', f'{archive} is missing or its sha256 differs', 'resend the archive')
    record = {'schema': READY_SCHEMA, 'run_id': args.run_id, 'stage': args.stage, 'archive': expected_name, 'sha256': args.sha256,
              'adapter_sha256': args.adapter_sha256, 'written_at': time.time()}
    write_json_atomic(stage_dir / 'onpolicy-ready.json', record)      # written last: the trainer polls for it
    return record


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, default=Path('/workspace/lilith'))
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('start')
    s.add_argument('--manifest', type=Path, required=True)
    s.add_argument('--approval', type=Path, required=True)
    s.add_argument('--main-sha', required=True)
    s.add_argument('--max-wall-seconds', type=int, required=True)
    s.add_argument('--payload-root', type=Path, default=None)
    s.add_argument('--resume', action='store_true')
    s.add_argument('--init-adapter', type=Path, default=None)
    s.add_argument('--init-adapter-sha256', default=None)
    s.add_argument('--onpolicy-wait-mode', choices=('poll', 'exit'), default='poll')
    s.add_argument('--startup-check-seconds', type=float, default=5.0)
    t = sub.add_parser('status')
    t.add_argument('--run-id', required=True)
    k = sub.add_parser('stop')
    k.add_argument('--run-id', required=True)
    k.add_argument('--grace', type=float, default=600.0)
    k.add_argument('--kill-after-grace', action='store_true')
    r = sub.add_parser('onpolicy-ready')
    r.add_argument('--run-id', required=True)
    r.add_argument('--stage', required=True)
    r.add_argument('--archive', required=True)
    r.add_argument('--sha256', required=True)
    r.add_argument('--adapter-sha256', required=True)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    passthrough: list[str] = []
    if '--' in argv:
        split = argv.index('--')
        argv, passthrough = argv[:split], argv[split + 1:]
    args = build_parser().parse_args(argv)
    os.umask(0o077)
    try:
        if args.command == 'start':
            if bool(args.init_adapter) != bool(args.init_adapter_sha256):
                raise ProgramError('lilith::program::init_adapter_pin', '--init-adapter and its sha256 go together', 'pass both or neither')
            result = start(args, passthrough)
        elif args.command == 'status':
            result = status(args.root, args.run_id)
        elif args.command == 'stop':
            result = stop(args.root, args.run_id, args.grace, args.kill_after_grace)
        else:
            result = onpolicy_ready(args)
    except ProgramError as error:
        print(f'run_program_pod: error{error}', file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 1 if args.command == 'start' and result.get('exited_immediately') else 0


if __name__ == '__main__':
    raise SystemExit(main())
