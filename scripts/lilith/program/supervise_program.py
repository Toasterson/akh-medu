#!/usr/bin/env python3
"""Mac host guard for one resource window of the Lilith training program.

Generalizes ``history/supervise.py``: it never creates or deletes a pod. It optionally starts the
approved pod (``--resume-window``), starts or resumes the trainer on it, polls it every 5 minutes
(pid + marker alive, newest-write age, GPU utilisation, handshake files), runs the on-policy
hand-off, and at the window deadline checkpoints the trainer (SIGTERM), stops the pod through the
same OpenBao-delivered ``agent-tools runpod -- runpodctl`` path ``history/supervise.py`` uses, verifies
EXITED and records the spend window. The pod is stopped on every exit path, including exceptions.

    supervise_program.py --pod-id ID --ssh root@HOST --port N --identity KEY --window-seconds S \\
        --approval approval.json --dest DIR [--resume-window] [--hourly-usd 2.69] [--poll-seconds 300]
        [--onpolicy-wait-mode poll|exit] [--init-adapter DIR --init-adapter-sha256 H]

The on-policy rounds are run by the supervising session, not by this script. On
``onpolicy-needed.json`` the script serves the stage adapter on the pod (serve_candidate.sh) and
writes ``<dest>/onpolicy/<stage>/serve-ready.json`` (proxy URL, model, key file path; Python
clients must send ``User-Agent: curl/8.7.1``). The session runs the dojo rounds against it and
writes ``<dest>/onpolicy/<stage>/onpolicy-<stage>.tokens.jsonl`` and then
``archive-ready.json`` {"archive": "<file name>", "sha256": "<hex>"}. The script rsyncs the archive,
stops vLLM and only then writes ``onpolicy-ready.json`` on the pod, so the trainer never shares the
GPU with vLLM after the hand-off. In ``exit`` wait mode it also restarts the trainer with ``--resume``.

Approval (same file the trainer checks) must carry: decision "go", run_id, pod_id, main_sha,
manifest_sha256, stop_pod_at_completion_or_deadline true and max_window_seconds >= --window-seconds.

Window arithmetic (D = window start + window-seconds): the trainer gets ``--max-wall-seconds`` =
D - now - 300 and stops itself 300 s before that; the host sends SIGTERM at D - 480 if it is still
alive, then stops the pod. Exit codes: 0 program complete, 75 window ended (resume me), 1 failure.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import json
from pathlib import Path
import re
import secrets
import shlex
import subprocess
import sys
import time
from typing import Any, Callable, Sequence

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / 'history'))
sys.path.insert(0, str(HERE.parent))

from checkpoint import DEFAULT_CUDA_LANE, CUDA_LANE_ENV, ProgramError, cuda_lane  # noqa: E402

AGENT_TOOLS = Path.home() / '.local/bin/agent-tools'
TRAINER_MARGIN_SECONDS = 300
HOST_TERM_LEAD_SECONDS = 480
MIN_TRAINER_WALL_SECONDS = 900
EXIT_COMPLETE, EXIT_FAILURE, EXIT_WINDOW = 0, 1, 75
USER_AGENT = 'curl/8.7.1'
_SHA256 = re.compile(r'^[0-9a-f]{64}$')
_RUN_ID = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{2,79}$')
SMALL_ARTIFACTS = ('program-complete.json', 'candidate-manifest.json', 'budget.json', 'windows.jsonl', 'window-exit.json',
                   'failure.json', 'progress.json', 'identity.json', 'steps.jsonl', 'events.jsonl')


class HostError(RuntimeError):
    def __init__(self, code: str, message: str, help: str):
        super().__init__(f'[{code}] {message} (help: {help})')
        self.code = code
        self.help = help


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def utc(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).isoformat()


# ----------------------------------------------------------------------- approval

@dataclasses.dataclass(frozen=True)
class HostApproval:
    run_id: str
    pod_id: str
    main_sha: str
    manifest_sha256: str
    max_window_seconds: int


def load_host_approval(path: Path, pod_id: str, window_seconds: int) -> HostApproval:
    consent = json.loads(Path(path).read_text())
    problems = []
    if consent.get('decision') != 'go':
        problems.append('decision is not "go"')
    if consent.get('pod_id') != pod_id:
        problems.append('go does not bind this pod')
    if consent.get('stop_pod_at_completion_or_deadline') is not True:
        problems.append('go does not authorize the bounded pod stop')
    if not _RUN_ID.match(str(consent.get('run_id', ''))):
        problems.append('run_id missing or unsafe')
    if not re.fullmatch(r'[0-9a-f]{40}', str(consent.get('main_sha', ''))):
        problems.append('main_sha missing')
    if not _SHA256.match(str(consent.get('manifest_sha256', ''))):
        problems.append('manifest_sha256 missing')
    cap = consent.get('max_window_seconds')
    if not isinstance(cap, int) or isinstance(cap, bool) or cap < window_seconds:
        problems.append(f'max_window_seconds {cap!r} does not cover --window-seconds {window_seconds}')
    if problems:
        raise HostError('lilith::supervise::approval', '; '.join(problems), 'the operator go must bind this pod, run, main SHA and window length')
    return HostApproval(consent['run_id'], pod_id, consent['main_sha'], consent['manifest_sha256'], cap)


# ------------------------------------------------------------------------- decide

@dataclasses.dataclass(frozen=True)
class Decision:
    action: str          # continue | complete | failed | window_ended | handshake | crashed | stalled
    stage: str | None = None
    detail: str = ''


def is_stall_probe(probe: dict[str, Any], stall_seconds: float) -> bool:
    """Alive, nothing written for a whole stall interval, and the GPU idle (or unreadable)."""
    age, gpu = probe.get('newest_write_age_seconds'), probe.get('gpu_util')
    return bool(probe.get('alive')) and age is not None and age > stall_seconds and (gpu is None or gpu <= 5)


def decide(probe: dict[str, Any], *, stall_seconds: float, stalled_strikes: int) -> Decision:
    """Pure reading of one ``run_program_pod.py status`` result; ``stalled_strikes`` counts prior stalled probes."""
    if probe.get('program_complete'):
        return Decision('complete')
    pending = probe.get('onpolicy_pending') or []
    if pending:
        return Decision('handshake', pending[0])
    if probe.get('alive'):
        if is_stall_probe(probe, stall_seconds):
            detail = f'no write for {int(probe["newest_write_age_seconds"])} s, gpu {probe.get("gpu_util")}'
            return Decision('stalled' if stalled_strikes >= 1 else 'continue', detail=detail)
        return Decision('continue')
    verdict = probe.get('window_exit')
    if verdict:
        code = verdict.get('exit_code')
        if code == 75:
            return Decision('window_ended', detail=str(verdict.get('reason')))
        return Decision('failed', detail=str(verdict.get('reason') or (probe.get('failure') or {}).get('message') or code))
    return Decision('crashed', detail='trainer gone without a window verdict (killed or OOM); resume uses the last checkpoint')


# ------------------------------------------------------------------------- ssh

class Ssh:
    def __init__(self, target: str, port: int, identity: Path, known_hosts: Path,
                 runner: Callable[..., subprocess.CompletedProcess] = subprocess.run):
        self.target, self.port, self.runner = target, port, runner
        self.opts = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'StrictHostKeyChecking=accept-new',
                     '-o', f'UserKnownHostsFile={known_hosts}']
        # `--identity agent`: authenticate with the running ssh-agent (the pod's registered key may
        # exist only there); otherwise pin the given key file.
        if str(identity) != 'agent':
            self.opts += ['-i', str(identity), '-o', 'IdentitiesOnly=yes']

    def run(self, command: str, *, timeout: float, input: str | None = None) -> subprocess.CompletedProcess:
        return self.runner(['ssh', *self.opts, '-p', str(self.port), self.target, command], capture_output=True, text=True,
                           timeout=timeout, input=input)

    def json(self, command: str, *, timeout: float = 60) -> dict[str, Any]:
        result = self.run(command, timeout=timeout)
        if result.returncode:
            raise HostError('lilith::supervise::remote', f'remote command failed ({result.returncode}): {result.stderr.strip()[-300:]}',
                            'the pod may be unreachable or the payload missing')
        text = result.stdout.strip()
        try:                       # whole-output JSON (pretty-printed files such as onpolicy-needed.json)
            return json.loads(text)
        except json.JSONDecodeError:
            try:                   # a status line after other output: the last line is the JSON
                return json.loads(text.splitlines()[-1])
            except (json.JSONDecodeError, IndexError) as error:
                raise HostError('lilith::supervise::remote', f'remote output is not JSON: {text[-200:]!r}',
                                'the pod printed something other than the expected JSON document') from error

    def put(self, local: Path, remote: str, *, timeout: float = 900) -> None:
        # Plain flags only: macOS ships an old rsync/openrsync; scp is the fallback.
        rsync = self.runner(['rsync', '-a', '--partial', '-e', 'ssh ' + shlex.join([*self.opts, '-p', str(self.port)]),
                             str(local), f'{self.target}:{remote}'], capture_output=True, text=True, timeout=timeout)
        if rsync.returncode:
            scp = self.runner(['scp', *self.opts, '-P', str(self.port), str(local), f'{self.target}:{remote}'],
                              capture_output=True, text=True, timeout=timeout)
            if scp.returncode:
                raise HostError('lilith::supervise::transfer', f'rsync and scp both failed: {scp.stderr.strip()[-200:]}', 'check disk space on the pod')

    def get(self, remote: str, local: Path, *, timeout: float = 300) -> bool:
        result = self.runner(['scp', *self.opts, '-P', str(self.port), f'{self.target}:{remote}', str(local)],
                             capture_output=True, text=True, timeout=timeout)
        return result.returncode == 0


# ---------------------------------------------------------------------- pod control

def _agent_tools(args: Sequence[str], runner: Callable[..., subprocess.CompletedProcess], timeout: float) -> subprocess.CompletedProcess:
    return runner([str(AGENT_TOOLS), 'runpod', '--', 'runpodctl', *args], capture_output=True, text=True, timeout=timeout)


def default_start_pod(pod: str, runner: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> dict[str, Any]:
    result = _agent_tools(['pod', 'start', pod], runner, 90)
    if result.returncode:
        raise HostError('lilith::supervise::pod_start', f'pod start failed: {result.stderr.strip()[-200:]}',
                        'EU-FR-1 H100 capacity may be exhausted; nothing is billing if the pod did not start')
    return {'started': True}


def default_pod_get(pod: str, runner: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> dict[str, Any]:
    result = _agent_tools(['pod', 'get', pod], runner, 45)
    return json.loads(result.stdout) if result.returncode == 0 else {}


def default_stop_pod(pod: str) -> dict[str, Any]:
    from supervise import stop_pod  # noqa: PLC0415  (history/supervise.py: stop + verify EXITED, reused as is)
    return stop_pod(pod)


# ------------------------------------------------------------------- the supervisor

@dataclasses.dataclass(frozen=True)
class HostArgs:
    pod_id: str
    ssh: str
    port: int
    identity: Path
    window_seconds: int
    approval: Path
    dest: Path
    resume_window: bool = False
    resume_trainer: bool = False   # resume the trainer on an already running pod (no pod start)
    hourly_usd: float | None = None
    poll_seconds: float = 300.0
    stall_seconds: float = 1800.0
    onpolicy_wait_mode: str = 'poll'
    serve_env: tuple[str, ...] = ()   # KEY=VALUE pairs prefixed to serve_candidate.sh (e.g. LILITH_SERVE_MAX_MODEL_LEN=65536)
    init_adapter: str | None = None
    init_adapter_sha256: str | None = None
    root: str = '/workspace/lilith'
    pod_payload: str | None = None
    pod_data_dir: str | None = None
    fetch_candidate: bool = False
    endpoint_from_pod: bool = False
    driver_poll_seconds: float = 10.0
    ssh_ready_seconds: float = 900.0


class Supervisor:
    def __init__(self, args: HostArgs, approval: HostApproval, *, ssh: Ssh | None = None,
                 clock: Callable[[], float] = time.time, sleep: Callable[[float], Any] = time.sleep,
                 start_pod: Callable[[str], dict[str, Any]] = default_start_pod,
                 stop_pod: Callable[[str], dict[str, Any]] = default_stop_pod,
                 pod_get: Callable[[str], dict[str, Any]] = default_pod_get):
        self.args, self.approval = args, approval
        self.dest = Path(args.dest)
        self.ssh = ssh or Ssh(args.ssh, args.port, args.identity, self.dest / 'known_hosts')
        self.clock, self.sleep = clock, sleep
        self.start_pod, self.stop_pod, self.pod_get = start_pod, stop_pod, pod_get
        self.run_id = approval.run_id
        short = approval.main_sha[:12]
        self.payload = args.pod_payload or f'{args.root}/payload/main-{short}'
        self.data_dir = args.pod_data_dir or f'{args.root}/program-data/{self.run_id}'
        self.run_dir = f'{args.root}/program/{self.run_id}'
        try:
            self.lane = cuda_lane()
        except ProgramError as error:
            raise HostError(error.code, str(error), error.help) from error
        # The default lane adds nothing to the pod commands (byte-for-byte as before); cu128 is forwarded
        # so bootstrap --check, the trainer and vLLM all pick <root>/env-cu128 on the pod.
        lane_prefix = '' if self.lane == DEFAULT_CUDA_LANE else f'{CUDA_LANE_ENV}={self.lane} '
        self.pod_script = f'{lane_prefix}python3 {shlex.quote(self.payload)}/scripts/lilith/program/run_program_pod.py --root {shlex.quote(args.root)}'
        for pair in args.serve_env:
            if not re.fullmatch(r'LILITH_SERVE_[A-Z0-9_]+=[A-Za-z0-9._-]+', pair):
                raise HostError('lilith::supervise::serve_env', f'--serve-env {pair!r} is not a LILITH_SERVE_* KEY=VALUE pair',
                                'only vLLM serve settings may be injected, as plain values')
        serve_prefix = ''.join(f'{pair} ' for pair in args.serve_env)
        self.serve_script = f'{lane_prefix}{serve_prefix}bash {shlex.quote(self.payload)}/scripts/lilith/program/serve_candidate.sh'
        self.t0 = clock()
        self.deadline = self.t0 + args.window_seconds
        self.stall_strikes = 0
        self.report: dict[str, Any] = {'schema': 'lilith/program-supervisor@1', 'run_id': self.run_id, 'pod_id': args.pod_id,
                                       'main_sha': approval.main_sha, 'window_start_utc': utc(self.t0), 'deadline_utc': utc(self.deadline),
                                       'resume_window': args.resume_window, 'polls': 0, 'handshakes': []}
        self.vllm_running = False

    # ---------------------------------------------------------------- pod I/O

    def status(self) -> dict[str, Any]:
        return self.ssh.json(f'{self.pod_script} status --run-id {shlex.quote(self.run_id)}')

    def wait_ssh(self) -> None:
        limit = self.clock() + self.args.ssh_ready_seconds
        while True:
            try:
                if self.ssh.run('true', timeout=20).returncode == 0:
                    return
            except subprocess.TimeoutExpired:
                pass
            if self.clock() >= limit:
                raise HostError('lilith::supervise::ssh', f'pod not reachable over ssh within {int(self.args.ssh_ready_seconds)} s',
                                'a restarted pod can get a new host/port: read it from `pod get` and rerun with --ssh/--port, or use --endpoint-from-pod')
            self.sleep(15)

    def discover_endpoint(self) -> None:
        body = self.pod_get(self.args.pod_id)
        host, mappings = body.get('publicIp'), body.get('portMappings') or {}
        port = mappings.get('22')
        if host and port:       # shape of the Runpod REST pod object; unverified against a live restart
            self.ssh = Ssh(f'root@{host}', int(port), self.args.identity, self.dest / 'known_hosts', self.ssh.runner)
            self.report['endpoint'] = {'host': host, 'port': int(port)}

    def launch(self, resume: bool) -> dict[str, Any]:
        remaining = int(self.deadline - self.clock())
        wall = remaining - TRAINER_MARGIN_SECONDS
        if wall < MIN_TRAINER_WALL_SECONDS:
            raise HostError('lilith::supervise::window', f'only {remaining} s left; the trainer needs at least {MIN_TRAINER_WALL_SECONDS + TRAINER_MARGIN_SECONDS} s',
                            'use a longer --window-seconds')
        approval_remote = f'{self.data_dir}/approval.json'
        manifest_remote = f'{self.data_dir}/program-manifest.json'
        check = self.ssh.run(f'test -f {shlex.quote(manifest_remote)}', timeout=30)
        if check.returncode:
            raise HostError('lilith::supervise::data', f'{manifest_remote} is missing on the pod',
                            'rsync the frozen program-manifest.json and its archives to the pod data directory first')
        self.ssh.put(self.args.approval, approval_remote)
        command = (f'{self.pod_script} start --manifest {shlex.quote(manifest_remote)} --approval {shlex.quote(approval_remote)} '
                   f'--main-sha {self.approval.main_sha} --payload-root {shlex.quote(self.payload)} --max-wall-seconds {wall} '
                   f'--onpolicy-wait-mode {self.args.onpolicy_wait_mode}')
        if resume:
            command += ' --resume'
        if self.args.init_adapter:
            command += f' --init-adapter {shlex.quote(self.args.init_adapter)} --init-adapter-sha256 {self.args.init_adapter_sha256}'
        started = self.ssh.json(command, timeout=900)
        self.report['trainer'] = {'marker': started.get('marker'), 'window': started.get('window'), 'max_wall_seconds': wall,
                                  'log': started.get('log')}
        if started.get('exited_immediately'):
            raise HostError('lilith::supervise::trainer_start', f'trainer exited immediately (exit {started.get("exit_code")})',
                            f'read {started.get("log")} on the pod')
        return started

    # ------------------------------------------------------------- on-policy

    def handshake(self, stage: str) -> str:
        """Serve the stage adapter, wait for the session's archive, deliver it. Returns delivered|deadline."""
        local = self.dest / 'onpolicy' / stage
        local.mkdir(parents=True, exist_ok=True, mode=0o700)
        needed = self.ssh.json(f'cat {shlex.quote(f"{self.run_dir}/stages/{stage}/onpolicy-needed.json")}')
        (local / 'needed.json').write_text(json.dumps(needed, indent=2) + '\n')
        key = secrets.token_hex(32)
        key_file = local / 'serve-key'
        key_file.write_text(key + '\n')
        key_file.chmod(0o600)
        entry: dict[str, Any] = {'stage': stage, 'started_utc': utc(self.clock())}
        self.report['handshakes'].append(entry)
        served = self.ssh.run(f'{self.serve_script} start --root {shlex.quote(self.args.root)} --run-id {shlex.quote(self.run_id)} '
                              f'--stage {stage} --main-sha {self.approval.main_sha}', timeout=1800, input=key + '\n')
        if served.returncode:
            raise HostError('lilith::supervise::serve', f'serve_candidate.sh start failed: {served.stderr.strip()[-300:]}',
                            'with trainer GPU memory resident, retry with --onpolicy-wait-mode exit')
        self.vllm_running = True
        (local / 'serve-ready.json').write_text(json.dumps({
            'schema': 'lilith/program-serve-ready@1', 'run_id': self.run_id, 'stage': stage, 'served_model': 'lilith',
            'proxy_url': f'https://{self.args.pod_id}-8000.proxy.runpod.net/v1', 'api_key_file': str(key_file),
            'user_agent_required': USER_AGENT, 'needed': needed, 'written_at': utc(self.clock())}, indent=2) + '\n')
        archive_ready = local / 'archive-ready.json'
        while not archive_ready.is_file():
            if self.clock() >= self.deadline - HOST_TERM_LEAD_SECONDS:
                self.stop_vllm()
                key_file.unlink(missing_ok=True)
                entry['outcome'] = 'deadline'
                return 'deadline'
            self.sleep(self.args.driver_poll_seconds)
        marker = json.loads(archive_ready.read_text())
        name = f'onpolicy-{stage}.tokens.jsonl'
        if marker.get('archive') != name or not _SHA256.match(str(marker.get('sha256', ''))):
            raise HostError('lilith::supervise::archive', f'archive-ready.json must name {name} with a sha256', 'see the supervisor docstring')
        archive = local / name
        if not archive.is_file() or sha256_file(archive) != marker['sha256']:
            raise HostError('lilith::supervise::archive', f'{archive} is missing or differs from archive-ready.json', 'rewrite the archive and its marker')
        self.ssh.put(archive, f'{self.run_dir}/stages/{stage}/{name}')
        self.stop_vllm()                       # free the GPU before the trainer is told to continue
        key_file.unlink(missing_ok=True)
        ready = self.ssh.json(f'{self.pod_script} onpolicy-ready --run-id {shlex.quote(self.run_id)} --stage {stage} --archive {name} '
                              f'--sha256 {marker["sha256"]} --adapter-sha256 {needed["adapter_sha256"]}')
        entry.update({'outcome': 'delivered', 'archive_sha256': marker['sha256'], 'finished_utc': utc(self.clock()), 'ready': ready.get('stage')})
        return 'delivered'

    def stop_vllm(self) -> None:
        if self.vllm_running:
            result = self.ssh.run(f'{self.serve_script} stop --root {shlex.quote(self.args.root)}', timeout=300)
            if result.returncode:
                raise HostError('lilith::supervise::serve_stop', f'vLLM did not stop: {result.stderr.strip()[-200:]}', 'the GPU is held; stop the pod')
            self.vllm_running = False

    # --------------------------------------------------------------- the loop

    def loop(self) -> None:
        resume_needed = False
        while True:
            if self.clock() >= self.deadline - HOST_TERM_LEAD_SECONDS:
                self.report['run_status'] = 'window_ended'
                return
            probe = self.status()
            self.report['polls'] += 1
            with (self.dest / 'poll.jsonl').open('a') as stream:
                stream.write(json.dumps({'at': utc(self.clock()), 'alive': probe.get('alive'), 'age': probe.get('newest_write_age_seconds'),
                                         'gpu': probe.get('gpu_util'), 'progress': (probe.get('progress') or {}).get('status'),
                                         'step': (probe.get('progress') or {}).get('global_step'),
                                         'pending': probe.get('onpolicy_pending')}) + '\n')
            decision = decide(probe, stall_seconds=self.args.stall_seconds, stalled_strikes=self.stall_strikes)
            self.stall_strikes = self.stall_strikes + 1 if is_stall_probe(probe, self.args.stall_seconds) else 0
            if decision.action == 'continue':
                self.sleep(min(self.args.poll_seconds, max(1.0, self.deadline - HOST_TERM_LEAD_SECONDS - self.clock())))
            elif decision.action == 'handshake':
                outcome = self.handshake(decision.stage or '')
                if outcome == 'deadline':
                    self.report['run_status'] = 'window_ended'
                    return
                if not probe.get('alive'):       # exit wait mode: the trainer left its GPU free; resume it now
                    self.launch(resume=True)
            else:
                self.report['run_status'] = {'complete': 'complete', 'window_ended': 'window_ended', 'failed': 'failed',
                                             'crashed': 'crashed', 'stalled': 'stalled'}[decision.action]
                self.report['detail'] = decision.detail
                return

    def wind_down(self) -> None:
        try:
            status = self.status()
            if status.get('alive'):
                stopped = self.ssh.json(f'{self.pod_script} stop --run-id {shlex.quote(self.run_id)} --grace 420', timeout=520)
                self.report['trainer_stop'] = stopped
        except Exception as error:   # noqa: BLE001 - the pod stop below must still run
            self.report['trainer_stop_error'] = f'{type(error).__name__}: {error}'
        try:
            self.stop_vllm()
        except Exception as error:   # noqa: BLE001
            self.report['vllm_stop_error'] = f'{type(error).__name__}: {error}'
        self.collect()
        try:
            self.report['pod_stop'] = self.stop_pod(self.args.pod_id)
        except Exception as error:   # noqa: BLE001
            self.report['stop_error'] = f'{type(error).__name__}: {error}'
        end = self.clock()
        spend = {'window_start_utc': utc(self.t0), 'window_end_utc': utc(end), 'seconds': round(end - self.t0, 1),
                 'hourly_usd': self.args.hourly_usd,
                 'estimated_usd': None if self.args.hourly_usd is None else round((end - self.t0) / 3600 * self.args.hourly_usd, 2),
                 'run_status': self.report.get('run_status'), 'pod_id': self.args.pod_id}
        self.report['spend_window'] = spend
        with (self.dest / 'spend.jsonl').open('a') as stream:
            stream.write(json.dumps(spend) + '\n')

    def collect(self) -> None:
        artifacts = self.dest / 'artifacts'
        artifacts.mkdir(parents=True, exist_ok=True)
        fetched, missing = [], []
        for name in SMALL_ARTIFACTS:
            (fetched if self.try_get(f'{self.run_dir}/{name}', artifacts / name) else missing).append(name)
        if self.args.fetch_candidate and self.report.get('run_status') == 'complete':
            if self.try_get(f'{self.run_dir}/candidate-adapter', artifacts / 'candidate-adapter', recursive=True):
                fetched.append('candidate-adapter')
        self.report['artifacts'] = {'fetched': fetched, 'absent': missing}
        complete, candidate = artifacts / 'program-complete.json', artifacts / 'candidate-manifest.json'
        if complete.is_file() and candidate.is_file():
            expected = json.loads(complete.read_text()).get('candidate_manifest_sha256')
            self.report['artifacts']['candidate_manifest_verified'] = expected == sha256_file(candidate)

    def try_get(self, remote: str, local: Path, recursive: bool = False) -> bool:
        try:
            if recursive:
                result = self.ssh.runner(['scp', *self.ssh.opts, '-P', str(self.ssh.port), '-r', f'{self.ssh.target}:{remote}', str(local)],
                                         capture_output=True, text=True, timeout=900)
                return result.returncode == 0
            return self.ssh.get(remote, local)
        except Exception:   # noqa: BLE001 - an unreachable pod just means fewer artifacts
            return False

    def run(self) -> int:
        self.dest.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            if self.args.resume_window:
                self.start_pod(self.args.pod_id)
                self.report['pod_started'] = True
                if self.args.endpoint_from_pod:
                    self.sleep(30)
                    self.discover_endpoint()
            self.wait_ssh()
            self.launch(resume=self.args.resume_window or self.args.resume_trainer)
            self.loop()
        except BaseException as error:   # noqa: BLE001 - every path ends in wind_down, which stops the pod
            self.report['error'] = f'{type(error).__name__}: {error}'
            self.report.setdefault('run_status', 'failed')
        finally:
            self.wind_down()
            (self.dest / 'supervisor.json').write_text(json.dumps(self.report, indent=2) + '\n')
            (self.dest / 'supervisor.json').chmod(0o600)
        status = self.report.get('run_status')
        stopped = 'pod_stop' in self.report
        if not stopped:
            print('SUPERVISOR: pod stop NOT confirmed; check billing immediately', file=sys.stderr)
        print(json.dumps(self.report))
        if status == 'complete' and stopped:
            return EXIT_COMPLETE
        if status == 'window_ended' and stopped:
            return EXIT_WINDOW
        return EXIT_FAILURE


def parse_args(argv: Sequence[str] | None = None) -> HostArgs:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pod-id', required=True)
    p.add_argument('--ssh', required=True, help='user@host of the pod ssh endpoint')
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--identity', type=Path, required=True,
                   help="ssh private key file, or the literal 'agent' to use the running ssh-agent")
    p.add_argument('--window-seconds', type=int, required=True)
    p.add_argument('--approval', type=Path, required=True)
    p.add_argument('--dest', type=Path, required=True, help='new local evidence directory for this window')
    p.add_argument('--resume-window', action='store_true', help='start the stopped pod, then resume the trainer')
    p.add_argument('--resume-trainer', action='store_true', help='resume the trainer on a pod that is already running (no pod start)')
    p.add_argument('--hourly-usd', type=float, default=None, help='pod price refreshed at launch, for the spend record')
    p.add_argument('--poll-seconds', type=float, default=300.0)
    p.add_argument('--stall-seconds', type=float, default=1800.0)
    p.add_argument('--onpolicy-wait-mode', choices=('poll', 'exit'), default='poll')
    p.add_argument('--serve-env', action='append', default=[], metavar='LILITH_SERVE_KEY=VALUE',
                   help='vLLM serve setting prefixed to serve_candidate.sh (repeatable), e.g. LILITH_SERVE_MAX_MODEL_LEN=65536')
    p.add_argument('--init-adapter', default=None, help='pod path of the starting adapter (first window only)')
    p.add_argument('--init-adapter-sha256', default=None)
    p.add_argument('--root', default='/workspace/lilith')
    p.add_argument('--pod-payload', default=None, help='pod path of the extracted make_payload tarball (default <root>/payload/main-<sha12>)')
    p.add_argument('--pod-data-dir', default=None, help='pod path holding program-manifest.json + archives (default <root>/program-data/<run_id>)')
    p.add_argument('--fetch-candidate', action='store_true', help='scp the candidate adapter back when complete')
    p.add_argument('--endpoint-from-pod', action='store_true')
    ns = p.parse_args(argv)
    if bool(ns.init_adapter) != bool(ns.init_adapter_sha256):
        p.error('--init-adapter and --init-adapter-sha256 go together')
    if not re.fullmatch(r'[a-zA-Z0-9]+', ns.pod_id) or not re.fullmatch(r'[\w.-]+@[\w.:-]+', ns.ssh) or not 1 <= ns.port <= 65535:
        p.error('invalid pod id or ssh endpoint')
    if ns.window_seconds < MIN_TRAINER_WALL_SECONDS + TRAINER_MARGIN_SECONDS + 600:
        p.error(f'--window-seconds must be at least {MIN_TRAINER_WALL_SECONDS + TRAINER_MARGIN_SECONDS + 600}')
    return HostArgs(**{f.name: getattr(ns, f.name) for f in dataclasses.fields(HostArgs) if hasattr(ns, f.name)})


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        approval = load_host_approval(args.approval, args.pod_id, args.window_seconds)
    except (HostError, ValueError, OSError) as error:
        print(f'supervise_program: error: {error}', file=sys.stderr)
        return EXIT_FAILURE
    if Path(args.dest).exists():
        print(f'supervise_program: error: {args.dest} already exists; one evidence directory per window', file=sys.stderr)
        return EXIT_FAILURE
    return Supervisor(args, approval).run()


if __name__ == '__main__':
    raise SystemExit(main())
