#!/usr/bin/env python3
"""T3's existing Grok ACP driver used solely as a transport for Roman akhs.

No Grok model/API is used. Ignore CLI permission flags: the daemon owns grants.
The executable basename selects an operator-provisioned, individually scoped
credential; project cwd and T3 filesystem capabilities never become agent tools.

Lilith (ADR 116, Lilith C2; ADR 121) is served by llama-server on hephaistos, the operator's
Linux box (`lilith_serving` = "hephaistos" in config/t3-acp.json, the default). T3 only
health-checks it: when it is down this script logs one clear line and starts nothing, because
the box is the operator's and nothing here may power it on or bill anything. With
`lilith_serving` = "runpod" (training-era tests) her endpoint is a Runpod pod that is normally
stopped: when T3 starts her transport and her endpoint is down, this script starts
`lilith_serve.py start` detached (budget-checked by that script) and carries on. Either way the
ACP handshake never waits for the model.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROMANS = ('janus', 'minerva', 'vesta')
LILITH = 'lilith'
MODEL_LABELS = {LILITH: 'Gemma 4 12B (Lilith) on hephaistos (llama.cpp), in-process in akhomed'}
DEFAULT_MODEL_LABEL = 'GLM-5.3 on Ollama Cloud, in-process in akhomed'
# Where her model is served (config/t3-acp.json `lilith_serving`, rendered from config/lilith.json).
SERVING_HEPHAISTOS = 'hephaistos'
SERVING_RUNPOD = 'runpod'
DEFAULT_SERVING = SERVING_HEPHAISTOS
# Runpod's proxy answers Python's default User-Agent with 403.
HEALTH_USER_AGENT = 'curl/8.7.1'
HEALTH_TIMEOUT_SECONDS = 3
SERVE_PYTHON = '/opt/homebrew/bin/python3'
SERVE_SCRIPT = Path('scripts/lilith/serve/lilith_serve.py')
# `lilith_serve.py start` waits up to 30 min for vLLM; a lock older than that is stale.
SERVE_LOCK_MAX_AGE_SECONDS = 35 * 60
# A start that refused or failed exits within seconds; do not re-ask (each `start` reads the
# Runpod key from 1Password, whose account-wide read allowance is shared) for this long.
SERVE_RETRY_COOLDOWN_SECONDS = 5 * 60
# While hephaistos is down every T3 session start would log the same line; once per this long.
HEPHAISTOS_LOG_COOLDOWN_SECONDS = 10 * 60


def allowed_roles(settings):
    """The roles this deployment renders (`akhs` in config/t3-acp.json), else the three Romans."""
    akhs = settings.get('akhs')
    if isinstance(akhs, list) and akhs and all(isinstance(a, str) and a for a in akhs):
        return tuple(akhs)
    return ROMANS


def resolve_role(argv0, settings):
    """The role named by the executable basename, or None when it is not a rendered akh."""
    role = Path(argv0).name.removeprefix('roman-')
    return role if role in allowed_roles(settings) else None


def lilith_healthy(root, opener=urllib.request.urlopen):
    """True when her model server answers `/health` (llama-server, vLLM). Any failure (no URL file,
    network, non-2xx; llama-server answers 503 while it loads the model) is 'down'."""
    try:
        base = (root / 'secrets' / 'lilith-base-url').read_text().strip().rstrip('/').removesuffix('/v1')
        if not base:
            return False
        request = urllib.request.Request(base + '/health', headers={'User-Agent': HEALTH_USER_AGENT})
        with opener(request, timeout=HEALTH_TIMEOUT_SECONDS) as response:
            return 200 <= response.status < 300
    except Exception:
        return False


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _log(log_path, message):
    try:
        with open(log_path, 'a') as log:
            log.write(time.strftime('%Y-%m-%dT%H:%M:%S%z') + ' t3_entrypoint: ' + message + '\n')
    except OSError:
        pass


def _take_lock(lock_path, now):
    """Create the start lock exclusively.

    False while a start is in flight (live pid, younger than SERVE_LOCK_MAX_AGE_SECONDS) or
    the last one began less than SERVE_RETRY_COOLDOWN_SECONDS ago (it may have been refused).
    """
    for _ in range(2):
        try:
            fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            try:
                held = int(lock_path.read_text().strip() or 0)
                age = now - lock_path.stat().st_mtime
            except (OSError, ValueError):
                held, age = 0, SERVE_LOCK_MAX_AGE_SECONDS
            if age < SERVE_RETRY_COOLDOWN_SECONDS or (
                    held and age < SERVE_LOCK_MAX_AGE_SECONDS and _pid_alive(held)):
                return False
            try:
                lock_path.unlink()  # stale: the start ended or crashed
            except FileNotFoundError:
                pass
            continue
        with os.fdopen(fd, 'w') as file:
            file.write(str(os.getpid()))
        return True
    return False


def ensure_lilith_serving(root, settings, opener=urllib.request.urlopen, popen=subprocess.Popen,
                          now=time.time):
    """Make sure her model server is up, as far as this script may.

    `settings['lilith_serving']` (default 'hephaistos'):
    - 'hephaistos': health-check only. Returns 'up', or 'down' after logging one clear line to
      ROOT/logs/lilith-serve-start.log (the log lines are throttled to one per cooldown). It never
      starts a pod, never powers the box on and spawns nothing: the repo has no wake mechanism for
      hephaistos, and inventing one here is not this script's job.
    - 'runpod': starts her Runpod serve pod in the background when her endpoint is down. Returns
      'up', 'spawned', 'locked' (a start is already in flight), or 'skipped' (no checkout or
      script; the reason is in the log). A refusal from `lilith_serve.py start` (budget, zero GPU)
      lands in the same log because the child writes its stdout/stderr there, and her turn fails
      with the backend error until the pod is ready (first message after a cold start: ~8-10 min).
    - anything else: 'skipped', logged. An unknown value must never start a billable pod.

    Never raises: the ACP handshake must not depend on the model server.
    """
    log_path = root / 'logs' / 'lilith-serve-start.log'
    serving = None
    try:
        serving = settings.get('lilith_serving', DEFAULT_SERVING)
        if serving not in (SERVING_HEPHAISTOS, SERVING_RUNPOD):
            _log(log_path, f'unknown lilith_serving {serving!r} in config/t3-acp.json '
                           '(hephaistos or runpod); starting nothing')
            return 'skipped'
        if lilith_healthy(root, opener):
            return 'up'
        if serving == SERVING_HEPHAISTOS:
            _note_hephaistos_down(root, log_path, now())
            return 'down'
        return _spawn_runpod_serve(root, settings, log_path, popen, now)
    except Exception as error:  # the ACP handshake must not depend on the model server
        what = 'start the serve pod' if serving == SERVING_RUNPOD else 'check her model server'
        _log(log_path, f'could not {what}: {type(error).__name__}: {error}')
        return 'skipped'


def _note_hephaistos_down(root, log_path, now):
    """One log line per cooldown: each T3 session start would otherwise append another."""
    stamp = root / 'state' / 'lilith' / 't3-hephaistos-down.stamp'
    try:
        if now - stamp.stat().st_mtime < HEPHAISTOS_LOG_COOLDOWN_SECONDS:
            return
    except OSError:
        pass
    _log(log_path, 'her model server on hephaistos is not answering /health. Nothing was started: '
                   'hephaistos is the operator\'s box. Check that it is powered on and that '
                   'lilith-serve.service is active (docs/ops/lilith-serve-hephaistos.md); her turns fail '
                   'with the backend error until then')
    try:
        stamp.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        stamp.write_text(str(now))
        os.utime(stamp, (now, now))
    except OSError:
        pass


def _spawn_runpod_serve(root, settings, log_path, popen, now):
    checkout = settings.get('lilith_serve')
    script = Path(checkout).expanduser() / SERVE_SCRIPT if checkout else None
    if script is None or not script.is_file():
        _log(log_path, f'endpoint is down but no lilith_serve.py to start ({script}); '
                       'set lilith_serve in config/t3-acp.json (serve_checkout in config/lilith.json)')
        return 'skipped'
    state = root / 'state' / 'lilith'
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = state / 't3-serve-start.pid'
    if not _take_lock(lock, now()):
        return 'locked'
    env = dict(os.environ)
    env['PATH'] = '/opt/homebrew/bin:/usr/local/bin:' + env.get('PATH', '/usr/bin:/bin')
    _log(log_path, f'endpoint is down; starting {script} start (detached)')
    with open(log_path, 'a') as log:
        child = popen([SERVE_PYTHON, str(script), 'start'], stdin=subprocess.DEVNULL, stdout=log,
                      stderr=subprocess.STDOUT, start_new_session=True, close_fds=True, env=env)
    lock.write_text(str(child.pid))
    return 'spawned'


def main(argv):
    root = Path(__file__).resolve().parent.parent
    # Phase 53f cutover (ADR 113, 53d notes): `config/t3-acp.json`, rendered by
    # configure_core.py, points this transport at akhomed's own ACP TCP listener via
    # `akh-acp --akhomed`. Without that file (or with ROMAN_T3_ACP_ENDPOINT set to a
    # ws:// URL) the transport stays on the akhsessd broker. Deleting the file is the
    # rollback.
    settings_path = root / 'config' / 't3-acp.json'
    settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
    role = resolve_role(argv[0], settings)
    if role is None:
        raise SystemExit('unknown Roman Akh')
    endpoint = os.environ.get('ROMAN_T3_ACP_ENDPOINT', settings.get('endpoint', 'ws://127.0.0.1:18431/acp'))
    health_url = os.environ.get('ROMAN_T3_HEALTH_URL', settings.get('health_url', 'http://127.0.0.1:18431/v1/sessions'))
    akhomed = endpoint.startswith('tcp://')
    if '--version' in argv:
        print('Roman Akh ACP bridge 0.1.0')
    elif 'models' in argv:
        token = (root / 'secrets' / ('t3-' + role + '.key')).read_text().strip()
        request = urllib.request.Request(health_url, headers={'Authorization': 'Bearer ' + token})
        try:
            with urllib.request.urlopen(request, timeout=5):
                pass
            model = MODEL_LABELS.get(role, DEFAULT_MODEL_LABEL) if akhomed else 'Mistral Large 3 via Ollama Cloud'
            print('You are logged in to the Roman Akh session daemon.\nDefault model: grok-build\nAvailable models:\n * grok-build (default)\nTransport alias only: configured daemon model is ' + model + '.')
        except Exception:
            raise SystemExit('Roman session daemon not authenticated or unavailable')
    else:
        if role == LILITH:
            ensure_lilith_serving(root, settings)
        binary = Path(settings.get('binary', root / 'bin/akh-acp')) if akhomed else root / 'bin/akh-acp'
        token_file = str(root / 'secrets' / ('t3-' + role + '.key'))
        if akhomed:
            # akhomed has no workspace profiles; `--akhomed` injects the bearer token
            # and the akh selection into initialize / session/new.
            args = [str(binary), '--akhomed', '--endpoint', endpoint, '--token-file', token_file, '--akh', role, '--t3-compat']
        else:
            args = [str(binary), '--endpoint', endpoint, '--token-file', token_file, '--akh', role, '--workspace-profile', role, '--t3-compat']
        os.execv(binary, args)


if __name__ == '__main__':
    main(sys.argv)
