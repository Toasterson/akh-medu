#!/usr/bin/env python3
"""Run a training worker through a private akhsessd/akhrtd/executor stack."""
import argparse
import copy
import hashlib
import json
import secrets
import shutil
import socket
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path

from lilith_training_transport import digest_file
from lilith_paths import resolve as resolve_paths
from lilith_runtime_training_worker import canonical, digest, write
from lilith_deferred_review import DeferredReviewQueue, reconcile_with_reviewer

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
DEFAULT_RETIREMENT_CANCEL_GRACE_MS = 2 * 60_000


def configure_training_session_timeout(config, training_timeout_seconds):
    """Keep the private training session alive through its worker deadline."""
    if training_timeout_seconds <= 0:
        raise ValueError('training timeout must be positive')
    cleanup_grace_ms = config.get(
        'retirement_cancel_grace_ms', DEFAULT_RETIREMENT_CANCEL_GRACE_MS)
    if cleanup_grace_ms < 0:
        raise ValueError('retirement cancellation grace must not be negative')
    config['retirement_cancel_grace_ms'] = cleanup_grace_ms
    config['session_stall_threshold_ms'] = training_timeout_seconds * 1000 + cleanup_grace_ms
    return config['session_stall_threshold_ms']


def capture_session_events(api, session_path, events_path, after):
    """Append unseen durable session events and return the new sequence cursor."""
    while True:
        events = api('akhsessd', session_path + '/events?after=' + str(after))
        for event in events:
            sequence = event['sequence']
            if sequence <= after:
                raise ValueError('session events are not strictly increasing')
            with events_path.open('ab') as stream:
                stream.write(canonical(event) + b'\n')
            after = sequence
        if len(events) < 1000:
            return after


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker-config', type=Path)
    parser.add_argument('--timeout', type=int, default=180)
    parser.add_argument('--base-port', type=int, default=18620)
    parser.add_argument('--paths-config', type=Path,
                        help='lilith/paths@1 JSON; else LILITH_* env, else hephaistos defaults')
    args = parser.parse_args()
    paths = resolve_paths(args.paths_config)
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    prior = paths.templates_dir
    private = paths.private_root / ('lilith-runtime-training-' + uuid.uuid4().hex)
    private.mkdir(mode=0o700)
    (private / 'credentials').mkdir(mode=0o700)
    (private / 'workspaces').mkdir(mode=0o700)
    names = ('akhexecd', 'akhsbxd', 'akhrtd', 'akhsessd')
    configs = {}
    for name in names:
        text = (prior / (name + '.json')).read_text().replace(str(prior), str(private))
        for index in range(4):
            text = text.replace('127.0.0.1:' + str(18420 + index), '127.0.0.1:' + str(args.base_port + index))
        configs[name] = json.loads(text)
        for principal in configs[name]['principals']:
            principal['akhs'] = ['lilith-training']
        host, port = configs[name]['listen'].rsplit(':', 1)
        if host != '127.0.0.1':
            raise ValueError('training stack must remain loopback-only')
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind((host, int(port)))
    session_stall_threshold_ms = configure_training_session_timeout(
        configs['akhsessd'], args.timeout)
    for name in ('operator',) + names:
        token = secrets.token_urlsafe(32)
        for suffix, value in [('token', token), ('sha256', hashlib.sha256(token.encode()).hexdigest())]:
            path = private / 'credentials' / (name + '.' + suffix)
            path.write_text(value + '\n'); path.chmod(0o600)
    inference = copy.deepcopy(configs['akhsessd']['akhs']['lilith'])
    reviewer = copy.deepcopy(configs['akhsessd']['akhs']['ptah-inline-reviewer'])
    reviewer['tools'] = []
    write(output / 'inference-tools.json', inference['tools'])
    tools_hash = digest(inference['tools'])
    review_tool = next(tool for tool in inference['tools'] if tool['name'] == 'request_review')
    review_queue = DeferredReviewQueue(output / 'review-queue',
        review_tool['input_schema']['properties']['path']['enum'], tools_hash)
    reviewer_name = 'ptah-deferred-reviewer'
    python = paths.venv_python
    worker_script = SCRIPTS / 'lilith_runtime_training_worker.py'
    if args.worker_config:
        worker = json.loads(args.worker_config.read_text())
        if worker['tools_sha256'] != tools_hash:
            raise ValueError('requested worker catalog differs from Lilith inference')
        # The caller freezes the experiment; the stack only supplies its output location.
        if worker['output'] != str(output / 'worker'):
            raise ValueError('worker output differs from declared stack artifact location')
    else:
        worker = {'schema':'lilith/runtime-training-worker@1', 'mode':'fixture',
            'output':str(output / 'worker'), 'tools_sha256':tools_hash, 'artifact_path':'clamp_u8.py',
            'max_context_tokens':32768, 'max_output_tokens':16384, 'max_iterations':16,
            'source_pins':{str(SCRIPTS / name):digest_file(SCRIPTS / name) for name in
                ('lilith_runtime_training_worker.py','lilith_training_transport.py','lilith_gemma_tool_frames.py')}}
    worker_path = private / 'worker-config.json'; write(worker_path, worker)
    write(output / 'worker-config.json', worker)
    runtime = copy.deepcopy(configs['akhrtd']['runtimes']['lilith'])
    runtime.pop('native_candle', None); runtime.pop('model_endpoint', None)
    runtime['revision'] = 'lilith-runtime-training-v1'
    runtime['timeout_secs'] = args.timeout
    runtime['environment'] = {'PATH':str(python.parent) + ':/usr/bin:/bin', 'PYTHONUNBUFFERED':'1',
        'PYTHONNOUSERSITE':'1', 'HF_HUB_OFFLINE':'1', 'OMP_NUM_THREADS':'2'}
    runtime['isolation'] = {'directory':str(private / 'worker-isolation')}
    Path(runtime['working_dir']).mkdir(parents=True, exist_ok=True, mode=0o700)
    Path(runtime['isolation']['directory']).mkdir(mode=0o700)
    runtime['training_worker'] = {'executable':str(python), 'executable_sha256':digest_file(python),
        'script':str(worker_script), 'script_sha256':digest_file(worker_script),
        'worker_config_path':str(worker_path), 'worker_config_sha256':digest_file(worker_path),
        'model':inference['runtime']['model'], 'max_iterations':worker['max_iterations'],
        'max_context_tokens':worker['max_context_tokens'], 'max_output_tokens':worker['max_output_tokens'],
        'max_tool_calls':2048}
    ptah_runtime = copy.deepcopy(configs['akhrtd']['runtimes']['ptah'])
    ptah_runtime['working_dir'] = str(private / 'runtime/ptah')
    ptah_runtime['isolation']['directory'] = str(private / 'runtime/ptah/profile')
    Path(ptah_runtime['working_dir']).mkdir(parents=True, exist_ok=True, mode=0o700)
    Path(ptah_runtime['isolation']['directory']).mkdir(parents=True, exist_ok=True, mode=0o700)
    configs['akhrtd']['runtimes'] = {'lilith-training':runtime}
    reviewer_runtime_config = copy.deepcopy(configs['akhrtd'])
    reviewer_runtime_config['runtimes'] = {'ptah':ptah_runtime}
    inference['runtime'].update(profile='lilith-training', revision=runtime['revision'])
    configs['akhsessd']['akhs'] = {'lilith-training':inference, reviewer_name:reviewer}
    for config in configs.values():
        for principal in config['principals']:
            principal['akhs'] = ['lilith-training', reviewer_name]
    for principal in reviewer_runtime_config['principals']:
        principal['akhs'] = ['lilith-training', reviewer_name]
    binaries = {name:paths.runtime_binary(name) for name in names}
    binaries['akhrtd'] = args.runtime_binary.resolve()
    for name, config in configs.items():
        write(private / (name + '.json'), config)
    write(private / 'akhrtd-review.json', reviewer_runtime_config)
    reviewer_binary = paths.reviewer_runtime_binary.resolve(strict=True)
    reviewer_binary_sha256 = digest_file(reviewer_binary)
    write(output / 'protocol.json', {'schema':'lilith/managed-training-stack@1',
        'driver_sha256':digest_file(__file__), 'worker_config_sha256':digest_file(worker_path),
        'review_controller_sha256':digest_file(SCRIPTS / 'lilith_deferred_review.py'),
        'review_verifier_sources':{name:digest_file(SCRIPTS / name) for name in
            ('adam_kaine_code_rl_tasks.py', 'adam_kaine_code_environment.py',
             'lilith_rl_tasks_v2.py', 'lilith_rl_tasks.py', 'lilith_vm_executor.py')},
        'review_runtime_config_sha256':digest_file(private / 'akhrtd-review.json'),
        'binaries':{name:{'path':str(path), 'sha256':digest_file(path)} for name,path in binaries.items()},
        'reviewer_binary':{'path':str(reviewer_binary), 'sha256':reviewer_binary_sha256},
        'reviewer_model_sha256':ptah_runtime['native_candle']['model_sha256'],
        'reviewer_tokenizer_sha256':ptah_runtime['native_candle']['tokenizer_sha256'],
        'inference_tools_sha256':tools_hash, 'training_tools_sha256':digest(inference['tools']),
        'approval_policy':inference['approval_policy'], 'private_root':str(private),
        'mode':worker['mode'], 'timeout_seconds':args.timeout, 'paths':paths.to_record(),
        'session_stall_threshold_ms':session_stall_threshold_ms,
        'retirement_cancel_grace_ms':configs['akhsessd']['retirement_cancel_grace_ms'],
        'scope':'Actual runtime-managed worker and session/executor callbacks; fixture mode performs no model update.'})
    operator_token = (private / 'credentials/operator.token').read_text().strip()
    def api(name, path, body=None):
        request = urllib.request.Request('http://' + configs[name]['listen'] + path,
            data=None if body is None else canonical(body),
            headers={'Authorization':'Bearer ' + operator_token, 'Content-Type':'application/json'})
        with urllib.request.urlopen(request, timeout=15) as response:
            return json.load(response)
    processes, logs = [], []
    result = {'status':'starting', 'mode':worker['mode'], 'deployed_weights_changed':False}
    write(output / 'result.json', result)
    try:
        for name in names:
            log = (output / (name + '.log')).open('w'); logs.append(log)
            process = subprocess.Popen([str(binaries[name]), '--config', str(private / (name + '.json'))],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            processes.append(process)
            for _ in range(100):
                if process.poll() is not None:
                    raise RuntimeError(name + ' exited before readiness')
                try:
                    if api(name, '/healthz')['service_id'] == name:
                        break
                except (OSError, ValueError):
                    time.sleep(0.2)
            else:
                raise TimeoutError(name + ' readiness')
        session = api('akhsessd', '/v1/sessions', {'request_id':uuid.uuid4().hex,
                      'akh':'lilith-training', 'workspace_profile':'local-dev'})
        path = '/v1/sessions/' + session['id']
        turn = api('akhsessd', path + '/inputs', {'request_id':uuid.uuid4().hex,
                   'text':'Execute the pinned operator-owned Lilith learning run within this assigned workspace.'})
        result.update(status='running', session_id=session['id'], turn_id=turn['id'])
        write(output / 'result.json', result)
        events_path = output / 'events.jsonl'
        events_path.touch(mode=0o600, exist_ok=False)
        event_cursor = 0
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            session = api('akhsessd', path)
            current = next(item for item in session['turns'] if item['id'] == turn['id'])
            write(output / 'session.json', session)
            event_cursor = capture_session_events(api, path, events_path, event_cursor)
            if current['status'] in ('completed', 'failed', 'cancelled', 'unknown'):
                break
            # Never fabricate client review outcomes or approve a new permission.
            pending = api('akhsessd', path + '/pending')
            if pending:
                write(output / 'pending.json', pending)
            for item in pending:
                active = sorted((output / 'worker').glob('**/active-task.json'),
                                key=lambda candidate: candidate.stat().st_mtime_ns)
                if not active:
                    raise RuntimeError('review request has no frozen active task')
                task = json.loads(active[-1].read_text())
                queued = review_queue.admit(session, item, {'task_id':task['task_id'],
                    'split':task['split'], 'verifier_sha256':worker.get('verifier_sha256'),
                    'active_task_sha256':digest_file(active[-1])})
                if queued is None:
                    continue
                decision = api('akhsessd', path + '/input-decisions', {
                    'request_id':queued['queue_id'] + '-ack', 'turn_id':item['turn_id'],
                    'input_id':item['id'], 'generation':item['generation'],
                    'response':review_queue.response(queued)})
                review_queue.record_decision(queued['queue_id'], decision)
            time.sleep(1)
        else:
            raise TimeoutError('managed training turn deadline')
        records = api('akhsessd', path + '/tools')['records']
        write(output / 'tool-records.json', records)
        if current['status'] != 'completed':
            raise RuntimeError('managed training ended ' + current['status'])
        if digest(session['tools']) != tools_hash:
            raise ValueError('runtime session changed the inference tool catalog')
        queued_reviews = review_queue.rows()
        write(output / 'review-queue.json', queued_reviews)
        if queued_reviews:
            # The training runtime may retain Gemma VRAM after the job. Stop and reap it
            # before starting the distinct Ptah process; sessd/executor remain live.
            training_runtime = processes[2]
            training_runtime.terminate()
            training_runtime.wait(timeout=30)
            if digest_file(reviewer_binary) != reviewer_binary_sha256:
                raise RuntimeError('reviewer runtime binary changed before launch')
            review_log = (output / 'akhrtd-review.log').open('w'); logs.append(review_log)
            review_runtime = subprocess.Popen([str(reviewer_binary), '--config',
                str(private / 'akhrtd-review.json')], cwd=ROOT,
                stdout=review_log, stderr=subprocess.STDOUT)
            processes.append(review_runtime)
            for _ in range(100):
                if review_runtime.poll() is not None:
                    raise RuntimeError('reviewer akhrtd exited before readiness')
                try:
                    if api('akhrtd', '/healthz')['service_id'] == 'akhrtd':
                        break
                except (OSError, ValueError):
                    time.sleep(0.2)
            else:
                raise TimeoutError('reviewer akhrtd readiness')

            import adam_kaine_code_environment as code_environment
            import adam_kaine_code_rl_tasks as code_tasks
            from lilith_vm_executor import local_executor
            indexed_tasks = {task.id:task for task in code_tasks.tasks()}
            executor = local_executor()
            native_root = output / 'native-review'; native_root.mkdir(mode=0o700)
            def verify_review(row, artifact):
                task_id = (row.get('context') or {}).get('task_id')
                if task_id not in indexed_tasks:
                    raise ValueError('queued review has no pinned curriculum task')
                repository = native_root / row['queue_id']; repository.mkdir(mode=0o700)
                evidence = code_environment.native_audit(
                    artifact.decode('utf-8'), indexed_tasks[task_id], executor, repository)
                evidence['verifier_sha256'] = code_tasks.verifier_identity()
                evidence['task_id'] = task_id
                write(repository / 'evidence.json', evidence)
                return evidence
            reviews = reconcile_with_reviewer(
                lambda request_path, body=None: api('akhsessd', request_path, body),
                review_queue, reviewer_name, ptah_runtime['native_candle']['model_sha256'],
                verify_review, timeout_seconds=300)
            write(output / 'review-queue.json', reviews)
            review_queue.require_terminal()
        result.update(status='completed', output=current.get('output'), tool_records=len(records),
                      tools_identical_to_inference=True)
    except BaseException as error:
        result.update(status='failed', error=type(error).__name__ + ': ' + str(error))
        raise
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
        for log in logs:
            log.close()
        result['services_stopped'] = all(process.poll() is not None for process in processes)
        shutil.rmtree(private)
        result['private_root_removed'] = not private.exists()
        write(output / 'result.json', result)


if __name__ == '__main__':
    main()
