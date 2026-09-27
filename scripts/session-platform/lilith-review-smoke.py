#!/usr/bin/env python3
"""Exercise isolated Lilith code creation, Ptah review and return to Lilith.

A development integration fixture, not a coding benchmark or illumos validation.
Starts/stops only the supplied project's loopback daemons. Keeps run evidence.
"""
import argparse
import json
import os
import pathlib
import socket
import subprocess
import time
import urllib.error
import urllib.request
import uuid

from lilith_verification import verify_clamp
from lilith_contracts import CONTRACTS, DEFAULT_CONTRACT
from lilith_coordination import Coordinator, Ledger, DONE, INLINE_REVIEWER



def run(root, bin_dir, output, timeout, coordinate=False, contract_id=DEFAULT_CONTRACT):
    contract = CONTRACTS[contract_id]
    if not coordinate and contract_id != DEFAULT_CONTRACT:
        raise ValueError('additional contracts require --coordinate')
    output.mkdir(parents=True, exist_ok=False)
    names = ['akhexecd', 'akhsbxd', 'akhrtd', 'akhsessd']
    configs = {name: json.loads((root / f'{name}.json').read_text()) for name in names}
    if coordinate:
        reviewer = configs['akhsessd']['akhs'].get(INLINE_REVIEWER)
        if reviewer is None or reviewer['tools']:
            raise ValueError('coordination requires a tool-free inline Ptah reviewer')
        grant = next(t for t in configs['akhsessd']['akhs']['lilith']['tools'] if t['name'] == 'request_review')
        if contract.artifact not in grant['input_schema']['properties']['path']['enum']:
            raise ValueError('selected contract is not granted by the session profile')
    for config in configs.values():
        host, port = config['listen'].rsplit(':', 1)
        if host != '127.0.0.1':
            raise ValueError('integration fixture requires loopback services')
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((host, int(port)))
    token = (root / 'credentials/operator.token').read_text().strip()
    processes, logs = {}, {}
    result = {'kind': ('local_durable_coordination_integration' if coordinate
                       else 'local_model_switch_and_review_integration'), 'status': 'running', 'phases': []}
    run_id = uuid.uuid4().hex
    coordinator = None
    ledger = None
    env = dict(os.environ, AKH_GPU_MATMUL='0', AKH_GPU_QUANT='0',
               RUST_LOG='info,akh_session_services::internal::native=debug')

    def request(name, path, body=None):
        req = urllib.request.Request('http://' + configs[name]['listen'] + path,
            data=None if body is None else json.dumps(body).encode(),
            headers={'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=15) as response:
            return json.load(response)

    def phase(name, **fields):
        row = {'phase': name, **fields}
        result['phases'].append(row)
        print(json.dumps(row), flush=True)

    def turn(session, text, label):
        nonlocal coordinator, ledger
        path = '/v1/sessions/' + session['id']
        started = time.monotonic()
        created = request('akhsessd', path + '/inputs', {'request_id': run_id + '-' + label, 'text': text})
        if coordinate and label == 'lilith-write':
            ledger = Ledger(root / 'state' / 'lilith-coordination.sqlite')
            coordinator = Coordinator(lambda path, body=None: request('akhsessd', path, body),
                ledger, session['id'], created['id'], root / 'workspaces', contract.verifier,
                artifact=contract.artifact, contract=contract.grant())
            phase('coordination_scope', session_id=session['id'], turn_id=created['id'])
        deadline = started + timeout
        while time.monotonic() < deadline:
            for name, process in processes.items():
                if process.poll() is not None:
                    raise RuntimeError(f'{name} exited during {label}')
            state = request('akhsessd', path)
            if state.get('dispatch_failures', 0) >= 3:
                (output / f'{label}-dispatch-failed.json').write_text(json.dumps(state, indent=2) + '\n')
                raise RuntimeError(f'{label}: repeated dispatch failures; inspect session and service logs')
            current = next((t for t in state['turns'] if t['id'] == created['id']), None)
            if current and current['status'] in ('failed', 'unknown', 'cancelled'):
                (output / f'{label}-failed.json').write_text(json.dumps(state, indent=2) + '\n')
                raise RuntimeError(f'{label} ended {current["status"]}: {current}')
            if current and current['status'] == 'completed':
                (output / f'{label}.json').write_text(json.dumps(state, indent=2) + '\n')
                phase(label, seconds=time.monotonic() - started, output=current.get('output'))
                return state, current
            if coordinator is not None:
                coordinator.tick()
            if request('akhsessd', path + '/pending') and coordinator is None:
                raise RuntimeError(f'{label} requested input outside the fixture contract')
            time.sleep(.5)
        raise TimeoutError(f'{label} exceeded {timeout}s')

    def receipts(session, required):
        records = request('akhsessd', '/v1/sessions/' + session['id'] + '/tools')['records']
        (output / f'{session["akh"]}-tools.json').write_text(json.dumps(records, indent=2) + '\n')
        for name in required:
            assert any(r['call']['name'] == name and not r['unknown'] and r.get('result')
                       and r['result']['success'] and r['result'].get('receipt') for r in records), name

    try:
        for name in names:
            logs[name] = (output / f'{name}.log').open('w')
            processes[name] = subprocess.Popen([str((bin_dir / name).resolve()), '--config', str(root / f'{name}.json')],
                env=env, stdout=logs[name], stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 30
            while True:
                if processes[name].poll() is not None:
                    raise RuntimeError(f'{name} exited on startup')
                try:
                    health = request(name, '/healthz')
                    assert health['service_id'] == configs[name]['service_id']
                    break
                except urllib.error.URLError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(name + ' startup')
                    time.sleep(.2)
        runtime_pid = processes['akhrtd'].pid
        phase('stack_ready', runtime_pid=runtime_pid,
              devices={k: v['native_candle']['device'] for k, v in configs['akhrtd']['runtimes'].items()})
        lilith = request('akhsessd', '/v1/sessions', {'request_id': run_id + '-lilith',
            'akh': 'lilith', 'workspace_profile': 'local-dev'})
        first, _ = turn(lilith, 'Remember the word cedar. Use write_file to create ' + contract.artifact + ' with create=true. '
            + contract.task + ' No imports, calls, decorators or loops. Write actual Python, not markdown. '
            + (f'Then use request_review with path={contract.artifact} and objective=Review the {contract.id} contract. '
               'After it acknowledges the queued review, finish your turn briefly.' if coordinate else 'Then confirm briefly.'),
            'lilith-write')
        workspace = pathlib.Path(first['workspace']['root'])
        source = (workspace / contract.artifact).read_text()
        verdict = contract.verifier(source)
        verdict.update(contract=contract.grant())
        (output / contract.artifact).write_text(source)
        (output / 'verification.json').write_text(json.dumps(verdict, indent=2) + '\n')
        receipts(first, ['write_file'])
        phase('artifact_verified', **verdict)
        if coordinate:
            # Re-open the controller after admission to exercise durable local recovery.
            parent_id, author_turn = coordinator.parent, coordinator.author_turn
            ledger.close()
            ledger = Ledger(root / 'state' / 'lilith-coordination.sqlite')
            coordinator = Coordinator(lambda path, body=None: request('akhsessd', path, body),
                ledger, parent_id, author_turn, root / 'workspaces', contract.verifier,
                artifact=contract.artifact, contract=contract.grant())
            phase('controller_reopened')
            deadline = time.monotonic() + timeout * 3
            while time.monotonic() < deadline:
                rows = coordinator.tick()
                (output / 'assignments.json').write_text(json.dumps(rows, indent=2) + '\n')
                if rows and all(row['status'] in DONE for row in rows):
                    assert len(rows) == 1 and rows[0]['status'] == 'completed', rows
                    assignment = rows[0]
                    final = request('akhsessd', '/v1/sessions/' + parent_id)
                    review = request('akhsessd', '/v1/sessions/' + assignment['child'])
                    (output / 'lilith-final.json').write_text(json.dumps(final, indent=2) + '\n')
                    (output / 'ptah-review.json').write_text(json.dumps(review, indent=2) + '\n')
                    assert final['provider_session'] == first['provider_session']
                    receipts(final, ['write_file', 'request_review'])
                    assert assignment['report'], 'empty human-facing report'
                    phase('autonomous_delegation_completed', assignment_id=assignment['id'],
                          reviewer=assignment['reviewer'], report=assignment['report'])
                    break
                time.sleep(.5)
            else:
                raise TimeoutError('coordination did not settle')
            assert processes['akhrtd'].pid == runtime_pid
            runtime_log = (output / 'akhrtd.log').read_text()
            assert runtime_log.count('unloaded previous native model') >= 2
            phase('same_process_model_roundtrip_verified', runtime_pid=runtime_pid)
        else:
            ptah = request('akhsessd', '/v1/sessions', {'request_id': run_id + '-ptah',
                'akh': 'ptah-validator', 'workspace_profile': 'local-dev'})
            # Workspace allocation happens at dispatch, not session creation.
            ptah_state, _ = turn(ptah, 'Reply READY. An artifact review will follow. Do not use tools yet.', 'ptah-ready')
            review_root = pathlib.Path(ptah_state['workspace']['root'])
            review_root.mkdir(parents=True, exist_ok=True)
            assert review_root.resolve() != workspace.resolve(), 'reviewer shares author workspace'
            (review_root / 'clamp_u8.py').write_text(source)
            (review_root / 'verification.json').write_text(json.dumps(verdict) + '\n')
            phase('explicit_review_handoff', source_sha256=verdict['source_sha256'])
            review, review_turn = turn(ptah, 'Use read_file to inspect clamp_u8.py. Review this integer-clamping '
                'function against the contract: below 0 returns 0, above 255 returns 255, otherwise unchanged. '
                'The supplied verification.json contains independent test evidence if needed. '
                'Report specific defects or say no defect found. Be concise; do not edit files.', 'ptah-review')
            receipts(review, ['read_file'])
            assert (review_turn.get('output') or '').strip(), 'empty review'
            final, final_turn = turn(lilith, 'What word did I originally ask you to remember? '
                'Reply with only that word, without tools.', 'lilith-return')
            assert (final_turn.get('output') or '').strip().lower().strip('.') == 'cedar'
            assert first['provider_session'], 'no durable provider session binding'
            assert final['provider_session'] == first['provider_session']
            assert processes['akhrtd'].pid == runtime_pid, 'runtime restarted instead of switching'
            runtime_log = (output / 'akhrtd.log').read_text()
            assert runtime_log.count('unloaded previous native model') >= 2, 'model unload roundtrip not observed'
            assert runtime_log.count('Candle GGUF model loaded successfully') >= 3, 'three model loads not observed'
            phase('same_process_model_roundtrip_verified', runtime_pid=runtime_pid)
        result['status'] = 'passed'
    except Exception as error:
        result.update(status='failed', error=str(error))
        print(json.dumps({'error': str(error)}), flush=True)
    finally:
        if ledger is not None:
            ledger.close()
        for process in reversed(list(processes.values())):
            if process.poll() is None:
                process.terminate()
        for process in processes.values():
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for log in logs.values():
            log.close()
        result['services_stopped'] = True
        (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    return result['status'] == 'passed'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=pathlib.Path, required=True)
    parser.add_argument('--bin-dir', type=pathlib.Path, default=pathlib.Path('target/release'))
    parser.add_argument('--output', type=pathlib.Path, required=True)
    parser.add_argument('--coordinate', action='store_true', help='Exercise model-requested durable Ptah delegation')
    parser.add_argument('--contract', choices=CONTRACTS, default=DEFAULT_CONTRACT)
    parser.add_argument('--turn-timeout', type=int, default=300)
    args = parser.parse_args()
    if args.turn_timeout <= 0:
        parser.error('turn timeout must be positive')
    return 0 if run(args.root.resolve(), args.bin_dir, args.output, args.turn_timeout, args.coordinate, args.contract) else 1


if __name__ == '__main__':
    raise SystemExit(main())
