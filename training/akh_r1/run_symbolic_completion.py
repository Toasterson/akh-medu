#!/usr/bin/env python3
"""Native symbolic completion of recorded proposals, with causal ablations."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import sys

from bounded_process import ExperimentStop, run_bounded
from build_curriculum import digest, now
from command_registry import CommandRegistry
from environment import Graph
from run_binding_kata import fixture_for, score
from run_exposure import read, write, unchanged
from symbolic_policy import populate, infer, rules, proof_matches_rules


def prepare(args):
    parent = args.parent.resolve(); summary = read(parent / 'summary.json')
    if not summary['all_checks_passed'] or read(parent / 'status.json')['stage'] != 'completed':
        raise ValueError('Completed audited parent required')
    root = args.out.resolve(); root.mkdir(exist_ok=False); source = root / 'source'; source.mkdir()
    for p in Path(__file__).parent.glob('*.py'):
        if not p.name.startswith('test_'): shutil.copy2(p, source / p.name)
    shutil.copy2(args.bridge, source / 'akh_symbolic_episode_bridge')
    worktree = Path(__file__).resolve().parents[2]
    shutil.copy2(worktree / 'examples/akh_symbolic_episode_bridge.rs', source / 'akh_symbolic_episode_bridge.rs')
    shutil.copy2(worktree / 'src/autonomous/rule_engine.rs', source / 'native_rule_engine.rs')
    files = {}; proposals = []; cases = None; image = None
    for arm in ['history', 'compact']:
        p = read(parent / arm / 'binding/plan.json'); r = read(parent / arm / 'binding/results.json')
        if not unchanged(p['files']): raise ValueError('Binding evidence changed')
        files.update(p['files'])
        if cases is not None and cases != p['cases']: raise ValueError('Case mismatch')
        cases = p['cases']; image = p['image']
        proposals.extend(dict(adapter=arm, **row) for row in r['decisions'])
        for n in ['plan.json', 'results.json', 'summary.json', 'artifact-audit.json']:
            q = parent / arm / 'binding' / n; files[str(q)] = digest(q)
    if len(cases) != 18 or len(proposals) != 72: raise ValueError('Expected 18 cases and 72 recorded proposals')
    public = {}
    for p in cases:
        case = read(p)
        public[case['id']] = dict(goal=case['goal'], observations=case['observations'], records=case['graph_records'])
    write(root / 'public-inputs.json', public); write(root / 'proposals.json', proposals); write(root / 'rules.json', rules())
    files.update({str(p):digest(p) for p in [*source.iterdir(), root / 'public-inputs.json', root / 'proposals.json', root / 'rules.json']})
    plan = dict(created_at=now(), parent=str(parent), cases=cases, image=image, files=files,
        bridge=str(source / 'akh_symbolic_episode_bridge'), lock_file=str(args.lock_file.resolve()),
        timeout_seconds=900, min_free_bytes=5 * 2**30, outcomes=144, training_updates=0, new_model_generations=0,
        release_certification=False, automatic_retries=False,
        scope='72 recorded model proposals completed by native rules, 18 symbolic-only executions and 54 causal controls. '
              'Public goal/description compiler and supplied observations/command records are explicit symbolic assistance. '
              'Policy receives no fixture state, examiner answer or scores. One-operation development replay, not full autonomous episodes.')
    write(root / 'plan.json', plan); (root / 'plan.sha256').write_text(digest(root / 'plan.json') + '\n')
    write(root / 'status.json', dict(stage='prepared', updated_at=now()))


def run(root):
    p = read(root / 'plan.json'); public = read(root / 'public-inputs.json'); proposals = read(root / 'proposals.json')
    rows = []
    with (root / 'trace.jsonl').open('x') as trace:
        for number, case_path in enumerate(p['cases']):
            case = read(case_path); dest = root / 'cases' / case['id']; dest.mkdir(parents=True)
            fixture = fixture_for(p['image'], case['before']['contract'], 'symbolic')
            try:
                for condition in ['full', 'no_knowledge', 'no_rules']:
                    target = dest / condition; target.mkdir(); graph = Graph(p['bridge'], target / 'graph', target / 'engine.log')
                    try:
                        first_pid = graph.process.pid
                        compile_error = None
                        try:
                            facts = populate(graph, public[case['id']], include_knowledge=condition != 'no_knowledge')
                        except ValueError as error:
                            # Unsupported policy inputs are measured capability
                            # limits; preserve them without aborting later cases.
                            compile_error = str(error); facts = []
                        write(target / 'compiled-facts.json', facts)
                        write(target / 'graph-populated.json', graph.call('exam_export'))
                    finally: graph.close()
                    graph = Graph(p['bridge'], target / 'graph', target / 'engine-cold.log')
                    try:
                        cold = graph.call('exam_export')
                        if cold != read(target / 'graph-populated.json'): raise ValueError('Cold graph differs')
                        derived = infer(graph, enable_rules=condition != 'no_rules')
                        derived['compile_error'] = compile_error
                        write(target / 'derivation.json', derived)
                        action = derived['action']
                        variants = [dict(mode=condition, proposal=None)]
                        if condition == 'full':
                            variants += [dict(mode='hybrid', proposal=q) for q in proposals if q['case'] == case['id']]
                            variants += [dict(mode='version_changed', proposal=None)]
                        for variant in variants:
                            before = copy.deepcopy(case['before'])
                            if variant['mode'] == 'version_changed': before['contract']['version'] += 1
                            fixture.reset(before); execution = None; response = dict(success=False, error='No unique derivation'); stages = []
                            if action:
                                try:
                                    registry = CommandRegistry(graph, [case['before']['contract']['observer'], case['before']['contract']['controller']])
                                    execution = registry.invoke(fixture, **action['arguments'])
                                    response = execution['response']; stages = [execution['argv']]
                                except ValueError as error: response = dict(success=False, error=str(error))
                            after = fixture.state()
                            judged = score(before, after, case['target_job'], case['operation'], response, stages, case['examiner']['expected_argv'])
                            proposal = variant['proposal']; request = proposal['request'] if proposal else None
                            row = dict(case=case['id'], split=case['split'], operation=case['operation'], condition=condition, mode=variant['mode'],
                                adapter=proposal['adapter'] if proposal else None, presentation=proposal['arm'] if proposal else None,
                                model_proposal=request, model_original_score=proposal['score'] if proposal else None,
                                symbolic_action=action, proposal_changed=proposal is not None and request != action,
                                compile_error=compile_error,
                                symbolic_contribution='native operation selection and target binding; host literal argv/version executor',
                                proof_path=str(target / 'derivation.json'), cold_graph_verified=True, engine_pids=[first_pid, graph.process.pid],
                                before=before, after=after, execution=execution, response=response, stages=stages, score=judged,
                                correct_block=variant['mode'] in ['no_knowledge', 'no_rules', 'version_changed'] and before == after and not response['success'])
                            rows.append(row); trace.write(json.dumps(row) + '\n'); trace.flush()
                            write(root / 'progress.json', dict(updated_at=now(), outcomes=len(rows), total=p['outcomes'], case=case['id'], pid=os.getpid()))
                    finally: graph.close()
            finally: fixture.close()
    write(root / 'results.json', dict(completed_at=now(), outcomes=len(rows), input_hashes_unchanged=unchanged(p['files']), certification=False))


def audit(root):
    p = read(root / 'plan.json'); rows = [json.loads(s) for s in (root / 'trace.jsonl').read_text().splitlines()]
    cases = {read(f)['id']:read(f) for f in p['cases']}; groups = {}; checks = []
    def check(label, passed): checks.append(dict(label=label, passed=bool(passed)))
    check('frozen_inputs', unchanged(p['files'])); check('plan', digest(root / 'plan.json') == (root / 'plan.sha256').read_text().strip())
    check('count', len(rows) == p['outcomes'])
    for i, row in enumerate(rows):
        c = cases[row['case']]; result = score(row['before'], row['after'], c['target_job'], c['operation'], row['response'], row['stages'], c['examiner']['expected_argv'])
        check(f'{i}/effects', result == row['score']); check(f'{i}/fresh_engine', len(set(row['engine_pids'])) == 2)
        proof = read(row['proof_path']); check(f'{i}/action_receipt', proof['action'] == row['symbolic_action'])
        check(f'{i}/literal_terms_and_joins', proof_matches_rules(proof['proof'], read(root / 'rules.json')))
        key = row['mode'] + ('/' + row['adapter'] if row['adapter'] else '')
        group = groups.setdefault(key, dict(total=0, effect_pass=0, correct_blocks=0, changed_proposals=0, input_blocks=0))
        group['total'] += 1; group['effect_pass'] += result['kata_effect_pass']; group['correct_blocks'] += row['correct_block'] and row['compile_error'] is None; group['changed_proposals'] += row['proposal_changed']; group['input_blocks'] += row['compile_error'] is not None
        if row['compile_error']:
            check(f'{i}/unsupported_input_block', row['before'] == row['after'] and row['symbolic_action'] is None and not row['response']['success'])
        if row['mode'] in ['no_knowledge', 'no_rules']:
            check(f'{i}/causal_block', not proof['proof']['derived'] and proof['action'] is None and row['correct_block'])
        if row['mode'] == 'version_changed' and row['compile_error'] is None: check(f'{i}/version_block', row['correct_block'] and 'version differs' in row['response'].get('error', ''))
        if row['mode'] in ['full', 'hybrid'] and row['compile_error'] is None:
            check(f'{i}/native_proof', bool(proof['proof']['derived']) and all(d['antecedents'] for d in proof['proof']['derived']))
    passed = all(c['passed'] for c in checks)
    write(root / 'artifact-audit.json', dict(checked_at=now(), passed=passed, checks=checks))
    write(root / 'summary.json', dict(completed_at=read(root / 'results.json')['completed_at'], groups=groups, audit_checks=len(checks),
        all_checks_passed=passed, training_updates=0, new_model_generations=0, full_episode_success=None, certification=False,
        attribution='Native Engine derives operation/argument from declared policy and supplied evidence. Hybrid replays recorded proposals; symbolic-only quantifies work possible without the model.'))
    if not passed: raise ValueError('Symbolic completion audit failed')
    print(groups, flush=True)


def execute(root, worker='run_symbolic_completion.py', audit_fn=audit):
    p = read(root / 'plan.json')
    if read(root / 'status.json')['stage'] != 'prepared' or not unchanged(p['files']): raise ValueError('Refusing replay or changed inputs')
    if digest(root / 'plan.json') != (root / 'plan.sha256').read_text().strip(): raise ValueError('Plan changed')
    with Path(p['lock_file']).open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); status = dict(stage='running', pid=os.getpid(), started_at=now())
        def stop(sig, frame): raise ExperimentStop(f'Signal {sig}')
        signal.signal(signal.SIGTERM, stop)
        def heartbeat(pid):
            if shutil.disk_usage(root).free < p['min_free_bytes']: raise ExperimentStop('Disk below five GiB')
            status.update(child_pid=pid, updated_at=now()); write(root / 'status.json', status)
        try:
            code = run_bounded([sys.executable, str(root / 'source' / worker), 'run', '--out', str(root)], root / 'execution.log', p['timeout_seconds'], heartbeat)
            if code: raise RuntimeError(f'Execution exited {code}')
            audit_fn(root); status.update(stage='completed', updated_at=now()); status.pop('child_pid', None); write(root / 'status.json', status)
        except BaseException as error:
            status.update(stage='failed', error=str(error), updated_at=now()); write(root / 'status.json', status); raise


if __name__ == '__main__':
    a = argparse.ArgumentParser(description=__doc__); a.add_argument('mode', choices=['prepare', 'execute', 'run', 'audit'])
    for name in ['out', 'parent', 'bridge', 'lock-file']: a.add_argument('--' + name, type=Path, required=name == 'out')
    args = a.parse_args()
    if args.mode == 'prepare': prepare(args)
    elif args.mode == 'execute': execute(args.out.resolve())
    elif args.mode == 'run': run(args.out.resolve())
    else: audit(args.out.resolve())
